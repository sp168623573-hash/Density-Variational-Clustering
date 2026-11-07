"""Original EVM/W-SVM mechanisms and class-balanced author CVAE networks."""

from pathlib import Path
import contextlib
import io
import json
import subprocess
import time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import dump_svmlight_file
import torch
from torch import nn
import torch.distributions as D
from common import BASE, native_tool, save_json
import run_evm as old_evm
import run_cvae_evt as old_cvae


class EVM:
    def __init__(self, tail_size=50, margin_scale=0.5, nfuse=4):
        self.tail_size, self.margin_scale, self.nfuse = tail_size, margin_scale, nfuse

    def fit(self, x, y, seed=1234, distances=None):
        self.x = np.asarray(x, dtype="float64")
        self.y = np.asarray(y)
        dist = cdist(self.x, self.x) if distances is None else distances
        tails = np.empty((len(x), self.tail_size))
        for c in np.unique(y):
            pos = np.flatnonzero(y == c)
            neg = np.flatnonzero(y != c)
            for start in range(0, len(pos), 256):
                ix = pos[start : start + 256]
                tails[ix] = (
                    np.partition(dist[np.ix_(ix, neg)], self.tail_size - 1, axis=1)[
                        :, : self.tail_size
                    ]
                    * self.margin_scale
                )
        params = np.zeros((len(x), 6))
        self.invalid_fits = old_evm.native().mr_fit_rows(
            np.ascontiguousarray(tails), len(x), self.tail_size, 1, params
        )
        order = np.random.default_rng(seed).permutation(len(x))
        keep = old_evm.reduce(dist, params, y, order)
        if not len(keep):
            raise ValueError("No valid EVM extreme vectors")
        self.points, self.params, self.labels = self.x[keep], params[keep], y[keep]
        self.keep = keep
        return self

    def scores(self, x):
        result = []
        for start in range(0, len(x), 512):
            p = old_evm.probability(
                cdist(self.points, np.asarray(x[start : start + 512], dtype="float64")),
                self.params,
            )
            fused = []
            for c in range(9):
                v = p[self.labels == c]
                n = min(self.nfuse, len(v))
                fused.append(
                    np.partition(v, -n, axis=0)[-n:].mean(0)
                    if n
                    else np.zeros(p.shape[1])
                )
            result.append(np.stack(fused, axis=1))
        score = np.concatenate(result)
        return score.argmax(1), score.max(1)

    def save(self, path):
        np.savez_compressed(
            path,
            points=self.points,
            params=self.params,
            labels=self.labels,
            indices=self.keep,
            nfuse=self.nfuse,
            tail_size=self.tail_size,
            margin_scale=self.margin_scale,
        )


class WSVM:
    def __init__(self, path, C=1, gamma_multiplier=1, nu=0.05):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.C, self.gamma_multiplier, self.nu = C, gamma_multiplier, nu
        self.model = self.path / "model"

    def fit(self, x, y, seed=1234):
        train = self.path / "train.libsvm"
        dump_svmlight_file(x, y + 1, str(train), zero_based=False)
        gamma = self.gamma_multiplier / x.shape[1]
        args = [
            str(native_tool("svm-train.exe")),
            "-s",
            "8",
            "-t",
            "2",
            "-c",
            str(self.C),
            "-g",
            str(gamma),
            "-a",
            str(gamma),
            "-n",
            str(self.nu),
            "-o",
            "1",
            "-m",
            "256",
            "-q",
            str(train),
            str(self.model),
        ]
        with (self.path / "train.log").open("w", encoding="utf-8") as log:
            r = subprocess.run(
                args, stdout=log, stderr=subprocess.STDOUT, cwd=self.path
            )
        text = (self.path / "train.log").read_text("utf-8")
        if (
            r.returncode
            or "Error:" in text
            or "fit weibull positive -" in text
            or "fit weibull compliment -" in text
        ):
            raise ValueError(
                "Native W-SVM did not produce valid Weibull models: " + text[-350:]
            )
        return self

    def scores(self, x, tag="score"):
        source = self.path / (tag + ".libsvm")
        target = self.path / (tag + ".pred")
        dump_svmlight_file(x, np.ones(len(x)), str(source), zero_based=False)
        with (self.path / (tag + ".log")).open("w") as log:
            subprocess.run(
                [
                    str(native_tool("svm-predict.exe")),
                    "-P",
                    "0",
                    "-C",
                    "0.001",
                    str(source),
                    str(self.model),
                    str(target),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                cwd=self.path,
            )
        raw = np.loadtxt(target, delimiter=":")
        pred = raw[:, 0].astype("int64") - 1
        if np.any((pred < 0) | (pred > 8)) or not np.isfinite(raw[:, 1]).all():
            raise ValueError("Invalid W-SVM output")
        return pred, raw[:, 1]


def safe_normal(net, x):
    mu, logstd = net(x).chunk(2, -1)
    return D.Normal(mu, logstd.clamp(-8, 5).exp())


class CVAE:
    def __init__(self, setting, learning_rate=0.001, batch_size=512, epochs=100):
        self.setting, self.lr, self.batch, self.epochs = (
            setting,
            learning_rate,
            batch_size,
            epochs,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_classes = 9
        self.benign_clusters = 1

    def fit(self, x, y, seed=1234, path=None, callback=None):
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.c, self.r, _ = old_cvae.load_author_networks(self.setting, x.shape[1])
        self.c = self.c.to(self.device)
        self.r = self.r.to(self.device)
        if self.n_classes != 9:
            latent = self.c.l_z_x[-1].out_features // 2
            self.c.l_z_xy[0] = nn.Linear(
                x.shape[1] + self.n_classes, self.c.l_z_xy[0].out_features
            ).to(self.device)
            self.c.l_y_xz[-2] = nn.Linear(
                self.c.l_y_xz[-2].in_features, self.n_classes
            ).to(self.device)
            self.r.l_z_xy[0] = nn.Linear(
                x.shape[1] + self.n_classes, self.r.l_z_xy[0].out_features
            ).to(self.device)
            self.r.l_x_yz[0] = nn.Linear(
                latent + self.n_classes, self.r.l_x_yz[0].out_features
            ).to(self.device)
            for p in self.r.l_z_xy.parameters():
                p.requires_grad = False
        self.x = torch.tensor(x, device=self.device)
        self.y = torch.tensor(y, device=self.device, dtype=torch.long)
        counts = np.bincount(y, minlength=self.n_classes)
        active = np.count_nonzero(counts)
        weight = np.zeros(self.n_classes, dtype="float32")
        weight[counts > 0] = len(y) / (active * counts[counts > 0])
        w = torch.tensor(weight, device=self.device)
        self.history = []
        started = time.perf_counter()
        optim = torch.optim.Adam(
            self.c.parameters(), lr=self.lr, fused=self.device == "cuda"
        )
        for epoch in range(1, self.epochs + 1):
            self.c.train()
            total = torch.zeros((), device=self.device)
            for ix in torch.randperm(len(x), device=self.device).split(self.batch):
                a = self.x[ix]
                one = nn.functional.one_hot(self.y[ix], self.n_classes).float()
                posterior = safe_normal(self.c.l_z_xy, torch.cat((a, one), 1))
                prior = safe_normal(self.c.l_z_x, a)
                output = self.c.y_xz(a, posterior.rsample())
                losses = D.kl.kl_divergence(posterior, prior).sum(
                    1
                ) + nn.functional.binary_cross_entropy(
                    output, one, reduction="none"
                ).sum(
                    1
                )
                loss = (losses * w[self.y[ix]]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("CVAE stage1 diverged")
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.c.parameters(), 10, foreach=True)
                optim.step()
                total += loss.detach() * len(ix)
            self.history.append(
                dict(
                    stage=1,
                    epoch=epoch,
                    loss=float(total) / len(x),
                    elapsed=time.perf_counter() - started,
                )
            )
            if path and epoch % 10 == 0:
                save_json(Path(path) / "progress.json", self.history[-1])
        self.c.eval()
        self.r.l_z_xy.load_state_dict(self.c.l_z_xy.state_dict())
        for p in self.r.l_z_xy.parameters():
            p.requires_grad = False
        optim = torch.optim.Adam(
            [p for p in self.r.parameters() if p.requires_grad],
            lr=self.lr,
            fused=self.device == "cuda",
        )
        for epoch in range(1, self.epochs + 1):
            self.r.train()
            total = torch.zeros((), device=self.device)
            for ix in torch.randperm(len(x), device=self.device).split(self.batch):
                a = self.x[ix]
                one = nn.functional.one_hot(self.y[ix], self.n_classes).float()
                posterior = safe_normal(self.r.l_z_xy, torch.cat((a, one), 1))
                output = self.r.x_yz(one, posterior.rsample())
                loss = ((output - a).square().mean(1) * w[self.y[ix]]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("CVAE stage2 diverged")
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.r.parameters(), 10, foreach=True)
                optim.step()
                total += loss.detach() * len(ix)
            self.r.eval()
            self.history.append(
                dict(
                    stage=2,
                    epoch=epoch,
                    loss=float(total) / len(x),
                    elapsed=time.perf_counter() - started,
                )
            )
            if path and epoch % 10 == 0:
                save_json(Path(path) / "progress.json", self.history[-1])
            if callback:
                callback(self, epoch, self.history[-1])
        self.training_seconds = time.perf_counter() - started
        self.fit_tails()
        return self

    def fit_tails(self):
        errors = self.errors(self.x, self.y)
        self.tails = []
        y = self.y.cpu().numpy()
        for c in range(self.n_classes):
            v = errors[y == c]
            self.tails.append(
                old_cvae.fit_tail(v, 0.90)
                if len(v)
                else dict(cutoff=1.0, method="heldout_class_not_trained")
            )
        self.scales = np.maximum([t["cutoff"] for t in self.tails], 1e-8)

    @torch.inference_mode()
    def scores(self, x):
        a = torch.tensor(x, device=self.device)
        p = old_cvae.classification(self.c, a)
        err = self.errors(a, p)
        pred = p.cpu().numpy()
        # A monotone score in each class. Known-CAL coverage calibrates the
        # GPD scale, including finite-support saturation and sparse tails.
        return pred, -err / self.scales[pred]

    @torch.inference_mode()
    def errors(self, x, y):
        result = []
        for a, b in zip(x.split(1024), y.split(1024)):
            one = nn.functional.one_hot(b, self.n_classes).float()
            result.append((self.r(a, one) - a).square().sum(1).cpu().numpy())
        return np.concatenate(result).astype("float64")

    def calibration_labels(self, x, y):
        return y

    def save(self, path):
        torch.save(
            dict(
                classifier=self.c.state_dict(),
                reconstructor=self.r.state_dict(),
                tails=self.tails,
                feature_dim=self.x.shape[1],
                setting=self.setting,
                n_classes=self.n_classes,
                benign_clusters=self.benign_clusters,
            ),
            path,
        )


class CVAEBC(CVAE):
    """The original paper's CVAE-EVT* with eight FIT-only benign clusters."""

    def __init__(
        self,
        setting,
        learning_rate=0.001,
        batch_size=512,
        epochs=100,
        benign_clusters=8,
    ):
        super().__init__(setting, learning_rate, batch_size, epochs)
        self.n_classes = benign_clusters + 8
        self.benign_clusters = benign_clusters

    def calibration_labels(self, x, y):
        labels = np.asarray(y).copy() + self.benign_clusters - 1
        benign = np.asarray(y) == 0
        if benign.any():
            labels[benign] = self.clustering.predict(x[benign])
        return labels

    def fit(self, x, y, seed=1234, path=None, callback=None):
        from sklearn.cluster import KMeans

        self.clustering = KMeans(
            n_clusters=self.benign_clusters, n_init=10, random_state=seed
        ).fit(x[y == 0])
        mapped = self.calibration_labels(x, y)
        return super().fit(x, mapped, seed, path, callback)

    def save(self, path):
        import joblib

        super().save(path)
        joblib.dump(self.clustering, Path(path).with_suffix(".clusters.joblib"))
