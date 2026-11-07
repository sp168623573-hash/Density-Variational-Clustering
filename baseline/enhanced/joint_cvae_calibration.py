"""Explicit CVAE-EVT engineering variant: reconstruction plus CVAE loss evidence.

All variant/alpha choices use known-family holdouts only. This is separately
named CVAE-EVT-Joint, not claimed to be the paper's unchanged baseline.
"""

import argparse
import json
import time
import joblib
import numpy as np
import torch
from torch import nn
import torch.distributions as D
from threadpoolctl import threadpool_limits
from common import *
from models import CVAE, safe_normal
import run_cvae_evt as author_adapter

MODES = (
    "reconstruction",
    "reconstruction_KL",
    "reconstruction_BCE",
    "all_max",
    "all_mean",
)
ALPHAS = (0.01, 0.025, 0.05, 0.1)


@torch.inference_mode()
def auxiliary(model, x, truth=None):
    predictions = []
    features = []
    for start in range(0, len(x), 1024):
        a = torch.tensor(x[start : start + 1024], device=model.device)
        output = model.c(a)
        pred = output.argmax(1)
        condition = (
            pred
            if truth is None
            else torch.tensor(
                truth[start : start + 1024], device=model.device, dtype=torch.long
            )
        )
        one = nn.functional.one_hot(condition, 9).float()
        q = safe_normal(model.c.l_z_xy, torch.cat((a, one), 1))
        p = safe_normal(model.c.l_z_x, a)
        kl = D.kl.kl_divergence(q, p).sum(1)
        bce = nn.functional.binary_cross_entropy(output, one, reduction="none").sum(1)
        predictions.append(pred.cpu().numpy())
        features.append(torch.stack((kl, bce), 1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(features).astype("float64")


def fit_aux(model, x, y):
    _, features = auxiliary(model, x, truth=y)
    tails = []
    scale = np.ones((9, 2))
    for c in range(9):
        row = []
        for j in range(2):
            v = features[y == c, j]
            tail = (
                author_adapter.fit_tail(v, 0.90)
                if len(v)
                else dict(cutoff=1.0, method="held_family")
            )
            row.append(tail)
            scale[c, j] = max(float(tail["cutoff"]), 1e-8)
        tails.append(row)
    return scale, tails


def merge(reconstruction, aux, mode):
    r = -np.asarray(reconstruction)
    if mode == "reconstruction":
        return -r
    if mode == "reconstruction_KL":
        return -np.maximum(r, aux[:, 0])
    if mode == "reconstruction_BCE":
        return -np.maximum(r, aux[:, 1])
    if mode == "all_max":
        return -np.maximum(r, aux.max(1))
    if mode == "all_mean":
        return -(r + aux.sum(1)) / 3
    raise ValueError(mode)


def select(settings):
    for setting in settings:
        started = time.perf_counter()
        base = HERE / "selection" / setting / "CVAE-EVT/FROZEN_SELECTION.json"
        original = json.loads(base.read_text())
        cfg = original["chosen"]["config"]
        out = HERE / "selection" / setting / "CVAE-EVT-Joint"
        out.mkdir(parents=True, exist_ok=False)
        save_json(
            out / "BEFORE_SELECTION.json",
            dict(
                config=cfg,
                modes=MODES,
                alphas=ALPHAS,
                base_selection_sha256=digest(base),
                description="Engineering extension using reconstruction, conditional/prior KL, predicted-class BCE; not unchanged paper model",
                query_values_read=False,
                query_labels_read=False,
            ),
        )
        k = load_known(setting)
        scores = {(mode, a): [] for mode in MODES for a in ALPHAS}
        for held in heldout_families(k["y"], 2):
            fm = k["y"] != held
            cm = k["yc"] != held
            view = View(setting, cfg["view"]).fit(k["x"][fm])
            x = view.transform(k["x"][fm])
            cal = view.transform(k["cal"][cm])
            proxy = view.transform(np.concatenate([k["x"][~fm], k["cal"][~cm]]))
            model = CVAE(setting, learning_rate=cfg["learning_rate"]).fit(
                x, k["y"][fm], path=out / f"held_{held}"
            )
            scale, tails = fit_aux(model, x, k["y"][fm])
            pc, rc = model.scores(cal)
            pp, rp = model.scores(proxy)
            ac_pred, ac = auxiliary(model, cal)
            ap_pred, ap = auxiliary(model, proxy)
            assert np.array_equal(pc, ac_pred) and np.array_equal(pp, ap_pred)
            ac = ac / scale[pc]
            ap = ap / scale[pp]
            for mode, alpha in scores:
                sc, sp = merge(rc, ac, mode), merge(rp, ap, mode)
                cuts, _ = probability_cuts(sc, pc, k["yc"][cm], alpha)
                cp = probability_decision(pc, sc, cuts)
                up = probability_decision(pp, sp, cuts)
                scores[(mode, alpha)].append(selector_score(k["yc"][cm], cp, up))
            save_json(out / f"held_{held}" / "auxiliary_GPD_tails.json", tails)
            print(
                setting,
                "joint known validation held",
                held,
                "seconds",
                round(time.perf_counter() - started, 1),
                flush=True,
            )
        trials = [
            dict(
                config=cfg,
                mode=mode,
                alpha=alpha,
                folds=v,
                **{n: float(np.mean([q[n] for q in v])) for n in v[0]},
            )
            for (mode, alpha), v in scores.items()
        ]
        best = max(
            trials,
            key=lambda q: (
                q["objective"],
                q["known_macro_recall"],
                -q["alpha"],
                -MODES.index(q["mode"]),
            ),
        )
        save_json(
            out / "FROZEN_SELECTION.json",
            dict(
                model="CVAE-EVT-Joint",
                setting=setting,
                chosen=best,
                trials=trials,
                known_sha256=digest(DATA / setting / "known.npz"),
                selection_seconds=time.perf_counter() - started,
                query_values_read=False,
                query_labels_read=False,
                base_selection_sha256=digest(base),
            ),
        )
        print(
            "FROZEN",
            setting,
            "joint",
            best["mode"],
            best["alpha"],
            best["objective"],
            flush=True,
        )


def apply(settings):
    root = BASE / "results/comparison_20261102_calibrated"
    for setting in settings:
        started = time.perf_counter()
        sp = HERE / "selection" / setting / "CVAE-EVT-Joint/FROZEN_SELECTION.json"
        sel = json.loads(sp.read_text())
        cfg = sel["chosen"]["config"]
        mode = sel["chosen"]["mode"]
        alpha = sel["chosen"]["alpha"]
        source = root / setting / "CVAE-EVT"
        out = root / setting / "CVAE-EVT-Joint"
        out.mkdir(parents=True, exist_ok=False)
        checkpoint = torch.load(
            source / "model_final.pt", map_location="cpu", weights_only=True
        )
        model = CVAE(setting, learning_rate=cfg["learning_rate"])
        model.c, model.r, _ = author_adapter.load_author_networks(
            setting, checkpoint["feature_dim"]
        )
        model.c.load_state_dict(checkpoint["classifier"])
        model.r.load_state_dict(checkpoint["reconstructor"])
        model.c = model.c.to(model.device).eval()
        model.r = model.r.to(model.device).eval()
        k = load_known(setting)
        view = joblib.load(source / "preprocessing.joblib")
        x = view.transform(k["x"])
        cal = view.transform(k["cal"])
        save_json(
            out / "FROZEN_BEFORE_QUERY.json",
            dict(
                selection_sha256=digest(sp),
                base_100_epochs_sha256=digest(source / "predictions_all_epochs.npz"),
                query_labels_read=False,
            ),
        )
        query, ids = load_query(setting)
        query = view.transform(query)
        scale, tails = fit_aux(model, x, k["y"])
        original = np.load(source / "predictions_all_epochs.npz")
        extra = {}
        preds = {}
        cps = []
        for role, values in [("train", x), ("cal", cal), ("query", query)]:
            p, a = auxiliary(model, values)
            assert np.all(original[role + "_fine"] == p[None, :])
            extra[role] = a / scale[p]
            preds[role] = p
        output = {role: [] for role in ("train", "cal", "query")}
        new_scores = {role: [] for role in output}
        for epoch in range(100):
            scores = {
                role: merge(original[role + "_score"][epoch], extra[role], mode)
                for role in output
            }
            cuts, _ = probability_cuts(scores["cal"], preds["cal"], k["yc"], alpha)
            cps.append(cuts.tolist())
            for role in output:
                output[role].append(
                    probability_decision(preds[role], scores[role], cuts)
                )
                new_scores[role].append(scores[role])
        np.savez_compressed(
            out / "predictions_all_epochs.npz",
            **{r: np.stack(v) for r, v in output.items()},
            **{r + "_score": np.stack(v) for r, v in new_scores.items()},
            **{r + "_fine": original[r + "_fine"] for r in output},
            fit_ids=k["ids"],
            cal_ids=k["cal_ids"],
            query_ids=ids,
        )
        save_json(out / "auxiliary_GPD_tails.json", tails)
        save_json(
            out / "known_calibration_all_epochs.json", dict(alpha=alpha, cuts=cps)
        )
        import shutil

        shutil.copyfile(source / "training_history.csv", out / "training_history.csv")
        sm = json.loads((source / "run_manifest.json").read_text())
        seconds = time.perf_counter() - started
        save_json(
            out / "run_manifest.json",
            dict(
                model="CVAE-EVT-Joint",
                setting=setting,
                epochs=100,
                row_semantics="known_only_joint_calibration_of_100_actual_CVAE_reconstruction_epochs",
                config=cfg,
                mode=mode,
                alpha=alpha,
                training_complete=True,
                query_truth_read_during_training=False,
                selected_epoch=100,
                base_model=str(source.relative_to(BASE)),
                base_model_sha256=digest(source / "model_final.pt"),
                fit_seconds=sm["fit_seconds"],
                total_seconds=sm["total_seconds"] + seconds,
                postprocessing_seconds=seconds,
                selection_seconds=sel["selection_seconds"],
                engineering_extension=True,
                published=False,
                final_test_executed=False,
            ),
        )
        print("COMPLETE joint calibration", setting, round(seconds, 1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["select", "apply"])
    p.add_argument("--settings", nargs="+", default=list(SETTINGS))
    a = p.parse_args()
    with threadpool_limits(limits=4):
        (select if a.phase == "select" else apply)(a.settings)
