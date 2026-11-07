"""Explicit transductive enhancement, calibrated using known data only.

Pointwise known-reference support is compared with unlabeled batch density.
Feature duplicates are represented once in geometric calculations and then
mapped back to every original row. No query labels enter any model method.
"""

import argparse
import json
import time
import numpy as np
import joblib
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from common import *

KS = (3, 7, 15)
VIEWS = ("original", "log_standard", "log_robust", "log_whiten", "log_class_whiten")
ALPHAS = (0.01, 0.025, 0.05, 0.1)
MIN_SUPPORT = (0.0, 0.05, 0.2, 0.5)


class DensityReference:
    def fit(self, x, y):
        references = []
        labels = []
        radius = {k: [] for k in KS}
        self.dimensions = {}
        self.counts = {}
        for c in np.unique(y):
            points = np.unique(np.asarray(x[y == c], dtype="float64"), axis=0)
            n = len(points)
            references.append(points)
            labels.append(np.full(n, c, dtype="int64"))
            self.counts[int(c)] = n
            if n < 2:
                self.dimensions[int(c)] = 5.0
                for k in KS:
                    radius[k].append(np.ones(n))
                continue
            d = cdist(points, points)
            np.fill_diagonal(d, np.inf)
            maxk = min(20, n - 1)
            nearest = np.sort(np.partition(d, maxk - 1, axis=1)[:, :maxk], axis=1)
            if maxk >= 3:
                logs = np.log(
                    np.maximum(nearest[:, -1, None], 1e-12)
                    / np.maximum(nearest[:, :-1], 1e-12)
                )
                dimensions = 1 / np.maximum(logs.mean(1), 1e-6)
                self.dimensions[int(c)] = float(np.clip(np.median(dimensions), 1, 20))
            else:
                self.dimensions[int(c)] = 5.0
            for k in KS:
                radius[k].append(nearest[:, min(k, n - 1) - 1])
        self.x = np.concatenate(references)
        self.y = np.concatenate(labels)
        self.radii = {k: np.concatenate(v) for k, v in radius.items()}
        self.floors = {
            k: max(float(np.median(v[v > 0])) * 1e-6, 1e-12)
            for k, v in self.radii.items()
        }
        return self

    def scores(self, x):
        unique, inverse = np.unique(
            np.asarray(x, dtype="float64"), axis=0, return_inverse=True
        )
        nearest = []
        distance = []
        for start in range(0, len(unique), 512):
            d = cdist(unique[start : start + 512], self.x)
            ix = d.argmin(1)
            nearest.append(ix)
            distance.append(d[np.arange(len(ix)), ix])
        nearest = np.concatenate(nearest)
        distance = np.concatenate(distance)
        pred = self.y[nearest]
        results = {
            k: dict(
                support=distance / np.maximum(self.radii[k][nearest], self.floors[k]),
                density=np.full(len(unique), -np.inf),
            )
            for k in KS
        }
        for c in np.unique(pred):
            ix = np.flatnonzero(pred == c)
            points = unique[ix]
            reference = self.x[self.y == c]
            nq, nf = len(ix), len(reference)
            if nq < 3 or nf < 3:
                continue
            dtrain = cdist(points, reference)
            dbatch = cdist(points, points)
            np.fill_diagonal(dbatch, np.inf)
            for k in KS:
                kt = min(k, nf)
                kq = min(k, nq - 1)
                rt = np.partition(dtrain, kt - 1, axis=1)[:, kt - 1]
                rq = np.partition(dbatch, kq - 1, axis=1)[:, kq - 1]
                score = (
                    self.dimensions[int(c)]
                    * np.log(
                        np.maximum(rt, self.floors[k]) / np.maximum(rq, self.floors[k])
                    )
                    + np.log(nf / nq)
                    + np.log(kq / kt)
                )
                results[k]["density"][ix] = score
        return (
            {
                k: {name: values[inverse] for name, values in parts.items()}
                for k, parts in results.items()
            },
            pred[inverse],
            dict(
                original_rows=len(x),
                unique_feature_groups=len(unique),
                known_references=len(self.x),
            ),
        )


def cuts(cal, truth, alpha):
    values = []
    for coarse in [0, 1]:
        a = cal[(truth != 0) == bool(coarse)]
        a = a[np.isfinite(a)]
        if len(a):
            values.append(float(np.quantile(a, 1 - alpha, method="higher")))
    return max(values) if values else 1e100


def score_decision(parts, threshold, min_support):
    return (parts["density"] > threshold) & (parts["support"] > min_support)


def select(settings):
    for setting in settings:
        start = time.perf_counter()
        k = load_known(setting)
        out = HERE / "selection_density" / setting
        out.mkdir(parents=True, exist_ok=False)
        config = dict(
            views=VIEWS,
            neighbors=KS,
            alphas=ALPHAS,
            min_support=MIN_SUPPORT,
            held_families=heldout_families(k["y"], 3),
            query_labels_used=False,
            transductive=True,
            training_sample_resampling=False,
            geometry_groups="one per distinct encoded feature vector, all output rows retained",
        )
        save_json(out / "BEFORE_SELECTION.json", config)
        trials = []
        for view_name in VIEWS:
            values = {(n, a, m): [] for n in KS for a in ALPHAS for m in MIN_SUPPORT}
            for held in config["held_families"]:
                fm = k["y"] != held
                cm = k["yc"] != held
                view = View(setting, view_name).fit(k["x"][fm], k["y"][fm])
                x = view.transform(k["x"][fm])
                cal = view.transform(k["cal"][cm])
                proxy = view.transform(np.concatenate([k["x"][~fm], k["cal"][~cm]]))
                model = DensityReference().fit(x, k["y"][fm])
                cs, _, _ = model.scores(cal)
                mixed = np.concatenate([cal, proxy])
                ms, pred, meta = model.scores(mixed)
                for n, a, m in values:
                    threshold = cuts(cs[n]["density"], k["yc"][cm], a)
                    rejected = score_decision(ms[n], threshold, m)
                    p = np.where(pred == 0, 0, 1).astype("int8")
                    p[rejected] = 2
                    result = selector_score(k["yc"][cm], p[: len(cal)], p[len(cal) :])
                    result["threshold"] = threshold
                    values[(n, a, m)].append(result)
            for (n, a, m), folds in values.items():
                trial = dict(view=view_name, k=n, alpha=a, min_support=m, folds=folds)
                trial.update(
                    {
                        field: float(np.mean([f[field] for f in folds]))
                        for field in [
                            "objective",
                            "known_macro_recall",
                            "proxy_unknown_recall",
                            "known_BEN_recall",
                        ]
                    }
                )
                trials.append(trial)
            save_json(out / "trials.json", trials)
            best = max(
                trials,
                key=lambda t: (t["objective"], t["known_macro_recall"], -t["alpha"]),
            )
            print(
                setting,
                "density view",
                view_name,
                "best known-proxy",
                round(best["objective"], 4),
                "seconds",
                round(time.perf_counter() - start, 1),
                flush=True,
            )
        best = max(
            trials, key=lambda t: (t["objective"], t["known_macro_recall"], -t["alpha"])
        )
        save_json(
            out / "FROZEN_SELECTION.json",
            dict(
                setting=setting,
                chosen=best,
                selection_seconds=time.perf_counter() - start,
                known_sha256=digest(DATA / setting / "known.npz"),
                query_labels_used=False,
                query_values_read=False,
                transductive=True,
                algorithm="known-reference versus unlabeled batch conditional density contrast",
            ),
        )
        print(
            "FROZEN density",
            setting,
            {n: best[n] for n in ["view", "k", "alpha", "min_support", "objective"]},
            flush=True,
        )


def apply(settings):
    root = BASE / "results/comparison_20261102_density_enhanced"
    root.mkdir(exist_ok=True)
    for setting in settings:
        start = time.perf_counter()
        sp = HERE / "selection_density" / setting / "FROZEN_SELECTION.json"
        selected = json.loads(sp.read_text())
        cfg = selected["chosen"]
        k = load_known(setting)
        out = root / setting
        out.mkdir(parents=True, exist_ok=False)
        view = View(setting, cfg["view"]).fit(k["x"], k["y"])
        x = view.transform(k["x"])
        cal = view.transform(k["cal"])
        reference = DensityReference().fit(x, k["y"])
        cs, cp, cm = reference.scores(cal)
        threshold = cuts(cs[cfg["k"]]["density"], k["yc"], cfg["alpha"])
        joblib.dump(
            dict(view=view, reference=reference), out / "known_density_reference.joblib"
        )
        save_json(
            out / "FROZEN_BEFORE_QUERY.json",
            dict(
                selection_sha256=digest(sp),
                threshold=threshold,
                min_support=cfg["min_support"],
                k=cfg["k"],
                known_reference_sha256=digest(out / "known_density_reference.joblib"),
                query_labels_used=False,
                transductive=True,
            ),
        )
        query, query_ids = load_query(setting)
        query = view.transform(query)
        qs, qp, qmeta = reference.scores(query)
        fs, fp, fmeta = reference.scores(x)
        qmask = score_decision(qs[cfg["k"]], threshold, cfg["min_support"])
        fmask = score_decision(fs[cfg["k"]], threshold, cfg["min_support"])
        cmask = score_decision(cs[cfg["k"]], threshold, cfg["min_support"])
        np.savez_compressed(
            out / "density_scores_before_truth.npz",
            query_ids=query_ids,
            density=qs[cfg["k"]]["density"],
            support=qs[cfg["k"]]["support"],
            guard_reject=qmask,
            fit_ids=k["ids"],
            fit_guard_reject=fmask,
            cal_ids=k["cal_ids"],
            cal_guard_reject=cmask,
        )
        for model in ["CVAE-EVT", "EVM", "W-SVM"]:
            source = BASE / "results/comparison_20261102_calibrated" / setting / model
            if setting == "NSL1" and model == "W-SVM":
                source = (
                    BASE / "results/comparison_20261102_whitening" / setting / model
                )
            original = np.load(source / "predictions_all_epochs.npz")
            assert np.array_equal(original["query_ids"], query_ids)
            pquery = original["query"].copy()
            pquery[:, qmask] = 2
            pfit = original["train"].copy()
            pfit[:, fmask] = 2
            pcal = original["cal"].copy()
            pcal[:, cmask] = 2
            dest = out / (model + "-DC")
            dest.mkdir(exist_ok=False)
            np.savez_compressed(
                dest / "predictions_all_epochs.npz",
                query=pquery,
                train=pfit,
                cal=pcal,
                query_ids=query_ids,
                fit_ids=k["ids"],
                cal_ids=k["cal_ids"],
            )
            sm = json.loads((source / "run_manifest.json").read_text())
            import shutil

            shutil.copyfile(
                source / "training_history.csv", dest / "base_training_history.csv"
            )
            save_json(
                dest / "run_manifest.json",
                dict(
                    model=model + "-DC",
                    base_model=model,
                    setting=setting,
                    epochs=100,
                    row_semantics="density-enhanced outputs from 100 actual base model epochs/refits",
                    training_complete=True,
                    base_run=str(source.relative_to(BASE)),
                    base_predictions_sha256=digest(
                        source / "predictions_all_epochs.npz"
                    ),
                    selection_sha256=digest(sp),
                    threshold=threshold,
                    config=cfg,
                    query_truth_read_during_training=False,
                    query_labels_used_for_inference=False,
                    unlabeled_query_batch_used=True,
                    transductive=True,
                    query_geometry=qmeta,
                    fit_geometry=fmeta,
                    cal_geometry=cm,
                    guard_seconds=time.perf_counter() - start,
                    base_seconds=sm["total_seconds"],
                    total_seconds=sm["total_seconds"] + time.perf_counter() - start,
                    selection_seconds=selected["selection_seconds"],
                    selected_epoch=100,
                    published=False,
                    final_test_executed=False,
                ),
            )
        print(
            "COMPLETE density enhancement",
            setting,
            "unlabeled rejected",
            int(qmask.sum()),
            "of",
            len(qmask),
            "seconds",
            round(time.perf_counter() - start, 1),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["select", "apply"])
    p.add_argument("--settings", nargs="+", default=list(SETTINGS))
    a = p.parse_args()
    with threadpool_limits(limits=4):
        (select if a.phase == "select" else apply)(a.settings)
