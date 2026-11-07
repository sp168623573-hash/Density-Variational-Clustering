"""Known-only BEN protection and stricter two-source rejection calibration.

Current truth is opened only by the report phase. All parameter choices use
known calibration and known-family holdout proxies. The 100 actual native
W-SVM fits and the old complete enhanced vector remain preserved.
"""

import argparse
import csv
import hashlib
import json
import time
import shutil
import numpy as np
import joblib
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from common import *
from density_enhancement import DensityReference
from cic2_anchored_density import controls
from models import WSVM

ROOT = BASE / "results/CIC2_WSVM_LOW_FPR_20261102"
SELECT = HERE / "selection_cic2_low_fpr_v2"
OLD = BASE / "results/comparison_20261102_source_anchored/CIC2/W-SVM-DCA"
BASE_RUN = BASE / "results/comparison_20261102_calibrated/CIC2/W-SVM"
RATES = (0.005, 0.01, 0.025)
NEIGHBORS = (3, 7, 15)


def benign_quantile(scores, pred, y, alpha):
    values = scores[(y == 0) & (pred == 0)]
    if not len(values):
        raise ValueError("No correctly predicted BEN controls")
    return float(np.quantile(values, alpha, method="lower"))


def density_ben_quantile(score, y, alpha):
    values = score[(y == 0) & np.isfinite(score)]
    if not len(values):
        raise ValueError("No finite BEN density controls")
    return float(np.quantile(values, 1 - alpha, method="higher"))


class BenignSupport:
    def fit(self, x, y):
        self.x = np.ascontiguousarray(x, dtype="float64")
        self.y = np.asarray(y)
        self.bn = NearestNeighbors(n_neighbors=16, n_jobs=1).fit(self.x[y == 0])
        self.an = NearestNeighbors(n_neighbors=3, n_jobs=1).fit(self.x[y != 0])
        self.all = NearestNeighbors(n_neighbors=15, n_jobs=1).fit(self.x)
        self.known_ben = self.x[y == 0]
        distances, _ = self.bn.kneighbors(self.known_ben)
        self.radii = {k: np.maximum(distances[:, k], 1e-8) for k in NEIGHBORS}
        return self

    def score(self, x):
        bd, bi = self.bn.kneighbors(x)
        ad, _ = self.an.kneighbors(x)
        _, ii = self.all.kneighbors(x)
        values = {}
        for k in NEIGHBORS:
            votes = np.mean(self.y[ii[:, :k]] == 0, axis=1)
            support = bd[:, 0] / self.radii[k][bi[:, 0]]
            ratio = bd[:, : min(3, k)].mean(1) / np.maximum(ad.mean(1), 1e-8)
            # Smaller is stronger evidence for BEN. A single opposite-class
            # neighbor raises the score beyond every all-BEN neighborhood.
            values[k] = np.maximum(support, ratio) + 10 * (1 - votes)
        return values


def guard_cut(sa, ya, sb, yb, alpha):
    attack = np.concatenate([sa[ya != 0], sb[yb != 0]])
    return float(np.quantile(attack, alpha, method="lower"))


def thresholds(
    pred_a, score_a, ya, pred_b, score_b, yb, density_a, density_b, alpha, original
):
    base_cuts = np.asarray(original, dtype="float64").copy()
    base_cuts[0] = min(
        benign_quantile(score_a, pred_a, ya, alpha),
        benign_quantile(score_b, pred_b, yb, alpha),
    )
    density_cut = max(
        density_ben_quantile(density_a, ya, alpha),
        density_ben_quantile(density_b, yb, alpha),
    )
    return base_cuts, density_cut


def predictions(
    pred, score, density, base_cuts, density_cut, ben_score=None, ben_cut=None
):
    result = np.where(pred == 0, 0, 1).astype("int8")
    result[score < base_cuts[pred]] = 2
    result[density > density_cut] = 2
    if ben_score is not None:
        # Recover only W-SVM predicted BEN, never rewrite an accepted attack
        # with a guard that learned exclusively BEN/KNA reference support.
        result[(pred == 0) & (ben_score < ben_cut)] = 0
    return result


def score_fold(y, p, proxy, domains):
    fprs = [
        float(np.mean(p[(y == 0) & (domains == d)] != 0)) for d in np.unique(domains)
    ]
    attacks = [float(np.mean(p[(y == c)] == 1)) for c in np.unique(y) if c > 0]
    return dict(
        max_source_BEN_FPR=max(fprs),
        source_BEN_FPR=fprs,
        known_KNA_macro_recall=float(np.mean(attacks)),
        proxy_UNA_recall=float(np.mean(proxy == 2)),
        known_overall_accuracy=float(np.mean(p == (y != 0))),
    )


def select():
    started = time.perf_counter()
    SELECT.mkdir(parents=True, exist_ok=False)
    k = load_known("CIC2")
    s = controls()
    x = np.concatenate([k["x"], s["x"]])
    y = np.concatenate([k["y"], s["y"]])
    configs = [
        dict(alpha=a, ben_view=v, ben_k=n, ben_attack_allowance=b)
        for a in RATES
        for v in ["none", "log_standard", "log_robust"]
        for n in ([0] if v == "none" else NEIGHBORS)
        for b in ([0.0] if v == "none" else [0.0, 0.005, 0.01])
    ]
    folds = heldout_families(y, 3)
    save_json(
        SELECT / "BEFORE_SELECTION.json",
        dict(
            candidates=configs,
            held_known_families=folds,
            objective="Minimize known BEN FPR first under a 3% maximum source BEN error gate; then maximize mean proxy-UNA recall and known-KNA macro recall",
            query_features_opened=False,
            query_labels_opened=False,
            source_reference_rows=len(s["x"]),
            source_cal_rows=len(s["cal"]),
        ),
    )
    all_scores = [[] for _ in configs]
    for held in folds:
        fm = y != held
        ca = k["yc"] != held
        cb = s["yc"] != held
        proxy_raw = np.concatenate([x[~fm], k["cal"][~ca], s["cal"][~cb]])
        a_raw, b_raw = k["cal"][ca], s["cal"][cb]
        ya, yb = k["yc"][ca], s["yc"][cb]
        mixed = np.concatenate([a_raw, b_raw, proxy_raw])
        known_y = np.concatenate([ya, yb])
        domains = np.r_[np.zeros(len(ya)), np.ones(len(yb))].astype("int64")
        view = View("CIC2", "log_standard").fit(x[fm], y[fm])
        w = WSVM(SELECT / f"held_{held}/native", C=16, gamma_multiplier=4, nu=0.05).fit(
            view.transform(x[fm]), y[fm]
        )
        pa, sa = w.scores(view.transform(a_raw), "cal_a")
        pb, sb = w.scores(view.transform(b_raw), "cal_b")
        pm, sm = w.scores(view.transform(mixed), "mixed")
        # Recalibrate the entire fold without borrowing full-model cuts.
        old_cuts, _ = probability_cuts(np.r_[sa, sb], np.r_[pa, pb], known_y, 0.05)
        dv = View("CIC2", "log_whiten").fit(x[fm], y[fm])
        dr = DensityReference().fit(dv.transform(x[fm]), y[fm])
        da, _, _ = dr.scores(dv.transform(a_raw))
        db, _, _ = dr.scores(dv.transform(b_raw))
        dm, _, _ = dr.scores(dv.transform(mixed))
        support = {}
        for v in ["log_standard", "log_robust"]:
            bv = View("CIC2", v).fit(x[fm], y[fm])
            br = BenignSupport().fit(bv.transform(x[fm]), y[fm])
            support[v] = (
                br.score(bv.transform(a_raw)),
                br.score(bv.transform(b_raw)),
                br.score(bv.transform(mixed)),
            )
        for i, c in enumerate(configs):
            cut, dc = thresholds(
                pa,
                sa,
                ya,
                pb,
                sb,
                yb,
                da[7]["density"],
                db[7]["density"],
                c["alpha"],
                old_cuts,
            )
            bs, bc = None, None
            if c["ben_view"] != "none":
                aa, bb, mm = support[c["ben_view"]]
                n = c["ben_k"]
                bs = mm[n]
                bc = guard_cut(aa[n], ya, bb[n], yb, c["ben_attack_allowance"])
            p = predictions(pm, sm, dm[7]["density"], cut, dc, bs, bc)
            result = score_fold(known_y, p[: len(known_y)], p[len(known_y) :], domains)
            all_scores[i].append(result)
        save_json(
            SELECT / f"held_{held}/scores.json",
            dict(
                candidates=configs,
                results=[a[-1] for a in all_scores],
                query_labels_used=False,
            ),
        )
        print(
            "Known-only low-FPR fold",
            held,
            "seconds",
            round(time.perf_counter() - started, 1),
            flush=True,
        )
    trials = []
    for config, values in zip(configs, all_scores):
        r = dict(
            config=config,
            folds=values,
            max_source_BEN_FPR=max(v["max_source_BEN_FPR"] for v in values),
            known_KNA_macro_recall=float(
                np.mean([v["known_KNA_macro_recall"] for v in values])
            ),
            proxy_UNA_recall=float(np.mean([v["proxy_UNA_recall"] for v in values])),
        )
        r["objective"] = 0.5 * (r["known_KNA_macro_recall"] + r["proxy_UNA_recall"])
        r["BEN_gate_passed"] = r["max_source_BEN_FPR"] <= 0.03
        trials.append(r)
    feasible = [r for r in trials if r["BEN_gate_passed"]]
    best = (
        max(feasible, key=lambda r: (r["objective"], -r["max_source_BEN_FPR"]))
        if feasible
        else min(trials, key=lambda r: (r["max_source_BEN_FPR"], -r["objective"]))
    )
    save_json(
        SELECT / "FROZEN_SELECTION.json",
        dict(
            chosen=best,
            trials=trials,
            selection_seconds=time.perf_counter() - started,
            query_labels_used=False,
            query_values_opened=False,
            reference_data_hashes=dict(
                known=digest(DATA / "CIC2/known.npz"),
                source=digest(
                    BASE
                    / "results/comparison_20261102_density_matched/CIC2/source_known_controls.npz"
                ),
            ),
        ),
    )
    print("FROZEN low-FPR", json.dumps(best), flush=True)


def apply():
    started = time.perf_counter()
    ROOT.mkdir(parents=True, exist_ok=False)
    selection = json.loads((SELECT / "FROZEN_SELECTION.json").read_text())
    cfg = selection["chosen"]["config"]
    k = load_known("CIC2")
    s = controls()
    x = np.concatenate([k["x"], s["x"]])
    y = np.concatenate([k["y"], s["y"]])
    # Existing 100 native fits share the same byte-identical final model.
    with (BASE_RUN / "training_history.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        history = list(csv.DictReader(f))
    assert len(history) == 100 and len({r["Model_SHA256"] for r in history}) == 1
    view = joblib.load(BASE_RUN / "preprocessing.joblib")
    w = WSVM(ROOT / "native_replay")
    w.model = BASE_RUN / "model_final"
    pa, sa = w.scores(view.transform(k["cal"]), "cal_original")
    pb, sb = w.scores(view.transform(s["cal"]), "cal_source")
    anchored = BASE / "results/comparison_20261102_source_anchored/CIC2"
    artifact = joblib.load(anchored / "reference.joblib")
    dv, dr = artifact["view"], artifact["reference"]
    da, _, _ = dr.scores(dv.transform(k["cal"]))
    db, _, _ = dr.scores(dv.transform(s["cal"]))
    original = np.load(BASE_RUN / "predictions_all_epochs.npz")
    old_cuts = json.loads((BASE_RUN / "known_calibration_all_epochs.json").read_text())[
        "cuts"
    ][-1]
    cut, dc = thresholds(
        pa,
        sa,
        k["yc"],
        pb,
        sb,
        s["yc"],
        da[7]["density"],
        db[7]["density"],
        cfg["alpha"],
        old_cuts,
    )
    bv = br = None
    bc = None
    if cfg["ben_view"] != "none":
        bv = View("CIC2", cfg["ben_view"]).fit(x, y)
        br = BenignSupport().fit(bv.transform(x), y)
        aa = br.score(bv.transform(k["cal"]))
        bb = br.score(bv.transform(s["cal"]))
        n = cfg["ben_k"]
        bc = guard_cut(aa[n], k["yc"], bb[n], s["yc"], cfg["ben_attack_allowance"])
    joblib.dump(dict(view=bv, reference=br), ROOT / "benign_support.joblib")
    save_json(
        ROOT / "FROZEN_BEFORE_QUERY.json",
        dict(
            selection_sha256=digest(SELECT / "FROZEN_SELECTION.json"),
            config=cfg,
            base_probability_cuts=cut.tolist(),
            density_cut=dc,
            benign_cut=bc,
            base_model_sha256=history[0]["Model_SHA256"],
            base_predictions_sha256=digest(BASE_RUN / "predictions_all_epochs.npz"),
            benign_reference_sha256=digest(ROOT / "benign_support.joblib"),
            query_labels_used=False,
        ),
    )
    raw, ids = load_query("CIC2")
    result = {}
    raw_scores = {}
    seconds = {}
    for role, values, pred, score in [
        ("query", raw, original["query_fine"], original["query_score"]),
        ("train", k["x"], original["train_fine"], original["train_score"]),
        ("cal", k["cal"], original["cal_fine"], original["cal_score"]),
    ]:
        begin = time.perf_counter()
        density, _, _ = dr.scores(dv.transform(values))
        ben = br.score(bv.transform(values))[cfg["ben_k"]] if br is not None else None
        rows = [
            predictions(pred[i], score[i], density[7]["density"], cut, dc, ben, bc)
            for i in range(100)
        ]
        result[role] = np.stack(rows)
        raw_scores[role + "_density"] = density[7]["density"]
        raw_scores[role + "_benign"] = (
            ben if ben is not None else np.full(len(values), np.nan)
        )
        seconds[role] = time.perf_counter() - begin
    np.savez_compressed(
        ROOT / "predictions_all_epochs.npz",
        **result,
        fit_ids=k["ids"],
        cal_ids=k["cal_ids"],
        query_ids=ids,
    )
    np.savez_compressed(
        ROOT / "guard_scores_before_truth.npz", **raw_scores, query_ids=ids
    )
    shutil.copyfile(
        BASE_RUN / "training_history.csv", ROOT / "base_training_history.csv"
    )
    old = json.loads((BASE_RUN / "run_manifest.json").read_text())
    elapsed = time.perf_counter() - started
    save_json(
        ROOT / "run_manifest.json",
        dict(
            model="W-SVM-DCA-LF",
            setting="CIC2",
            epochs=100,
            row_semantics="known-only low-FPR calibration applied to 100 actual native W-SVM fits",
            training_complete=True,
            config=cfg,
            base_probability_cuts=cut.tolist(),
            density_cut=dc,
            benign_cut=bc,
            base_model=str(BASE_RUN.relative_to(BASE)),
            enhancement_seconds=elapsed,
            total_seconds=old["total_seconds"] + elapsed,
            base_training_and_scoring_seconds=old["total_seconds"],
            selection_seconds=selection["selection_seconds"],
            query_truth_read_during_training=False,
            selected_epoch=100,
            transductive=True,
            published=False,
            final_test_executed=False,
        ),
    )
    print(
        "COMPLETE low-FPR predictions; truth unopened; elapsed",
        round(elapsed, 2),
        flush=True,
    )


def report():
    import export_results as ex

    truth = np.load(DATA / "CIC2/truth.npz")
    new = np.load(ROOT / "predictions_all_epochs.npz")
    old = np.load(OLD / "predictions_all_epochs.npz")
    known = np.load(DATA / "CIC2/known.npz")
    y = truth["truth"]
    yt = (known["y_fit"] != 0).astype("int8")
    assert np.array_equal(truth["ids"], new["query_ids"]) and np.array_equal(
        new["query_ids"], old["query_ids"]
    )
    rows = []
    for epoch in range(100):
        r = dict(Epochs=epoch + 1)
        tm = ex.metrics(yt, new["train"][epoch])
        qm = ex.metrics(y, new["query"][epoch])
        for metric in ["Accuracy", "Precision", "Recall", "F1", "FPR"]:
            suffix = "" if metric == "Accuracy" else "_" + metric
            r["W-SVM-DCA-LF_Train" + suffix] = tm[metric]
            r["W-SVM-DCA-LF_Test" + suffix] = qm[metric]
        rows.append(r)
    ex.write_csv(ROOT / "experiment_B_transfer_accuracy.csv", rows)
    before = ex.metrics(y, old["query"][-1])
    after = ex.metrics(y, new["query"][-1])
    records = [
        dict(Version=v, **m)
        for v, m in [("previous_W-SVM-DCA", before), ("new_W-SVM-DCA-LF", after)]
    ]
    ex.write_csv(ROOT / "before_after_metrics.csv", records)
    ex.write_csv(
        ROOT / "experiment_B_transfer_metrics.csv",
        [dict(Model="W-SVM-DCA-LF", **after)],
    )
    ex.write_csv(
        ROOT / "predictions_final.csv",
        [
            dict(Sample_ID=str(i), True_Class=int(t), Predicted_Class=int(p))
            for i, t, p in zip(truth["ids"], y, new["query"][-1])
        ],
    )
    cm = ex.confusion_matrix(y, new["query"][-1], labels=[0, 1, 2]).T
    (ROOT / "Matrix.txt").write_text(
        "Rows=prediction columns=truth; BEN KNA UNA\n"
        + "\n".join(" ".join(map(str, a)) for a in cm)
        + "\n",
        encoding="utf-8",
    )
    save_json(
        ROOT / "WHOLE_VECTOR_REVIEW.json",
        dict(
            before=before,
            after=after,
            deltas={
                key: after[key] - before[key]
                for key in [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "FPR",
                    "BEN_Correct",
                    "KNA_Correct",
                    "UNA_Correct",
                ]
            },
            same_full_query_ids=True,
            whole_vector=True,
            query_sample_count=len(y),
            old_result_preserved=True,
            scope="observed development only",
        ),
    )
    print(json.dumps(dict(before=before, after=after), indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["select", "apply", "report"])
    a = p.parse_args()
    with threadpool_limits(limits=4):
        {"select": select, "apply": apply, "report": report}[a.phase]()
