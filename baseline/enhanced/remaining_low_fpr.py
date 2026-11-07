"""Known-only low-FPR calibration for the four requested EVM/CVAE settings.

Selection never loads current query features or truth. Models, support rules,
and all cutoffs freeze before apply opens unlabeled current features. The
report phase alone opens current labels and reports full-vector tradeoffs.
"""

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import joblib
import torch
from threadpoolctl import threadpool_limits
from common import *
from models import EVM, CVAE
from density_enhancement import DensityReference
from cic2_anchored_density import controls
from reduce_cic2_wsvm_fpr import BenignSupport

TARGETS = (("CIC2", "CVAE-EVT"), ("CIC2", "EVM"), ("NSL2", "EVM"), ("NSL2", "CVAE-EVT"))
SELECTION = HERE / "selection_remaining_low_fpr"
ROOT = BASE / "results/REMAINING_FOUR_LOW_FPR_20261102"
OLD = BASE / "results/ENHANCED_BASELINES_LOW_FPR_20261102"
BASE_ROOT = BASE / "results/comparison_20261102_calibrated"
BASE_ALPHA = (0.005, 0.01, 0.025, 0.05, 0.1)
DENSITY_ALPHA = (0.01, 0.025, 0.05, 0.1)
SUPPORT_VIEWS = ("log_standard", "log_robust")
SUPPORT_K = (3, 7, 15)


def calibration_sources(setting, k):
    result = [dict(x=k["cal"], y=k["yc"], ids=k["cal_ids"], name="original_known_CAL")]
    supplemental = None
    if setting == "CIC2":
        supplemental = controls()
        result.append(
            dict(
                x=supplemental["cal"],
                y=supplemental["yc"],
                ids=supplemental["cal_ids"],
                name="source_matched_known_CAL",
            )
        )
    return result, supplemental


def geometry_recipe(setting):
    if setting == "CIC2":
        return dict(view="log_whiten", k=7)
    return dict(view="log_robust", k=3)


def base_recipe(setting, model):
    source = RECIPES / "selection" / setting / model / "FROZEN_SELECTION.json"
    a = json.loads(source.read_text())
    return a["chosen"]["config"], float(a["chosen"]["alpha"]), digest(source)


def build_base(setting, model, cfg):
    params = {k: v for k, v in cfg.items() if k != "view"}
    return EVM(**params) if model == "EVM" else CVAE(setting, **params)


def calibrate_base(cal, original, alpha):
    cuts = np.asarray(original).copy()
    candidates = []
    for a in cal:
        score = a["score"][(a["y"] == 0) & (a["pred"] == 0)]
        if len(score) < 20:
            raise ValueError("Not enough BEN calibration evidence")
        candidates.append(float(np.quantile(score, alpha, method="lower")))
    cuts[0] = min(candidates)
    return cuts


def calibrate_density(cal, alpha):
    vals = []
    for a in cal:
        score = a["density"][(a["y"] == 0) & np.isfinite(a["density"])]
        if len(score) < 20:
            raise ValueError("Not enough density BEN calibration evidence")
        vals.append(float(np.quantile(score, 1 - alpha, method="higher")))
    return max(vals)


def calibrate_support(cal, view, n, allowance):
    attack = np.concatenate([a["support"][view][n][a["y"] != 0] for a in cal])
    if not len(attack):
        raise ValueError("Missing known attack support controls")
    return float(np.quantile(attack, allowance, method="lower"))


def decide(
    pred,
    score,
    density,
    cuts,
    dc,
    support=None,
    sc=None,
    scope="predicted_ben",
    guard_order="after_density",
):
    output = np.where(pred == 0, 0, 1).astype("int8")
    output[score < cuts[pred]] = 2
    output[density > dc] = 2
    if support is not None:
        safe = support < sc
        if scope == "predicted_ben":
            safe &= pred == 0
        output[safe] = 0
    if guard_order == "before_density":
        output[density > dc] = 2
    return output


def candidates():
    guards = [
        dict(
            ben_view="none",
            ben_k=0,
            ben_attack_allowance=0.0,
            guard_scope="predicted_ben",
        )
    ]
    guards.extend(
        dict(ben_view=v, ben_k=k, ben_attack_allowance=a, guard_scope=scope)
        for v in SUPPORT_VIEWS
        for k in SUPPORT_K
        for a in (0.0, 0.005, 0.01)
        for scope in ("predicted_ben", "any_prediction")
    )
    return [
        dict(base_alpha=a, density_alpha=d, **guard)
        for a in BASE_ALPHA
        for d in DENSITY_ALPHA
        for guard in guards
    ]


def summarize_fold(y, p, proxy, domains):
    fprs = [
        float(np.mean(p[(y == 0) & (domains == d)] != 0)) for d in np.unique(domains)
    ]
    attack_recalls = [float(np.mean(p[y == c] == 1)) for c in np.unique(y) if c != 0]
    return dict(
        max_source_BEN_FPR=max(fprs),
        source_BEN_FPR=fprs,
        known_KNA_macro_recall=float(np.mean(attack_recalls)),
        proxy_UNA_recall=float(np.mean(proxy == 2)),
        known_coarse_accuracy=float(np.mean(p == (y != 0))),
    )


def select(setting, name):
    start = time.perf_counter()
    out = SELECTION / setting / name
    out.mkdir(parents=True, exist_ok=False)
    cfg, base_alpha, base_sha = base_recipe(setting, name)
    k = load_known(setting)
    sources, supplemental = calibration_sources(setting, k)
    ref_x = (
        k["x"] if supplemental is None else np.concatenate([k["x"], supplemental["x"]])
    )
    ref_y = k["y"] if supplemental is None else np.r_[k["y"], supplemental["y"]]
    folds = heldout_families(k["y"], 3)
    grid = candidates()
    stats = [[] for _ in grid]
    gr = geometry_recipe(setting)
    save_json(
        out / "BEFORE_SELECTION.json",
        dict(
            setting=setting,
            model=name,
            base_recipe=cfg,
            base_recipe_sha256=base_sha,
            geometry_recipe=gr,
            held_known_families=folds,
            candidates=grid,
            max_known_BEN_FPR=0.03,
            criterion="among candidates with max per-source known BEN FPR <= 3%, maximize mean known-KNA macro recall and held-known-family UNA recall",
            query_features_read=False,
            query_labels_read=False,
            known_sha256=digest(DATA / setting / "known.npz"),
            supplemental_source_reference_rows=(
                0 if supplemental is None else len(supplemental["x"])
            ),
        ),
    )
    for held in folds:
        fm = k["y"] != held
        rm = ref_y != held
        view = View(setting, cfg["view"]).fit(k["x"][fm], k["y"][fm])
        base = build_base(setting, name, cfg)
        fit = view.transform(k["x"][fm])
        path = out / f"held_{held}"
        path.mkdir(exist_ok=True)
        if name == "CVAE-EVT":
            base.fit(fit, k["y"][fm], seed=1234, path=path / "training")
        else:
            base.fit(fit, k["y"][fm], seed=1234)
        cal_raw = [
            dict(x=a["x"][a["y"] != held], y=a["y"][a["y"] != held], name=a["name"])
            for a in sources
        ]
        proxy_parts = [k["x"][~fm]] + [a["x"][a["y"] == held] for a in sources]
        if supplemental is not None:
            proxy_parts.append(supplemental["x"][supplemental["y"] == held])
        proxy = np.concatenate(proxy_parts)
        mixed = np.concatenate([a["x"] for a in cal_raw] + [proxy])
        pm, sm = base.scores(view.transform(mixed))
        cal = []
        for a in cal_raw:
            p, s = base.scores(view.transform(a["x"]))
            cal.append(dict(a, pred=p, score=s, support={}))
        all_y = np.concatenate([a["y"] for a in cal])
        all_p = np.concatenate([a["pred"] for a in cal])
        all_s = np.concatenate([a["score"] for a in cal])
        oldcuts, _ = probability_cuts(all_s, all_p, all_y, base_alpha)
        dv = View(setting, gr["view"]).fit(ref_x[rm], ref_y[rm])
        dr = DensityReference().fit(dv.transform(ref_x[rm]), ref_y[rm])
        ds, _, _ = dr.scores(dv.transform(mixed))
        dm = ds[gr["k"]]["density"]
        for a in cal:
            d, _, _ = dr.scores(dv.transform(a["x"]))
            a["density"] = d[gr["k"]]["density"]
        support_mixed = {}
        for v in SUPPORT_VIEWS:
            bv = View(setting, v).fit(ref_x[rm], ref_y[rm])
            br = BenignSupport().fit(bv.transform(ref_x[rm]), ref_y[rm])
            support_mixed[v] = br.score(bv.transform(mixed))
            for a in cal:
                a["support"][v] = br.score(bv.transform(a["x"]))
        domains = np.concatenate(
            [np.full(len(a["y"]), i, dtype="int64") for i, a in enumerate(cal)]
        )
        # Calibration arrays cache all scores before any current query data.
        np.savez_compressed(
            path / "known_proxy_arrays.npz",
            pred=pm,
            score=sm,
            density=dm,
            known_y=all_y,
            domains=domains,
            proxy_rows=len(proxy),
            **{
                v + "_" + str(n): support_mixed[v][n]
                for v in SUPPORT_VIEWS
                for n in SUPPORT_K
            },
        )
        for i, c in enumerate(grid):
            cuts = calibrate_base(cal, oldcuts, c["base_alpha"])
            dc = calibrate_density(cal, c["density_alpha"])
            bs = bc = None
            if c["ben_view"] != "none":
                bs = support_mixed[c["ben_view"]][c["ben_k"]]
                bc = calibrate_support(
                    cal, c["ben_view"], c["ben_k"], c["ben_attack_allowance"]
                )
            p = decide(
                pm,
                sm,
                dm,
                cuts,
                dc,
                bs,
                bc,
                c["guard_scope"],
                c.get("guard_order", "after_density"),
            )
            stats[i].append(
                summarize_fold(all_y, p[: len(all_y)], p[len(all_y) :], domains)
            )
        if name == "CVAE-EVT":
            save_json(path / "history.json", base.history)
        print(
            setting,
            name,
            "known-only fold",
            held,
            "seconds",
            round(time.perf_counter() - start, 1),
            flush=True,
        )
    trials = []
    for c, values in zip(grid, stats):
        r = dict(
            config=c,
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
    feasible = [t for t in trials if t["BEN_gate_passed"]]
    best = (
        max(feasible, key=lambda a: (a["objective"], -a["max_source_BEN_FPR"]))
        if feasible
        else min(trials, key=lambda a: (a["max_source_BEN_FPR"], -a["objective"]))
    )
    save_json(
        out / "FROZEN_SELECTION.json",
        dict(
            setting=setting,
            model=name,
            chosen=best,
            trials=trials,
            base_recipe=cfg,
            base_alpha=base_alpha,
            geometry_recipe=gr,
            selection_seconds=time.perf_counter() - start,
            query_values_read=False,
            query_labels_used=False,
            known_sha256=digest(DATA / setting / "known.npz"),
        ),
    )
    print("FROZEN", setting, name, json.dumps(best), flush=True)


def load_final_base(setting, name, cfg):
    source = BASE_ROOT / setting / name
    if name == "EVM":
        p = np.load(source / "model_final.npz")
        m = EVM(
            nfuse=int(p["nfuse"]),
            tail_size=int(p["tail_size"]),
            margin_scale=float(p["margin_scale"]),
        )
        m.points = p["points"]
        m.params = p["params"]
        m.labels = p["labels"]
        return m
    from verify_models import load_cvae

    return load_cvae(source, setting)


def final_apply(setting, name):
    started = time.perf_counter()
    sp = SELECTION / setting / name / "FROZEN_SELECTION.json"
    sel = json.loads(sp.read_text())
    cfg = sel["chosen"]["config"]
    k = load_known(setting)
    sources, supp = calibration_sources(setting, k)
    ref_x = k["x"] if supp is None else np.concatenate([k["x"], supp["x"]])
    ref_y = k["y"] if supp is None else np.r_[k["y"], supp["y"]]
    source = BASE_ROOT / setting / name
    original = np.load(source / "predictions_all_epochs.npz")
    view = joblib.load(source / "preprocessing.joblib")
    out = ROOT / setting / name
    out.mkdir(parents=True, exist_ok=False)
    gr = sel["geometry_recipe"]
    dv = View(setting, gr["view"]).fit(ref_x, ref_y)
    dr = DensityReference().fit(dv.transform(ref_x), ref_y)
    cal = []
    for a in sources:
        ds, _, _ = dr.scores(dv.transform(a["x"]))
        cal.append(dict(a, density=ds[gr["k"]]["density"], support={}))
    bv = br = None
    bc = None
    if cfg["ben_view"] != "none":
        bv = View(setting, cfg["ben_view"]).fit(ref_x, ref_y)
        br = BenignSupport().fit(bv.transform(ref_x), ref_y)
        for a in cal:
            a["support"][cfg["ben_view"]] = br.score(bv.transform(a["x"]))
        bc = calibrate_support(
            cal, cfg["ben_view"], cfg["ben_k"], cfg["ben_attack_allowance"]
        )
    dc = calibrate_density(cal, cfg["density_alpha"])
    joblib.dump(
        dict(
            density_view=dv, density_reference=dr, benign_view=bv, benign_reference=br
        ),
        out / "guard_reference.joblib",
    )
    save_json(
        out / "FROZEN_BEFORE_QUERY.json",
        dict(
            config=cfg,
            selection_sha256=digest(sp),
            density_cut=dc,
            benign_cut=bc,
            geometry_k=gr["k"],
            known_sha256=digest(DATA / setting / "known.npz"),
            guard_reference_sha256=digest(out / "guard_reference.joblib"),
            query_labels_used=False,
            source_control_count=0 if supp is None else len(supp["cal"]),
        ),
    )
    raw, ids = load_query(setting)
    assert np.array_equal(ids, original["query_ids"])
    raws = {"train": k["x"], "cal": k["cal"], "query": raw}
    role_features = {r: view.transform(a) for r, a in raws.items()}
    dscores = {}
    bscores = {}
    for role, a in raws.items():
        dd, _, _ = dr.scores(dv.transform(a))
        dscores[role] = dd[gr["k"]]["density"]
        bscores[role] = (
            br.score(bv.transform(a))[cfg["ben_k"]] if br is not None else None
        )
    new = {r: [] for r in raws}
    preds = {r: [] for r in raws}
    scores = {r: [] for r in raws}
    cut_rows = []
    records = []
    supplemental_scores = []
    supplemental_pred = []

    def consume(
        epoch, base_predictions, base_scores, extra_pred=None, extra_score=None
    ):
        cal[0]["pred"] = base_predictions["cal"]
        cal[0]["score"] = base_scores["cal"]
        if supp is not None:
            cal[1]["pred"] = extra_pred
            cal[1]["score"] = extra_score
            supplemental_scores.append(extra_score.copy())
            supplemental_pred.append(extra_pred.copy())
        all_scores = np.concatenate([a["score"] for a in cal])
        all_pred = np.concatenate([a["pred"] for a in cal])
        all_y = np.concatenate([a["y"] for a in cal])
        # All class cuts are frozen by true-known controls; only BEN coverage
        # is made stricter through the selected nominal false-rejection rate.
        original_cuts, _ = (
            probability_cuts(
                original["cal_score"][epoch - 1],
                original["cal_fine"][epoch - 1],
                k["yc"],
                sel["base_alpha"],
            )
            if setting == "NSL2"
            else probability_cuts(all_scores, all_pred, all_y, sel["base_alpha"])
        )
        cuts = calibrate_base(cal, original_cuts, cfg["base_alpha"])
        cut_rows.append(cuts.tolist())
        for role in raws:
            p = base_predictions[role]
            sc = base_scores[role]
            new[role].append(
                decide(
                    p,
                    sc,
                    dscores[role],
                    cuts,
                    dc,
                    bscores[role],
                    bc,
                    cfg["guard_scope"],
                    cfg.get("guard_order", "after_density"),
                )
            )
            preds[role].append(p.copy())
            scores[role].append(sc.copy())

    if setting == "NSL2":
        # All required known and query scores were saved for each real epoch.
        for epoch in range(1, 101):
            consume(
                epoch,
                {r: original[r + "_fine"][epoch - 1] for r in raws},
                {r: original[r + "_score"][epoch - 1] for r in raws},
            )
            records.append(
                dict(Epochs=epoch, Elapsed_Seconds=time.perf_counter() - started)
            )
        base_wall = json.loads((source / "run_manifest.json").read_text())[
            "total_seconds"
        ]
        model_origin = "saved_100_actual_base_states"
    else:
        x = role_features["train"]
        y = k["y"]
        extra = view.transform(supp["cal"])
        base_cfg = sel["base_recipe"]
        if name == "EVM":
            from scipy.spatial.distance import cdist

            distances = cdist(x.astype("float64"), x.astype("float64"))
            base_started = time.perf_counter()
            for epoch in range(1, 101):
                m = build_base(setting, name, base_cfg).fit(
                    x, y, seed=1234 + epoch, distances=distances
                )
                pp = {}
                ss = {}
                for role, a in role_features.items():
                    pp[role], ss[role] = m.scores(a)
                ep, es = m.scores(extra)
                consume(epoch, pp, ss, ep, es)
                records.append(
                    dict(
                        Epochs=epoch,
                        Elapsed_Seconds=time.perf_counter() - started,
                        Extreme_Vectors=len(m.points),
                    )
                )
                if epoch % 10 == 0:
                    save_json(out / "progress.json", records[-1])
                    print(
                        setting,
                        name,
                        "actual refit",
                        epoch,
                        "seconds",
                        round(records[-1]["Elapsed_Seconds"], 1),
                        flush=True,
                    )
            m.save(out / "base_model_final.npz")
            base_wall = time.perf_counter() - base_started
        else:
            m = build_base(setting, name, base_cfg)
            base_started = time.perf_counter()

            def callback(model, epoch, history):
                model.fit_tails()
                pp = {}
                ss = {}
                for role, a in role_features.items():
                    pp[role], ss[role] = model.scores(a)
                ep, es = model.scores(extra)
                consume(epoch, pp, ss, ep, es)
                records.append(
                    dict(
                        Epochs=epoch,
                        Loss=history["loss"],
                        Elapsed_Seconds=time.perf_counter() - started,
                    )
                )
                if epoch % 10 == 0:
                    save_json(out / "progress.json", records[-1])
                    print(
                        setting,
                        name,
                        "epoch",
                        epoch,
                        "seconds",
                        round(records[-1]["Elapsed_Seconds"], 1),
                        flush=True,
                    )

            m.fit(x, y, seed=1234, path=out / "training", callback=callback)
            m.save(out / "base_model_final.pt")
            base_wall = time.perf_counter() - base_started
            save_json(out / "base_stage_histories.json", m.history)
        model_origin = "fresh_100_base_states_with_extra_known_source_CAL_scores"
    fields = {
        **{r: np.stack(a) for r, a in new.items()},
        **{r + "_fine": np.stack(a) for r, a in preds.items()},
        **{r + "_score": np.stack(a) for r, a in scores.items()},
    }
    np.savez_compressed(
        out / "predictions_all_epochs.npz",
        **fields,
        query_ids=ids,
        fit_ids=k["ids"],
        cal_ids=k["cal_ids"],
    )
    np.savez_compressed(
        out / "guard_scores_before_truth.npz",
        **{r + "_density": v for r, v in dscores.items()},
        **{
            r + "_benign": v if v is not None else np.full(len(raws[r]), np.nan)
            for r, v in bscores.items()
        },
        query_ids=ids,
    )
    if supplemental_scores:
        np.savez_compressed(
            out / "source_known_CAL_scores.npz",
            scores=np.stack(supplemental_scores),
            predicted_fine=np.stack(supplemental_pred),
            ids=supp["cal_ids"],
            y_known=supp["yc"],
        )
    save_json(
        out / "calibration_all_epochs.json",
        dict(cuts=cut_rows, density_cut=dc, benign_cut=bc, config=cfg),
    )
    with (out / "training_history.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    if setting == "NSL2":
        shutil.copyfile(
            source / "training_history.csv", out / "base_training_history.csv"
        )
    elapsed = time.perf_counter() - started
    save_json(
        out / "run_manifest.json",
        dict(
            setting=setting,
            model=name + ("-DCA-LF" if setting == "CIC2" else "-DC-LF"),
            family=name,
            epochs=100,
            selected_epoch=100,
            training_complete=True,
            config=cfg,
            model_origin=model_origin,
            base_source=str(source.relative_to(BASE) if source.is_relative_to(BASE) else source.relative_to(ARCHIVE)),
            base_source_sha256=digest(source / "predictions_all_epochs.npz"),
            selection_sha256=digest(sp),
            operation_seconds=elapsed,
            base_training_and_scoring_seconds=base_wall,
            total_seconds=elapsed if setting == "CIC2" else elapsed + base_wall,
            selection_seconds=sel["selection_seconds"],
            query_truth_read_during_training=False,
            query_labels_used_for_calibration=False,
            transductive=True,
            published=False,
            final_test_executed=False,
        ),
    )
    print(
        "COMPLETE",
        setting,
        name,
        "predictions before truth",
        round(elapsed, 2),
        flush=True,
    )


def report_one(setting, name):
    import export_results as ex

    out = ROOT / setting / name
    manifest = json.loads((out / "run_manifest.json").read_text())
    model = manifest["model"]
    z = np.load(out / "predictions_all_epochs.npz")
    t = np.load(DATA / setting / "truth.npz")
    k = np.load(DATA / setting / "known.npz")
    y = t["truth"]
    yt = (k["y_fit"] != 0).astype("int8")
    old_model = name + ("-DCA" if setting == "CIC2" else "-DC")
    old = np.load(OLD / setting / old_model / "predictions_all_epochs.npz")
    assert np.array_equal(t["ids"], z["query_ids"]) and np.array_equal(
        z["query_ids"], old["query_ids"]
    )
    rows = []
    for i in range(100):
        tr = ex.metrics(yt, z["train"][i])
        te = ex.metrics(y, z["query"][i])
        row = dict(Epochs=i + 1)
        for metric in ["Accuracy", "Precision", "Recall", "F1", "FPR"]:
            suffix = "" if metric == "Accuracy" else "_" + metric
            row[model + "_Train" + suffix] = tr[metric]
            row[model + "_Test" + suffix] = te[metric]
        rows.append(row)
    ex.write_csv(out / "experiment_B_transfer_accuracy.csv", rows)
    before = ex.metrics(y, old["query"][-1])
    after = ex.metrics(y, z["query"][-1])
    ex.write_csv(
        out / "before_after_metrics.csv",
        [dict(Version="previous", **before), dict(Version="new", **after)],
    )
    ex.write_csv(
        out / "experiment_B_transfer_metrics.csv",
        [dict(Model=model, Setting=setting, **after)],
    )
    ex.write_csv(
        out / "predictions_final.csv",
        [
            dict(Sample_ID=str(i), True_Class=int(t), Predicted_Class=int(p))
            for i, t, p in zip(t["ids"], y, z["query"][-1])
        ],
    )
    cm = ex.confusion_matrix(y, z["query"][-1], labels=[0, 1, 2]).T
    (out / "Matrix.txt").write_text(
        "Rows=prediction; columns=truth; BEN KNA UNA\n"
        + "\n".join(" ".join(map(str, a)) for a in cm)
        + "\n",
        encoding="utf-8",
    )
    save_json(
        out / "WHOLE_VECTOR_REVIEW.json",
        dict(
            before=before,
            after=after,
            same_ids=True,
            rows=len(y),
            deltas={
                n: after[n] - before[n]
                for n in [
                    "Accuracy",
                    "F1",
                    "FPR",
                    "BEN_Correct",
                    "KNA_Correct",
                    "UNA_Correct",
                ]
            },
            accuracy_above70=after["Accuracy"] > 0.7,
            lower_fpr=after["FPR"] < before["FPR"],
            scope="observed development",
        ),
    )
    print(setting, name, json.dumps(dict(before=before, after=after)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["select", "apply", "report"])
    p.add_argument("--model", choices=["EVM", "CVAE-EVT"])
    p.add_argument("--setting", choices=["CIC2", "NSL2"])
    a = p.parse_args()
    targets = [
        (s, m)
        for s, m in TARGETS
        if (a.model is None or m == a.model) and (a.setting is None or s == a.setting)
    ]
    with threadpool_limits(limits=4):
        for s, m in targets:
            {"select": select, "apply": final_apply, "report": report_one}[a.phase](
                s, m
            )
