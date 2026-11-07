"""Retrain all 100 base states and replay the frozen explicit density layers."""

from datetime import datetime
import argparse
import csv
import json
import time
import shutil
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from common import *
from models import CVAE, EVM, WSVM
from density_enhancement import DensityReference, score_decision
from cic2_anchored_density import decide
from run_final import run as train_base
import export_results as report
from reduce_cic2_wsvm_fpr import predictions as low_fpr_predictions, BenignSupport

REMAINING_LOW_FPR = {
    ("CIC2", "CVAE-EVT"): "selection_remaining_low_fpr",
    ("CIC2", "EVM"): "selection_remaining_low_fpr",
    ("NSL2", "EVM"): "selection_remaining_low_fpr_strict_ben",
    ("NSL2", "CVAE-EVT"): "selection_remaining_low_fpr_order",
}


def train_remaining_low_fpr(setting, model, run_name, workers):
    """Replay the newly frozen controls with fresh real 100-state base runs."""
    import remaining_low_fpr as core

    saved = (core.SELECTION, core.ROOT, core.BASE_ROOT)
    try:
        core.SELECTION = RECIPES / REMAINING_LOW_FPR[(setting, model)]
        core.ROOT = BASE / "results" / (run_name + "_low_fpr_states")
        if setting == "NSL2":
            train_base(
                setting,
                model,
                workers,
                selection_group="selection",
                run_name=run_name + "_base",
            )
            core.BASE_ROOT = BASE / "results" / (run_name + "_base")
        else:
            # CIC2 final_apply itself trains both complete CVAE stages or
            # the 100 EVM refits and scores the extra known CAL each round.
            core.BASE_ROOT = ARCHIVE / "results/comparison_20261102_calibrated"
        core.final_apply(setting, model)
        return core.ROOT / setting / model
    finally:
        core.SELECTION, core.ROOT, core.BASE_ROOT = saved


def apply_low_fpr(base_predictions):
    start = time.perf_counter()
    source = ARCHIVE / "results/CIC2_WSVM_LOW_FPR_SELECTIVE_20261102"
    cfg = json.loads((source / "FROZEN_BEFORE_QUERY.json").read_text())
    cut = np.array(cfg["base_probability_cuts"])
    artifact = joblib.load(source / "benign_support.joblib")
    bv, br = artifact["view"], artifact["reference"]
    density = joblib.load(
        ARCHIVE / "results/comparison_20261102_source_anchored/CIC2/reference.joblib"
    )
    dv, dr = density["view"], density["reference"]
    k = load_known("CIC2")
    q, ids = load_query("CIC2")
    assert np.array_equal(ids, base_predictions["query_ids"])
    result = {}
    for role, x in [("query", q), ("train", k["x"]), ("cal", k["cal"])]:
        ds, _, _ = dr.scores(dv.transform(x))
        bs = br.score(bv.transform(x))[cfg["config"]["ben_k"]]
        result[role] = np.stack(
            [
                low_fpr_predictions(
                    base_predictions[role + "_fine"][i],
                    base_predictions[role + "_score"][i],
                    ds[7]["density"],
                    cut,
                    cfg["density_cut"],
                    bs,
                    cfg["benign_cut"],
                )
                for i in range(100)
            ]
        )
    return result, time.perf_counter() - start


def guard_for(setting, run_out):
    start = time.perf_counter()
    k = load_known(setting)
    raw, ids = load_query(setting)
    if setting == "CIC2":
        source = ARCHIVE / "results/comparison_20261102_source_anchored/CIC2"
        artifact = joblib.load(source / "reference.joblib")
        manifest = json.loads((source / "FROZEN_BEFORE_QUERY.json").read_text())
        cfg = manifest["selection"]["chosen"]
        rule = lambda a: decide(
            a[cfg["k"]], manifest["thresholds"], cfg["mode"], cfg["support_floor"]
        )
    else:
        source = ARCHIVE / "results/comparison_20261102_density_enhanced" / setting
        artifact = joblib.load(source / "known_density_reference.joblib")
        manifest = json.loads((source / "FROZEN_BEFORE_QUERY.json").read_text())
        rule = lambda a: score_decision(
            a[manifest["k"]], manifest["threshold"], manifest["min_support"]
        )
    view, reference = artifact["view"], artifact["reference"]
    masks = {}
    for role, x in [("query", raw), ("train", k["x"]), ("cal", k["cal"])]:
        scores, _, _ = reference.scores(view.transform(x))
        masks[role] = rule(scores)
    save_json(
        run_out / "DENSITY_REPLAY.json",
        dict(
            source=str(source.relative_to(ARCHIVE)),
            configuration=manifest,
            query_labels_used=False,
            transductive=True,
            seconds=time.perf_counter() - start,
        ),
    )
    return masks, time.perf_counter() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS)
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=["CVAE-EVT", "EVM", "W-SVM"],
        default=["CVAE-EVT", "EVM", "W-SVM"],
    )
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--run-name",
        default="enhanced_replay_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    a = p.parse_args()
    if a.workers < 1:
        raise ValueError("workers must be positive")
    root = BASE / "results" / a.run_name
    root.mkdir(parents=True, exist_ok=False)
    all_rows = []
    for setting in a.settings:
        setting_out = root / setting
        setting_out.mkdir()
        legacy_needed = any(
            (setting, m) not in REMAINING_LOW_FPR
            and not (setting == "CIC2" and m == "W-SVM")
            for m in a.models
        )
        masks, guard_seconds = (
            guard_for(setting, setting_out) if legacy_needed else ({}, 0.0)
        )
        known = np.load(DATA / setting / "known.npz")
        yt = (known["y_fit"] != 0).astype("int8")
        combined = [
            dict(Epochs=e + 1, Scope="observed_development") for e in range(100)
        ]
        for model in a.models:
            group = (
                "selection_whitening"
                if setting == "NSL1" and model == "W-SVM"
                else "selection"
            )
            remaining = (setting, model) in REMAINING_LOW_FPR
            if remaining:
                base = train_remaining_low_fpr(setting, model, a.run_name, a.workers)
            else:
                train_base(
                    setting,
                    model,
                    a.workers,
                    selection_group=group,
                    run_name=a.run_name + "_base",
                )
                base = BASE / "results" / (a.run_name + "_base") / setting / model
            pred = np.load(base / "predictions_all_epochs.npz")
            name = model + ("-DCA" if setting == "CIC2" else "-DC")
            low_fpr = setting == "CIC2" and model == "W-SVM"
            if low_fpr or remaining:
                name += "-LF"
            out = setting_out / name
            out.mkdir()
            values = {}
            model_guard_seconds = guard_seconds
            if remaining:
                values = {r: pred[r].copy() for r in ["train", "query", "cal"]}
                model_guard_seconds = 0.0
            elif low_fpr:
                values, model_guard_seconds = apply_low_fpr(pred)
            else:
                for role in ("train", "query", "cal"):
                    values[role] = pred[role].copy()
                    values[role][:, masks[role]] = 2
            np.savez_compressed(
                out / "predictions_all_epochs.npz",
                **values,
                fit_ids=pred["fit_ids"],
                query_ids=pred["query_ids"],
                cal_ids=pred["cal_ids"],
            )
            # Save predictions before loading evaluation labels.
            truth = np.load(DATA / setting / "truth.npz")
            assert np.array_equal(truth["ids"], pred["query_ids"])
            y = truth["truth"]
            rows = []
            for i in range(100):
                row = dict(Epochs=i + 1)
                tr = report.metrics(yt, values["train"][i])
                te = report.metrics(y, values["query"][i])
                for field in ["Accuracy", "Precision", "Recall", "F1", "FPR"]:
                    suffix = "" if field == "Accuracy" else "_" + field
                    for label, data in [("Train", tr), ("Test", te)]:
                        row[name + "_" + label + suffix] = data[field]
                        combined[i][name + "_" + label + suffix] = data[field]
                rows.append(row)
            report.write_csv(out / "experiment_B_transfer_accuracy.csv", rows)
            final = report.metrics(y, values["query"][-1])
            bm = json.loads((base / "run_manifest.json").read_text())
            record = dict(
                Setting=setting,
                Model=name,
                Epochs=100,
                **final,
                Total_Seconds=bm["total_seconds"] + model_guard_seconds,
                Scope="observed_development",
                Transductive=True,
            )
            all_rows.append(record)
            report.write_csv(out / "experiment_B_transfer_metrics.csv", [record])
            cm = report.confusion_matrix(y, values["query"][-1], labels=[0, 1, 2]).T
            (out / "Matrix.txt").write_text(
                "Rows=prediction; columns=truth; BEN KNA UNA\n"
                + "\n".join(" ".join(map(str, r)) for r in cm)
                + "\n",
                encoding="utf-8",
            )
            history_source = (
                base / "base_training_history.csv"
                if (base / "base_training_history.csv").exists()
                else base / "training_history.csv"
            )
            shutil.copyfile(history_source, out / "base_training_history.csv")
            save_json(
                out / "run_manifest.json",
                dict(
                    model=name,
                    base_run=str(base.relative_to(BASE)),
                    epochs=100,
                    training_complete=True,
                    prediction_sha256=digest(out / "predictions_all_epochs.npz"),
                    query_labels_used_for_training_or_calibration=False,
                    transductive=True,
                    total_seconds=record["Total_Seconds"],
                    source="fixed known-selected baseline config and frozen known-calibrated density enhancement",
                    published=False,
                    final_test_executed=False,
                ),
            )
            print(
                "COMPLETE enhanced replay",
                setting,
                name,
                round(final["Accuracy"] * 100, 2),
                flush=True,
            )
        report.write_csv(setting_out / "experiment_B_transfer_accuracy.csv", combined)
    report.write_csv(root / "comparison_summary.csv", all_rows)
    print("Saved replay to", root, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
