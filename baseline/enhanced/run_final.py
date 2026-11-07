"""Final 100-round runs with configurations frozen by known-only selection."""

from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import hashlib
import json
import shutil
import time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from common import *
from models import EVM, WSVM, CVAE, CVAEBC

RUN = "comparison_20261102_calibrated"


def collect(model, x, cal, yc, query, alpha, tagged=False):
    pc, sc = model.scores(cal, tag="cal") if tagged else model.scores(cal)
    cy = (
        model.calibration_labels(cal, yc)
        if hasattr(model, "calibration_labels")
        else yc
    )
    cuts, meta = probability_cuts(
        sc, pc, cy, alpha, n_classes=getattr(model, "n_classes", 9)
    )
    benign = getattr(model, "benign_clusters", 1)
    pt, st = model.scores(x, tag="train_eval") if tagged else model.scores(x)
    pq, sq = model.scores(query, tag="query") if tagged else model.scores(query)
    return dict(
        train=probability_decision(pt, st, cuts, benign),
        query=probability_decision(pq, sq, cuts, benign),
        cal=probability_decision(pc, sc, cuts, benign),
        train_fine=pt,
        query_fine=pq,
        cal_fine=pc,
        train_score=st,
        query_score=sq,
        cal_score=sc,
        cuts=cuts,
        calibration=meta,
    )


def run(setting, name, workers, selection_group="selection", run_name=RUN):
    started = time.perf_counter()
    selection_path = RECIPES / selection_group / setting / name / "FROZEN_SELECTION.json"
    selection = json.loads(selection_path.read_text("utf-8"))
    assert not selection["query_labels_read"] and not selection["query_values_read"]
    assert selection["known_sha256"] == digest(DATA / setting / "known.npz")
    cfg = selection["chosen"]["config"]
    alpha = selection["chosen"]["alpha"]
    k = load_known(setting)
    view = View(setting, cfg["view"]).fit(k["x"], k["y"])
    x = view.transform(k["x"])
    cal = view.transform(k["cal"])
    out = BASE / "results" / run_name / setting / name
    out.mkdir(parents=True, exist_ok=False)
    joblib.dump(view, out / "preprocessing.joblib")
    save_json(
        out / "FROZEN_BEFORE_QUERY.json",
        dict(
            selection_sha256=digest(selection_path),
            selection=selection,
            data_hashes=data_signature(setting),
            preprocessor_sha256=digest(out / "preprocessing.joblib"),
            query_labels_read=False,
        ),
    )
    query, query_ids = load_query(setting)
    query = view.transform(query)
    args = {key: value for key, value in cfg.items() if key != "view"}
    records = []
    predictions = []
    cuts = []
    if name.startswith("CVAE-EVT"):
        model = (CVAEBC if name == "CVAE-EVT-BC" else CVAE)(setting, **args)

        def callback(fitted, epoch, history):
            ep = time.perf_counter()
            fitted.fit_tails()
            result = collect(fitted, x, cal, k["yc"], query, alpha)
            predictions.append(result)
            cuts.append(result["cuts"].tolist())
            record = dict(
                Epochs=epoch,
                Loss=history["loss"],
                Evaluation_Seconds=time.perf_counter() - ep,
                Elapsed_Seconds=time.perf_counter() - started,
            )
            records.append(record)
            save_json(out / "progress.json", record)
            if epoch % 10 == 0:
                print(
                    setting,
                    name,
                    epoch,
                    "seconds",
                    round(record["Elapsed_Seconds"], 1),
                    flush=True,
                )

        model.fit(x, k["y"], seed=1234, path=out / "training", callback=callback)
        model.save(out / "model_final.pt")
        save_json(out / "stage1_stage2_history.json", model.history)
        save_json(out / "gpd_tails_final.json", model.tails)
        row_semantics = "stage2_training_epoch_after_100_stage1_epochs"
        fit_seconds = model.training_seconds - sum(
            r["Evaluation_Seconds"] for r in records
        )
    elif name == "EVM":
        from scipy.spatial.distance import cdist

        distances = cdist(x.astype("float64"), x.astype("float64"))
        for epoch in range(1, 101):
            ep = time.perf_counter()
            model = EVM(**args).fit(x, k["y"], seed=1234 + epoch, distances=distances)
            fit_time = time.perf_counter() - ep
            result = collect(model, x, cal, k["yc"], query, alpha)
            predictions.append(result)
            cuts.append(result["cuts"].tolist())
            record = dict(
                Epochs=epoch,
                Fit_Seconds=fit_time,
                Epoch_Seconds=time.perf_counter() - ep,
                Elapsed_Seconds=time.perf_counter() - started,
                Extreme_Vectors=len(model.points),
                Invalid_Weibull_Fits=model.invalid_fits,
            )
            records.append(record)
            save_json(out / "progress.json", record)
            if epoch % 10 == 0:
                print(
                    setting,
                    name,
                    epoch,
                    "seconds",
                    round(record["Elapsed_Seconds"], 1),
                    flush=True,
                )
        model.save(out / "model_final.npz")
        row_semantics = "actual_EVM_full_refit"
        fit_seconds = sum(r["Fit_Seconds"] for r in records)
    else:

        def batch(worker):
            result_list = []
            path = out / "workers" / str(worker)
            path.mkdir(parents=True, exist_ok=True)
            for epoch in range(worker + 1, 101, workers):
                ep = time.perf_counter()
                model = WSVM(path, **args).fit(x, k["y"])
                fit_time = time.perf_counter() - ep
                values = collect(model, x, cal, k["yc"], query, alpha, tagged=True)
                model_sha = hashlib.sha256(
                    model.model.read_bytes()
                    + Path(str(model.model) + "_one_wsvm").read_bytes()
                ).hexdigest()
                record = dict(
                    Epochs=epoch,
                    Fit_Seconds=fit_time,
                    Epoch_Seconds=time.perf_counter() - ep,
                    Elapsed_Seconds=time.perf_counter() - started,
                    Model_SHA256=model_sha,
                )
                result_list.append((record, values))
                save_json(path / "progress.json", record)
                if epoch == 100:
                    for source, target in [
                        (model.model, out / "model_final"),
                        (
                            Path(str(model.model) + "_one_wsvm"),
                            out / "model_final_one_wsvm",
                        ),
                        (path / "query.pred", out / "query.pred"),
                        (path / "train_eval.pred", out / "train.pred"),
                        (path / "cal.pred", out / "cal.pred"),
                    ]:
                        shutil.copyfile(source, target)
                if epoch % 10 == 0:
                    print(
                        setting,
                        name,
                        epoch,
                        "seconds",
                        round(record["Elapsed_Seconds"], 1),
                        flush=True,
                    )
            return result_list

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = [
                item for group in pool.map(batch, range(workers)) for item in group
            ]
        results.sort(key=lambda pair: pair[0]["Epochs"])
        records = [r for r, v in results]
        predictions = [v for r, v in results]
        cuts = [v["cuts"].tolist() for r, v in results]
        row_semantics = "actual_native_W_SVM_full_refit_parallel_workers"
        fit_seconds = sum(r["Fit_Seconds"] for r in records)
        (out / "native_train.log").write_text(
            "NATIVE TOOL LOG ONLY. Truth-free model fitting and scoring; each actual refit is recorded below.\n"
            + "".join("FIT_ROUND " + str(r["Epochs"]) + "\n" for r in records),
            encoding="utf-8",
        )
    fields = {
        key: np.stack([p[key] for p in predictions])
        for key in [
            "train",
            "query",
            "cal",
            "train_fine",
            "query_fine",
            "cal_fine",
            "train_score",
            "query_score",
            "cal_score",
        ]
    }
    np.savez_compressed(
        out / "predictions_all_epochs.npz",
        **fields,
        fit_ids=k["ids"],
        cal_ids=k["cal_ids"],
        query_ids=query_ids,
    )
    save_json(out / "known_calibration_all_epochs.json", dict(alpha=alpha, cuts=cuts))
    with (out / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    total = time.perf_counter() - started
    save_json(
        out / "run_manifest.json",
        dict(
            model=name,
            setting=setting,
            epochs=100,
            config=cfg,
            alpha=alpha,
            row_semantics=row_semantics,
            training_complete=True,
            selection_sha256=digest(selection_path),
            query_truth_read_during_training=False,
            selection_inputs="known CAL and held-out-known-family proxies only",
            selected_epoch=100,
            fit_seconds=fit_seconds,
            total_seconds=total,
            selection_seconds=selection["selection_seconds"],
            performance_target_not_used_as_a_measurement=True,
            published=False,
            final_test_executed=False,
        ),
    )
    print("COMPLETE", setting, name, round(total, 2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", required=True, choices=["CVAE-EVT", "CVAE-EVT-BC", "EVM", "W-SVM"]
    )
    p.add_argument("--settings", nargs="+", default=list(SETTINGS))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--selection-group", default="selection")
    p.add_argument("--run-name", default=RUN)
    a = p.parse_args()
    with threadpool_limits(limits=4):
        for s in a.settings:
            run(s, a.model, a.workers, a.selection_group, a.run_name)
