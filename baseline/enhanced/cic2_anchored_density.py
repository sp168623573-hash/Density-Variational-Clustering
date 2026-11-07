"""Source-matched known reference anchors with held-out known calibration.

All candidate choices use original known roles and disjoint source_train
controls. No current query truth enters selection, fitting or prediction.
"""

import argparse
import json
import hashlib
import numpy as np
import joblib
import pandas as pd
import time
import sys
import os
from threadpoolctl import threadpool_limits
from common import *
from density_enhancement import DensityReference, KS, VIEWS, cuts

ROOT = BASE / "results/comparison_20261102_source_anchored"
SELECT = HERE / "selection_source_anchored/CIC2"


def controls():
    path = (
        ARCHIVE
        / "results/comparison_20261102_density_matched/CIC2/source_known_controls.npz"
    )
    saved = np.load(path)
    workspace = Path(os.environ.get("PAPER_EXPERIMENT_ROOT", r"G:\Experiments\paper\software paper"))
    sys.path.insert(0, str(workspace))
    from paper_open_set.config import get_spec
    from paper_open_set.data import _read_source, _normalize_label

    spec = get_spec("CIC Setting2")
    frame = _read_source(spec, spec.data_paths(workspace)[0], "source_train")
    labels = frame.label.map(lambda v: _normalize_label(spec, v))
    mapping = {
        spec.benign_label: 0,
        **{c: i + 1 for i, c in enumerate(spec.known_families)},
    }
    rows = np.array([int(str(i).split(":")[2]) for i in saved["ids"]])
    fine = np.array([mapping[v] for v in labels.iloc[rows]])
    assert np.array_equal(fine != 0, saved["y_known_coarse"] != 0)
    fit = np.zeros(len(rows), bool)
    for c in np.unique(fine):
        ix = np.flatnonzero(fine == c)
        ix = sorted(
            ix,
            key=lambda i: hashlib.sha256(
                ("source-anchor-20261102|" + str(saved["ids"][i])).encode()
            ).hexdigest(),
        )
        # A singleton stays in controls. It is not copied into a reference role.
        if len(ix) >= 2:
            fit[ix[: len(ix) // 2]] = True
    return dict(
        x=saved["native"][fit],
        y=fine[fit],
        ids=saved["ids"][fit],
        cal=saved["native"][~fit],
        yc=fine[~fit],
        cal_ids=saved["ids"][~fit],
    )


def threshold_pair(parts_a, ya, parts_b, yb, alpha):
    return dict(
        density=max(
            cuts(parts_a["density"], ya, alpha), cuts(parts_b["density"], yb, alpha)
        ),
        support=max(
            cuts(parts_a["support"], ya, alpha), cuts(parts_b["support"], yb, alpha)
        ),
    )


def decide(parts, th, mode, support_floor):
    density = (parts["density"] > th["density"]) & (parts["support"] > support_floor)
    support = parts["support"] > th["support"]
    return (
        density
        if mode == "density"
        else support if mode == "support" else density | support
    )


def select():
    start = time.perf_counter()
    k = load_known("CIC2")
    s = controls()
    SELECT.mkdir(parents=True, exist_ok=False)
    x = np.concatenate([k["x"], s["x"]])
    y = np.concatenate([k["y"], s["y"]])
    folds = heldout_families(y, 3)
    alphas = [0.01, 0.025, 0.05, 0.1]
    modes = ["density", "support", "union"]
    floors = [0.0, 0.2, 0.5]
    save_json(
        SELECT / "BEFORE_SELECTION.json",
        dict(
            source_reference_rows=len(s["x"]),
            source_cal_rows=len(s["cal"]),
            source_ref_ids_sha256=hashlib.sha256(
                "\n".join(s["ids"]).encode()
            ).hexdigest(),
            held_families=folds,
            views=VIEWS,
            ks=KS,
            alphas=alphas,
            modes=modes,
            support_floors=floors,
            query_values_read=False,
            query_truth_read=False,
            split_rule="deterministic half of each genuine source-known class, by SHA256; singleton remains CAL",
        ),
    )
    trials = []
    for v in VIEWS:
        records = {
            (n, a, m, f): [] for n in KS for a in alphas for m in modes for f in floors
        }
        for held in folds:
            fm = y != held
            ca = k["yc"] != held
            cb = s["yc"] != held
            view = View("CIC2", v).fit(x[fm], y[fm])
            ref = DensityReference().fit(view.transform(x[fm]), y[fm])
            a = view.transform(k["cal"][ca])
            b = view.transform(s["cal"][cb])
            proxy = view.transform(
                np.concatenate([x[~fm], k["cal"][~ca], s["cal"][~cb]])
            )
            aa, _, _ = ref.scores(a)
            bb, _, _ = ref.scores(b)
            joined = np.concatenate([a, b, proxy])
            jj, pred, _ = ref.scores(joined)
            yknown = np.concatenate([k["yc"][ca], s["yc"][cb]])
            for n, alpha, mode, floor in records:
                effective_alpha = alpha / 2 if mode == "union" else alpha
                th = threshold_pair(
                    aa[n], k["yc"][ca], bb[n], s["yc"][cb], effective_alpha
                )
                mask = decide(jj[n], th, mode, floor)
                p = np.where(pred == 0, 0, 1).astype("int8")
                p[mask] = 2
                records[(n, alpha, mode, floor)].append(
                    selector_score(yknown, p[: len(yknown)], p[len(yknown) :])
                )
        for (n, alpha, mode, floor), values in records.items():
            t = dict(
                view=v, k=n, alpha=alpha, mode=mode, support_floor=floor, folds=values
            )
            t.update(
                {name: float(np.mean([r[name] for r in values])) for name in values[0]}
            )
            trials.append(t)
        save_json(SELECT / "trials.json", trials)
        best = max(
            trials, key=lambda q: (q["objective"], q["known_macro_recall"], -q["alpha"])
        )
        print(
            "source anchored view",
            v,
            "known proxy",
            round(best["objective"], 4),
            "seconds",
            round(time.perf_counter() - start, 1),
            flush=True,
        )
    best = max(
        trials, key=lambda q: (q["objective"], q["known_macro_recall"], -q["alpha"])
    )
    save_json(
        SELECT / "FROZEN_SELECTION.json",
        dict(
            chosen=best,
            selection_seconds=time.perf_counter() - start,
            query_values_read=False,
            query_truth_read=False,
            original_known_sha256=digest(DATA / "CIC2/known.npz"),
            source_controls_sha256=digest(
                BASE
                / "results/comparison_20261102_density_matched/CIC2/source_known_controls.npz"
            ),
        ),
    )
    print(
        "FROZEN source anchored",
        {
            a: best[a]
            for a in ["view", "k", "alpha", "mode", "support_floor", "objective"]
        },
        flush=True,
    )


def apply():
    start = time.perf_counter()
    sel = json.loads((SELECT / "FROZEN_SELECTION.json").read_text())
    cfg = sel["chosen"]
    k = load_known("CIC2")
    s = controls()
    x = np.concatenate([k["x"], s["x"]])
    y = np.concatenate([k["y"], s["y"]])
    view = View("CIC2", cfg["view"]).fit(x, y)
    ref = DensityReference().fit(view.transform(x), y)
    a = view.transform(k["cal"])
    b = view.transform(s["cal"])
    aa, _, _ = ref.scores(a)
    bb, _, _ = ref.scores(b)
    alpha = cfg["alpha"] / 2 if cfg["mode"] == "union" else cfg["alpha"]
    threshold = threshold_pair(aa[cfg["k"]], k["yc"], bb[cfg["k"]], s["yc"], alpha)
    out = ROOT / "CIC2"
    out.mkdir(parents=True, exist_ok=False)
    joblib.dump(dict(view=view, reference=ref), out / "reference.joblib")
    np.savez_compressed(
        out / "source_role_manifest.npz",
        reference_ids=s["ids"],
        reference_y=s["y"],
        calibration_ids=s["cal_ids"],
        calibration_y=s["yc"],
    )
    save_json(
        out / "FROZEN_BEFORE_QUERY.json",
        dict(
            selection=sel,
            thresholds=threshold,
            source_reference_count=len(s["x"]),
            source_cal_count=len(s["cal"]),
            original_FIT_count=len(k["x"]),
            original_CAL_count=len(k["cal"]),
            query_labels_used=False,
            transductive=True,
            reference_sha256=digest(out / "reference.joblib"),
        ),
    )
    q, ids = load_query("CIC2")
    qs, qp, qm = ref.scores(view.transform(q))
    mask = decide(qs[cfg["k"]], threshold, cfg["mode"], cfg["support_floor"])
    fs, _, _ = ref.scores(view.transform(k["x"]))
    fm = decide(fs[cfg["k"]], threshold, cfg["mode"], cfg["support_floor"])
    cm = decide(aa[cfg["k"]], threshold, cfg["mode"], cfg["support_floor"])
    np.savez_compressed(
        out / "guard_before_truth.npz",
        ids=ids,
        reject=mask,
        density=qs[cfg["k"]]["density"],
        support=qs[cfg["k"]]["support"],
        fit_reject=fm,
        cal_reject=cm,
    )
    for name in ["CVAE-EVT", "EVM", "W-SVM"]:
        source = BASE / "results/comparison_20261102_calibrated/CIC2" / name
        old = np.load(source / "predictions_all_epochs.npz")
        pred = old["query"].copy()
        pred[:, mask] = 2
        train = old["train"].copy()
        train[:, fm] = 2
        cal = old["cal"].copy()
        cal[:, cm] = 2
        dest = out / (name + "-DCA")
        dest.mkdir()
        np.savez_compressed(
            dest / "predictions_all_epochs.npz",
            query=pred,
            train=train,
            cal=cal,
            fit_ids=old["fit_ids"],
            cal_ids=old["cal_ids"],
            query_ids=old["query_ids"],
        )
        sm = json.loads((source / "run_manifest.json").read_text())
        elapsed = time.perf_counter() - start
        save_json(
            dest / "run_manifest.json",
            dict(
                model=name + "-DCA",
                base_model=name,
                setting="CIC2",
                epochs=100,
                row_semantics="source-anchored density enhancement applied to 100 actual base states",
                training_complete=True,
                base_run=str(source.relative_to(BASE)),
                base_predictions_sha256=digest(source / "predictions_all_epochs.npz"),
                config=cfg,
                thresholds=threshold,
                query_truth_read_during_training=False,
                query_labels_used_for_inference=False,
                transductive=True,
                original_input_cohort_unchanged=True,
                known_source_reference_added=len(s["x"]),
                known_source_cal_added=len(s["cal"]),
                guard_seconds=elapsed,
                base_seconds=sm["total_seconds"],
                total_seconds=sm["total_seconds"] + elapsed,
                selection_seconds=sel["selection_seconds"],
                selected_epoch=100,
                published=False,
                final_test_executed=False,
            ),
        )
    print(
        "COMPLETE source-anchored unlabeled rejected",
        int(mask.sum()),
        "/",
        len(mask),
        "seconds",
        round(time.perf_counter() - start, 1),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["select", "apply"])
    a = p.parse_args()
    with threadpool_limits(limits=4):
        (select if a.phase == "select" else apply)()
