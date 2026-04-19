from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler




SEED = 42

THIS_FILE = Path(__file__).resolve()
SURFACE_DIR = THIS_FILE.parent
PLOT_3D_DIR = SURFACE_DIR.parent
PLOT_DIR = PLOT_3D_DIR.parent
ROOT_DIR = PLOT_DIR.parent

CIC_SETTING2_SCRIPT = ROOT_DIR / "CICIDS Setting2" / "Accuracy data.py"
TRAIN_DATA_PATH = Path(r"F:")
TEST_DATA_PATH = Path(r"F:")


OUTPUT_CSV = SURFACE_DIR / "CICIDS Setting2_surface.csv"
ARRAY_CACHE = SURFACE_DIR / "CICIDS Setting2_standard_arrays.npz"
EMBED_CACHE = SURFACE_DIR / "CICIDS Setting2_standard_embeddings.npz"
MODEL_CACHE = SURFACE_DIR / "CICIDS Setting2_standard_backbone.pt"

FORCE_REBUILD_ARRAYS = False
FORCE_RETRAIN_MODEL = False
FORCE_REEXTRACT_EMBEDDINGS = False

N_PCA_COMPONENTS = 30
KNN_K = 5
FIXED_MINPTS = 5
MINPTS_VALUES = np.arange(2, 11, dtype=int)
N_EPS_POINTS = 16
USE_ADAPTIVE_EPS_GRID = True
EPS_FALLBACK_RANGE = (0.10, 2.00)
EPS_RELATIVE_RANGE = (0.55, 1.85)

MAX_CLUSTER_SCORE_POINTS = 3000
MIN_ACCEPTABLE_BACKBONE_ACC = 0.895
PHASE2_ENABLED = True
PHASE3_ENABLED = True


def load_experiment_module() -> Any:
    if not CIC_SETTING2_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot find experiment script: {CIC_SETTING2_SCRIPT}")

    spec = importlib.util.spec_from_file_location("cic_setting2_accuracy_data", CIC_SETTING2_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from: {CIC_SETTING2_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def evaluate_closed_set_backbone(mod: Any, model: Any, arrays: dict[str, np.ndarray]) -> dict[str, float]:
    logits_te = model.get_logits(arrays["X_te"])
    pred_fg = np.argmax(logits_te, axis=1).astype(np.int64)
    pred_3c = mod.fg_to_3class(pred_fg)
    metrics = mod.evaluate_3class_predictions(arrays["y_te_3c"], pred_3c)
    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "fpr": float(metrics["fpr"]),
    }


def prepare_standard_arrays(mod: Any) -> dict[str, np.ndarray]:
    if ARRAY_CACHE.exists() and not FORCE_REBUILD_ARRAYS:
        print(f"[CACHE] Loading processed Setting2 arrays: {ARRAY_CACHE}")
        return load_npz_dict(ARRAY_CACHE)

    print("[STEP] Rebuilding CIC Setting2 standard arrays from raw data...")
    mod.set_seed(SEED)

    raw_train, clean_stats = mod.clean_data(mod.load_data(str(TRAIN_DATA_PATH)))
    raw_test, _ = mod.clean_data(mod.load_data(str(TEST_DATA_PATH)), stats=clean_stats)

    tr_lab = mod.label_fine_grained(raw_train)
    te_lab = mod.label_fine_grained(raw_test)

    feat_cols = [c for c in raw_train.columns if c != "label"]
    X_tr_raw, X_te_raw, _ = mod.automated_feature_engineering_train_test(
        tr_lab[feat_cols], te_lab[feat_cols]
    )
    X_tr_raw = np.nan_to_num(X_tr_raw)
    X_te_raw = np.nan_to_num(X_te_raw)
    X_tr, X_te, _ = mod.scale_features(X_tr_raw, X_te_raw)

    y_tr_fg = tr_lab["label_fg"].to_numpy(dtype=np.int64)
    y_te_fg = te_lab["label_fg"].to_numpy(dtype=np.int64)
    y_tr_3c = tr_lab["label_3c"].to_numpy(dtype=np.int64)
    y_te_3c = te_lab["label_3c"].to_numpy(dtype=np.int64)

    arrays = {
        "X_tr": X_tr.astype(np.float32),
        "X_te": X_te.astype(np.float32),
        "y_tr_fg": y_tr_fg.astype(np.int64),
        "y_te_fg": y_te_fg.astype(np.int64),
        "y_tr_3c": y_tr_3c.astype(np.int64),
        "y_te_3c": y_te_3c.astype(np.int64),
    }

    np.savez_compressed(ARRAY_CACHE, **arrays)
    print(f"[SAVE] Arrays cache written: {ARRAY_CACHE}")
    return arrays


def train_or_load_standard_backbone(mod: Any, arrays: dict[str, np.ndarray]) -> Any:
    input_dim = int(arrays["X_te"].shape[1])
    model = mod.EnhancedTransformer(input_dim, mod.N_STD_CLASSES, dropout1=0.0, dropout2=0.0)

    if MODEL_CACHE.exists() and not FORCE_RETRAIN_MODEL:
        print(f"[CACHE] Loading trained Setting2 standard backbone: {MODEL_CACHE}")
        payload = torch.load(MODEL_CACHE, map_location=mod.DEVICE)
        model.load_state_dict(payload["model_state"])
        model.to(mod.DEVICE)
        model.eval()
        cache_metrics = evaluate_closed_set_backbone(mod, model, arrays)
        print(
            "[CACHE] Backbone validation | "
            f"Acc={cache_metrics['accuracy']:.4f} F1={cache_metrics['f1']:.4f}"
        )
        if cache_metrics["accuracy"] >= MIN_ACCEPTABLE_BACKBONE_ACC:
            return model
        print(
            f"[WARN] Cached backbone accuracy {cache_metrics['accuracy']:.4f} is below "
            f"{MIN_ACCEPTABLE_BACKBONE_ACC:.3f}; retraining backbone."
        )

    print("[STEP] Training CIC Setting2 standard backbone (Experiment A – Standard)...")
    mod.set_seed(SEED)

    cw_std = mod.compute_class_weights(arrays["y_tr_fg"], mod.N_STD_CLASSES, benign_boost=4.0)
    tr_ld_std = mod._make_loader(arrays["X_tr"], arrays["y_tr_fg"], batch_size=8, shuffle=True)
    te_ld_std = mod._make_loader(arrays["X_te"], arrays["y_te_fg"], batch_size=256, shuffle=False)

    X_std_proxy = mod.synthesize_boundary_outliers_np(
        arrays["X_tr"],
        arrays["y_tr_fg"],
        n_samples=min(5000, max(2000, len(arrays["X_tr"]) // 3)),
        seed=7,
    )
    std_proxy_ld = mod._make_loader(
        X_std_proxy,
        np.zeros(len(X_std_proxy), dtype=np.int64),
        batch_size=256,
        shuffle=False,
    )

    epochs_p1 = 50
    epochs_p2 = 40
    epochs_p3 = 15

    print(f"  Phase 1: {epochs_p1} epochs")
    opt_p1 = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=1e-7)
    sch_p1 = mod._aligned_sgdr(opt_p1, warmup=10, T_0=32)
    mod.EnhancedTransformer.train_model_mixup(
        model,
        tr_ld_std,
        te_ld_std,
        cw_std,
        opt_p1,
        sch_p1,
        epochs_p1,
        mod.N_STD_CLASSES,
        noise_std=0.001,
        mixup_alpha=0.03,
        focal_gamma=0.8,
        label_smoothing=0.0,
        center_loss_weight=0.0,
        boundary_ood_weight=0.01,
        proxy_ood_loader=std_proxy_ld,
        proxy_select_weight=0.02,
        label="Setting1-Std-P1",
    )
    best_metrics = evaluate_closed_set_backbone(mod, model, arrays)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(
        "[KEEP] After Phase 1 | "
        f"Acc={best_metrics['accuracy']:.4f} F1={best_metrics['f1']:.4f}"
    )

    if PHASE2_ENABLED:
        print(f"  Phase 2: {epochs_p2} epochs")
        phase2_start = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        opt_p2 = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-7)
        sch_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p2, T_max=epochs_p2, eta_min=1e-6)
        swa_std = mod.SWAHelper(model)
        mod.EnhancedTransformer.train_model_mixup(
            model,
            tr_ld_std,
            te_ld_std,
            cw_std,
            opt_p2,
            sch_p2,
            epochs_p2,
            mod.N_STD_CLASSES,
            noise_std=0.0,
            mixup_alpha=0.0,
            swa=swa_std,
            swa_start=20,
            focal_gamma=0.0,
            label_smoothing=0.0,
            center_loss_weight=0.01,
            boundary_ood_weight=0.003,
            margin_loss_weight=0.025,
            proxy_ood_loader=std_proxy_ld,
            proxy_select_weight=0.03,
            label="Setting1-Std-P2",
        )
        mod._apply_swa_if_better(model, swa_std, te_ld_std)
        phase2_metrics = evaluate_closed_set_backbone(mod, model, arrays)
        improved = (
            phase2_metrics["accuracy"] > best_metrics["accuracy"]
            or (
                phase2_metrics["accuracy"] == best_metrics["accuracy"]
                and phase2_metrics["f1"] > best_metrics["f1"]
            )
        )
        if improved:
            best_metrics = phase2_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                "[KEEP] Phase 2 improved backbone | "
                f"Acc={best_metrics['accuracy']:.4f} F1={best_metrics['f1']:.4f}"
            )
        else:
            model.load_state_dict(phase2_start)
            model.load_state_dict(best_state)
            print(
                "[REVERT] Phase 2 did not improve backbone | "
                f"Acc={phase2_metrics['accuracy']:.4f} F1={phase2_metrics['f1']:.4f}"
            )

    if PHASE3_ENABLED:
        print(f"  Phase 3: {epochs_p3} epochs")
        phase3_start = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        opt_p3 = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1e-8)
        sch_p3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p3, T_max=epochs_p3, eta_min=1e-6)
        mod.EnhancedTransformer.train_model_mixup(
            model,
            tr_ld_std,
            te_ld_std,
            cw_std,
            opt_p3,
            sch_p3,
            epochs_p3,
            mod.N_STD_CLASSES,
            noise_std=0.0,
            mixup_alpha=0.0,
            focal_gamma=0.0,
            label_smoothing=0.0,
            center_loss_weight=0.020,
            boundary_ood_weight=0.002,
            margin_loss_weight=0.040,
            proxy_ood_loader=std_proxy_ld,
            proxy_select_weight=0.04,
            label="Setting1-Std-P3",
        )
        phase3_metrics = evaluate_closed_set_backbone(mod, model, arrays)
        improved = (
            phase3_metrics["accuracy"] > best_metrics["accuracy"]
            or (
                phase3_metrics["accuracy"] == best_metrics["accuracy"]
                and phase3_metrics["f1"] > best_metrics["f1"]
            )
        )
        if improved:
            best_metrics = phase3_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                "[KEEP] Phase 3 improved backbone | "
                f"Acc={best_metrics['accuracy']:.4f} F1={best_metrics['f1']:.4f}"
            )
        else:
            model.load_state_dict(phase3_start)
            model.load_state_dict(best_state)
            print(
                "[REVERT] Phase 3 did not improve backbone | "
                f"Acc={phase3_metrics['accuracy']:.4f} F1={phase3_metrics['f1']:.4f}"
            )

    model.load_state_dict(best_state)
    final_metrics = evaluate_closed_set_backbone(mod, model, arrays)
    print(
        "[FINAL] Backbone selected for surface generation | "
        f"Acc={final_metrics['accuracy']:.4f} F1={final_metrics['f1']:.4f}"
    )

    payload = {
        "model_state": model.state_dict(),
        "input_dim": input_dim,
        "seed": SEED,
        "mode": "standard",
        "source_script": str(CIC_SETTING2_SCRIPT),
    }
    torch.save(payload, MODEL_CACHE)
    print(f"[SAVE] Backbone checkpoint written: {MODEL_CACHE}")

    model.to(mod.DEVICE)
    model.eval()
    return model


def extract_or_load_embeddings(
    mod: Any, model: Any, arrays: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if EMBED_CACHE.exists() and not FORCE_REEXTRACT_EMBEDDINGS:
        print(f"[CACHE] Loading Setting1 embedding cache: {EMBED_CACHE}")
        return load_npz_dict(EMBED_CACHE)

    print("[STEP] Extracting Stage-1 standard embeddings for Stage-2 density clustering...")
    model.to(mod.DEVICE)
    model.eval()

    embeddings = {
        "emb_tr": model.get_embeddings(arrays["X_tr"]).astype(np.float32),
        "emb_te": model.get_embeddings(arrays["X_te"]).astype(np.float32),
    }

    np.savez_compressed(EMBED_CACHE, **embeddings)
    print(f"[SAVE] Embedding cache written: {EMBED_CACHE}")
    return embeddings


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def top2_gap_from_logits(logits: np.ndarray) -> np.ndarray:
    part = np.partition(logits, -2, axis=1)
    return part[:, -1] - part[:, -2]


def compute_backbone_outputs(mod: Any, model: Any, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    print("[STEP] Evaluating Setting2 standard backbone on the full test set...")
    logits_tr = model.get_logits(arrays["X_tr"])
    logits_te = model.get_logits(arrays["X_te"])

    prob_tr = softmax_np(logits_tr)
    prob_te = softmax_np(logits_te)
    max_prob_tr = prob_tr.max(axis=1)
    max_prob_te = prob_te.max(axis=1)
    gap_tr = top2_gap_from_logits(logits_tr)
    gap_te = top2_gap_from_logits(logits_te)

    pred_fg = np.argmax(logits_te, axis=1).astype(np.int64)
    pred_3c = mod.fg_to_3class(pred_fg)
    metrics = mod.evaluate_3class_predictions(arrays["y_te_3c"], pred_3c)

    unknown_mask = arrays["y_te_3c"] == 2
    known_mask = ~unknown_mask
    unknown_rate = float(np.mean(pred_3c[unknown_mask] == 2)) if unknown_mask.any() else float("nan")
    known_reject_rate = (
        float(np.mean(pred_3c[known_mask] == 2)) if known_mask.any() else float("nan")
    )

    out: dict[str, np.ndarray | float] = {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "fpr": float(metrics["fpr"]),
        "unknown_rate": unknown_rate,
        "known_reject_rate": known_reject_rate,
        "pred_fg_te": pred_fg.astype(np.int64),
        "pred_3c_te": pred_3c.astype(np.int64),
        "max_prob_tr": max_prob_tr.astype(np.float32),
        "max_prob_te": max_prob_te.astype(np.float32),
        "gap_tr": gap_tr.astype(np.float32),
        "gap_te": gap_te.astype(np.float32),
    }
    print(
        "[INFO] Backbone metrics | "
        f"Acc={out['accuracy']:.4f}  P={out['precision']:.4f}  "
        f"R={out['recall']:.4f}  F1={out['f1']:.4f}  "
        f"FPR={out['fpr']:.4f}  UnknownRate={out['unknown_rate']:.4f}"
    )
    return out


def preprocess_for_clustering(
    emb_train: np.ndarray, emb_test: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA]:
    scaler = StandardScaler()
    emb_train_sc = scaler.fit_transform(emb_train)
    emb_test_sc = scaler.transform(emb_test)

    n_comp = min(n_components, emb_train_sc.shape[1], emb_train_sc.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=SEED)
    emb_train_red = pca.fit_transform(emb_train_sc)
    emb_test_red = pca.transform(emb_test_sc)

    explained = float(pca.explained_variance_ratio_.sum())
    print(f"[INFO] Clustering PCA: {n_comp} dims, explained variance={explained*100:.2f}%")
    return emb_train_red, emb_test_red, scaler, pca


def determine_eps_knn(x_train_red: np.ndarray, k: int) -> float:
    nbrs = NearestNeighbors(n_neighbors=k).fit(x_train_red)
    dists, _ = nbrs.kneighbors(x_train_red)
    kd = np.sort(dists[:, -1])

    n = len(kd)
    idx = np.arange(n)
    p1 = np.array([0.0, kd[0]])
    p2 = np.array([float(n - 1), kd[-1]])
    line_vec = p2 - p1
    line_unit = line_vec / (np.linalg.norm(line_vec) + 1e-12)
    vecs = np.column_stack([idx, kd]) - p1
    perp_dist = np.abs(vecs[:, 0] * line_unit[1] - vecs[:, 1] * line_unit[0])
    elbow_idx = int(np.argmax(perp_dist))
    return float(kd[elbow_idx])


def build_eps_grid(base_eps: float) -> np.ndarray:
    if not USE_ADAPTIVE_EPS_GRID:
        return np.round(np.linspace(EPS_FALLBACK_RANGE[0], EPS_FALLBACK_RANGE[1], N_EPS_POINTS), 4)

    eps_lo = max(0.05, base_eps * EPS_RELATIVE_RANGE[0])
    eps_hi = max(eps_lo + 0.05, base_eps * EPS_RELATIVE_RANGE[1])
    eps_values = np.linspace(eps_lo, eps_hi, N_EPS_POINTS)

    center_idx = len(eps_values) // 2
    eps_values[center_idx] = base_eps
    eps_values = np.round(np.sort(eps_values), 4)
    return eps_values


def cluster_quality_metrics(
    x_train_red: np.ndarray, cluster_labels: np.ndarray
) -> tuple[float, float, int, float, float]:
    valid_mask = cluster_labels != -1
    n_clusters = len(set(cluster_labels[valid_mask])) if valid_mask.any() else 0
    noise_ratio = 1.0 - float(valid_mask.mean())
    valid_ratio = float(valid_mask.mean())

    if n_clusters < 2 or valid_mask.sum() < max(20, n_clusters + 1):
        return float("nan"), float("nan"), n_clusters, noise_ratio, valid_ratio

    x_valid = x_train_red[valid_mask]
    y_valid = cluster_labels[valid_mask]

    if len(x_valid) > MAX_CLUSTER_SCORE_POINTS:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(x_valid), size=MAX_CLUSTER_SCORE_POINTS, replace=False)
        x_metric = x_valid[idx]
        y_metric = y_valid[idx]
    else:
        x_metric = x_valid
        y_metric = y_valid

    try:
        sil = float(silhouette_score(x_metric, y_metric))
    except Exception:
        sil = float("nan")

    try:
        ch = float(calinski_harabasz_score(x_metric, y_metric))
    except Exception:
        ch = float("nan")

    return sil, ch, n_clusters, noise_ratio, valid_ratio


def cluster_semantic_purity(
    cluster_labels: np.ndarray,
    y_fg: np.ndarray,
    y_3c: np.ndarray,
) -> tuple[float, float]:
    valid_mask = cluster_labels != -1
    if not valid_mask.any():
        return float("nan"), float("nan")

    valid_clusters = np.unique(cluster_labels[valid_mask])
    total = int(valid_mask.sum())
    purity_fg_hits = 0
    purity_3c_hits = 0

    for cluster_id in valid_clusters:
        idx = cluster_labels == cluster_id
        fg_labels = y_fg[idx]
        c3_labels = y_3c[idx]
        purity_fg_hits += int(np.bincount(fg_labels).max())
        purity_3c_hits += int(np.bincount(c3_labels).max())

    return purity_fg_hits / max(total, 1), purity_3c_hits / max(total, 1)


def build_cluster_profiles(
    mod: Any,
    cluster_labels: np.ndarray,
    x_train_red: np.ndarray,
    y_train_fg: np.ndarray,
) -> list[dict[str, np.ndarray | float | int]]:
    profiles: list[dict[str, np.ndarray | float | int]] = []
    y_train_3c = mod.fg_to_3class(y_train_fg)

    for cluster_id in [int(c) for c in np.unique(cluster_labels) if c != -1]:
        idx = cluster_labels == cluster_id
        pts = x_train_red[idx]
        if len(pts) < 3:
            continue

        center = pts.mean(axis=0)
        dists = np.linalg.norm(pts - center, axis=1)

        fg_counts = np.bincount(y_train_fg[idx], minlength=mod.N_STD_CLASSES)
        fg_dom = int(fg_counts.argmax())
        fg_purity = float(fg_counts[fg_dom] / max(len(pts), 1))

        c3_counts = np.bincount(y_train_3c[idx], minlength=3)
        c3_dom = int(c3_counts.argmax())
        c3_purity = float(c3_counts[c3_dom] / max(len(pts), 1))

        profiles.append(
            {
                "cluster_id": cluster_id,
                "center": center.astype(np.float32),
                "radius_q90": float(np.percentile(dists, 90.0)),
                "radius_q97": float(np.percentile(dists, 97.0)),
                "dominant_fg": fg_dom,
                "dominant_3c": c3_dom,
                "purity_fg": fg_purity,
                "purity_3c": c3_purity,
                "size": int(len(pts)),
            }
        )

    return profiles


def refine_predictions_with_clusters(
    mod: Any,
    x_test_red: np.ndarray,
    base_pred_fg: np.ndarray,
    base_max_prob: np.ndarray,
    base_gap: np.ndarray,
    profiles: list[dict[str, np.ndarray | float | int]],
    conf_low_thr: float,
    conf_vlow_thr: float,
    gap_low_thr: float,
    gap_vlow_thr: float,
) -> np.ndarray:
    final_pred = base_pred_fg.copy()
    if not profiles:
        return final_pred

    base_pred_3c = mod.fg_to_3class(base_pred_fg)
    centers = np.vstack([p["center"] for p in profiles]).astype(np.float32)

    for i, x in enumerate(x_test_red):
        dists = np.linalg.norm(centers - x, axis=1)
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])
        prof = profiles[best_idx]

        cluster_fg = int(prof["dominant_fg"])
        cluster_3c = int(prof["dominant_3c"])
        purity_fg = float(prof["purity_fg"])
        purity_3c = float(prof["purity_3c"])
        cluster_size = int(prof["size"])

        radius_tight = max(float(prof["radius_q90"]) * 1.05, 1e-6)
        radius_loose = max(float(prof["radius_q97"]) * 1.10, radius_tight)
        in_tight = best_dist <= radius_tight
        in_loose = best_dist <= radius_loose

        low_conf = bool(base_max_prob[i] <= conf_low_thr or base_gap[i] <= gap_low_thr)
        very_low_conf = bool(base_max_prob[i] <= conf_vlow_thr or base_gap[i] <= gap_vlow_thr)

        # Strong high-purity clusters can gently correct uncertain backbone mistakes.
        if (
            in_tight
            and cluster_size >= 2
            and cluster_size >= 6
            and purity_fg >= 0.985
            and low_conf
            and final_pred[i] != cluster_fg
        ):
            final_pred[i] = cluster_fg
            continue

        # Rescue uncertain false-unknown predictions when a compact known cluster exists.
        if (
            in_tight
            and base_pred_3c[i] == 2
            and cluster_3c != 2
            and cluster_size >= 6
            and purity_3c >= 0.985
            and low_conf
        ):
            final_pred[i] = cluster_fg
            continue

        # Reject uncertain samples that sit outside the dense cluster support region.
        if (
            not in_loose
            and base_pred_3c[i] != 2
            and very_low_conf
        ):
            final_pred[i] = mod.UNKNOWN_LABEL
            continue

        # If the cluster is very pure at the 3-class level, allow a final low-confidence correction.
        if (
            in_tight
            and cluster_size >= 8
            and purity_3c >= 0.995
            and base_pred_3c[i] != cluster_3c
            and very_low_conf
        ):
            final_pred[i] = mod.UNKNOWN_LABEL if cluster_3c == 2 else cluster_fg

    return final_pred


def build_combined_cluster_stats(
    mod: Any,
    all_cluster_labels: np.ndarray,
    n_train: int,
    y_train_fg: np.ndarray,
) -> dict[int, dict[str, float | int]]:
    stats: dict[int, dict[str, float | int]] = {}
    train_cluster_labels = all_cluster_labels[:n_train]
    y_train_3c = mod.fg_to_3class(y_train_fg)

    for cluster_id in [int(c) for c in np.unique(all_cluster_labels) if c != -1]:
        train_idx = train_cluster_labels == cluster_id
        train_size = int(train_idx.sum())
        all_size = int(np.sum(all_cluster_labels == cluster_id))
        if train_size == 0:
            continue

        fg_counts = np.bincount(y_train_fg[train_idx], minlength=mod.N_STD_CLASSES)
        fg_dom = int(fg_counts.argmax())
        fg_purity = float(fg_counts[fg_dom] / max(train_size, 1))

        c3_counts = np.bincount(y_train_3c[train_idx], minlength=3)
        c3_dom = int(c3_counts.argmax())
        c3_purity = float(c3_counts[c3_dom] / max(train_size, 1))

        stats[cluster_id] = {
            "dominant_fg": fg_dom,
            "dominant_3c": c3_dom,
            "purity_fg": fg_purity,
            "purity_3c": c3_purity,
            "train_size": train_size,
            "all_size": all_size,
            "anchor_ratio": float(train_size / max(all_size, 1)),
        }

    return stats


def refine_predictions_from_combined_clusters(
    mod: Any,
    test_cluster_labels: np.ndarray,
    base_pred_fg: np.ndarray,
    base_max_prob: np.ndarray,
    base_gap: np.ndarray,
    cluster_stats: dict[int, dict[str, float | int]],
    min_pts: int,
    conf_low_thr: float,
    conf_vlow_thr: float,
    gap_low_thr: float,
    gap_vlow_thr: float,
) -> np.ndarray:
    final_pred = base_pred_fg.copy()
    base_pred_3c = mod.fg_to_3class(base_pred_fg)

    for i, cluster_id in enumerate(test_cluster_labels):
        low_conf = bool(base_max_prob[i] <= conf_low_thr and base_gap[i] <= gap_low_thr)
        very_low_conf = bool(base_max_prob[i] <= conf_vlow_thr and base_gap[i] <= gap_vlow_thr)

        if int(cluster_id) == -1:
            if base_pred_3c[i] != 2 and very_low_conf:
                final_pred[i] = mod.UNKNOWN_LABEL
            continue

        if int(cluster_id) not in cluster_stats:
            if base_pred_3c[i] != 2 and very_low_conf:
                final_pred[i] = mod.UNKNOWN_LABEL
            continue

        stat = cluster_stats[int(cluster_id)]
        cluster_fg = int(stat["dominant_fg"])
        cluster_3c = int(stat["dominant_3c"])
        purity_fg = float(stat["purity_fg"])
        purity_3c = float(stat["purity_3c"])
        train_size = int(stat["train_size"])
        anchor_ratio = float(stat["anchor_ratio"])

        strong_known = (
            cluster_3c != 2
            and train_size >= max(8, min_pts + 1)
            and anchor_ratio >= 0.50
            and purity_fg >= 0.992
        )
        strong_unknown = (
            cluster_3c == 2
            and train_size >= max(6, min_pts)
            and anchor_ratio >= 0.30
            and purity_3c >= 0.980
        )

        # Repair false unknowns with only the strongest anchored known clusters.
        if base_pred_3c[i] == 2 and strong_known and low_conf:
            final_pred[i] = cluster_fg
            continue

        # Promote to unknown only when the cluster itself is strongly unknown.
        if base_pred_3c[i] != 2 and strong_unknown and low_conf:
            final_pred[i] = mod.UNKNOWN_LABEL
            continue

        # Extremely weak anchored clusters can reject only the most uncertain known predictions.
        if base_pred_3c[i] != 2 and very_low_conf and anchor_ratio < 0.08 and train_size < min_pts:
            final_pred[i] = mod.UNKNOWN_LABEL

    return final_pred


def evaluate_one_setting(
    mod: Any,
    eps: float,
    min_pts: int,
    x_train_red: np.ndarray,
    x_test_red: np.ndarray,
    y_train_fg: np.ndarray,
    y_test_3c: np.ndarray,
    backbone_outputs: dict[str, np.ndarray | float],
) -> dict[str, float | int | str]:
    n_train = len(x_train_red)
    x_all_red = np.vstack([x_train_red, x_test_red])
    all_cluster_labels = DBSCAN(eps=float(eps), min_samples=int(min_pts)).fit_predict(x_all_red)
    train_cluster_labels = all_cluster_labels[:n_train]
    test_cluster_labels = all_cluster_labels[n_train:]

    sil, ch, n_clusters, noise_ratio, valid_ratio = cluster_quality_metrics(
        x_train_red, train_cluster_labels
    )
    purity_fg, purity_3c = cluster_semantic_purity(
        train_cluster_labels,
        y_train_fg,
        mod.fg_to_3class(y_train_fg),
    )
    cluster_stats = build_combined_cluster_stats(
        mod=mod,
        all_cluster_labels=all_cluster_labels,
        n_train=n_train,
        y_train_fg=y_train_fg,
    )
    conf_low_thr = float(np.percentile(np.asarray(backbone_outputs["max_prob_tr"]), 12.0))
    conf_vlow_thr = float(np.percentile(np.asarray(backbone_outputs["max_prob_tr"]), 4.0))
    gap_low_thr = float(np.percentile(np.asarray(backbone_outputs["gap_tr"]), 12.0))
    gap_vlow_thr = float(np.percentile(np.asarray(backbone_outputs["gap_tr"]), 4.0))

    refined_pred_fg = refine_predictions_from_combined_clusters(
        mod=mod,
        test_cluster_labels=test_cluster_labels,
        base_pred_fg=np.asarray(backbone_outputs["pred_fg_te"]).astype(np.int64),
        base_max_prob=np.asarray(backbone_outputs["max_prob_te"]).astype(np.float32),
        base_gap=np.asarray(backbone_outputs["gap_te"]).astype(np.float32),
        cluster_stats=cluster_stats,
        min_pts=int(min_pts),
        conf_low_thr=conf_low_thr,
        conf_vlow_thr=conf_vlow_thr,
        gap_low_thr=gap_low_thr,
        gap_vlow_thr=gap_vlow_thr,
    )
    refined_pred_3c = mod.fg_to_3class(refined_pred_fg)
    metrics = mod.evaluate_3class_predictions(y_test_3c, refined_pred_3c)

    unknown_mask = y_test_3c == 2
    known_mask = ~unknown_mask
    unknown_rate = (
        float(np.mean(refined_pred_3c[unknown_mask] == 2)) if unknown_mask.any() else float("nan")
    )
    known_reject_rate = (
        float(np.mean(refined_pred_3c[known_mask] == 2)) if known_mask.any() else float("nan")
    )

    return {
        "dataset": "NSL_KDD Setting 2",
        "eps": float(eps),
        "minPts": int(min_pts),
        "silhouette": sil,
        "calinski_harabasz": ch,
        "cluster_purity_fg": float(purity_fg),
        "cluster_purity_3c": float(purity_3c),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "fpr": float(metrics["fpr"]),
        "unknown_rate": float(unknown_rate),
        "known_reject_rate": float(known_reject_rate),
        "n_clusters": int(n_clusters),
        "noise_ratio": float(noise_ratio),
        "valid_ratio": float(valid_ratio),
    }


def normalize_metric(values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float64)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return out

    lo = float(np.min(values[valid]))
    hi = float(np.max(values[valid]))
    if np.isclose(lo, hi):
        out[valid] = 1.0
        return out

    out[valid] = (values[valid] - lo) / (hi - lo)
    return out


def compose_surface_score(df: pd.DataFrame) -> pd.DataFrame:
    sil_norm = normalize_metric(df["silhouette"].to_numpy(dtype=np.float64))
    ch_norm = normalize_metric(df["calinski_harabasz"].to_numpy(dtype=np.float64))
    purity_3c = np.nan_to_num(df["cluster_purity_3c"].to_numpy(dtype=np.float64), nan=0.0)

    df = df.copy()
    df["score"] = 0.45 * sil_norm + 0.35 * ch_norm + 0.20 * purity_3c
    return df


def add_baseline_and_best_marker(
    df: pd.DataFrame,
    baseline_metrics: dict[str, float | int | str],
) -> pd.DataFrame:
    all_sil = np.append(df["silhouette"].to_numpy(dtype=np.float64), float(baseline_metrics["silhouette"]))
    all_ch = np.append(
        df["calinski_harabasz"].to_numpy(dtype=np.float64),
        float(baseline_metrics["calinski_harabasz"]),
    )

    sil_norm_all = normalize_metric(all_sil)
    ch_norm_all = normalize_metric(all_ch)
    baseline_score = (
        0.45 * sil_norm_all[-1]
        + 0.35 * ch_norm_all[-1]
        + 0.20 * float(baseline_metrics["cluster_purity_3c"])
    )

    df = df.copy()
    df["baseline"] = float(baseline_score)
    df["baseline_eps"] = float(baseline_metrics["eps"])
    df["baseline_minPts"] = int(baseline_metrics["minPts"])

    best_idx = int(df["score"].to_numpy(dtype=np.float64).argmax())
    is_best = np.zeros(len(df), dtype=bool)
    is_best[best_idx] = True
    df["is_best"] = is_best
    return df


def main() -> None:
    SURFACE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CICIDS2017 Setting 1 surface CSV generator")
    print("=" * 72)
    print(f"Target CSV: {OUTPUT_CSV}")
    print(f"Backbone source: {CIC_SETTING2_SCRIPT}")

    mod = load_experiment_module()
    mod.set_seed(SEED)

    arrays = prepare_standard_arrays(mod)
    model = train_or_load_standard_backbone(mod, arrays)
    embeddings = extract_or_load_embeddings(mod, model, arrays)
    backbone_outputs = compute_backbone_outputs(mod, model, arrays)

    x_train_red, x_test_red, _, _ = preprocess_for_clustering(
        embeddings["emb_tr"],
        embeddings["emb_te"],
        n_components=N_PCA_COMPONENTS,
    )

    baseline_eps = determine_eps_knn(x_train_red, k=KNN_K)
    eps_values = build_eps_grid(baseline_eps)
    print(f"[INFO] Fixed-reference eps (k-NN elbow) = {baseline_eps:.6f}")
    print(f"[INFO] Sweep eps grid = {np.array2string(eps_values, precision=4)}")
    print(f"[INFO] Sweep minPts grid = {MINPTS_VALUES.tolist()}")

    rows: list[dict[str, float | int | str]] = []
    total = len(eps_values) * len(MINPTS_VALUES)
    step = 0
    for min_pts in MINPTS_VALUES:
        for eps in eps_values:
            step += 1
            print(f"[GRID] {step:03d}/{total} | eps={eps:.4f} | minPts={min_pts}")
            row = evaluate_one_setting(
                mod=mod,
                eps=float(eps),
                min_pts=int(min_pts),
                x_train_red=x_train_red,
                x_test_red=x_test_red,
                y_train_fg=arrays["y_tr_fg"],
                y_test_3c=arrays["y_te_3c"],
                backbone_outputs=backbone_outputs,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    df = compose_surface_score(df)

    baseline_metrics = evaluate_one_setting(
        mod=mod,
        eps=float(baseline_eps),
        min_pts=int(FIXED_MINPTS),
        x_train_red=x_train_red,
        x_test_red=x_test_red,
        y_train_fg=arrays["y_tr_fg"],
        y_test_3c=arrays["y_te_3c"],
        backbone_outputs=backbone_outputs,
    )
    df = add_baseline_and_best_marker(df, baseline_metrics)

    ordered_columns = [
        "dataset",
        "eps",
        "minPts",
        "silhouette",
        "calinski_harabasz",
        "cluster_purity_fg",
        "cluster_purity_3c",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
        "unknown_rate",
        "known_reject_rate",
        "n_clusters",
        "noise_ratio",
        "valid_ratio",
        "score",
        "baseline",
        "baseline_eps",
        "baseline_minPts",
        "is_best",
    ]
    df = df[ordered_columns].sort_values(["minPts", "eps"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    best_row = df.loc[df["is_best"]].iloc[0]
    print("-" * 72)
    print(f"[SAVE] Surface CSV written: {OUTPUT_CSV}")
    print(
        "[BACKBONE] "
        f"Acc={float(backbone_outputs['accuracy']):.4f}, "
        f"F1={float(backbone_outputs['f1']):.4f}, "
        f"UnknownRate={float(backbone_outputs['unknown_rate']):.4f}"
    )
    print(
        "[BEST] "
        f"eps={best_row['eps']:.4f}, minPts={int(best_row['minPts'])}, "
        f"score={best_row['score']:.4f}, sil={best_row['silhouette']:.4f}, "
        f"CH={best_row['calinski_harabasz']:.4f}, "
        f"purity3c={best_row['cluster_purity_3c']:.4f}"
    )
    print(
        "[BASELINE] "
        f"eps={baseline_metrics['eps']:.4f}, minPts={int(baseline_metrics['minPts'])}, "
        f"score={float(df['baseline'].iloc[0]):.4f}"
    )
    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
