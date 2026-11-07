"""Known-only model selection and immutable-cohort preprocessing."""

from pathlib import Path
import hashlib
import json
import sys
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from asset_paths import ARCHIVE, DATA, RECIPES, native_tool
SETTINGS = ("CIC1", "CIC2", "NSL1", "NSL2")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def load_known(setting):
    z = np.load(DATA / setting / "known.npz", allow_pickle=False)
    old = joblib.load(DATA / setting / "preprocessing.joblib")["scaler"]
    # Invert the scaler to recover native or encoded coordinates.

    return dict(
        x=old.inverse_transform(z["x_fit"].astype("float64")),
        cal=old.inverse_transform(z["x_cal"].astype("float64")),
        y=z["y_fit"],
        yc=z["y_cal"],
        ids=z["fit_ids"],
        cal_ids=z["cal_ids"],
    )


def load_query(setting):
    z = np.load(DATA / setting / "query.npz", allow_pickle=False)
    old = joblib.load(DATA / setting / "preprocessing.joblib")["scaler"]
    return old.inverse_transform(z["x"].astype("float64")), z["ids"]


class View:
    def __init__(self, setting, kind):
        self.setting, self.kind = setting, kind

    def _base(self, x):
        if self.setting.startswith("CIC") and self.kind != "original":
            return np.sign(x) * np.log1p(np.abs(x))
        return x

    def fit(self, x, y=None):
        z = self._base(x)
        self.binary = np.all((np.abs(z) < 1e-7) | (np.abs(z - 1) < 1e-7), axis=0)
        if self.kind == "original":
            from sklearn.preprocessing import MinMaxScaler

            self.scaler = MinMaxScaler().fit(z)
        elif self.kind in ("log_standard", "log_whiten", "log_class_whiten"):
            self.scaler = StandardScaler().fit(z)
        elif self.kind == "log_robust":
            self.scaler = RobustScaler(quantile_range=(10, 90)).fit(z)
            std = z.std(0)
            self.scaler.scale_ = np.where(
                self.scaler.scale_ > 1e-8, self.scaler.scale_, np.maximum(std, 1.0)
            )
        else:
            raise ValueError(self.kind)
        if self.kind in ("log_whiten", "log_class_whiten"):
            from sklearn.covariance import LedoitWolf

            values = self.scaler.transform(z)
            values[:, self.binary] = z[:, self.binary]
            self.white_center = values.mean(0)
            residual = values - self.white_center
            if self.kind == "log_class_whiten":
                if y is None:
                    raise ValueError("Within-class whitening requires FIT labels")
                residual = values.copy()
                for c in np.unique(y):
                    residual[y == c] -= values[y == c].mean(0)
            fitted = LedoitWolf(assume_centered=True).fit(residual)
            eig, vec = np.linalg.eigh(fitted.covariance_)
            floor = max(float(np.trace(fitted.covariance_) / len(eig)) * 1e-6, 1e-10)
            self.whitening = (vec / np.sqrt(np.maximum(eig, floor))) @ vec.T
            self.shrinkage = float(fitted.shrinkage_)
        return self

    def transform(self, x):
        z = self._base(x)
        v = self.scaler.transform(z)
        if self.kind != "original":
            v[:, self.binary] = z[:, self.binary]
        if hasattr(self, "whitening"):
            v = (v - self.white_center) @ self.whitening
        if not np.isfinite(v).all():
            raise ValueError("nonfinite preprocessing")
        return np.ascontiguousarray(v, dtype="float32")


def heldout_families(y, count=3):
    eligible = [i for i in range(1, 9) if np.sum(y == i) >= 20]
    # Select evenly spaced families by index.
    return [
        eligible[i]
        for i in np.unique(np.linspace(0, len(eligible) - 1, count).round().astype(int))
    ]


def balanced_known(y, p):
    return float(
        np.mean([np.mean(p[y == c] == (0 if c == 0 else 1)) for c in np.unique(y)])
    )


def probability_cuts(scores, pred, y, alpha, n_classes=9):
    # Scores are larger for inclusion. True known CAL labels establish
    # coverage only; current query labels never enter calibration.
    correct = pred == y
    pooled = scores[correct]
    if len(pooled) < 5:
        pooled = scores
    pooled_cut = float(np.quantile(pooled, alpha, method="lower"))
    cuts = []
    for c in range(n_classes):
        values = scores[correct & (y == c)]
        cuts.append(
            float(np.quantile(values, alpha, method="lower"))
            if len(values) >= 8
            else pooled_cut
        )
    return np.array(cuts), dict(
        alpha=alpha,
        pooled_correct_n=len(pooled),
        correct_class_n=[int(np.sum(correct & (y == c))) for c in range(n_classes)],
    )


def probability_decision(pred, score, cuts, benign_classes=1):
    out = np.where(pred < benign_classes, 0, 1).astype("int8")
    out[score < cuts[pred]] = 2
    return out


def selector_score(known_labels, known_pred, proxy_pred):
    known = balanced_known(known_labels, known_pred)
    unknown = float(np.mean(proxy_pred == 2))
    return dict(
        known_macro_recall=known,
        proxy_unknown_recall=unknown,
        objective=0.5 * (known + unknown),
        known_BEN_recall=float(np.mean(known_pred[known_labels == 0] == 0)),
    )


def data_signature(setting):
    return {
        n: digest(DATA / setting / n)
        for n in ("known.npz", "query.npz", "truth.npz", "manifest.json")
    }
