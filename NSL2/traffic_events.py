"""Known-only event relationships. Rounded rates are features, not event counts or p-values."""

from rate210 import *
import rate210 as Old

METHODS = ("relative7", "lof7")
Old.METHODS = METHODS
FIELDS = (
    "duration",
    "src_bytes",
    "dst_bytes",
    "count",
    "srv_count",
    "dst_host_count",
    "dst_host_srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "hot",
    "num_failed_logins",
    "num_compromised",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "logged_in",
    "root_shell",
    "su_attempted",
    "is_host_login",
    "is_guest_login",
)

NAMES = FIELDS


def log_values(setting, A, x, duration):
    if setting == "NSL1":
        enc = read(
            ROOT
            / "data/external_dev/processed/nsl1_expanded_FIT6000_all41_20261009_v1/material/new_unique_FIT6000_all41_encoder_before_training.json"
        )
        v = {}
        for n, i in enumerate(enc["numeric_indices"]):
            g = enc["original_field_groups"][i]
            v[g["field"]] = (
                x[:, g["selected_columns"][0]].astype(float)
                * enc["log_numeric_FIT_RMS"][n]
            )
    else:
        enc = read(A.C.Adapter.manifest()["encoding"]["path"])
        v = {
            name: x[:, j].astype(float) * enc["std"][j] + enc["mean"][j]
            for j, name in enumerate(enc["numeric"])
        }
        v["duration"] = np.log1p(duration)
    z = np.column_stack([v[k] for k in FIELDS])
    assert np.isfinite(z).all() and z.min() > -2e-5
    return np.maximum(z, 0)


def physical(l, kind):
    v = {k: l[:, j] for j, k in enumerate(FIELDS)}
    r = {k: np.expm1(x) for k, x in v.items()}
    if kind == "activity":
        total = np.logaddexp(v["src_bytes"], v["dst_bytes"])
        den = 1 + np.log1p(np.maximum(np.exp(total) - 2, 0))
        events = FIELDS[21:28]
        return np.column_stack(
            [v[k] for k in FIELDS[:3]]
            + [v[k] / den for k in events]
            + [v[k] / (1 + v["duration"]) for k in events]
            + [r[k] for k in FIELDS[28:]]
        )
    assert kind == "errors"
    pairs = (
        ("serror_rate", "count", "srv_serror_rate", "srv_count"),
        ("rerror_rate", "count", "srv_rerror_rate", "srv_count"),
        (
            "dst_host_serror_rate",
            "dst_host_count",
            "dst_host_srv_serror_rate",
            "dst_host_srv_count",
        ),
        (
            "dst_host_rerror_rate",
            "dst_host_count",
            "dst_host_srv_rerror_rate",
            "dst_host_srv_count",
        ),
        ("serror_rate", "count", "dst_host_serror_rate", "dst_host_count"),
        ("rerror_rate", "count", "dst_host_rerror_rate", "dst_host_count"),
        (
            "srv_serror_rate",
            "srv_count",
            "dst_host_srv_serror_rate",
            "dst_host_srv_count",
        ),
        (
            "srv_rerror_rate",
            "srv_count",
            "dst_host_srv_rerror_rate",
            "dst_host_srv_count",
        ),
        ("same_srv_rate", "count", "dst_host_same_srv_rate", "dst_host_count"),
        ("diff_srv_rate", "count", "dst_host_diff_srv_rate", "dst_host_count"),
        (
            "srv_diff_host_rate",
            "srv_count",
            "dst_host_srv_diff_host_rate",
            "dst_host_srv_count",
        ),
    )
    out = []
    for pn, nn, qn, mn in pairs:
        n, m = r[nn], r[mn]
        p, q = np.clip(r[pn], 0, 1), np.clip(r[qn], 0, 1)
        a = (n * p + 0.5) / (n + 1)
        b = (m * q + 0.5) / (m + 1)
        out.append(
            np.sqrt((n + 1) * (m + 1) / (n + m + 2))
            * (np.arcsin(np.sqrt(a)) - np.arcsin(np.sqrt(b)))
        )
    return np.column_stack([v[k] for k in FIELDS[3:7]] + out)


def fit(l, out, fine, cells, ids, kind, check):
    z = physical(l, kind)
    state = dict(
        kind=kind,
        ids=ids,
        fine=fine,
        cells={},
        feature_names=[kind + "_relation_" + str(j) for j in range(z.shape[1])],
    )
    for cell in sorted(set(cells)):
        at = np.flatnonzero(cells == cell)
        ben = at[fine[at] == 0]
        if len(ben) < 20 or "<UNSEEN>" in cell:
            continue
        check()
        mu = np.median(z[ben], 0)
        sd = np.quantile(z[ben], 0.75, axis=0) - np.quantile(z[ben], 0.25, axis=0)
        sd = np.where(sd > 1e-6, sd, 1.0)
        bank = (z[at] - mu) / sd
        bb, ix = np.unique(bank[fine[at] == 0], axis=0, return_index=True)
        if len(bb) < 20:
            continue
        d = cdist(bb, bb)
        np.fill_diagonal(d, np.inf)
        rad = np.partition(d, 6, axis=1)[:, 6]
        reach = np.asarray(
            [
                np.maximum(rad[row <= rad[j]], row[row <= rad[j]]).mean()
                for j, row in enumerate(d)
            ]
        )
        assert np.all(rad > 0)
        state["cells"][str(cell)] = dict(
            indices=at,
            mean=mu,
            scale=sd,
            bank=bank,
            BEN_bank=bb,
            BEN_ids=ids[ben[ix]],
            radius=rad,
            reach=reach,
            KNA_available=int(np.sum(fine[at] > 0)) >= 20,
        )
    return state


def score(l, out, cells, available, state, check):
    z = physical(l, state["kind"])
    n = len(l)
    sc = {k: np.zeros(n) for k in METHODS}
    app = np.zeros(n, bool)
    cert = np.zeros(n, bool)
    margin = np.zeros(n)
    pred = np.where(
        (out["coarse_logits"].argmax(1) == 0) & (out["fine_logits"].argmax(1) == 0),
        0,
        1,
    )
    for cell, b in state["cells"].items():
        at = np.flatnonzero((cells == cell) & available)
        app[at] = True
        for first in range(0, len(at), 256):
            check()
            ix = at[first : first + 256]
            x = (z[ix] - b["mean"]) / b["scale"]
            d = cdist(x, b["BEN_bank"])
            rad = np.partition(d, 6, axis=1)[:, 6]
            all_d = cdist(x, b["bank"])
            labels = state["fine"][b["indices"]]
            for j, row in enumerate(ix):
                near = d[j] <= rad[j]
                reach = np.maximum(b["radius"][near], d[j, near]).mean()
                sc["relative7"][row] = rad[j] / b["radius"][near].mean()
                sc["lof7"][row] = np.mean(reach / b["reach"][near])
                ar = np.partition(all_d[j], 6)[6]
                pure = np.all(labels[all_d[j] <= ar] == 0)
                if b["KNA_available"]:
                    margin[row] = np.log(
                        max(all_d[j, labels > 0].min(), 1e-9) / max(rad[j], 1e-9)
                    )
                    cert[row] = (
                        pred[row] == 0
                        and pure
                        and np.sum(d[j, near] <= b["radius"][near]) >= 4
                    )
    assert all(np.isfinite(v).all() for v in sc.values())
    return sc, dict(
        applicable=app, cells=cells, predicted=pred, certificate=cert, margin=margin
    )


def fixtures():
    z = np.zeros((2, len(FIELDS)))
    a = physical(z, "errors")
    b = physical(z, "activity")
    assert np.isfinite(a).all() and np.isfinite(b).all() and not a.any() and not b.any()
    row = np.zeros((1, len(FIELDS)))
    row[0, FIELDS.index("count")] = np.log1p(9)
    row[0, FIELDS.index("srv_count")] = np.log1p(9)
    row[0, FIELDS.index("serror_rate")] = np.log1p(0.2)
    row[0, FIELDS.index("srv_serror_rate")] = np.log1p(0.2)
    assert physical(row, "errors")[0, 4] == 0
    row[0, FIELDS.index("serror_rate")] = np.log1p(0.8)
    assert physical(row, "errors")[0, 4] > 0
    return dict(
        zero_counts_finite_and_symmetric=True,
        equal_rates_equal_counts_zero_contrast=True,
        positive_difference_sign_checked=True,
        no_independent_binomial_or_integer_event_claim=True,
        source_neighbors_exact_aliases_collapsed=True,
    )
