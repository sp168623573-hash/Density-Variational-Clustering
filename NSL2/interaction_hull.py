"""Feasible local convex support and a certified local-hull distance lower bound.

Query weights are temporary inference coordinates. No reconstructed point is
used for training or added to DVC. This is not a global-knownness certificate.
"""

from rate210 import *
import rate210 as Old

METHODS = ("hull_lower", "hull_relative")
Old.METHODS = METHODS

import events241 as Events

Old.METHODS = METHODS
log_values = Events.log_values


def coords(l, out, kind):
    assert kind in ("activity", "rates")
    if kind == "activity":
        return Events.physical(l, "activity")
    # Use duration, bidirectional byte counts, and connection counts in the ratios.
    import rate210 as Physical

    selected = np.column_stack([l[:, Events.FIELDS.index(k)] for k in Physical.NAMES])
    return Physical.physical(selected, "rates")


def simplex(v):
    u = np.sort(v, axis=1)[:, ::-1]
    cs = np.cumsum(u, axis=1) - 1
    idx = np.arange(1, v.shape[1] + 1)
    active = u - cs / idx > 0
    rho = active.sum(1) - 1
    theta = cs[np.arange(len(v)), rho] / (rho + 1)
    return np.maximum(v - theta[:, None], 0)


def project(q, b):
    # Center each neighborhood on its query before computing the Gram matrix.
    a = b - q[:, None, :]
    g = np.einsum("nkd,njd->nkj", a, a, optimize=False)
    step = 1 / np.maximum(np.abs(g).sum(2).max(1), 1e-12)
    w = np.full(a.shape[:2], 1 / a.shape[1])
    v = w.copy()
    t = 1.0
    for _ in range(64):
        new = simplex(v - step[:, None] * np.einsum("nkj,nj->nk", g, v, optimize=False))
        nt = (1 + np.sqrt(1 + 4 * t * t)) / 2
        v = new + (t - 1) / nt * (new - w)
        w = new
        t = nt
    residual = np.einsum("nk,nkd->nd", w, a, optimize=False)
    upper = np.linalg.norm(residual, axis=1)
    dots = np.einsum("nkd,nd->nk", a, residual, optimize=False)
    lower = np.maximum(
        np.divide(dots.min(1), upper, out=np.zeros(len(q)), where=upper > 1e-12), 0
    )
    lower = np.minimum(lower, upper)
    effective = 1 / np.maximum(np.sum(w * w, 1), 1e-12)
    assert np.all(w >= -1e-12) and np.allclose(w.sum(1), 1, atol=1e-10)
    return lower, upper, effective


def fit(l, out, fine, cells, ids, kind, check):
    z = coords(l, out, kind)
    state = dict(
        kind=kind,
        ids=ids,
        fine=fine,
        cells={},
        feature_names=[kind + "_coordinate_" + str(j) for j in range(z.shape[1])],
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
        allbank = (z[at] - mu) / sd
        bb, idx = np.unique(allbank[fine[at] == 0], axis=0, return_index=True)
        if len(bb) < 20:
            continue
        state["cells"][str(cell)] = dict(
            indices=at, mean=mu, scale=sd, bank=allbank, BEN=bb, BEN_ids=ids[ben[idx]]
        )
    return state


def score(l, out, cells, available, state, check):
    z = coords(l, out, state["kind"])
    n = len(z)
    sc = {k: np.zeros(n) for k in METHODS}
    app = np.zeros(n, bool)
    upper = np.zeros(n)
    eff = np.zeros(n)
    cert = np.zeros(n, bool)
    pred = np.where(
        (out["fine_logits"].argmax(1) == 0) & (out["coarse_logits"].argmax(1) == 0),
        0,
        1,
    )
    for cell, b in state["cells"].items():
        at = np.flatnonzero((cells == cell) & available)
        app[at] = True
        for first in range(0, len(at), 128):
            check()
            ix = at[first : first + 128]
            q = (z[ix] - b["mean"]) / b["scale"]
            dist = cdist(q, b["BEN"])
            near = np.argsort(dist, axis=1, kind="stable")[:, :20]
            radius = np.take_along_axis(dist, near, 1)[:, -1]
            low, up, ef = project(q, b["BEN"][near])
            sc["hull_lower"][ix] = low
            sc["hull_relative"][ix] = low / np.maximum(radius, 1e-9)
            upper[ix] = up / np.maximum(radius, 1e-9)
            eff[ix] = ef
            all_d = cdist(q, b["bank"])
            r20 = np.partition(all_d, 19, axis=1)[:, 19]
            labels = state["fine"][b["indices"]]
            cert[ix] = np.asarray(
                [np.all(labels[dd <= rr] == 0) for dd, rr in zip(all_d, r20)]
            ) & (ef >= 4)
    return sc, dict(
        applicable=app,
        cells=cells,
        predicted=pred,
        certificate=cert,
        margin=np.zeros(n),
        upper=upper,
        effective_support=eff,
    )


def calibrate(qs, qd, qy, qa, ad, ay, old, od, oy, cal, cd, cy, ci):
    cuts = Old.calibrate(qs, qd, qy, qa, ad, ay, old, od, oy, cal, cd, cy, ci)
    for cell in cuts["cuts"]:
        qb = (qd["cells"] == cell) & qd["applicable"] & (qy == 0)
        a = (ad["cells"] == cell) & ad["applicable"]
        o = (od["cells"] == cell) & od["applicable"]
        c = ci[(cd["cells"][ci] == cell) & cd["applicable"][ci]]
        negatives = np.r_[
            qd["upper"][(qd["cells"] == cell) & qd["applicable"] & (qy > 0)],
            od["upper"][o & (oy > 0)],
            cd["upper"][c[cy[c] > 0]],
        ]
        bound = float(np.median(qd["upper"][qb]))
        bound = min(bound, float(negatives.min())) if len(negatives) else bound
        proposed = (
            a & (ad["predicted"] == 0) & ad["certificate"] & (ad["upper"] < bound)
        )
        tp = int(np.sum(proposed & (ay == 0)))
        fp = int(np.sum(proposed & (ay > 0)))
        cuts["known"][cell] = dict(
            upper_cut=bound,
            qualified=tp >= 20 and fp == 0,
            QAUDIT_BEN=tp,
            QAUDIT_KNA_wrong=fp,
            source_KNA_bound_count=len(negatives),
            not_UNA_safety_proof=True,
        )
    return cuts


def decisions(sc, det, cal):
    raw = {k: np.zeros(len(det["applicable"]), bool) for k in METHODS}
    known = np.zeros(len(det["applicable"]), bool)
    for cell, by in cal["cuts"].items():
        at = (det["cells"] == cell) & det["applicable"] & (det["predicted"] == 0)
        for k, cfg in by.items():
            if cfg["qualified"]:
                raw[k] |= at & (sc[k] > cfg["cut"])
        c = cal["known"][cell]
        if c["qualified"]:
            known |= at & det["certificate"] & (det["upper"] < c["upper_cut"])
    return raw, known


def fixtures():
    b = np.array([[[0.0], [1.0]], [[0.0], [1.0]]])
    lo, up, e = project(np.array([[0.5], [2.0]]), b)
    assert up[0] < 1e-12 and np.isclose(lo[1], 1) and np.isclose(up[1], 1)
    tri = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    lo, up, e = project(np.array([[0.2, 0.3]]), tri)
    assert lo[0] <= up[0] < 1e-7
    return dict(
        simplex_weights_feasible=True,
        one_dimensional_exact_distance=True,
        triangle_interior_reconstruction=True,
        lower_bound_not_optimizer_error=True,
        no_reconstructed_density_point=True,
    )
