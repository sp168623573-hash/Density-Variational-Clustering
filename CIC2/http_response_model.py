from io34 import *
from scipy.spatial.distance import cdist

METHODS = (
    "unseen_any",
    "continuation_any",
    "continuation_success",
    "continuation_error",
)


def unique(m, HTTP=False, known_conflicts=False):
    buckets = {}
    for j in range(len(m["ids"])):
        a, b = m["offsets"][j : j + 2]
        if not m["packet_available"][j] or b <= a:
            continue
        if HTTP and b - a != 1:
            continue
        if len(set(m["view_tuple_keys"][a:b])) != 1:
            continue
        buckets.setdefault(str(m["view_tuple_keys"][a]), []).append(j)
    out = []
    for key, at in buckets.items():
        if known_conflicts and len(set(m["fine"][at])) != 1:
            continue
        if HTTP:
            at = [j for j in at if m["available"][j]]
        if at:
            out.append(
                min(
                    at, key=lambda j: hashlib.sha256(str(m["ids"][j]).encode()).digest()
                )
            )
    return np.asarray(sorted(out), int)


def fit(m):
    at = unique(m, True, True)
    vi = m["offsets"][at]
    assert len(at) <= 6000
    seen = []
    prefix = []
    for part in range(6):
        kk = []
        pp = []
        for j in vi:
            a, b = m["gram_offsets"][6 * j + part : 6 * j + part + 2]
            kk.extend(m["gram_keys"][a:b])
            pp.extend(m["gram_prefix"][a:b])
        seen.append(np.unique(np.asarray(kk, np.uint64)))
        prefix.append(np.unique(np.asarray(pp, np.uint64)))
    supported = np.flatnonzero(np.bincount(m["fine"][at], minlength=9) >= 20)
    return dict(
        source_ids=m["ids"][at],
        source_tuples=m["view_tuple_keys"][vi],
        supported=supported,
        seen=seen,
        prefix=prefix,
        banks={"source_dictionary": {"principal_dimension": 0}},
        dictionary_sizes=[len(k) for k in seen],
        source_only_dictionary=True,
    )


def contains(reference, values):
    if len(reference) == 0:
        return np.zeros(len(values), bool)
    ix = np.searchsorted(reference, values)
    return (ix < len(reference)) & (
        reference[np.minimum(ix, len(reference) - 1)] == values
    )


def score(m, state, check):
    n = len(m["view_available"])
    u = np.zeros((n, 6))
    c = u.copy()
    for j in range(n):
        if j % 128 == 0:
            check()
        if not m["view_available"][j]:
            continue
        for part in range(6):
            a, b = m["gram_offsets"][6 * j + part : 6 * j + part + 2]
            k = m["gram_keys"][a:b]
            pr = m["gram_prefix"][a:b]
            w = m["gram_mass"][a:b]
            if not len(state["seen"][part]):
                continue
            novel = ~contains(state["seen"][part], k)
            u[j, part] = w[novel].sum()
            c[j, part] = w[novel & contains(state["prefix"][part], pr)].sum()
    view = dict(
        unseen_any=u.max(1),
        continuation_any=c.max(1),
        continuation_success=c[:, :2].max(1),
        continuation_error=c[:, 4:6].max(1),
    )
    index = np.repeat(np.arange(len(m["ids"])), np.diff(m["offsets"]))
    scores = {}
    for k, v in view.items():
        row = np.full(len(m["ids"]), np.inf)
        np.minimum.at(row, index, v)
        row[~m["available"]] = 0
        assert np.isfinite(row).all()
        scores[k] = row
    return scores, np.zeros((len(m["ids"]), 9))


def finite(v):
    rank = (99 * (len(v) + 1) + 99) // 100
    assert len(v) >= 99 and rank <= len(v)
    return float(np.sort(v)[rank - 1])


def calibrate(qs, qm, qy, qa, cs, cm, cy, ci, extra, em, ey, ei):
    cuts = {}
    detail = {}
    for k in METHODS:
        primary = [finite(qs[k][qa[(qy[qa] > 0) == c]]) for c in (0, 1)]
        safety = float(cs[k][ci].max(initial=0))
        added = []
        counts = []
        for c in (0, 1):
            values = extra[k][ei[(ey[ei] > 0) == c]]
            counts.append(len(values))
            added.append(
                finite(values) if len(values) >= 99 else float(values.max(initial=0))
            )
        assert min(counts) >= 20, "INSUFFICIENT_GENUINE_ADDITIONAL_KNOWN_CONTROL"
        cuts[k] = max(*primary, safety, *added)
        detail[k] = dict(
            primary_sourceQA_finite99=primary,
            original_CAL_observed_max=safety,
            additional_known_control_bounds=added,
            additional_control_counts=counts,
            small_CAL_not_claimed_finite99=True,
        )
    known = []
    proof = []
    for c in range(9):
        coarse = int(c > 0)
        cut = float(
            max(
                0,
                qm[qa[((qy[qa] > 0).astype(int)) != coarse], c].max(initial=0),
                cm[ci[cy[ci] != coarse], c].max(initial=0),
                em[ei[((ey[ei] > 0).astype(int)) != coarse], c].max(initial=0),
            )
        )
        ok = qm[qa, c] > cut
        row = dict(
            correct=int(np.sum(ok & ((qy[qa] > 0) == coarse))),
            wrong=int(np.sum(ok & ((qy[qa] > 0) != coarse))),
        )
        known.append(cut)
        proof.append(row)
    return dict(
        novelty=cuts,
        known_cuts=known,
        known_qualified=[v["correct"] >= 20 and v["wrong"] == 0 for v in proof],
        sourceQA_proof=proof,
        calibration_details=detail,
        calibration_denominator="SourceQA primaryfinite99; disjoint additional-known VALID provides independent new-confidence control; originalavailable CAL usesmaximum soeveryobservedknown score protected. No standalone99finite guarantee fromsmallCAL.",
    )


def fixtures():
    keys = np.array([1, 7, 10], np.uint64)
    q = np.array([0, 1, 2, 7, 100], np.uint64)
    assert contains(keys, q).tolist() == [False, True, False, True, False]
    assert not contains(np.array([], np.uint64), q).any()
    w = np.array([0.25, 0.75])
    new = ~contains(np.array([1], np.uint64), np.array([1, 2], np.uint64))
    prefix = contains(np.array([5], np.uint64), np.array([5, 5], np.uint64))
    assert w[new].sum() == 0.75 and w[new & prefix].sum() == 0.75
    return dict(
        full64bit_keys_not_modulo256=True,
        sorted_dictionary_membership_literal=True,
        novel_continuation_weight_literal=True,
        no_QUERY_vocabulary_fit=True,
    )
