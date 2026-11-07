from io34 import *
import runtime as E
import routing as R
import rate210 as M
import mc_core as Core
from union_cache import UnionMetric
from vector_review210 import review
from threadpoolctl import threadpool_limits
import pickle, time

S, A = E.SETTING, E.A
assert S in ("NSL1", "NSL2")
KIND = os.environ["R210_KIND"]
assert KIND in ("volumes", "rates")
D = E.D / KIND
E.D = D
MAT = I / "conditional_known_calibration_20261104_r148" / S
ARMS = {
    name + "_" + g: (name, g == "guarded", False)
    for name in M.METHODS
    for g in ("guarded", "reviewed")
}
ARMS["BEN_return"] = (None, False, True)
ARMS["joint"] = ("relative7", True, True)


def check():
    E.check()


def eq(a, b):
    for k, v in a.items():
        assert np.array_equal(v, b[k]), k


def duration(role, ids):
    if S == "NSL1":
        return None, np.ones(len(ids), bool)
    d = load(MAT / "duration" / (role + "_duration.npz"))
    assert np.array_equal(d["ids"], ids)
    return np.where(d["available"], d["duration"], 0), d["available"]


def apply(q, z, sc, det, cuts, g, method, guard, ret):
    novel, known = M.decisions(sc, det, cuts)
    raw = (
        (q["parent"] == 0) & novel[method] if method else np.zeros(len(q["ids"]), bool)
    )
    back = (q["parent"] != 0) & known if ret else np.zeros(len(raw), bool)
    if guard:
        raw &= ~q["veto"]
    raw &= ~back
    for mask in (raw, back):
        for group in np.unique(q["groups"][mask]):
            for before in (0, 1, 2):
                at = (q["groups"] == group) & (q["parent"] == before)
                if mask[at].any() and not mask[at].all():
                    mask[at] = False
    context = ((q["parent"] == 2) & ~back) | raw
    den = (
        R.route(q, z, context, g)
        if raw.any()
        else dict(retained=np.zeros(len(raw), bool))
    )
    pred = q["parent"].copy()
    pred[back] = 0
    pred[raw & den["retained"]] = 2
    return (
        dict(
            prediction=pred,
            raw=raw,
            retained=den["retained"],
            context=context,
            known_reclassification=back,
        ),
        den,
    )


def run():
    T = time.perf_counter()
    recipe = read(H / "FROZEN_RECIPE210.json")
    for v in recipe["code"] + recipe["inputs"][S]:
        assert desc(v["path"]) == v
    b = M.BIND[S]
    source = load(b["source_roster"]["path"])
    save(D / "source.npz", **source)
    p = M.Parent.Predictor(S, A)
    x = Core.inputs(S, A, source, p.owner, True)
    raw = p.raw(x, check)
    base = p.clean(raw)
    cache = load(b["neural_source_forward"]["path"])
    eq(base, {k: cache[k] for k in base})
    sourcecells, _ = M.metadata(S, A, x)
    du, available = duration("source", source["ids"])
    l = M.log_values(S, A, x, du)
    fit = source["fit"][available[source["fit"]]]
    state = M.fit(
        l[fit],
        {k: v[fit] for k, v in base.items()},
        source["fine"][fit],
        sourcecells[fit],
        source["ids"][fit],
        KIND,
        check,
    )
    with (D / "rate_bank.pkl").open("xb") as f:
        pickle.dump(state, f, protocol=5)
    save(
        D / "source_physical_log_values.npz",
        ids=source["ids"],
        log_values=l,
        available=available,
    )
    put(
        D / "FIT_AUDIT.json",
        dict(
            only_known_FIT=True,
            no_neural_weight_updates=True,
            actual_FIT=len(fit),
            source_ids=source["ids"][fit],
            physical_features=state["feature_names"],
            cells=len(state["cells"]),
            missing_duration_abstains=True,
            **STATUS,
        ),
    )
    extra = {}
    for role in ("QCAL", "QAUDIT"):
        q = load(MAT / (role + ".npz"))
        cells, _ = M.metadata(S, A, q["neural"])
        ok = cells == q["cells"]
        assert all("<UNSEEN>" in c for c in cells[~ok])
        rr = p.raw(q["neural"], check)
        out = p.clean(rr)
        ll = M.log_values(S, A, q["neural"], None if S == "NSL1" else q["duration"])
        sc, det = M.score(ll, out, cells, ok, state, check)
        save(D / (role + "_scores.npz"), ids=q["ids"], **sc)
        save(D / (role + "_details.npz"), ids=q["ids"], **det)
        extra[role] = (sc, det, q["fine"])
        print(
            "RATE_KNOWN_CONTROL",
            S,
            KIND,
            role,
            len(q["ids"]),
            time.perf_counter() - T,
            flush=True,
        )
    qa = M.Parent.split_source(source)["qa"]
    old, od = M.score(
        l[qa],
        {k: v[qa] for k, v in base.items()},
        sourcecells[qa],
        available[qa],
        state,
        check,
    )
    metricmod = mod(b["own_metric_module"]["path"], "r210_metric_" + S)
    metric = metricmod.PaperVariationalPairMetric.from_state(
        read(b["own_metric"]["path"])
    )
    units = load(b["source_geometry"]["path"])["units"]
    outputs = {a: {} for a in ARMS}

    def compute(role, q):
        xx = Core.inputs(S, A, q, p.owner)
        rr = p.raw(xx, check)
        out = p.clean(rr)
        cells, _ = M.metadata(S, A, xx)
        du, ok = duration(role, q["ids"])
        ll = M.log_values(S, A, xx, du)
        sc, det = M.score(ll, out, cells, ok, state, check)
        return rr, ll, sc, det, np.c_[q["x"], out["embedding"] / units]

    for role in ("CAL", "CP", "current"):
        q = R.role(role)
        rr, ll, sc, det, z = compute(role, q)
        save(D / (role + "_raw.npz"), ids=q["ids"], **rr)
        save(D / (role + "_physical.npz"), ids=q["ids"], log_values=ll)
        save(D / (role + "_scores.npz"), ids=q["ids"], **sc)
        save(D / (role + "_details.npz"), ids=q["ids"], **det)
        save(D / (role + "_geometry.npz"), ids=q["ids"], Z=z)
        if role == "CAL":
            ids, y = q["control_labels"]
            cuts = M.calibrate(
                *extra["QCAL"],
                *extra["QAUDIT"],
                old,
                od,
                source["fine"][qa],
                sc,
                det,
                y,
                A.calibration_indices(A.role(role)),
            )
            put(D / "cuts.json", dict(cuts, **STATUS))
            print("RATE_QUALIFIED_CELLS", S, KIND, len(cuts["cuts"]), flush=True)
        g = dict(metric=UnionMetric(metric, z), routes={})
        for arm, cfg in ARMS.items():
            pred, den = apply(q, z, sc, det, cuts, g, *cfg)
            folder = D / arm / role
            folder.mkdir(parents=True)
            pr = save(
                folder / "prediction_before_truth.npz",
                ids=q["ids"],
                groups=q["groups"],
                parent_prediction=q["parent"],
                **pred,
            )
            if "manifest" in den:
                put(folder / "DVC.json", den["manifest"])
            seal = put(
                folder / "seal.json",
                dict(
                    prediction=pr,
                    recipe=desc(H / "FROZEN_RECIPE210.json"),
                    current_truth_reads=0,
                    **STATUS,
                ),
            )
            outputs[arm][role] = dict(prediction=pr, seal=seal)
            print(
                "RATE_SEALED",
                S,
                KIND,
                role,
                arm,
                int(pred["raw"].sum()),
                int(pred["known_reclassification"].sum()),
                flush=True,
            )
    put(
        D / "PREDICTIONS_SEALED.json",
        dict(outputs=outputs, all_complete=True, current_truth_reads=0, **STATUS),
    )
    with (D / "rate_bank.pkl").open("rb") as f:
        state = pickle.load(f)

    def replay(role, q):
        rr, ll, sc, det, z = compute(role, q)
        eq(rr, load(D / (role + "_raw.npz")))
        eq(sc, load(D / (role + "_scores.npz")))
        eq(det, load(D / (role + "_details.npz")))
        assert np.array_equal(
            ll, load(D / (role + "_physical.npz"))["log_values"]
        ) and np.array_equal(z, load(D / (role + "_geometry.npz"))["Z"])
        if role == "CAL":
            ids, y = q["control_labels"]
            assert (
                M.calibrate(
                    *extra["QCAL"],
                    *extra["QAUDIT"],
                    old,
                    od,
                    source["fine"][qa],
                    sc,
                    det,
                    y,
                    A.calibration_indices(A.role(role)),
                )
                == cuts
            )
        g = dict(metric=UnionMetric(metric, z), routes={})
        return {
            arm: apply(q, z, sc, det, cuts, g, *cfg)[0] for arm, cfg in ARMS.items()
        }

    summary, report = review(S, outputs, R.role, replay, check)
    good = [
        a
        for a, v in summary.items()
        if all(t["no_metric_regression"] and t["group_consistent"] for t in v.values())
        and sum(v["current"]["delta"]) > 0
    ]
    selected = (
        max(good, key=lambda a: sum(summary[a]["current"]["correct"])) if good else None
    )
    put(D / "VERIFICATION.json", dict(candidates=summary, selected=selected, **STATUS))
    put(D / "REPORT.json", dict(results=report, **STATUS))
    print("RATE_ADMISSION", S, KIND, selected, time.perf_counter() - T)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
