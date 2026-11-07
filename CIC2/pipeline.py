from io34 import *
import runtime as E
import routing as R
import gradient_model as Base
import mc_core as Core
import symbol_model266 as M
from union_cache import UnionMetric, fixture
from vector_review266 import review
from threadpoolctl import threadpool_limits
import pickle, time

S, A = E.SETTING, E.A
assert S == "CIC2"
KIND = os.environ["R266_KIND"]
assert KIND in ("http",)
D = E.D / KIND
E.D = D
MAT = H / "material" / KIND
ARMS = {
    k + "_" + g: (k, g == "guarded", (), "known")
    for k in M.METHODS
    for g in ("guarded", "reviewed")
}


def check():
    E.check()


def eq(a, b):
    for k, v in a.items():
        assert np.array_equal(v, b[k]), k


def supported_scores(role, m, state, sc):
    cached = load(
        I / "source_covered_mhfd_20261104_r147/CIC2/flow/source_parts" / (role + ".npz")
    )
    assert np.array_equal(cached["ids"], m["ids"])
    gate = np.isin(cached["fine_logits"].argmax(1), state["supported"])
    return {k: np.where(gate, v, 0.0) for k, v in sc.items()}


def apply(q, z, sc, margin, cuts, g, method, guard, classes, scope):
    raw = (
        ((q["parent"] == 0) if scope == "BEN" else np.isin(q["parent"], [0, 1]))
        & q["supported_view"]
        & (sc[method] > cuts["novelty"][method])
        if method
        else np.zeros(len(q["ids"]), bool)
    )
    cert = np.full(len(raw), -1, int)
    for c in classes:
        if cuts["known_qualified"][c] and c in cuts["supported"]:
            cert[margin[:, c] > cuts["known_cuts"][c]] = int(c > 0)
    back = (cert >= 0) & (cert != q["parent"])
    if guard:
        raw &= ~q["veto"]
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
    pred[back] = cert[back]
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
    recipe = read(H / "FROZEN_RECIPE266.json")
    for ref in recipe["code"] + recipe["inputs"]:
        assert desc(ref["path"]) == ref
    fm = load(MAT / "FIT.npz")
    state = M.fit(fm)
    with (D / "symbol_bank.pkl").open("xb") as f:
        pickle.dump(state, f, protocol=5)
    put(
        D / "FIT_AUDIT.json",
        dict(
            only_known_FIT=True,
            no_neural_weight_updates=True,
            actual_FIT=len(state["source_ids"]),
            source_ids=state["source_ids"],
            supported_original_fine=state["supported"],
            one_reference_per_actual_capture_tuple=True,
            native_span_reference=True,
            feature_kind=KIND,
            dimensions={k: v["principal_dimension"] for k, v in state["banks"].items()},
            **STATUS,
        ),
    )
    binding = Base.BIND[S]
    source = load(binding["source_roster"]["path"])
    save(D / "source.npz", **source)
    p = Base.Predictor(S, A)
    base = p.clean(p.raw(Core.inputs(S, A, source, p.owner, True), check))
    cache = load(binding["neural_source_forward"]["path"])
    eq(base, {k: cache[k] for k in base})
    qm = load(MAT / "QA.npz")
    qs, qmargin = M.score(qm, state, check)
    qs = supported_scores("QA", qm, state, qs)
    qa = M.unique(qm, False, True)
    save(D / "source_QA_scores.npz", ids=qm["ids"], margin=qmargin, **qs)
    vm = load(MAT / "VALID.npz")
    vs, vmargin = M.score(vm, state, check)
    vs = supported_scores("VALID", vm, state, vs)
    extraids = set(
        load(I / "known_subtype_coverage_20261104_r146/material/VALID.npz")["ids"]
    )
    qa_tuples = {str(qm["view_tuple_keys"][qm["offsets"][j]]) for j in qa}
    va = np.asarray(
        [
            j
            for j in M.unique(vm, False, True)
            if vm["ids"][j] in extraids
            and str(vm["view_tuple_keys"][vm["offsets"][j]]) not in qa_tuples
        ],
        int,
    )
    assert len(set(vm["ids"][va])) == len(va)
    save(
        D / "additional_known_CAL_scores.npz",
        ids=vm["ids"],
        used_indices=va,
        fine=vm["fine"],
        margin=vmargin,
        **vs,
    )
    put(
        D / "ADDITIONAL_CAL_ROLE_PROOF.json",
        dict(
            source_ids=vm["ids"][va],
            only_declared_existing_known_supplemental_VALID=True,
            new_confidence_statistics_fit_only_FIT=True,
            sourceQA_tuple_overlap_zero=True,
            no_extra_accuracy_rows=True,
            ancestor_auxiliary_validation_exposure_disclosed=True,
            **STATUS,
        ),
    )
    metricmod = mod(binding["own_metric_module"]["path"], "r189_metric")
    metric = metricmod.PaperVariationalPairMetric.from_state(
        read(binding["own_metric"]["path"])
    )
    units = load(binding["source_geometry"]["path"])["units"]
    outputs = {a: {} for a in ARMS}
    role_metrics = {}
    put(
        D / "DISTANCE_CACHE_SOURCE_FIXTURE.json",
        dict(
            fixture(metric, np.c_[source["x"][:64], base["embedding"][:64] / units]),
            **STATUS,
        ),
    )

    def compute(role, q):
        m = load(MAT / (role + ".npz"))
        assert np.array_equal(m["ids"], q["ids"])
        rr = p.raw(Core.inputs(S, A, q, p.owner), check)
        o = p.clean(rr)
        sc, margin = M.score(m, state, check)
        q["supported_view"] = np.isin(o["fine_logits"].argmax(1), state["supported"])
        sc = {k: np.where(q["supported_view"], v, 0.0) for k, v in sc.items()}
        z = np.c_[q["x"], o["embedding"] / units]
        return m, rr, sc, margin, z

    for role in ("CAL", "CP", "current"):
        q = R.role(role)
        m, rr, sc, margin, z = compute(role, q)
        save(D / (role + "_raw.npz"), ids=q["ids"], **rr)
        save(D / (role + "_scores.npz"), ids=q["ids"], margin=margin, **sc)
        save(D / (role + "_coordinates.npz"), ids=q["ids"], Z=z)
        if role == "CAL":
            ids, y = q["control_labels"]
            ci = np.intersect1d(M.unique(m, False), A.calibration_indices(A.role(role)))
            cuts = M.calibrate(
                qs,
                qmargin,
                qm["fine"],
                qa,
                sc,
                margin,
                y,
                ci,
                vs,
                vmargin,
                vm["fine"],
                va,
            )
            cuts["supported"] = state["supported"].tolist()
            put(D / "cuts.json", dict(cuts, **STATUS))
            print("HTTP_SYMBOL_CUTS", cuts, flush=True)
        role_metrics[role] = UnionMetric(metric, z)
        g = dict(metric=role_metrics[role], routes={})
        for arm, cfg in ARMS.items():
            check()
            pred, den = apply(q, z, sc, margin, cuts, g, *cfg)
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
                    recipe=desc(H / "FROZEN_RECIPE266.json"),
                    current_truth_reads=0,
                    **STATUS,
                ),
            )
            outputs[arm][role] = dict(prediction=pr, seal=seal)
            print(
                "HTTP_SYMBOL_SEALED",
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
    with (D / "symbol_bank.pkl").open("rb") as f:
        state = pickle.load(f)

    def replay(role, q):
        m, rr, sc, margin, z = compute(role, q)
        eq(rr, load(D / (role + "_raw.npz")))
        eq(dict(sc, margin=margin), load(D / (role + "_scores.npz")))
        assert np.array_equal(z, load(D / (role + "_coordinates.npz"))["Z"])
        if role == "CAL":
            ids, y = q["control_labels"]
            assert (
                dict(
                    M.calibrate(
                        qs,
                        qmargin,
                        qm["fine"],
                        qa,
                        sc,
                        margin,
                        y,
                        np.intersect1d(
                            M.unique(m, False), A.calibration_indices(A.role(role))
                        ),
                        vs,
                        vmargin,
                        vm["fine"],
                        va,
                    ),
                    supported=state["supported"].tolist(),
                )
                == cuts
            )
        g = dict(metric=role_metrics[role], routes={})
        return {a: apply(q, z, sc, margin, cuts, g, *cfg)[0] for a, cfg in ARMS.items()}

    summary, report = review(S, outputs, R.role, replay, check)
    put(
        D / "DISTANCE_CACHE_REPLAY.json",
        dict(
            whole_raw_neural_scores_and_geometry_recomputed=True,
            geometry_bitwise_equal_before_cache_reuse=True,
            original_DVC_and_noise_reexecuted=True,
            only_immutable_pair_distances_reused=True,
            no_new_DVC_vertices=True,
            **STATUS,
        ),
    )
    good = [
        a
        for a, v in summary.items()
        if all(x["no_metric_regression"] and x["group_consistent"] for x in v.values())
        and sum(v["current"]["delta"]) > 0
    ]
    chosen = (
        max(good, key=lambda a: sum(summary[a]["current"]["correct"])) if good else None
    )
    put(D / "VERIFICATION.json", dict(candidates=summary, selected=chosen, **STATUS))
    put(D / "REPORT.json", dict(results=report, **STATUS))
    print("HTTP_SYMBOL_ADMISSION", chosen, time.perf_counter() - T)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
