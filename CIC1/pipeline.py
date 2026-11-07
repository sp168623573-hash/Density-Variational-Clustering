"""Whole preserved masked-MHFD32 auxiliaryview and unchanged126 paper DVC."""

import argparse, json, sys, time, types
from pathlib import Path
import numpy as np
from scipy.spatial import distance as spatial
from threadpoolctl import threadpool_limits
from . import cic1_MHFD32_soft_affinity_core_v142 as core
from . import cic1_soft_raw_affinity126_v133 as source
from . import cic1_native78_soft_view126_v141 as previous

ROOT, files, require = source.ROOT, source.files, source.require
old = source.old
MODULE = "paper_open_set.cic1_MHFD32_multiview126_v142"
DIR = ROOT / "tmp/cic1_MHFD32_multiview126_20261012_v142"
BASE = ROOT / "data/external_dev/processed/cic1_MHFD32_multiview126_20261012_v142"
REG = DIR / "registry_before_known.json"


class SemanticEuclidean:
    dimension = 32

    def pairwise_distances(self, q, r):
        require(
            q.ndim == r.ndim == 2
            and q.shape[1] == r.shape[1] == 32
            and np.isfinite(q).all()
            and np.isfinite(r).all(),
            "complete actual MHFD32 Euclidean input",
        )
        return spatial.cdist(q, r, metric="euclidean")


def decisions(raw, first, suspected, retained, native, full, semantic):
    require(
        raw.shape
        == first.shape
        == suspected.shape
        == retained.shape
        == native.shape
        == full.shape
        == semantic.shape
        and raw.dtype
        == suspected.dtype
        == retained.dtype
        == native.dtype
        == full.dtype
        == semantic.dtype
        == bool
        and np.all(first >= -2),
        "realwhole126 primary/secondary sentinel andthree fixedcue masks",
    )
    ordinary = raw & retained
    secondary = raw & (first >= 0) & suspected & ~retained
    noise = raw & (first == -1) & native & (full | semantic)
    return ordinary | secondary | noise, secondary, noise


def policy():
    return dict(
        schema_version="CIC1-fullpreserved-maskedMHFD32-native78-full126-multiview-v142",
        recipe=core.recipe(),
        hypotheses=1,
        representation="whole Z126[:,78:110] actual V48 full78 fieldmask-MHFD final80 cleanembedding32/genuineFIT RMS; not earlier legacyselected50 primaryclassifier embedding",
        actualproducer="frozen trueBEN+KNA6000 full78 mask-MHFD80, actualsaved raw5embedding beforeguards; preservedoriginalprimary coarse/fine classification remains unchanged",
        auxiliary="realSciPy cdistEuclidean32 auxiliarygraph, no newNNfit/forward/units/metric; mainpaperDVC remains owned126",
        initial="sameactual V126 rawcandidate function every CAL/CP/current, includesfalsepositives no truthfilter or rolezero",
        graph="allgenuineknown6000anchors0; fixedk7/sigma7/Gaussian/.9/20/softinitial/no hardqueryclamp; oneCALfinite99 plus engineeringfloor.5",
        previouscuts="native78 V141 and full126 V133 frozenknowncuts byteidentical, no recalibration",
        candidate="baseline!=2 AND (native78cue OR full126cue OR newMHFD32cue), allthreecompleteview fixedcut masks",
        primary_secondary="anycue candidate+real126primarylabel>=0 mayreviewsecondarysuspected rejection; originalordinaryretained preserved",
        primary_noise="requires native78cue AND(full126cue OR newMHFD32cue), first==-1 only; -2 unassessed neverrescued",
        density="allnewanycue raw OR allwhole3124oldpred2, nativephysicalgroupsonce, actualsame126 SC/CH/knee andnoise",
        preservation="complete646/247/3124 old2 including mistakes intact; original3121/3114/3059/2402 retained separately",
        correlatedviews_not_independent_probability=True,
        extension_not_verbatim_paper=True,
        NN_fit=0,
        NN_forward=0,
        metric_fit=0,
        R7_fit=0,
        trueUNA_fit=False,
        query_fit=False,
        parameter_grid=False,
        timecap_coded=False,
        status="FAIL_TARGET99",
        published=False,
        final_test_executed=False,
    )


def closure():
    found = {}
    seen = set()
    todo = [sys.modules[__name__]]
    while todo:
        m = todo.pop()
        if m is None or id(m) in seen:
            continue
        seen.add(id(m))
        p = getattr(m, "__file__", None)
        if not p or not Path(p).resolve().is_relative_to(ROOT / "paper_open_set"):
            continue
        found[MODULE if m is sys.modules[__name__] else m.__name__] = files.item(p)
        for value in vars(m).values():
            child = (
                value
                if isinstance(value, types.ModuleType)
                else sys.modules.get(getattr(value, "__module__", ""))
            )
            if child is not None:
                todo.append(child)
    return found


def archive(a, keys, pins):
    return old.archive(a, keys, pins)


def save(out, r, key, value):
    return old.save(out, r, key, value)


def synthetic_contract():
    contract = core.synthetic_contract()
    raw = np.ones(6, bool)
    first = np.array([-2, -1, -1, -1, 0, 1])
    suspected = np.ones(6, bool)
    retained = np.array([False, False, False, False, False, True])
    native = np.array([True, False, True, True, False, False])
    full = np.array([True, True, False, True, False, False])
    semantic = np.array([True, True, True, False, True, False])
    accepted, secondary, noise = decisions(
        raw, first, suspected, retained, native, full, semantic
    )
    require(
        accepted.tolist() == [False, False, True, True, True, True]
        and noise.tolist() == [False, False, True, True, False, False],
        "uniformnative AND(full ORsemantic), no semantic-alone primarynoise rescue, -2 blocked",
    )
    x = np.zeros((3, 32))
    x[1, 31] = 1.0
    d = SemanticEuclidean().pairwise_distances(x, x)
    require(
        d[0, 1] == 1 and d[1, 0] == 1 and np.all(np.diag(d) == 0),
        "actualcdist lastcomplete32field+self",
    )
    return dict(
        contract,
        actualcdist_all32=True,
        wholethreeview_secondary=True,
        native_plus_full_orsemantic_primarynoise=True,
        minus2_unassessed_neverrescued=True,
    )


def freeze():
    started = time.perf_counter()
    pins = {}
    sr = files.item(source.REG)
    s = files.js(sr, pins)
    sk = files.item(source.BASE / "known/report.json")
    sc = files.item(source.BASE / "current/report.json")
    ks = files.js(sk, pins)
    cs = files.js(sc, pins)
    pr = files.item(previous.REG)
    p = files.js(pr, pins)
    pk = files.item(previous.BASE / "known/report.json")
    pc = files.item(previous.BASE / "current/report.json")
    kp = files.js(pk, pins)
    cp = files.js(pc, pins)
    require(
        kp["known_gate_passed"]
        and cp["prediction_sealed"]
        and cp["original_class_gates"]["UNA"]["correct"] == 3124,
        "actualwhole3124 physicalview baseline coupled",
    )
    a = dict(s["assets"])
    a.update(
        known_baseline=kp["files"]["complete_multiviewprediction_beforetruth"],
        baseline=cp["files"]["complete_multiviewprediction_beforetruth"],
        native_cut=kp["files"]["rule"],
        CP_nativegraph=kp["files"]["CP_graph"],
        current_nativegraph=cp["files"]["current_graph"],
        full_cut=ks["files"]["rule"],
        CP_fullgraph=ks["files"]["CP_graph"],
        current_fullgraph=cs["files"]["current_graph"],
    )
    ownerknown_ref = files.item(old.cic.BASE / "known/report.json")
    owner = files.js(ownerknown_ref, pins)
    oreg = files.js(owner["registry"], pins)
    a["semantic_units"] = owner["files"]["units"]
    for role in ("FIT", "CAL", "CP", "current"):
        a[role + "_clean32"] = oreg["assets"][role + "_clean32"]
    mk = oreg["known_parent"]
    masked = files.js(mk, pins)
    mr = files.js(masked["registry"], pins)
    mc = files.item(
        ROOT
        / "data/external_dev/processed/cic1_mask78_clean110_20261011_v48/current/report.json"
    )
    maskedcurrent = files.js(mc, pins)
    a.update(
        MHFD_known_report=mk,
        MHFD_current_report=mc,
        MHFD_producer_report=mr["assets"]["producer_report"],
        MHFD_model=mr["assets"]["mask_model"],
    )
    producer = files.js(a["MHFD_producer_report"], pins)
    require(
        producer["epochs_completed"] == 80
        and producer["unknown_values_opened"] == False
        and producer["files"]["model"] == a["MHFD_model"]
        and producer["model_state_sha256"] == mr["mask_state_sha"],
        "actualfrozenfull78 known-only80 MHFD producer state identity",
    )
    src = closure()
    contract = synthetic_contract()
    for item in [*a.values(), *src.values()]:
        files.read(item, pins)
    runtime = dict(p["cdist_runtime_sources"])
    for item in runtime.values():
        require(
            files.item(item["path"]) == item, "exactinstalledcdist runtime immutable"
        )
        pins[str(Path(item["path"]).resolve())] = item["sha256"]
    priorclose = files.item(
        previous.DIR / "CIC1_V141_NATIVE78_AUXVIEW_3124_UP_COMPLETED_CLOSEOUT.json"
    )
    prior = files.js(priorclose, pins)
    require(
        not REG.exists() and not BASE.exists(),
        "exclusivewhole32 semanticview recipe namespace",
    )
    DIR.mkdir(parents=True, exist_ok=True)
    return files.write(
        REG,
        dict(
            policy=policy(),
            assets=a,
            sources=src,
            synthetic_contract=contract,
            cdist_runtime_sources=runtime,
            original_graph_registry=sr,
            original_graph_known=sk,
            original_graph_current=sc,
            prior_native_registry=pr,
            prior_native_known=pk,
            prior_native_current=pc,
            semantic126_owner_known=ownerknown_ref,
            semantic126_owner_registry=owner["registry"],
            MHFD_source_registry=masked["registry"],
            MHFD_actual_state_sha256=mr["mask_state_sha"],
            actual_representation="V48full78mask-MHFD80 clean32, primarylegacy classifier unchanged",
            ancestry_closeout=priorclose,
            prior_preparation_seconds=prior["actual_preparation"][
                "conservative_allprep_plus_ancestry_seconds"
            ],
            freeze_seconds=time.perf_counter() - started,
            input_sha256=pins,
            current_numeric_decodes=0,
        ),
    )


def bind(ref, pins):
    reg = files.js(ref, pins)
    require(
        reg["policy"] == policy() and reg["sources"] == closure(),
        "exactfrozen full32+allthreeview policy/source",
    )
    runtimes = {
        str(Path(x["path"]).resolve()) for x in reg["cdist_runtime_sources"].values()
    }
    for p, s in reg["input_sha256"].items():
        if str(Path(p).resolve()) in runtimes:
            require(
                files.item(p)["sha256"] == s, "registeredinstalledruntime unchanged"
            )
            pins[str(Path(p).resolve())] = s
        else:
            files.read(dict(path=p, sha256=s), pins)
    return reg


def semantic_proof(reg, role, ids, z, out, r, pins):
    cache = archive(reg["assets"][role + "_clean32"], ("embedding32",), pins)
    units = np.asarray(files.js(reg["assets"]["semantic_units"], pins)["clean32"])
    embedding = cache["embedding32"]
    require(
        embedding.shape == (len(ids), 32)
        and units.shape == (32,)
        and np.all(units > 0)
        and np.array_equal(embedding / units, z[:, 78:110]),
        "actualall32 unscaled MHFD cache/genuineFIT RMS exactlysameoriginal126block",
    )
    source_report = files.js(
        reg["assets"][
            "MHFD_current_report" if role == "current" else "MHFD_known_report"
        ],
        pins,
    )
    tag = "current4056" if role == "current" else role
    rawpieces = []
    forward = []
    for first in range(0, len(ids), 512):
        a = source_report["files"][
            tag + "_clean_raw5_batch" + str(first) + "_before_guard"
        ]
        raw = archive(a, ("embedding",), pins)
        require(
            raw["embedding"].shape == (min(512, len(ids) - first), 32)
            and raw["embedding"].dtype == np.float32,
            "actualsaved cleanforward raw32 dtype andbatchcoverage",
        )
        rawpieces.append(raw["embedding"])
        forward.append(a)
    require(
        np.array_equal(np.concatenate(rawpieces).astype(np.float64), embedding),
        "allactualrawforward FP32 outputs exactlyproducer cache32, no neural reforward",
    )
    if role == "current":
        coupled = archive(
            source_report["files"]["complete4056_MASK78_CLEAN110_before_scores"],
            ("ids", "Z110"),
            pins,
        )
        linkedids = coupled["ids"]
    else:
        coupled = archive(
            source_report["files"]["coupled_MASK78_CLEAN110_known_before_metric"],
            (role + "_ids", role + "_Z110"),
            pins,
        )
        linkedids = coupled[role + "_ids"]
    require(
        np.array_equal(linkedids, ids),
        "savedactualMHFD roleoutput orderedIDs match current126 role",
    )
    if role == "FIT":
        require(
            np.array_equal(
                units, old.cic.previous.geometry_helper.bridge.rms_unique(embedding)
            ),
            "actualgenuine6000 Fitter RMS32 verifieswithoutqueryfit",
        )
    r["files"][role + "_semantic_cache_source_proof"] = files.write(
        out / (role + "_full32_actualmaskedMHFD80_cache_units_proof.json"),
        dict(
            cache=reg["assets"][role + "_clean32"],
            units=reg["assets"]["semantic_units"],
            actual_forward_packets=forward,
            raw32_cache_and126block_bitexact=True,
            role_ids=old.cic.learned._array_descriptor(ids),
            model=reg["assets"]["MHFD_model"],
            actual_model_state_sha256=reg["MHFD_actual_state_sha256"],
            new_NN_forward=0,
            legacy_primary_encoder_claimed=False,
        ),
    )


def anchors(reg, metric, out, r, pins):
    mat = archive(
        reg["assets"]["material"],
        ("FIT_ids", "FIT_groups", "FIT_fine", "FIT_Z126"),
        pins,
    )
    labels = archive(reg["assets"]["native_known"], ("FIT_ids", "FIT_y_fine"), pins)
    require(
        len(mat["FIT_ids"]) == len(np.unique(mat["FIT_groups"])) == 6000
        and np.array_equal(mat["FIT_ids"], labels["FIT_ids"])
        and np.array_equal(mat["FIT_fine"], labels["FIT_y_fine"])
        and np.bincount(mat["FIT_fine"], minlength=9).min() >= 20,
        "actualgenuineknown6000 all9floor20 source",
    )
    require(
        metric.to_state()["prefit_descriptor"]["FIT_MHFD_Z"]
        == old.cic.learned._array_descriptor(mat["FIT_Z126"]),
        "originalcomplete126metric owner unchanged",
    )
    semantic_proof(reg, "FIT", mat["FIT_ids"], mat["FIT_Z126"], out, r, pins)
    order = np.argsort(mat["FIT_groups"], kind="stable")
    return dict(
        ids=mat["FIT_ids"][order],
        groups=mat["FIT_groups"][order],
        Z126=mat["FIT_Z126"][order],
        MHFD32=mat["FIT_Z126"][order, 78:110],
        fine=mat["FIT_fine"][order],
    )


def role_input(reg, role, out, r, pins):
    if role == "CAL":
        mat = archive(
            reg["assets"]["material"],
            tuple("CAL_" + k for k in ("ids", "groups", "Z126", "stage1")),
            pins,
        )
        q = {k: mat["CAL_" + k] for k in ("ids", "groups", "Z126", "stage1")}
        j = archive(reg["assets"]["CAL_joint"], ("ids", "score"), pins)
        g0 = archive(reg["assets"]["CAL_bank0"], ("ids", "eligible"), pins)
        g1 = archive(reg["assets"]["CAL_bank1"], ("ids", "eligible"), pins)
        raw = core.raw_candidate(
            j["score"],
            g0["eligible"] | g1["eligible"],
            q["stage1"],
            files.js(reg["assets"]["prior_rule"], pins),
        )
        baseline = None
    else:
        q, baseline, _ = old.query(
            "CIC1", "known" if role == "CP" else "current", reg["assets"], pins
        )
        j = archive(
            reg["assets"][role + "_joint"], ("ids", "score", "eligible", "raw"), pins
        )
        raw = core.raw_candidate(
            j["score"],
            j["eligible"],
            q["stage1"],
            files.js(reg["assets"]["prior_rule"], pins),
        )
        require(
            np.array_equal(raw, j["raw"]),
            "sameactualV126rawinitial everyrole no targetfilter",
        )
    require(np.array_equal(j["ids"], q["ids"]), "orderedwholeV126raw source identity")
    semantic_proof(reg, role, q["ids"], q["Z126"], out, r, pins)
    return q, raw, baseline


def graph(q, initial, anchor, out, r, role):
    nodes = core.collapse(q["Z126"][:, 78:110], q["groups"], q["ids"], initial)
    require(
        not set(nodes["groups"].tolist()) & set(anchor["groups"].tolist()),
        "samegenuinegroupsdisjointFIT knownanchors",
    )
    started = time.perf_counter()
    value = core.diffusion(
        SemanticEuclidean(), anchor["MHFD32"], nodes["Z32"], nodes["initial_raw"]
    )
    r["graph_seconds"][role] = time.perf_counter() - started
    r["files"][role + "_graph"] = save(
        out,
        r,
        role + "_actualfullMHFD32_uniformraw_softgraph_beforecut",
        dict(
            ids=q["ids"],
            stage1=q["stage1"],
            initial_raw_rows=initial,
            group32=nodes["Z32"],
            groups=nodes["groups"],
            representatives=nodes["representatives"],
            inverse=nodes["inverse"],
            score_rows=value["score"][nodes["inverse"]],
            **{k: v for k, v in value.items() if k != "manifest"},
        ),
    )
    r["files"][role + "_graph_manifest"] = files.write(
        out / (role + "_MHFD32_Euclidean_softgraph_manifest.json"),
        dict(
            value["manifest"],
            Euclidean_auxiliary32_not_paper126=True,
            initial_raw_rows=int(initial.sum()),
        ),
    )
    return value["score"][nodes["inverse"]]


def run(phase, ref, knownref=None):
    started = time.perf_counter()
    pins = {}
    reg = bind(ref, pins)
    control = None
    if phase == "current":
        control = files.js(knownref, pins)
        require(
            control["known_gate_passed"]
            and control["prediction_sealed"]
            and control["registry"] == ref,
            "newknownintegercontrol beforesemanticcurrent",
        )
    out = BASE / phase
    out.mkdir(parents=True, exist_ok=False)
    r = dict(
        policy=policy(),
        phase=phase,
        registry=ref,
        files={},
        graph_seconds={},
        NN_fit=0,
        NN_forward=0,
        metric_fit=0,
        R7_fit=0,
        units_fit=0,
        CAL_fit=0,
        DVC=0,
        current_numeric_decodes=0,
        prediction_sealed=False,
        truth_opened=False,
        status="FAIL_TARGET99",
        published=False,
        final_test_executed=False,
    )
    try:
        with threadpool_limits(limits=2):
            metric = old.cic.learned.PaperVariationalPairMetric.from_state(
                files.js(reg["assets"]["metric"], pins)["metric_payload"]
            )
            anchor = anchors(reg, metric, out, r, pins)
            r["files"]["metric"] = reg["assets"]["metric"]
            r["files"]["anchors"] = save(
                out,
                r,
                "allknown6000_actualcompleteMHFD32_original126_anchor0_beforegraphs",
                dict(**anchor, anchor_zero=np.zeros(6000)),
            )
            if phase == "known":
                q, initial, _ = role_input(reg, "CAL", out, r, pins)
                score = graph(q, initial, anchor, out, r, "CAL")
                labels = archive(
                    reg["assets"]["native_known"], ("CAL_ids", "CAL_y_coarse"), pins
                )
                require(
                    np.array_equal(labels["CAL_ids"], q["ids"]),
                    "actualknownCALall900identity",
                )
                rank = old.cic.dvc.calibrate_known_cutoffs(
                    score, labels["CAL_y_coarse"]
                )
                rule = {
                    k: dict(
                        v,
                        finite99_cutoff=v["cutoff"],
                        cutoff=max(0.5, v["cutoff"]) if v["enabled"] else None,
                        fixed_majority_floor=0.5,
                    )
                    for k, v in rank.items()
                }
                r["CAL_fit"] = 1
                r["files"]["rule"] = files.write(
                    out / "all900_trueknown_oncefinite99_MHFD32_fixedhalf_floor.json",
                    rule,
                )
                role = "CP"
            else:
                rule = files.js(control["files"]["rule"], pins)
                r["files"]["rule"] = control["files"]["rule"]
                r["known_report"] = knownref
                role = "current"
            q, initial, baseline = role_input(reg, role, out, r, pins)
            r["current_numeric_decodes"] = len(q["ids"]) if phase == "current" else 0
            score = graph(q, initial, anchor, out, r, role)
            semantic = old.cic.dvc.candidate_mask(score, q["stage1"], rule)
            nativecache = archive(
                reg["assets"][role + "_nativegraph"],
                ("ids", "stage1", "score_rows"),
                pins,
            )
            fullcache = archive(
                reg["assets"][role + "_fullgraph"],
                ("ids", "stage1", "score_rows"),
                pins,
            )
            require(
                np.array_equal(nativecache["ids"], q["ids"])
                and np.array_equal(fullcache["ids"], q["ids"])
                and np.array_equal(nativecache["stage1"], q["stage1"])
                and np.array_equal(fullcache["stage1"], q["stage1"]),
                "samewhole identity actualstage allthree frozenviews",
            )
            native = old.cic.dvc.candidate_mask(
                nativecache["score_rows"],
                q["stage1"],
                files.js(reg["assets"]["native_cut"], pins),
            )
            full = old.cic.dvc.candidate_mask(
                fullcache["score_rows"],
                q["stage1"],
                files.js(reg["assets"]["full_cut"], pins),
            )
            raw = (baseline != 2) & (native | full | semantic)
            density = raw | (baseline == 2)
            _, rep, inv = np.unique(q["groups"], return_index=True, return_inverse=True)
            gd = np.zeros(len(rep), bool)
            np.logical_or.at(gd, inv, density)
            routed = old.cic.dvc.route_current_candidates(
                metric, q["Z126"][rep], gd, seed=20261012
            )
            r["DVC"] = 1
            route = {k: v[inv] for k, v in routed.items() if k != "manifest"}
            accepted, secondary, noise = decisions(
                raw,
                route["first"],
                route["suspected"],
                route["retained"],
                native,
                full,
                semantic,
            )
            p = baseline.copy()
            p[accepted] = 2
            require(
                np.all(p[baseline == 2] == 2),
                "whole3124old2includingmistakes staysintact",
            )
            save(
                out,
                r,
                "matching126_anycompleteviewcue_plusold3124_DVC_beforetruth",
                route,
            )
            r["files"]["DVC_manifest"] = files.write(
                out / "matching126_anycompleteviewcue_plusold3124_DVC_manifest.json",
                dict(
                    routed["manifest"],
                    matching_metric_state_sha256=metric.state_sha256,
                    density_copies=0,
                ),
            )
            values = dict(
                ids=q["ids"],
                prediction=p,
                baseline=baseline,
                MHFD32_score=score,
                native78_score=nativecache["score_rows"],
                full126_score=fullcache["score_rows"],
                semantic_cue=semantic,
                native_cue=native,
                full_cue=full,
                raw=raw,
                density=density,
                accepted=accepted,
                secondary_review=secondary,
                primarynoise_review=noise,
                initial_raw=initial,
            )
            sealed = save(out, r, "complete_threeviewprediction_beforetruth", values)
            r["files"]["seal"] = files.write(
                out / "complete_threeviewprediction_seal_beforetruth.json",
                dict(
                    predictions=sealed,
                    registry=ref,
                    new_rule=r["files"]["rule"],
                    native_rule=reg["assets"]["native_cut"],
                    full_rule=reg["assets"]["full_cut"],
                    metric=reg["assets"]["metric"],
                    querytruth_used=False,
                ),
            )
            r["prediction_sealed"] = True
            if phase == "known":
                y = archive(reg["assets"]["known_truth"], ("CP_truth",), pins)[
                    "CP_truth"
                ]
            else:
                t = archive(reg["assets"]["truth"], ("ids", "truth"), pins)
                require(
                    np.array_equal(t["ids"], q["ids"]),
                    "truequery onlyafternewwholethreeviewseal",
                )
                y = t["truth"]
            r["truth_opened"] = True
            gates, delta, binary = old.helpers.metrics(y, baseline, p)
            r.update(
                execution_status="CIC1_completeMHFD32_threeview126_"
                + phase
                + "_completed",
                original_class_gates=gates,
                EffectDelta=delta,
                binary_attack_detection=binary,
                known_gate_passed=all(gates[n]["passed"] for n in ("BEN", "KNA")),
                candidate_accepted_for_retention=False,
                raw_by_trueclass={
                    n: int(((y == c) & raw).sum())
                    for c, n in enumerate(("BEN", "KNA", "UNA"))
                },
                accepted_by_trueclass={
                    n: int(((y == c) & accepted).sum())
                    for c, n in enumerate(("BEN", "KNA", "UNA"))
                },
                secondary_review_by_trueclass={
                    n: int(((y == c) & secondary).sum())
                    for c, n in enumerate(("BEN", "KNA", "UNA"))
                },
                primarynoise_review_by_trueclass={
                    n: int(((y == c) & noise).sum())
                    for c, n in enumerate(("BEN", "KNA", "UNA"))
                },
            )
            save(out, r, "postseal_complete_threeview_pair", dict(**values, truth=y))
            if phase == "current":
                r["all_three_integer99_passed"] = all(
                    v["passed"] for v in gates.values()
                )
                r["perfect_anyview_raw_UNA_ceiling"] = int(
                    ((y == 2) & ((baseline == 2) | raw)).sum()
                )
        for p, s in pins.items():
            require(
                files.item(p)["sha256"] == s,
                "frozenactualMHFD80/raw32/units/oldcuts/metrics/wholebaselines unchanged",
            )
    except Exception as error:
        r.update(
            execution_status="execution_failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        r.update(input_sha256=pins, wall_seconds=time.perf_counter() - started)
        files.write(out / "report.json", r)
    return files.item(out / "report.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("freeze", "known", "current"), required=True)
    for key in ("registry", "registry-sha256", "known-report", "known-sha256"):
        p.add_argument("--" + key)
    a = p.parse_args()
    ref = (
        None
        if a.registry is None
        else dict(path=str(Path(a.registry).resolve()), sha256=a.registry_sha256)
    )
    k = (
        None
        if a.known_report is None
        else dict(path=str(Path(a.known_report).resolve()), sha256=a.known_sha256)
    )
    print(json.dumps(freeze() if a.phase == "freeze" else run(a.phase, ref, k)))
