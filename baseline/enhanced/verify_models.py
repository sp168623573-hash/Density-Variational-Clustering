"""Independently reload final models and reproduce complete query predictions."""

import json
import numpy as np
import torch
from torch import nn
import joblib
from threadpoolctl import threadpool_limits
from common import *
from models import CVAE, EVM, WSVM
import run_cvae_evt as author
from joint_cvae_calibration import auxiliary, merge

ROOT = BASE / "results/comparison_20261102_calibrated"


def load_cvae(folder, setting):
    ck = torch.load(folder / "model_final.pt", map_location="cpu", weights_only=True)
    model = CVAE(setting)
    model.n_classes = ck["n_classes"]
    model.benign_clusters = ck["benign_clusters"]
    model.c, model.r, _ = author.load_author_networks(setting, ck["feature_dim"])
    if model.n_classes != 9:
        n = model.n_classes
        dim = ck["feature_dim"]
        latent = model.c.l_z_x[-1].out_features // 2
        model.c.l_z_xy[0] = nn.Linear(dim + n, model.c.l_z_xy[0].out_features)
        model.c.l_y_xz[-2] = nn.Linear(model.c.l_y_xz[-2].in_features, n)
        model.r.l_z_xy[0] = nn.Linear(dim + n, model.r.l_z_xy[0].out_features)
        model.r.l_x_yz[0] = nn.Linear(latent + n, model.r.l_x_yz[0].out_features)
    model.c.load_state_dict(ck["classifier"])
    model.r.load_state_dict(ck["reconstructor"])
    model.c = model.c.to(model.device).eval()
    model.r = model.r.to(model.device).eval()
    model.scales = np.maximum([t["cutoff"] for t in ck["tails"]], 1e-8)
    return model


def main():
    torch.set_num_threads(4)
    checks = []
    for setting in SETTINGS:
        raw, ids = load_query(setting)
        for name in ["CVAE-EVT", "CVAE-EVT-BC", "CVAE-EVT-Joint", "EVM", "W-SVM"]:
            out = ROOT / setting / name
            saved = np.load(out / "predictions_all_epochs.npz")
            manifest = json.loads((out / "run_manifest.json").read_text())
            cal = json.loads((out / "known_calibration_all_epochs.json").read_text())
            source = out if name != "CVAE-EVT-Joint" else ROOT / setting / "CVAE-EVT"
            view = joblib.load(source / "preprocessing.joblib")
            x = view.transform(raw)
            benign = 8 if name == "CVAE-EVT-BC" else 1
            if name in ["CVAE-EVT", "CVAE-EVT-BC"]:
                model = load_cvae(out, setting)
                pred, score = model.scores(x)
            elif name == "CVAE-EVT-Joint":
                model = load_cvae(source, setting)
                pred, features = auxiliary(model, x)
                scales = np.maximum(
                    np.array(
                        [
                            [t["cutoff"] for t in row]
                            for row in json.loads(
                                (out / "auxiliary_GPD_tails.json").read_text()
                            )
                        ]
                    ),
                    1e-8,
                )
                _, recon = model.scores(x)
                score = merge(recon, features / scales[pred], manifest["mode"])
            elif name == "EVM":
                state = np.load(out / "model_final.npz")
                model = EVM(nfuse=int(state["nfuse"]))
                model.points = state["points"]
                model.params = state["params"]
                model.labels = state["labels"]
                pred, score = model.scores(x)
            else:
                model = WSVM(out / "replay_check")
                model.model = out / "model_final"
                pred, score = model.scores(x)
            assert np.array_equal(ids, saved["query_ids"]) and np.array_equal(
                pred, saved["query_fine"][-1]
            )
            assert np.allclose(
                score, saved["query_score"][-1], rtol=1e-6, atol=1e-6
            ), name
            got = probability_decision(pred, score, np.array(cal["cuts"][-1]), benign)
            assert np.array_equal(got, saved["query"][-1]), (setting, name)
            checks.append(
                dict(
                    setting=setting,
                    model=name,
                    rows=len(ids),
                    full_model_replay_matches=True,
                )
            )
            print("PASS replay", setting, name, len(ids), flush=True)
    save_json(
        ROOT / "MODEL_REPLAY_VERIFICATION.json",
        dict(status="PASS", checks=checks, query_truth_used=False),
    )


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
