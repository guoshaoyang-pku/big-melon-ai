#!/usr/bin/env python3
"""Policy/replay diagnostics for Suika AlphaZero checkpoints."""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from net import build_net, pick_device
from common import encode_state, input_dim


HERE = os.path.dirname(os.path.abspath(__file__))


def _load_replay(path, in_dim, K, limit):
    d = np.load(path, allow_pickle=False)
    n = int(d["size"][0]) if "size" in d.files else len(d["vecs"])
    vecs = d["vecs"][:n].astype(np.float32)
    pis = d["pis"][:n].astype(np.float32)
    zs = d["zs"][:n].astype(np.float32)
    if vecs.shape[1] != in_dim:
        raise ValueError("replay vec dim %d != expected %d" % (vecs.shape[1], in_dim))
    if pis.shape[1] != K:
        raise ValueError("replay policy dim %d != expected %d" % (pis.shape[1], K))
    if limit and len(vecs) > limit:
        idx = np.linspace(0, len(vecs) - 1, num=limit, dtype=np.int64)
        vecs, pis, zs = vecs[idx], pis[idx], zs[idx]
    return vecs, pis, zs


def _entropy(p):
    p = np.asarray(p, dtype=np.float64)
    return -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)


def _center_edge_mass(p):
    K = p.shape[1]
    center = np.arange(K)
    center_lo = K // 2 - 1
    center_hi = K // 2 + 1
    center_mask = (center >= center_lo) & (center <= center_hi)
    edge_mask = (center <= 1) | (center >= K - 2)
    return float(p[:, center_mask].sum(axis=1).mean()), float(p[:, edge_mask].sum(axis=1).mean())


def _stress_metrics(net, device, cfg, seed):
    from stress_states import stress_state_pairs

    K = int(cfg["K"])
    probs, mirror_l1, near_wall = [], [], []
    with torch.no_grad():
        for state, mirrored in stress_state_pairs(seed):
            vec = encode_state(
                state, K, cfg["max_fruits"],
                boundary_features=cfg.get("boundary_features", False))
            mvec = encode_state(
                mirrored, K, cfg["max_fruits"],
                boundary_features=cfg.get("boundary_features", False))
            x = torch.as_tensor(np.stack([vec, mvec]), device=device)
            p = F.softmax(net(x)[0], dim=-1).cpu().numpy()
            probs.append(p[0])
            mirror_l1.append(float(np.abs(p[0] - p[1][::-1]).sum()))
            if "wall" in state["name"] or "corner" in state["name"]:
                near_wall.append(float(p[0][[0, 1, K - 2, K - 1]].sum()))
    probs = np.asarray(probs, dtype=np.float64)
    center_mass, edge_mass = _center_edge_mass(probs)
    return {
        "stress_states": int(len(probs)),
        "stress_policy_entropy": float(_entropy(probs).mean()),
        "stress_center_mass": center_mass,
        "stress_edge_mass": edge_mass,
        "mirror_consistency_l1": float(np.mean(mirror_l1)) if mirror_l1 else 0.0,
        "near_wall_edge_mass": float(np.mean(near_wall)) if near_wall else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "latest.pt"))
    ap.add_argument("--replay", default=os.path.join(HERE, "data", "replay.npz"))
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    ap.add_argument("--stress", action="store_true",
                    help="add synthetic near-wall/corner/danger mirror diagnostics")
    ap.add_argument("--stress-seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    K = int(cfg["K"])
    in_dim = input_dim(cfg["max_fruits"], cfg.get("boundary_features", False))
    vecs, teacher_pi, zs = _load_replay(args.replay, in_dim, K, args.limit)

    device = pick_device(args.device)
    net = build_net(cfg).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ck["net"])
    net.eval()

    probs, values = [], []
    with torch.no_grad():
        for i in range(0, len(vecs), args.batch_size):
            x = torch.as_tensor(vecs[i:i + args.batch_size], device=device)
            logits, value = net(x)
            probs.append(F.softmax(logits, dim=-1).cpu().numpy())
            values.append(value.cpu().numpy())
    model_pi = np.concatenate(probs, axis=0)
    values = np.concatenate(values, axis=0)

    kl = (teacher_pi * (np.log(np.clip(teacher_pi, 1e-12, 1.0))
                       - np.log(np.clip(model_pi, 1e-12, 1.0)))).sum(axis=1)
    ent = _entropy(model_pi)
    t_ent = _entropy(teacher_pi)
    center_mass, edge_mass = _center_edge_mass(model_pi)
    t_center_mass, t_edge_mass = _center_edge_mass(teacher_pi)
    value_err = values - zs
    out = {
        "checkpoint": args.ckpt,
        "checkpoint_step": int(ck.get("step", 0)),
        "replay": args.replay,
        "samples": int(len(vecs)),
        "policy_entropy": float(ent.mean()),
        "policy_perplexity": float(np.exp(ent.mean())),
        "teacher_entropy": float(t_ent.mean()),
        "teacher_perplexity": float(np.exp(t_ent.mean())),
        "root_kl_teacher_to_model": float(kl.mean()),
        "center_mass": center_mass,
        "edge_mass": edge_mass,
        "teacher_center_mass": t_center_mass,
        "teacher_edge_mass": t_edge_mass,
        "value_mae": float(np.abs(value_err).mean()),
        "value_rmse": float(np.sqrt((value_err ** 2).mean())),
        "value_bias": float(value_err.mean()),
        "top1_counts": [int(x) for x in np.bincount(model_pi.argmax(axis=1), minlength=K)],
        "teacher_top1_counts": [int(x) for x in np.bincount(teacher_pi.argmax(axis=1), minlength=K)],
    }
    if args.stress:
        out.update(_stress_metrics(net, device, cfg, args.stress_seed))
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
