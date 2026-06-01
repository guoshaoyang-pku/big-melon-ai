#!/usr/bin/env python3
"""Lightweight policy-RL data loop via top-trajectory imitation.

The script rolls out an existing policy checkpoint, keeps the best games, and
writes a standalone NPZ dataset with policy targets from the model's own action
distribution (or one-hot sampled actions).  It never merges into live replay;
use ``merge_replay.py`` explicitly after inspecting the output.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from common import SCORE_NORM, col_to_x, encode_state, input_dim
from net import build_net, pick_device
from suika_env import SuikaEnv


HERE = os.path.dirname(os.path.abspath(__file__))


def _policy(net, device, vec, temperature):
    with torch.no_grad():
        x = torch.as_tensor(vec[None, :], device=device)
        logits, _value = net(x)
        logits = logits / max(float(temperature), 1e-6)
        return F.softmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)


def rollout(cfg, net, device, seed, args):
    rng = np.random.default_rng(seed)
    env = SuikaEnv(seed=seed)
    state = env.get_state()
    pending = []
    move = 0
    while not state["game_over"]:
        if args.max_moves and move >= args.max_moves:
            return [], float(state["score"]), move, True
        vec = encode_state(
            state, cfg["K"], cfg["max_fruits"],
            boundary_features=cfg.get("boundary_features", False))
        pi = _policy(net, device, vec, args.temperature)
        if args.action_mode == "argmax":
            col = int(np.argmax(pi))
        else:
            col = int(rng.choice(int(cfg["K"]), p=pi / max(float(pi.sum()), 1e-12)))
        target = np.zeros_like(pi) if args.target_mode == "onehot" else pi.copy()
        if args.target_mode == "onehot":
            target[col] = 1.0
        pending.append((vec, target, float(state["score"])))
        state, _r, done, _info = env.step(col_to_x(col, cfg["K"]))
        move += 1
        if done:
            break
    final = float(state["score"])
    samples = []
    for vec, pi, before in pending:
        z = np.float32(0.0 if args.placeholder_z else (final - before) / SCORE_NORM)
        samples.append((vec, pi, z, np.float32(args.value_weight)))
    return samples, final, move, False


def _save(path, samples, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vecs = np.asarray([s[0] for s in samples], dtype=np.float32)
    pis = np.asarray([s[1] for s in samples], dtype=np.float32)
    zs = np.asarray([s[2] for s in samples], dtype=np.float32)
    ws = np.asarray([s[3] for s in samples], dtype=np.float32)
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp, vecs=vecs, pis=pis, zs=zs, ws=ws,
        pos=np.array([len(vecs)], dtype=np.int64),
        size=np.array([len(vecs)], dtype=np.int64),
        meta=json.dumps(meta, ensure_ascii=False),
    )
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "latest.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "policy_elite.npz"))
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--keep-frac", type=float, default=0.25)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--action-mode", choices=["sample", "argmax"], default="sample")
    ap.add_argument("--target-mode", choices=["policy", "onehot"], default="policy")
    ap.add_argument("--value-weight", type=float, default=0.0)
    ap.add_argument("--placeholder-z", action="store_true")
    ap.add_argument("--max-moves", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if os.path.basename(args.out) == "replay.npz":
        raise SystemExit("Refusing to write directly to replay.npz")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    in_dim = input_dim(cfg["max_fruits"], cfg.get("boundary_features", False))
    device = pick_device(args.device)
    net = build_net(cfg).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ck["net"])
    net.eval()

    rows = []
    t0 = time.time()
    for i in range(args.games):
        seed = int(args.seed + i)
        samples, score, moves, truncated = rollout(cfg, net, device, seed, args)
        rows.append({
            "seed": seed,
            "score": score,
            "moves": moves,
            "truncated": truncated,
            "samples": samples,
        })
        print("[policy_rl] game=%d/%d score=%.0f moves=%d truncated=%s" %
              (i + 1, args.games, score, moves, truncated), flush=True)

    completed = [r for r in rows if not r["truncated"] and r["samples"]]
    completed.sort(key=lambda r: r["score"], reverse=True)
    keep_n = max(1, int(np.ceil(len(completed) * float(args.keep_frac)))) if completed else 0
    kept = completed[:keep_n]
    samples = [s for r in kept for s in r["samples"]]
    meta = {
        "generator": "policy_rollout_imitation.py",
        "schema": "policy_elite_v1",
        "checkpoint": os.path.abspath(args.ckpt),
        "checkpoint_step": int(ck.get("step", 0)),
        "games": int(args.games),
        "kept_games": int(len(kept)),
        "keep_frac": float(args.keep_frac),
        "temperature": float(args.temperature),
        "action_mode": args.action_mode,
        "target_mode": args.target_mode,
        "value_weight": float(args.value_weight),
        "input_dim": int(in_dim),
        "seconds": time.time() - t0,
        "score_cutoff": float(kept[-1]["score"]) if kept else 0.0,
    }
    if samples:
        _save(args.out, samples, meta)
    print(json.dumps({
        "out": args.out,
        "samples": len(samples),
        "kept_games": len(kept),
        "score_mean_all": float(np.mean([r["score"] for r in rows])) if rows else 0.0,
        "score_mean_kept": float(np.mean([r["score"] for r in kept])) if kept else 0.0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
