#!/usr/bin/env python3
"""Collect score-filtered policy rollouts for isolated RFT/CEM training.

The dataset is intentionally written as a standalone NPZ under this directory.
It never appends to or mutates the live self-play replay buffer.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from common import encode_state
from net import build_net
from suika_env import SuikaEnv

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _sample_action(probs, rng, temperature=1.0, epsilon=0.0):
    p = np.asarray(probs, dtype=np.float64)
    p = np.maximum(p, 1e-12)
    if temperature and abs(float(temperature) - 1.0) > 1e-6:
        p = np.power(p, 1.0 / float(temperature))
    p = p / p.sum()
    eps = float(np.clip(epsilon, 0.0, 1.0))
    if eps > 0.0:
        p = (1.0 - eps) * p + eps / len(p)
    p = p / p.sum()
    return int(rng.choice(len(p), p=p)), p.astype(np.float32)


def load_policy(cfg, ckpt, device):
    net = build_net(cfg).to(device)
    if ckpt:
        ck = torch.load(ckpt, map_location=device)
        net.load_state_dict(ck["net"])
    net.eval()
    return net


def collect_game(net, device, cfg, seed, temperature, epsilon, max_moves=0):
    rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
    env = SuikaEnv(seed=seed)
    state = env.get_state()
    pending = []
    move = 0
    while not state["game_over"]:
        if max_moves > 0 and move >= max_moves:
            return [], float(state["score"]), move, True
        vec = encode_state(
            state, cfg["K"], cfg["max_fruits"],
            boundary_features=cfg.get("boundary_features", False),
        )
        with torch.no_grad():
            probs, _ = net.infer_batch(vec[None, :], device)
        action, behavior_pi = _sample_action(probs[0], rng, temperature, epsilon)
        target = np.zeros((cfg["K"],), dtype=np.float32)
        target[action] = 1.0
        pending.append((vec, target, behavior_pi, action, float(state["score"])))
        state, _reward, done, _info = env.step_column(action, num_columns=cfg["K"])
        move += 1
        if done:
            break
    return pending, float(state["score"]), move, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config_rft_cem.yaml"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--games", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--elite-frac", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--weight-clip", type=float, default=None)
    ap.add_argument("--max-transitions", type=int, default=None)
    ap.add_argument("--max-moves", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    ckpt = _resolve(args.ckpt or cfg.get("rft_init_checkpoint", ""))
    out = _resolve(args.out or cfg.get("rft_data", "data/rft_elite_latest.npz"))
    games = int(args.games or cfg.get("rft_rollout_games", 50))
    seed0 = int(args.seed or cfg.get("rft_rollout_seed", 2026060101))
    temp = float(args.temperature if args.temperature is not None else cfg.get("rft_temperature", 1.35))
    eps = float(args.epsilon if args.epsilon is not None else cfg.get("rft_epsilon", 0.12))
    elite_frac = float(args.elite_frac if args.elite_frac is not None else cfg.get("rft_elite_frac", 0.25))
    beta = float(args.beta if args.beta is not None else cfg.get("rft_beta", 400.0))
    weight_clip = float(args.weight_clip if args.weight_clip is not None else cfg.get("rft_weight_clip", 6.0))
    max_transitions = int(args.max_transitions or cfg.get("rft_max_transitions", 100000))
    device = torch.device(args.device or cfg.get("device", "cpu"))

    net = load_policy(cfg, ckpt, device)
    all_games = []
    t0 = time.time()
    for i in range(games):
        seed = seed0 + i
        traj, score, moves, truncated = collect_game(
            net, device, cfg, seed, temp, eps, max_moves=args.max_moves)
        if not truncated and traj:
            all_games.append({"seed": seed, "score": score, "moves": moves, "traj": traj})
        print("[collect] game=%d/%d seed=%d score=%.0f moves=%d kept_games=%d" %
              (i + 1, games, seed, score, moves, len(all_games)), flush=True)

    if not all_games:
        raise SystemExit("no completed games collected")
    scores = np.array([g["score"] for g in all_games], dtype=np.float32)
    cutoff = float(np.quantile(scores, 1.0 - elite_frac))
    baseline = float(np.mean(scores))
    elite = [g for g in all_games if g["score"] >= cutoff]
    elite.sort(key=lambda g: g["score"], reverse=True)

    vecs, pis, behavior_pis, ws, game_ids, sample_scores, actions = [], [], [], [], [], [], []
    for gid, g in enumerate(elite):
        w = np.exp((float(g["score"]) - baseline) / max(beta, 1e-6))
        w = float(np.clip(w, 1.0 / max(weight_clip, 1.0), weight_clip))
        for vec, pi, bpi, action, _score_before in g["traj"]:
            vecs.append(vec)
            pis.append(pi)
            behavior_pis.append(bpi)
            ws.append(w)
            game_ids.append(gid)
            sample_scores.append(float(g["score"]))
            actions.append(int(action))
            if len(vecs) >= max_transitions:
                break
        if len(vecs) >= max_transitions:
            break

    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = {
        "generator": "rft_collect.py",
        "schema": "rft_cem_elite_v1",
        "checkpoint": ckpt,
        "games_requested": games,
        "games_completed": len(all_games),
        "elite_frac": elite_frac,
        "elite_cutoff": cutoff,
        "beta": beta,
        "weight_clip": weight_clip,
        "temperature": temp,
        "epsilon": eps,
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "elite_score_mean": float(np.mean([g["score"] for g in elite])),
        "elite_games": len(elite),
        "seconds": time.time() - t0,
    }
    tmp = out + ".tmp.npz"
    np.savez_compressed(
        tmp,
        vecs=np.asarray(vecs, dtype=np.float32),
        pis=np.asarray(pis, dtype=np.float32),
        behavior_pis=np.asarray(behavior_pis, dtype=np.float32),
        ws=np.asarray(ws, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        game_ids=np.asarray(game_ids, dtype=np.int64),
        sample_scores=np.asarray(sample_scores, dtype=np.float32),
        game_scores=scores,
        pos=np.array([len(vecs)], dtype=np.int64),
        size=np.array([len(vecs)], dtype=np.int64),
        meta=json.dumps(meta, ensure_ascii=False),
    )
    os.replace(tmp, out)
    print(json.dumps({"out": out, "samples": len(vecs), **meta}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
