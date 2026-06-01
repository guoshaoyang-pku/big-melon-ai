#!/usr/bin/env python3
"""Fixed-seed full-game eval for RFT/CEM policy checkpoints."""
import argparse
import json
import os

import numpy as np
import torch
import yaml

from common import encode_state
from net import build_net
from suika_env import SuikaEnv
from eval import eval_random, eval_heuristic

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def eval_policy(cfg, ckpt, seeds, device, temperature=0.0, epsilon=0.0, max_moves=0):
    net = build_net(cfg).to(device)
    c = torch.load(ckpt, map_location=device)
    net.load_state_dict(c["net"])
    net.eval()
    scores, moves, maxf = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(int(seed) ^ 0xC0FFEE)
        env = SuikaEnv(seed=int(seed))
        state = env.get_state()
        move = 0
        while not state["game_over"]:
            if max_moves > 0 and move >= max_moves:
                break
            vec = encode_state(state, cfg["K"], cfg["max_fruits"],
                               boundary_features=cfg.get("boundary_features", False))
            with torch.no_grad():
                probs, _ = net.infer_batch(vec[None, :], device)
            p = np.asarray(probs[0], dtype=np.float64)
            if temperature and temperature > 0.0:
                p = np.maximum(p, 1e-12) ** (1.0 / float(temperature))
                p = p / p.sum()
                if epsilon > 0.0:
                    p = (1.0 - epsilon) * p + epsilon / len(p)
                action = int(rng.choice(len(p), p=p / p.sum()))
            else:
                action = int(np.argmax(p))
            state, _reward, done, _info = env.step_column(action, num_columns=cfg["K"])
            move += 1
            if done:
                break
        scores.append(float(state["score"]))
        moves.append(move)
        maxf.append(max((f["type"] for f in state["fruits"]), default=-1))
        print("[eval_policy] seed=%s score=%.0f moves=%d" % (seed, scores[-1], move), flush=True)
    return {"mean": float(np.mean(scores)), "median": float(np.median(scores)),
            "min": float(np.min(scores)), "max": float(np.max(scores)),
            "scores": scores, "moves": moves, "max_fruit": int(max(maxf))}


def summarise(scores):
    return {"mean": float(np.mean(scores)), "median": float(np.median(scores)),
            "min": float(np.min(scores)), "max": float(np.max(scores)),
            "scores": [float(x) for x in scores]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config_rft_cem.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "latest.pt"))
    ap.add_argument("--seeds", default=None, help="comma separated seeds")
    ap.add_argument("--device", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--max-moves", type=int, default=None)
    ap.add_argument("--include-random", action="store_true")
    ap.add_argument("--include-heuristic", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "logs", "rft_eval_latest.json"))
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else list(cfg.get("rft_eval_seeds", range(10)))
    max_moves = int(args.max_moves if args.max_moves is not None else cfg.get("rft_eval_max_moves", 0))
    device = torch.device(args.device or cfg.get("device", "cpu"))
    result = {
        "checkpoint": _resolve(args.ckpt),
        "seeds": seeds,
        "policy": eval_policy(cfg, _resolve(args.ckpt), seeds, device,
                              temperature=args.temperature, epsilon=args.epsilon,
                              max_moves=max_moves),
    }
    if args.include_random:
        _mean, _maxf, scores = eval_random(seeds, K=cfg["K"], max_moves=max_moves)
        result["random"] = summarise(scores)
    if args.include_heuristic:
        _mean, _maxf, scores = eval_heuristic(seeds, K=cfg["K"], max_moves=max_moves)
        result["heuristic"] = summarise(scores)
    os.makedirs(os.path.dirname(_resolve(args.out)), exist_ok=True)
    with open(_resolve(args.out), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
