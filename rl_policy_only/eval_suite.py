#!/usr/bin/env python3
"""Fixed-seed evaluation suite for Suika agents.

This script is intentionally evaluation-only: it does not touch replay buffers,
training logs, or checkpoints except for reading the selected checkpoint.
"""
import argparse
import csv
import json
import os
import time
from statistics import mean, median

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from common import encode_state, col_to_x
from net import pick_device
import eval as ev


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _parse_seeds(text):
    if "," in text:
        return [int(x) for x in text.split(",") if x.strip()]
    if ":" in text:
        a, b = text.split(":", 1)
        return list(range(int(a), int(b)))
    n = int(text)
    return list(range(n))


def _summary(name, scores, maxfruit=None, seconds=None, moves=None, extra=None):
    arr = np.asarray(scores, dtype=np.float64)
    moves_arr = np.asarray(moves or [], dtype=np.float64)
    out = {
        "name": name,
        "games": int(len(arr)),
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "median": float(np.median(arr)) if len(arr) else 0.0,
        "p25": float(np.percentile(arr, 25)) if len(arr) else 0.0,
        "max": float(arr.max()) if len(arr) else 0.0,
        "min": float(arr.min()) if len(arr) else 0.0,
        "std": float(arr.std(ddof=0)) if len(arr) else 0.0,
        "maxfruit": int(maxfruit) if maxfruit is not None else None,
        "seconds": float(seconds) if seconds is not None else None,
        "moves": float(moves_arr.mean()) if len(moves_arr) else None,
        "sec_per_move": (
            float(seconds) / max(1.0, float(moves_arr.sum()))
            if seconds is not None and len(moves_arr) else None
        ),
    }
    if extra:
        out.update(extra)
    return out


def _run_random(cfg, seeds):
    t0 = time.time()
    from suika_env import SuikaEnv

    scores, maxf, moves = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        env = SuikaEnv(seed=seed)
        state = env.get_state()
        move = 0
        while not state["game_over"]:
            if cfg.get("_eval_max_moves", 0) and move >= cfg["_eval_max_moves"]:
                break
            action = int(rng.integers(0, cfg["K"]))
            state, _r, done, _ = env.step_column(action, num_columns=cfg["K"])
            move += 1
            if done:
                break
        scores.append(float(state["score"]))
        maxf.append(max((f["type"] for f in state["fruits"]), default=-1))
        moves.append(move)
    return _summary("random", scores, max(maxf) if maxf else -1,
                    time.time() - t0, moves=moves)


def _run_heuristic(cfg, seeds):
    from ai_agent import HeuristicLookaheadAgent
    from suika_env import SuikaEnv

    t0 = time.time()
    scores, maxf, moves = [], [], []
    for seed in seeds:
        env = SuikaEnv(seed=seed)
        state = env.get_state()
        agent = HeuristicLookaheadAgent(num_columns=cfg["K"], seed=seed)
        move = 0
        while not state["game_over"]:
            if cfg.get("_eval_max_moves", 0) and move >= cfg["_eval_max_moves"]:
                break
            x = agent.decide(state)
            state, _r, done, _ = env.step(x)
            move += 1
            if done:
                break
        scores.append(float(state["score"]))
        maxf.append(max((f["type"] for f in state["fruits"]), default=-1))
        moves.append(move)
    return _summary("heuristic_1step", scores, max(maxf) if maxf else -1,
                    time.time() - t0, moves=moves)


def _run_uniform_mcts(cfg, seeds, sims):
    t0 = time.time()
    avg, mf, scores = ev.eval_uniform_mcts(cfg, seeds, num_simulations=sims,
                                           max_moves=cfg.get("_eval_max_moves", 0))
    return _summary("uniform_mcts", scores, mf, time.time() - t0,
                    {"simulations": int(sims)})


def _run_net_mcts(cfg, seeds, ckpt, sims, device_name):
    device = pick_device(device_name)
    net, step = ev.load_eval_net(cfg, ckpt, device)
    t0 = time.time()
    avg, mf, scores = ev.eval_net(net, device, cfg, seeds,
                                  num_simulations=sims,
                                  max_moves=cfg.get("_eval_max_moves", 0))
    return _summary("net_mcts", scores, mf, time.time() - t0,
                    {"simulations": int(sims), "checkpoint": ckpt,
                     "checkpoint_step": int(step), "device": str(device)})


def _run_robust(cfg, seeds, args):
    from bench_robust_search import RobustBeamSearchAgent
    from suika_env import SuikaEnv

    scores, maxf, steps = [], [], []
    t0 = time.time()
    for seed in seeds:
        env = SuikaEnv(seed=seed)
        state = env.get_state()
        agent = RobustBeamSearchAgent(
            K=cfg["K"],
            depth=args.robust_depth,
            beam_size=args.robust_beam,
            samples_per_action=args.robust_samples,
            branch_width=args.robust_branch,
            risk_lambda=args.robust_lambda,
            risk_mode=args.robust_mode,
            seed=seed,
            fast_steps=args.robust_fast_steps,
            fast_check_every=args.robust_check_every,
            iterations=args.robust_iterations,
        )
        move = 0
        while not state["game_over"]:
            if cfg.get("_eval_max_moves", 0) and move >= cfg["_eval_max_moves"]:
                break
            x, _dbg = agent.decide(state)
            state, _r, done, _info = env.step(x)
            move += 1
            if done:
                break
        scores.append(float(state["score"]))
        maxf.append(max((f["type"] for f in state["fruits"]), default=-1))
        steps.append(move)
    return _summary(
        "robust_beam", scores, max(maxf) if maxf else -1, time.time() - t0,
        {
            "depth": args.robust_depth,
            "beam": args.robust_beam,
            "samples_per_action": args.robust_samples,
            "branch_width": args.robust_branch,
            "risk_mode": args.robust_mode,
            "risk_lambda": args.robust_lambda,
            "mean_steps": float(mean(steps)) if steps else 0.0,
        },
    )


class PolicyOnlyAgent:
    def __init__(self, net, device, cfg):
        self.net = net
        self.device = device
        self.cfg = cfg
        self.K = int(cfg["K"])

    def policy(self, state):
        vec = encode_state(
            state, self.K, self.cfg["max_fruits"],
            boundary_features=self.cfg.get("boundary_features", False))
        with torch.no_grad():
            x = torch.as_tensor(vec[None, :], device=self.device)
            logits, _value = self.net(x)
            return F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    def decide(self, state):
        p = self.policy(state)
        col = int(np.argmax(p))
        return col_to_x(col, self.K), {
            "best_col": col,
            "policy": p.astype(np.float32).tolist(),
            "top_actions": [{"col": int(c), "score": float(p[c])}
                            for c in np.argsort(p)[::-1][:5]],
        }


def _run_agent_loop(name, cfg, seeds, agent_factory):
    from suika_env import SuikaEnv

    scores, maxf, moves = [], [], []
    t0 = time.time()
    for seed in seeds:
        env = SuikaEnv(seed=seed)
        state = env.get_state()
        agent = agent_factory(seed)
        move = 0
        while not state["game_over"]:
            if cfg.get("_eval_max_moves", 0) and move >= cfg["_eval_max_moves"]:
                break
            x, _dbg = agent.decide(state)
            state, _r, done, _info = env.step(x)
            move += 1
            if done:
                break
        scores.append(float(state["score"]))
        maxf.append(max((f["type"] for f in state["fruits"]), default=-1))
        moves.append(move)
    return _summary(name, scores, max(maxf) if maxf else -1,
                    time.time() - t0, moves=moves)


def _policy_stats(policy_rows):
    if not policy_rows:
        return {}
    probs = np.asarray(policy_rows, dtype=np.float64)
    K = probs.shape[1]
    ent = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    center = np.arange(K)
    center_mask = (center >= K // 2 - 1) & (center <= K // 2 + 1)
    edge_mask = (center <= 1) | (center >= K - 2)
    return {
        "policy_entropy": float(ent.mean()),
        "policy_top1_counts": [int(x) for x in np.bincount(
            probs.argmax(axis=1), minlength=K)],
        "center_mass": float(probs[:, center_mask].sum(axis=1).mean()),
        "edge_mass": float(probs[:, edge_mask].sum(axis=1).mean()),
    }


def _run_stress(name, cfg, policy_fn, seed=0):
    from stress_states import stress_state_pairs

    rows, mirror_diffs, near_wall = [], [], []
    for state, mirrored in stress_state_pairs(seed):
        p = np.asarray(policy_fn(state), dtype=np.float64)
        pm = np.asarray(policy_fn(mirrored), dtype=np.float64)
        p = p / max(float(p.sum()), 1e-12)
        pm = pm / max(float(pm.sum()), 1e-12)
        rows.append(p)
        mirror_diffs.append(float(np.abs(p - pm[::-1]).sum()))
        if "wall" in state["name"] or "corner" in state["name"]:
            K = int(cfg["K"])
            near_wall.append(float(p[[0, 1, K - 2, K - 1]].sum()))
    out = {"name": name + "_stress", "stress_states": len(rows)}
    out.update(_policy_stats(rows))
    out["mirror_l1"] = float(np.mean(mirror_diffs)) if mirror_diffs else 0.0
    out["near_wall_edge_mass"] = float(np.mean(near_wall)) if near_wall else 0.0
    return out


def _write_csv(rows, path):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "latest.pt"))
    ap.add_argument("--seeds", default="0:20",
                    help="'N' for range(N), 'a:b' for range(a,b), or csv list")
    ap.add_argument("--agents", default="random,heuristic,net,uniform",
                    help=("comma list: random,heuristic,heuristic_plus,two_step,"
                          "robust,robust_v2,net,policy,uniform"))
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--device", default="cpu",
                    help="cpu is usually faster for small eval batches")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--max-moves", type=int, default=0,
                    help="debug cap; 0 plays to game_over")
    ap.add_argument("--stress", action="store_true",
                    help="also run synthetic boundary/mirror policy diagnostics")
    ap.add_argument("--policy-temperature", type=float, default=100.0)
    ap.add_argument("--policy-top-k", type=int, default=8)
    ap.add_argument("--policy-stochasticity", type=float, default=0.0)
    ap.add_argument("--lookahead-steps", type=int, default=220)
    ap.add_argument("--max-candidates", type=int, default=28)
    ap.add_argument("--chance-mode", default="enumerate",
                    choices=["enumerate", "sample"])
    # robust beam knobs
    ap.add_argument("--robust-depth", type=int, default=4)
    ap.add_argument("--robust-beam", type=int, default=8)
    ap.add_argument("--robust-samples", type=int, default=2)
    ap.add_argument("--robust-branch", type=int, default=4)
    ap.add_argument("--robust-mode", default="mean_std")
    ap.add_argument("--robust-lambda", type=float, default=0.5)
    ap.add_argument("--robust-fast-steps", type=int, default=50)
    ap.add_argument("--robust-check-every", type=int, default=5)
    ap.add_argument("--robust-iterations", type=int, default=5)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seeds = _parse_seeds(args.seeds)
    cfg["_eval_max_moves"] = int(args.max_moves)
    sims = int(args.sims or cfg.get("eval_simulations", cfg["num_simulations"]))
    agents = {a.strip() for a in args.agents.split(",") if a.strip()}

    rows = []
    stress_rows = []
    if "random" in agents:
        rows.append(_run_random(cfg, seeds))
    if "heuristic" in agents:
        rows.append(_run_heuristic(cfg, seeds))
    if "uniform" in agents:
        rows.append(_run_uniform_mcts(cfg, seeds, sims))
    if "net" in agents:
        rows.append(_run_net_mcts(cfg, seeds, args.ckpt, sims, args.device))
        if args.stress:
            device = pick_device(args.device)
            net, _step = ev.load_eval_net(cfg, args.ckpt, device)
            pa = PolicyOnlyAgent(net, device, cfg)
            stress_rows.append(_run_stress("policy", cfg, pa.policy))
    if "policy" in agents:
        device = pick_device(args.device)
        net, step = ev.load_eval_net(cfg, args.ckpt, device)
        pa = PolicyOnlyAgent(net, device, cfg)
        rows.append(_run_agent_loop(
            "policy_only", cfg, seeds,
            lambda _seed: pa))
        rows[-1].update({"checkpoint": args.ckpt, "checkpoint_step": int(step),
                         "device": str(device)})
        if args.stress:
            stress_rows.append(_run_stress("policy", cfg, pa.policy))
    if "robust" in agents:
        rows.append(_run_robust(cfg, seeds, args))
    if "heuristic_plus" in agents or "two_step" in agents or "robust_v2" in agents:
        from teacher_agents import build_teacher_agent

        for name in ("heuristic_plus", "two_step", "robust_v2"):
            if name not in agents:
                continue
            rows.append(_run_agent_loop(
                name, cfg, seeds,
                lambda seed, n=name: build_teacher_agent(n, cfg, args, seed=seed)))
            if args.stress:
                agent = build_teacher_agent(name, cfg, args, seed=0)
                stress_rows.append(_run_stress(
                    name, cfg,
                    lambda s, a=agent: np.asarray(a.decide(s)[1].get("policy"))))

    payload = {"seeds": seeds, "rows": rows, "stress": stress_rows}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    _write_csv(rows, args.out_csv)


if __name__ == "__main__":
    main()
