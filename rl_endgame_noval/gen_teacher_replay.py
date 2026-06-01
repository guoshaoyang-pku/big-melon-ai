#!/usr/bin/env python3
"""Generate offline teacher replay from robust-search / heuristic rollouts.

The output is a standalone NPZ compatible with ReplayBuffer.load fields
(`vecs`, `pis`, `zs`, optional `ws`). It never modifies the live replay unless
the caller explicitly merges it with another tool.
"""
import argparse
import json
import os
import time

import numpy as np
import yaml

from bench_robust_search import RobustBeamSearchAgent
from common import encode_state, input_dim, SCORE_NORM
from suika_env import SuikaEnv
from teacher_agents import build_teacher_agent, normalised_score_value


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _policy_from_debug(debug, K, temperature, top_k=0, stochasticity=0.0):
    if debug.get("policy") is not None:
        pi = np.asarray(debug["policy"], dtype=np.float32)
        if pi.shape == (K,) and float(pi.sum()) > 0.0:
            return (pi / max(float(pi.sum()), 1e-12)).astype(np.float32)
    rows = debug.get("top_actions") or []
    pi = np.zeros((K,), dtype=np.float32)
    if not rows:
        pi[int(debug.get("best_col", K // 2))] = 1.0
        return pi
    rows = sorted(rows, key=lambda r: float(r.get("score", 0.0)), reverse=True)
    if top_k and int(top_k) > 0:
        rows = rows[:int(top_k)]
    cols = np.array([int(r["col"]) for r in rows], dtype=np.int64)
    scores = np.array([float(r.get("score", 0.0)) for r in rows], dtype=np.float64)
    if temperature <= 1e-6:
        pi[int(cols[int(np.argmax(scores))])] = 1.0
        return pi
    scores = scores - scores.max()
    probs = np.exp(scores / float(temperature))
    probs = probs / max(probs.sum(), 1e-12)
    for c, p in zip(cols, probs):
        pi[int(c)] += float(p)
    if stochasticity > 0.0:
        eps = float(np.clip(stochasticity, 0.0, 1.0))
        pi = (1.0 - eps) * pi + eps / float(K)
    pi = pi / max(float(pi.sum()), 1e-12)
    return pi.astype(np.float32)


def _save_npz(path, samples, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vecs = np.asarray([s[0] for s in samples], dtype=np.float32)
    pis = np.asarray([s[1] for s in samples], dtype=np.float32)
    zs = np.asarray([s[2] for s in samples], dtype=np.float32)
    ws = np.asarray([s[3] for s in samples], dtype=np.float32)
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        vecs=vecs,
        pis=pis,
        zs=zs,
        ws=ws,
        pos=np.array([len(vecs)], dtype=np.int64),
        size=np.array([len(vecs)], dtype=np.int64),
        meta=json.dumps(meta, ensure_ascii=False),
    )
    os.replace(tmp, path)


def play_teacher_game(cfg, seed, args):
    env = SuikaEnv(seed=seed)
    state = env.get_state()
    if args.teacher == "robust":
        agent = RobustBeamSearchAgent(
            K=cfg["K"],
            depth=args.depth,
            beam_size=args.beam_size,
            samples_per_action=args.samples_per_action,
            branch_width=args.branch_width,
            risk_lambda=args.risk_lambda,
            quantile=args.quantile,
            risk_mode=args.risk_mode,
            seed=seed,
            fast_steps=args.fast_steps,
            fast_settle_v=args.fast_settle_v,
            fast_check_every=args.fast_check_every,
            iterations=args.iterations,
        )
    else:
        agent = build_teacher_agent(args.teacher, cfg, args, seed=seed)
    pending = []
    move = 0
    t0 = time.time()
    while not state["game_over"]:
        if args.max_moves > 0 and move >= args.max_moves:
            return [], float(state["score"]), move, True
        if args.max_seconds > 0 and (time.time() - t0) >= args.max_seconds:
            return [], float(state["score"]), move, True
        vec = encode_state(state, cfg["K"], cfg["max_fruits"],
                           boundary_features=cfg.get("boundary_features", False))
        x, debug = agent.decide(state)
        pi = _policy_from_debug(
            debug, cfg["K"], args.policy_temperature,
            top_k=args.policy_top_k,
            stochasticity=args.policy_stochasticity)
        pending.append((vec, pi, float(state["score"])))
        state, _reward, done, _info = env.step(x)
        move += 1
        if done:
            break
    final_score = float(state["score"])
    out = []
    for vec, pi, score_before in pending:
        z = 0.0 if args.placeholder_z else normalised_score_value(
            final_score, score_before)
        out.append((vec, pi, np.float32(z), np.float32(args.value_weight)))
    return out, final_score, move, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "teacher_replay.npz"))
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--teacher", default="robust",
                    choices=["robust", "heuristic_plus", "two_step",
                             "robust_v2", "ensemble"],
                    help="teacher used to label the soft policy")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--beam-size", type=int, default=8)
    ap.add_argument("--samples-per-action", type=int, default=2)
    ap.add_argument("--branch-width", type=int, default=4)
    ap.add_argument("--risk-mode", default="mean_std")
    ap.add_argument("--risk-lambda", type=float, default=0.5)
    ap.add_argument("--quantile", type=float, default=0.2)
    ap.add_argument("--policy-temperature", type=float, default=100.0,
                    help="softmax temperature over teacher scores")
    ap.add_argument("--policy-top-k", type=int, default=8,
                    help="keep only top-K teacher actions before softmax; 0 keeps all")
    ap.add_argument("--policy-stochasticity", type=float, default=0.0,
                    help="mix this much uniform mass into teacher policy")
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--policy-only", action="store_true",
                    help="alias for --value-weight 0; z is saved but ignored by training")
    ap.add_argument("--placeholder-z", action="store_true",
                    help="store z=0 placeholders for pure policy datasets")
    ap.add_argument("--lookahead-steps", type=int, default=220)
    ap.add_argument("--max-candidates", type=int, default=28)
    ap.add_argument("--chance-mode", default="enumerate",
                    choices=["enumerate", "sample"])
    ap.add_argument("--ensemble-include-robust", action="store_true")
    ap.add_argument("--fast-steps", type=int, default=60)
    ap.add_argument("--fast-settle-v", type=float, default=4.0)
    ap.add_argument("--fast-check-every", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--max-moves", type=int, default=0,
                    help="debug cap only; capped games are discarded")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="debug cap only; capped games are discarded")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.policy_only:
        args.value_weight = 0.0

    if os.path.basename(args.out) == "replay.npz":
        raise SystemExit("Refusing to write directly to replay.npz")

    all_samples = []
    scores, moves, discarded = [], [], 0
    t0 = time.time()
    for i in range(args.games):
        seed = int(args.seed + i)
        samples, score, n_moves, was_truncated = play_teacher_game(cfg, seed, args)
        if was_truncated:
            discarded += 1
        else:
            all_samples.extend(samples)
            scores.append(score)
            moves.append(n_moves)
        if (i + 1) % max(1, args.save_every) == 0 and all_samples:
            _save_npz(args.out, all_samples, {
                "generator": "gen_teacher_replay.py",
                "schema": "policy_teacher_v2",
                "teacher": args.teacher,
                "games_requested": args.games,
                "games_completed": len(scores),
                "discarded": discarded,
                "depth": args.depth,
                "beam_size": args.beam_size,
                "samples_per_action": args.samples_per_action,
                "branch_width": args.branch_width,
                "risk_mode": args.risk_mode,
                "risk_lambda": args.risk_lambda,
                "policy_top_k": args.policy_top_k,
                "policy_stochasticity": args.policy_stochasticity,
                "value_weight": args.value_weight,
                "boundary_features": bool(cfg.get("boundary_features", False)),
                "K": cfg["K"],
                "max_fruits": cfg["max_fruits"],
            })
        print("[teacher] game=%d/%d score=%.0f moves=%d samples=%d discarded=%d" %
              (i + 1, args.games, score, n_moves, len(all_samples), discarded),
              flush=True)

    if all_samples:
        _save_npz(args.out, all_samples, {
            "generator": "gen_teacher_replay.py",
            "schema": "policy_teacher_v2",
            "teacher": args.teacher,
            "games_requested": args.games,
            "games_completed": len(scores),
            "discarded": discarded,
            "seconds": time.time() - t0,
            "score_mean": float(np.mean(scores)) if scores else 0.0,
            "score_median": float(np.median(scores)) if scores else 0.0,
            "score_max": float(np.max(scores)) if scores else 0.0,
            "move_mean": float(np.mean(moves)) if moves else 0.0,
            "depth": args.depth,
            "beam_size": args.beam_size,
            "samples_per_action": args.samples_per_action,
            "branch_width": args.branch_width,
            "risk_mode": args.risk_mode,
            "risk_lambda": args.risk_lambda,
            "policy_temperature": args.policy_temperature,
            "policy_top_k": args.policy_top_k,
            "policy_stochasticity": args.policy_stochasticity,
            "value_weight": args.value_weight,
            "placeholder_z": bool(args.placeholder_z),
            "boundary_features": bool(cfg.get("boundary_features", False)),
            "physics": {
                "fast_steps": args.fast_steps,
                "fast_settle_v": args.fast_settle_v,
                "fast_check_every": args.fast_check_every,
                "iterations": args.iterations,
            },
            "K": cfg["K"],
            "max_fruits": cfg["max_fruits"],
        })
    print(json.dumps({
        "out": args.out,
        "samples": len(all_samples),
        "games_completed": len(scores),
        "discarded": discarded,
        "score_mean": float(np.mean(scores)) if scores else 0.0,
        "score_median": float(np.median(scores)) if scores else 0.0,
        "score_max": float(np.max(scores)) if scores else 0.0,
        "seconds": time.time() - t0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
