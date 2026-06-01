#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight, decoupled overnight eval tracker for the policy-only run.

Every --interval seconds it READS checkpoints/latest.pt (never writes replay,
never touches training), plays --games fixed-seed FULL games from an empty
board (game_over), and appends mean/median/max/moves to logs/eval_track.csv.
When mean improves it copies latest.pt -> checkpoints/best.pt (atomic). Runs at
low priority on CPU so it cannot steal cycles from the MPS learner.
Random baseline reference on these seeds is ~1156.
"""
import argparse, csv, os, shutil, time, datetime
import numpy as np
import torch
import yaml

from net import build_net
from mcts import MCTS
from suika_env import SuikaEnv

HERE = os.path.dirname(os.path.abspath(__file__))


def _max_fruit(state):
    return max((f["type"] for f in state["fruits"]), default=-1)


def play_one(net, device, cfg, seed, sims, max_moves=1000, game_timeout=180.0):
    K = cfg["K"]
    env = SuikaEnv(seed=seed)
    state = env.get_state()
    mcts = MCTS(net, device, K=K, num_simulations=sims,
                c_puct=cfg["c_puct"], max_fruits=cfg["max_fruits"],
                fast_model=True, seed=seed,
                eval_batch=cfg.get("eval_batch", 16),
                root_forced_visits=cfg.get("root_forced_visits", 0),
                boundary_features=cfg.get("boundary_features", False),
                use_value_net=cfg.get("use_value_net", False),
                leaf_value_mode=cfg.get("leaf_value_mode", "heuristic"),
                leaf_safety_w=cfg.get("leaf_safety_w", 0.15),
                leaf_fill_w=cfg.get("leaf_fill_w", 0.05),
                leaf_value_clip=cfg.get("leaf_value_clip", 0.5))
    moves = 0
    t0 = time.time()
    while not state["game_over"]:
        # safety guards: a collapsed policy can yield games that never end,
        # which previously hung the monitor indefinitely (e.g. after ~03:46).
        if (max_moves and moves >= max_moves) or \
           (game_timeout and (time.time() - t0) > game_timeout):
            print(f"[monitor] WARN seed={seed} safety-cap hit "
                  f"(moves={moves}, {time.time()-t0:.0f}s); ending game early",
                  flush=True)
            break
        action, _ = mcts.policy(state, temperature=0.0, add_noise=False)
        state, _r, done, _ = env.step_column(action, num_columns=K)
        moves += 1
        if done:
            break
    return float(state["score"]), int(_max_fruit(state)), moves


def evaluate(cfg, ckpt, seeds, sims, device, max_moves=1000, game_timeout=180.0):
    net = build_net(cfg).to(device)
    ck = torch.load(ckpt, map_location=device)
    net.load_state_dict(ck["net"]); net.eval()
    step = int(ck.get("step", 0))
    scores, maxf, moves = [], [], []
    for sd in seeds:
        try:
            sc, mf, mv = play_one(net, device, cfg, sd, sims,
                                  max_moves=max_moves, game_timeout=game_timeout)
        except Exception as exc:
            print(f"[monitor] WARN seed={sd} game crashed: {exc!r}; "
                  f"recording 0", flush=True)
            sc, mf, mv = 0.0, -1, 0
        scores.append(sc); maxf.append(mf); moves.append(mv)
    return step, scores, maxf, moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config_policy_only.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "latest.pt"))
    ap.add_argument("--best", default=os.path.join(HERE, "checkpoints", "best.pt"))
    ap.add_argument("--csv", default=os.path.join(HERE, "logs", "eval_track.csv"))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--interval", type=float, default=1800.0)
    ap.add_argument("--max-moves", type=int, default=1000,
                    help="per-game move cap (anti-hang)")
    ap.add_argument("--game-timeout", type=float, default=180.0,
                    help="per-game wall-clock cap in seconds (anti-hang)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    try:
        os.nice(10)
    except Exception:
        pass

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    header = ["timestamp", "step", "n_games", "sims", "mean", "median",
              "max", "min", "avg_moves", "eval_sec", "rand_baseline"]
    if not os.path.exists(args.csv):
        with open(args.csv, "w", newline="") as f:
            csv.writer(f).writerow(header)

    best_mean = -1.0
    if os.path.exists(args.best):
        try:
            best_mean = float(torch.load(args.best, map_location="cpu")
                              .get("eval_mean", -1.0))
        except Exception:
            best_mean = -1.0

    print(f"[monitor] start interval={args.interval}s seeds={seeds} "
          f"sims={args.sims} csv={args.csv}", flush=True)
    while True:
        t0 = time.time()
        if os.path.exists(args.ckpt):
            try:
                step, scores, maxf, moves = evaluate(
                    cfg, args.ckpt, seeds, args.sims, device,
                    max_moves=args.max_moves, game_timeout=args.game_timeout)
                mean = float(np.mean(scores)); med = float(np.median(scores))
                row = [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       step, len(scores), args.sims, f"{mean:.1f}", f"{med:.1f}",
                       f"{max(scores):.0f}", f"{min(scores):.0f}",
                       f"{np.mean(moves):.1f}", f"{time.time()-t0:.1f}", 1156]
                with open(args.csv, "a", newline="") as f:
                    csv.writer(f).writerow(row)
                print(f"[monitor] step={step} mean={mean:.0f} med={med:.0f} "
                      f"max={max(scores):.0f} moves={np.mean(moves):.0f} "
                      f"(vs rand 1156)", flush=True)
                if mean > best_mean:
                    best_mean = mean
                    tmp = args.best + ".tmp"
                    d = torch.load(args.ckpt, map_location="cpu")
                    d["eval_mean"] = mean
                    torch.save(d, tmp); os.replace(tmp, args.best)
                    print(f"[monitor] NEW BEST mean={mean:.0f} -> best.pt", flush=True)
            except Exception as exc:
                print(f"[monitor] eval failed: {exc!r}", flush=True)
        else:
            print("[monitor] latest.pt not present yet; waiting", flush=True)
        dt = time.time() - t0
        time.sleep(max(5.0, args.interval - dt))


if __name__ == "__main__":
    main()
