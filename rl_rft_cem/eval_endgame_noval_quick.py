#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""隔离实验快速评测：整局从空盘玩，去 value + MCTS 叶子启发式，与本实验一致。
只读 checkpoints/latest.pt，不写 replay。带时间预算，超时提前停止。"""
import argparse, json, os, time
import numpy as np
import torch
import yaml
from net import build_net
from mcts import MCTS
from suika_env import SuikaEnv

HERE = os.path.dirname(os.path.abspath(__file__))

def max_fruit(state):
    return max((f["type"] for f in state["fruits"]), default=-1)

def play_one(net, device, cfg, seed, sims):
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
    actions = []
    while not state["game_over"]:
        action, _ = mcts.policy(state, temperature=0.0, add_noise=False)
        actions.append(int(action))
        state, _r, done, _ = env.step_column(action, num_columns=cfg["K"])
        moves += 1
        if done:
            break
    dt = time.time() - t0
    # 动作多样性：唯一动作数 / 是否塌缩
    uniq = len(set(actions))
    return {"seed": seed, "score": float(state["score"]), "max_fruit": int(max_fruit(state)),
            "moves": moves, "sec": dt, "sec_per_move": dt/max(moves,1),
            "uniq_actions": uniq, "K": cfg["K"]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.effective.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "latest.pt"))
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sims", type=int, default=0)  # 0 -> 用 config
    ap.add_argument("--budget", type=float, default=420.0)  # 秒；超出后停止
    ap.add_argument("--out", default=os.path.join(HERE, "plots", "eval_quick.json"))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sims = args.sims or cfg["num_simulations"]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    device = torch.device(args.device)
    net = build_net(cfg).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ck["net"]); net.eval()
    step = int(ck.get("step", 0))
    print(f"[eval] ckpt step={step} device={device} sims={sims} seeds={seeds} budget={args.budget}s", flush=True)

    rows = []
    t_start = time.time()
    for sd in seeds:
        r = play_one(net, device, cfg, sd, sims)
        rows.append(r)
        print(f"[eval] seed={sd:>2} score={r['score']:.0f} maxfruit={r['max_fruit']} "
              f"moves={r['moves']} sec={r['sec']:.1f} sec/mv={r['sec_per_move']:.2f} "
              f"uniqAct={r['uniq_actions']}/{r['K']}", flush=True)
        if time.time() - t_start > args.budget:
            print(f"[eval] budget exceeded after {len(rows)} games, stopping early", flush=True)
            break

    scores = [r["score"] for r in rows]
    moves = [r["moves"] for r in rows]
    spm = [r["sec_per_move"] for r in rows]
    maxf = [r["max_fruit"] for r in rows]
    summary = {
        "ckpt_step": step, "n_games": len(rows), "sims": sims, "device": str(device),
        "score_mean": float(np.mean(scores)), "score_median": float(np.median(scores)),
        "score_max": float(np.max(scores)), "score_min": float(np.min(scores)),
        "max_fruit": int(np.max(maxf)), "avg_moves": float(np.mean(moves)),
        "sec_per_move": float(np.mean(spm)),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[eval] SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
