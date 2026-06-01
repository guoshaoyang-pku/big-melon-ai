#!/usr/bin/env python3
"""Read-only benchmark for a risk-sensitive beam-search Suika baseline.

This script intentionally does not import or touch the AlphaZero training loop,
replay buffer, checkpoints, or config files. It evaluates a greedy player in the
real headless SuikaEnv while using SuikaModel only for lookahead scoring.
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from statistics import mean, median

import numpy as np

# Make imports robust when launched from either suika/ or suika/rl/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
_PART2 = os.path.join(_ROOT, "part2")
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PART2 not in sys.path:
    sys.path.insert(0, _PART2)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from common import (  # noqa: E402
    KILLY,
    PLAY_BOT,
    PLAY_LEFT,
    PLAY_RIGHT,
    PLAY_TOP,
    PLAY_W,
    SPAWN_TYPES,
    col_to_x,
    config,
    radius_of,
)
from model import SuikaModel  # noqa: E402
from suika_env import SuikaEnv  # noqa: E402


FRUIT_NAMES = list(config.fruit_names)


@dataclass
class Node:
    value: float
    gain: float
    state: dict
    root_col: int
    done: bool
    depth: int


class RobustBeamSearchAgent:
    """Risk-sensitive finite-depth beam-search baseline.

    The root considers all K columns. Future layers can use a narrower static
    candidate set to keep the benchmark affordable while training is running.
    """

    W_GAIN = 4.0
    W_SAFETY = 1.0
    W_POTENTIAL = 0.6
    W_FRUITS = 1.5
    W_DANGER = 350.0
    W_GAMEOVER = 1.0e6

    def __init__(
        self,
        K=16,
        depth=4,
        beam_size=8,
        samples_per_action=2,
        branch_width=None,
        risk_lambda=0.5,
        quantile=0.2,
        risk_mode="mean_std",
        seed=0,
        fast_steps=70,
        fast_settle_v=4.0,
        fast_check_every=3,
        iterations=None,
    ):
        self.K = int(K)
        self.depth = int(depth)
        self.beam_size = int(beam_size)
        self.samples_per_action = int(samples_per_action)
        self.branch_width = int(branch_width or K)
        self.risk_lambda = float(risk_lambda)
        self.quantile = float(quantile)
        self.risk_mode = str(risk_mode)
        self.seed = int(seed)
        self.model = SuikaModel(
            K=self.K,
            fast=True,
            fast_steps=int(fast_steps),
            fast_settle_v=float(fast_settle_v),
            fast_check_every=int(fast_check_every),
            iterations=iterations,
        )
        self.decisions = 0
        self.model_steps = 0
        self.eval_seconds = 0.0
        self.last_debug = None

    def decide(self, state):
        if state.get("game_over"):
            return col_to_x(self.K // 2, self.K), {"reason": "game_over"}
        t0 = time.perf_counter()
        col, debug = self._search(state)
        elapsed = time.perf_counter() - t0
        self.decisions += 1
        self.eval_seconds += elapsed
        debug["eval_sec"] = elapsed
        self.last_debug = debug
        return col_to_x(col, self.K), debug

    def _sample_fruits(self, rng):
        n = max(1, self.samples_per_action)
        replace = n > SPAWN_TYPES
        return [int(x) for x in rng.choice(SPAWN_TYPES, size=n, replace=replace)]

    def _search(self, state):
        # Stable per-decision RNG keeps runs reproducible independent of wall time.
        rng = np.random.default_rng(self.seed + 1000003 * self.decisions)
        root_samples = self._sample_fruits(rng)
        values_by_depth = [dict() for _ in range(max(1, self.depth))]
        nodes = []
        root_model_steps = 0

        for root_col in range(self.K):
            for next_fruit in root_samples:
                ns, reward, done = self.model.step(state, root_col, next_fruit, fast=True)
                root_model_steps += 1
                gain = float(reward)
                value = self._node_value(ns, gain, done)
                nodes.append(Node(value, gain, ns, root_col, done, 1))
                values_by_depth[0].setdefault(root_col, []).append(value)

        nodes = self._top_nodes(nodes, self.beam_size)
        depth_reached = 1
        for depth_idx in range(1, self.depth):
            expanded = []
            for node in nodes:
                if node.done:
                    expanded.append(node)
                    values_by_depth[depth_idx].setdefault(node.root_col, []).append(node.value)
                    continue
                cols = self._candidate_cols(node.state, self.branch_width)
                for col in cols:
                    for next_fruit in self._sample_fruits(rng):
                        ns, reward, done = self.model.step(
                            node.state, col, next_fruit, fast=True
                        )
                        root_model_steps += 1
                        gain = node.gain + float(reward)
                        value = self._node_value(ns, gain, done)
                        expanded.append(Node(value, gain, ns, node.root_col, done, depth_idx + 1))
                        values_by_depth[depth_idx].setdefault(node.root_col, []).append(value)
            if not expanded:
                break
            nodes = self._top_nodes(expanded, self.beam_size)
            depth_reached = depth_idx + 1

        self.model_steps += root_model_steps
        action_rows = []
        for root_col in range(self.K):
            vals = None
            used_depth = 0
            for d in range(depth_reached - 1, -1, -1):
                vals = values_by_depth[d].get(root_col)
                if vals:
                    used_depth = d + 1
                    break
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            row = {
                "col": root_col,
                "n": int(arr.size),
                "depth": used_depth,
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "q": float(np.quantile(arr, self.quantile)),
            }
            row["mean_std"] = row["mean"] - self.risk_lambda * row["std"]
            row["score"] = row["q"] if self.risk_mode == "quantile" else row["mean_std"]
            action_rows.append(row)

        center = (self.K - 1) / 2.0
        action_rows.sort(key=lambda r: (r["score"], -abs(r["col"] - center)), reverse=True)
        best = action_rows[0]
        return int(best["col"]), {
            "best_col": int(best["col"]),
            "best_score": float(best["score"]),
            "best_mean": float(best["mean"]),
            "best_std": float(best["std"]),
            "best_q": float(best["q"]),
            "best_n": int(best["n"]),
            "best_depth": int(best["depth"]),
            "depth_reached": int(depth_reached),
            "root_samples": root_samples,
            "model_steps": int(root_model_steps),
            "top_actions": action_rows[:5],
        }

    @staticmethod
    def _top_nodes(nodes, limit):
        if len(nodes) <= limit:
            return nodes
        nodes.sort(key=lambda n: n.value, reverse=True)
        return nodes[:limit]

    def _candidate_cols(self, state, width):
        width = max(1, min(int(width), self.K))
        if width >= self.K:
            return list(range(self.K))
        scored = [(self._static_action_score(state, col), col) for col in range(self.K)]
        center = (self.K - 1) / 2.0
        scored.sort(key=lambda t: (t[0], -abs(t[1] - center)), reverse=True)
        return [col for _, col in scored[:width]]

    def _static_action_score(self, state, col):
        cur_type = int(state["current"]["type"])
        radius = radius_of(cur_type)
        x = col_to_x(col, self.K)
        fruits = state["fruits"]
        same = [f for f in fruits if int(f["type"]) == cur_type]
        same_bonus = 0.0
        if same:
            nearest = min(abs(float(f["x"]) - x) for f in same)
            same_bonus = 3.0 * config[cur_type, "points"] * max(0.0, 1.0 - nearest / 90.0)
        overlap = [f for f in fruits if abs(float(f["x"]) - x) < float(f["radius"]) + radius]
        surface_y = min((float(f["y"]) - float(f["radius"]) for f in overlap), default=PLAY_BOT)
        danger = 500.0 if surface_y < KILLY + 60 else 0.0
        center_penalty = 0.03 * abs(x - (PLAY_LEFT + PLAY_RIGHT) / 2.0)
        return same_bonus + surface_y - danger - center_penalty

    def _node_value(self, state, gain, done):
        if done or state.get("game_over"):
            return -self.W_GAMEOVER + self.W_GAIN * float(gain)
        fruits = state["fruits"]
        top_y = self._top_y(fruits)
        danger = max(0.0, (KILLY + 80.0) - top_y)
        return (
            self.W_GAIN * float(gain)
            + self.W_SAFETY * top_y
            + self.W_POTENTIAL * self._merge_potential(fruits)
            - self.W_FRUITS * len(fruits)
            - self.W_DANGER * danger / 80.0
        )

    @staticmethod
    def _top_y(fruits):
        if not fruits:
            return float(PLAY_BOT)
        return min(float(f["y"]) - float(f["radius"]) for f in fruits)

    @staticmethod
    def _merge_potential(fruits):
        bonus = 0.0
        n = len(fruits)
        for i in range(n):
            fi = fruits[i]
            ti = int(fi["type"])
            for j in range(i + 1, n):
                fj = fruits[j]
                if ti != int(fj["type"]):
                    continue
                dx = float(fi["x"]) - float(fj["x"])
                dy = float(fi["y"]) - float(fj["y"])
                d = math.hypot(dx, dy)
                reach = float(fi["radius"]) + float(fj["radius"])
                if d < reach * 1.6:
                    closeness = max(0.0, 1.0 - (d - reach) / (reach * 0.6 + 1e-9))
                    bonus += closeness * config[ti, "points"]
        return float(bonus)


def run_episode(args, seed):
    env = SuikaEnv(seed=int(seed))
    state = env.get_state()
    if args.v2:
        from teacher_agents import RobustBeamSearchV2Agent
        agent = RobustBeamSearchV2Agent(
            K=args.K,
            depth=args.depth,
            beam_size=args.beam_size,
            samples_per_action=args.samples_per_action,
            branch_width=args.branch_width,
            risk_lambda=args.risk_lambda,
            quantile=args.quantile,
            risk_mode=args.risk_mode,
            chance_mode=args.chance_mode,
            seed=args.agent_seed + int(seed) * 10007,
            fast_steps=args.fast_steps,
            fast_settle_v=args.fast_settle_v,
            fast_check_every=args.fast_check_every,
            iterations=args.iterations,
        )
    else:
        agent = RobustBeamSearchAgent(
            K=args.K,
            depth=args.depth,
            beam_size=args.beam_size,
            samples_per_action=args.samples_per_action,
            branch_width=args.branch_width,
            risk_lambda=args.risk_lambda,
            quantile=args.quantile,
            risk_mode=args.risk_mode,
            seed=args.agent_seed + int(seed) * 10007,
            fast_steps=args.fast_steps,
            fast_settle_v=args.fast_settle_v,
            fast_check_every=args.fast_check_every,
            iterations=args.iterations,
        )
    rewards = []
    debug_first = None
    start = time.perf_counter()
    truncated = False
    while not state.get("game_over"):
        action_x, debug = agent.decide(state)
        if debug_first is None:
            debug_first = debug
        state, reward, done, info = env.step(action_x)
        rewards.append(float(reward))
        if done:
            break
        if args.max_moves and len(rewards) >= args.max_moves:
            truncated = True
            break
    elapsed = time.perf_counter() - start
    fruits = state.get("fruits", [])
    max_type = max((int(f["type"]) for f in fruits), default=-1)
    return {
        "seed": int(seed),
        "score": int(state.get("score", 0)),
        "steps": int(len(rewards)),
        "max_type": int(max_type),
        "max_fruit": FRUIT_NAMES[max_type] if max_type >= 0 else "none",
        "seconds": float(elapsed),
        "sec_per_step": float(elapsed / max(1, len(rewards))),
        "model_steps": int(agent.model_steps),
        "model_steps_per_move": float(agent.model_steps / max(1, len(rewards))),
        "eval_sec_per_move": float(agent.eval_seconds / max(1, agent.decisions)),
        "truncated": bool(truncated),
        "first_debug": debug_first,
    }


def summarize(rows):
    scores = [r["score"] for r in rows]
    steps = [r["steps"] for r in rows]
    secs = [r["seconds"] for r in rows]
    per_step = [r["sec_per_step"] for r in rows]
    return {
        "episodes": len(rows),
        "score_mean": float(mean(scores)) if scores else 0.0,
        "score_median": float(median(scores)) if scores else 0.0,
        "score_max": int(max(scores)) if scores else 0,
        "score_min": int(min(scores)) if scores else 0,
        "score_std": float(np.std(scores, ddof=0)) if scores else 0.0,
        "steps_mean": float(mean(steps)) if steps else 0.0,
        "seconds_total": float(sum(secs)),
        "sec_per_step_mean": float(mean(per_step)) if per_step else 0.0,
        "max_type_max": int(max((r["max_type"] for r in rows), default=-1)),
        "max_fruit_max": FRUIT_NAMES[max((r["max_type"] for r in rows), default=-1)] if rows else "none",
        "any_truncated": any(r["truncated"] for r in rows),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seeds", type=str, default=None, help="comma-separated seeds; overrides --episodes")
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--beam-size", type=int, default=8)
    p.add_argument("--samples-per-action", type=int, default=2)
    p.add_argument("--branch-width", type=int, default=8, help="future-layer candidate columns; root still uses all K")
    p.add_argument("--risk-lambda", type=float, default=0.5)
    p.add_argument("--quantile", type=float, default=0.2)
    p.add_argument("--risk-mode", choices=["mean_std", "quantile"], default="mean_std")
    p.add_argument("--agent-seed", type=int, default=12345)
    p.add_argument("--fast-steps", type=int, default=70)
    p.add_argument("--fast-settle-v", type=float, default=4.0)
    p.add_argument("--fast-check-every", type=int, default=3)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--v2", action="store_true",
                   help="use precise-root per-root robust beam v2 teacher")
    p.add_argument("--chance-mode", choices=["enumerate", "sample"],
                   default="enumerate")
    p.add_argument("--max-moves", type=int, default=0, help="safety cap only; 0 means play until game_over")
    p.add_argument("--json", action="store_true", help="also print JSON lines for parsing")
    return p.parse_args()


def main():
    args = parse_args()
    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    else:
        seeds = list(range(args.episodes))
    cfg = {
        "K": args.K,
        "depth": args.depth,
        "beam_size": args.beam_size,
        "samples_per_action": args.samples_per_action,
        "branch_width": args.branch_width,
        "risk_mode": args.risk_mode,
        "risk_lambda": args.risk_lambda,
        "quantile": args.quantile,
        "fast_steps": args.fast_steps,
        "fast_settle_v": args.fast_settle_v,
        "fast_check_every": args.fast_check_every,
        "iterations": args.iterations,
        "v2": bool(args.v2),
        "chance_mode": args.chance_mode,
        "seeds": seeds,
        "max_moves": args.max_moves,
    }
    print("CONFIG " + json.dumps(cfg, ensure_ascii=False, sort_keys=True), flush=True)
    rows = []
    for seed in seeds:
        row = run_episode(args, seed)
        rows.append(row)
        print(
            "EPISODE seed={seed} score={score} steps={steps} max={max_type}:{max_fruit} "
            "sec={seconds:.2f} sec_step={sec_per_step:.3f} model_steps_move={model_steps_per_move:.1f} "
            "truncated={truncated}".format(**row),
            flush=True,
        )
        if args.json:
            print("JSON_EPISODE " + json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    summary = summarize(rows)
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if args.json:
        print("JSON_ALL " + json.dumps({"config": cfg, "episodes": rows, "summary": summary}, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
