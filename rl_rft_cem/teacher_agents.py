"""Policy-first teacher agents for Suika offline distillation.

The classes in this module expose a common ``decide(state) -> (x, debug)``
surface.  ``debug["top_actions"]`` is intentionally compatible with
``gen_teacher_replay.py`` and ``eval_suite.py`` so teacher policies can be
saved as soft targets instead of only as scalar value targets.
"""
import math
import time

import numpy as np

from action_candidates import refined_x_candidates
from common import (
    KILLY,
    PLAY_BOT,
    PLAY_LEFT,
    PLAY_RIGHT,
    PLAY_TOP,
    PLAY_W,
    SCORE_NORM,
    SPAWN_TYPES,
    col_to_x,
    config,
    radius_of,
)
from model import SuikaModel


def x_to_col(x, K):
    rel = (float(x) - PLAY_LEFT) / max(1e-9, PLAY_W)
    return int(np.clip(math.floor(rel * int(K)), 0, int(K) - 1))


def policy_from_action_rows(rows, K, temperature=100.0, top_k=0,
                            stochasticity=0.0, fallback_col=None):
    """Convert scored candidate rows into a soft K-bin policy."""
    pi = np.zeros((int(K),), dtype=np.float32)
    if not rows:
        col = int(fallback_col if fallback_col is not None else K // 2)
        pi[col] = 1.0
        return pi

    ordered = sorted(rows, key=lambda r: float(r.get("score", 0.0)),
                     reverse=True)
    if top_k and int(top_k) > 0:
        ordered = ordered[:int(top_k)]
    cols = np.array([int(r["col"]) for r in ordered], dtype=np.int64)
    scores = np.array([float(r.get("score", 0.0)) for r in ordered],
                      dtype=np.float64)
    if temperature <= 1e-9:
        pi[int(cols[int(np.argmax(scores))])] = 1.0
    else:
        scores = scores - float(scores.max())
        probs = np.exp(scores / float(temperature))
        probs = probs / max(float(probs.sum()), 1e-12)
        for c, p in zip(cols, probs):
            pi[int(c)] += float(p)
    if stochasticity > 0.0:
        eps = float(np.clip(stochasticity, 0.0, 1.0))
        pi = (1.0 - eps) * pi + eps / float(K)
    return (pi / max(float(pi.sum()), 1e-12)).astype(np.float32)


def _dedupe_xs(xs, min_gap=2.0):
    out = []
    for x in sorted(float(np.clip(x, PLAY_LEFT, PLAY_RIGHT)) for x in xs):
        if not out or abs(x - out[-1]) >= min_gap:
            out.append(x)
    return out


def _top_y(fruits):
    if not fruits:
        return float(PLAY_BOT)
    return min(float(f["y"]) - float(f["radius"]) for f in fruits)


def _merge_potential(fruits):
    bonus = 0.0
    for i, fi in enumerate(fruits):
        ti = int(fi["type"])
        for fj in fruits[i + 1:]:
            if ti != int(fj["type"]):
                continue
            d = math.hypot(float(fi["x"]) - float(fj["x"]),
                           float(fi["y"]) - float(fj["y"]))
            reach = float(fi["radius"]) + float(fj["radius"])
            if d < reach * 1.7:
                closeness = max(0.0, 1.0 - (d - reach) / (reach * 0.7 + 1e-9))
                bonus += closeness * config[ti, "points"]
    return float(bonus)


def _wall_pressure(fruits):
    pressure = 0.0
    for f in fruits:
        r = float(f["radius"])
        x = float(f["x"])
        near = min(x - PLAY_LEFT, PLAY_RIGHT - x)
        pressure += max(0.0, (r * 1.4 - near) / max(r, 1.0))
    return float(pressure)


def heuristic_state_value(state, gain=0.0, done=False):
    if done or state.get("game_over"):
        return -1.0e6 + 4.0 * float(gain)
    fruits = state.get("fruits") or []
    top_y = _top_y(fruits)
    danger = max(0.0, (KILLY + 90.0) - top_y)
    dense = max(0.0, len(fruits) - 28)
    return float(
        4.0 * float(gain)
        + 1.0 * top_y
        + 0.65 * _merge_potential(fruits)
        - 1.7 * len(fruits)
        - 380.0 * danger / 90.0
        - 22.0 * _wall_pressure(fruits)
        - 6.0 * dense
    )


class HeuristicOneStepPlusAgent:
    """High-fidelity one-step teacher with broad continuous candidates."""

    def __init__(self, K=16, lookahead_steps=220, max_candidates=28,
                 policy_temperature=100.0, policy_top_k=8,
                 policy_stochasticity=0.0, seed=0):
        self.K = int(K)
        self.lookahead_steps = int(lookahead_steps)
        self.max_candidates = int(max_candidates)
        self.policy_temperature = float(policy_temperature)
        self.policy_top_k = int(policy_top_k)
        self.policy_stochasticity = float(policy_stochasticity)
        self.rng = np.random.default_rng(seed)
        self.decisions = 0

    def _candidate_xs(self, state):
        cur_t = int(state["current"]["type"])
        r = radius_of(cur_t)
        xs = [float(np.clip(col_to_x(c, self.K), PLAY_LEFT + r, PLAY_RIGHT - r))
              for c in range(self.K)]
        xs.extend(refined_x_candidates(
            state, self.K, policy=None, top_k=4, jitter=max(8.0, r * 0.35),
            max_candidates=self.max_candidates))
        for f in state.get("fruits") or []:
            if int(f.get("type", -1)) == cur_t:
                xs.extend([float(f["x"]) - 0.5 * r, float(f["x"]),
                           float(f["x"]) + 0.5 * r])
        return _dedupe_xs(xs)[:max(self.K, self.max_candidates)]

    def evaluate(self, state):
        from ai_agent import simulate_drop

        rows = []
        for x in self._candidate_xs(state):
            try:
                out = simulate_drop(state, x, max_steps=self.lookahead_steps)
                score = heuristic_state_value(
                    {"fruits": out["fruits"], "score": state.get("score", 0),
                     "game_over": out["game_over"]},
                    gain=out["score_gain"], done=out["game_over"])
                row = {
                    "x": float(x),
                    "col": x_to_col(x, self.K),
                    "score": float(score),
                    "gain": float(out["score_gain"]),
                    "game_over": bool(out["game_over"]),
                    "n_fruits": int(len(out["fruits"])),
                    "top_y": float(_top_y(out["fruits"])),
                }
            except Exception as exc:
                row = {
                    "x": float(x),
                    "col": x_to_col(x, self.K),
                    "score": -1.0e9,
                    "error": repr(exc),
                }
            rows.append(row)
        rows.sort(key=lambda r: (float(r["score"]),
                                 -abs(float(r["x"]) - (PLAY_LEFT + PLAY_RIGHT) / 2.0)),
                  reverse=True)
        return rows

    def decide(self, state):
        if state.get("game_over"):
            return col_to_x(self.K // 2, self.K), {"reason": "game_over"}
        t0 = time.perf_counter()
        rows = self.evaluate(state)
        best = rows[0] if rows else {"x": col_to_x(self.K // 2, self.K),
                                     "col": self.K // 2, "score": 0.0}
        pi = policy_from_action_rows(
            rows, self.K, temperature=self.policy_temperature,
            top_k=self.policy_top_k, stochasticity=self.policy_stochasticity,
            fallback_col=int(best["col"]))
        self.decisions += 1
        return float(best["x"]), {
            "teacher": "heuristic_1step_plus",
            "best_col": int(best["col"]),
            "best_x": float(best["x"]),
            "best_score": float(best["score"]),
            "top_actions": rows[:max(1, self.policy_top_k or 8)],
            "policy": pi.tolist(),
            "eval_sec": time.perf_counter() - t0,
        }


class TwoStepExpectimaxAgent:
    """Two-step expectimax teacher using current + next fruit preview."""

    def __init__(self, K=16, branch_width=8, first_fast=False,
                 second_fast=True, chance_mode="enumerate",
                 policy_temperature=100.0, policy_top_k=8,
                 policy_stochasticity=0.0, seed=0, fast_steps=70,
                 fast_settle_v=4.0, fast_check_every=3):
        self.K = int(K)
        self.branch_width = int(branch_width)
        self.first_fast = bool(first_fast)
        self.second_fast = bool(second_fast)
        self.chance_mode = str(chance_mode)
        self.policy_temperature = float(policy_temperature)
        self.policy_top_k = int(policy_top_k)
        self.policy_stochasticity = float(policy_stochasticity)
        self.rng = np.random.default_rng(seed)
        self.model = SuikaModel(K=self.K, fast=True, fast_steps=fast_steps,
                                fast_settle_v=fast_settle_v,
                                fast_check_every=fast_check_every)
        self.decisions = 0
        self.model_steps = 0

    def _chance_fruits(self):
        if self.chance_mode == "sample":
            return [int(self.rng.integers(0, SPAWN_TYPES))]
        return list(range(SPAWN_TYPES))

    def _candidate_cols(self, state):
        if self.branch_width >= self.K:
            return list(range(self.K))
        scored = []
        cur_type = int(state["current"]["type"])
        r = radius_of(cur_type)
        for col in range(self.K):
            x = col_to_x(col, self.K)
            fruits = state.get("fruits") or []
            same = [f for f in fruits if int(f["type"]) == cur_type]
            same_bonus = 0.0
            if same:
                nearest = min(abs(float(f["x"]) - x) for f in same)
                same_bonus = 3.0 * config[cur_type, "points"] * max(
                    0.0, 1.0 - nearest / 90.0)
            overlap = [f for f in fruits
                       if abs(float(f["x"]) - x) < float(f["radius"]) + r]
            surface_y = min((float(f["y"]) - float(f["radius"]) for f in overlap),
                            default=PLAY_BOT)
            wall_pen = 0.04 * abs(x - (PLAY_LEFT + PLAY_RIGHT) / 2.0)
            scored.append((same_bonus + surface_y - wall_pen, col))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [c for _s, c in scored[:max(1, self.branch_width)]]

    def _root_value(self, state, root_col):
        vals = []
        for nf1 in self._chance_fruits():
            s1, g1, d1 = self.model.step(state, root_col, nf1,
                                         fast=self.first_fast)
            self.model_steps += 1
            if d1:
                vals.append(heuristic_state_value(s1, gain=g1, done=True))
                continue
            second_vals = []
            for col2 in self._candidate_cols(s1):
                chance_vals = []
                for nf2 in self._chance_fruits():
                    s2, g2, d2 = self.model.step(s1, col2, nf2,
                                                 fast=self.second_fast)
                    self.model_steps += 1
                    chance_vals.append(
                        4.0 * float(g1) + heuristic_state_value(
                            s2, gain=g2, done=d2))
                second_vals.append(float(np.mean(chance_vals)))
            vals.append(max(second_vals) if second_vals else
                        heuristic_state_value(s1, gain=g1, done=d1))
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "q20": float(np.quantile(arr, 0.2)),
            "n": int(arr.size),
            "score": float(arr.mean() - 0.25 * arr.std(ddof=0)),
        }

    def decide(self, state):
        if state.get("game_over"):
            return col_to_x(self.K // 2, self.K), {"reason": "game_over"}
        t0 = time.perf_counter()
        before_steps = self.model_steps
        rows = []
        for col in range(self.K):
            val = self._root_value(state, col)
            val.update({"col": int(col), "x": col_to_x(col, self.K)})
            rows.append(val)
        center = (self.K - 1) / 2.0
        rows.sort(key=lambda r: (float(r["score"]), -abs(r["col"] - center)),
                  reverse=True)
        pi = policy_from_action_rows(
            rows, self.K, temperature=self.policy_temperature,
            top_k=self.policy_top_k, stochasticity=self.policy_stochasticity,
            fallback_col=int(rows[0]["col"]))
        self.decisions += 1
        return col_to_x(int(rows[0]["col"]), self.K), {
            "teacher": "two_step_expectimax",
            "best_col": int(rows[0]["col"]),
            "best_score": float(rows[0]["score"]),
            "top_actions": rows[:max(1, self.policy_top_k or 8)],
            "policy": pi.tolist(),
            "model_steps": int(self.model_steps - before_steps),
            "eval_sec": time.perf_counter() - t0,
        }


class RobustBeamSearchV2Agent:
    """Risk-aware beam teacher with precise root and per-root beams."""

    def __init__(self, K=16, depth=3, beam_size=8, samples_per_action=2,
                 branch_width=8, risk_lambda=0.5, quantile=0.2,
                 risk_mode="mean_std", chance_mode="enumerate", seed=0,
                 fast_steps=70, fast_settle_v=4.0, fast_check_every=3,
                 iterations=None, policy_temperature=100.0, policy_top_k=8,
                 policy_stochasticity=0.0):
        self.K = int(K)
        self.depth = int(depth)
        self.beam_size = int(beam_size)
        self.samples_per_action = int(samples_per_action)
        self.branch_width = int(branch_width)
        self.risk_lambda = float(risk_lambda)
        self.quantile = float(quantile)
        self.risk_mode = str(risk_mode)
        self.chance_mode = str(chance_mode)
        self.policy_temperature = float(policy_temperature)
        self.policy_top_k = int(policy_top_k)
        self.policy_stochasticity = float(policy_stochasticity)
        self.rng = np.random.default_rng(seed)
        self.model = SuikaModel(K=self.K, fast=True, fast_steps=fast_steps,
                                fast_settle_v=fast_settle_v,
                                fast_check_every=fast_check_every,
                                iterations=iterations)
        self.decisions = 0
        self.model_steps = 0

    def _chance_fruits(self):
        if self.chance_mode == "enumerate":
            return list(range(SPAWN_TYPES))
        n = max(1, self.samples_per_action)
        replace = n > SPAWN_TYPES
        return [int(x) for x in self.rng.choice(SPAWN_TYPES, size=n,
                                                replace=replace)]

    def _risk_score(self, vals):
        arr = np.asarray(vals, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        q = float(np.quantile(arr, self.quantile))
        score = q if self.risk_mode == "quantile" else mean - self.risk_lambda * std
        return score, mean, std, q

    def _candidate_cols(self, state):
        return TwoStepExpectimaxAgent(
            K=self.K, branch_width=self.branch_width)._candidate_cols(state)

    def _expand_root(self, state, root_col):
        nodes = []
        vals = []
        for nf in self._chance_fruits():
            ns, gain, done = self.model.step(state, root_col, nf, fast=False)
            self.model_steps += 1
            value = heuristic_state_value(ns, gain=gain, done=done)
            vals.append(value)
            if not done:
                nodes.append((value, float(gain), ns, 1))
        nodes.sort(key=lambda n: n[0], reverse=True)
        return nodes[:self.beam_size], vals

    def _search_root(self, state, root_col):
        nodes, root_vals = self._expand_root(state, root_col)
        values = list(root_vals)
        for depth_idx in range(1, self.depth):
            expanded = []
            for _value, gain_so_far, node_state, _d in nodes:
                for col in self._candidate_cols(node_state):
                    chance_vals = []
                    for nf in self._chance_fruits():
                        ns, gain, done = self.model.step(
                            node_state, col, nf, fast=True)
                        self.model_steps += 1
                        total_gain = gain_so_far + float(gain)
                        v = heuristic_state_value(ns, gain=total_gain,
                                                  done=done)
                        values.append(v)
                        chance_vals.append(v)
                        if not done:
                            expanded.append((v, total_gain, ns, depth_idx + 1))
                    if chance_vals:
                        values.append(float(np.mean(chance_vals)))
            if not expanded:
                break
            expanded.sort(key=lambda n: n[0], reverse=True)
            nodes = expanded[:self.beam_size]
        score, mean, std, q = self._risk_score(values)
        return {"score": score, "mean": mean, "std": std, "q": q,
                "n": len(values), "depth": self.depth}

    def decide(self, state):
        if state.get("game_over"):
            return col_to_x(self.K // 2, self.K), {"reason": "game_over"}
        t0 = time.perf_counter()
        before_steps = self.model_steps
        rows = []
        for col in range(self.K):
            row = self._search_root(state, col)
            row.update({"col": int(col), "x": col_to_x(col, self.K)})
            rows.append(row)
        center = (self.K - 1) / 2.0
        rows.sort(key=lambda r: (float(r["score"]), -abs(r["col"] - center)),
                  reverse=True)
        pi = policy_from_action_rows(
            rows, self.K, temperature=self.policy_temperature,
            top_k=self.policy_top_k, stochasticity=self.policy_stochasticity,
            fallback_col=int(rows[0]["col"]))
        self.decisions += 1
        return col_to_x(int(rows[0]["col"]), self.K), {
            "teacher": "robust_beam_v2",
            "best_col": int(rows[0]["col"]),
            "best_score": float(rows[0]["score"]),
            "best_mean": float(rows[0]["mean"]),
            "best_std": float(rows[0]["std"]),
            "best_q": float(rows[0]["q"]),
            "top_actions": rows[:max(1, self.policy_top_k or 8)],
            "policy": pi.tolist(),
            "model_steps": int(self.model_steps - before_steps),
            "eval_sec": time.perf_counter() - t0,
        }


class EnsembleTeacherAgent:
    """Average soft policies from several teachers and play the aggregate top-1."""

    def __init__(self, agents, K=16, weights=None, policy_temperature=1.0):
        self.agents = list(agents)
        self.K = int(K)
        if weights is None:
            weights = [1.0] * len(self.agents)
        w = np.asarray(weights, dtype=np.float64)
        self.weights = w / max(float(w.sum()), 1e-12)
        self.policy_temperature = float(policy_temperature)

    def decide(self, state):
        policies, debugs = [], []
        for agent in self.agents:
            _x, dbg = agent.decide(state)
            pi = np.asarray(dbg.get("policy") or policy_from_action_rows(
                dbg.get("top_actions") or [], self.K,
                temperature=self.policy_temperature,
                fallback_col=dbg.get("best_col")), dtype=np.float64)
            policies.append(pi / max(float(pi.sum()), 1e-12))
            debugs.append(dbg)
        agg = np.zeros((self.K,), dtype=np.float64)
        for w, pi in zip(self.weights, policies):
            agg += float(w) * pi
        agg = agg / max(float(agg.sum()), 1e-12)
        col = int(np.argmax(agg))
        rows = [{"col": int(c), "x": col_to_x(c, self.K),
                 "score": float(agg[c])} for c in range(self.K)]
        rows.sort(key=lambda r: r["score"], reverse=True)
        return col_to_x(col, self.K), {
            "teacher": "ensemble",
            "best_col": col,
            "best_score": float(agg[col]),
            "policy": agg.astype(np.float32).tolist(),
            "top_actions": rows[:8],
            "members": [d.get("teacher", "unknown") for d in debugs],
        }


def build_teacher_agent(name, cfg, args, seed=0):
    """Factory shared by replay generation and eval scripts."""
    def arg(*names, default=None):
        for n in names:
            if hasattr(args, n):
                return getattr(args, n)
        return default

    K = int(cfg["K"])
    common = {
        "K": K,
        "policy_temperature": float(arg("policy_temperature", default=100.0)),
        "policy_top_k": int(arg("policy_top_k", default=8)),
        "policy_stochasticity": float(arg("policy_stochasticity", default=0.0)),
        "seed": int(seed),
    }
    name = str(name).lower()
    if name in ("heuristic_plus", "heuristic_1step_plus", "heuristic"):
        return HeuristicOneStepPlusAgent(
            lookahead_steps=int(arg("lookahead_steps", default=220)),
            max_candidates=int(arg("max_candidates", default=28)),
            **common)
    if name in ("two_step", "two_step_expectimax", "expectimax"):
        return TwoStepExpectimaxAgent(
            branch_width=int(arg("branch_width", "robust_branch", default=8)),
            chance_mode=str(arg("chance_mode", default="enumerate")),
            fast_steps=int(arg("fast_steps", "robust_fast_steps", default=70)),
            fast_settle_v=float(arg("fast_settle_v", default=4.0)),
            fast_check_every=int(arg("fast_check_every", "robust_check_every", default=3)),
            **common)
    if name in ("robust_v2", "robust_beam_v2"):
        return RobustBeamSearchV2Agent(
            depth=int(arg("depth", "robust_depth", default=3)),
            beam_size=int(arg("beam_size", "robust_beam", default=8)),
            samples_per_action=int(arg("samples_per_action", "robust_samples", default=2)),
            branch_width=int(arg("branch_width", "robust_branch", default=8)),
            risk_lambda=float(arg("risk_lambda", "robust_lambda", default=0.5)),
            quantile=float(arg("quantile", default=0.2)),
            risk_mode=str(arg("risk_mode", "robust_mode", default="mean_std")),
            chance_mode=str(arg("chance_mode", default="enumerate")),
            fast_steps=int(arg("fast_steps", "robust_fast_steps", default=70)),
            fast_settle_v=float(arg("fast_settle_v", default=4.0)),
            fast_check_every=int(arg("fast_check_every", "robust_check_every", default=3)),
            iterations=arg("iterations", "robust_iterations", default=None),
            **common)
    if name == "ensemble":
        agents = [
            HeuristicOneStepPlusAgent(**common),
            TwoStepExpectimaxAgent(
                branch_width=int(arg("branch_width", "robust_branch", default=8)),
                chance_mode=str(arg("chance_mode", default="enumerate")),
                **common),
        ]
        if bool(arg("ensemble_include_robust", default=False)):
            agents.append(RobustBeamSearchV2Agent(
                depth=int(arg("depth", "robust_depth", default=3)),
                beam_size=int(arg("beam_size", "robust_beam", default=8)),
                branch_width=int(arg("branch_width", "robust_branch", default=8)),
                **common))
        return EnsembleTeacherAgent(agents, K=K)
    raise ValueError("unknown teacher agent: %s" % name)


def normalised_score_value(final_score, score_before):
    return np.float32((float(final_score) - float(score_before)) / SCORE_NORM)
