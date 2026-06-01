#!/usr/bin/env python3
"""Tiny CEM-style improvement skeleton for Qwen Suika VLA.

This is deliberately a data export loop, not a trainer: rollout the current
Qwen policy, keep elite trajectories by score, and write their state/action
pairs as SFT JSONL for the next LoRA round.  Keep --games/--max-moves small
while debugging; --dry-run avoids model loading entirely.
"""
import argparse
import json
import os
import time
from typing import Any, Dict, List

from qwen_policy_agent import (
    DEFAULT_DRY_RUN_RESPONSE,
    DEFAULT_MODEL_ID,
    QwenCoordinatePolicyAgent,
)
from qwen_vla_dataset import record_from_state_action, write_jsonl


def _rollout(seed: int, agent: QwenCoordinatePolicyAgent, max_moves: int) -> Dict[str, Any]:
    from suika_env import SuikaEnv

    env = SuikaEnv(seed=seed)
    state = env.get_state()
    samples = []
    move = 0
    while not state.get("game_over"):
        if max_moves and move >= max_moves:
            break
        decision = agent.decide_with_debug(state)
        samples.append({
            "state": state,
            "x_rel": decision.x_rel,
            "x_abs": decision.x_abs,
            "raw_text": decision.raw_text,
            "source": decision.source,
            "parsed": decision.parsed,
            "fallback_used": decision.fallback_used,
        })
        state, _reward, done, _info = env.step(decision.x_abs)
        move += 1
        if done:
            break
    return {
        "seed": int(seed),
        "score": int(state.get("score", 0)),
        "moves": int(move),
        "game_over": bool(state.get("game_over", False)),
        "samples": samples,
    }


def _elite_trajectories(trajectories: List[Dict[str, Any]], elite_frac: float) -> List[Dict[str, Any]]:
    if not trajectories:
        return []
    keep = max(1, int(round(len(trajectories) * max(0.0, min(1.0, float(elite_frac))))))
    return sorted(trajectories, key=lambda t: int(t["score"]), reverse=True)[:keep]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--out", required=True, help="elite SFT JSONL output")
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-moves", type=int, default=8)
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--action-format", choices=["x_bin", "x_rel", "action_token"], default="x_bin")
    ap.add_argument("--answer-style", choices=["json", "plain", "think"], default="json")
    ap.add_argument("--prompt-style", choices=["verbose", "compact"], default="verbose")
    ap.add_argument("--bins", type=int, default=101)
    ap.add_argument("--include-image-placeholder", action="store_true")
    ap.add_argument("--max-prompt-fruits", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--fallback", choices=["heuristic", "center"], default="heuristic")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dry-response", default=DEFAULT_DRY_RUN_RESPONSE)
    args = ap.parse_args()

    if args.dry_run:
        args.games = min(args.games, 3)
        args.max_moves = min(args.max_moves, 3)

    agent = QwenCoordinatePolicyAgent(
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        max_prompt_fruits=args.max_prompt_fruits,
        fallback=args.fallback,
        prompt_style=args.prompt_style,
        dry_run=args.dry_run,
        dry_run_response=args.dry_response,
        trust_remote_code=args.trust_remote_code,
    )

    t0 = time.time()
    trajectories = [
        _rollout(seed=int(args.seed) + i, agent=agent, max_moves=int(args.max_moves))
        for i in range(max(1, int(args.games)))
    ]
    elites = _elite_trajectories(trajectories, elite_frac=args.elite_frac)
    records = []
    for traj in elites:
        for move, sample in enumerate(traj["samples"]):
            records.append(record_from_state_action(
                sample["state"],
                x_rel=float(sample["x_rel"]),
                action_format=args.action_format,
                answer_style=args.answer_style,
                bins=args.bins,
                include_image_placeholder=args.include_image_placeholder,
                max_prompt_fruits=args.max_prompt_fruits,
                prompt_style=args.prompt_style,
                source="cem_elite:qwen_policy",
                extra={
                    "seed": int(traj["seed"]),
                    "move": int(move),
                    "trajectory_score": int(traj["score"]),
                    "policy_source": sample["source"],
                    "fallback_used": bool(sample["fallback_used"]),
                },
            ))
    count = write_jsonl(args.out, records)

    summary = {
        "out": args.out,
        "records": int(count),
        "games": int(args.games),
        "elite_games": len(elites),
        "scores": [int(t["score"]) for t in trajectories],
        "elite_scores": [int(t["score"]) for t in elites],
        "max_moves": int(args.max_moves),
        "action_format": args.action_format,
        "answer_style": args.answer_style,
        "prompt_style": args.prompt_style,
        "dry_run": bool(args.dry_run),
        "elapsed_sec": round(time.time() - t0, 3),
        "note": "Skeleton only: inspect elite data before starting a real LoRA round.",
    }
    if args.summary_out:
        directory = os.path.dirname(args.summary_out)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
