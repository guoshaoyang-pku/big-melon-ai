#!/usr/bin/env python3
"""Evaluate a Qwen VLA LoRA adapter on short Suika rollouts.

Use --dry-run to validate prompt, parser, adapter CLI plumbing, and env stepping
without loading the base model or adapter weights.
"""
import argparse
import json
import time
from typing import List

from qwen_policy_agent import (
    DEFAULT_DRY_RUN_RESPONSE,
    DEFAULT_MODEL_ID,
    QwenCoordinatePolicyAgent,
)


def _parse_seeds(text: str) -> List[int]:
    if "," in text:
        return [int(x) for x in text.split(",") if x.strip()]
    if ":" in text:
        start, stop = text.split(":", 1)
        return list(range(int(start), int(stop)))
    return list(range(int(text)))


def _rollout(seed: int, agent: QwenCoordinatePolicyAgent, max_moves: int) -> dict:
    from suika_env import SuikaEnv

    env = SuikaEnv(seed=seed)
    state = env.get_state()
    trace = []
    move = 0
    while not state.get("game_over"):
        if max_moves and move >= max_moves:
            break
        decision = agent.decide_with_debug(state)
        state, reward, done, info = env.step(decision.x_abs)
        trace.append({
            "move": int(move),
            "x_abs": round(float(decision.x_abs), 3),
            "x_rel": round(float(decision.x_rel), 4),
            "reward": float(reward),
            "score": int(info.get("score", state.get("score", 0))),
            "source": decision.source,
            "parsed": bool(decision.parsed),
            "fallback_used": bool(decision.fallback_used),
            "raw_text": decision.raw_text,
        })
        move += 1
        if done:
            break
    return {
        "seed": int(seed),
        "score": int(state.get("score", 0)),
        "moves": len(trace),
        "game_over": bool(state.get("game_over", False)),
        "trace": trace,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--adapter-dir", default=None,
                    help="LoRA adapter directory from train_qwen_vla_lora.py")
    ap.add_argument("--seeds", default="0:1")
    ap.add_argument("--max-moves", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--fallback", choices=["heuristic", "center"], default="heuristic")
    ap.add_argument("--max-prompt-fruits", type=int, default=24)
    ap.add_argument("--prompt-style", choices=["verbose", "compact"], default="verbose")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dry-response", default=DEFAULT_DRY_RUN_RESPONSE)
    args = ap.parse_args()

    if not args.dry_run and not args.adapter_dir:
        raise SystemExit("Provide --adapter-dir for real adapter eval, or use --dry-run")

    seeds = _parse_seeds(args.seeds)
    agent = QwenCoordinatePolicyAgent(
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        max_prompt_fruits=args.max_prompt_fruits,
        prompt_style=args.prompt_style,
        fallback=args.fallback,
        dry_run=args.dry_run,
        dry_run_response=args.dry_response,
        trust_remote_code=args.trust_remote_code,
    )
    t0 = time.time()
    results = [_rollout(seed, agent, max_moves=args.max_moves) for seed in seeds]
    print(json.dumps({
        "model_id": args.model_id,
        "adapter_dir": args.adapter_dir,
        "dry_run": bool(args.dry_run),
        "seeds": seeds,
        "max_moves": int(args.max_moves),
        "elapsed_sec": round(time.time() - t0, 3),
        "results": results,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
