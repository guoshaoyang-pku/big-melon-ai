#!/usr/bin/env python3
"""Lightweight rollout entry for the Qwen coordinate policy.

Use ``--dry-run`` to exercise prompt serialization, parsing, clamping, fallback,
and the Suika env loop without loading or downloading the HuggingFace model.
"""
import argparse
import json
import time
from typing import List

from qwen_policy_agent import (
    DEFAULT_DRY_RUN_RESPONSE,
    DEFAULT_MODEL_ID,
    QwenCoordinatePolicyAgent,
    parse_coordinate_text,
    state_to_prompt,
)


def _parse_seeds(text: str) -> List[int]:
    if "," in text:
        return [int(x) for x in text.split(",") if x.strip()]
    if ":" in text:
        start, stop = text.split(":", 1)
        return list(range(int(start), int(stop)))
    return list(range(int(text)))


def _parser_smoke(state):
    samples = [
        '{"x_rel":0.25}',
        '{"x_abs":99999}',
        "0.75",
        "not a coordinate",
    ]
    rows = []
    for text in samples:
        x_abs, x_rel, source = parse_coordinate_text(text, state)
        rows.append({
            "input": text,
            "parsed": x_abs is not None,
            "x_abs": x_abs,
            "x_rel": x_rel,
            "source": source,
        })
    return rows


def _rollout_seed(seed, agent, max_moves, print_prompts):
    from suika_env import SuikaEnv

    env = SuikaEnv(seed=seed)
    state = env.get_state()
    moves = []
    move = 0
    while not state["game_over"]:
        if max_moves and move >= max_moves:
            break
        decision = agent.decide_with_debug(state)
        if print_prompts and seed == 0 and move == 0:
            print(json.dumps({
                "prompt_preview": decision.prompt,
                "raw_model_text": decision.raw_text,
            }, ensure_ascii=False, indent=2))
        state, reward, done, info = env.step(decision.x_abs)
        moves.append({
            "move": move,
            "x_abs": decision.x_abs,
            "x_rel": decision.x_rel,
            "reward": float(reward),
            "score": int(info.get("score", state.get("score", 0))),
            "source": decision.source,
            "parsed": decision.parsed,
            "fallback_used": decision.fallback_used,
            "raw_text": decision.raw_text,
        })
        move += 1
        if done:
            break
    max_fruit = max((int(f["type"]) for f in state["fruits"]), default=-1)
    return {
        "seed": int(seed),
        "score": int(state["score"]),
        "moves": len(moves),
        "max_fruit_type": max_fruit,
        "game_over": bool(state["game_over"]),
        "trace": moves,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--seeds", default="0:1",
                    help="'N' for range(N), 'a:b' for range(a,b), or csv list")
    ap.add_argument("--max-moves", type=int, default=3,
                    help="debug cap; 0 plays until game_over")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip model loading and use --dry-response")
    ap.add_argument("--dry-response", default=DEFAULT_DRY_RUN_RESPONSE,
                    help="mock model text used only with --dry-run")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--device", default="auto",
                    help="auto, cpu, mps, cuda, cuda:0, ...")
    ap.add_argument("--dtype", default="auto",
                    help="auto, float32, float16, bfloat16")
    ap.add_argument("--max-prompt-fruits", type=int, default=24)
    ap.add_argument("--prompt-style", choices=["verbose", "compact"], default="verbose")
    ap.add_argument("--fallback", default="heuristic",
                    choices=["heuristic", "center"])
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--print-prompts", action="store_true",
                    help="print the first prompt and raw response")
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    agent = QwenCoordinatePolicyAgent(
        model_id=args.model_id,
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
    results = [
        _rollout_seed(seed, agent, args.max_moves, args.print_prompts)
        for seed in seeds
    ]
    elapsed = time.time() - t0

    parser_smoke = []
    if args.dry_run:
        from suika_env import SuikaEnv

        smoke_state = SuikaEnv(seed=seeds[0] if seeds else 0).get_state()
        parser_smoke = _parser_smoke(smoke_state)
        # Also build once explicitly so dry-run validates prompt construction
        # even if the rollout was skipped with an empty seed list.
        state_to_prompt(smoke_state, max_fruits=args.max_prompt_fruits)

    payload = {
        "model_id": args.model_id,
        "dry_run": bool(args.dry_run),
        "seeds": seeds,
        "max_moves": int(args.max_moves),
        "elapsed_sec": elapsed,
        "results": results,
        "parser_smoke": parser_smoke,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
