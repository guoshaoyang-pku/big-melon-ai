#!/usr/bin/env python3
"""Build Qwen VLA SFT JSONL for Suika coordinate policies.

The primary path converts existing teacher replay NPZ files (`vecs` + `pis`)
into HF-style chat records.  Because the historical flat replay stores encoded
features rather than full game snapshots, states reconstructed from `vecs` are
approximate: fruit x/y/type/radius and current/next fruit are preserved well
enough for text prompts and board sketches, while hidden physics state is not.
"""
import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from common import (
    KILLY,
    MAX_RADIUS,
    PLAY_BOT,
    PLAY_H,
    PLAY_LEFT,
    PLAY_RIGHT,
    PLAY_TOP,
    PLAY_W,
    SCORE_NORM,
    col_to_x,
    config,
    decode_flat_vec_to_tokens,
    input_dim,
    radius_of,
)


SCHEMA = "suika_qwen_vla_sft_v1"


def _clamp(value: float, lo: float, hi: float) -> float:
    if hi < lo:
        lo, hi = hi, lo
    return max(lo, min(hi, value))


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _round(value: Any, digits: int = 3) -> float:
    return round(float(value), digits)


def legal_x_bounds(state: Dict[str, Any]) -> Tuple[float, float]:
    current = state.get("current") or {}
    radius = _finite_float(current.get("radius"))
    if radius is None:
        radius = radius_of(int(current.get("type", 0)))
    left = float((state.get("play_area") or {}).get("left", PLAY_LEFT))
    right = float((state.get("play_area") or {}).get("right", PLAY_RIGHT))
    return left + radius, right - radius


def x_abs_to_rel(x_abs: float, state: Dict[str, Any]) -> float:
    left, right = legal_x_bounds(state)
    span = max(right - left, 1e-9)
    return _clamp((float(x_abs) - left) / span, 0.0, 1.0)


def x_rel_to_abs(x_rel: float, state: Dict[str, Any]) -> float:
    left, right = legal_x_bounds(state)
    return left + _clamp(float(x_rel), 0.0, 1.0) * (right - left)


def _fruit_name(ftype: int) -> str:
    try:
        return str(config.fruit_names[int(ftype)])
    except Exception:
        return str(int(ftype))


def approximate_state_from_vec(
    vec: np.ndarray,
    max_fruits: int,
    boundary_features: bool = False,
) -> Dict[str, Any]:
    """Approximate a structured state from the flat replay vector layout."""

    global_feats, fruit_tokens, mask = decode_flat_vec_to_tokens(
        vec, max_fruits=max_fruits, boundary_features=boundary_features
    )
    current_type = int(np.clip(round(float(global_feats[0]) * 10.0), 0, 10))
    next_type = int(np.clip(round(float(global_feats[1]) * 10.0), 0, 10))
    score = int(max(0.0, float(global_feats[4]) * SCORE_NORM)) if len(global_feats) > 4 else 0

    fruits: List[Dict[str, Any]] = []
    for tok, valid in zip(fruit_tokens, mask):
        if not bool(valid):
            continue
        ftype = int(np.clip(round(float(tok[3]) * 10.0), 0, 10))
        radius = float(tok[2]) * MAX_RADIUS
        if radius <= 0.0:
            radius = radius_of(ftype)
        fruits.append({
            "type": ftype,
            "name": _fruit_name(ftype),
            "x": float(PLAY_LEFT + float(tok[0]) * PLAY_W),
            "y": float(PLAY_TOP + float(tok[1]) * PLAY_H),
            "radius": float(radius),
        })

    return {
        "fruits": fruits,
        "current": {
            "type": current_type,
            "name": _fruit_name(current_type),
            "radius": radius_of(current_type),
        },
        "next": {
            "type": next_type,
            "name": _fruit_name(next_type),
        },
        "score": score,
        "game_over": False,
        "play_area": {
            "left": PLAY_LEFT,
            "right": PLAY_RIGHT,
            "top": PLAY_TOP,
            "bottom": PLAY_BOT,
            "danger_y": KILLY,
        },
        "approx_from_replay_vec": True,
    }


def _compact_state_payload(
    state: Dict[str, Any],
    max_fruits: int = 32,
    include_image_placeholder: bool = False,
) -> Dict[str, Any]:
    left, right = legal_x_bounds(state)
    fruits = sorted(
        state.get("fruits") or [],
        key=lambda f: (float(f.get("y", PLAY_BOT)) - float(f.get("radius", 0.0)),
                       -float(f.get("radius", 0.0)),
                       float(f.get("x", PLAY_LEFT))),
    )[:max_fruits]
    return {
        "image": (
            "BOARD_IMAGE_PLACEHOLDER; use image_path when a rendered board PNG is attached"
            if include_image_placeholder else None
        ),
        "current": {
            "type": int((state.get("current") or {}).get("type", 0)),
            "name": (state.get("current") or {}).get("name", ""),
            "radius": _round((state.get("current") or {}).get("radius", radius_of(0))),
        },
        "next": {
            "type": int((state.get("next") or {}).get("type", 0)),
            "name": (state.get("next") or {}).get("name", ""),
        },
        "score": int(state.get("score", 0)),
        "play_area": {
            "left": _round((state.get("play_area") or {}).get("left", PLAY_LEFT)),
            "right": _round((state.get("play_area") or {}).get("right", PLAY_RIGHT)),
            "top": _round((state.get("play_area") or {}).get("top", PLAY_TOP)),
            "bottom": _round((state.get("play_area") or {}).get("bottom", PLAY_BOT)),
            "danger_y": _round((state.get("play_area") or {}).get("danger_y", KILLY)),
            "legal_left": _round(left),
            "legal_right": _round(right),
        },
        "fruits_top_first": [{
            "type": int(f.get("type", 0)),
            "name": f.get("name", ""),
            "x": _round(f.get("x", 0.0)),
            "y": _round(f.get("y", 0.0)),
            "radius": _round(f.get("radius", 0.0)),
        } for f in fruits],
        "approx_from_replay_vec": bool(state.get("approx_from_replay_vec", False)),
    }


def state_to_vla_prompt(
    state: Dict[str, Any],
    action_format: str = "x_bin",
    bins: int = 101,
    include_image_placeholder: bool = False,
    max_prompt_fruits: int = 32,
    prompt_style: str = "verbose",
) -> str:
    if prompt_style == "compact":
        return compact_state_prompt(
            state,
            action_format=action_format,
            bins=bins,
            max_prompt_fruits=max_prompt_fruits,
        )
    payload = _compact_state_payload(
        state,
        max_fruits=max_prompt_fruits,
        include_image_placeholder=include_image_placeholder,
    )
    if action_format == "x_bin":
        answer_spec = (
            '{"x_bin":42} where x_bin is an integer in [0,%d] mapped left-to-right'
            % (int(bins) - 1)
        )
    elif action_format == "action_token":
        answer_spec = "ACTION_N where N is an integer in [0,%d]" % (int(bins) - 1)
    else:
        answer_spec = '{"x_rel":0.42} where x_rel is a float in [0,1]'
    return_prefix = (
        "Return exactly one action token and no explanation: "
        if action_format == "action_token"
        else "Return exactly one strict JSON object and no explanation: "
    )
    return (
        "You are a Suika visual-language-action policy. Choose the horizontal "
        "drop location for the current fruit.\n"
        "Use the board image if present; otherwise use the structured state text. "
        "Coordinates use the legal interval between legal_left and legal_right.\n"
        f"{return_prefix}"
        f"{answer_spec}.\n"
        "STATE="
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def compact_state_prompt(
    state: Dict[str, Any],
    action_format: str = "x_bin",
    bins: int = 101,
    max_prompt_fruits: int = 16,
) -> str:
    """Very short, self-contained state encoding for local LoRA training.

    The assistant should emit a tiny reasoning tag plus the action, e.g.
    ``T=merge A=42``. Keeping all fields positional-ish reduces context length
    while preserving current/next fruit, legal bounds, danger line, and fruit
    geometry.
    """
    action_word = "ACTION_N" if action_format == "action_token" else "42"
    payload = _compact_state_payload(state, max_fruits=max_prompt_fruits)
    cur = payload["current"]
    nxt = payload["next"]
    area = payload["play_area"]
    fruits = payload["fruits_top_first"]
    fruit_bits = []
    for f in fruits:
        fruit_bits.append("%d,%d,%d,%d" % (
            int(f["type"]),
            int(round(float(f["x"]))),
            int(round(float(f["y"]))),
            int(round(float(f["radius"]))),
        ))
    ftxt = ";".join(fruit_bits) if fruit_bits else "-"
    return (
        "Suika. Choose the drop bin. Output only the action.\n"
        "Format: %s where bin range is 0-%d.\n"
        "C%d N%d S%d X%d,%d D%d F=%s"
        % (
            action_word,
            int(bins) - 1,
            int(cur["type"]),
            int(nxt["type"]),
            int(payload["score"]),
            int(round(float(area["legal_left"]))),
            int(round(float(area["legal_right"]))),
            int(round(float(area["danger_y"]))),
            ftxt,
        )
    )


def action_answer_from_x_rel(x_rel: float, action_format: str, bins: int) -> Dict[str, Any]:
    rel = _clamp(float(x_rel), 0.0, 1.0)
    if action_format == "x_rel":
        return {"x_rel": round(rel, 4)}
    x_bin = int(round(rel * (int(bins) - 1)))
    return {"x_bin": int(_clamp(x_bin, 0, int(bins) - 1))}


def _thinking_tag(state: Dict[str, Any], x_rel: float) -> str:
    """Small supervised rationale tag, not a long chain-of-thought."""
    x_abs = x_rel_to_abs(x_rel, state)
    cur_type = int((state.get("current") or {}).get("type", 0))
    fruits = list(state.get("fruits") or [])
    same = [f for f in fruits if int(f.get("type", -1)) == cur_type]
    if same:
        nearest = min(abs(float(f.get("x", x_abs)) - x_abs) for f in same)
        if nearest <= 80.0:
            return "merge"
    top = min((float(f.get("y", PLAY_BOT)) - float(f.get("radius", 0.0)) for f in fruits), default=PLAY_BOT)
    if top < KILLY + 90.0:
        return "danger"
    if x_rel < 0.2:
        return "left"
    if x_rel > 0.8:
        return "right"
    return "safe"


def action_answer_text(
    state: Dict[str, Any],
    x_rel: float,
    action_format: str,
    bins: int,
    answer_style: str,
) -> str:
    answer = action_answer_from_x_rel(x_rel, action_format=action_format, bins=bins)
    if answer_style == "json":
        return json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
    if action_format == "x_rel":
        action_text = str(answer["x_rel"])
    else:
        action_text = str(answer["x_bin"])
    if action_format == "action_token":
        action_text = "ACTION_%02d" % int(answer["x_bin"])
    if answer_style == "plain":
        return action_text
    if answer_style == "think":
        return "T=%s A=%s" % (_thinking_tag(state, x_rel), action_text)
    raise ValueError("unknown answer_style: %s" % answer_style)


def record_from_state_action(
    state: Dict[str, Any],
    x_rel: float,
    action_format: str = "x_bin",
    bins: int = 101,
    include_image_placeholder: bool = False,
    max_prompt_fruits: int = 32,
    source: str = "unknown",
    image_path: Optional[str] = None,
    prompt_style: str = "verbose",
    answer_style: str = "json",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prompt = state_to_vla_prompt(
        state,
        action_format=action_format,
        bins=bins,
        include_image_placeholder=include_image_placeholder,
        max_prompt_fruits=max_prompt_fruits,
        prompt_style=prompt_style,
    )
    answer = action_answer_from_x_rel(x_rel, action_format=action_format, bins=bins)
    answer_text = action_answer_text(
        state,
        x_rel=x_rel,
        action_format=action_format,
        bins=bins,
        answer_style=answer_style,
    )
    record: Dict[str, Any] = {
        "schema": SCHEMA,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You output only the Suika drop action token requested by the user."
                ),
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer_text},
        ],
        "prompt": prompt,
        "answer": answer,
        "action_format": action_format,
        "answer_style": answer_style,
        "prompt_style": prompt_style,
        "bins": int(bins),
        "x_rel": round(_clamp(float(x_rel), 0.0, 1.0), 6),
        "x_abs": round(x_rel_to_abs(x_rel, state), 3),
        "source": source,
        "image_path": image_path,
        "state": _compact_state_payload(
            state,
            max_fruits=max_prompt_fruits,
            include_image_placeholder=include_image_placeholder,
        ),
    }
    if extra:
        record.update(extra)
    return record


def _npz_len(data: Any) -> int:
    if "size" in data.files:
        return int(np.asarray(data["size"]).reshape(-1)[0])
    for key in ("x_rels", "x_rel", "x_abs", "xs", "actions", "pis", "vecs"):
        if key in data.files:
            return len(data[key])
    return 0


def _read_json_state(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            obj = json.loads(value)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None
    return None


def _state_for_npz_row(
    data: Any,
    index: int,
    max_fruits: int,
    boundary_features: bool,
) -> Dict[str, Any]:
    for key in ("states", "state_json", "states_json"):
        if key in data.files:
            state = _read_json_state(data[key][index])
            if state is not None:
                return state
    if "vecs" not in data.files:
        raise ValueError("NPZ must contain states/state_json or vecs")
    return approximate_state_from_vec(
        data["vecs"][index],
        max_fruits=max_fruits,
        boundary_features=boundary_features,
    )


def _field_at(data: Any, keys: Iterable[str], index: int) -> Optional[float]:
    for key in keys:
        if key not in data.files:
            continue
        arr = data[key]
        if len(arr) <= index:
            continue
        value = _finite_float(np.asarray(arr[index]).reshape(-1)[0])
        if value is not None:
            return value
    return None


def _bin_to_x_rel(col: float, num_bins: int, state: Dict[str, Any]) -> float:
    x_abs = col_to_x(int(round(col)), int(num_bins))
    left, right = legal_x_bounds(state)
    x_abs = _clamp(x_abs, left, right)
    return x_abs_to_rel(x_abs, state)


def _policy_to_x_rel(
    pi: np.ndarray,
    state: Dict[str, Any],
    reduction: str = "expectation",
) -> Tuple[float, str]:
    p = np.asarray(pi, dtype=np.float64).reshape(-1)
    if p.size == 0 or not np.isfinite(p).any() or float(np.maximum(p, 0.0).sum()) <= 0.0:
        return 0.5, "policy:fallback_center"
    p = np.maximum(p, 0.0)
    p = p / max(float(p.sum()), 1e-12)
    if reduction == "argmax":
        col = int(np.argmax(p))
        return _bin_to_x_rel(col, p.size, state), "policy:argmax"
    xs = np.array([col_to_x(i, p.size) for i in range(p.size)], dtype=np.float64)
    x_abs = float(np.sum(p * xs))
    left, right = legal_x_bounds(state)
    return x_abs_to_rel(_clamp(x_abs, left, right), state), "policy:expectation"


def _action_for_npz_row(
    data: Any,
    index: int,
    state: Dict[str, Any],
    policy_reduction: str,
) -> Tuple[float, str]:
    rel = _field_at(data, ("x_rel", "x_rels", "rel_x", "rel_xs"), index)
    if rel is not None:
        return _clamp(rel, 0.0, 1.0), "metadata:x_rel"

    x_abs = _field_at(data, ("x_abs", "x", "xs", "action_x", "action_abs"), index)
    if x_abs is not None:
        if 0.0 <= x_abs <= 1.0:
            return _clamp(x_abs, 0.0, 1.0), "metadata:x_rel_like"
        return x_abs_to_rel(x_abs, state), "metadata:x_abs"

    col = _field_at(data, ("col", "cols", "action_bin", "action_bins"), index)
    if col is not None:
        k = int(data["pis"].shape[1]) if "pis" in data.files and data["pis"].ndim == 2 else 101
        return _bin_to_x_rel(col, k, state), "metadata:bin"

    if "actions" in data.files:
        action = _finite_float(np.asarray(data["actions"][index]).reshape(-1)[0])
        if action is not None:
            if 0.0 <= action <= 1.0:
                return _clamp(action, 0.0, 1.0), "metadata:actions_rel"
            if PLAY_LEFT - 1.0 <= action <= PLAY_RIGHT + 1.0:
                return x_abs_to_rel(action, state), "metadata:actions_abs"

    if "pis" in data.files:
        return _policy_to_x_rel(data["pis"][index], state, reduction=policy_reduction)

    return 0.5, "fallback:center"


def _load_meta(data: Any) -> Dict[str, Any]:
    if "meta" not in data.files:
        return {}
    raw = data["meta"]
    try:
        text = str(np.asarray(raw).reshape(-1)[0])
        obj = json.loads(text)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _read_config(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    return obj if isinstance(obj, dict) else {}


def _infer_layout(args: argparse.Namespace, data: Any) -> Tuple[int, bool]:
    meta = _load_meta(data)
    cfg = _read_config(args.config)
    boundary = bool(args.boundary_features)
    if "boundary_features" in meta:
        boundary = bool(meta["boundary_features"])
    elif "boundary_features" in cfg:
        boundary = bool(cfg["boundary_features"])

    max_fruits = int(args.max_fruits or meta.get("max_fruits") or cfg.get("max_fruits") or 80)
    if "vecs" not in data.files:
        return max_fruits, boundary

    dim = int(data["vecs"].shape[1])
    if dim == input_dim(max_fruits, boundary):
        return max_fruits, boundary
    for candidate_boundary in (boundary, False, True):
        gdim = 13 if candidate_boundary else 9
        if dim >= gdim and (dim - gdim) % 4 == 0:
            return (dim - gdim) // 4, candidate_boundary
    raise ValueError("Cannot infer replay vec layout for dim=%d" % dim)


def iter_npz_records(args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    data = np.load(args.input, allow_pickle=True)
    max_fruits, boundary_features = _infer_layout(args, data)
    n = min(_npz_len(data), int(args.max_samples or _npz_len(data)))
    for i in range(n):
        state = _state_for_npz_row(data, i, max_fruits, boundary_features)
        x_rel, action_source = _action_for_npz_row(
            data, i, state, policy_reduction=args.policy_reduction
        )
        yield record_from_state_action(
            state,
            x_rel=x_rel,
            action_format=args.action_format,
            bins=args.bins,
            include_image_placeholder=args.include_image_placeholder,
            max_prompt_fruits=args.max_prompt_fruits,
            prompt_style=args.prompt_style,
            answer_style=args.answer_style,
            source="npz:%s:%s" % (os.path.basename(args.input), action_source),
            extra={"sample_index": int(i)},
        )


def _center_action(state: Dict[str, Any]) -> float:
    left, right = legal_x_bounds(state)
    return (left + right) / 2.0


def iter_rollout_records(args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    from suika_env import SuikaEnv

    if args.rollout_teacher == "heuristic":
        from ai_agent import HeuristicLookaheadAgent

        teacher = HeuristicLookaheadAgent(num_columns=14, lookahead_steps=80, seed=args.seed)
    else:
        teacher = None

    made = 0
    for game in range(max(1, int(args.rollout_games))):
        env = SuikaEnv(seed=int(args.seed) + game)
        state = env.get_state()
        move = 0
        while not state.get("game_over"):
            if args.max_samples and made >= args.max_samples:
                return
            if args.rollout_max_moves and move >= args.rollout_max_moves:
                break
            if teacher is None:
                x_abs = _center_action(state)
                action_source = "rollout:center"
            else:
                x_abs = float(teacher.decide(state))
                action_source = "rollout:heuristic"
            x_rel = x_abs_to_rel(x_abs, state)
            yield record_from_state_action(
                state,
                x_rel=x_rel,
                action_format=args.action_format,
                bins=args.bins,
                include_image_placeholder=args.include_image_placeholder,
                max_prompt_fruits=args.max_prompt_fruits,
                prompt_style=args.prompt_style,
                answer_style=args.answer_style,
                source=action_source,
                extra={"game": int(game), "move": int(move)},
            )
            state, _reward, done, _info = env.step(x_abs)
            made += 1
            move += 1
            if done:
                break


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> int:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            if rec.get("answer_style", "json") == "json":
                # Ensure the assistant answer is strict JSON before writing.
                json.loads(rec["messages"][-1]["content"])
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="teacher replay NPZ; if omitted, use rollout")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--action-format", choices=["x_bin", "x_rel", "action_token"], default="x_bin")
    ap.add_argument("--answer-style", choices=["json", "plain", "think"], default="json")
    ap.add_argument("--prompt-style", choices=["verbose", "compact"], default="verbose")
    ap.add_argument("--bins", type=int, default=101)
    ap.add_argument("--include-image-placeholder", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="keep generation tiny and avoid expensive teachers")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--max-fruits", type=int, default=0,
                    help="override replay layout inference")
    ap.add_argument("--boundary-features", action="store_true")
    ap.add_argument("--policy-reduction", choices=["expectation", "argmax"], default="expectation")
    ap.add_argument("--max-prompt-fruits", type=int, default=32)
    ap.add_argument("--rollout-games", type=int, default=1)
    ap.add_argument("--rollout-max-moves", type=int, default=0)
    ap.add_argument("--rollout-teacher", choices=["center", "heuristic"], default="center")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.bins < 2:
        raise ValueError("--bins must be >= 2")
    if args.dry_run:
        if not args.max_samples:
            args.max_samples = 3
        if not args.rollout_max_moves:
            args.rollout_max_moves = args.max_samples
        args.rollout_teacher = "center"

    if args.input:
        records = iter_npz_records(args)
    else:
        records = iter_rollout_records(args)
    count = write_jsonl(args.out, records)
    print(json.dumps({
        "out": args.out,
        "samples": count,
        "action_format": args.action_format,
        "bins": int(args.bins),
        "source": "npz" if args.input else "rollout",
        "dry_run": bool(args.dry_run),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
