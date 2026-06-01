#!/usr/bin/env python3
"""Render lightweight Suika board PNGs from states or replay vectors.

This renderer intentionally uses Pillow instead of a pygame display.  It draws
the play area, danger line, current/next fruit labels, and approximate fruit
circles.  Replay-vector renders are approximate because flat replay does not
store velocities, collision flags, or hidden RNG state.
"""
import argparse
import json
import os
from typing import Any, Dict

import numpy as np

from common import KILLY, PLAY_BOT, PLAY_LEFT, PLAY_RIGHT, PLAY_TOP, config
from qwen_vla_dataset import approximate_state_from_vec


FRUIT_COLORS = [
    (236, 70, 70),
    (244, 132, 54),
    (250, 196, 70),
    (156, 206, 71),
    (78, 184, 97),
    (72, 184, 174),
    (72, 145, 210),
    (112, 104, 204),
    (168, 92, 188),
    (217, 107, 166),
    (76, 163, 80),
]


def _load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError(
            "render_state_image.py requires Pillow. Install with: "
            "pip install pillow"
        ) from exc
    return Image, ImageDraw, ImageFont


def _read_json_state(text_or_path: str) -> Dict[str, Any]:
    if os.path.exists(text_or_path):
        with open(text_or_path, "r", encoding="utf-8") as f:
            text_or_path = f.read()
    obj = json.loads(text_or_path)
    if not isinstance(obj, dict):
        raise ValueError("state JSON must decode to an object")
    return obj


def _dry_run_state() -> Dict[str, Any]:
    from suika_env import SuikaEnv

    env = SuikaEnv(seed=0)
    state = env.get_state()
    for x in (520, 650, 590):
        state, _reward, done, _info = env.step(x)
        if done:
            break
    return state


def _state_from_npz(path: str, index: int, max_fruits: int, boundary_features: bool) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    if "states" in data.files:
        value = data["states"][index]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            return _read_json_state(value)
        if isinstance(value, dict):
            return value
    if "state_json" in data.files:
        return _read_json_state(data["state_json"][index])
    if "vecs" not in data.files:
        raise ValueError("NPZ must contain states/state_json or vecs")
    return approximate_state_from_vec(
        data["vecs"][index],
        max_fruits=max_fruits,
        boundary_features=boundary_features,
    )


def render_state_png(
    state: Dict[str, Any],
    out: str,
    width: int = 448,
    height: int = 590,
    margin: int = 28,
) -> None:
    Image, ImageDraw, ImageFont = _load_pillow()
    canvas_w = int(width) + 2 * int(margin)
    canvas_h = int(height) + 2 * int(margin) + 48
    image = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 238))
    draw = ImageDraw.Draw(image)

    play = state.get("play_area") or {}
    left = float(play.get("left", PLAY_LEFT))
    right = float(play.get("right", PLAY_RIGHT))
    top = float(play.get("top", PLAY_TOP))
    bottom = float(play.get("bottom", PLAY_BOT))
    danger_y = float(play.get("danger_y", KILLY))
    sx = float(width) / max(right - left, 1e-9)
    sy = float(height) / max(bottom - top, 1e-9)

    def tx(x: float) -> float:
        return margin + (float(x) - left) * sx

    def ty(y: float) -> float:
        return margin + (float(y) - top) * sy

    board_box = (margin, margin, margin + width, margin + height)
    draw.rectangle(board_box, fill=(255, 252, 240), outline=(45, 45, 45), width=2)
    y_danger = ty(danger_y)
    draw.line((margin, y_danger, margin + width, y_danger), fill=(220, 70, 70), width=2)
    draw.text((margin + 6, y_danger + 4), "danger", fill=(180, 40, 40))

    for fruit in sorted(state.get("fruits") or [], key=lambda f: float(f.get("radius", 0.0)), reverse=True):
        ftype = int(fruit.get("type", 0)) % len(FRUIT_COLORS)
        x = tx(float(fruit.get("x", left)))
        y = ty(float(fruit.get("y", bottom)))
        r = max(2.0, float(fruit.get("radius", 1.0)) * (sx + sy) * 0.5)
        color = FRUIT_COLORS[ftype]
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(40, 40, 40), width=1)
        label = str(int(fruit.get("type", 0)))
        draw.text((x - 4, y - 6), label, fill=(20, 20, 20))

    current = state.get("current") or {}
    next_fruit = state.get("next") or {}
    footer_y = margin + height + 10
    footer = (
        "score=%s  current=%s:%s  next=%s:%s"
        % (
            int(state.get("score", 0)),
            int(current.get("type", 0)),
            current.get("name", ""),
            int(next_fruit.get("type", 0)),
            next_fruit.get("name", ""),
        )
    )
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((margin, footer_y), footer, fill=(30, 30, 30), font=font)

    directory = os.path.dirname(out)
    if directory:
        os.makedirs(directory, exist_ok=True)
    image.save(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-json", default=None,
                    help="state JSON string or path to a JSON file")
    ap.add_argument("--input", default=None, help="replay NPZ")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-fruits", type=int, default=80)
    ap.add_argument("--boundary-features", action="store_true")
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--height", type=int, default=590)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.state_json:
        state = _read_json_state(args.state_json)
        source = "state_json"
    elif args.input:
        state = _state_from_npz(args.input, args.index, args.max_fruits, args.boundary_features)
        source = "npz"
    elif args.dry_run:
        state = _dry_run_state()
        source = "dry_run_env"
    else:
        raise SystemExit("Provide --state-json, --input, or --dry-run")

    render_state_png(state, args.out, width=args.width, height=args.height)
    print(json.dumps({
        "out": args.out,
        "source": source,
        "fruits": len(state.get("fruits") or []),
        "approx_from_replay_vec": bool(state.get("approx_from_replay_vec", False)),
        "limit": "Replay-vector renders omit velocity/collision/RNG state.",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
