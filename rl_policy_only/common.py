"""Shared utilities for the Suika AlphaZero pipeline.

Handles sys.path / cwd setup so that the part2 game modules import cleanly,
defines geometry / fruit constants, and provides both the historical flat state
encoder and the token views used by the Transformer policy/value network.
"""
import os
import sys

# --- locate project layout -------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # .../suika/rl
_ROOT = os.path.dirname(_THIS_DIR)                          # .../suika
_PART2 = os.path.join(_ROOT, "part2")                       # .../suika/part2

# part2/config.py opens "part2/config.yaml" and loads blits relative to the
# suika root, so we must be on that cwd before importing it. We also force a
# headless SDL driver so importing pygame never opens a window.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
if _PART2 not in sys.path:
    sys.path.insert(0, _PART2)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
os.chdir(_ROOT)

import numpy as np  # noqa: E402

# Importing config triggers pygame asset loading (headless). Done once here so
# every other module can `from common import config, ...`.
from config import config, CollisionTypes  # noqa: E402

# --- geometry / fruit constants --------------------------------------------
PLAY_LEFT = config.pad.left          # 415
PLAY_RIGHT = config.pad.right        # 863
PLAY_TOP = config.pad.top            # 85
PLAY_BOT = config.pad.bot            # 675
KILLY = config.pad.killy             # 170
PLAY_W = PLAY_RIGHT - PLAY_LEFT      # 448
PLAY_H = PLAY_BOT - PLAY_TOP         # 590

NUM_FRUIT_TYPES = 11                 # cherry .. watermelon (0..10)
SPAWN_TYPES = 5                      # PreParticle samples uniformly from 0..4
MAX_RADIUS = config[10, "radius"]    # watermelon radius = 125

# Reward / value normalisation. Heuristic baseline ~1300, strong play a few k.
SCORE_NORM = 2000.0

GLOBAL_FEATS = 9                     # legacy global feature block length
BOUNDARY_EXTRA_FEATS = 4             # opt-in wall/death-line features
FRUIT_FEATS = 4                      # x_norm, y_norm, radius_norm, type_norm


def use_boundary_features(cfg_or_flag=False):
    """Return whether the optional boundary-aware flat layout is enabled."""
    if isinstance(cfg_or_flag, dict):
        return bool(cfg_or_flag.get("boundary_features", False))
    return bool(cfg_or_flag)


def global_feat_count(boundary_features=False):
    return GLOBAL_FEATS + (BOUNDARY_EXTRA_FEATS if boundary_features else 0)


def col_to_x(col, K):
    """Centre x (pixels) of discrete column ``col`` out of ``K`` columns."""
    col = int(np.clip(col, 0, K - 1))
    return PLAY_LEFT + PLAY_W * (col + 0.5) / K


def radius_of(ftype):
    return float(config[int(ftype) % NUM_FRUIT_TYPES, "radius"])


def _top_y(fruits):
    """Top-most edge y over all fruits (small == near ceiling == dangerous)."""
    if not fruits:
        return float(PLAY_BOT)
    return min(f["y"] - f["radius"] for f in fruits)


def _global_features(state, max_fruits, boundary_features=False):
    fruits = state["fruits"]
    cur_t = float(state["current"]["type"])
    nxt_t = float(state["next"]["type"])
    cur_r = float(state["current"].get("radius", radius_of(cur_t)))
    n = len(fruits)

    if fruits:
        ys = np.array([f["y"] for f in fruits], dtype=np.float32)
        rs = np.array([f["radius"] for f in fruits], dtype=np.float32)
        mean_y = float(ys.mean())
        max_r = float(rs.max())
        # crude fill fraction = sum of fruit areas / play-area box
        fill = float(np.sum(np.pi * rs ** 2) / (PLAY_W * PLAY_H))
    else:
        mean_y, max_r, fill = float(PLAY_BOT), 0.0, 0.0

    feats = [
        cur_t / 10.0,
        nxt_t / 10.0,
        cur_r / MAX_RADIUS,
        n / float(max_fruits),
        float(state["score"]) / SCORE_NORM,
        (_top_y(fruits) - PLAY_TOP) / PLAY_H,
        (mean_y - PLAY_TOP) / PLAY_H,
        max_r / MAX_RADIUS,
        min(fill, 2.0),
    ]
    if boundary_features:
        if fruits:
            left_clear = min(max(0.0, float(f["x"]) - float(f["radius"]) - PLAY_LEFT)
                             for f in fruits)
            right_clear = min(max(0.0, PLAY_RIGHT - (float(f["x"]) + float(f["radius"])))
                              for f in fruits)
            bottom_clear = min(max(0.0, PLAY_BOT - (float(f["y"]) + float(f["radius"])))
                               for f in fruits)
        else:
            left_clear = right_clear = PLAY_W
            bottom_clear = PLAY_H
        top_margin = max(0.0, _top_y(fruits) - KILLY)
        legal_span = max(0.0, (PLAY_W - 2.0 * cur_r) / PLAY_W)
        feats.extend([
            min(left_clear, PLAY_W) / PLAY_W,
            min(right_clear, PLAY_W) / PLAY_W,
            min(top_margin, PLAY_H) / PLAY_H,
            0.5 * min(bottom_clear, PLAY_H) / PLAY_H + 0.5 * legal_span,
        ])
    return np.array(feats, dtype=np.float32)


def _fruit_tokens(state, max_fruits):
    tokens = np.zeros((max_fruits, FRUIT_FEATS), dtype=np.float32)
    mask = np.zeros((max_fruits,), dtype=np.bool_)
    fr = sorted(state["fruits"], key=lambda f: (f["y"], f["x"]))[:max_fruits]
    for i, f in enumerate(fr):
        tokens[i, 0] = (f["x"] - PLAY_LEFT) / PLAY_W
        tokens[i, 1] = (f["y"] - PLAY_TOP) / PLAY_H
        tokens[i, 2] = f["radius"] / MAX_RADIUS
        tokens[i, 3] = f["type"] / 10.0
        mask[i] = True
    return tokens, mask


def encode_state_tokens(state, K, max_fruits, boundary_features=False):
    """Return ``(global_feats, fruit_tokens, fruit_mask)`` for object models.

    ``fruit_tokens`` has shape ``(max_fruits, 4)`` with x/y/radius/type
    normalised to the same layout as the historical flat encoder. ``fruit_mask``
    is True only for real fruits, so cherry (type 0) remains distinguishable
    from zero padding.
    """
    del K  # the state layout is intentionally independent of action count.
    g = _global_features(state, max_fruits, boundary_features=boundary_features)
    tokens, mask = _fruit_tokens(state, max_fruits)
    return g, tokens, mask


def encode_state(state, K, max_fruits, boundary_features=False):
    """Object-centric flat state vector: globals + max_fruits*4 floats.

    Layout (MattJacobs30-style):
      globals[9]: current type, next type, current radius, #fruits, score,
                  top-y (danger), mean-y, max radius present, fill fraction.
      optional globals[4]: left-wall clearance, right-wall clearance,
                  death-line margin, bottom/legal-span mix.
      per fruit[4]: x_norm, y_norm, radius_norm, type_norm.
    Fruits are sorted by (y, x) and zero-padded / truncated to ``max_fruits``.
    The legacy layout remains the default so existing replay/checkpoints stay
    reusable; set ``boundary_features=True`` only for new policy-only runs.
    """
    g, tokens, _mask = encode_state_tokens(
        state, K, max_fruits, boundary_features=boundary_features)
    return np.concatenate([g, tokens.reshape(-1)]).astype(np.float32)


def decode_flat_vec_to_tokens(vec, max_fruits, boundary_features=False):
    """Convert old flat replay vectors into ``(global, fruit_tokens, mask)``.

    Accepts one vector ``[D]`` or a batch ``[B, D]``. Padding is detected by
    ``radius_norm > 0`` so cherry tokens (type_norm == 0) are preserved.
    """
    arr = np.asarray(vec, dtype=np.float32)
    one = arr.ndim == 1
    if one:
        arr = arr[None, :]
    expected = input_dim(max_fruits, boundary_features=boundary_features)
    if arr.shape[-1] != expected:
        raise ValueError("flat vec dim %d != expected %d" % (arr.shape[-1], expected))
    gdim = global_feat_count(boundary_features)
    global_feats = arr[:, :gdim]
    fruit_tokens = arr[:, gdim:].reshape(arr.shape[0], max_fruits, FRUIT_FEATS)
    fruit_mask = fruit_tokens[:, :, 2] > 0.0
    if one:
        return global_feats[0], fruit_tokens[0], fruit_mask[0]
    return global_feats, fruit_tokens, fruit_mask


def mirror_flat_vec(vec, max_fruits, boundary_features=False):
    """Mirror a flat encoded state horizontally without activating padding."""
    out = np.asarray(vec, dtype=np.float32).copy()
    gdim = global_feat_count(boundary_features)
    if boundary_features:
        out[GLOBAL_FEATS + 0], out[GLOBAL_FEATS + 1] = (
            out[GLOBAL_FEATS + 1], out[GLOBAL_FEATS + 0])
    tokens = out[gdim:].reshape(max_fruits, FRUIT_FEATS)
    valid = tokens[:, 2] > 0.0
    tokens[valid, 0] = 1.0 - tokens[valid, 0]
    tokens[~valid] = 0.0
    return out.astype(np.float32)


def input_dim(max_fruits, boundary_features=False):
    return global_feat_count(boundary_features) + max_fruits * FRUIT_FEATS
