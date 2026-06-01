"""Continuous action-candidate helpers for future refined MCTS.

The current AlphaZero policy still trains on coarse bins. These helpers provide
the refinement layer described in the roadmap: start from coarse columns, then
add physically meaningful continuous x candidates around merges, gaps, edges,
and local jitters. They are side-effect free and can be plugged into search
experiments without changing replay format.
"""
import numpy as np

from common import PLAY_LEFT, PLAY_RIGHT, col_to_x


def _clip_x(x):
    return float(np.clip(float(x), PLAY_LEFT, PLAY_RIGHT))


def _dedupe(xs, min_gap=3.0):
    out = []
    for x in sorted(_clip_x(x) for x in xs):
        if not out or abs(x - out[-1]) >= min_gap:
            out.append(x)
    return out


def refined_x_candidates(state, K, policy=None, top_k=4, jitter=10.0,
                         max_candidates=24):
    """Return continuous x candidates for refined drop search.

    Sources:
    * policy top-k or central coarse bins,
    * positions near same-type fruits,
    * horizontal gaps between nearby fruits,
    * edges/corners,
    * small jitter around the strongest coarse candidate.
    """
    xs = []
    if policy is not None:
        p = np.asarray(policy, dtype=np.float64)
        cols = np.argsort(p)[::-1][:max(1, int(top_k))]
    else:
        mid = K // 2
        cols = np.array(sorted({mid, max(0, mid - 1), min(K - 1, mid + 1)}))
    for col in cols:
        xs.append(col_to_x(int(col), K))

    fruits = list(state.get("fruits") or [])
    cur_type = int(state.get("current", {}).get("type", 0))
    same = [f for f in fruits if int(f.get("type", -1)) == cur_type]
    for f in same:
        xs.extend([float(f["x"]), float(f["x"]) - jitter, float(f["x"]) + jitter])

    by_x = sorted(fruits, key=lambda f: float(f["x"]))
    for a, b in zip(by_x, by_x[1:]):
        ax, bx = float(a["x"]), float(b["x"])
        gap = bx - ax - float(a.get("radius", 0.0)) - float(b.get("radius", 0.0))
        if gap > 4.0:
            xs.append((ax + bx) / 2.0)

    xs.extend([PLAY_LEFT, PLAY_LEFT + jitter, PLAY_RIGHT - jitter, PLAY_RIGHT])
    if xs:
        best = xs[0]
        xs.extend([best - jitter, best + jitter, best - 0.5 * jitter,
                   best + 0.5 * jitter])
    return _dedupe(xs)[:max(1, int(max_candidates))]


def progressive_width(visits, min_width=4, growth=2.0, max_width=24):
    """Number of refined candidates allowed at a node by visit count."""
    width = int(min_width + growth * np.sqrt(max(0, int(visits))))
    return max(1, min(int(max_width), width))
