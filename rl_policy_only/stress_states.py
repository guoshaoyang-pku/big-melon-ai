"""Synthetic boundary/stress states for policy and teacher diagnostics."""
import copy

import numpy as np

from common import KILLY, PLAY_BOT, PLAY_LEFT, PLAY_RIGHT, PLAY_TOP, config, radius_of


def _fruit(x, y, t):
    return {
        "x": float(x),
        "y": float(y),
        "radius": float(radius_of(t)),
        "type": int(t),
        "name": config.fruit_names[int(t)],
    }


def _state(name, fruits, current=0, nxt=1, score=0):
    return {
        "name": name,
        "fruits": list(fruits),
        "current": {
            "type": int(current),
            "name": config.fruit_names[int(current)],
            "radius": float(radius_of(current)),
        },
        "next": {
            "type": int(nxt),
            "name": config.fruit_names[int(nxt)],
        },
        "score": int(score),
        "game_over": False,
        "play_area": {
            "left": PLAY_LEFT,
            "right": PLAY_RIGHT,
            "top": PLAY_TOP,
            "bottom": PLAY_BOT,
        },
    }


def mirror_state(state):
    out = copy.deepcopy(state)
    out["fruits"] = []
    for f in state.get("fruits") or []:
        nf = dict(f)
        nf["x"] = float(PLAY_LEFT + PLAY_RIGHT - float(f["x"]))
        out["fruits"].append(nf)
    out["name"] = str(state.get("name", "state")) + "_mirror"
    return out


def make_stress_states(seed=0):
    """Return representative states without touching replay/checkpoints."""
    rng = np.random.default_rng(seed)
    states = []

    # Same-type fruit squeezed near the left wall: good policies should not
    # blindly over-center if the merge is safely reachable.
    states.append(_state("near_wall_left", [
        _fruit(PLAY_LEFT + 23, PLAY_BOT - 18, 0),
        _fruit(PLAY_LEFT + 58, PLAY_BOT - 18, 0),
        _fruit(PLAY_LEFT + 108, PLAY_BOT - 26, 1),
        _fruit(PLAY_LEFT + 165, PLAY_BOT - 38, 2),
    ], current=0, nxt=2, score=42))

    states.append(_state("corner_right", [
        _fruit(PLAY_RIGHT - 28, PLAY_BOT - 20, 0),
        _fruit(PLAY_RIGHT - 64, PLAY_BOT - 24, 1),
        _fruit(PLAY_RIGHT - 118, PLAY_BOT - 44, 2),
        _fruit(PLAY_RIGHT - 170, PLAY_BOT - 72, 3),
    ], current=1, nxt=0, score=90))

    states.append(_state("top_line_danger", [
        _fruit(PLAY_LEFT + 105, KILLY + 22, 3),
        _fruit(PLAY_LEFT + 175, KILLY + 40, 2),
        _fruit(PLAY_LEFT + 245, KILLY + 30, 3),
        _fruit(PLAY_LEFT + 325, KILLY + 68, 4),
        _fruit(PLAY_LEFT + 385, KILLY + 84, 2),
    ], current=2, nxt=1, score=520))

    dense = []
    for row in range(4):
        y = PLAY_BOT - 18 - row * 42
        for col in range(7):
            x = PLAY_LEFT + 34 + col * 62 + (row % 2) * 18
            dense.append(_fruit(x, y, int((row + col) % 4)))
    states.append(_state("dense_board", dense, current=3, nxt=1, score=880))

    states.append(_state("narrow_gap", [
        _fruit(PLAY_LEFT + 120, PLAY_BOT - 36, 2),
        _fruit(PLAY_LEFT + 180, PLAY_BOT - 44, 2),
        _fruit(PLAY_LEFT + 255, PLAY_BOT - 32, 1),
        _fruit(PLAY_LEFT + 312, PLAY_BOT - 44, 1),
        _fruit(PLAY_LEFT + 214, PLAY_BOT - 112, 3),
    ], current=1, nxt=2, score=360))

    for i in range(3):
        fruits = []
        for _ in range(10 + i * 4):
            t = int(rng.integers(0, 5))
            r = radius_of(t)
            x = float(rng.uniform(PLAY_LEFT + r, PLAY_RIGHT - r))
            y = float(rng.uniform(PLAY_BOT - 190, PLAY_BOT - r))
            fruits.append(_fruit(x, y, t))
        states.append(_state("random_dense_%d" % i, fruits,
                             current=int(rng.integers(0, 5)),
                             nxt=int(rng.integers(0, 5)),
                             score=int(rng.integers(0, 1000))))
    return states


def stress_state_pairs(seed=0):
    for state in make_stress_states(seed):
        yield state, mirror_state(state)
