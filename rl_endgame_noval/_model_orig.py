"""Environment model for MCTS.

Rebuilds an independent pymunk space from a ``get_state()`` snapshot, drops the
current fruit into a discrete column, settles the physics, and returns the new
state + immediate merge score + game-over flag. This is the same
"reconstruct-from-snapshot" trick used by ``part2/ai_agent.py`` (SuikaEnv /
Particle / Wall cannot be deep-copied), factored out as a reusable model so
MCTS can expand nodes without ever touching the live environment.

Two fidelities:
  * precise  -> matches SuikaEnv (max_settle_steps=240, settle_velocity=2.0)
  * fast     -> fewer steps / looser threshold for cheap MCTS rollouts.
The mean per-step score/positional difference between the two can be probed
with ``measure_fidelity_gap`` (used in the smoke test / report).
"""
import numpy as np

from common import (config, CollisionTypes, PLAY_LEFT, PLAY_RIGHT, PLAY_TOP,
                    KILLY, col_to_x, radius_of, SPAWN_TYPES, NUM_FRUIT_TYPES)
from particle import Particle
from wall import Wall
from collision import collide

FPS = config.screen.fps


def _build_space():
    space = pymunk_space()
    return space


def pymunk_space():
    import pymunk
    space = pymunk.Space()
    space.gravity = (0, config.physics.gravity)
    space.damping = config.physics.damping
    space.collision_bias = config.physics.bias
    # walls must be kept alive; stash them on the space object.
    space._walls = [
        Wall(config.top_left, config.bot_left, space),
        Wall(config.bot_left, config.bot_right, space),
        Wall(config.bot_right, config.top_right, space),
    ]
    handler = space.add_collision_handler(
        CollisionTypes.PARTICLE, CollisionTypes.PARTICLE
    )
    handler.begin = collide
    handler.data["score"] = 0
    space._handler = handler
    return space


def _live(space):
    return [s for s in space.shapes
            if isinstance(s, Particle) and getattr(s, "alive", False)]


def _check_game_over(space):
    for p in _live(space):
        if p.pos[1] < KILLY and p.has_collided:
            return True
    return False


def _settle(space, max_steps, settle_velocity):
    over = False
    for _ in range(max_steps):
        space.step(1 / FPS)
        if _check_game_over(space):
            over = True
            break
        vmax = max((p.body.velocity.length for p in _live(space)), default=0.0)
        if vmax < settle_velocity:
            break
    return over


class SuikaModel:
    """Discrete-column physics model used by MCTS / self-play.

    Parameters control the K-column action space and the fast/precise settle
    fidelity. ``step`` is pure: it takes a state dict and returns a brand-new
    state dict, never mutating the input.
    """

    def __init__(self, K=16, fast=True,
                 fast_steps=70, fast_settle_v=4.0,
                 precise_steps=240, precise_settle_v=2.0):
        self.K = K
        self.fast = fast
        self.fast_steps = fast_steps
        self.fast_settle_v = fast_settle_v
        self.precise_steps = precise_steps
        self.precise_settle_v = precise_settle_v

    # -- transition ---------------------------------------------------------
    def step(self, state, col, next_fruit, fast=None):
        """Apply dropping the current fruit into ``col``.

        ``next_fruit`` is the *new* next-preview type (sampled by the caller /
        MCTS chance handling). Returns ``(new_state, reward, game_over)`` where
        reward is the raw merge score gained this drop.
        """
        if state.get("game_over"):
            return state, 0.0, True
        fast = self.fast if fast is None else fast
        max_steps = self.fast_steps if fast else self.precise_steps
        settle_v = self.fast_settle_v if fast else self.precise_settle_v

        space = pymunk_space()
        # rebuild settled fruits (already-collided so game-over rule matches).
        for f in state["fruits"]:
            p = Particle((float(f["x"]), float(f["y"])), int(f["type"]), space)
            p.has_collided = True

        cur_type = int(state["current"]["type"])
        r = radius_of(cur_type)
        x = float(np.clip(col_to_x(col, self.K),
                          PLAY_LEFT + r, PLAY_RIGHT - r))
        Particle((x, PLAY_TOP), cur_type, space)

        over = _settle(space, max_steps, settle_v)
        gain = float(space._handler.data["score"])

        fruits = [{
            "x": float(p.pos[0]), "y": float(p.pos[1]),
            "radius": float(p.radius), "type": int(p.n),
            "name": config.fruit_names[p.n],
        } for p in _live(space)]

        new_next_type = int(next_fruit) % SPAWN_TYPES
        new_cur_type = int(state["next"]["type"])
        new_state = {
            "fruits": fruits,
            "current": {
                "type": new_cur_type,
                "name": config.fruit_names[new_cur_type],
                "radius": radius_of(new_cur_type),
            },
            "next": {
                "type": new_next_type,
                "name": config.fruit_names[new_next_type],
            },
            "score": int(state["score"]) + int(gain),
            "game_over": bool(over),
            "play_area": state.get("play_area"),
        }
        return new_state, gain, bool(over)


def sample_fruit(rng):
    """Draw a spawn fruit type uniformly from 0..SPAWN_TYPES-1."""
    return int(rng.integers(0, SPAWN_TYPES))


def measure_fidelity_gap(env_state, K=16, trials=12, seed=0):
    """Compare fast vs precise settling on the same drops; returns mean abs
    score difference and mean fruit-count difference (diagnostic only)."""
    rng = np.random.default_rng(seed)
    m = SuikaModel(K=K)
    ds, dn = [], []
    for _ in range(trials):
        col = int(rng.integers(0, K))
        nf = sample_fruit(rng)
        sf, rf, _ = m.step(env_state, col, nf, fast=True)
        sp, rp, _ = m.step(env_state, col, nf, fast=False)
        ds.append(abs(rf - rp))
        dn.append(abs(len(sf["fruits"]) - len(sp["fruits"])))
    return float(np.mean(ds)), float(np.mean(dn))
