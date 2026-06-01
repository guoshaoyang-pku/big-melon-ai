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

Settle hot-path optimisations (behaviour-preserving):
  * one combined live-particle scan per check folds the old game-over pass and
    the old max-velocity pass into a single ``space.shapes`` traversal;
  * squared-speed comparison removes a per-particle ``sqrt`` and the per-call
    ``Particle.pos`` numpy allocation;
  * an optional ``check_every`` runs the Python-side settle/game-over check only
    every N physics steps for cheap fast rollouts (precise stays N=1 so it is
    bit-for-bit identical to the original per-step logic);
  * rollouts can skip the new fruit's collision-free free fall and place it just
    before the first possible contact, preserving a plausible impact speed while
    spending pymunk steps on the post-contact settling phase.
"""
import math

import numpy as np

from common import (config, CollisionTypes, PLAY_LEFT, PLAY_RIGHT, PLAY_TOP,
                    PLAY_BOT,
                    KILLY, col_to_x, radius_of, SPAWN_TYPES, NUM_FRUIT_TYPES)
from particle import Particle
from wall import Wall
from collision import collide

FPS = config.screen.fps
BOTTOM_WALL_RADIUS = 2.0
FREEFALL_CLEARANCE = 1.0
FREEFALL_MIN_DISTANCE = 4.0


def _build_space():
    space = pymunk_space()
    return space


def pymunk_space(iterations=None):
    """Build a fresh space + 3 walls + particle/particle merge handler.

    ``iterations`` is optional and defaults to ``None`` which leaves pymunk's
    default solver iteration count (10) untouched -- existing callers see no
    change. It is exposed only so benchmarks can probe the speed / fidelity
    trade-off of lowering it.
    """
    import pymunk
    space = pymunk.Space()
    space.gravity = (0, config.physics.gravity)
    space.damping = config.physics.damping
    space.collision_bias = config.physics.bias
    if iterations is not None:
        space.iterations = int(iterations)
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


def _scan_live(space, killy):
    """Single traversal over live particles.

    Returns ``(game_over, max_velocity_squared)``. ``game_over`` uses the exact
    same ``has_collided`` + below-kill-line rule as the old ``_check_game_over``
    (``body.position.y`` equals ``Particle.pos[1]`` without the numpy round-trip)
    and short-circuits on the first offending particle. The max squared speed is
    accumulated in the same pass so the settle threshold can be evaluated
    without a second scan and without a per-particle ``sqrt``.
    """
    over = False
    vmax2 = 0.0
    for s in space.shapes:
        if isinstance(s, Particle) and s.alive:
            if s.has_collided and s.body.position.y < killy:
                over = True
                break
            vel = s.body.velocity
            v2 = vel.x * vel.x + vel.y * vel.y
            if v2 > vmax2:
                vmax2 = v2
    return over, vmax2


def _check_game_over(space):
    """Backward-compatible game-over probe (delegates to the shared scan)."""
    over, _ = _scan_live(space, KILLY)
    return over


def _first_contact_y(fruits, x, radius):
    """Return the new fruit centre-y just before its first vertical contact.

    The pre-contact phase has no horizontal motion, so the earliest possible
    contact is either the bottom wall or the upper arc of an existing fruit whose
    circle overlaps the drop column.
    """
    best_y = PLAY_BOT - float(radius) - BOTTOM_WALL_RADIUS
    for f in fruits:
        fx = float(f["x"])
        fy = float(f["y"])
        reach = float(radius) + float(f["radius"])
        dx = fx - float(x)
        if abs(dx) >= reach:
            continue
        dy2 = reach * reach - dx * dx
        contact_y = fy - math.sqrt(max(0.0, dy2))
        if PLAY_TOP <= contact_y < best_y:
            best_y = contact_y
    return best_y


def _freefall_state_after(frames):
    """Exact integer-frame free-fall state for pymunk's default integrator."""
    frames = int(frames)
    if frames <= 0:
        return PLAY_TOP, 0.0
    dt = 1.0 / FPS
    gravity = float(config.physics.gravity)
    d = float(config.physics.damping) ** dt
    a = gravity * dt
    if abs(1.0 - d) < 1e-12:
        velocity = a * frames
        distance = dt * a * frames * (frames + 1) * 0.5
    else:
        velocity = a * (1.0 - d ** frames) / (1.0 - d)
        distance = (
            dt * a * (frames - d * (1.0 - d ** frames) / (1.0 - d))
            / (1.0 - d)
        )
    return PLAY_TOP + distance, velocity


def _freefall_frame_count(distance, max_frames=None):
    """Largest integer free-fall frame count that stays within ``distance``."""
    distance = float(distance)
    if distance <= 0.0:
        return 0
    lo, hi = 0, 1
    while _freefall_state_after(hi)[0] - PLAY_TOP <= distance:
        hi *= 2
        if max_frames is not None and hi >= max_frames:
            hi = int(max_frames)
            break
    if max_frames is not None:
        hi = min(hi, int(max_frames))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _freefall_state_after(mid)[0] - PLAY_TOP <= distance:
            lo = mid
        else:
            hi = mid - 1
    return int(lo)


def _skip_freefall(state, particle, x, radius, max_skip_steps=None):
    """Fast-forward the newly dropped fruit to just before first contact.

    This is intentionally used only for approximate fast rollouts. Once a
    contact can happen, pymunk takes over so rolling, merging and impulses still
    use the same engine path.
    """
    contact_y = _first_contact_y(state.get("fruits") or [], x, radius)
    safe_distance = contact_y - FREEFALL_CLEARANCE - PLAY_TOP
    skipped = _freefall_frame_count(safe_distance, max_skip_steps)
    target_y, velocity = _freefall_state_after(skipped)
    if skipped <= 0 or target_y - PLAY_TOP < FREEFALL_MIN_DISTANCE:
        return 0
    particle.body.position = (float(x), float(target_y))
    particle.body.velocity = (0.0, velocity)
    return int(skipped)


def _settle(space, max_steps, settle_velocity, check_every=1):
    """Advance physics until quiescent (max speed < threshold) or game over.

    ``check_every`` controls how often the Python-side settle/game-over check
    runs: every physics step for ``check_every == 1`` (the original behaviour),
    or once per ``N`` steps otherwise. The final step is always checked so a
    capped rollout never skips its terminal evaluation. The settle test uses the
    squared speed against ``settle_velocity ** 2`` (mathematically identical to
    ``max speed < settle_velocity`` since both sides are non-negative).
    """
    if check_every < 1:
        check_every = 1
    settle_v2 = float(settle_velocity) * float(settle_velocity)
    killy = KILLY
    dt = 1.0 / FPS
    last = max_steps - 1
    over = False
    for i in range(max_steps):
        space.step(dt)
        if i != last and (i % check_every) != (check_every - 1):
            continue
        over, vmax2 = _scan_live(space, killy)
        if over:
            break
        if vmax2 < settle_v2:
            break
    return over


class SuikaModel:
    """Discrete-column physics model used by MCTS / self-play.

    Parameters control the K-column action space and the fast/precise settle
    fidelity. ``step`` is pure: it takes a state dict and returns a brand-new
    state dict, never mutating the input.

    ``fast_check_every`` / ``precise_check_every`` set how often the settle loop
    runs its Python-side check for each fidelity. Precise defaults to 1 (per
    step), while ``precise_freefall_skip`` only skips the collision-free
    pre-contact fall; contacts and settling still use precise parameters.
    ``iterations`` is forwarded to ``pymunk_space`` (``None`` keeps the pymunk
    default solver iterations).
    """

    def __init__(self, K=16, fast=True,
                 fast_steps=70, fast_settle_v=4.0,
                 precise_steps=240, precise_settle_v=2.0,
                 fast_check_every=3, precise_check_every=1,
                 iterations=None, fast_freefall_skip=True,
                 precise_freefall_skip=True):
        self.K = K
        self.fast = fast
        self.fast_steps = fast_steps
        self.fast_settle_v = fast_settle_v
        self.precise_steps = precise_steps
        self.precise_settle_v = precise_settle_v
        self.fast_check_every = fast_check_every
        self.precise_check_every = precise_check_every
        self.iterations = iterations
        self.fast_freefall_skip = bool(fast_freefall_skip)
        self.precise_freefall_skip = bool(precise_freefall_skip)

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
        check_every = (self.fast_check_every if fast
                       else self.precise_check_every)

        space = pymunk_space(iterations=self.iterations)
        # rebuild settled fruits (already-collided so game-over rule matches).
        for f in state["fruits"]:
            p = Particle((float(f["x"]), float(f["y"])), int(f["type"]), space)
            p.has_collided = True

        cur_type = int(state["current"]["type"])
        r = radius_of(cur_type)
        x = float(np.clip(col_to_x(col, self.K),
                          PLAY_LEFT + r, PLAY_RIGHT - r))
        dropped = Particle((x, PLAY_TOP), cur_type, space)
        skipped_steps = 0
        if ((fast and self.fast_freefall_skip)
                or ((not fast) and self.precise_freefall_skip)):
            # Fast rollouts still need a post-contact settle budget; subtracting
            # skipped free-fall frames under-settles crowded boards and produces
            # misleading MCTS states. Precise mode keeps a fixed total frame
            # budget so fidelity comparisons remain bounded.
            max_skip = None if fast else max(0, max_steps - 1)
            skipped_steps = _skip_freefall(state, dropped, x, r, max_skip)

        settle_steps = max_steps if fast else max(1, max_steps - skipped_steps)
        over = _settle(space, settle_steps, settle_v, check_every)
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
