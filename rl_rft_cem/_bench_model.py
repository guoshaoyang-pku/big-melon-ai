"""Benchmark + correctness harness for the optimised SuikaModel.

Compares the pre-optimisation model (kept as ``_model_orig.py``) against the
current ``model.py`` on a representative late-game board (~dozens of fruits):

  * per-step wall-clock for fast and precise fidelities (steps/s + speedup);
  * precise-fidelity output parity (fruit count / positions / score / over);
  * fast-fidelity fidelity gap (vs precise) old vs new -- must stay same order;
  * optional space.iterations sweep (speed vs precise fidelity trade-off).

Light by design (small rep counts) so it does not hog CPU shared with any
running coldstart/self-play workers.
"""
import time
import numpy as np

import common  # noqa: F401  (sets up sys.path / cwd / headless SDL)
from common import config, radius_of, SPAWN_TYPES

import model as new_model
import _model_orig as old_model


def empty_state(seed_types):
    cur, nxt = seed_types
    return {
        "fruits": [],
        "current": {"type": cur, "name": config.fruit_names[cur],
                    "radius": radius_of(cur)},
        "next": {"type": nxt, "name": config.fruit_names[nxt]},
        "score": 0,
        "game_over": False,
        "play_area": None,
    }


def build_board(n_drops=45, seed=7, K=16):
    """Grow a busy board with deterministic precise drops (original model)."""
    rng = np.random.default_rng(seed)
    m = old_model.SuikaModel(K=K, fast=False)
    st = empty_state((old_model.sample_fruit(rng), old_model.sample_fruit(rng)))
    for _ in range(n_drops):
        if st["game_over"]:
            break
        col = int(rng.integers(0, K))
        nf = old_model.sample_fruit(rng)
        st, _, _ = m.step(st, col, nf, fast=False)
    return st


def time_step(mod, state, fast, reps, K=16, iterations=None, seed=123):
    rng = np.random.default_rng(seed)
    drops = [(int(rng.integers(0, K)), old_model.sample_fruit(rng))
             for _ in range(reps)]
    if iterations is None:
        m = mod.SuikaModel(K=K)
    else:
        m = mod.SuikaModel(K=K, iterations=iterations)
    # warm up
    m.step(state, drops[0][0], drops[0][1], fast=fast)
    t0 = time.perf_counter()
    for col, nf in drops:
        m.step(state, col, nf, fast=fast)
    dt = time.perf_counter() - t0
    return dt / reps


def parity_check(state, n=24, K=16, seed=321):
    """Precise old vs new must match; report fast divergence too."""
    rng = np.random.default_rng(seed)
    mo = old_model.SuikaModel(K=K)
    mn = new_model.SuikaModel(K=K)
    max_pos_d = 0.0
    score_mismatch = 0
    count_mismatch = 0
    over_mismatch = 0
    for _ in range(n):
        col = int(rng.integers(0, K))
        nf = old_model.sample_fruit(rng)
        so, ro, oo = mo.step(state, col, nf, fast=False)
        sn, rn, on = mn.step(state, col, nf, fast=False)
        if ro != rn:
            score_mismatch += 1
        if len(so["fruits"]) != len(sn["fruits"]):
            count_mismatch += 1
        if oo != on:
            over_mismatch += 1
        if len(so["fruits"]) == len(sn["fruits"]):
            fo = sorted(so["fruits"], key=lambda f: (round(f["y"], 3), round(f["x"], 3)))
            fn = sorted(sn["fruits"], key=lambda f: (round(f["y"], 3), round(f["x"], 3)))
            for a, b in zip(fo, fn):
                d = abs(a["x"] - b["x"]) + abs(a["y"] - b["y"])
                if d > max_pos_d:
                    max_pos_d = d
    return dict(n=n, score_mismatch=score_mismatch, count_mismatch=count_mismatch,
                over_mismatch=over_mismatch, max_pos_delta=max_pos_d)


def main():
    K = 16
    print("Building representative late-game board (precise drops)...")
    state = build_board(n_drops=45, seed=7, K=K)
    print("  board fruits: %d   score: %d   game_over: %s"
          % (len(state["fruits"]), state["score"], state["game_over"]))

    reps_fast, reps_prec = 60, 25

    print("\n=== Per-step timing (lower is better) ===")
    of = time_step(old_model, state, fast=True, reps=reps_fast, K=K)
    nf = time_step(new_model, state, fast=True, reps=reps_fast, K=K)
    op = time_step(old_model, state, fast=False, reps=reps_prec, K=K)
    np_ = time_step(new_model, state, fast=False, reps=reps_prec, K=K)

    def line(tag, told, tnew):
        sp = (told / tnew - 1.0) * 100.0
        print("  %-8s old %8.3f ms (%6.1f st/s) -> new %8.3f ms (%6.1f st/s)"
              "  | speedup %+6.1f%%" %
              (tag, told * 1e3, 1.0 / told, tnew * 1e3, 1.0 / tnew, sp))

    line("fast", of, nf)
    line("precise", op, np_)

    print("\n=== Precise parity (old vs new, fast=False) ===")
    par = parity_check(state, n=24, K=K)
    print("  ", par)

    print("\n=== Fidelity gap (fast vs precise) ===")
    go = old_model.measure_fidelity_gap(state, K=K, trials=12, seed=0)
    gn = new_model.measure_fidelity_gap(state, K=K, trials=12, seed=0)
    print("  old: mean|dscore|=%.3f  mean|dcount|=%.3f" % go)
    print("  new: mean|dscore|=%.3f  mean|dcount|=%.3f" % gn)

    print("\n=== space.iterations sweep (new model, precise) ===")
    base = time_step(new_model, state, fast=False, reps=reps_prec, K=K, iterations=None)
    for it in (10, 8, 6):
        t = time_step(new_model, state, fast=False, reps=reps_prec, K=K, iterations=it)
        # fidelity vs default-iterations precise on a few drops
        rng = np.random.default_rng(55)
        mref = new_model.SuikaModel(K=K)                  # default iters
        mit = new_model.SuikaModel(K=K, iterations=it)
        dpos = 0.0
        dsc = 0
        for _ in range(10):
            col = int(rng.integers(0, K)); fr = old_model.sample_fruit(rng)
            sr, rr, _ = mref.step(state, col, fr, fast=False)
            si, ri, _ = mit.step(state, col, fr, fast=False)
            if rr != ri:
                dsc += 1
            if len(sr["fruits"]) == len(si["fruits"]):
                a = sorted(sr["fruits"], key=lambda f: (round(f["y"],3), round(f["x"],3)))
                b = sorted(si["fruits"], key=lambda f: (round(f["y"],3), round(f["x"],3)))
                dpos = max(dpos, max((abs(x["x"]-y["x"])+abs(x["y"]-y["y"]) for x,y in zip(a,b)), default=0.0))
        print("  iters=%2s  %8.3f ms (%6.1f st/s)  vs default: score_mism=%d  max_pos_delta=%.2f"
              % (it, t*1e3, 1.0/t, dsc, dpos))

    print("\nDONE")


if __name__ == "__main__":
    main()
