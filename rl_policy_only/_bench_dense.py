"""Dense-board addendum: builds a packed ~dozens-of-fruits board (varied types
so nothing merges instantly) to stress the per-step live scan, then reuses the
timing/parity helpers from _bench_model."""
import numpy as np
import common  # noqa
from common import config, radius_of, PLAY_LEFT, PLAY_RIGHT, PLAY_TOP, PLAY_BOT
import model as new_model
import _model_orig as old_model
from _bench_model import empty_state, time_step, parity_check


def build_dense_board(seed=3):
    """Grid of alternating small types near the bottom; no instant merges."""
    rng = np.random.default_rng(seed)
    fruits = []
    r0 = radius_of(1)            # small-ish fruit radius for spacing
    step = 2.2 * r0
    x = PLAY_LEFT + r0 + 4
    row = 0
    while x < PLAY_RIGHT - r0 - 4:
        y = PLAY_BOT - r0 - 4
        col_i = 0
        while y > PLAY_TOP + 200:
            # alternate types so 4-neighbourhood never has equal pair
            t = (row + col_i) % 4          # 0..3, all distinct neighbours
            fruits.append({"x": float(x), "y": float(y),
                           "radius": radius_of(t), "type": int(t),
                           "name": config.fruit_names[t]})
            y -= step
            col_i += 1
        x += step
        row += 1
    cur = int(rng.integers(0, 5)); nxt = int(rng.integers(0, 5))
    st = empty_state((cur, nxt))
    st["fruits"] = fruits
    return st


def main():
    K = 16
    st = build_dense_board()
    print("Dense board fruits: %d" % len(st["fruits"]))
    reps_fast, reps_prec = 50, 20

    of = time_step(old_model, st, fast=True, reps=reps_fast, K=K)
    nf = time_step(new_model, st, fast=True, reps=reps_fast, K=K)
    op = time_step(old_model, st, fast=False, reps=reps_prec, K=K)
    npr = time_step(new_model, st, fast=False, reps=reps_prec, K=K)

    def line(tag, told, tnew):
        sp = (told / tnew - 1.0) * 100.0
        print("  %-8s old %8.3f ms (%6.1f st/s) -> new %8.3f ms (%6.1f st/s) | speedup %+6.1f%%"
              % (tag, told*1e3, 1.0/told, tnew*1e3, 1.0/tnew, sp))

    print("\n=== Dense-board per-step timing ===")
    line("fast", of, nf)
    line("precise", op, npr)

    print("\n=== Dense-board precise parity (old vs new) ===")
    print("  ", parity_check(st, n=20, K=K))

    print("\n=== Dense-board fidelity gap (fast vs precise) ===")
    go = old_model.measure_fidelity_gap(st, K=K, trials=12, seed=0)
    gn = new_model.measure_fidelity_gap(st, K=K, trials=12, seed=0)
    print("  old: mean|dscore|=%.3f mean|dcount|=%.3f" % go)
    print("  new: mean|dscore|=%.3f mean|dcount|=%.3f" % gn)
    print("DONE")


if __name__ == "__main__":
    main()
