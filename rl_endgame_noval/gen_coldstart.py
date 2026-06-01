"""Cold-start replay generator for Suika AlphaZero.

Runs *large-batch, parallel* MCTS self-play **without any trained weights** to
bootstrap a replay buffer ("cold start"). The search uses a UNIFORM prior
(``ColdStartNet``) and a cheap heuristic / zero leaf value, so it depends on no
checkpoint. The real merge rewards collected by the pymunk environment model
inside the tree (mcts._backup) give the visit distribution pi real meaning, and
the value target z is generated only after a real game_over as remaining
return-to-go from each recorded state.

Design notes
------------
* We REUSE the existing search stack untouched: ``selfplay.play_game`` ->
  ``mcts.MCTS`` -> ``model.SuikaModel`` -> ``common.encode_state``. The only new
  piece is ``ColdStartNet`` which mimics the ``infer_batch`` / ``infer`` API of
  the real networks but returns uniform priors + a heuristic value.
* The physics (pymunk CPU settle) dominates wall-clock, so the only real
  speed-up is multi-core parallelism: one process per game, ``--workers`` of
  them via a spawn Pool (matching selfplay's multiprocessing model). GPU/MPS is
  intentionally NOT used -- the "network" is a trivial uniform constant, so
  there is nothing worth batching onto the GPU. This is honest: physics is the
  bottleneck.
* Output is written incrementally (atomic tmp -> replace) in a format that is
  byte-compatible with ``train.ReplayBuffer.load`` (vecs/pis/zs/pos/size), so it
  can be loaded or merged later. We DEFAULT to ``rl/data/coldstart.npz`` and
  never touch ``replay.npz`` unless the user explicitly runs ``--merge-into``.

Usage
-----
Smoke test::

    python rl/gen_coldstart.py --games 4 --sims 16 --workers 4

Large background run::

    nohup python rl/gen_coldstart.py --games 500 --sims 100 --workers 9 \
        > rl/logs/coldstart.log 2>&1 & echo $! > rl/logs/coldstart.pid

Safe merge into replay (backs up replay.npz first, validates dims)::

    python rl/gen_coldstart.py --merge-into rl/data/replay.npz \
        --out rl/data/coldstart.npz
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np

# common.py fixes sys.path / cwd so part2 + rl modules import cleanly.
import common  # noqa: F401  (side effects: path/cwd setup)
from common import input_dim


# --------------------------------------------------------------------------- #
# Cold-start "network": uniform prior + heuristic/zero value. Picklable so it
# survives the spawn Pool. Exposes the same infer_batch/infer surface the MCTS
# and selfplay code expect from a real net.
# --------------------------------------------------------------------------- #
class ColdStartNet:
    """Weightless stand-in for the policy/value net.

    * priors:  uniform 1/K over the K drop columns (true cold start).
    * value:   ``zero``      -> 0.0 leaf bootstrap; Q becomes the pure
                              return-to-go of REAL merge rewards collected along
                              the search path (a real, if shallow, rollout).
               ``heuristic`` -> a small, bounded safety term decoded from the
                              global features of the encoded state (lower / less
                              crammed boards are safer), added on top of the real
                              reward backups to nudge pi away from game-over.
    """

    # global-feature indices inside the flat encode_state vector (see common.py)
    _IDX_TOPY = 5      # (top_y - PLAY_TOP) / PLAY_H : larger == fruits lower == safer
    _IDX_FILL = 8      # fill fraction : larger == more crammed == riskier

    def __init__(self, K, value_mode="heuristic",
                 safety_w=0.15, fill_w=0.05, value_clip=0.5):
        self.K = int(K)
        self.value_mode = str(value_mode)
        self.safety_w = float(safety_w)
        self.fill_w = float(fill_w)
        self.value_clip = float(value_clip)

    def _values(self, vecs):
        if self.value_mode == "zero":
            return np.zeros((vecs.shape[0],), dtype=np.float32)
        topy = vecs[:, self._IDX_TOPY]
        fill = vecs[:, self._IDX_FILL]
        v = self.safety_w * topy - self.fill_w * fill
        return np.clip(v, -self.value_clip, self.value_clip).astype(np.float32)

    def infer_batch(self, vecs, device=None):
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        b = vecs.shape[0]
        probs = np.full((b, self.K), 1.0 / self.K, dtype=np.float32)
        return probs, self._values(vecs)

    def infer(self, vec, device=None):
        probs, vals = self.infer_batch(np.asarray(vec)[None, :], device)
        return probs[0], float(vals[0])

    # the real nets are nn.Module; play_game never calls .eval()/.to() on the
    # net it receives, but provide harmless no-ops just in case.
    def eval(self):
        return self

    def to(self, *_a, **_k):
        return self


# --------------------------------------------------------------------------- #
# Worker: play exactly one game with the cold-start net. Top-level so the spawn
# Pool can pickle it.
# --------------------------------------------------------------------------- #
def _play_one(args):
    cfg, seed, value_mode = args
    import torch
    torch.set_num_threads(1)                 # avoid CPU oversubscription
    from selfplay import play_game           # reuse the real game loop
    net = ColdStartNet(cfg["K"], value_mode=value_mode)
    t0 = time.time()
    samples, final_score, n_moves = play_game(net, "cpu", cfg, seed)
    return samples, float(final_score), int(n_moves), time.time() - t0


# --------------------------------------------------------------------------- #
# Incremental, crash-safe NPZ writer (ReplayBuffer.load-compatible).
# --------------------------------------------------------------------------- #
def _save(path, vecs_list, pis_list, zs_list):
    """Stack accumulated samples and write atomically. Returns n samples."""
    if not vecs_list:
        return 0
    vecs = np.asarray(np.concatenate(vecs_list, axis=0), dtype=np.float32)
    pis = np.asarray(np.concatenate(pis_list, axis=0), dtype=np.float32)
    zs = np.asarray(np.concatenate(zs_list, axis=0), dtype=np.float32)
    n = len(vecs)
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp, vecs=vecs, pis=pis, zs=zs,
        pos=np.array([n], dtype=np.int64),
        size=np.array([n], dtype=np.int64),
    )
    os.replace(tmp, path)
    return n


# --------------------------------------------------------------------------- #
# Safe merge of coldstart.npz INTO an existing replay file.
# --------------------------------------------------------------------------- #
def merge_into(source_path, target_path):
    if not os.path.exists(source_path):
        raise SystemExit("merge: source %s not found" % source_path)
    if not os.path.exists(target_path):
        raise SystemExit("merge: target %s not found" % target_path)

    src = np.load(source_path)
    tgt = np.load(target_path)
    for k in ("vecs", "pis", "zs"):
        if k not in src.files or k not in tgt.files:
            raise SystemExit("merge: missing field %r in source/target" % k)
    if src["vecs"].shape[1] != tgt["vecs"].shape[1]:
        raise SystemExit("merge: vec dim mismatch %d vs %d"
                         % (src["vecs"].shape[1], tgt["vecs"].shape[1]))
    if src["pis"].shape[1] != tgt["pis"].shape[1]:
        raise SystemExit("merge: policy dim mismatch %d vs %d"
                         % (src["pis"].shape[1], tgt["pis"].shape[1]))

    # back up target before touching it.
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = "%s.bak_%s" % (target_path, ts)
    shutil.copy2(target_path, bak)

    vecs = np.concatenate([tgt["vecs"], src["vecs"]], axis=0).astype(np.float32)
    pis = np.concatenate([tgt["pis"], src["pis"]], axis=0).astype(np.float32)
    zs = np.concatenate([tgt["zs"], src["zs"]], axis=0).astype(np.float32)
    n = len(vecs)
    tmp = target_path + ".tmp.npz"
    np.savez_compressed(
        tmp, vecs=vecs, pis=pis, zs=zs,
        pos=np.array([n], dtype=np.int64),
        size=np.array([n], dtype=np.int64),
    )
    os.replace(tmp, target_path)
    print("[merge] backed up -> %s" % bak)
    print("[merge] %s: %d + %d = %d samples"
          % (target_path, len(tgt["vecs"]), len(src["vecs"]), n))


# --------------------------------------------------------------------------- #
# Config assembled for the search (subset of config.yaml + CLI overrides).
# --------------------------------------------------------------------------- #
def _load_yaml_cfg():
    import yaml
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def build_cfg(args, ycfg):
    return {
        "K": int(ycfg["K"]),
        "max_fruits": int(ycfg["max_fruits"]),
        "num_simulations": int(args.sims),
        "c_puct": float(ycfg.get("c_puct", 1.5)),
        # cold start: keep exploration noise so uniform-prior search spreads out.
        "dirichlet_alpha": float(ycfg.get("dirichlet_alpha", 0.3)),
        "dirichlet_eps": float(ycfg.get("dirichlet_eps", 0.25)),
        "eval_batch": int(args.eval_batch),
        "temp_moves": int(args.temp_moves),
        "temp_final": float(ycfg.get("temp_final", 0.25)),
        "max_moves": int(args.max_moves),
        # Truncated games are discarded so value targets always use real game_over.
        "allow_truncated_games": False,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=500, help="number of games to play")
    ap.add_argument("--sims", type=int, default=100, help="MCTS simulations per move")
    ap.add_argument("--workers", type=int, default=9, help="parallel game processes")
    ap.add_argument("--max-moves", dest="max_moves", type=int, default=0,
                    help="0 = no move cap; positive caps are discarded")
    ap.add_argument("--temp-moves", dest="temp_moves", type=int, default=12,
                    help="moves played at temperature 1.0 before exploiting")
    ap.add_argument("--eval-batch", dest="eval_batch", type=int, default=16,
                    help="MCTS leaf-eval batch size (virtual loss)")
    ap.add_argument("--out", default="rl/data/coldstart.npz",
                    help="output npz (NEVER use replay.npz)")
    ap.add_argument("--seed", type=int, default=12345, help="base RNG seed")
    ap.add_argument("--value-mode", dest="value_mode", default="heuristic",
                    choices=["heuristic", "zero"],
                    help="leaf value: heuristic safety term or pure 0 bootstrap")
    ap.add_argument("--save-every-sec", dest="save_every_sec", type=float,
                    default=30.0, help="checkpoint cadence (seconds)")
    ap.add_argument("--merge-into", dest="merge_into", default=None,
                    help="MERGE MODE: safely append --out into this replay file "
                         "(backs it up first) and exit; no generation")
    args = ap.parse_args()

    # --- merge mode -------------------------------------------------------- #
    if args.merge_into:
        src = args.out if os.path.isabs(args.out) else os.path.join(common._ROOT, args.out)
        tgt = args.merge_into if os.path.isabs(args.merge_into) else os.path.join(common._ROOT, args.merge_into)
        merge_into(src, tgt)
        return

    # Anchor relative paths to the project root. common.py chdir()s the
    # process into the suika root on import, so a relative --out is resolved
    # against that root no matter where the script was launched from.
    if not os.path.isabs(args.out):
        args.out = os.path.join(common._ROOT, args.out)

    # --- guard: never clobber replay.npz ----------------------------------- #
    if os.path.basename(args.out) == "replay.npz":
        raise SystemExit("refusing to write replay.npz; pick another --out")

    ycfg = _load_yaml_cfg()
    cfg = build_cfg(args, ycfg)
    K = cfg["K"]
    exp_dim = input_dim(cfg["max_fruits"])
    assert exp_dim == 329, "unexpected input dim %d" % exp_dim

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print("[cfg] K=%d max_fruits=%d in_dim=%d sims=%d workers=%d max_moves=%d "
          "value_mode=%s temp_moves=%d eval_batch=%d truncation=discard"
          % (K, cfg["max_fruits"], exp_dim, cfg["num_simulations"], args.workers,
             cfg["max_moves"], args.value_mode, cfg["temp_moves"], cfg["eval_batch"]))
    print("[out] %s  (games=%d, seed=%d)" % (args.out, args.games, args.seed))
    sys.stdout.flush()

    jobs = ((cfg, args.seed + i * 100003, args.value_mode)
            for i in range(args.games))

    vecs_list, pis_list, zs_list = [], [], []
    scores, moves = [], []
    n_done = 0
    n_samples = 0
    t_start = time.time()
    t_last_save = t_start

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    def checkpoint(tag):
        nonlocal n_samples
        n_samples = _save(args.out, vecs_list, pis_list, zs_list)
        elapsed = time.time() - t_start
        gpm = (n_done / elapsed * 60.0) if elapsed > 0 else 0.0
        sps = (n_samples / elapsed) if elapsed > 0 else 0.0
        mean_sc = float(np.mean(scores)) if scores else 0.0
        print("[%s] games=%d/%d samples=%d | %.1f games/min %.1f samp/s | "
              "mean_score=%.0f mean_moves=%.1f | elapsed=%.0fs"
              % (tag, n_done, args.games, n_samples, gpm, sps, mean_sc,
                 (float(np.mean(moves)) if moves else 0.0), elapsed))
        sys.stdout.flush()

    try:
        if args.workers <= 1:
            for job in jobs:
                s, fs, nm, _dt = _play_one(job)
                n_done += 1
                scores.append(fs); moves.append(nm)
                for vec, pi, z in s:
                    vecs_list.append(vec[None, :]); pis_list.append(pi[None, :])
                    zs_list.append(np.asarray([z], dtype=np.float32))
                if time.time() - t_last_save >= args.save_every_sec:
                    checkpoint("ckpt"); t_last_save = time.time()
        else:
            with ctx.Pool(processes=args.workers) as pool:
                for s, fs, nm, _dt in pool.imap_unordered(_play_one, jobs, chunksize=1):
                    n_done += 1
                    scores.append(fs); moves.append(nm)
                    for vec, pi, z in s:
                        vecs_list.append(vec[None, :]); pis_list.append(pi[None, :])
                        zs_list.append(np.asarray([z], dtype=np.float32))
                    if time.time() - t_last_save >= args.save_every_sec:
                        checkpoint("ckpt"); t_last_save = time.time()
    except KeyboardInterrupt:
        print("[interrupt] saving partial results...")
    finally:
        checkpoint("final")

    print("[done] %d games -> %d samples in %s" % (n_done, n_samples, args.out))


if __name__ == "__main__":
    main()
