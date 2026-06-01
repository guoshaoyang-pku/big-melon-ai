"""AlphaZero actor-learner continuous training loop for Suika (off-policy).

Architecture (no per-round synchronization barrier):
  * N resident self-play workers (CPU, spawn) play games back to back and push
    (state, pi, z) samples into a shared queue; each worker reloads latest.pt
    periodically so it tracks the improving policy.
  * The learner (this process, MPS) continuously drains the queue into a large
    bounded FIFO replay buffer and trains mini-batches OFF-POLICY, throttled to
    a target sample-reuse ratio. It periodically refreshes latest.pt (workers
    pick it up), snapshots checkpoints, evaluates, and saves the buffer.

Usage:
  python run_pipeline.py --config config.yaml [--resume] [--smoke]
                         [--set key=val ...]

--resume warm-starts net + optimizer + step + replay buffer from latest.pt.
--smoke runs a tiny, fully ISOLATED end-to-end pass under rl/_smoke/ (it never
touches the real checkpoints/ or data/). SIGTERM/SIGINT trigger a clean
shutdown: stop workers, drain the queue, save checkpoint + buffer.
"""
import argparse
import os
import time
import datetime
import signal
import queue
import resource
import multiprocessing as mp
from collections import defaultdict, deque

import yaml
import numpy as np
import torch

from common import input_dim
from net import build_net, pick_device
from selfplay import continuous_worker
from train import (ReplayBuffer, train_steps, save_checkpoint,
                   load_checkpoint)
import eval as ev

HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def prune_snapshots(ckpt_dir, keep):
    """Keep only the newest ``keep`` step_*.pt snapshots; never touch best.pt
    or latest.pt. Prevents the overnight run from filling the disk."""
    import glob
    try:
        keep = int(keep)
        if keep <= 0:
            return
        snaps = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")),
                       key=lambda f: os.path.getmtime(f))
        for old in snaps[:-keep]:
            try:
                os.remove(old)
                log(f"pruned old snapshot {os.path.basename(old)}")
            except OSError:
                pass
    except Exception as exc:
        log(f"snapshot prune skipped: {exc!r}")


def reuse_for(buffer_size):
    """Adaptive off-policy reuse schedule keyed on replay size."""
    n = int(buffer_size)
    if n < 10_000:
        return 6
    if n < 50_000:
        return 8
    # capped at 8x to prevent off-policy over-reuse (entropy/policy collapse)
    if n < 200_000:
        return 8
    return 8


def _coerce(v):
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


# new actor-learner knobs (added with safe defaults; config.yaml may override)
_DEFAULTS = {
    "target_sample_reuse": 16,   # off-policy: avg gradient exposures per sample
    "train_chunk": 8,            # train steps per learner loop iteration
    "latest_save_sec": 20,       # how often to refresh latest.pt for workers
    "log_every_sec": 30,         # STATS line cadence
    "eval_every_sec": 900,       # periodic eval cadence (0 disables)
    "buffer_save_sec": 300,      # replay buffer disk snapshot cadence
    "snapshot_sec": 1800,        # milestone checkpoint cadence
    "worker_reload_sec": 20,     # how often workers reload latest.pt
    "queue_maxsize": 0,          # 0 -> num_workers*4
    "max_game_seconds": 0,       # 0 -> no wall-clock truncation
    "allow_truncated_games": False,  # discard capped games by default
    "run_seconds": 0,            # 0 -> run forever (until killed)
    "arch": "transformer",
    "token_dim": 128,
    "d_model": 128,
    "n_heads": 4,
    "n_transformer_layers": 2,
    "dropout": 0.1,
    "mirror_augmentation": True,
    "adaptive_reuse": False,
    "entropy_coef": 0.0,
    "collect_tree_samples": False,
    "tree_min_visits": 16,
    "tree_max_nodes": 6,
    "tree_value_weight": 0.25,
    "root_forced_visits": 0,
    "eval_baseline_ckpt": "",
    "boundary_features": False,
    # --- policy-only (no learned value) + endgame-reseed additions ---
    "use_value_net": True,        # False -> freeze value head, policy-only train
    "leaf_value_mode": "heuristic",  # MCTS leaf bootstrap when value net is off
    "value_loss_weight": None,    # None -> fall back to value_coef
    "leaf_safety_w": 0.15,
    "leaf_fill_w": 0.05,
    "leaf_value_clip": 0.5,
    "endgame_reseed": False,      # start self-play from mined near-endgame states
    "endgame_tail_min": 20,
    "endgame_tail_max": 50,
    "endgame_pool_capacity": 200,
    "endgame_refresh_games": 20,
    "endgame_explore_eps": 0.25,
}


def load_cfg(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        cfg[k] = _coerce(v)
    if args.smoke:
        cfg.update({
            "num_simulations": 24, "num_workers": 2, "max_moves": 20,
            "max_game_seconds": 0, "allow_truncated_games": False,
            "batch_size": 32, "train_chunk": 4,
            "min_buffer_to_train": 16, "target_sample_reuse": 4,
            "replay_capacity": 5000, "queue_maxsize": 8,
            "latest_save_sec": 4, "log_every_sec": 3, "worker_reload_sec": 4,
            "buffer_save_sec": 9999, "snapshot_sec": 99999,
            "eval_every_sec": 20, "eval_simulations": 8, "eval_seeds": [0],
            "run_seconds": 40,
            "d_model": 64, "token_dim": 64,
            "n_transformer_layers": 1, "n_heads": 4,
            "collect_tree_samples": True, "tree_min_visits": 4,
            "tree_max_nodes": 6, "tree_value_weight": 0.25,
            "adaptive_reuse": True,
        })
    for k, v in _DEFAULTS.items():
        cfg.setdefault(k, v)
    if not cfg["queue_maxsize"]:
        cfg["queue_maxsize"] = cfg["num_workers"] * 4
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_cfg(args)

    base = os.path.join(HERE, "_smoke") if args.smoke else HERE
    ckpt_dir = os.path.join(base, "checkpoints")
    log_dir = os.path.join(base, "logs")
    data_dir = os.path.join(base, "data")
    latest = os.path.join(ckpt_dir, "latest.pt")
    buffer_path = os.path.join(data_dir, "replay.npz")
    for d in (ckpt_dir, log_dir, data_dir):
        os.makedirs(d, exist_ok=True)

    with open(os.path.join(base, "config.effective.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    np.random.seed(cfg["base_seed"])
    torch.manual_seed(cfg["base_seed"])

    device = pick_device(cfg.get("device", "mps"))
    in_dim = input_dim(cfg["max_fruits"], cfg.get("boundary_features", False))
    K = cfg["K"]
    net = build_net(cfg).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    # policy-only mode: freeze the value head so NO gradient (and no weight-decay
    # erosion -- Adam skips params whose grad stays None) ever touches it, while
    # keeping its params/structure in the checkpoint so ai_play_live stays
    # loadable and the change is fully reversible. value_coef is forced to 0.
    use_value_net = bool(cfg.get("use_value_net", True))
    if use_value_net:
        _vlw = cfg.get("value_loss_weight", None)
        value_coef = float(cfg["value_coef"] if _vlw is None else _vlw)
    else:
        value_coef = 0.0
        if hasattr(net, "value_head"):
            for _p in net.value_head.parameters():
                _p.requires_grad_(False)
    buffer = ReplayBuffer(cfg["replay_capacity"], in_dim, K,
                          max_fruits=cfg["max_fruits"],
                          boundary_features=cfg.get("boundary_features", False))

    start_step = 0
    if args.resume and os.path.exists(latest):
        # checkpoint and buffer loads are INDEPENDENT: a checkpoint failure must
        # never silently clobber latest.pt with fresh random weights (that would
        # wipe real training progress); a buffer failure must not abort the run.
        try:
            step, _ = load_checkpoint(latest, net, optimizer, device=device)
            start_step = step
            log(f"RESUME checkpoint loaded from step {start_step}")
        except Exception as exc:
            log(f"RESUME ABORT: failed to load checkpoint {latest} "
                f"(arch={cfg.get('arch')}): {exc!r}. Refusing to overwrite "
                f"latest.pt with random weights -- fix the checkpoint/arch and "
                f"retry. No files modified.")
            raise SystemExit(1)
        try:
            buffer.load(buffer_path)
            log(f"RESUME buffer loaded: size={buffer.size}")
        except Exception as exc:
            log(f"RESUME buffer load failed ({exc!r}); continuing with empty buffer")
    elif args.resume:
        save_checkpoint(latest, net, optimizer, start_step, cfg)
        log("RESUME requested but no latest.pt found; initialised new run")
    else:
        save_checkpoint(latest, net, optimizer, start_step, cfg)
        log("INIT new run; saved initial checkpoint")
    # Enforce config lr / weight_decay AFTER any optimizer-state restore: Adam's
    # load_state_dict() reinstates the checkpoint's param-group lr, silently
    # overriding a deliberate lr change on resume. Re-apply from cfg so the new
    # (lower) lr actually takes effect for the anti-collapse continuation.
    for _pg in optimizer.param_groups:
        _pg["lr"] = float(cfg["lr"])
        _pg["weight_decay"] = float(cfg["weight_decay"])
    log(f"optimizer lr={cfg['lr']} weight_decay={cfg['weight_decay']} "
        f"(re-applied from cfg after resume)")

    # refresh latest.pt so workers boot from current weights. Safe: weights are
    # either freshly loaded from a good checkpoint or a brand-new run -- we never
    # reach here after a failed load (that path raised SystemExit above).
    save_checkpoint(latest, net, optimizer, start_step, cfg)

    log(f"value_net={'on' if use_value_net else 'OFF (frozen head)'} "
        f"value_coef={value_coef} leaf_value_mode={cfg.get('leaf_value_mode')} "
        f"endgame_reseed={cfg.get('endgame_reseed')} "
        f"(tail {cfg.get('endgame_tail_min')}-{cfg.get('endgame_tail_max')} "
        f"pool={cfg.get('endgame_pool_capacity')} "
        f"refresh/{cfg.get('endgame_refresh_games')}g)")
    bytes_per = in_dim * 4 + K * 4 + 4
    log(f"device={device} arch={cfg.get('arch')} in_dim={in_dim} K={K} sims={cfg['num_simulations']} "
        f"workers={cfg['num_workers']} max_moves={cfg['max_moves']} "
        f"max_game_seconds={cfg['max_game_seconds']} "
        f"truncated={'keep' if cfg.get('allow_truncated_games') else 'discard'} "
        f"reuse={'adaptive' if cfg.get('adaptive_reuse') else str(cfg['target_sample_reuse'])+'x'} "
        f"tree_samples={'on' if cfg.get('collect_tree_samples') else 'off'} "
        f"buffer_cap={cfg['replay_capacity']} (~{cfg['replay_capacity']*bytes_per/1e9:.2f}GB)")

    ctx = mp.get_context("spawn")
    sample_q = ctx.Queue(maxsize=cfg["queue_maxsize"])
    stop_event = ctx.Event()
    base_seed = (cfg["base_seed"] + start_step * 1_000_003
                 + int(time.time()) % 100000)
    n_workers = cfg["num_workers"]
    workers = []
    for w in range(n_workers):
        p = ctx.Process(target=continuous_worker,
                        args=(latest, cfg, base_seed, w, sample_q, stop_event,
                              cfg["worker_reload_sec"], cfg["max_game_seconds"]))
        p.start()
        workers.append(p)
    log(f"started {n_workers} resident self-play workers "
        f"pids={[p.pid for p in workers]}")

    def _handle(sig, _frame):
        log(f"received signal {sig}; shutting down gracefully")
        stop_event.set()
    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    adaptive_reuse = bool(cfg.get("adaptive_reuse", False))
    base_R = cfg["target_sample_reuse"]
    R = reuse_for(buffer.size) if adaptive_reuse else base_R
    batch = cfg["batch_size"]
    chunk = cfg["train_chunk"]
    base_produced = buffer.size
    session_produced = 0
    trained_samples = 0
    train_steps_total = 0
    games_done = 0
    total_moves = 0
    per_worker = defaultdict(int)
    score_win = deque(maxlen=200)
    last_pl = last_vl = last_ent = last_t1 = 0.0
    # one-shot comparison eval vs a reference checkpoint (e.g. pre-restart net)
    baseline_ckpt = cfg.get("eval_baseline_ckpt") or ""
    if baseline_ckpt and not os.path.isabs(baseline_ckpt):
        baseline_ckpt = os.path.join(HERE, baseline_ckpt)
    did_baseline_eval = False

    t_start = time.time()
    last_log = last_ckpt = last_eval = last_bufsave = last_snap = t_start
    win_t0 = t_start
    win_games0 = win_moves0 = win_steps0 = 0
    run_seconds = cfg["run_seconds"]

    def drain(timeout):
        nonlocal session_produced, games_done, total_moves
        n = 0
        first = True
        while n < 1024:
            try:
                wid, samples, fs, nm = sample_q.get(
                    timeout=(timeout if first else 0))
            except queue.Empty:
                break
            buffer.add_many(samples, mirror=cfg.get("mirror_augmentation", False))
            session_produced += len(samples)
            games_done += 1
            total_moves += nm
            per_worker[wid] += 1
            score_win.append(fs)
            n += 1
            first = False
        return n

    try:
        while not stop_event.is_set():
            if run_seconds and (time.time() - t_start) >= run_seconds:
                break
            drain(0.0)
            produced = base_produced + session_produced
            if adaptive_reuse:
                R = reuse_for(buffer.size)
            can_train = (buffer.size >= cfg["min_buffer_to_train"]
                         and trained_samples < produced * R)
            if can_train:
                pl, vl, ent, t1 = train_steps(net, optimizer, buffer, device,
                                     chunk, batch, value_coef,
                                     entropy_coef=cfg.get("entropy_coef", 0.0))
                last_pl, last_vl = pl, vl
                last_ent, last_t1 = ent, t1
                trained_samples += chunk * batch
                train_steps_total += chunk
            else:
                # caught up to the reuse budget (or warming up): wait for data
                drain(0.5)

            now = time.time()
            if now - last_ckpt >= cfg["latest_save_sec"]:
                save_checkpoint(latest, net, optimizer,
                                start_step + train_steps_total, cfg)
                last_ckpt = now
            if now - last_log >= cfg["log_every_sec"]:
                dt = max(1e-6, now - win_t0)
                gpm = (games_done - win_games0) / (dt / 60.0)
                mvps = (total_moves - win_moves0) / dt
                mvpg = (total_moves - win_moves0) / max(1, games_done - win_games0)
                stps = (train_steps_total - win_steps0) / dt
                maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
                buf_mb = buffer.size * bytes_per / 1e6
                reuse_act = trained_samples / max(1, produced)
                pw = " ".join(f"w{w}={per_worker[w]}" for w in range(n_workers))
                smean = np.mean(score_win) if score_win else 0.0
                smax = max(score_win) if score_win else 0.0
                log(f"STATS buffer={buffer.size}/{cfg['replay_capacity']} "
                    f"(~{buf_mb:.0f}MB) | produced={session_produced} "
                    f"games={games_done} ({gpm:.2f} g/min {mvps:.2f} mv/s "
                    f"{mvpg:.1f} mv/game) | "
                    f"steps={train_steps_total} ({stps:.2f} st/s "
                    f"{trained_samples} samp reuse~{reuse_act:.1f}x tgt{R}x) | "
                    f"loss p={last_pl:.4f} v={last_vl:.4f} "
                    f"ent={last_ent:.3f} top1={last_t1:.3f} | "
                    f"score mean={smean:.0f} max={smax:.0f} | "
                    f"learnerRSS={maxrss:.0f}MB | workers[{pw}]")
                last_log = now
                win_t0 = now
                win_games0 = games_done
                win_moves0 = total_moves
                win_steps0 = train_steps_total
            if now - last_snap >= cfg["snapshot_sec"]:
                st = start_step + train_steps_total
                save_checkpoint(os.path.join(ckpt_dir, f"step_{st:08d}.pt"),
                                net, optimizer, st, cfg)
                log(f"snapshot step_{st:08d}.pt")
                prune_snapshots(ckpt_dir, cfg.get("snapshot_keep", 6))
                last_snap = now
            if now - last_bufsave >= cfg["buffer_save_sec"]:
                buffer.save(buffer_path)
                last_bufsave = now
            if cfg["eval_every_sec"] and now - last_eval >= cfg["eval_every_sec"]:
                t_ev = time.time()
                n_avg, n_maxfruit, n_scores = ev.eval_net(
                    net, device, cfg, cfg["eval_seeds"],
                    num_simulations=cfg["eval_simulations"])
                n_max = max(n_scores) if n_scores else 0.0
                log(f"EVAL net(sims={cfg['eval_simulations']} "
                    f"games={len(cfg['eval_seeds'])}) mean={n_avg:.0f} "
                    f"max={n_max:.0f} maxfruit={n_maxfruit} "
                    f"in {time.time()-t_ev:.1f}s")
                # one-shot comparison against the reference (pre-restart) net
                if (not did_baseline_eval and baseline_ckpt
                        and os.path.exists(baseline_ckpt)):
                    try:
                        t_b = time.time()
                        base_net = build_net(cfg).to(device)
                        load_checkpoint(baseline_ckpt, base_net, None,
                                        device=device)
                        b_avg, b_maxfruit, b_scores = ev.eval_net(
                            base_net, device, cfg, cfg["eval_seeds"],
                            num_simulations=cfg["eval_simulations"])
                        b_max = max(b_scores) if b_scores else 0.0
                        ref = os.path.basename(os.path.dirname(baseline_ckpt))
                        log(f"EVAL baseline({ref}) mean={b_avg:.0f} "
                            f"max={b_max:.0f} maxfruit={b_maxfruit} "
                            f"in {time.time()-t_b:.1f}s")
                        log(f"EVAL COMPARE current(mean={n_avg:.0f} "
                            f"max={n_max:.0f} maxfruit={n_maxfruit}) vs "
                            f"baseline(mean={b_avg:.0f} max={b_max:.0f} "
                            f"maxfruit={b_maxfruit}) delta_mean={n_avg-b_avg:+.0f}")
                        del base_net
                    except Exception as exc:
                        log(f"EVAL baseline comparison failed: {exc!r}")
                    did_baseline_eval = True
                last_eval = time.time()
    finally:
        log("stopping workers...")
        stop_event.set()
        deadline = time.time() + (cfg["max_game_seconds"] or 90) + 30
        for p in workers:
            p.join(timeout=max(1.0, deadline - time.time()))
        for p in workers:
            if p.is_alive():
                log(f"terminating stuck worker pid {p.pid}")
                p.terminate()
        # final drain of whatever the workers already produced
        try:
            while True:
                wid, samples, fs, nm = sample_q.get_nowait()
                buffer.add_many(samples, mirror=cfg.get("mirror_augmentation", False))
                session_produced += len(samples)
        except Exception:
            pass
        st = start_step + train_steps_total
        save_checkpoint(latest, net, optimizer, st, cfg)
        buffer.save(buffer_path)
        log(f"shutdown complete: step={st} buffer={buffer.size} "
            f"session_produced={session_produced}")


if __name__ == "__main__":
    main()
