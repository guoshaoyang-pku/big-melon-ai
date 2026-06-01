"""Self-play: drive the real SuikaEnv with MCTS and record training samples.

Real games use the *precise* SuikaEnv physics; MCTS internally uses the fast
model for its rollouts. Each move records (state_vector, MCTS visit dist pi,
score_before); after the game the return-to-go target z is computed for every
recorded state. Games run in parallel CPU worker processes (spawn) so the MPS
GPU stays reserved for training -- the MLP is tiny and CPU inference is far
cheaper than the pymunk settle that dominates each step.

Two modes:
  * generate_selfplay(): legacy batched (one round of N games) -- kept for
    eval / compatibility.
  * continuous_worker(): resident actor-learner worker. Plays games back to
    back with no synchronization barrier, pushes per-game samples into a shared
    queue, and periodically reloads latest.pt to track the improving policy.
"""
import os
import copy
import time
import queue
import numpy as np
import torch

from common import encode_state, SCORE_NORM
from net import build_net
from mcts import MCTS
from model import SuikaModel, sample_fruit

# SuikaEnv import has side effects (pygame init); common.py already fixed cwd.
from suika_env import SuikaEnv


def _make_mcts(net, device, cfg, seed):
    """Construct an MCTS configured from ``cfg``, including the policy-only leaf
    valuation knobs (use_value_net / leaf_value_mode / heuristic weights)."""
    return MCTS(
        net, device, K=cfg["K"],
        num_simulations=cfg["num_simulations"],
        c_puct=cfg["c_puct"],
        dirichlet_alpha=cfg["dirichlet_alpha"],
        dirichlet_eps=cfg["dirichlet_eps"],
        max_fruits=cfg["max_fruits"],
        fast_model=True,
        seed=seed,
        eval_batch=cfg.get("eval_batch", 16),
        root_forced_visits=cfg.get("root_forced_visits", 0),
        boundary_features=cfg.get("boundary_features", False),
        use_value_net=cfg.get("use_value_net", True),
        leaf_value_mode=cfg.get("leaf_value_mode", "heuristic"),
        leaf_safety_w=cfg.get("leaf_safety_w", 0.15),
        leaf_fill_w=cfg.get("leaf_fill_w", 0.05),
        leaf_value_clip=cfg.get("leaf_value_clip", 0.5),
    )


def play_game(net, device, cfg, seed, max_seconds=0):
    """Play one full game with MCTS. Returns (samples, final_score, n_moves).

    ``samples`` is a list of ``(vec, pi, target, w)`` tuples. Root self-play
    states use the true Monte-Carlo return ``z`` as target with weight 1.0.
    When ``cfg['collect_tree_samples']`` is set, high-visit MCTS internal nodes
    are additionally mined per move (excluding the root) and appended with the
    node's bootstrap value Q as target and a reduced weight
    ``cfg['tree_value_weight']`` so they do not pollute the value head.
    ``max_moves <= 0`` means no move cap: keep playing until real game_over.
    If a debug wall-clock or move cap truncates the game, samples are discarded
    unless ``allow_truncated_games`` is explicitly enabled for local debugging.
    """
    K = cfg["K"]
    env = SuikaEnv(seed=seed)
    state = env.get_state()
    mcts = _make_mcts(net, device, cfg, seed)
    samples = []
    tree_samples = []  # (vec, pi, q_value, w): final bootstrap targets
    collect_tree = bool(cfg.get("collect_tree_samples", False))
    tree_min_visits = int(cfg.get("tree_min_visits", 16))
    tree_max_nodes = int(cfg.get("tree_max_nodes", 6))
    tree_w = float(cfg.get("tree_value_weight", 0.25))
    temp_moves = cfg["temp_moves"]
    max_moves = int(cfg.get("max_moves") or 0)
    allow_truncated = bool(cfg.get("allow_truncated_games", False))
    t0 = time.time()
    move = 0
    truncated = False
    while not state["game_over"]:
        if max_moves > 0 and move >= max_moves:
            truncated = True
            break
        if max_seconds and (time.time() - t0) > max_seconds:
            truncated = True
            break
        temp = 1.0 if move < temp_moves else cfg["temp_final"]
        action, pi = mcts.policy(state, temperature=temp, add_noise=True)
        vec = encode_state(state, K, cfg["max_fruits"],
                           boundary_features=cfg.get("boundary_features", False))
        samples.append([vec, pi.astype(np.float32), float(state["score"])])
        if collect_tree:
            root = getattr(mcts, "last_root", None)
            if root is not None:
                for tvec, tpi, tval in mcts.extract_training_nodes(
                        root, min_visits=tree_min_visits,
                        max_nodes=tree_max_nodes):
                    tree_samples.append((tvec, tpi, np.float32(tval),
                                         np.float32(tree_w)))
        state, _reward, done, _info = env.step_column(action, num_columns=K)
        move += 1
        if done:
            break
    final_score = float(state["score"])
    if truncated and not allow_truncated:
        return [], final_score, len(samples)
    # return-to-go target z for each recorded root state: remaining score.
    out = []
    for vec, pi, score_before in samples:
        z = (final_score - score_before) / SCORE_NORM
        out.append((vec, pi, np.float32(z), np.float32(1.0)))
    # mined internal-node samples carry bootstrap Q targets (down-weighted).
    out.extend(tree_samples)
    return out, final_score, len(samples)


def rollout_states(net, device, cfg, seed, max_steps=250):
    """Cheap full game using the policy head (greedy + epsilon) on the precise
    model; returns the trajectory of pre-move states up to game_over. Used only
    to mine realistic near-endgame start positions for the reseed pool, so it
    runs NO MCTS and writes NO training samples -- far cheaper than a search
    game, and it deliberately replays the opening here (once per refresh) so the
    expensive search budget is spent only on endgames."""
    K = cfg["K"]
    bf = cfg.get("boundary_features", False)
    rng = np.random.default_rng(seed)
    model = SuikaModel(K=K, fast=False)
    state = SuikaEnv(seed=seed).get_state()
    eps = float(cfg.get("endgame_explore_eps", 0.25))
    states = []
    steps = 0
    while not state["game_over"] and steps < max_steps:
        states.append(state)
        vec = encode_state(state, K, cfg["max_fruits"], boundary_features=bf)
        probs, _ = net.infer_batch(vec[None, :], device)
        if rng.random() < eps:
            a = int(rng.integers(0, K))
        else:
            a = int(np.argmax(probs[0]))
        state, _gain, _over = model.step(state, a, int(sample_fruit(rng)))
        steps += 1
    return states


def play_game_from_state(net, device, cfg, seed, start_state):
    """Self-play a *policy-only* episode beginning from ``start_state`` (a near-
    endgame position from the reseed pool). Real transitions use the precise
    model (SuikaEnv cannot be reset to an arbitrary state); MCTS uses the fast
    model internally, exactly as in ``play_game``. Returns
    ``(samples, final_score, n_moves)`` with the same tuple layout as
    ``play_game``. Value targets ``z`` are still filled in but go unused while
    the value head is frozen."""
    K = cfg["K"]
    bf = cfg.get("boundary_features", False)
    mcts = _make_mcts(net, device, cfg, seed)
    real_model = SuikaModel(K=K, fast=False)
    rng = np.random.default_rng((seed * 2654435761) & 0xFFFFFFFF)
    collect_tree = bool(cfg.get("collect_tree_samples", False))
    tree_min_visits = int(cfg.get("tree_min_visits", 16))
    tree_max_nodes = int(cfg.get("tree_max_nodes", 6))
    tree_w = float(cfg.get("tree_value_weight", 0.25))
    temp_moves = cfg["temp_moves"]
    state = start_state
    samples = []
    tree_samples = []
    move = 0
    while not state["game_over"]:
        temp = 1.0 if move < temp_moves else cfg["temp_final"]
        action, pi = mcts.policy(state, temperature=temp, add_noise=True)
        vec = encode_state(state, K, cfg["max_fruits"], boundary_features=bf)
        samples.append([vec, pi.astype(np.float32), float(state["score"])])
        if collect_tree:
            root = getattr(mcts, "last_root", None)
            if root is not None:
                for tvec, tpi, tval in mcts.extract_training_nodes(
                        root, min_visits=tree_min_visits,
                        max_nodes=tree_max_nodes):
                    tree_samples.append((tvec, tpi, np.float32(tval),
                                         np.float32(tree_w)))
        state, _gain, done = real_model.step(
            state, action, int(sample_fruit(rng)))
        move += 1
        if done:
            break
    final_score = float(state["score"])
    out = []
    for vec, pi, score_before in samples:
        z = (final_score - score_before) / SCORE_NORM
        out.append((vec, pi, np.float32(z), np.float32(1.0)))
    out.extend(tree_samples)
    return out, final_score, len(samples)


def _load_net(ckpt_path, cfg):
    net = build_net(cfg)
    net.eval()
    return net


def _try_reload(net, ckpt_path, last_mtime):
    """Reload weights from ckpt_path if it changed. Returns new mtime."""
    try:
        m = os.path.getmtime(ckpt_path)
    except OSError:
        return last_mtime
    if m == last_mtime:
        return last_mtime
    try:
        ck = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(ck["net"])
        net.eval()
        return m
    except Exception:
        # atomic os.replace in save_checkpoint makes this rare; just retry later
        return last_mtime


def continuous_worker(ckpt_path, cfg, base_seed, worker_id,
                      sample_q, stop_event, reload_sec, max_game_seconds):
    """Resident actor: play games forever, push samples, reload weights.

    Pushes ``(worker_id, samples, final_score, n_moves)`` tuples onto
    ``sample_q``. Exits cleanly when ``stop_event`` is set (after finishing the
    current game).
    """
    torch.set_num_threads(1)               # avoid oversubscription across procs
    device = torch.device("cpu")
    net = _load_net(ckpt_path, cfg)
    last_mtime = 0.0
    if ckpt_path and os.path.exists(ckpt_path):
        last_mtime = _try_reload(net, ckpt_path, 0.0)
    g = 0
    last_reload = time.time()
    # Endgame reseed: keep a per-worker pool of realistic near-endgame start
    # states and begin most episodes from there (skipping the opening) so the
    # learner concentrates the search budget on dangerous endgames. The pool is
    # periodically regenerated from a cheap policy rollout as the policy
    # improves. Disabled -> classic full games from an empty board.
    endgame = bool(cfg.get("endgame_reseed", False))
    pool = []
    pool_cap = int(cfg.get("endgame_pool_capacity", 200))
    refresh_every = int(cfg.get("endgame_refresh_games", 20))
    tail_min = int(cfg.get("endgame_tail_min", 20))
    tail_max = int(cfg.get("endgame_tail_max", 50))
    since_refresh = refresh_every            # force an initial pool fill
    prng = np.random.default_rng(base_seed * 7919 + worker_id + 1)
    while not stop_event.is_set():
        seed = base_seed + worker_id * 100003 + g
        g += 1
        if endgame:
            if not pool or since_refresh >= refresh_every:
                traj = rollout_states(net, device, cfg, seed)
                n = len(traj)
                lo = max(0, n - tail_max)
                hi = max(lo + 1, n - tail_min)   # exclude the last tail_min
                for st in traj[lo:hi]:
                    pool.append(st)
                if len(pool) > pool_cap:
                    del pool[:len(pool) - pool_cap]
                since_refresh = 0
                if stop_event.is_set():
                    break
            start = pool[int(prng.integers(0, len(pool)))]
            samples, fs, nm = play_game_from_state(
                net, device, cfg, seed, copy.deepcopy(start))
            since_refresh += 1
        else:
            samples, fs, nm = play_game(net, device, cfg, seed,
                                        max_seconds=max_game_seconds)
        if not samples:
            continue
        # push the finished game; retry so we don't drop data, but stay
        # responsive to shutdown.
        payload = (worker_id, samples, fs, nm)
        while not stop_event.is_set():
            try:
                sample_q.put(payload, timeout=1.0)
                break
            except queue.Full:
                continue
        if time.time() - last_reload >= reload_sec:
            last_mtime = _try_reload(net, ckpt_path, last_mtime)
            last_reload = time.time()


def _worker(args):
    """Legacy batched worker: load net on CPU and play ``n_games``."""
    ckpt_path, cfg, base_seed, n_games, worker_id = args
    torch.set_num_threads(1)
    device = torch.device("cpu")
    net = _load_net(ckpt_path, cfg)
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(ck["net"])
    net.eval()
    all_samples, scores, moves = [], [], []
    for g in range(n_games):
        seed = base_seed + worker_id * 100003 + g
        s, fs, nm = play_game(net, device, cfg, seed)
        all_samples.extend(s)
        scores.append(fs)
        moves.append(nm)
    return all_samples, scores, moves


def generate_selfplay(ckpt_path, cfg, base_seed, num_games, num_workers):
    """Legacy parallel self-play across ``num_workers`` processes (one round).

    Returns (samples, scores, total_moves, elapsed_sec).
    """
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    per = [num_games // num_workers] * num_workers
    for i in range(num_games % num_workers):
        per[i] += 1
    jobs = [(ckpt_path, cfg, base_seed, per[w], w)
            for w in range(num_workers) if per[w] > 0]

    t0 = time.time()
    if num_workers == 1 or len(jobs) == 1:
        results = [_worker(jobs[0])] if jobs else []
    else:
        with ctx.Pool(processes=len(jobs)) as pool:
            results = pool.map(_worker, jobs)
    elapsed = time.time() - t0

    samples, scores, moves = [], [], []
    for s, sc, mv in results:
        samples.extend(s)
        scores.extend(sc)
        moves.extend(mv)
    return samples, scores, sum(moves), elapsed
