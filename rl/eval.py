"""Evaluation: current net+MCTS vs heuristic and random baselines.

Runs a fixed set of seeds (greedy, no Dirichlet noise) and reports mean score
and the largest fruit reached, alongside the HeuristicLookaheadAgent and a
uniform-random agent on the same seeds.
"""
import numpy as np
import torch

from net import build_net
from mcts import MCTS
from suika_env import SuikaEnv
from ai_agent import HeuristicLookaheadAgent


def _max_fruit(state):
    return max((f["type"] for f in state["fruits"]), default=-1)


def eval_net(net, device, cfg, seeds, num_simulations=None, max_moves=None):
    K = cfg["K"]
    sims = num_simulations or cfg["num_simulations"]
    mm = cfg["max_moves"] if max_moves is None else max_moves
    mm = int(mm or 0)
    scores, maxf = [], []
    for seed in seeds:
        env = SuikaEnv(seed=seed)
        state = env.get_state()
        mcts = MCTS(net, device, K=K, num_simulations=sims,
                    c_puct=cfg["c_puct"], max_fruits=cfg["max_fruits"],
                    fast_model=True, seed=seed,
                    eval_batch=cfg.get("eval_batch", 16),
                    root_forced_visits=cfg.get("root_forced_visits", 0),
                    boundary_features=cfg.get("boundary_features", False))
        move = 0
        while not state["game_over"]:
            if mm > 0 and move >= mm:
                break
            action, _ = mcts.policy(state, temperature=0.0, add_noise=False)
            state, _r, done, _ = env.step_column(action, num_columns=K)
            move += 1
            if done:
                break
        scores.append(float(state["score"]))
        maxf.append(_max_fruit(state))
    return float(np.mean(scores)), int(max(maxf)), scores


def eval_heuristic(seeds, K=16, max_moves=0):
    scores, maxf = [], []
    for seed in seeds:
        env = SuikaEnv(seed=seed)
        state = env.get_state()
        agent = HeuristicLookaheadAgent(num_columns=K, seed=seed)
        move = 0
        while not state["game_over"]:
            if max_moves > 0 and move >= max_moves:
                break
            x = agent.decide(state)
            state, _r, done, _ = env.step(x)
            move += 1
            if done:
                break
        scores.append(float(state["score"]))
        maxf.append(_max_fruit(state))
    return float(np.mean(scores)), int(max(maxf)), scores


def eval_random(seeds, K=16, max_moves=0):
    scores, maxf = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        env = SuikaEnv(seed=seed)
        state = env.get_state()
        move = 0
        while not state["game_over"]:
            if max_moves > 0 and move >= max_moves:
                break
            action = int(rng.integers(0, K))
            state, _r, done, _ = env.step_column(action, num_columns=K)
            move += 1
            if done:
                break
        scores.append(float(state["score"]))
        maxf.append(_max_fruit(state))
    return float(np.mean(scores)), int(max(maxf)), scores


class UniformPriorNet:
    """Network-like object for evaluating search quality without learned priors."""

    def __init__(self, K, value=0.0):
        self.K = int(K)
        self.value = float(value)

    def infer_batch(self, vecs, device=None):
        b = len(vecs)
        probs = np.full((b, self.K), 1.0 / self.K, dtype=np.float32)
        values = np.full((b,), self.value, dtype=np.float32)
        return probs, values

    def infer(self, vec, device=None):
        probs, values = self.infer_batch(np.asarray(vec)[None, :], device)
        return probs[0], float(values[0])

    def eval(self):
        return self


def eval_uniform_mcts(cfg, seeds, num_simulations=None, max_moves=None):
    """Evaluate MCTS with uniform priors / constant value, isolating search."""
    net = UniformPriorNet(cfg["K"])
    return eval_net(net, torch.device("cpu"), cfg, seeds,
                    num_simulations=num_simulations, max_moves=max_moves)


def load_eval_net(cfg, ckpt_path, device):
    """Build the configured net and load a checkpoint for standalone eval."""
    net = build_net(cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ck["net"])
    net.eval()
    return net, int(ck.get("step", 0))
