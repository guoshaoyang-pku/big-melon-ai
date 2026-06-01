"""AlphaZero-style MCTS for single-player Suika (batched / leaf-parallel).

* PUCT selection with network priors P(a) and value-guided backups.
* Leaves are evaluated by the value network (return-to-go estimate); no random
  rollout to the end.
* Stochasticity (the fruit-after-next type) is handled by determinised
  sampling: a child's ``next`` preview is sampled once at creation with a
  seeded RNG (allowed per spec).
* BATCHED EVALUATION: instead of one network call per simulation, we gather up
  to ``eval_batch`` leaves using *virtual loss* (so concurrent descents pick
  different paths), do the pymunk expansion for each, then run a single batched
  network forward. This amortises MPS per-call overhead and slashes net cost;
  physics (pymunk) remains the dominant cost. Set eval_batch=1 for the classic
  one-at-a-time behaviour.

Edge stats: immediate normalised reward R(s,a); the backed-up value of an edge
is R + return-to-go from the child, so Q is a real return-to-go estimate
consistent with the value-network target. Robust to game-over leaves.
"""
import math
import numpy as np

from common import encode_state, SCORE_NORM
from model import SuikaModel, sample_fruit


class _Node:
    __slots__ = ("state", "is_terminal", "expanded", "pending", "P", "N", "W",
                 "VL", "children", "reward", "vec")

    def __init__(self, state, is_terminal=False):
        self.state = state
        self.is_terminal = is_terminal
        self.expanded = False
        self.pending = False     # awaiting batched network evaluation
        self.P = None            # prior over K actions
        self.N = None            # visit counts per action
        self.W = None            # total value per action
        self.VL = None           # virtual-loss counts per action
        self.children = {}       # action -> _Node
        self.reward = None       # immediate norm reward per action
        self.vec = None


class MCTS:
    def __init__(self, net, device, K=16, num_simulations=500,
                 c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25,
                 max_fruits=80, fast_model=True, seed=0,
                 eval_batch=16, virtual_loss=1.0, root_forced_visits=0,
                 boundary_features=False):
        self.net = net
        self.device = device
        self.K = K
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dir_alpha = dirichlet_alpha
        self.dir_eps = dirichlet_eps
        self.max_fruits = max_fruits
        self.model = SuikaModel(K=K, fast=fast_model)
        self.rng = np.random.default_rng(seed)
        self.eval_batch = max(1, int(eval_batch))
        self.vl = float(virtual_loss)
        self.root_forced_visits = max(0, int(root_forced_visits))
        self.boundary_features = bool(boundary_features)

    # -- helpers ------------------------------------------------------------
    def _vec(self, node):
        if node.vec is None:
            node.vec = encode_state(node.state, self.K, self.max_fruits,
                                    boundary_features=self.boundary_features)
        return node.vec

    def _init_arrays(self, node):
        node.N = np.zeros(self.K, dtype=np.float64)
        node.W = np.zeros(self.K, dtype=np.float64)
        node.VL = np.zeros(self.K, dtype=np.float64)
        node.reward = np.full(self.K, np.nan, dtype=np.float64)

    def _set_priors(self, node, probs):
        node.P = probs.astype(np.float64)
        node.expanded = True
        node.pending = False

    def _select_action(self, node):
        eff_n = node.N + node.VL                    # virtual loss inflates N
        total = eff_n.sum()
        sqrt_total = math.sqrt(max(1.0, total))
        q = np.zeros(self.K)
        nz = eff_n > 0
        q[nz] = node.W[nz] / eff_n[nz]              # pending paths pull Q down
        u = self.c_puct * node.P * sqrt_total / (1.0 + eff_n)
        return int(np.argmax(q + u))

    # -- one descent collecting a leaf, applying virtual loss ---------------
    def _descend(self, root):
        """Return (path, leaf) where path is [(node, action), ...]. The leaf is
        an unexpanded/terminal node needing evaluation, or None if we hit an
        already-pending node (caller flushes the current batch)."""
        path = []
        node = root
        while True:
            if node.is_terminal:
                return path, node
            if not node.expanded:
                # reached an unexpanded node; if already queued this round, bail
                return (path, None) if node.pending else (path, node)
            a = self._select_action(node)
            node.VL[a] += self.vl
            path.append((node, a))
            if a in node.children:
                node = node.children[a]
                continue
            # create child via physics with a freshly sampled next-preview
            nf = sample_fruit(self.rng)
            ns, gain, over = self.model.step(node.state, a, nf)
            node.reward[a] = gain / SCORE_NORM
            child = _Node(ns, is_terminal=over)
            node.children[a] = child
            return path, child

    def _backup(self, path, leaf_value):
        G = leaf_value
        for parent, a in reversed(path):
            G = parent.reward[a] + G
            parent.N[a] += 1
            parent.W[a] += G
            parent.VL[a] -= self.vl                 # release virtual loss

    def _expand_root_action(self, root, action):
        """Force one rollout through a specific root action.

        This is used only for root exploration diagnostics. It consumes a real
        simulation and backs up the same reward/value units as normal PUCT.
        """
        nf = sample_fruit(self.rng)
        ns, gain, over = self.model.step(root.state, action, nf)
        root.reward[action] = gain / SCORE_NORM
        child = _Node(ns, is_terminal=over)
        root.children[action] = child
        if over:
            self._backup([(root, action)], 0.0)
            return
        self._init_arrays(child)
        probs, vals = self.net.infer_batch(self._vec(child)[None, :],
                                           self.device)
        self._set_priors(child, probs[0])
        self._backup([(root, action)], float(vals[0]))

    def _release_vl(self, path):
        for parent, a in path:
            parent.VL[a] -= self.vl

    # -- public search ------------------------------------------------------
    def run(self, state, add_noise=True):
        root = _Node(state, is_terminal=bool(state.get("game_over")))
        if root.is_terminal:
            return np.ones(self.K) / self.K, root
        # expand root (single batched eval of size 1)
        self._init_arrays(root)
        probs, _ = self.net.infer_batch(self._vec(root)[None, :], self.device)
        self._set_priors(root, probs[0])
        if add_noise:
            noise = self.rng.dirichlet([self.dir_alpha] * self.K)
            root.P = (1 - self.dir_eps) * root.P + self.dir_eps * noise

        sims = 0
        for _ in range(self.root_forced_visits):
            for action in range(self.K):
                if sims >= self.num_simulations:
                    break
                self._expand_root_action(root, action)
                sims += 1
            if sims >= self.num_simulations:
                break
        while sims < self.num_simulations:
            batch_paths, batch_leaves = [], []
            target = min(self.eval_batch, self.num_simulations - sims)
            while len(batch_leaves) < target:
                path, leaf = self._descend(root)
                if leaf is None:
                    self._release_vl(path)          # hit pending node -> flush
                    break
                sims += 1
                if leaf.is_terminal:
                    self._backup(path, 0.0)
                elif leaf.expanded:
                    # already-evaluated node selected as leaf: backup its
                    # bootstrap value via a quick single eval (rare).
                    _, v = self.net.infer_batch(self._vec(leaf)[None, :],
                                                self.device)
                    self._backup(path, float(v[0]))
                else:
                    leaf.pending = True
                    self._init_arrays(leaf)
                    batch_paths.append(path)
                    batch_leaves.append(leaf)
            if batch_leaves:
                vecs = np.stack([self._vec(lf) for lf in batch_leaves])
                probs_b, vals_b = self.net.infer_batch(vecs, self.device)
                for lf, path, pr, v in zip(batch_leaves, batch_paths,
                                           probs_b, vals_b):
                    self._set_priors(lf, pr)
                    self._backup(path, float(v))
            elif target > 0 and sims < self.num_simulations:
                # nothing collected and not advanced (all paths pending); force
                # one sequential sim to guarantee progress.
                path, leaf = self._descend(root)
                if leaf is None:
                    self._release_vl(path)
                    # give virtual loss a chance to clear; advance counter
                    sims += 1
                    continue
                sims += 1
                if leaf.is_terminal:
                    self._backup(path, 0.0)
                else:
                    self._init_arrays(leaf)
                    _, v = self.net.infer_batch(self._vec(leaf)[None, :],
                                                self.device)
                    probs_b, _ = self.net.infer_batch(self._vec(leaf)[None, :],
                                                      self.device)
                    self._set_priors(leaf, probs_b[0])
                    self._backup(path, float(v[0]))
        self.last_root = root
        return root.N / max(1.0, root.N.sum()), root

    def extract_training_nodes(self, root, min_visits=8, max_nodes=16,
                               include_root=False):
        """Export high-visit search nodes (root excluded by default) as extra
        training tuples ``(vec, pi, value)``.

        ``pi`` is the node's normalised visit distribution; ``value`` is the
        visit-weighted mean action value Q -- a return-to-go bootstrap estimate
        in the same normalised units as the value-network target. The root is
        skipped by default since the caller already records it with the true
        Monte-Carlo return ``z``; descendant nodes carry only the cheaper
        bootstrap target, so callers typically down-weight their value loss.
        High-visit children are prioritised when ``max_nodes`` caps the export.
        """
        out = []
        stack = [(root, True)]
        while stack and len(out) < max_nodes:
            node, is_root = stack.pop()
            exportable = node.expanded and node.N is not None and not (
                is_root and not include_root)
            if exportable:
                visits = float(node.N.sum())
                if visits >= min_visits:
                    pi = node.N / max(1.0, visits)
                    nz = node.N > 0
                    value = float(node.W[nz].sum() / max(1.0, node.N[nz].sum())) if nz.any() else 0.0
                    vec = node.vec if node.vec is not None else encode_state(
                        node.state, self.K, self.max_fruits,
                        boundary_features=self.boundary_features)
                    out.append((vec.astype(np.float32), pi.astype(np.float32), np.float32(value)))
            if node.children:
                kids = sorted(node.children.items(),
                              key=lambda kv: float(node.N[kv[0]]) if node.N is not None else 0.0)
                stack.extend((child, False) for _a, child in kids)
        return out

    def policy(self, state, temperature=1.0, add_noise=True):
        counts, root = self.run(state, add_noise=add_noise)
        self.last_root = root
        n = root.N
        if temperature <= 1e-6:
            pi = np.zeros(self.K)
            pi[int(np.argmax(n))] = 1.0
        else:
            nt = np.power(n, 1.0 / temperature)
            pi = nt / max(nt.sum(), 1e-9)
        action = int(self.rng.choice(self.K, p=pi))
        return action, pi
