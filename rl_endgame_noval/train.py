"""Replay buffer, training step and checkpoint I/O for the AlphaZero loop.

Loss = policy cross-entropy(pi_target || policy_logits) + value MSE + L2.
Replay remains flat-vector based; mirror augmentation is applied at insertion
so both MLP and Transformer models can consume the same saved buffer.

Each sample also carries a value-loss weight ``w`` (default 1.0). Root self-play
samples use w=1.0 (their value target is the true Monte-Carlo return ``z``);
optional mined MCTS internal nodes use w<1.0 because their value target is only
a bootstrap Q estimate -- down-weighting keeps them from polluting the value
head. The on-disk format stays backward compatible: replay files without a
``ws`` field load with all weights = 1.0.
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from common import mirror_flat_vec


class ReplayBuffer:
    """Capped ring buffer of (state_vec, pi, z, w) samples, disk-persistable."""

    def __init__(self, capacity, in_dim, K, max_fruits=None,
                 boundary_features=False):
        self.capacity = int(capacity)
        self.in_dim = int(in_dim)
        self.K = int(K)
        self.max_fruits = int(max_fruits) if max_fruits is not None else (in_dim - 9) // 4
        self.boundary_features = bool(boundary_features)
        self.vecs = np.zeros((self.capacity, self.in_dim), dtype=np.float32)
        self.pis = np.zeros((self.capacity, self.K), dtype=np.float32)
        self.zs = np.zeros((self.capacity,), dtype=np.float32)
        self.ws = np.ones((self.capacity,), dtype=np.float32)
        self.size = 0
        self.pos = 0

    def _add_one(self, vec, pi, z, w=1.0):
        self.vecs[self.pos] = vec
        self.pis[self.pos] = pi
        self.zs[self.pos] = z
        self.ws[self.pos] = w
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_many(self, samples, mirror=False):
        for s in samples:
            # accept (vec, pi, z) or (vec, pi, z, w)
            if len(s) == 4:
                vec, pi, z, w = s
            else:
                vec, pi, z = s
                w = 1.0
            vec = np.asarray(vec, dtype=np.float32)
            pi = np.asarray(pi, dtype=np.float32)
            w = float(w)
            self._add_one(vec, pi, np.float32(z), w)
            if mirror:
                self._add_one(mirror_flat_vec(vec, self.max_fruits,
                                              self.boundary_features),
                              pi[::-1].copy(), np.float32(z), w)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.vecs[idx], self.pis[idx], self.zs[idx], self.ws[idx]

    def save(self, path):
        np.savez_compressed(
            path,
            vecs=self.vecs[:self.size], pis=self.pis[:self.size],
            zs=self.zs[:self.size], ws=self.ws[:self.size],
            pos=np.array([self.pos]), size=np.array([self.size]),
        )

    def load(self, path):
        if not os.path.exists(path):
            return
        d = np.load(path)
        n = int(d["size"][0]) if "size" in d.files else len(d["vecs"])
        vecs = d["vecs"][:n]
        pis = d["pis"][:n]
        zs = d["zs"][:n]
        # backward compatible: older replay files have no per-sample weight.
        if "ws" in d.files:
            ws = d["ws"][:n]
        else:
            ws = np.ones((len(vecs),), dtype=np.float32)
        if vecs.shape[1] != self.in_dim:
            raise ValueError("replay vec dim %d != expected %d" % (vecs.shape[1], self.in_dim))
        if pis.shape[1] != self.K:
            raise ValueError("replay policy dim %d != expected %d" % (pis.shape[1], self.K))
        if n > self.capacity:
            vecs, pis, zs, ws = (vecs[-self.capacity:], pis[-self.capacity:],
                                 zs[-self.capacity:], ws[-self.capacity:])
            n = self.capacity
        self.vecs[:n] = vecs
        self.pis[:n] = pis
        self.zs[:n] = zs
        self.ws[:n] = ws
        self.size = n
        self.pos = n % self.capacity


def train_steps(net, optimizer, buffer, device, n_steps, batch_size,
                value_coef=1.0):
    net.train()
    tot_p = tot_v = 0.0
    for _ in range(n_steps):
        vecs, pis, zs, ws = buffer.sample(batch_size)
        v = torch.as_tensor(vecs, device=device)
        tgt_pi = torch.as_tensor(pis, device=device)
        tgt_z = torch.as_tensor(zs, device=device)
        w = torch.as_tensor(ws, device=device)
        logits, value = net(v)
        logp = F.log_softmax(logits, dim=-1)
        policy_loss = -(tgt_pi * logp).sum(dim=1).mean()
        if value_coef != 0.0:
            # weighted value MSE: down-weighted bootstrap targets contribute less.
            sq = (value - tgt_z) ** 2
            value_loss = (w * sq).sum() / w.sum().clamp_min(1e-6)
            loss = policy_loss + value_coef * value_loss
            vloss_item = float(value_loss.item())
        else:
            # policy-only training: value head frozen, contributes no loss/grad.
            loss = policy_loss
            vloss_item = 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_p += float(policy_loss.item())
        tot_v += vloss_item
    return tot_p / n_steps, tot_v / n_steps


def save_checkpoint(path, net, optimizer, step, cfg, extra=None):
    tmp = path + ".tmp"
    payload = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "cfg": cfg,
        "time": time.time(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, net, optimizer=None, device="cpu", strict=True):
    ck = torch.load(path, map_location=device)
    net.load_state_dict(ck["net"], strict=strict)
    if optimizer is not None and ck.get("optimizer") is not None:
        optimizer.load_state_dict(ck["optimizer"])
    return ck.get("step", 0), ck.get("cfg", {})
