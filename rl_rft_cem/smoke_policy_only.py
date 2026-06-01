#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isolated smoke: verifies the policy-only / full-game setup WITHOUT writing
to checkpoints/, data/ or logs/. Checks:
  1) endgame_reseed off -> a self-play game runs full length to game_over (~100+)
  2) resume: init latest.pt (step 8900) loads into the configured net
  3) value head frozen + value_coef 0 -> value loss == 0
  4) batch_size=1024 forward+backward on MPS does not OOM
"""
import os, time, copy
import numpy as np
import torch
import yaml

from common import input_dim
from net import build_net, pick_device
from train import train_steps, ReplayBuffer, load_checkpoint
from selfplay import play_game

HERE = os.path.dirname(os.path.abspath(__file__))

cfg = yaml.safe_load(open(os.path.join(HERE, "config_policy_only.yaml")))
print(f"[smoke] endgame_reseed={cfg['endgame_reseed']} use_value_net={cfg['use_value_net']} "
      f"value_coef={cfg['value_coef']} batch_size={cfg['batch_size']} lr={cfg['lr']}")

device = pick_device(cfg.get("device", "mps"))
in_dim = input_dim(cfg["max_fruits"], cfg.get("boundary_features", False))
K = cfg["K"]

# 2) resume: load init checkpoint
net = build_net(cfg).to(device)
latest = os.path.join(HERE, "checkpoints", "latest.pt")
step, _ = load_checkpoint(latest, net, None, device=device)
print(f"[smoke] RESUME ok: init latest.pt step={step}")

# 3) freeze value head (policy-only)
value_coef = 0.0
frozen = 0
if hasattr(net, "value_head"):
    for p in net.value_head.parameters():
        p.requires_grad_(False); frozen += 1
print(f"[smoke] froze {frozen} value-head params; value_coef={value_coef}")

# 4) batch=1024 fwd/bwd on device
opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                       lr=cfg["lr"], weight_decay=cfg["weight_decay"])
buf = ReplayBuffer(8192, in_dim, K, max_fruits=cfg["max_fruits"])
rng = np.random.default_rng(0)
fake = []
for _ in range(4096):
    vec = rng.standard_normal(in_dim).astype(np.float32)
    pi = rng.random(K).astype(np.float32); pi /= pi.sum()
    fake.append((vec, pi, np.float32(rng.standard_normal()), np.float32(1.0)))
buf.add_many(fake)
if device.type == "mps":
    try: torch.mps.empty_cache()
    except Exception: pass
t0 = time.time()
pl, vl, ent, t1 = train_steps(net, opt, buf, device, n_steps=8, batch_size=cfg["batch_size"],
                     value_coef=value_coef, entropy_coef=cfg.get("entropy_coef", 0.0))
dt = time.time() - t0
mem = 0.0
if device.type == "mps":
    try: mem = torch.mps.current_allocated_memory() / 1e6
    except Exception: mem = -1
print(f"[smoke] train_steps device={device} batch={cfg['batch_size']} "
      f"8 steps in {dt:.2f}s ({8/dt:.2f} st/s) policy_loss={pl:.4f} value_loss={vl:.4f} "
      f"entropy={ent:.3f} top1={t1:.3f} "
      f"mps_alloc={mem:.0f}MB  -> {'NO OOM' if True else ''}")
assert vl == 0.0, "value loss must be 0 in policy-only mode"

# 1) full-game length with endgame off (low sims for speed; length is policy-driven)
scfg = copy.deepcopy(cfg); scfg["num_simulations"] = 24
net.eval()
t0 = time.time()
samples, fs, nm = play_game(net, torch.device("cpu"), scfg, seed=0, max_seconds=0)
print(f"[smoke] full game (endgame off, sims=24): moves={nm} final_score={fs:.0f} "
      f"samples={len(samples)} in {time.time()-t0:.1f}s -> "
      f"{'FULL-GAME DISTRIBUTION OK (>=80 moves)' if nm >= 80 else 'WARN short game'}")

print("[smoke] ALL CHECKS DONE")
