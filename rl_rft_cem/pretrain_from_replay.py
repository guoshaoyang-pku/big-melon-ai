"""Warm-start a Transformer policy/value net from existing flat replay.

The replay file is not modified. Old flattened vectors are sampled directly;
the Transformer converts them to global/fruit tokens inside ``forward``.
"""
import argparse
import os
import time

import yaml
import numpy as np
import torch
import torch.nn.functional as F

from common import input_dim, mirror_flat_vec
from net import build_net, pick_device
from train import save_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))


def _coerce(v):
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def _resolve_path(path):
    if path is None or os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    here_path = os.path.join(HERE, path)
    if os.path.exists(here_path) or os.path.dirname(path):
        return here_path
    return path


def load_cfg(path, overrides, smoke=False):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["arch"] = "transformer"
    if smoke:
        cfg.update({
            "d_model": 64,
            "token_dim": 64,
            "n_transformer_layers": 1,
            "n_heads": 4,
            "dropout": 0.0,
            "device": "cpu",
        })
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        cfg[k] = _coerce(v)
    return cfg


def _read_replay(path, in_dim, K):
    if not os.path.exists(path):
        print("[pretrain] replay not found, skipping: %s" % path)
        return None
    d = np.load(path)
    n = int(d["size"][0]) if "size" in d.files else len(d["vecs"])
    if n <= 0:
        print("[pretrain] replay empty, skipping: %s" % path)
        return None
    vecs = d["vecs"][:n].astype(np.float32)
    pis = d["pis"][:n].astype(np.float32)
    zs = d["zs"][:n].astype(np.float32)
    if vecs.shape[1] != in_dim:
        raise ValueError("replay vec dim %d != expected %d" % (vecs.shape[1], in_dim))
    if pis.shape[1] != K:
        raise ValueError("replay policy dim %d != expected %d" % (pis.shape[1], K))
    return vecs, pis, zs


def _augment_batch(vecs_b, pis_b, max_fruits, boundary_features=False):
    flip = np.random.random(len(vecs_b)) < 0.5
    if not flip.any():
        return vecs_b, pis_b
    vecs_b = vecs_b.copy()
    pis_b = pis_b.copy()
    for i in np.where(flip)[0]:
        vecs_b[i] = mirror_flat_vec(vecs_b[i], max_fruits, boundary_features)
        pis_b[i] = pis_b[i][::-1]
    return vecs_b, pis_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--replay", default=os.path.join(HERE, "data", "replay.npz"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--policy-only", action="store_true",
                    help="set value loss coefficient to 0 for pure policy distillation")
    ap.add_argument("--value-coef", type=float, default=None,
                    help="override cfg value_coef without editing config.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    if args.smoke:
        args.steps = min(args.steps, 20)
        args.batch_size = min(args.batch_size, 64)

    args.config = _resolve_path(args.config)
    args.replay = _resolve_path(args.replay)
    args.out = _resolve_path(args.out)

    cfg = load_cfg(args.config, args.set, smoke=args.smoke)
    if args.policy_only:
        cfg["value_coef"] = 0.0
        cfg["policy_only"] = True
    if args.value_coef is not None:
        cfg["value_coef"] = float(args.value_coef)
    if args.device:
        cfg["device"] = args.device
    device = pick_device(cfg.get("device", "mps"))
    boundary_features = bool(cfg.get("boundary_features", False))
    in_dim = input_dim(cfg["max_fruits"], boundary_features)
    K = int(cfg["K"])
    data = _read_replay(args.replay, in_dim, K)
    if data is None:
        return 0
    vecs, pis, zs = data

    np.random.seed(int(cfg.get("base_seed", 12345)))
    torch.manual_seed(int(cfg.get("base_seed", 12345)))
    net = build_net(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg.get("lr", 1e-3)),
                           weight_decay=float(cfg.get("weight_decay", 1e-4)))

    t0 = time.time()
    last_p = last_v = 0.0
    for step in range(1, args.steps + 1):
        idx = np.random.randint(0, len(vecs), size=args.batch_size)
        xb = vecs[idx].copy()
        pb = pis[idx].copy()
        zb = zs[idx]
        if cfg.get("mirror_augmentation", True):
            xb, pb = _augment_batch(
                xb, pb, int(cfg["max_fruits"]), boundary_features)
        x = torch.as_tensor(xb, device=device)
        target_pi = torch.as_tensor(pb, device=device)
        target_z = torch.as_tensor(zb, device=device)
        logits, value = net(x)
        policy_loss = -(target_pi * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()
        value_loss = F.mse_loss(value, target_z)
        loss = policy_loss + float(cfg.get("value_coef", 1.0)) * value_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_p, last_v = float(policy_loss.item()), float(value_loss.item())
        if step == 1 or step == args.steps or step % 50 == 0:
            print("[pretrain] step=%d/%d policy_loss=%.4f value_loss=%.4f" %
                  (step, args.steps, last_p, last_v), flush=True)

    if args.out:
        out = args.out
    elif args.smoke:
        out = os.path.join(HERE, "_smoke", "checkpoints", "pretrain_from_replay.pt")
    else:
        out = os.path.join(HERE, "checkpoints", "pretrain_transformer.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_checkpoint(out, net, opt, args.steps, cfg, extra={
        "source_replay": os.path.abspath(args.replay),
        "pretrain_samples": int(len(vecs)),
        "pretrain_seconds": time.time() - t0,
        "last_policy_loss": last_p,
        "last_value_loss": last_v,
        "value_coef": float(cfg.get("value_coef", 1.0)),
        "policy_only": bool(cfg.get("policy_only", False)),
    })
    print("[pretrain] saved %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
