#!/usr/bin/env python3
"""Train a policy-only RFT student from elite/CEM trajectories."""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from common import mirror_flat_vec
from net import build_net
from train import save_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _load_npz(path, cfg, mirror=True):
    d = np.load(path, allow_pickle=False)
    n = int(d["size"][0]) if "size" in d.files else len(d["vecs"])
    vecs = np.asarray(d["vecs"][:n], dtype=np.float32)
    pis = np.asarray(d["pis"][:n], dtype=np.float32)
    ws = np.asarray(d["ws"][:n], dtype=np.float32) if "ws" in d.files else np.ones(n, dtype=np.float32)
    if len(ws) and float(np.max(ws)) <= 0.0:
        # Teacher replay uses ws as value-loss weight; policy-only exports set it to 0.
        ws = np.ones_like(ws, dtype=np.float32)
    if mirror:
        mv = np.asarray([
            mirror_flat_vec(v, cfg["max_fruits"], cfg.get("boundary_features", False))
            for v in vecs
        ], dtype=np.float32)
        mp = pis[:, ::-1].copy()
        vecs = np.concatenate([vecs, mv], axis=0)
        pis = np.concatenate([pis, mp], axis=0)
        ws = np.concatenate([ws, ws], axis=0)
    return vecs, pis, ws, json.loads(str(d["meta"])) if "meta" in d.files else {}


def _batch(rng, n, batch_size):
    return rng.integers(0, n, size=int(batch_size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config_rft_cem.yaml"))
    ap.add_argument("--data", default=None)
    ap.add_argument("--init", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "checkpoints", "rft_latest.pt"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--entropy-coef", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    data_path = _resolve(args.data or cfg.get("rft_data", "data/rft_elite_latest.npz"))
    init = _resolve(args.init or cfg.get("rft_init_checkpoint", ""))
    out = _resolve(args.out)
    device = torch.device(args.device or cfg.get("device", "cpu"))
    epochs = int(args.epochs or cfg.get("rft_train_epochs", 12))
    steps_override = int(args.steps if args.steps is not None else cfg.get("rft_train_steps", 0))
    batch_size = int(args.batch_size or cfg.get("rft_batch_size", cfg.get("batch_size", 512)))
    lr = float(args.lr if args.lr is not None else cfg.get("rft_lr", cfg.get("lr", 1e-4)))
    entropy_coef = float(args.entropy_coef if args.entropy_coef is not None else cfg.get("rft_entropy_coef", 0.0))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else cfg.get("rft_weight_decay", cfg.get("weight_decay", 0.0)))
    save_every = int(args.save_every if args.save_every is not None else cfg.get("rft_save_every", 0))

    vecs, pis, ws, data_meta = _load_npz(data_path, cfg, mirror=not args.no_mirror)
    if len(vecs) == 0:
        raise SystemExit("empty RFT dataset")
    net = build_net(cfg).to(device)
    ck = torch.load(init, map_location=device)
    net.load_state_dict(ck["net"])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(args.seed)
    n = len(vecs)
    steps_per_epoch = max(1, int(np.ceil(n / float(batch_size))))
    total_steps = steps_override if steps_override > 0 else steps_per_epoch * epochs
    log_path = os.path.join(HERE, "logs", "rft_train.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    header_needed = not os.path.exists(log_path)
    step = 0
    t0 = time.time()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["time", "step", "loss", "policy_ce", "entropy", "top1", "w_mean", "data_n"])
        while step < total_steps:
            idx = _batch(rng, n, batch_size)
            v = torch.as_tensor(vecs[idx], dtype=torch.float32, device=device)
            target = torch.as_tensor(pis[idx], dtype=torch.float32, device=device)
            w = torch.as_tensor(ws[idx], dtype=torch.float32, device=device)
            logits, _value = net(v)
            logp = F.log_softmax(logits, dim=-1)
            probs = logp.exp()
            ce = -(target * logp).sum(dim=1)
            policy_loss = (w * ce).sum() / w.sum().clamp_min(1e-6)
            entropy = -(probs * logp).sum(dim=1).mean()
            loss = policy_loss - entropy_coef * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            if step == 1 or step % 25 == 0 or step == total_steps:
                top1 = probs.max(dim=1).values.mean().item()
                row = [time.time(), step, float(loss.item()), float(policy_loss.item()),
                       float(entropy.item()), float(top1), float(w.mean().item()), int(n)]
                writer.writerow(row)
                f.flush()
                print("[rft_train] step=%d/%d loss=%.4f ce=%.4f ent=%.3f top1=%.3f" %
                      (step, total_steps, row[2], row[3], row[4], row[5]), flush=True)
            if save_every > 0 and (step % save_every == 0) and step < total_steps:
                partial = os.path.splitext(out)[0] + "_partial.pt"
                save_checkpoint(partial, net, opt, int(ck.get("step", 0)) + step,
                                cfg, extra={"rft_partial": {
                                    "data": data_path,
                                    "step": int(step),
                                    "target_steps": int(total_steps),
                                    "batch_size": batch_size,
                                    "lr": lr,
                                    "entropy_coef": entropy_coef,
                                }})
                save_checkpoint(os.path.join(HERE, "checkpoints", "latest.pt"),
                                net, opt, int(ck.get("step", 0)) + step, cfg,
                                extra={"rft_partial": {
                                    "data": data_path,
                                    "step": int(step),
                                    "target_steps": int(total_steps),
                                    "batch_size": batch_size,
                                    "lr": lr,
                                    "entropy_coef": entropy_coef,
                                }})
                print("[rft_train] saved partial checkpoint step=%d path=%s" %
                      (step, partial), flush=True)

    extra = {
        "rft": {
            "data": data_path,
            "data_meta": data_meta,
            "samples": int(n),
            "epochs": epochs,
            "steps": int(total_steps),
            "batch_size": batch_size,
            "lr": lr,
            "entropy_coef": entropy_coef,
            "weighted_ce": True,
            "value_off": True,
            "seconds": time.time() - t0,
        }
    }
    save_checkpoint(out, net, opt, int(ck.get("step", 0)) + total_steps, cfg, extra=extra)
    save_checkpoint(os.path.join(HERE, "checkpoints", "latest.pt"), net, opt,
                    int(ck.get("step", 0)) + total_steps, cfg, extra=extra)
    print(json.dumps({"checkpoint": out, "latest": os.path.join(HERE, "checkpoints", "latest.pt"), **extra["rft"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
