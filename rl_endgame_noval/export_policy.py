#!/usr/bin/env python3
"""Export a trained policy/value checkpoint for lightweight deployment."""
import argparse
import json
import os

import torch
import yaml

from common import input_dim
from net import build_net


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "latest.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "checkpoints", "policy_value_scripted.pt"))
    ap.add_argument("--metadata-out", default="")
    ap.add_argument("--policy-only", action="store_true",
                    help="metadata flag for deployments that consume only logits")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    net = build_net(cfg).cpu()
    ck = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(ck["net"])
    net.eval()

    in_dim = input_dim(cfg["max_fruits"], cfg.get("boundary_features", False))
    dummy = torch.zeros((1, in_dim), dtype=torch.float32)
    scripted = torch.jit.trace(net, dummy)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    scripted.save(args.out)

    meta = {
        "source_checkpoint": os.path.abspath(args.ckpt),
        "checkpoint_step": int(ck.get("step", 0)),
        "arch": cfg.get("arch", "mlp"),
        "K": int(cfg["K"]),
        "max_fruits": int(cfg["max_fruits"]),
        "input_dim": int(in_dim),
        "boundary_features": bool(cfg.get("boundary_features", False)),
        "policy_only": bool(args.policy_only or ck.get("policy_only", False)
                            or ck.get("cfg", {}).get("policy_only", False)),
        "output": os.path.abspath(args.out),
    }
    meta_path = args.metadata_out or (args.out + ".json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
