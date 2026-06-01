#!/usr/bin/env python3
"""Safely merge standalone replay NPZ files into a target replay buffer."""
import argparse
import os
import shutil
import time

import numpy as np


def _load(path):
    d = np.load(path, allow_pickle=False)
    n = int(d["size"][0]) if "size" in d.files else len(d["vecs"])
    vecs = d["vecs"][:n].astype(np.float32)
    pis = d["pis"][:n].astype(np.float32)
    zs = d["zs"][:n].astype(np.float32)
    if "ws" in d.files:
        ws = d["ws"][:n].astype(np.float32)
    else:
        ws = np.ones((n,), dtype=np.float32)
    return vecs, pis, zs, ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--capacity", type=int, default=0,
                    help="0 keeps all samples; positive keeps most recent N")
    args = ap.parse_args()

    if not os.path.exists(args.target):
        raise SystemExit("target replay not found: %s" % args.target)
    chunks = [_load(args.target)]
    base_dim = chunks[0][0].shape[1]
    base_k = chunks[0][1].shape[1]
    for src in args.sources:
        if os.path.abspath(src) == os.path.abspath(args.target):
            raise SystemExit("source equals target: %s" % src)
        if not os.path.exists(src):
            raise SystemExit("source not found: %s" % src)
        chunk = _load(src)
        if chunk[0].shape[1] != base_dim:
            raise SystemExit("vec dim mismatch for %s" % src)
        if chunk[1].shape[1] != base_k:
            raise SystemExit("policy dim mismatch for %s" % src)
        chunks.append(chunk)

    vecs = np.concatenate([c[0] for c in chunks], axis=0)
    pis = np.concatenate([c[1] for c in chunks], axis=0)
    zs = np.concatenate([c[2] for c in chunks], axis=0)
    ws = np.concatenate([c[3] for c in chunks], axis=0)
    if args.capacity and len(vecs) > args.capacity:
        vecs, pis, zs, ws = (vecs[-args.capacity:], pis[-args.capacity:],
                             zs[-args.capacity:], ws[-args.capacity:])
    n = len(vecs)

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = "%s.bak_%s" % (args.target, ts)
    shutil.copy2(args.target, backup)
    tmp = args.target + ".tmp.npz"
    np.savez_compressed(
        tmp, vecs=vecs, pis=pis, zs=zs, ws=ws,
        pos=np.array([n], dtype=np.int64),
        size=np.array([n], dtype=np.int64),
    )
    os.replace(tmp, args.target)
    print("[merge_replay] backup:", backup)
    print("[merge_replay] target:", args.target)
    print("[merge_replay] size:", n)


if __name__ == "__main__":
    main()
