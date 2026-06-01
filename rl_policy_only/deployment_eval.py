#!/usr/bin/env python3
"""Convenience entry for policy/limited-search/full-teacher cost evaluation."""
import argparse
import json
import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "latest.pt"))
    ap.add_argument("--seeds", default="0:20")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default=os.path.join(HERE, "logs", "deployment_eval.json"))
    ap.add_argument("--limited-sims", type=int, default=32)
    ap.add_argument("--teacher-depth", type=int, default=3)
    ap.add_argument("--teacher-beam", type=int, default=8)
    ap.add_argument("--max-moves", type=int, default=0)
    ap.add_argument("--run", action="store_true",
                    help="execute the eval command; default only prints it")
    args = ap.parse_args()

    cmd = [
        sys.executable,
        os.path.join(HERE, "eval_suite.py"),
        "--config", args.config,
        "--ckpt", args.ckpt,
        "--seeds", args.seeds,
        "--agents", "policy,net,robust_v2",
        "--sims", str(args.limited_sims),
        "--device", args.device,
        "--stress",
        "--out-json", args.out_json,
        "--robust-depth", str(args.teacher_depth),
        "--robust-beam", str(args.teacher_beam),
    ]
    if args.max_moves:
        cmd.extend(["--max-moves", str(args.max_moves)])
    print(json.dumps({"command": cmd, "out_json": args.out_json},
                     ensure_ascii=False, indent=2))
    if args.run:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
