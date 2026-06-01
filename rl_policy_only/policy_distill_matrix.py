#!/usr/bin/env python3
"""Run or print a small matrix of policy-only distillation jobs.

This is intentionally a thin orchestrator around ``pretrain_from_replay.py`` so
existing checkpoint/replay formats remain untouched.  By default it only prints
commands; pass ``--run`` for local experiments.
"""
import argparse
import json
import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))


def _split_csv(text, cast=str):
    return [cast(x) for x in str(text).split(",") if str(x).strip()]


def build_jobs(args):
    replays = _split_csv(args.replays)
    seeds = _split_csv(args.seeds, int)
    value_coefs = _split_csv(args.value_coefs, float)
    jobs = []
    for replay in replays:
        stem = os.path.splitext(os.path.basename(replay))[0]
        for seed in seeds:
            for vc in value_coefs:
                tag = "policy" if vc == 0.0 else "pv%.3g" % vc
                out = os.path.join(args.out_dir, "%s_seed%d_%s.pt" %
                                   (stem, seed, tag))
                cmd = [
                    sys.executable,
                    os.path.join(HERE, "pretrain_from_replay.py"),
                    "--config", args.config,
                    "--replay", replay,
                    "--out", out,
                    "--steps", str(args.steps),
                    "--batch-size", str(args.batch_size),
                    "--device", args.device,
                    "--value-coef", str(vc),
                    "--set", "base_seed=%d" % seed,
                ]
                if vc == 0.0:
                    cmd.append("--policy-only")
                if args.boundary_features:
                    cmd.extend(["--set", "boundary_features=true"])
                jobs.append({
                    "replay": replay,
                    "seed": seed,
                    "value_coef": vc,
                    "out": out,
                    "cmd": cmd,
                })
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--replays", default=os.path.join(HERE, "data", "teacher_replay.npz"),
                    help="comma-separated replay npz paths")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "checkpoints", "distill_matrix"))
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seeds", default="12345")
    ap.add_argument("--value-coefs", default="0",
                    help="comma list; use 0 for pure policy-only")
    ap.add_argument("--boundary-features", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="execute jobs sequentially instead of only printing")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs(args)
    if args.json:
        print(json.dumps(jobs, ensure_ascii=False, indent=2))
    else:
        for job in jobs:
            print(" ".join('"%s"' % x if " " in x else x for x in job["cmd"]))
    if args.run:
        os.makedirs(args.out_dir, exist_ok=True)
        for job in jobs:
            subprocess.run(job["cmd"], check=True)


if __name__ == "__main__":
    main()
