#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析 AlphaZero 合成大西瓜训练日志，绘制 loss / score 曲线（可重复运行 / 可 --watch）。

只读训练日志，不触碰训练进程与 rl/ 下训练核心文件。

支持两类日志行：
  1) 旧版按轮:
       iter N | selfplay 16 games, 1732 moves, 1732 samples in 548.0s (1.75 games/min, 3.16 moves/s) | score mean=780 max=1263 | buffer=1732
       iter N | train 400 steps in 2.0s | policy_loss=2.6164 value_loss=0.0063
  2) 新版 off-policy（主）:
       [2026-05-31 22:19:50] STATS buffer=1515/500000 (~2MB) | produced=1515 games=13 (1.98 g/min 3.82 mv/s) | steps=96 (0.26 st/s 24576 samp reuse~16.2x) | loss p=1.4794 v=0.0096 | score mean=907 max=1113 | ...
"""
import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime

DEFAULT_LOG = "/Users/guoshaoyang/Desktop/workdir/兴趣项目/合成大西瓜/suika/rl/logs/train.log"
DEFAULT_OUT = "/Users/guoshaoyang/Desktop/workdir/兴趣项目/合成大西瓜/traces/loss_curve.png"
DEFAULT_CSV = "/Users/guoshaoyang/Desktop/workdir/兴趣项目/合成大西瓜/traces/metrics.csv"

# ----- 稳健的字段级正则（字段可能缺失也不报错）-----
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
RE_BUFFER = re.compile(r"buffer=(\d+)")
RE_PRODUCED = re.compile(r"produced=(\d+)")
RE_GAMES = re.compile(r"games=(\d+)")
RE_STEPS = re.compile(r"steps=(\d+)")
RE_GPM = re.compile(r"\(([\d.]+)\s*g/min")
RE_MVS = re.compile(r"([\d.]+)\s*mv/s")
RE_STS = re.compile(r"([\d.]+)\s*st/s")
RE_SAMP = re.compile(r"st/s\s+(\d+)\s*samp")
RE_REUSE = re.compile(r"reuse~([\d.]+)x")
RE_LOSS_P = re.compile(r"loss\s+p=([-\d.]+)")
RE_LOSS_V = re.compile(r"v=([-\d.]+)")
RE_SCORE_MEAN = re.compile(r"score\s+mean=(\d+)")
RE_SCORE_MAX = re.compile(r"max=(\d+)")

# 旧版 iter 行
RE_ITER = re.compile(r"iter\s+(\d+)")
RE_OLD_TRAIN = re.compile(r"train\s+(\d+)\s+steps.*policy_loss=([-\d.]+)\s+value_loss=([-\d.]+)")
RE_OLD_SELFPLAY_GPM = re.compile(r"\(([\d.]+)\s*games/min")
RE_OLD_SELFPLAY_MVS = re.compile(r"([\d.]+)\s*moves/s")
RE_OLD_MOVES = re.compile(r"(\d+)\s+moves")


def _g1f(rx, s):
    m = rx.search(s)
    return float(m.group(1)) if m else None


def _g1i(rx, s):
    m = rx.search(s)
    return int(m.group(1)) if m else None


def _parse_ts(line):
    m = TS_RE.search(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def parse_log(path):
    """返回 records 列表，每条是统一 schema 的 dict。"""
    records = []
    old_cum_steps = 0  # 旧版 iter 训练步数累加（用于和 steps 横轴对齐）
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        print("[plot_loss] 找不到日志: %s" % path, file=sys.stderr)
        return records

    for line in lines:
        ts = _parse_ts(line)
        rec = None

        # 新版 off-policy STATS 行
        if "STATS" in line:
            rec = {
                "kind": "stats",
                "ts": ts,
                "iter": None,
                "steps": _g1i(RE_STEPS, line),
                "buffer": _g1i(RE_BUFFER, line),
                "produced": _g1i(RE_PRODUCED, line),
                "games": _g1i(RE_GAMES, line),
                "policy_loss": _g1f(RE_LOSS_P, line),
                "value_loss": _g1f(RE_LOSS_V, line),
                "score_mean": _g1i(RE_SCORE_MEAN, line),
                "score_max": _g1i(RE_SCORE_MAX, line),
                "g_per_min": _g1f(RE_GPM, line),
                "mv_per_s": _g1f(RE_MVS, line),
                "st_per_s": _g1f(RE_STS, line),
                "samp": _g1i(RE_SAMP, line),
                "reuse": _g1f(RE_REUSE, line),
            }

        # 旧版 train 行
        elif "| train" in line and "policy_loss" in line:
            mt = RE_OLD_TRAIN.search(line)
            it = _g1i(RE_ITER, line)
            if mt:
                old_cum_steps += int(mt.group(1))
                rec = {
                    "kind": "old_train",
                    "ts": ts,
                    "iter": it,
                    "steps": old_cum_steps,
                    "buffer": _g1i(RE_BUFFER, line),
                    "produced": None,
                    "games": None,
                    "policy_loss": float(mt.group(2)),
                    "value_loss": float(mt.group(3)),
                    "score_mean": None,
                    "score_max": None,
                    "g_per_min": None,
                    "mv_per_s": None,
                    "st_per_s": None,
                    "samp": None,
                    "reuse": None,
                }

        # 旧版 selfplay 行
        elif "| selfplay" in line:
            rec = {
                "kind": "old_selfplay",
                "ts": ts,
                "iter": _g1i(RE_ITER, line),
                "steps": None,
                "buffer": _g1i(RE_BUFFER, line),
                "produced": _g1i(RE_MOVES if False else RE_OLD_MOVES, line),
                "games": None,
                "policy_loss": None,
                "value_loss": None,
                "score_mean": _g1i(RE_SCORE_MEAN, line),
                "score_max": _g1i(RE_SCORE_MAX, line),
                "g_per_min": _g1f(RE_OLD_SELFPLAY_GPM, line),
                "mv_per_s": _g1f(RE_OLD_SELFPLAY_MVS, line),
                "st_per_s": None,
                "samp": None,
                "reuse": None,
            }

        if rec is not None:
            records.append(rec)

    return records


CSV_FIELDS = [
    "kind", "ts", "iter", "steps", "buffer", "produced", "games",
    "policy_loss", "value_loss", "score_mean", "score_max",
    "g_per_min", "mv_per_s", "st_per_s", "samp", "reuse",
]


def write_csv(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            row = dict(r)
            if isinstance(row.get("ts"), datetime):
                row["ts"] = row["ts"].strftime("%Y-%m-%d %H:%M:%S")
            w.writerow({k: row.get(k) for k in CSV_FIELDS})


def _series(records, kind, xkey, ykey, require_pos_y=False):
    xs, ys = [], []
    for r in records:
        if r["kind"] != kind:
            continue
        x, y = r.get(xkey), r.get(ykey)
        if x is None or y is None:
            continue
        if require_pos_y and (y is None or y <= 0):
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def make_plot(records, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # 选择一个支持中文的字体，避免方框乱码；找不到则退回默认
    _cjk = ["PingFang SC", "Arial Unicode MS", "Hiragino Sans GB",
            "Heiti SC", "STHeiti", "Songti SC", "Hiragino Sans",
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    _names = {f.name for f in font_manager.fontManager.ttflist}
    for _c in _cjk:
        if _c in _names:
            plt.rcParams["font.sans-serif"] = [_c]
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("合成大西瓜 AlphaZero 训练曲线", fontsize=14)

    # --- 子图1: policy loss vs steps ---
    ax = axes[0][0]
    sx, sy = _series(records, "stats", "steps", "policy_loss", require_pos_y=True)
    ox, oy = _series(records, "old_train", "steps", "policy_loss", require_pos_y=True)
    if sx:
        ax.plot(sx, sy, "-o", ms=3, color="tab:blue", label="off-policy (新)")
    if ox:
        ax.plot(ox, oy, "s--", ms=6, color="tab:orange", label="iter (旧)")
    ax.set_title("policy loss vs train steps")
    ax.set_xlabel("train steps")
    ax.set_ylabel("policy loss (交叉熵)")
    ax.grid(True, alpha=0.3)
    if sx or ox:
        ax.legend(fontsize=8)

    # --- 子图2: value loss vs steps（量级很小，用 log y 轴）---
    ax = axes[0][1]
    sx, sy = _series(records, "stats", "steps", "value_loss", require_pos_y=True)
    ox, oy = _series(records, "old_train", "steps", "value_loss", require_pos_y=True)
    if sx:
        ax.plot(sx, sy, "-o", ms=3, color="tab:green", label="off-policy (新)")
    if ox:
        ax.plot(ox, oy, "s--", ms=6, color="tab:red", label="iter (旧)")
    ax.set_title("value loss vs train steps (log y)")
    ax.set_xlabel("train steps")
    ax.set_ylabel("value loss (MSE)")
    if sy or oy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    if sx or ox:
        ax.legend(fontsize=8)

    # --- 子图3: score mean/max vs games ---
    ax = axes[1][0]
    gx, gmean = _series(records, "stats", "games", "score_mean", require_pos_y=True)
    _, gmax = _series(records, "stats", "games", "score_max", require_pos_y=True)
    if gx:
        ax.plot(gx, gmean, "-o", ms=3, color="tab:purple", label="score mean (新)")
    if gx and len(gmax) == len(gx):
        ax.plot(gx, gmax, "-^", ms=3, color="tab:pink", label="score max (新)")
    # 旧版 selfplay 用 iter 作为点（叠加散点）
    for r in records:
        if r["kind"] == "old_selfplay" and r.get("score_mean") is not None:
            ax.scatter([r.get("games") or 0], [r["score_mean"]], color="tab:gray",
                       marker="x", s=40, zorder=5)
    ax.set_title("score mean / max vs games")
    ax.set_xlabel("self-play games")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.3)
    if gx:
        ax.legend(fontsize=8)

    # --- 子图4: buffer 大小 与 产样速率 vs 时间 ---
    ax = axes[1][1]
    times, bufs, rates = [], [], []
    t0 = None
    for r in records:
        if r["kind"] != "stats" or r.get("ts") is None:
            continue
        if t0 is None:
            t0 = r["ts"]
        mins = (r["ts"] - t0).total_seconds() / 60.0
        if r.get("buffer") is not None:
            times.append(mins)
            bufs.append(r["buffer"])
            rates.append(r.get("mv_per_s"))
    if times:
        ax.plot(times, bufs, "-o", ms=3, color="tab:blue", label="buffer 大小")
        ax.set_ylabel("buffer 样本数", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax.twinx()
        rx = [t for t, v in zip(times, rates) if v is not None]
        ry = [v for v in rates if v is not None]
        if rx:
            ax2.plot(rx, ry, "-s", ms=3, color="tab:orange", label="产样速率 mv/s")
            ax2.set_ylabel("产样速率 (mv/s)", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax.set_title("buffer 与产样速率 vs 时间")
    ax.set_xlabel("训练时长 (分钟, 从首条 STATS 起)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_once(args):
    records = parse_log(args.log)
    write_csv(records, args.csv)
    make_plot(records, args.out)
    n_stats = sum(1 for r in records if r["kind"] == "stats")
    n_old = sum(1 for r in records if r["kind"].startswith("old"))
    stamp = datetime.now().strftime("%H:%M:%S")
    print("[%s] 解析 %d 条记录 (STATS=%d, 旧版=%d) -> %s / %s"
          % (stamp, len(records), n_stats, n_old, args.out, args.csv))
    return records


def main():
    ap = argparse.ArgumentParser(description="AlphaZero 合成大西瓜 loss 曲线可视化")
    ap.add_argument("--log", default=DEFAULT_LOG, help="训练日志路径")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出 PNG 路径")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="输出 metrics.csv 路径")
    ap.add_argument("--watch", type=int, default=0, metavar="N",
                    help="每 N 秒重画一次（默认 0=跑一次就退出）")
    args = ap.parse_args()

    if args.watch and args.watch > 0:
        print("[plot_loss] watch 模式，每 %ds 刷新一次，Ctrl-C 退出" % args.watch)
        try:
            while True:
                run_once(args)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n[plot_loss] 已退出 watch。")
    else:
        run_once(args)


if __name__ == "__main__":
    main()
