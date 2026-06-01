# Suika AlphaZero 自博弈训练管线

AlphaZero 式 MCTS + policy/value 网络自博弈，针对合成大西瓜（Suika）。

## 文件
- `common.py`  路径/常量 + 对象中心状态编码（9 全局 + max_fruits*4）。
- `model.py`   环境模型：从 state 快照重建独立 pymunk 空间 → 投放离散列 → 物理稳定。fast/precise 两档保真度。
- `net.py`     policy/value MLP / Object Transformer（PyTorch，MPS）。`infer_batch` 批量推理。
- `mcts.py`    PUCT + 网络先验 + value 引导 + 环境模型展开；**批量叶子并行 + virtual loss**（`eval_batch`）。
- `selfplay.py` 多进程并行自博弈（spawn，CPU 推理），记录 (state, π, z)。
- `train.py`   replay buffer + 训练步（policy CE + value MSE + L2）+ checkpoint。
- `eval.py`    当前网络 vs 启发式 vs 随机基线的基础函数。
- `eval_suite.py` 固定 seed 统一评测表：random / heuristic / net MCTS / uniform-prior MCTS / robust beam。
- `gen_teacher_replay.py` 用 robust-search teacher 生成独立 replay。
- `merge_replay.py` 安全合并 replay（自动备份，保留样本权重）。
- `diagnose_policy.py` 诊断 policy entropy / center mass / KL / value calibration。
- `action_candidates.py` coarse bin + continuous refinement 候选生成原型。
- `export_policy.py` 导出 TorchScript policy/value 网络用于部署或快速评测。
- `run_pipeline.py` AlphaZero 主循环：自博弈→训练→更新→评估，可 resume。
- `config.yaml` 超参。`config.effective.yaml` 为本次实际生效配置。

## 设备分工（实测后的最优）
- **训练：MPS**（batch 256，比 CPU 快 ~3.6×）。
- **自博弈/MCTS：CPU**（多进程；物理占 ~90% 是真瓶颈，小批量下 CPU 推理比 MPS 快；多进程+MPS 不稳）。
- 批量叶子评估把 MCTS 网络开销从 ~5% 降到 ~1%；MPS 在 eval_batch≥64 时才追平 CPU。

## 运行
```bash
PY=/Users/guoshaoyang/miniconda3/envs/suika/bin/python
# 全新启动
nohup $PY run_pipeline.py > logs/train.log 2>&1 & echo $! > logs/train.pid
# 续跑（从最近 checkpoint + buffer）
nohup $PY run_pipeline.py --resume >> logs/train.log 2>&1 & echo $! > logs/train.pid
# 冒烟
$PY run_pipeline.py --smoke
```

## 路线图执行入口
```bash
PY=/Users/guoshaoyang/miniconda3/envs/suika/bin/python

# 阶段 0：固定 seed 评测表（不修改训练产物）
$PY eval_suite.py --seeds 0:50 --agents random,heuristic,uniform,net \
  --sims 128 --device cpu \
  --out-json logs/fixed_seed_eval.json --out-csv logs/fixed_seed_eval.csv

# 阶段 1：强 teacher 评测探针
$PY eval_suite.py --seeds 0:20 --agents robust \
  --robust-depth 4 --robust-beam 8 --robust-samples 2 --robust-branch 4

# 阶段 2：生成 teacher replay（默认写独立文件，不碰 data/replay.npz）
$PY gen_teacher_replay.py --games 100 --depth 2 --beam-size 8 \
  --samples-per-action 2 --out data/teacher_replay.npz

# teacher replay 预训练 Transformer
$PY pretrain_from_replay.py --replay data/teacher_replay.npz \
  --out checkpoints/pretrain_teacher_transformer.pt --steps 5000 --batch-size 512

# 可选：显式合并 teacher replay 到主 replay（会先备份 target）
$PY merge_replay.py --target data/replay.npz --sources data/teacher_replay.npz \
  --capacity 500000

# 阶段 3：MCTS prior / root exploration 对照
$PY eval_suite.py --seeds 0:50 --agents uniform,net --sims 128
$PY eval_suite.py --seeds 0:50 --agents net --sims 128
# root forced visits 通过 config.yaml 或 --set root_forced_visits=N 在训练入口开启。

# 阶段 7：诊断 dashboard 的数据源
$PY diagnose_policy.py --ckpt checkpoints/latest.pt --replay data/replay.npz \
  --out logs/policy_diagnostics.json

# 阶段 8：蒸馏后的模型导出
$PY export_policy.py --ckpt checkpoints/pretrain_teacher_transformer.pt \
  --out checkpoints/policy_value_scripted.pt
```

## 监控 / 控制
```bash
./monitor.sh status   # PID / 最近迭代
./monitor.sh log      # tail 日志
./monitor.sh ckpt     # 看 checkpoint
./monitor.sh stop     # 停止（kill PID）
```
