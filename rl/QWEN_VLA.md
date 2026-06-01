# Qwen VLA Policy

This directory contains a lightweight pipeline for fine-tuning
`Qwen/Qwen3.5-0.8B` as a Suika visual-language-action policy.

The current implementation trains text+state SFT and preserves `image_path` /
image-placeholder fields for later true multimodal training.  The model card
uses `AutoProcessor` + `AutoModelForImageTextToText`, so real image training
should be added carefully after validating processor-specific image batching.

## Dependencies

Use the project Python environment:

```bash
/Users/guoshaoyang/miniconda3/envs/suika/bin/python -m pip install torch transformers peft accelerate pillow
```

Qwen3.5 may require a recent `transformers` build.  Do not run training until
the model and dependency versions are pinned for the target machine.

## 1. Build SFT JSONL

From an offline teacher replay:

```bash
cd /Users/guoshaoyang/Desktop/workdir/兴趣项目/合成大西瓜/suika
/Users/guoshaoyang/miniconda3/envs/suika/bin/python rl/qwen_vla_dataset.py \
  --input rl/data/teacher_replay.npz \
  --out rl/data/qwen_vla_teacher.jsonl \
  --max-samples 20000 \
  --action-format x_bin \
  --bins 101 \
  --include-image-placeholder
```

If the replay has explicit `x_rel`, `x_abs`, `xs`, or action-bin metadata, that
label is preferred.  Otherwise `pis` is treated as a K-bin teacher distribution
and converted to `x_rel` by expectation over bin centers.  Replay-vector states
are approximate because velocities, collision flags, and RNG state are not
stored in the flat replay.

Tiny rollout/dry-run sample:

```bash
/Users/guoshaoyang/miniconda3/envs/suika/bin/python rl/qwen_vla_dataset.py \
  --out rl/_smoke/qwen_vla_samples.jsonl \
  --max-samples 3 \
  --action-format x_bin \
  --bins 101 \
  --include-image-placeholder \
  --dry-run
```

Each JSONL row has `messages` in chat format.  The assistant answer is strict
JSON, for example `{"x_bin":50}` or `{"x_rel":0.5}`.

## 2. Render Board Images

Render a dry-run board:

```bash
/Users/guoshaoyang/miniconda3/envs/suika/bin/python rl/render_state_image.py \
  --out rl/_smoke/board.png \
  --dry-run
```

Render from replay vector:

```bash
/Users/guoshaoyang/miniconda3/envs/suika/bin/python rl/render_state_image.py \
  --input rl/data/teacher_replay.npz \
  --index 0 \
  --out rl/_smoke/replay_board.png
```

These PNGs are approximate sketches for VLA data inspection, not exact physics
snapshots.

## 3. LoRA SFT

Dry-run validates JSONL parsing and target masks without loading the model:

```bash
/Users/guoshaoyang/miniconda3/envs/suika/bin/python rl/train_qwen_vla_lora.py \
  --train-jsonl rl/_smoke/qwen_vla_samples.jsonl \
  --out-dir rl/_smoke/qwen_vla_lora \
  --dry-run
```

A real short training command:

```bash
/Users/guoshaoyang/miniconda3/envs/suika/bin/python rl/train_qwen_vla_lora.py \
  --model-id Qwen/Qwen3.5-0.8B \
  --train-jsonl rl/data/qwen_vla_teacher.jsonl \
  --out-dir rl/checkpoints/qwen_vla_lora_sft \
  --max-steps 1000 \
  --batch-size 1 \
  --lr 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --device auto \
  --dtype bfloat16
```

Only assistant/action JSON tokens contribute to loss; prompt tokens use
`label=-100`.

## 4. Adapter Evaluation

Dry-run:

```bash
/Users/guoshaoyang/miniconda3/envs/suika/bin/python rl/eval_qwen_vla_adapter.py \
  --adapter-dir rl/checkpoints/qwen_vla_lora_sft \
  --seeds 0:1 \
  --max-moves 3 \
  --dry-run
```

Real adapter eval:

```bash
/Users/guoshaoyang/miniconda3/envs/suika/bin/python rl/eval_qwen_vla_adapter.py \
  --model-id Qwen/Qwen3.5-0.8B \
  --adapter-dir rl/checkpoints/qwen_vla_lora_sft \
  --seeds 0:20 \
  --max-moves 0 \
  --device auto \
  --dtype bfloat16
```

## 5. CEM-Style Improvement Skeleton

This does not train.  It rolls out the current Qwen policy, keeps elite
trajectories by score, and exports their actions as a new SFT JSONL.

```bash
/Users/guoshaoyang/miniconda3/envs/suika/bin/python rl/qwen_vla_cem_loop.py \
  --adapter-dir rl/checkpoints/qwen_vla_lora_sft \
  --out rl/data/qwen_vla_cem_round1.jsonl \
  --summary-out rl/data/qwen_vla_cem_round1_summary.json \
  --games 20 \
  --max-moves 80 \
  --elite-frac 0.25 \
  --action-format x_bin \
  --bins 101
```

Start with very small `--games` and inspect samples before using the exported
data for another LoRA round.
