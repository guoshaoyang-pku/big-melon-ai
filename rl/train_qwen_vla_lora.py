#!/usr/bin/env python3
"""LoRA SFT for Qwen/Qwen3.5-0.8B Suika VLA policy records.

Dry-run mode intentionally does not import or load Transformers models.  It
only validates JSONL parsing and the assistant-only target mask construction.
The implemented training path is text+state SFT; image_path is preserved in the
dataset schema for a later true multimodal processor path.
"""
import argparse
import itertools
import json
import math
import os
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple


IGNORE_INDEX = -100


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") in ("image", "image_url"):
                    parts.append("[IMAGE]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def _answer_to_text(answer: Any) -> str:
    if isinstance(answer, str):
        return answer.strip()
    if isinstance(answer, dict):
        return json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
    raise ValueError("answer must be a JSON string or object")


def _messages_from_record(record: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        normalized: List[Dict[str, str]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user"))
            normalized.append({"role": role, "content": _content_to_text(msg.get("content", ""))})
        if normalized and normalized[-1]["role"] == "assistant":
            assistant_text = _answer_to_text(normalized[-1]["content"])
            return normalized[:-1], assistant_text

    prompt = record.get("prompt")
    answer = record.get("answer")
    if prompt is None or answer is None:
        raise ValueError("record must contain messages with assistant or prompt+answer")
    return [{"role": "user", "content": str(prompt)}], _answer_to_text(answer)


def _render_chat(messages: Sequence[Dict[str, str]], add_generation_prompt: bool) -> str:
    chunks: List[str] = []
    for msg in messages:
        chunks.append("<|%s|>\n%s\n" % (msg["role"], msg["content"]))
    if add_generation_prompt:
        chunks.append("<|assistant|>\n")
    return "".join(chunks)


class DryRunTokenizer:
    """Tiny deterministic tokenizer used only for no-model dry-run validation."""

    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "</s>"

    def apply_chat_template(
        self,
        messages: Sequence[Dict[str, str]],
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        **_: Any,
    ) -> Any:
        text = _render_chat(messages, add_generation_prompt=add_generation_prompt)
        if not tokenize:
            return text
        return self.encode(text)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        del add_special_tokens
        return [(ord(ch) % 251) + 3 for ch in text]


def _as_multimodal_text_messages(messages: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    return [{
        "role": msg["role"],
        "content": [{"type": "text", "text": msg["content"]}],
    } for msg in messages]


def _apply_chat_template_raw(
    template_owner: Any,
    messages: Sequence[Dict[str, Any]],
    add_generation_prompt: bool,
    tokenize: bool,
    **kwargs: Any,
) -> Any:
    call_kwargs = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": tokenize,
        **kwargs,
    }
    try:
        return template_owner.apply_chat_template(list(messages), enable_thinking=False, **call_kwargs)
    except TypeError:
        return template_owner.apply_chat_template(list(messages), **call_kwargs)


class ProcessorTokenizerBridge:
    """Use AutoProcessor chat templates while keeping tokenizer encode/pad APIs."""

    def __init__(self, processor: Any, tokenizer: Any):
        self.processor = processor
        self.tokenizer = tokenizer
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def apply_chat_template(
        self,
        messages: Sequence[Dict[str, str]],
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        **kwargs: Any,
    ) -> Any:
        mm_messages = _as_multimodal_text_messages(messages)
        if hasattr(self.processor, "apply_chat_template"):
            try:
                return _apply_chat_template_raw(
                    self.processor,
                    mm_messages,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=tokenize,
                    **kwargs,
                )
            except Exception:
                pass
        if hasattr(self.tokenizer, "apply_chat_template"):
            return _apply_chat_template_raw(
                self.tokenizer,
                mm_messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                **kwargs,
            )
        text = _render_chat(messages, add_generation_prompt=add_generation_prompt)
        return self.encode(text, add_special_tokens=False) if tokenize else text

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer(*args, **kwargs)


def _apply_chat_template(tokenizer: Any, messages: Sequence[Dict[str, str]], add_generation_prompt: bool) -> List[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        ids = _apply_chat_template_raw(
            tokenizer,
            list(messages),
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return list(ids)
    text = _render_chat(messages, add_generation_prompt=add_generation_prompt)
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def _apply_chat_template_text(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = _apply_chat_template_raw(
            tokenizer,
            list(messages),
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        if isinstance(rendered, str):
            return rendered
    return _render_chat(messages, add_generation_prompt=add_generation_prompt)


def _encode_text(tokenizer: Any, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def _tokenize_prompt_answer(
    tokenizer: Any,
    prompt_text: str,
    assistant_text: str,
) -> Tuple[List[int], List[int]]:
    full_text = prompt_text + assistant_text
    prompt_len = len(prompt_text)

    if callable(tokenizer):
        try:
            encoded = tokenizer(
                full_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            full_ids = list(encoded["input_ids"])
            offsets = encoded.get("offset_mapping")
            if offsets is not None and len(offsets) == len(full_ids):
                labels = []
                for token_id, offset in zip(full_ids, offsets):
                    end = int(offset[1])
                    labels.append(int(token_id) if end > prompt_len else IGNORE_INDEX)
                return [int(x) for x in full_ids], labels
        except Exception:
            pass

    prompt_ids = _encode_text(tokenizer, prompt_text)
    full_ids = _encode_text(tokenizer, full_text)
    labels = [IGNORE_INDEX] * len(full_ids)
    for i in range(len(prompt_ids), len(full_ids)):
        labels[i] = int(full_ids[i])
    return [int(x) for x in full_ids], labels


def build_tokenized_sample(
    tokenizer: Any,
    record: Dict[str, Any],
    max_length: int,
) -> Dict[str, List[int]]:
    prompt_messages, assistant_text = _messages_from_record(record)

    # Train on the exact inference prefix.  Qwen3.5's generation prompt includes
    # an empty thinking scaffold, so rendering a full assistant turn is not a
    # byte/token prefix of the eval prompt.
    prompt_text = _apply_chat_template_text(tokenizer, prompt_messages, add_generation_prompt=True)
    full_ids, labels = _tokenize_prompt_answer(tokenizer, prompt_text, assistant_text)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        full_ids.append(int(eos_token_id))
        labels.append(int(eos_token_id))

    if max_length and len(full_ids) > max_length:
        start = len(full_ids) - int(max_length)
        full_ids = full_ids[start:]
        labels = labels[start:]
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("assistant target was truncated away; increase --max-length")
    return {
        "input_ids": [int(x) for x in full_ids],
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d invalid JSONL: %s" % (path, line_no, exc)) from exc
    if not rows:
        raise ValueError("dataset is empty: %s" % path)
    return rows


def dry_run(args: argparse.Namespace) -> int:
    records = read_jsonl(args.train_jsonl)
    tokenizer = DryRunTokenizer()
    summaries = []
    for idx, rec in enumerate(records[:min(len(records), 8)]):
        sample = build_tokenized_sample(tokenizer, rec, max_length=args.max_length)
        target_tokens = sum(1 for label in sample["labels"] if label != IGNORE_INDEX)
        summaries.append({
            "index": idx,
            "tokens": len(sample["input_ids"]),
            "target_tokens": target_tokens,
            "assistant_text": _messages_from_record(rec)[1],
        })
    print(json.dumps({
        "dry_run": True,
        "records": len(records),
        "checked": len(summaries),
        "max_length": int(args.max_length),
        "samples": summaries,
    }, ensure_ascii=False, indent=2))
    return 0


class JsonlSftDataset:
    def __init__(self, records: Sequence[Dict[str, Any]], tokenizer: Any, max_length: int):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return build_tokenized_sample(self.tokenizer, self.records[index], self.max_length)


def _collate(features: Sequence[Dict[str, List[int]]], pad_token_id: int) -> Dict[str, Any]:
    import torch

    max_len = max(len(f["input_ids"]) for f in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for feature in features:
        pad = max_len - len(feature["input_ids"])
        batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * pad)
        batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
        batch["labels"].append(feature["labels"] + [IGNORE_INDEX] * pad)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def _resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(torch: Any, requested: str) -> Any:
    mapping = {
        "auto": None,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if requested not in mapping:
        raise ValueError("Unsupported --dtype %r" % requested)
    return mapping[requested]


def _mps_memory_stats(torch: Any, device: str) -> Dict[str, float]:
    if device != "mps":
        return {}
    mps = getattr(torch, "mps", None)
    if mps is None:
        return {}
    stats: Dict[str, float] = {}
    for key, attr in (
        ("mps_current_mb", "current_allocated_memory"),
        ("mps_driver_mb", "driver_allocated_memory"),
    ):
        fn = getattr(mps, attr, None)
        if not callable(fn):
            continue
        try:
            stats[key] = float(fn()) / (1024.0 * 1024.0)
        except Exception:
            continue
    return stats


def train(args: argparse.Namespace) -> int:
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "Training requires torch, transformers, peft, and accelerate. "
            "Install in the Suika env, for example: "
            "pip install torch transformers peft accelerate pillow"
        ) from exc

    records = read_jsonl(args.train_jsonl)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    template_tokenizer = ProcessorTokenizerBridge(processor, tokenizer)

    dataset = JsonlSftDataset(records, template_tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=lambda features: _collate(features, pad_token_id=pad_token_id),
    )

    dtype = _resolve_dtype(torch, args.dtype)
    model_kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    if args.gradient_checkpointing:
        if hasattr(model, "config"):
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    device = _resolve_device(torch, args.device)
    model.to(device)

    target_modules: Any = args.target_modules
    if target_modules != "all-linear":
        target_modules = [x.strip() for x in target_modules.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(args.lr),
    )

    grad_accum_steps = int(args.grad_accum_steps)
    if grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    max_optimizer_steps = int(args.max_steps)
    if max_optimizer_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    total_micro_steps = max_optimizer_steps * grad_accum_steps
    optimizer.zero_grad(set_to_none=True)

    iterator = itertools.cycle(loader)
    optimizer_step = 0
    total_micro_samples = 0
    total_tokens = 0
    skipped_nonfinite = 0
    accum_loss = 0.0
    accum_micro_steps = 0
    start_time = time.perf_counter()
    for micro_step in range(1, total_micro_steps + 1):
        batch = next(iterator)
        micro_samples = int(batch["input_ids"].shape[0])
        micro_tokens = int(batch["attention_mask"].sum().item())
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        raw_loss = float(loss.detach().cpu())
        if not math.isfinite(raw_loss):
            print("[qwen-vla-lora] non_finite_loss micro_step=%d/%d optimizer_step=%d/%d raw_loss=%s" % (
                micro_step, total_micro_steps, optimizer_step, max_optimizer_steps, raw_loss
            ), flush=True)
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            accum_micro_steps = 0
            if not args.skip_nonfinite_loss:
                return 2
            if skipped_nonfinite > int(args.max_nonfinite_skips):
                print("[qwen-vla-lora] too_many_nonfinite_skips=%d" % skipped_nonfinite, flush=True)
                return 2
            continue
        loss_scaled = loss / grad_accum_steps
        loss_scaled.backward()

        total_micro_samples += micro_samples
        total_tokens += micro_tokens
        accum_loss += raw_loss
        accum_micro_steps += 1

        should_step = (micro_step % grad_accum_steps == 0) or (micro_step == total_micro_steps)
        if should_step and accum_micro_steps > 0:
            if float(args.max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=float(args.max_grad_norm),
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            now = time.perf_counter()
            elapsed = max(now - start_time, 1e-9)
            avg_loss = accum_loss / max(1, accum_micro_steps)
            if (
                optimizer_step == 1
                or optimizer_step == max_optimizer_steps
                or optimizer_step % max(1, int(args.log_every)) == 0
            ):
                log_record = {
                    "optimizer_step": optimizer_step,
                    "max_optimizer_steps": max_optimizer_steps,
                    "micro_step": micro_step,
                    "total_micro_steps": total_micro_steps,
                    "grad_accum_steps": grad_accum_steps,
                    "raw_loss": raw_loss,
                    "avg_loss": avg_loss,
                    "micro_samples_per_s": total_micro_samples / elapsed,
                    "optimizer_steps_per_s": optimizer_step / elapsed,
                    "tokens_per_s": total_tokens / elapsed,
                    "skipped_nonfinite": skipped_nonfinite,
                }
                log_record.update(_mps_memory_stats(torch, device))
                print("[qwen-vla-lora] " + json.dumps(log_record, ensure_ascii=False), flush=True)
            accum_loss = 0.0
            accum_micro_steps = 0

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "out_dir": args.out_dir,
        "records": len(records),
        "max_steps": max_optimizer_steps,
        "grad_accum_steps": grad_accum_steps,
        "total_micro_steps": total_micro_steps,
        "skipped_nonfinite": skipped_nonfinite,
        "text_state_only": True,
        "image_path_preserved_but_not_trained": True,
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="number of micro-batches to accumulate per optimizer step")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--target-modules", default="all-linear")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="trade compute for lower activation memory during LoRA training")
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="clip trainable LoRA gradient norm; <=0 disables")
    ap.add_argument("--skip-nonfinite-loss", action="store_true",
                    help="skip isolated NaN/Inf micro-batches instead of aborting")
    ap.add_argument("--max-nonfinite-skips", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        return dry_run(args)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
