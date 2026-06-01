#!/usr/bin/env python3
"""Qwen3.5 coordinate policy for the headless Suika environment.

The agent is intentionally independent from the training pipeline. It reads the
structured ``SuikaEnv`` state, asks Qwen for a single x coordinate, parses the
answer strictly, and returns an absolute pixel x suitable for ``env.step(x)``.
"""
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from common import (
    KILLY,
    PLAY_BOT,
    PLAY_LEFT,
    PLAY_RIGHT,
    PLAY_TOP,
    PLAY_W,
    radius_of,
)


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_DRY_RUN_RESPONSE = '{"x_rel": 0.50}'

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_ACTION_RE = re.compile(r"(?:^|[^A-Za-z0-9_])A\s*=\s*([-+]?\d+(?:\.\d+)?)")
_ACTION_TOKEN_RE = re.compile(r"\bACTION[_-](\d{1,3})\b", re.IGNORECASE)


@dataclass
class CoordinateDecision:
    """Parsed policy decision returned by ``QwenCoordinatePolicyAgent``."""

    x_abs: float
    x_rel: float
    raw_text: str
    prompt: str
    source: str
    parsed: bool
    fallback_used: bool


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _clamp(value: float, lo: float, hi: float) -> float:
    if hi < lo:
        lo, hi = hi, lo
    return max(lo, min(hi, value))


def _round_float(value: Any, digits: int = 2) -> float:
    return round(float(value), digits)


def legal_x_bounds(state: Dict[str, Any]) -> Tuple[float, float]:
    """Return legal absolute x bounds for the current fruit center."""

    play_area = state.get("play_area") or {}
    left = float(play_area.get("left", PLAY_LEFT))
    right = float(play_area.get("right", PLAY_RIGHT))
    current = state.get("current") or {}
    radius = _as_float(current.get("radius"))
    if radius is None:
        radius = radius_of(int(current.get("type", 0)))
    legal_left = left + radius
    legal_right = right - radius
    if legal_right < legal_left:
        center = (left + right) / 2.0
        return center, center
    return float(legal_left), float(legal_right)


def _x_abs_to_rel(x_abs: float, legal_left: float, legal_right: float) -> float:
    span = max(legal_right - legal_left, 1e-9)
    return _clamp((x_abs - legal_left) / span, 0.0, 1.0)


def _x_rel_to_abs(x_rel: float, legal_left: float, legal_right: float) -> float:
    rel = _clamp(x_rel, 0.0, 1.0)
    return legal_left + rel * (legal_right - legal_left)


def _compact_fruits(state: Dict[str, Any], max_fruits: int) -> List[Dict[str, Any]]:
    """Keep the most decision-relevant fruits: dangerous/top first, then large."""

    fruits = list(state.get("fruits") or [])

    def sort_key(fruit: Dict[str, Any]) -> Tuple[float, float, float]:
        y = float(fruit.get("y", PLAY_BOT))
        radius = float(fruit.get("radius", 0.0))
        x = float(fruit.get("x", PLAY_LEFT + PLAY_W / 2.0))
        return (y - radius, -radius, x)

    out = []
    for fruit in sorted(fruits, key=sort_key)[:max(0, int(max_fruits))]:
        out.append({
            "type": int(fruit.get("type", 0)),
            "x": _round_float(fruit.get("x", 0.0)),
            "y": _round_float(fruit.get("y", 0.0)),
            "radius": _round_float(fruit.get("radius", 0.0)),
        })
    return out


def compact_state_prompt(state: Dict[str, Any], max_fruits: int = 16) -> str:
    """Minimal state prompt for compact VLA LoRA adapters."""
    legal_left, legal_right = legal_x_bounds(state)
    cur = state.get("current") or {}
    nxt = state.get("next") or {}
    fruit_bits = []
    for f in _compact_fruits(state, max_fruits=max_fruits):
        fruit_bits.append("%d,%d,%d,%d" % (
            int(f["type"]),
            int(round(float(f["x"]))),
            int(round(float(f["y"]))),
            int(round(float(f["radius"]))),
        ))
    ftxt = ";".join(fruit_bits) if fruit_bits else "-"
    return (
        "Suika. Choose the drop bin. Output only the action.\n"
        "Format: ACTION_N where N is an integer bin in 0-100.\n"
        "C%d N%d S%d X%d,%d D%d F=%s"
        % (
            int(cur.get("type", 0)),
            int(nxt.get("type", 0)),
            int(state.get("score", 0)),
            int(round(legal_left)),
            int(round(legal_right)),
            int(round(float((state.get("play_area") or {}).get("danger_y", KILLY)))),
            ftxt,
        )
    )


def state_to_prompt(state: Dict[str, Any], max_fruits: int = 24, prompt_style: str = "verbose") -> str:
    """Serialize a Suika state into a short, text-only Qwen prompt."""
    if prompt_style == "compact":
        return compact_state_prompt(state, max_fruits=min(max_fruits, 16))

    play_area = state.get("play_area") or {}
    legal_left, legal_right = legal_x_bounds(state)
    current = state.get("current") or {}
    next_fruit = state.get("next") or {}
    payload = {
        "current": {
            "type": int(current.get("type", 0)),
            "name": current.get("name", ""),
            "radius": _round_float(current.get("radius", radius_of(current.get("type", 0)))),
        },
        "next": {
            "type": int(next_fruit.get("type", 0)),
            "name": next_fruit.get("name", ""),
        },
        "score": int(state.get("score", 0)),
        "play_area": {
            "left": _round_float(play_area.get("left", PLAY_LEFT)),
            "right": _round_float(play_area.get("right", PLAY_RIGHT)),
            "top": _round_float(play_area.get("top", PLAY_TOP)),
            "bottom": _round_float(play_area.get("bottom", PLAY_BOT)),
            "danger_y": _round_float(play_area.get("danger_y", KILLY)),
            "legal_left": _round_float(legal_left),
            "legal_right": _round_float(legal_right),
        },
        "fruits_top_first": _compact_fruits(state, max_fruits=max_fruits),
    }
    state_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        "You choose one Suika drop coordinate for the current fruit.\n"
        "Coordinates are pixels; y grows downward. Avoid fruit tops near danger_y.\n"
        "Prefer output exactly one JSON object: {\"x_rel\":0.42}.\n"
        "x_rel must be 0..1, mapped from legal_left to legal_right.\n"
        "You may also output {\"x_bin\":42} where x_bin is an integer 0..100.\n"
        "If needed, you may output {\"x_abs\":312.5} in pixels.\n"
        "No explanation, no markdown, no extra keys.\n"
        f"STATE={state_json}"
    )


def _iter_json_objects(text: str) -> Iterable[Dict[str, Any]]:
    for match in _JSON_RE.finditer(text):
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def parse_coordinate_text(
    text: str,
    state: Dict[str, Any],
    strict_action_token: bool = False,
) -> Tuple[Optional[float], Optional[float], str]:
    """Parse Qwen text into ``(x_abs, x_rel, source)`` or ``(None, None, reason)``.

    The parser first accepts strict JSON with relative or absolute coordinate
    keys, then falls back to the first finite number in the response.
    In compact ACTION-token mode, only explicit ``ACTION_N`` text is accepted;
    otherwise copied state fields like ``C4 N3 S0`` can be mistaken for actions.
    """

    legal_left, legal_right = legal_x_bounds(state)

    token_match = _ACTION_TOKEN_RE.search(text or "")
    if token_match:
        value = _as_float(token_match.group(1))
        if value is not None:
            x_abs = _x_rel_to_abs(round(value) / 100.0, legal_left, legal_right)
            return x_abs, _x_abs_to_rel(x_abs, legal_left, legal_right), "text:ACTION_token"

    if strict_action_token:
        return None, None, "unparsed:missing_ACTION_token"

    action_match = _ACTION_RE.search(text or "")
    if action_match:
        value = _as_float(action_match.group(1))
        if value is not None:
            x_abs = _x_rel_to_abs(round(value) / 100.0, legal_left, legal_right)
            return x_abs, _x_abs_to_rel(x_abs, legal_left, legal_right), "text:A_bin"

    for obj in _iter_json_objects(text or ""):
        for key in ("x_bin", "bin", "action_bin"):
            value = _as_float(obj.get(key))
            if value is not None:
                x_abs = _x_rel_to_abs(round(value) / 100.0, legal_left, legal_right)
                return x_abs, _x_abs_to_rel(x_abs, legal_left, legal_right), "json:x_bin"
        for key in ("x_rel", "rel_x", "relative_x"):
            value = _as_float(obj.get(key))
            if value is not None:
                x_abs = _x_rel_to_abs(value, legal_left, legal_right)
                return x_abs, _x_abs_to_rel(x_abs, legal_left, legal_right), "json:x_rel"
        for key in ("x_abs", "abs_x", "absolute_x"):
            value = _as_float(obj.get(key))
            if value is not None:
                x_abs = _clamp(value, legal_left, legal_right)
                return x_abs, _x_abs_to_rel(x_abs, legal_left, legal_right), "json:x_abs"
        value = _as_float(obj.get("x"))
        if value is not None:
            if 0.0 <= value <= 1.0:
                x_abs = _x_rel_to_abs(value, legal_left, legal_right)
                return x_abs, _x_abs_to_rel(x_abs, legal_left, legal_right), "json:x_as_rel"
            x_abs = _clamp(value, legal_left, legal_right)
            return x_abs, _x_abs_to_rel(x_abs, legal_left, legal_right), "json:x_as_abs"

    match = _FLOAT_RE.search(text or "")
    if match:
        value = _as_float(match.group(0))
        if value is not None:
            if 0.0 <= value <= 1.0:
                x_abs = _x_rel_to_abs(value, legal_left, legal_right)
                return x_abs, _x_abs_to_rel(x_abs, legal_left, legal_right), "number:x_rel"
            x_abs = _clamp(value, legal_left, legal_right)
            return x_abs, _x_abs_to_rel(x_abs, legal_left, legal_right), "number:x_abs"

    return None, None, "unparsed"


class QwenCoordinatePolicyAgent:
    """Prompt Qwen3.5-0.8B for a Suika x coordinate."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        adapter_dir: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 24,
        device: str = "auto",
        dtype: str = "auto",
        max_prompt_fruits: int = 24,
        fallback: str = "heuristic",
        prompt_style: str = "verbose",
        dry_run: bool = False,
        dry_run_response: str = DEFAULT_DRY_RUN_RESPONSE,
        trust_remote_code: bool = False,
    ):
        self.model_id = model_id
        self.adapter_dir = adapter_dir
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.device_name = device
        self.dtype_name = dtype
        self.max_prompt_fruits = int(max_prompt_fruits)
        self.fallback = fallback
        self.prompt_style = prompt_style
        self.dry_run = bool(dry_run)
        self.dry_run_response = dry_run_response
        self.trust_remote_code = bool(trust_remote_code)
        self.processor = None
        self.tokenizer = None
        self.model = None
        self._fallback_agent = None

        if not self.dry_run:
            self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "QwenCoordinatePolicyAgent requires a recent transformers build "
                "with Qwen3.5 support and PyTorch installed. Install them in the "
                "Suika environment before running without --dry-run; install "
                "accelerate too if your Transformers setup requires it. This "
                "script does not modify the environment automatically."
            ) from exc

        device = self._resolve_device(torch, self.device_name)
        torch_dtype = self._resolve_dtype(torch, self.dtype_name)
        processor_kwargs: Dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        model_kwargs: Dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        self.processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **model_kwargs)
        if self.adapter_dir:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "Loading --adapter-dir requires peft. Install it in the "
                    "Suika environment before running adapter inference."
                ) from exc
            self.model = PeftModel.from_pretrained(self.model, self.adapter_dir)
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _resolve_device(torch: Any, requested: str) -> str:
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_dtype(torch: Any, requested: str) -> Any:
        if requested in ("auto", "", None):
            return None
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if requested not in mapping:
            raise ValueError("Unsupported dtype %r; use auto,float32,float16,bfloat16" % requested)
        return mapping[requested]

    def _messages(self, prompt: str) -> List[Dict[str, Any]]:
        # Match the SFT records exactly; Qwen chat templates are sensitive to
        # whether a system turn is present before the user prompt.
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You output only the Suika drop action token requested by the user."}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

    def _generate_text(self, prompt: str) -> str:
        if self.dry_run:
            return self.dry_run_response
        if self.processor is None or self.model is None:
            raise RuntimeError("Model is not loaded; use dry_run=True for prompt/parser tests.")

        messages = self._messages(prompt)
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        model_device = next(self.model.parameters()).device
        if hasattr(inputs, "to"):
            inputs = inputs.to(model_device)
        else:
            inputs = {
                key: value.to(model_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
        }
        tokenizer = self.tokenizer or getattr(self.processor, "tokenizer", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
        pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        if eos_token_id is not None:
            gen_kwargs["eos_token_id"] = eos_token_id
        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id
        elif eos_token_id is not None:
            gen_kwargs["pad_token_id"] = eos_token_id
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature

        outputs = self.model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[-1]
        generated = outputs[0][input_len:]
        if hasattr(self.processor, "decode"):
            return self.processor.decode(generated, skip_special_tokens=True).strip()
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _fallback_x(self, state: Dict[str, Any]) -> Tuple[float, str]:
        legal_left, legal_right = legal_x_bounds(state)
        center = (legal_left + legal_right) / 2.0
        if self.fallback == "center":
            return center, "fallback:center"
        if self.fallback != "heuristic":
            return center, "fallback:center"
        try:
            if self._fallback_agent is None:
                from ai_agent import HeuristicLookaheadAgent

                self._fallback_agent = HeuristicLookaheadAgent()
            value = self._fallback_agent.decide(state)
            if isinstance(value, tuple):
                value = value[0]
            x_abs = _as_float(value)
            if x_abs is None:
                return center, "fallback:center"
            return _clamp(x_abs, legal_left, legal_right), "fallback:heuristic"
        except Exception:
            return center, "fallback:center"

    def decide_with_debug(self, state: Dict[str, Any]) -> CoordinateDecision:
        prompt = state_to_prompt(
            state,
            max_fruits=self.max_prompt_fruits,
            prompt_style=self.prompt_style,
        )
        raw_text = self._generate_text(prompt)
        x_abs, x_rel, source = parse_coordinate_text(
            raw_text,
            state,
            strict_action_token=self.prompt_style == "compact",
        )
        if x_abs is None or x_rel is None:
            x_abs, source = self._fallback_x(state)
            legal_left, legal_right = legal_x_bounds(state)
            x_rel = _x_abs_to_rel(x_abs, legal_left, legal_right)
            return CoordinateDecision(
                x_abs=float(x_abs),
                x_rel=float(x_rel),
                raw_text=raw_text,
                prompt=prompt,
                source=source,
                parsed=False,
                fallback_used=True,
            )
        return CoordinateDecision(
            x_abs=float(x_abs),
            x_rel=float(x_rel),
            raw_text=raw_text,
            prompt=prompt,
            source=source,
            parsed=True,
            fallback_used=False,
        )

    def decide(self, state: Dict[str, Any]) -> float:
        """Return absolute x for ``SuikaEnv.step(x)``."""

        return self.decide_with_debug(state).x_abs
