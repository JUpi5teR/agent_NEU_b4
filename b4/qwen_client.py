from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from common.path_utils import resolve_from_file

_MODEL_CACHE: dict[tuple[str, ...], tuple[Any, Any]] = {}


class QwenJsonError(ValueError):
    """Raised when Qwen does not return a parseable JSON object."""


def qwen_generate_json(
    config_path: Path,
    config: dict[str, Any],
    task: str,
    payload: dict[str, Any],
    schema_hint: dict[str, Any],
    *,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are the B4 structured decision engine. "
                "Return exactly one valid JSON object. "
                "Do not output markdown, code fences, comments, or text outside JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": task,
                    "payload": payload,
                    "required_json_schema": schema_hint,
                    "output_rules": [
                        "Return only one JSON object.",
                        "Use only the fields listed in required_json_schema.",
                        "Use concise strings.",
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]
    completion = qwen_generate_completion(config_path, config, prompt_messages, max_new_tokens=max_new_tokens)
    usage_records = config.setdefault("_b4_usage_records", [])
    if isinstance(usage_records, list):
        model_config = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
        tool_calling = config.get("tool_calling", {}) if isinstance(config.get("tool_calling"), dict) else {}
        usage_records.append(
            {
                "task": task,
                "usage": completion.get("usage"),
                "model": model_config.get("model_name_or_path"),
                "model_profile": config.get("selected_model_profile"),
                "tool_calling_mode": tool_calling.get("mode"),
            }
        )
    return extract_json_object(completion["text"])


def qwen_generate_completion(
    config_path: Path,
    config: dict[str, Any],
    prompt_messages: list[dict[str, Any]],
    *,
    tools_schema: list[dict[str, Any]] | None = None,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    raw_text, usage = _qwen_generate(config_path, config, prompt_messages, tools_schema, max_new_tokens)
    return {"text": raw_text, "usage": usage}


def qwen_generate_text(
    config_path: Path,
    config: dict[str, Any],
    prompt_messages: list[dict[str, Any]],
    *,
    tools_schema: list[dict[str, Any]] | None = None,
    max_new_tokens: int | None = None,
) -> str:
    raw_text, _usage = _qwen_generate(config_path, config, prompt_messages, tools_schema, max_new_tokens)
    return raw_text


def _qwen_generate(
    config_path: Path,
    config: dict[str, Any],
    prompt_messages: list[dict[str, Any]],
    tools_schema: list[dict[str, Any]] | None,
    max_new_tokens: int | None,
) -> tuple[str, dict[str, int]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("prompt_json mode requires torch and transformers") from exc

    model_config = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    generation_config = config.get("generation", {}) if isinstance(config.get("generation"), dict) else {}
    tool_calling = config.get("tool_calling", {}) if isinstance(config.get("tool_calling"), dict) else {}
    model_setting = model_config.get("model_name_or_path")
    tokenizer_setting = model_config.get("tokenizer_name_or_path", model_setting)
    if not isinstance(model_setting, str) or not isinstance(tokenizer_setting, str):
        raise ValueError("model_name_or_path and tokenizer_name_or_path are required")

    model_path = resolve_from_file(model_setting, config_path)
    tokenizer_path = resolve_from_file(tokenizer_setting, config_path)
    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(f"local model path does not exist: {model_path}")

    dtype = _dtype_value(torch, str(model_config.get("torch_dtype", "auto")))
    tokenizer, model = _load_model_bundle(
        AutoModelForCausalLM,
        AutoTokenizer,
        model_path,
        tokenizer_path,
        bool(model_config.get("local_files_only", True)),
        bool(model_config.get("trust_remote_code", False)),
        dtype,
        model_config.get("device_map", "auto"),
        model_config.get("max_memory"),
    )

    template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
        "enable_thinking": False,
    }
    if tool_calling.get("mode") == "builtin_tools" and tools_schema:
        template_kwargs["tools"] = tools_schema
    inputs = tokenizer.apply_chat_template(prompt_messages, **template_kwargs)
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    input_length = inputs["input_ids"].shape[-1]
    options = {
        "max_new_tokens": int(max_new_tokens or generation_config.get("max_new_tokens", 1024)),
        "do_sample": bool(generation_config.get("do_sample", False)),
    }
    with torch.no_grad():
        generated = model.generate(**inputs, **options)
    new_tokens = generated[0][input_length:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    usage = {
        "input_tokens": int(input_length),
        "output_tokens": int(new_tokens.shape[-1]),
        "total_tokens": int(input_length + new_tokens.shape[-1]),
    }
    return raw_text, usage


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise QwenJsonError(f"Qwen output is not a JSON object: {raw_text[:200]}")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise QwenJsonError("Qwen JSON output must be an object")
    return value


def _dtype_value(torch_module: Any, configured: str) -> Any:
    if configured == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    if configured not in mapping:
        raise ValueError(f"unsupported torch_dtype: {configured}")
    return mapping[configured]


def _model_cache_key(
    model_path: Path,
    tokenizer_path: Path,
    local_only: bool,
    trust_remote_code: bool,
    dtype: Any,
    device_map: Any,
    max_memory: Any,
) -> tuple[str, ...]:
    try:
        device_map_key = json.dumps(device_map, sort_keys=True, separators=(",", ":"))
    except TypeError:
        device_map_key = repr(device_map)
    try:
        max_memory_key = json.dumps(max_memory, sort_keys=True, separators=(",", ":"))
    except TypeError:
        max_memory_key = repr(max_memory)
    return (
        str(model_path),
        str(tokenizer_path),
        str(local_only),
        str(trust_remote_code),
        str(dtype),
        device_map_key,
        max_memory_key,
    )


def _load_model_bundle(
    auto_model: Any,
    auto_tokenizer: Any,
    model_path: Path,
    tokenizer_path: Path,
    local_only: bool,
    trust_remote_code: bool,
    dtype: Any,
    device_map: Any,
    max_memory: Any,
) -> tuple[Any, Any]:
    cache_key = _model_cache_key(
        model_path,
        tokenizer_path,
        local_only,
        trust_remote_code,
        dtype,
        device_map,
        max_memory,
    )
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        print("qwen_model_cache=hit", file=sys.stderr, flush=True)
        return cached

    print("qwen_model_cache=miss", file=sys.stderr, flush=True)
    tokenizer = auto_tokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=local_only,
        trust_remote_code=trust_remote_code,
    )
    model = auto_model.from_pretrained(
        str(model_path),
        local_files_only=local_only,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
    )
    _MODEL_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model
