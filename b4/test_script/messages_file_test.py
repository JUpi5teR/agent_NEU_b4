#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))

from b4.service import generate_ai_message
from common.io_utils import read_json, write_json


DEFAULT_MESSAGES = ROOT / "data" / "messages" / "messages_no_tool.json"
DEFAULT_TOOLS_SCHEMA = ROOT / "data" / "messages" / "tools_schema_basic.json"
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "model.yaml"
DEFAULT_OUTDIR = ROOT / "outputs" / "B4" / "messages_file_test"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B4 file-input smoke test with real Qwen.")
    parser.add_argument("--messages", default=str(DEFAULT_MESSAGES))
    parser.add_argument("--tools_schema", default=str(DEFAULT_TOOLS_SCHEMA))
    parser.add_argument("--model_config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--mode", choices=["prompt_json"], default="prompt_json")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--artifact_stem", default="messages_file_test")
    return parser


def _brief(ai_message: dict[str, Any], raw_record: dict[str, Any]) -> dict[str, Any]:
    metadata = ai_message.get("metadata", {}) if isinstance(ai_message.get("metadata"), dict) else {}
    return {
        "status": raw_record.get("status"),
        "route": metadata.get("route"),
        "route_chain": metadata.get("route_chain"),
        "route_chain_state": metadata.get("route_chain_state"),
        "selected_model_profile": metadata.get("selected_model_profile"),
        "model_selection": metadata.get("model_selection"),
        "tool_calls": ai_message.get("tool_calls", []),
        "has_content": bool(str(ai_message.get("content", "")).strip()),
        "usage": raw_record.get("usage"),
        "native_tool_raw_text": raw_record.get("native_tool_raw_text"),
    }


def main() -> int:
    args = build_parser().parse_args()
    messages_path = Path(args.messages).resolve()
    tools_schema_path = Path(args.tools_schema).resolve()
    model_config_path = Path(args.model_config).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    result = generate_ai_message(
        str(model_config_path),
        read_json(messages_path),
        read_json(tools_schema_path),
        mode=args.mode,
        artifact_dir=str(outdir),
        artifact_stem=args.artifact_stem,
    )
    raw_path = outdir / f"{args.artifact_stem}_raw_model_output.json"
    raw_record = read_json(raw_path) if raw_path.exists() else {}
    summary = {
        "messages_path": str(messages_path),
        "tools_schema_path": str(tools_schema_path),
        "model_config_path": str(model_config_path),
        "outdir": str(outdir),
        "result": result,
        "brief": _brief(result["ai_message"], raw_record),
    }
    write_json(summary, outdir / f"{args.artifact_stem}_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
