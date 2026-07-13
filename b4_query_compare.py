#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from b4.capability import match_capabilities
from b4.service import generate_ai_message
from common.io_utils import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "configs" / "model.yaml"
TOOLS_SCHEMA = ROOT / "data" / "messages" / "tools_schema_basic.json"
QUERY_FILE = ROOT / "code" / "queries.txt"
OUTDIR = ROOT / "outputs" / "B4" / "B4_query_compare"
MODES = {
    "qwen_prompt_json": {"llm_mode": "prompt_json", "tool_calling_mode": "prompt_json"},
    "qwen_builtin_tools": {"llm_mode": "prompt_json", "tool_calling_mode": "builtin_tools"},
}


def load_queries() -> list[str]:
    queries = []
    for line in QUERY_FILE.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            queries.append(text)
    limit_text = os.environ.get("B4_QUERY_LIMIT")
    if limit_text:
        limit = int(limit_text)
        if limit > 0:
            return queries[:limit]
    return queries


def expected_tools(query: str) -> list[str]:
    lowered = query.lower()
    tools = []
    if re.search(r"\d+(?:\.\d+)?\s*[+*/-]\s*\d+", query) or any(marker in query for marker in ("计算", "求和")):
        tools.append("calculator")
    if any(marker in lowered or marker in query for marker in ("docs/", ".txt", ".md", "阅读", "读取", "文件")):
        tools.append("file_reader")
    if any(marker in lowered or marker in query for marker in ("search", "find", "搜索", "查找", "检索")):
        tools.append("local_file_search")
    if any(marker in lowered or marker in query for marker in ("csv", "table", "spreadsheet", "表格")):
        tools.append("table_analyzer")
    return dedupe(tools)


def dedupe(items: list[str]) -> list[str]:
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def route_category(ai_message: dict[str, Any], status: str) -> str:
    if status == "error":
        return "error"
    metadata = ai_message.get("metadata", {}) if isinstance(ai_message.get("metadata"), dict) else {}
    if isinstance(metadata.get("guard"), dict):
        return "error"
    route = metadata.get("route")
    if route in {"direct_answer", "tool_call", "plan_execute", "deep_think"}:
        return str(route)
    if ai_message.get("tool_calls"):
        return "tool_call"
    if ai_message.get("content"):
        return "direct_answer"
    return "error"


def build_feedback(query: str, ai_message: dict[str, Any], status: str, error: Any) -> str:
    content = ai_message.get("content", "")
    metadata = ai_message.get("metadata", {}) if isinstance(ai_message.get("metadata"), dict) else {}
    if isinstance(metadata.get("guard"), dict) and content:
        return content
    if status == "error":
        if error:
            return f"B4 error: {json.dumps(error, ensure_ascii=False)}"
        return "B4 returned error status."
    tool_calls = ai_message.get("tool_calls", [])
    if isinstance(metadata.get("guard"), dict):
        return content or f"B4 guard blocked unsafe action: {json.dumps(metadata['guard'], ensure_ascii=False)}"
    if tool_calls:
        names = ", ".join(call.get("name", "unknown") for call in tool_calls if isinstance(call, dict))
        return f"B4 requests tool call(s): {names}. No tools are executed in this comparison test."
    if content:
        return content
    expected = expected_tools(query)
    if expected:
        return f"B4 produced no content and no tool call, but query appears to need: {', '.join(expected)}."
    return "B4 produced empty content and no tool call."


def evaluate(query: str, ai_message: dict[str, Any], status: str) -> dict[str, Any]:
    expected = expected_tools(query)
    predicted = [call.get("name") for call in ai_message.get("tool_calls", []) if isinstance(call, dict)]
    content = ai_message.get("content", "")
    category = route_category(ai_message, status)
    guard = ai_message.get("metadata", {}).get("guard") if isinstance(ai_message.get("metadata"), dict) else None
    if guard:
        ok = True
    elif status == "error":
        ok = False
    elif expected:
        ok = bool(predicted) and all(tool in predicted for tool in expected)
    else:
        ok = bool(content.strip()) and not predicted
    return {
        "category": category,
        "expected_tools": expected,
        "predicted_tools": predicted,
        "ok": ok,
        "has_content": bool(content.strip()),
        "has_tool_calls": bool(predicted),
    }


def build_model_config(tool_calling_mode: str) -> Path:
    config = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("model.yaml must contain an object")
    config.setdefault("tool_calling", {})["mode"] = tool_calling_mode
    temp = tempfile.NamedTemporaryFile(
        "w",
        suffix=f"_{tool_calling_mode}.yaml",
        encoding="utf-8",
        delete=False,
    )
    with temp:
        yaml.safe_dump(config, temp, allow_unicode=True, sort_keys=False)
    return Path(temp.name)


def tool_success(ai_message: dict[str, Any], expected: list[str], status: str) -> bool:
    if status != "success":
        return False
    predicted = [call.get("name") for call in ai_message.get("tool_calls", []) if isinstance(call, dict)]
    if not expected:
        return not predicted and bool(str(ai_message.get("content", "")).strip())
    return bool(predicted) and all(tool in predicted for tool in expected)


def run_case(
    mode_name: str,
    mode_config: dict[str, str],
    query: str,
    index: int,
    tools_schema: list[dict[str, Any]],
    model_config_path: Path,
) -> dict[str, Any]:
    temp_dir = OUTDIR / mode_name / "_tmp" / f"query_{index:03d}"
    llm_mode = mode_config["llm_mode"]
    result = generate_ai_message(
        str(model_config_path),
        [{"role": "user", "content": query}],
        tools_schema,
        llm_mode,
        str(temp_dir),
        f"q{index:03d}",
    )
    ai_message = result["ai_message"]
    status = result.get("status", "error")
    category = route_category(ai_message, status)
    case_dir = OUTDIR / mode_name / category / f"query_{index:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    raw_path = temp_dir / f"q{index:03d}_raw_model_output.json"
    raw = read_json(raw_path) if raw_path.exists() else {}
    evaluation = evaluate(query, ai_message, status)
    capability_match = match_capabilities(
        [{"role": "user", "content": query}],
        tools_schema,
        raw.get("goal_analysis") if isinstance(raw.get("goal_analysis"), dict) else None,
        raw.get("complexity") if isinstance(raw.get("complexity"), dict) else None,
    ).to_dict()
    record = {
        "index": index,
        "mode": mode_name,
        "llm_mode": llm_mode,
        "tool_calling_mode": mode_config["tool_calling_mode"],
        "query": query,
        "status": status,
        "category": category,
        "feedback": build_feedback(query, ai_message, status, result.get("error")),
        "evaluation": evaluation,
        "route": ai_message.get("metadata", {}).get("route") if isinstance(ai_message.get("metadata"), dict) else None,
        "route_reason": ai_message.get("metadata", {}).get("route_reason") if isinstance(ai_message.get("metadata"), dict) else None,
        "route_chain_state": ai_message.get("metadata", {}).get("route_chain_state")
        if isinstance(ai_message.get("metadata"), dict)
        else None,
        "capability_match": capability_match,
        "selected_model_profile": ai_message.get("metadata", {}).get("selected_model_profile")
        if isinstance(ai_message.get("metadata"), dict)
        else None,
        "model_selection": raw.get("model_selection"),
        "usage": raw.get("usage"),
        "tool_success": tool_success(ai_message, evaluation["expected_tools"], status),
        "ai_message": ai_message,
        "raw_record": raw,
        "artifact_dir": str(case_dir),
    }
    write_json(ai_message, case_dir / "ai_message.json")
    write_json(raw, case_dir / "raw_model_output.json")
    write_json(record, case_dir / "result.json")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_profile = {
        profile: sum(1 for record in records if record.get("selected_model_profile") == profile)
        for profile in sorted({str(record.get("selected_model_profile")) for record in records})
    }
    return {
        "total": len(records),
        "ok_count": sum(1 for record in records if record["evaluation"]["ok"]),
        "finding_count": sum(1 for record in records if not record["evaluation"]["ok"]),
        "tool_success_count": sum(1 for record in records if record.get("tool_success")),
        "tool_success_rate": (
            sum(1 for record in records if record.get("tool_success")) / len(records)
            if records
            else 0.0
        ),
        "by_category": {
            category: sum(1 for record in records if record["category"] == category)
            for category in sorted({record["category"] for record in records})
        },
        "by_selected_model_profile": by_profile,
        "unsupported_capability_count": sum(
            1 for record in records if record.get("capability_match", {}).get("missing_required")
        ),
        "total_usage": {
            "input_tokens": sum((record.get("usage") or {}).get("input_tokens", 0) for record in records),
            "output_tokens": sum((record.get("usage") or {}).get("output_tokens", 0) for record in records),
            "total_tokens": sum((record.get("usage") or {}).get("total_tokens", 0) for record in records),
        },
    }


def run() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tools_schema = read_json(TOOLS_SCHEMA)
    queries = load_queries()
    all_records = []
    mode_summaries = {}
    for mode_name, mode_config in MODES.items():
        model_config_path = build_model_config(mode_config["tool_calling_mode"])
        try:
            mode_records = [
                run_case(mode_name, mode_config, query, index, tools_schema, model_config_path)
                for index, query in enumerate(queries, 1)
            ]
            write_json(mode_records, OUTDIR / mode_name / "results.json")
            mode_summaries[mode_name] = summarize(mode_records)
            all_records.extend(mode_records)
            tmp_dir = OUTDIR / mode_name / "_tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
        finally:
            try:
                model_config_path.unlink()
            except FileNotFoundError:
                pass
    summary = {
        "status": "success" if all(record["evaluation"]["ok"] for record in all_records) else "has_findings",
        "query_file": str(QUERY_FILE),
        "output_dir": str(OUTDIR),
        "modes": MODES,
        "mode_summaries": mode_summaries,
        "records": all_records,
    }
    write_json(summary, OUTDIR / "summary.json")
    return summary


def main() -> int:
    summary = run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
