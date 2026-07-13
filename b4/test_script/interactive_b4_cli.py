#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code"))

from b4.service import generate_ai_message
from common.io_utils import read_json
from common.schemas import make_skill_result, make_tool_message


DEFAULT_MODEL_CONFIG = ROOT / "configs" / "model.yaml"
DEFAULT_TOOLS_SCHEMA = ROOT / "data" / "messages" / "tools_schema_basic.json"
DEFAULT_OUTDIR = ROOT / "outputs" / "B4" / "interactive_cli"

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
BLUE = "\033[34m"


class Spinner:
    def __init__(self, text: str) -> None:
        self.text = text
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "Spinner":
        if sys.stdout.isatty():
            self._thread.start()
        else:
            print(f"{DIM}{self.text}...{RESET}", flush=True)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._done.set()
        if sys.stdout.isatty():
            self._thread.join(timeout=0.2)
            print("\r" + " " * (len(self.text) + 8) + "\r", end="", flush=True)

    def _run(self) -> None:
        frames = ["|", "/", "-", "\\"]
        i = 0
        while not self._done.is_set():
            print(f"\r{CYAN}{frames[i % len(frames)]}{RESET} {DIM}{self.text}{RESET}", end="", flush=True)
            i += 1
            time.sleep(0.1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive real-model B4 CLI test.")
    parser.add_argument("--model_config", default=str(DEFAULT_MODEL_CONFIG), help="Path to model.yaml.")
    parser.add_argument("--tools_schema", default=str(DEFAULT_TOOLS_SCHEMA), help="Path to tools_schema JSON.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Directory for raw artifacts and session logs.")
    parser.add_argument("--system", default="You are a local tool-using agent. Use tools when needed.")
    parser.add_argument("--mode", choices=["prompt_json", "builtin_tools"], default="prompt_json")
    return parser


def _now_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _usage_total(raw_record: dict[str, Any]) -> int | None:
    usage = raw_record.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
        return usage["total_tokens"]
    return None


def _usage_parts(raw_record: dict[str, Any]) -> tuple[str, str, str]:
    usage = raw_record.get("usage")
    if not isinstance(usage, dict):
        return "-", "-", "-"
    return (
        str(usage.get("input_tokens", "-")),
        str(usage.get("output_tokens", "-")),
        str(usage.get("total_tokens", "-")),
    )


def _model_path_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "-"
    return Path(value).name or value


def _latest_usage_records(raw_record: dict[str, Any]) -> list[dict[str, Any]]:
    records = raw_record.get("usage_records")
    if isinstance(records, list):
        return records
    usage = raw_record.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("stages"), list):
        return usage["stages"]
    return []


def _route_chain_text(state: Any) -> str:
    if not isinstance(state, dict):
        return "-"
    chain = state.get("route_chain")
    if isinstance(chain, list) and chain:
        current = state.get("current_route") or "?"
        status = state.get("status") or "?"
        return f"{' -> '.join(str(item) for item in chain)}  [{current}, {status}]"
    return str(state.get("current_route") or "-")


def _tool_line(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list) or not tool_calls:
        return "-"
    parts: list[str] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name", "unknown")
        args = call.get("args", {})
        arg_text = json.dumps(args, ensure_ascii=False, separators=(",", ":")) if isinstance(args, dict) else str(args)
        if len(arg_text) > 96:
            arg_text = arg_text[:93] + "..."
        parts.append(f"{name}({arg_text})")
    return "; ".join(parts) if parts else "-"


def _shorten(text: Any, limit: int = 500) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def _panel(title: str, rows: list[tuple[str, Any]], color: str = CYAN) -> None:
    label_width = max([len(label) for label, _ in rows] + [4])
    print(f"{color}{BOLD}{title}{RESET}")
    for label, value in rows:
        print(f"  {DIM}{label.ljust(label_width)}{RESET}  {value}")


def _print_ai_message(ai_message: dict[str, Any], raw_record: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    metadata = ai_message.get("metadata", {}) if isinstance(ai_message.get("metadata"), dict) else {}
    model_selection = metadata.get("model_selection") if isinstance(metadata.get("model_selection"), dict) else raw_record.get("model_selection")
    model_profile = metadata.get("selected_model_profile") or raw_record.get("selected_model_profile")
    route = metadata.get("route") or raw_record.get("route")
    route_chain_state = metadata.get("route_chain_state") or raw_record.get("route_chain_state")
    prompt_tokens, output_tokens, total_tokens = _usage_parts(raw_record)
    model_name = _model_path_name(raw_record.get("model"))
    if isinstance(model_selection, dict):
        model_name = _model_path_name(model_selection.get("model") or model_selection.get("model_name_or_path") or raw_record.get("model"))
    summary = {
        "route": route,
        "route_chain_state": route_chain_state,
        "selected_model_profile": model_profile,
        "model_selection": model_selection,
        "model": raw_record.get("model"),
        "usage": raw_record.get("usage"),
        "elapsed_s": round(elapsed_s, 3),
        "tool_calls": ai_message.get("tool_calls", []),
        "content": ai_message.get("content", ""),
    }

    _panel(
        "B4",
        [
            ("route", route or "-"),
            ("chain", _route_chain_text(route_chain_state)),
            ("profile", model_profile or "-"),
            ("model", model_name),
            ("tokens", f"in {prompt_tokens} / out {output_tokens} / total {total_tokens}"),
            ("latency", f"{elapsed_s:.2f}s"),
            ("tools", _tool_line(ai_message.get("tool_calls"))),
        ],
        CYAN,
    )
    content = _shorten(ai_message.get("content", ""), 900)
    if content:
        print(f"\n{GREEN}{BOLD}assistant{RESET}")
        print(content)
    if _latest_usage_records(raw_record):
        print(f"\n{MAGENTA}{BOLD}model calls{RESET}")
        for idx, record in enumerate(_latest_usage_records(raw_record), start=1):
            usage = record.get("usage") if isinstance(record, dict) else None
            usage_text = "-"
            if isinstance(usage, dict):
                usage_text = str(usage.get("total_tokens", "-"))
            print(
                f"  {idx:02d}. {record.get('task', '-')} | "
                f"{record.get('model_profile', '-') or '-'} | "
                f"{_model_path_name(record.get('model'))} | tokens {usage_text}"
            )
    print()
    return summary


def _load_raw(outdir: Path, stem: str) -> dict[str, Any]:
    path = outdir / f"{stem}_raw_model_output.json"
    return read_json(path) if path.exists() else {}


def _mock_tool_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    latest = next((item for item in reversed(messages) if item.get("role") == "assistant" and item.get("tool_calls")), None)
    if latest is None:
        print(f"{YELLOW}No previous assistant tool_calls found.{RESET}", flush=True)
        return None
    call = latest["tool_calls"][0]
    result = make_skill_result(
        str(call.get("name", "unknown")),
        "success",
        call.get("args", {}) if isinstance(call.get("args"), dict) else {},
        {"content": "Mock ToolMessage content for interactive B4 route-chain testing."},
        None,
        1.0,
    )
    return make_tool_message(str(call.get("id", "call_001")), str(call.get("name", "unknown")), json.dumps(result, ensure_ascii=False))


def _pending_tool_call(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    latest_assistant_index = None
    latest_calls = []
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            latest_assistant_index = index
            latest_calls = message.get("tool_calls", [])
            break
    if latest_assistant_index is None:
        return None
    following = messages[latest_assistant_index + 1 :]
    if any(message.get("role") not in {"tool"} for message in following):
        return None
    received_ids = {message.get("tool_call_id") for message in following if message.get("role") == "tool"}
    expected_ids = [call.get("id") for call in latest_calls if isinstance(call, dict) and call.get("id")]
    missing = [call_id for call_id in expected_ids if call_id not in received_ids]
    if not missing:
        return None
    return {"missing_tool_call_ids": missing, "tool_calls": latest_calls}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _append_transcript(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def _print_header(args: argparse.Namespace, session_dir: Path) -> None:
    print(f"{BOLD}B4 interactive CLI{RESET}")
    print(f"{DIM}root      {ROOT}{RESET}")
    print(f"{DIM}model     {Path(args.model_config).resolve()}{RESET}")
    print(f"{DIM}tools     {Path(args.tools_schema).resolve()}{RESET}")
    print(f"{DIM}mode      {args.mode}{RESET}")
    print(f"{DIM}logs      {session_dir}{RESET}")
    print(f"{DIM}commands  :quit  :reset  :history  :mock_tool{RESET}\n")


def main() -> int:
    args = build_parser().parse_args()
    model_config = str(Path(args.model_config).resolve())
    tools_schema = read_json(Path(args.tools_schema).resolve())
    outdir = Path(args.outdir).resolve()
    session_dir = outdir / f"session_{_now_compact()}"
    session_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = session_dir / "transcript.md"
    events_path = session_dir / "events.jsonl"
    history_path = session_dir / "messages_final.json"
    messages: list[dict[str, Any]] = [{"role": "system", "content": args.system}]
    turn = 0

    _print_header(args, session_dir)
    _write_json(
        session_dir / "session_config.json",
        {
            "created_at": _now_iso(),
            "root": str(ROOT),
            "model_config": model_config,
            "tools_schema": str(Path(args.tools_schema).resolve()),
            "mode": args.mode,
            "system": args.system,
        },
    )
    _append_transcript(transcript_path, f"# B4 interactive CLI\n\n- created_at: {_now_iso()}\n- mode: {args.mode}\n")

    while True:
        try:
            text = input(f"{BLUE}user>{RESET} ").strip()
        except EOFError:
            print()
            _write_json(history_path, messages)
            return 0
        if not text:
            continue
        if text in {":quit", ":q", "exit", "quit"}:
            _write_json(history_path, messages)
            print(f"{DIM}session saved to {session_dir}{RESET}")
            return 0
        if text == ":reset":
            messages = [{"role": "system", "content": args.system}]
            _append_jsonl(events_path, {"type": "reset", "time": _now_iso()})
            _append_transcript(transcript_path, "\n---\n\n`reset`\n")
            print(f"{DIM}Messages reset.{RESET}", flush=True)
            continue
        if text == ":history":
            print(json.dumps(messages, ensure_ascii=False, indent=2), flush=True)
            continue
        if text == ":mock_tool":
            tool_message = _mock_tool_message(messages)
            if tool_message is not None:
                messages.append(tool_message)
                _append_jsonl(events_path, {"type": "mock_tool", "time": _now_iso(), "message": tool_message})
                _append_transcript(transcript_path, f"\n**tool**\n\n```json\n{json.dumps(tool_message, ensure_ascii=False, indent=2)}\n```\n")
                print(f"{YELLOW}mock tool feedback appended; send the next user message to continue the route chain.{RESET}")
            continue

        pending_tool = _pending_tool_call(messages)
        if pending_tool is not None:
            missing = ", ".join(str(item) for item in pending_tool.get("missing_tool_call_ids", []))
            print(
                f"{YELLOW}Previous route is waiting for ToolMessage ({missing}). "
                f"Use :mock_tool to simulate B3 feedback, or :reset to start a new task.{RESET}",
                flush=True,
            )
            _append_jsonl(events_path, {"type": "blocked_pending_tool", "time": _now_iso(), "user": text, "pending_tool": pending_tool})
            continue

        messages.append({"role": "user", "content": text})
        _append_transcript(transcript_path, f"\n**user**\n\n{text}\n")
        turn += 1
        stem = f"turn_{turn:03d}"
        started = time.perf_counter()
        try:
            with Spinner("B4 is routing, selecting model, and generating"):
                result = generate_ai_message(
                    model_config,
                    messages,
                    tools_schema,
                    mode=args.mode,
                    artifact_dir=str(session_dir),
                    artifact_stem=stem,
                )
        except Exception as exc:
            elapsed_s = time.perf_counter() - started
            error_event = {"type": "error", "time": _now_iso(), "turn": turn, "elapsed_s": elapsed_s, "error": repr(exc)}
            _append_jsonl(events_path, error_event)
            _append_transcript(transcript_path, f"\n**error**\n\n```text\n{repr(exc)}\n```\n")
            print(f"{RED}B4 failed: {exc!r}{RESET}", flush=True)
            continue
        elapsed_s = time.perf_counter() - started
        ai_message = result["ai_message"]
        messages.append(ai_message)
        raw_record = _load_raw(session_dir, stem)
        summary = _print_ai_message(ai_message, raw_record, elapsed_s)
        _append_jsonl(
            events_path,
            {
                "type": "turn",
                "time": _now_iso(),
                "turn": turn,
                "user": text,
                "summary": summary,
                "status": result.get("status"),
                "error": result.get("error"),
                "artifact_stem": stem,
            },
        )
        _append_transcript(
            transcript_path,
            "\n**assistant**\n\n"
            + (_shorten(ai_message.get("content", ""), 4000) or "")
            + "\n\n```json\n"
            + json.dumps(summary, ensure_ascii=False, indent=2)
            + "\n```\n",
        )
        _write_json(history_path, messages)


if __name__ == "__main__":
    raise SystemExit(main())
