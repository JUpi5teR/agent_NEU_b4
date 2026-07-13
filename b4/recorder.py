from __future__ import annotations

from pathlib import Path
from typing import Any

from common.io_utils import append_jsonl, write_json


def _artifact_paths(artifact_dir: str | Path, stem: str | None) -> tuple[Path, Path, Path]:
    directory = Path(artifact_dir)
    prefix = f"{stem}_" if stem else ""
    return (
        directory / f"{prefix}raw_model_output.json",
        directory / f"{prefix}ai_message.json",
        directory / "llm_run_log.jsonl",
    )


class RunRecorder:
    """Persist B4 raw output, parsed message, and compact run log."""

    def record(
        self,
        artifact_dir: str | None,
        artifact_stem: str | None,
        raw_record: dict[str, Any],
        ai_message: dict[str, Any],
    ) -> None:
        if not artifact_dir:
            return
        raw_path, message_path, log_path = _artifact_paths(artifact_dir, artifact_stem)
        write_json(raw_record, raw_path)
        write_json(ai_message, message_path)
        append_jsonl(
            {
                "timestamp": raw_record.get("generated_at"),
                "mode": raw_record.get("mode"),
                "status": raw_record.get("status"),
                "raw_output_path": str(raw_path),
                "ai_message_path": str(message_path),
                "error": raw_record.get("error"),
                "route": raw_record.get("route"),
            },
            log_path,
        )
