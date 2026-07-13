from __future__ import annotations

from b4.model_adapters import (
    _MODEL_CACHE,
    _dtype_value,
    _extract_tool_result,
    _load_model_bundle,
    _mock_generate,
    _model_cache_key,
    _prompt_json_generate,
    _three_points,
)
from b4.output_parser import (
    _candidate_to_message,
    _parse_json_with_backtick_tail,
    _parse_model_output,
    _parse_tool_calls_fragment,
)
from b4.prompt_builder import _build_prompt_messages
from b4.recorder import _artifact_paths
from b4.service import PARSE_ERROR_CONTENT, _load_model_config, build_parser, generate_ai_message, main


if __name__ == "__main__":
    raise SystemExit(main())
