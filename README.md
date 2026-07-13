# 本地 Agent 框架（实训 B 方向）

本项目使用 Python 3.10 实现一个本地文件驱动的 Agent 框架。B1–B5 均保留独立命令行入口，使用服务器本地 Qwen3.5-4B。

模块边界如下：

| 模块 | 入口文件 | 职责 |
|---|---|---|
| B1 | `code/b1_agent_runtime.py` | Agent 总控、消息管理、循环控制和产物汇总。 |
| B2 | `code/b2_run_skill.py` | 独立运行五个基础 Skill。 |
| B3 | `code/b3_tool_layer.py` | 生成 tools schema，校验并执行 tool calls。 |
| B4 | `code/b4_local_agent_llm.py` | 使用 mock 或本地 LLM 生成标准 AIMessage，不执行工具。 |
| B5 | `code/b5_memory.py` | 查找、截断、保存 memory 文档并维护索引。 |
| 完整演示 | `code/run_full_demo.py` | 调用 B1 跑通完整 Agent，并生成汇总报告。 |

B4的mock模式不真实加载、运行模型，作为无 GPU、无模型或模块联调时的调试模式。`prompt_json`模式则加载本地模型真实运行。

## B4 Cognitive Engine

入口：`code/b4_local_agent_llm.py`

新的 B4 实现为“结构化任务规划与执行系统”，在现有 B4 文件内按插件化职责组织代码：

```text
code/b4_local_agent_llm.py
  B4Plugin              统一插件接口
  GoalParserPlugin      Goal Parser 插件
  PlannerPlugin         Planner 插件，内部包含 Tree of Thoughts 优化器
  CriticPlugin          Critic 插件
  SchedulerPlugin       Scheduler 插件
  ReflectionPlugin      Reflection 插件
  B4CognitiveEngine     核心调度器，只由它协调插件通信
```

固定执行流如下，顺序不可改变：

```text
Goal Parser
-> Planner
-> Tree of Thoughts (Planner 内部优化器)
-> Critic
-> Scheduler
-> Reflection Loop
```

`Tree of Thoughts` 不是独立插件，也不是独立流程，而是 `PlannerPlugin` 内部的优化器，按 `Expand -> Score -> Prune -> Expand` 优化初始计划。各插件之间不直接调用，只通过 `B4CognitiveEngine` 调度通信。

`GoalParserPlugin` 会额外生成 `decision` 字段，用于判断当前目标应进入哪种模式：

- `direct_answer`：直接回答
- `reasoning_answer`：推理回答，严格推理部分由 Planner 内部的 Tree of Thoughts 展开候选思路并量化比较
- `plan`：制定可执行计划
- `execute`：需要执行任务或调用工具

`decision` 同时包含 `needs_tool`、`reasoning_required`、`answer_mode`、`tool_candidates`、`confidence` 和 `evidence`，用于说明是否需要工具、是否需要推理以及判断依据。

`CriticPlugin` 除了输出总分 `score` 和 `reason`，还会输出量化评估 `metrics`：

- `completeness`
- `executability`
- `constraint_fit`
- `non_duplication`
- `tool_fit`
- `reasoning_quality`
- `issue_penalty`

运行示例：

```bash
cd code
python b4_local_agent_llm.py --goal "写一个Python学习路线，必须每一步可执行、可验证，优先安排基础语法和项目练习"
```

输出为 JSON，包含 `goal_json`、优化后的 `plan`、`critic` 评分、`schedule`、`reflection` 和完整 `trace`。

独立测试：

```bash
python -m py_compile code/b4_local_agent_llm.py
```

## 1. 环境准备

所有模块统一使用项目根目录下的 `requirements.txt`。推荐每位同学新建自己的conda环境，安装步骤如下：

```bash
conda create -n your_env python=3.10 -y
conda activate your_env
export PYTHONNOUSERSITE=1
pip install -r requirements.txt
```
其中"export PYTHONNOUSERSITE=1"的作用是：让Python启动时禁止加载用户级site-packages目录，保证只用当前环境自己的包

模型使用Qwen3.5-4B
统一配置文件为 `configs/model.yaml`，实际使用时需要把model_name_or_path和tokenizer_name_or_path改为模型所在的文件路径。

演示命令均从 `agent/code` 目录执行：

```bash
cd agent/code
```

## 2. 公共数据格式

### 2.1 SkillResult

B2 和 B3 使用以下 JSON 对象记录一次 Skill 执行：

```json
{
  "skill_name": "calculator",
  "status": "success",
  "input": {"expression": "23 * 17 + 9"},
  "output": {"result": 400},
  "error": null,
  "latency_ms": 0.5
}
```

失败时 `status` 为 `error`、`output` 为 `null`，`error` 包含异常类型和错误信息。

### 2.2 AIMessage

工具调用型 AIMessage：

```json
{
  "role": "assistant",
  "content": "",
  "tool_calls": [
    {
      "id": "call_001",
      "name": "file_reader",
      "args": {"path": "docs/agent_intro.txt", "max_chars": 2000}
    }
  ]
}
```

最终回答型 AIMessage 的 `content` 非空，`tool_calls` 为空数组。

### 2.3 ToolMessage

```json
{
  "role": "tool",
  "tool_call_id": "call_001",
  "name": "file_reader",
  "content": "{\"skill_name\":\"file_reader\",...}",
  "status": "success"
}
```

`content` 是序列化后的 SkillResult JSON 字符串；`tool_call_id` 用于关联前面的 AIMessage tool call。

## 3. B2：Skill 独立演示

入口：`code/b2_run_skill.py`

### 3.1 通用命令行输入

| 参数 | 说明 |
|---|---|
| `--skill` | Skill 名称：`calculator`、`file_reader`、`local_file_search`、`table_analyzer` 或 `format_converter`。 |
| `--input` | 对应 Skill 的 JSON 输入文件。顶层必须是 JSON 对象。 |
| `--outdir` | B2 输出目录。 |
| `--data_root` | 可选的数据根目录；未提供时使用项目的 `data/`。 |

### 3.2 每个 Skill 的输入文件

| Skill | 正常输入文件 | 关键输入字段 | 异常输入样例 |
|---|---|---|---|
| calculator | `data/tool_inputs/tool_input_calculator.json` | `expression`：数学表达式字符串。 | `tool_input_calculator_error.json` |
| file_reader | `data/tool_inputs/tool_input_file_reader.json` | `path`：相对 `data/` 的 txt/md 路径；`max_chars`：最大返回字符数。 | `tool_input_file_reader_error.json` |
| local_file_search | `data/tool_inputs/tool_input_file_search.json` | `query`、`root_dir`、`file_types`、`top_k`。 | `tool_input_file_search_error.json` |
| table_analyzer | `data/tool_inputs/tool_input_table_analyzer.json` | `path`：CSV/TSV 路径；`max_rows_preview`；`describe`。 | `tool_input_table_analyzer_error.json` |
| format_converter | `data/tool_inputs/tool_input_format_converter.json` | `text`；`target_format`：`markdown` 或 `json`；可选 `output_filename`。 | `tool_input_format_converter_error.json` |

文件类 Skill 的相对路径以 `data/` 为根。例如 `docs/agent_intro.txt` 实际对应 `data/docs/agent_intro.txt`。

### 3.3 演示命令

```bash
python b2_run_skill.py --skill calculator --input ../data/tool_inputs/tool_input_calculator.json --outdir ../outputs/B2_skills
python b2_run_skill.py --skill file_reader --input ../data/tool_inputs/tool_input_file_reader.json --outdir ../outputs/B2_skills
python b2_run_skill.py --skill local_file_search --input ../data/tool_inputs/tool_input_file_search.json --outdir ../outputs/B2_skills
python b2_run_skill.py --skill table_analyzer --input ../data/tool_inputs/tool_input_table_analyzer.json --outdir ../outputs/B2_skills
python b2_run_skill.py --skill format_converter --input ../data/tool_inputs/tool_input_format_converter.json --outdir ../outputs/B2_skills
```

### 3.4 B2 输出

| 输出文件 | 格式 | 说明 |
|---|---|---|
| `outputs/B2_skills/calculator_result.json` | JSON 对象（SkillResult） | calculator 最近一次运行的输入、结果或错误、耗时。 |
| `outputs/B2_skills/file_reader_result.json` | JSON 对象（SkillResult） | file_reader 最近一次运行结果；业务输出包含内容、字符数、来源和截断标志。 |
| `outputs/B2_skills/local_file_search_result.json` | JSON 对象（SkillResult） | 文件搜索结果；每项包含路径、匹配分数和命中片段。 |
| `outputs/B2_skills/table_analyzer_result.json` | JSON 对象（SkillResult） | 表格行列数、列名、预览和数值列统计。 |
| `outputs/B2_skills/format_converter_result.json` | JSON 对象（SkillResult） | 转换后的 Markdown 或 JSON 文本，以及生成文件路径。 |
| `outputs/B2_skills/skill_run_log.jsonl` | JSONL | B2 运行历史；每行记录时间、Skill、状态、结果路径和耗时。 |

同一个 Skill 再次运行时会覆盖对应的 `*_result.json`；日志采用追加写入。异常样例被正常捕获并写入 error SkillResult，CLI 仍返回 0。

## 4. B3：Tools Schema 与工具执行

入口：`code/b3_tool_layer.py`

### 4.1 输入文件

| 输入文件 | 说明 |
|---|---|
| `configs/tools.yaml` | 定义 toolsets、每个工具的 Python 模块/函数、描述、输入参数、必填参数、返回说明和 `data_root`。 |
| `data/messages/ai_message_with_tool_calls.json` | 工具执示演示的输入，正常file_reader调用的样例,包含标准 AIMessage 及其 `tool_calls`。 |
| `data/messages/b3_tool_call_format_converter_valid.json` | 正常 format_converter 调用样例，验证 B3 注入 `output_dir` 并生成文件。 |
| `data/messages/b3_tool_call_unknown_tool.json` | 未知工具错误样例。 |
| `data/messages/b3_tool_call_missing_required.json` | 缺少必填参数错误样例。 |

B3 会根据 Skill 函数签名自动注入 data_root 和 output_dir；前者用于读取文件类Skill定位输入文件夹data/，后者用于format_converter输出生成文件。

### 4.2 演示命令

生成tools_schema:
```bash
python b3_tool_layer.py --tools_config ../configs/tools.yaml --toolset basic_tools --export_schema --outdir ../outputs/B3_tools
```
执行tool_calls:
```bash
python b3_tool_layer.py --tools_config ../configs/tools.yaml --toolset basic_tools --tool_calls ../data/messages/ai_message_with_tool_calls.json --execute --outdir ../outputs/B3_tools
```

基础样例:
```bash
python b3_tool_layer.py --tools_config ../configs/tools.yaml --toolset basic_tools --tool_calls ../data/messages/b3_tool_call_format_converter_valid.json --execute --outdir ../outputs/B3_tools/format_converter_valid
python b3_tool_layer.py --tools_config ../configs/tools.yaml --toolset basic_tools --tool_calls ../data/messages/b3_tool_call_unknown_tool.json --execute --outdir ../outputs/B3_tools/unknown_tool
python b3_tool_layer.py --tools_config ../configs/tools.yaml --toolset basic_tools --tool_calls ../data/messages/b3_tool_call_missing_required.json --execute --outdir ../outputs/B3_tools/missing_required
```

错误样例不会让 CLI 崩溃；错误会写入 `tool_messages.json` 和 `tool_call_log.jsonl` 中的 `status=error` SkillResult。

### 4.3 B3 输出

| 输出文件 | 格式 | 说明 |
|---|---|---|
| `outputs/B3_tools/tools_schema.json` | JSON 数组 | 当前 OpenAI 风格的函数工具说明schema；`x-returns` 描述工具返回值。 |
| `outputs/B3_tools/tool_schema_report.json` | JSON 对象 | schema 导出报告，包含 toolset、工具数量和工具名称列表。 |
| `outputs/B3_tools/tool_messages.json` | JSON 数组 | 本次执行生成的 ToolMessage 列表。 |
| `outputs/B3_tools/tool_call_log.jsonl` | JSONL | 每个 tool call 的完整执行记录，含未转义 SkillResult、状态、参数和耗时。 |

schema/report/tool messages 会被最近一次运行覆盖；tool-call 日志追加写入。

## 5. B5：Memory 查找与保存

入口：`code/b5_memory.py`

### 5.1 输入文件

| 输入文件 | 说明 |
|---|---|
| `configs/memory.yaml` | 记忆根目录、全局/对话目录、索引路径和最大注入字符数。 |
| `memory/memory_index.json` | `memory_id → 元数据` 的索引，记录类型、标题、摘要、路径和时间。 |
| `memory/global/*.md` | 全局记忆文档。 |
| `memory/conversations/*.md` | 对话记忆文档。 |
| `data/memory_inputs/memory_save_input.json` | 保存对话类型记忆的样例，包含 conversation ID、保存类型及三个来源文件路径。 |
| `data/memory_inputs/memory_save_global_input.json` | 保存全局类型记忆的样例，会写入项目正式记忆目录 `memory/global/` 和 `memory/memory_index.json`。 |
| `data/memory_inputs/sample_messages.json` | 演示保存记忆时使用的消息数组。 |
| `data/memory_inputs/sample_trace.json` | 演示保存记忆时使用的 trace 对象。 |
| `data/memory_inputs/sample_final_answer.md` | 演示保存记忆时使用的最终回答。 |
| `configs/memory_small_limit.yaml` | 复用当前 `memory/`，仅降低 `max_memory_chars` 用于截断演示。 |

演示B5的查找模式时，直接通过命令行参数传入。`memory_save_input.json` 中的三个来源路径相对于该 JSON 文件所在目录解析，不依赖 B1 输出。

### 5.2 演示命令

查找记忆:
```bash
python b5_memory.py --config ../configs/memory.yaml --select_memory_ids mem_conversation_conv_000 --use_global_memory true --query "Agent 系统如何调用工具？" --outdir ../outputs/B5_memory
python b5_memory.py --config ../configs/memory.yaml --select_memory_ids mem_missing_001 --use_global_memory false --query "验证缺失 memory id 的错误记录。" --outdir ../outputs/B5_memory/missing_id
python b5_memory.py --config ../configs/memory_small_limit.yaml --select_memory_ids mem_course_001 --use_global_memory false --query "验证 max_memory_chars 截断。" --outdir ../outputs/B5_memory/truncate
```

保存对话记忆
```bash
python b5_memory.py --config ../configs/memory.yaml --save_type conversation --save_input_path ../data/memory_inputs/memory_save_input.json --outdir ../outputs/B5_memory
python b5_memory.py --config ../configs/memory.yaml --save_type global --save_input_path ../data/memory_inputs/memory_save_global_input.json --outdir ../outputs/B5_memory/save_global
```

B5 保存命令会更新项目正式记忆目录：生成或覆盖 `memory/conversations/*.md` / `memory/global/*.md`，并新增或更新 `memory/memory_index.json`。`memory_save_global_input.json` 使用唯一 `conversation_id=demo_global_memory_001`，避免覆盖已有示例。

### 5.3 B5 输出

| 输出文件 | 格式 | 说明 |
|---|---|---|
| `outputs/B5_memory/selected_memory.json` | JSON 对象 | 本次选中的记忆内容、字符统计、截断标志和缺失 ID 错误。 |
| `outputs/B5_memory/memory_log.jsonl` | JSONL | 记忆查找与保存历史。 |
| `outputs/B5_memory/saved_memory.json` | JSON 对象 | 已保存记忆的 ID、类型、标题、摘要、文档路径、来源路径和时间。 |
| `memory/conversations/conv_sample_001.md` | Markdown | 独立保存命令生成的对话记忆文档，内含最终回答、messages 和 trace。 |
| `memory/global/demo_global_memory_001.md` | Markdown | 全局保存样例生成的全局记忆文档。 |
| `memory/memory_index.json` | JSON 对象 | 保存后新增或更新对应记忆元数据。 |

`selected_memory.json` 和 `saved_memory.json` 会覆盖；`memory_log.jsonl` 追加；索引和 Markdown 记忆文档会新增或更新。

## 6. B4：真实调用模型 / Mock 调试决策

入口：`code/b4_local_agent_llm.py`

B4 只生成 AIMessage，不执行工具。

### 6.1 输入文件

| 输入文件 | 说明 |
|---|---|
| `configs/model.yaml` | 统一真实模型配置：服务器本地 Qwen3.5-4B、Transformers、bf16、`prompt_json`。 |
| `data/messages/messages_no_tool.json` | 第一阶段独立运行输入，只含 system/user，真实模型应生成 tool call。 |
| `data/messages/messages_with_tool.json` | 第二阶段独立运行输入，已含 ToolMessage，真实模型应生成最终回答。 |
| `data/messages/messages_with_error_tool.json` | 已含失败 ToolMessage，验证模型直接说明失败并保持 `tool_calls=[]`。 |
| `data/messages/tools_schema_basic.json` | B4个人演示使用的预设tools_schema工具说明，不依赖B3的预先运行。 |

### 6.2 演示命令

真实加载调用模型:
B4 第一阶段，生成 tool_call：
```bash
python b4_local_agent_llm.py --model_config ../configs/model.yaml --messages ../data/messages/messages_no_tool.json --tools_schema ../data/messages/tools_schema_basic.json --mode prompt_json --outdir ../outputs/B4_llm/no_tool_real
```

B4 第二阶段，生成 final_answer：
```bash
python b4_local_agent_llm.py --model_config ../configs/model.yaml --messages ../data/messages/messages_with_tool.json --tools_schema ../data/messages/tools_schema_basic.json --mode prompt_json --outdir ../outputs/B4_llm/with_tool_real
```

B4处理工具调用失败的ToolMessage结果
```bash
python b4_local_agent_llm.py --model_config ../configs/model.yaml --messages ../data/messages/messages_with_error_tool.json --tools_schema ../data/messages/tools_schema_basic.json --mode prompt_json --outdir ../outputs/B4_llm/error_tool_real
```

mock调试模式:
```bash
python b4_local_agent_llm.py --model_config ../configs/model.yaml --messages ../data/messages/messages_no_tool.json --tools_schema ../data/messages/tools_schema_basic.json --mode mock --outdir ../outputs/B4_llm
python b4_local_agent_llm.py --model_config ../configs/model.yaml --messages ../data/messages/messages_with_tool.json --tools_schema ../data/messages/tools_schema_basic.json --mode mock --outdir ../outputs/B4_llm
```

mock 不加载模型、不占用显存，适合无 GPU、无本地模型或其他模块同学联调，不作为正式基础演示截图。

### 6.3 B4 输出

| 输出文件 | 格式 | 说明 |
|---|---|---|
| `outputs/B4_llm/<case>/raw_model_output.json` | JSON 对象 | 原始生成文本、解析候选、模式、backend、状态、错误和生成时间。若`status`为 `error`，说明模型输出未能解析成合法 AIMessagemock。mock模式也会生成该记录。 |
| `outputs/B4_llm/<case>/ai_message.json` | JSON 对象（AIMessage） | 规范化后的工具调用或最终回答。 |
| `outputs/B4_llm/<case>/llm_run_log.jsonl` | JSONL | B4 独立运行历史及产物路径。 |

mock调试的演示命令共用 ../outputs/B4_llm，因此第二条会覆盖第一条的 raw_model_output.json 和 ai_message.json，但日志会追加。

## 7. B1：Agent Runtime

入口：`code/b1_agent_runtime.py`

B1 支持两种明确的执行模式：`fixture` 用于完全隔离的个人演示，直接消费预设 memory、tools schema、AIMessage 和 ToolMessage；`integrated` 用于全系统演示，只通过 B3、B4、B5 的公开函数进行编排。

### 7.1 直接输入文件

#### 个人演示 fixture 模式

| 输入文件 | 说明 |
|---|---|
| `data/b1_fixtures/b1_fixture_input.json` | B1 个人演示入口，设置 `execution_mode=fixture` 并引用全部预设响应。 |
| `data/b1_fixtures/preset_memory.json` | 模拟 B5 返回的 selected memory。 |
| `data/b1_fixtures/preset_ai_messages.json` | 模拟 B4 依次返回的工具调用 AIMessage 和最终回答 AIMessage。 |
| `data/b1_fixtures/preset_tool_messages.json` | 按 `tool_call_id` 模拟 B3 返回的 ToolMessage。 |
| `data/messages/tools_schema_basic.json` | 模拟 B3 返回的固定 tools schema。 |

#### 全系统演示 integrated 模式

| 输入文件 | 说明 |
|---|---|
| `data/runtime_input.json` | file_reader 主线任务，读取 `docs/agent_intro.txt` 并总结三条中文要点。 |
| `data/runtime_input_0.json` | 无工具倾向任务，验证模型直接回答。 |
| `data/runtime_input_2.json` | calculator 任务。 |
| `data/runtime_input_3.json` | local_file_search 任务。 |
| `data/runtime_input_4.json` | table_analyzer 任务。 |
| `data/runtime_input_5.json` | format_converter 任务。 |
| `configs/tools.yaml` | 传给 B3，确定可用工具及数据根目录。 |
| `configs/memory.yaml` | 传给 B5，确定 memory 路径和长度上限。 |
| `configs/model.yaml` | 统一真实模型配置。 |

### 7.2 间接读取文件

| 文件 | 读取原因 |
|---|---|
| `prompts/local_tool_agent.txt` | fixture 和 integrated 输入指定的 SystemMessage 模板。 |
| `memory/memory_index.json`、`memory/global/*.md` | 仅 integrated 模式读取；B5 查找并返回 memory 上下文。 |
| `data/docs/agent_intro.txt` | integrated file_reader 流程实际读取；fixture 模式使用预设 ToolMessage。 |

### 7.3 演示命令

个人演示 fixture 模式：

```bash
python b1_agent_runtime.py --input ../data/b1_fixtures/b1_fixture_input.json --outdir ../outputs/B1_fixture
```

该命令不会调用 B2–B5。

全系统 integrated 调试模式：

```bash
python b1_agent_runtime.py --input ../data/runtime_input.json --tools_config ../configs/tools.yaml --memory_config ../configs/memory.yaml --model_config ../configs/model.yaml --llm_mode mock --outdir ../outputs/B1_runtime
```

该命令以 B1 为入口真实调用 B3、B4、B5，但 LLM 使用mock；正式全系统演示推荐使用下一节的真实模型 full demo，或将'--llm_mode mock'改为'--llm_mode prompt_json'。

### 7.4 B1 输出

#### 个人演示 fixture 模式

| 输出文件 | 格式 | 说明 |
|---|---|---|
| `outputs/B1_fixture/messages.json` | JSON 数组 | B1 根据预设响应维护出的完整消息序列，顶层固定为数组。 |
| `outputs/B1_fixture/trace.json` | JSON 对象 | B1 的分支判断、消息追加、工具轮次和最终状态。 |
| `outputs/B1_fixture/final_answer.md` | Markdown | 从预设最终 AIMessage 提取的最终回答。 |

fixture模式只生成以上三个文件，不生成 runtime log、memory文档或其他模块的产物。

#### 全系统演示 integrated 模式

| 输出文件 | 格式 | 说明 |
|---|---|---|
| `outputs/B1_runtime/messages.json` | JSON 数组 | 完整消息序列，顶层固定为数组。 |
| `outputs/B1_runtime/trace.json` | JSON 对象 | 运行状态、工具轮次、LLM 次数、每轮 AI/ToolMessage、memory 保存状态和错误。 |
| `outputs/B1_runtime/final_answer.md` | Markdown | 最终给用户的回答。 |
| `outputs/B1_runtime/selected_memory.json` | JSON 对象 | B1 调用 B5 后保存的 memory 选择结果。 |
| `outputs/B1_runtime/tools_schema.json` | JSON 数组 | B1 调用 B3 后保存的当前工具 schema。 |
| `outputs/B1_runtime/tool_schema_report.json` | JSON 对象 | 当前 toolset 的 schema 摘要。 |
| `outputs/B1_runtime/tool_messages.json` | JSON 数组 | 本次 Agent 运行累计产生的 ToolMessage。 |
| `outputs/B1_runtime/tool_call_log.jsonl` | JSONL | B3 工具执行明细。 |
| `outputs/B1_runtime/saved_memory.json` | JSON 对象 | B5 保存本轮对话后的结果。 |
| `outputs/B1_runtime/memory_log.jsonl` | JSONL | B1 运行期间的 memory 查找和保存记录。 |
| `outputs/B1_runtime/runtime_log.jsonl` | JSONL | B1 每次完整任务的状态、模式、轮次、调用数和总耗时。 |
| `outputs/B1_runtime/llm_calls/llm_call_NNN_raw_model_output.json` | JSON 对象 | 第 N 次 B4 调用的原始输出记录。 |
| `outputs/B1_runtime/llm_calls/llm_call_NNN_ai_message.json` | JSON 对象 | 第 N 次 B4 调用生成的标准 AIMessage。 |
| `outputs/B1_runtime/llm_calls/llm_run_log.jsonl` | JSONL | 本次 B1 运行内所有 B4 调用的日志。 |
| `memory/conversations/conv_001.md` | Markdown | B1 按 `save_memory=conversation` 保存的对话 memory。 |
| `memory/memory_index.json` | JSON 对象 | B1 保存 memory 后更新的索引。 |

## 8. 完整一键演示

入口：`code/run_full_demo.py`

### 8.1 输入文件

输入与 B1 相同：

- `data/runtime_input.json`
- `configs/tools.yaml`
- `configs/memory.yaml`
- `configs/model.yaml`
- 以上配置间接引用的 prompt、memory 和演示文档

### 8.2 命令

```bash
python run_full_demo.py --input ../data/runtime_input.json --tools_config ../configs/tools.yaml --memory_config ../configs/memory.yaml --model_config ../configs/model.yaml --llm_mode prompt_json --outdir ../outputs/full_demo
```

该正式命令会按 `runtime_input.json` 中的 `save_memory=conversation` 更新 `memory/conversations/conv_001.md` 和 `memory/memory_index.json`。重复演示前请确认是否接受覆盖该 conversation memory。

### 8.3 输出

`run_full_demo.py` 使用 `outputs/full_demo` 作为 B1 integrated 模式的 outdir，因此会生成 integrated 模式的完整 artifacts，并额外生成：

| 输出文件 | 格式 | 说明 |
|---|---|---|
| `outputs/full_demo/demo_report.md` | Markdown | 汇总 conversation、运行状态、消息流、工具轮次、LLM 次数、memory 数量、工具数量、最终回答和文件清单。 |

典型成功演示的 `messages.json` 角色顺序为：

```text
system → user → assistant(tool_calls) → tool → assistant(final)
```

## 9. CLI 退出码

| 退出码 | 含义 |
|---|---|
| 0 | 成功，或业务错误已被捕获并写入结构化产物。 |
| 1 | 配置、输入文件、解析、模块加载、模型依赖或输出目录等致命错误。 |
| 2 | argparse 参数使用错误。 |

## 10.附录  `outputs/` 文件说明

本节描述输出目录中的文件。

### 10.1 `outputs/B1_fixture/`

| 输出文件 | 直接生成者 | 含义 | 格式 |
|---|---|---|---|
| `messages.json` | `b1_agent_runtime.py` fixture 模式 | 使用预设模块响应构造的完整 Agent 消息序列。 | JSON 数组 |
| `trace.json` | `b1_agent_runtime.py` fixture 模式 | B1 个人演示的轮次、分支、AIMessage 和 ToolMessage 轨迹。 | JSON 对象 |
| `final_answer.md` | `b1_agent_runtime.py` fixture 模式 | B1 从预设最终 AIMessage 中得到的回答。 | Markdown |

### 10.2 `outputs/B2_skills/`

| 输出文件 | 生成代码 | 含义 | 格式 |
|---|---|---|---|
| `calculator_result.json` | `b2_run_skill.py` | calculator 最近一次 SkillResult。 | JSON 对象 |
| `file_reader_result.json` | `b2_run_skill.py` | file_reader 最近一次 SkillResult。 | JSON 对象 |
| `local_file_search_result.json` | `b2_run_skill.py` | local_file_search 最近一次 SkillResult。 | JSON 对象 |
| `table_analyzer_result.json` | `b2_run_skill.py` | table_analyzer 最近一次 SkillResult。 | JSON 对象 |
| `format_converter_result.json` | `b2_run_skill.py` | format_converter 最近一次 SkillResult，包含 `formatted_text` 和 `generated_file_path`。 | JSON 对象 |
| `skill_run_log.jsonl` | `b2_run_skill.py` | 五类 Skill 的累计运行日志。 | JSONL |

### 10.3 `outputs/B3_tools/`

| 输出文件 | 生成代码 | 含义 | 格式 |
|---|---|---|---|
| `tools_schema.json` | `b3_tool_layer.py --export_schema` | `basic_tools` 的模型工具说明。 | JSON 数组 |
| `tool_schema_report.json` | `b3_tool_layer.py --export_schema` | 工具 schema 数量与名称摘要。 | JSON 对象 |
| `tool_messages.json` | `b3_tool_layer.py --execute` | 本次 tool calls 对应的 ToolMessage。 | JSON 数组 |
| `tool_call_log.jsonl` | `b3_tool_layer.py --execute` | tool calls 的累计执行明细。 | JSONL |

### 10.4 `outputs/B4_llm/`

| 输出文件 | 生成代码 | 含义 | 格式 |
|---|---|---|---|
| `<case>/raw_model_output.json` | `b4_local_agent_llm.py` | B4 独立运行案例的原始输出和解析状态。 | JSON 对象 |
| `<case>/ai_message.json` | `b4_local_agent_llm.py` | B4 独立运行案例的规范化 AIMessage。 | JSON 对象 |
| `<case>/llm_run_log.jsonl` | `b4_local_agent_llm.py` | B4 独立运行案例的运行日志。 | JSONL |

### 10.5 `outputs/B5_memory/`

| 输出文件 | 生成代码 | 含义 | 格式 |
|---|---|---|---|
| `selected_memory.json` | `b5_memory.py` 查找模式 | 最近一次 memory 选择及截断结果。 | JSON 对象 |
| `saved_memory.json` | `b5_memory.py` 保存模式 | 最近一次 memory 保存结果和目标路径。 | JSON 对象 |
| `memory_log.jsonl` | `b5_memory.py` | memory 查找/保存累计日志。 | JSONL |

### 10.6 `outputs/B1_runtime/`

| 输出文件 | 直接生成者 | 含义 | 格式 |
|---|---|---|---|
| `messages.json` | `b1_agent_runtime.py` | 完整 Agent 消息序列。 | JSON 数组 |
| `trace.json` | `b1_agent_runtime.py` | 完整运行轨迹、轮次、状态和错误。 | JSON 对象 |
| `final_answer.md` | `b1_agent_runtime.py` | 最终回答。 | Markdown |
| `runtime_log.jsonl` | `b1_agent_runtime.py` | B1 累计任务日志。 | JSONL |
| `selected_memory.json` | `b5_memory.load_memory`，由 B1 指定 outdir | 注入消息前选择的 memory。 | JSON 对象 |
| `memory_log.jsonl` | `b5_memory.py`，由 B1 调用 | 本轮 memory 查找及保存日志。 | JSONL |
| `saved_memory.json` | `b5_memory.save_memory`，由 B1 调用 | 本轮对话 memory 保存结果。 | JSON 对象 |
| `tools_schema.json` | `b3_tool_layer.get_tools_schema`，由 B1 调用 | 本轮可用工具 schema。 | JSON 数组 |
| `tool_schema_report.json` | `b3_tool_layer.py`，由 B1 调用 | 本轮工具集摘要。 | JSON 对象 |
| `tool_messages.json` | B3 生成、B1 汇总 | 本轮所有 ToolMessage。 | JSON 数组 |
| `tool_call_log.jsonl` | `b3_tool_layer.execute_tool_calls`，由 B1 调用 | 本轮工具执行明细。 | JSONL |
| `llm_calls/llm_call_001_raw_model_output.json` | `b4_local_agent_llm.generate_ai_message` | 第一次 LLM 决策的原始输出。 | JSON 对象 |
| `llm_calls/llm_call_001_ai_message.json` | `b4_local_agent_llm.generate_ai_message` | 第一次标准 AIMessage，包含 tool call。 | JSON 对象 |
| `llm_calls/llm_call_002_raw_model_output.json` | `b4_local_agent_llm.generate_ai_message` | 第二次 LLM 决策的原始输出。 | JSON 对象 |
| `llm_calls/llm_call_002_ai_message.json` | `b4_local_agent_llm.generate_ai_message` | 第二次标准 AIMessage，包含 final content。 | JSON 对象 |
| `llm_calls/llm_run_log.jsonl` | `b4_local_agent_llm.py`，由 B1 调用 | 本轮两次 LLM 调用日志。 | JSONL |

### 10.7 `outputs/full_demo/`

| 输出文件 | 直接生成者 | 含义 | 格式 |
|---|---|---|---|
| `messages.json` | `b1_agent_runtime.py`，由 `run_full_demo.py` 调用 | 完整演示消息序列。 | JSON 数组 |
| `trace.json` | `b1_agent_runtime.py` | 完整演示运行轨迹。 | JSON 对象 |
| `final_answer.md` | `b1_agent_runtime.py` | 完整演示最终回答。 | Markdown |
| `runtime_log.jsonl` | `b1_agent_runtime.py` | 完整演示运行日志。 | JSONL |
| `selected_memory.json` | `b5_memory.load_memory` | 完整演示选择的 memory。 | JSON 对象 |
| `memory_log.jsonl` | `b5_memory.py` | 完整演示的 memory 查找/保存日志。 | JSONL |
| `saved_memory.json` | `b5_memory.save_memory` | 完整演示保存的对话 memory 信息。 | JSON 对象 |
| `tools_schema.json` | `b3_tool_layer.get_tools_schema` | 完整演示使用的工具 schema。 | JSON 数组 |
| `tool_schema_report.json` | `b3_tool_layer.py` | 完整演示工具集摘要。 | JSON 对象 |
| `tool_messages.json` | B3 生成、B1 汇总 | 完整演示产生的 ToolMessage。 | JSON 数组 |
| `tool_call_log.jsonl` | `b3_tool_layer.execute_tool_calls` | 完整演示工具执行明细。 | JSONL |
| `llm_calls/llm_call_001_raw_model_output.json` | `b4_local_agent_llm.py` | 第一次 LLM 原始输出。 | JSON 对象 |
| `llm_calls/llm_call_001_ai_message.json` | `b4_local_agent_llm.py` | 第一次 AIMessage，包含 tool call。 | JSON 对象 |
| `llm_calls/llm_call_002_raw_model_output.json` | `b4_local_agent_llm.py` | 第二次 LLM 原始输出。 | JSON 对象 |
| `llm_calls/llm_call_002_ai_message.json` | `b4_local_agent_llm.py` | 第二次 AIMessage，包含最终回答。 | JSON 对象 |
| `llm_calls/llm_run_log.jsonl` | `b4_local_agent_llm.py` | 完整演示 LLM 调用日志。 | JSONL |
| `demo_report.md` | `run_full_demo.py` | 对完整演示状态、数据流、最终回答和文件清单的汇总。 | Markdown |
