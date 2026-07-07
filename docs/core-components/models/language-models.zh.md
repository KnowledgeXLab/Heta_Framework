# Language Models

Language Models 是 Heta 调用 LLM 的统一入口。构建步骤和 query engines 只依赖 `LanguageModel`、`ModelRequest` 和 `ModelResult`，不需要直接处理不同 provider 的请求格式。

当前实现使用 LiteLLM 作为底层适配层。Heta 负责提供稳定接口，LiteLLM 负责把请求发送到 OpenAI、DashScope OpenAI-compatible endpoint、Anthropic、Gemini 等具体服务。

## Quick Start

```python
from heta_framework.common.models import LanguageModel, ModelRequest

llm = LanguageModel(
    model_name="openai/gpt-4o-mini",
    api_key="...",
    request_timeout=120,
    max_retries=3,
    max_concurrent_requests=20,
)

result = await llm.invoke(
    ModelRequest(
        prompt="从下面文本中抽取实体：...",
        response_schema={"type": "object"},
        trace_context={"stage": "entity_extraction", "chunk_id": "chunk-001"},
    )
)

entities = result.parsed
```

调用 DashScope OpenAI-compatible endpoint 时，设置 `api_base` 并使用 LiteLLM 的 OpenAI 路由前缀：

```python
llm = LanguageModel(
    model_name="openai/qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="...",
    provider_options={"enable_thinking": False},
)
```

`model_name` 遵循 LiteLLM 命名规则。`openai/` 前缀表示按 OpenAI-compatible 协议发送请求，不一定表示模型来自 OpenAI。

## Core Objects

| 对象 | 说明 |
| --- | --- |
| `LanguageModel` | 长生命周期模型客户端，负责请求执行、并发限制和 LiteLLM 调用。 |
| `ToolCallingLanguageModel` | 显式 opt-in 的工具调用模型客户端，继承普通模型能力，并额外实现 `invoke_with_tools`。 |
| `LanguageModelProtocol` | 语言模型能力协议，用于 Recipe、steps、query engines 和自定义模型。 |
| `ToolCallingLanguageModelProtocol` | 工具调用增强协议，用于 agentic query engines 等需要 native tool calling 的组件。 |
| `ModelRequest` | 一次模型请求，包含 prompt、系统提示词、调用参数和 trace 信息。 |
| `ToolCallingModelRequest` | 一次工具调用模型请求，包含 chat messages、tools、tool choice 和 trace 信息。 |
| `TextPart` / `ImagePart` | 多模态请求内容片段，用于图文输入。 |
| `ModelOptions` | 单次请求参数，例如温度、输出长度、停止序列和结构化输出格式。 |
| `ModelResult` | 非流式调用结果，包含文本、结构化解析结果、token usage 和原始响应。 |
| `ToolDefinition` / `ToolCall` / `ToolMessage` | 工具 schema、模型发起的工具调用，以及 tool-calling 对话消息。 |
| `ToolCallingModelResult` | 工具调用响应，包含 assistant message、tool calls、token usage 和原始响应。 |
| `ModelChunk` | 流式调用结果，包含当前文本增量、结束原因和原始 chunk。 |

`LanguageModelProtocol` 是结构化协议，不要求用户继承某个父类。自定义语言模型只要实现 `invoke`、`invoke_many` 和 `stream`，就可以被 recipe 或自定义 step 接收。

`ToolCallingLanguageModelProtocol` 也是结构化协议，但它是 `LanguageModelProtocol` 的增强能力：除普通语言模型方法外，还必须实现 `invoke_with_tools`。普通 `LanguageModel` 不满足该协议；需要 native tool calling 时应显式使用 `ToolCallingLanguageModel` 或实现同等方法的自定义模型。

## Configuration

```python
llm = LanguageModel(
    model_name="openai/gpt-4o-mini",
    api_key="...",
    api_base=None,
    request_timeout=120,
    max_retries=3,
    max_concurrent_requests=10,
    default_temperature=0.1,
    drop_unsupported_params=True,
    provider_options=None,
)
```

| 参数 | 说明 |
| --- | --- |
| `model_name` | 传给 LiteLLM 的模型名。 |
| `api_key` | 模型服务 API key，也可以通过服务方支持的环境变量提供。 |
| `api_base` | 自定义 API endpoint，常用于 OpenAI-compatible 服务。 |
| `request_timeout` | 单次请求超时时间，单位为秒。 |
| `max_retries` | 底层请求失败后的重试次数。 |
| `max_concurrent_requests` | 当前模型实例允许的最大并发请求数。 |
| `default_temperature` | 请求未显式设置温度时使用的默认值。 |
| `drop_unsupported_params` | 让 LiteLLM 丢弃当前模型不支持的参数。 |
| `provider_options` | 长生命周期透传参数，适合放服务方固定选项。 |

同一个 `LanguageModel` 实例可以在 recipe 中复用。批量处理 chunks、图文描述、实体抽取或关系抽取时，并发请求会被 `max_concurrent_requests` 控制。

`ToolCallingLanguageModel` 接受同样的基础配置，并额外提供 `validate_function_calling_support`：

```python
from heta_framework.common.models import ToolCallingLanguageModel

tool_llm = ToolCallingLanguageModel(
    model_name="openai/gpt-4o-mini",
    api_key="...",
    validate_function_calling_support=True,
)
```

| 参数 | 说明 |
| --- | --- |
| `validate_function_calling_support` | 是否在发送带 tools 的请求前调用 LiteLLM `supports_function_calling(model=...)` 做能力检查。默认开启。 |

LiteLLM 文档建议用 `supports_function_calling(model=...)` 判断模型是否支持 function calling；`ToolCallingLanguageModel` 默认遵循这个检查。自定义 provider 或 LiteLLM 尚未登记能力的模型，可以设置 `validate_function_calling_support=False`，由调用方自行保证兼容性。

## Calling The Model

```python
result = await llm.invoke(request)
results = await llm.invoke_many([request_1, request_2])

async for chunk in llm.stream(request):
    print(chunk.text_delta, end="")
```

| 方法 | 说明 |
| --- | --- |
| `invoke` | 执行一次非流式请求，返回完整 `ModelResult`。 |
| `invoke_many` | 并发执行多次请求，返回顺序与输入顺序一致。 |
| `stream` | 执行一次流式请求，返回 `AsyncIterator[ModelChunk]`。 |

`stream` 当前不支持 `response_schema`。需要结构化输出时使用 `invoke`。

`ToolCallingLanguageModel` 额外提供：

```python
result = await tool_llm.invoke_with_tools(tool_request)
```

| 方法 | 说明 |
| --- | --- |
| `invoke_with_tools` | 执行一次 OpenAI-compatible native tool-calling 请求，返回 `ToolCallingModelResult`。 |

`invoke_with_tools` 是非流式接口。agentic query engine 应在收到 `ToolCall` 后执行对应工具，再把工具结果作为 `role="tool"` 的 `ToolMessage` 追加到下一轮请求。

## Request Format

纯文本任务通常使用 `prompt`：

```python
from heta_framework.common.models import ModelOptions, ModelRequest

request = ModelRequest(
    prompt="从文本中抽取实体和关系。",
    system_prompt="你是一个知识图谱抽取器。",
    options=ModelOptions(
        temperature=0.1,
        max_output_tokens=4096,
        top_p=0.9,
        stop_sequences=None,
        response_format={"type": "json_object"},
        provider_options={"enable_thinking": False},
    ),
    response_schema={"type": "object"},
    trace_context={"stage": "graph_extraction", "chunk_id": "chunk-001"},
)
```

图文输入使用 `content`。`prompt` 和 `content` 二选一。

```python
from heta_framework.common.models import ImagePart, ModelRequest, TextPart

request = ModelRequest(
    content=[
        TextPart(text="描述这张图，并提取图中的关键信息。"),
        ImagePart.from_uri("https://example.com/image.png", detail="high"),
    ],
    trace_context={"stage": "image_description", "document_id": "doc-001"},
)
```

`ImagePart` 支持三种图片来源：

```python
ImagePart.from_uri("https://example.com/image.png")
ImagePart.from_file("./images/page-001.png")
ImagePart.from_bytes(image_bytes, mime_type="image/png")
```

`path` 和 `data` 会被转换成 `data:image/...;base64,...` 后发送给模型。FastAPI 上传文件时通常使用 `from_bytes`：

```python
@app.post("/describe-image")
async def describe_image(file: UploadFile):
    image_bytes = await file.read()
    result = await llm.invoke(
        ModelRequest(
            content=[
                TextPart(text="描述这张图片。"),
                ImagePart.from_bytes(
                    image_bytes,
                    mime_type=file.content_type or "image/png",
                ),
            ]
        )
    )
    return {"description": result.text}
```

| 字段 | 说明 |
| --- | --- |
| `prompt` | 本次纯文本任务输入。 |
| `content` | 本次多模态任务输入，可包含 `TextPart` 和 `ImagePart`。 |
| `system_prompt` | 系统提示词，可选。 |
| `options` | 本次请求的模型参数，可选。 |
| `response_schema` | 结构化输出约束，可选。设置后解析结果写入 `ModelResult.parsed`。 |
| `trace_context` | 调用追踪信息，不会发送给模型。 |

`ModelOptions.provider_options` 会覆盖 `LanguageModel.provider_options` 中的同名字段，适合在单次请求中调整服务方专有参数。

## Tool Calling

Tool calling 使用单独的 `ToolCallingLanguageModel`，避免把普通语言模型都标记为 tool-capable。它复用 `LanguageModel` 的配置、并发限制、普通 `invoke`/`stream` 能力，但只有这个显式类会满足 `ToolCallingLanguageModelProtocol`。

```python
from heta_framework.common.models import (
    ToolCallingLanguageModel,
    ToolCallingModelRequest,
    ToolDefinition,
    ToolMessage,
)

tool_llm = ToolCallingLanguageModel(
    model_name="openai/gpt-4o-mini",
    api_key="...",
)

request = ToolCallingModelRequest(
    messages=(
        ToolMessage(role="system", content="Use tools when they help."),
        ToolMessage(role="user", content="Find pages about Heta query engines."),
    ),
    tools=(
        ToolDefinition(
            name="search_wiki",
            description="Search wiki pages by query.",
            parameters_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
    ),
    tool_choice="auto",
    trace_context={"stage": "agentic_query"},
)

result = await tool_llm.invoke_with_tools(request)
tool_calls = result.message.tool_calls
```

当模型请求工具调用后，执行工具，并把工具输出写回下一轮对话：

```python
messages = [*request.messages, result.message]

for call in result.message.tool_calls:
    tool_output = run_tool(call.name, call.arguments)
    messages.append(
        ToolMessage(
            role="tool",
            content=tool_output,
            tool_call_id=call.id,
        )
    )

final = await tool_llm.invoke_with_tools(
    ToolCallingModelRequest(
        messages=tuple(messages),
        tools=request.tools,
        tool_choice="auto",
    )
)
```

### Tool Objects

| 对象 | 说明 |
| --- | --- |
| `ToolDefinition` | 暴露给模型的工具 schema。Heta 会转换为 LiteLLM/OpenAI-compatible 的 `{"type": "function", "function": ...}` 格式。 |
| `ToolMessage` | tool-calling 对话消息。`role="tool"` 时必须提供 `tool_call_id` 和 `content`。 |
| `ToolCall` | 模型返回的工具调用请求，包含 `id`、工具名和 JSON 参数。 |
| `ToolCallingModelRequest` | 一轮带工具列表和 tool choice 的模型请求。 |
| `ToolCallingModelResult` | 模型返回的 assistant message、结束原因、token usage 和原始响应。 |

`tool_choice` 支持：

| 值 | 说明 |
| --- | --- |
| `"auto"` | 让模型自行决定回答或调用工具。 |
| `"none"` | 禁止模型调用工具。 |
| `"required"` | 要求模型调用某个可用工具。 |
| 工具名字符串 | 强制调用指定工具，Heta 会转换为 LiteLLM 支持的 function tool choice 对象。 |

### LiteLLM Compatibility

`ToolCallingLanguageModel` 使用 LiteLLM 的 Chat Completion 参数：`messages`、`tools`、`tool_choice`。LiteLLM 会把这些 OpenAI-compatible 参数翻译到具体 provider。参考 LiteLLM 文档：

- [Function Calling](https://docs.litellm.ai/docs/completion/function_call)
- [Input Params](https://docs.litellm.ai/docs/completion/input)

默认情况下，如果请求包含 tools，`ToolCallingLanguageModel` 会先调用 `litellm.supports_function_calling(model=...)`。如果 LiteLLM 判断模型不支持 function calling，会抛出 `ModelRequestError`，避免 agentic query 在不支持工具调用的模型上隐式失败。

如果你使用私有 OpenAI-compatible endpoint，且确认该模型支持 tools 但 LiteLLM 还无法识别，可以关闭检查：

```python
tool_llm = ToolCallingLanguageModel(
    model_name="openai/custom-tool-model",
    api_base="https://example.com/v1",
    api_key="...",
    validate_function_calling_support=False,
)
```

关闭检查只跳过 LiteLLM 的能力判定，不会改变请求格式；provider 仍需实际支持 `tools` 和 `tool_choice`。

## Result

```python
result.text
result.parsed
result.model_name
result.token_usage
result.finish_reason
result.trace_context
result.raw_response
```

| 字段 | 说明 |
| --- | --- |
| `text` | 模型原始文本输出。 |
| `parsed` | 按 `response_schema` 解析后的结构化结果。 |
| `model_name` | 当前 `LanguageModel` 配置的模型名。 |
| `token_usage` | token 消耗信息。 |
| `finish_reason` | 模型服务返回的结束原因。 |
| `trace_context` | 请求携带的追踪信息。 |
| `raw_response` | LiteLLM 返回的原始响应字典。 |

如果设置了 `response_schema`，业务代码通常读取 `parsed`；否则读取 `text`。

## Errors

模型层不会把失败请求转换为空字符串。

| 错误 | 含义 |
| --- | --- |
| `ModelError` | 模型层基础错误。 |
| `ModelRequestError` | 请求模型服务失败，或请求参数不适用于当前调用方式。 |
| `ModelResponseError` | 响应格式错误，或结构化输出解析失败。 |

错误对象会保留 `trace_context`，方便定位失败发生在哪个任务阶段、文档或 chunk。

Tool calling 也使用同一组模型错误：

- `ModelRequestError`：LiteLLM 请求失败、当前模型不支持 function calling，或 support check 无法完成。
- `ModelResponseError`：响应缺少 assistant message、tool call 参数不是 JSON object，或 tool-calling 响应结构无法解析。

## Scope

Models 层负责：

- 统一模型调用入口。
- 异步请求和批量并发。
- 文本和图文输入。
- 非流式与流式输出。
- 结构化 JSON 解析。
- token usage、原始响应和追踪上下文保留。
- 向 LiteLLM 透传模型服务专有参数。
- native tool-calling 请求、响应解析和能力检查。

Models 不负责文档解析、chunk 切分、prompt 业务内容、向量入库、图谱构建或 `KnowledgeBase` 生命周期管理。

模型服务覆盖范围由 LiteLLM 决定。接入新模型时，优先确认 LiteLLM 是否支持对应 `model_name` 和 provider 参数。
