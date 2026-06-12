# Language Models

Language Models 提供 Heta 与语言模型服务通信的统一入口。上层知识库构建流程只依赖 `LanguageModel`、`ModelRequest` 和 `ModelResult`，不需要直接处理不同模型服务的请求差异。

当前实现使用 LiteLLM 作为底层适配层。模型名、鉴权、重试、超时、流式输出、多模态输入和 provider 专有参数由 `LanguageModel` 统一接收，再交给 LiteLLM 执行真实请求。

## 快速开始

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

调用 DashScope OpenAI-compatible endpoint 时，可以设置 `api_base` 并使用 LiteLLM 的 OpenAI 路由前缀：

```python
llm = LanguageModel(
    model_name="openai/qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="...",
    provider_options={"enable_thinking": False},
)
```

## 核心对象

| 对象 | 说明 |
| --- | --- |
| `LanguageModel` | 长生命周期模型客户端，负责请求执行、并发限制和 LiteLLM 调用。 |
| `LanguageModelProtocol` | 语言模型能力协议，用于 Recipe、构建步骤和自定义模型的类型约束。 |
| `ModelRequest` | 一次模型请求，包含 prompt、系统提示词、调用参数和追踪信息。 |
| `TextPart` / `ImagePart` | 多模态请求内容片段，用于图文输入。 |
| `ModelOptions` | 单次请求参数，例如温度、输出长度、停止序列和结构化输出格式。 |
| `ModelResult` | 非流式调用结果，包含文本、结构化解析结果、token usage 和原始响应。 |
| `ModelChunk` | 流式调用结果，包含当前文本增量、结束原因和原始 chunk。 |

`LanguageModelProtocol` 是结构化协议，不要求用户继承某个父类。自定义语言模型只要实现 `invoke`、`invoke_many` 和 `stream`，就可以被后续 Recipe 或构建步骤接收。

## LanguageModel

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
| `model_name` | 传给 LiteLLM 的模型名。不同服务的命名规则遵循 LiteLLM。 |
| `api_key` | 模型服务 API key。也可以通过服务方支持的环境变量提供。 |
| `api_base` | 自定义 API endpoint，常用于 OpenAI-compatible 服务。 |
| `request_timeout` | 单次请求超时时间，单位为秒。 |
| `max_retries` | 底层请求失败后的重试次数。 |
| `max_concurrent_requests` | 当前模型实例允许的最大并发请求数。 |
| `default_temperature` | 请求未显式设置温度时使用的默认值。 |
| `drop_unsupported_params` | 让 LiteLLM 丢弃当前模型不支持的参数。 |
| `provider_options` | 长生命周期透传参数，适合放服务方固定选项。 |

`LanguageModel` 是异步优先接口，同一个实例可以被 pipeline 复用。并发请求会被 `max_concurrent_requests` 限制，适合批量处理 chunks、文档、图文理解或抽取任务。

## 调用方法

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

## 请求格式

```python
from heta_framework.common.models import ImagePart, ModelOptions, ModelRequest, TextPart

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

图文输入使用 `content`。`prompt` 和 `content` 二选一；纯文本任务优先使用 `prompt`。

```python
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

`path` 和 `data` 会被转换成 `data:image/...;base64,...` 后发送给模型。FastAPI 上传文件时通常使用 `data`：

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

## 返回结果

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

## 错误处理

模型层不会把失败请求转换为空字符串。

| 错误 | 含义 |
| --- | --- |
| `ModelError` | 模型层基础错误。 |
| `ModelRequestError` | 请求模型服务失败，或请求参数不适用于当前调用方式。 |
| `ModelResponseError` | 响应格式错误，或结构化输出解析失败。 |

错误对象会保留 `trace_context`，方便定位失败发生在哪个任务阶段、文档或 chunk。

## 支持范围

Heta 的 Models 层负责：

- 统一模型调用入口
- 异步请求和批量并发
- 文本和图文输入
- 非流式与流式输出
- 结构化 JSON 解析
- token usage、原始响应和追踪上下文保留
- 向 LiteLLM 透传模型服务专有参数

模型服务覆盖范围由 LiteLLM 决定。新模型接入时，优先确认 LiteLLM 是否已支持对应模型名和 provider 参数。
