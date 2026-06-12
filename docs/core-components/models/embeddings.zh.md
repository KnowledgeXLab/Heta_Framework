# Embeddings

Embeddings 提供 Heta 与向量模型服务通信的统一入口。它属于 Models 组件的一部分，负责把文本转换为向量，供后续知识库构建、向量检索、实体对齐和相似度计算使用。

当前实现使用 LiteLLM 作为底层适配层。上层代码只依赖 `EmbeddingModel`、`EmbeddingRequest` 和 `EmbeddingResult`，不需要直接处理不同模型服务的请求差异。

## 快速开始

```python
from heta_framework.common.models import EmbeddingModel, EmbeddingRequest

embedding = EmbeddingModel(
    model_name="openai/text-embedding-3-small",
    api_key="...",
    dimensions=1536,
    max_concurrent_requests=10,
)

result = await embedding.embed(
    EmbeddingRequest(
        texts=["第一个 chunk", "第二个 chunk"],
        trace_context={"stage": "chunk_embedding", "document_id": "doc-001"},
    )
)

vectors = result.vectors
```

调用 OpenAI-compatible endpoint 时，可以设置 `api_base`：

```python
embedding = EmbeddingModel(
    model_name="openai/text-embedding-v4",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="...",
)
```

`model_name` 遵循 LiteLLM 的命名规则。对于 OpenAI-compatible endpoint，`openai/` 前缀表示按 OpenAI-compatible embedding 协议发送请求，不表示模型一定来自 OpenAI。

## 核心对象

| 对象 | 说明 |
| --- | --- |
| `EmbeddingModel` | 长生命周期向量模型客户端，负责请求执行、并发限制和 LiteLLM 调用。 |
| `EmbeddingModelProtocol` | 向量模型能力协议，用于 Recipe、构建步骤和自定义模型的类型约束。 |
| `EmbeddingRequest` | 一次向量化请求，包含一组文本、调用参数和追踪信息。 |
| `EmbeddingOptions` | 单次请求参数，例如向量维度、编码格式和服务方专有参数。 |
| `EmbeddingResult` | 向量化结果，包含向量列表、token usage、原始响应和追踪信息。 |

`EmbeddingModelProtocol` 是结构化协议，不要求用户继承某个父类。自定义向量模型只要实现 `embed` 和 `embed_many`，就可以被后续 Recipe 或构建步骤接收。

## EmbeddingModel

```python
embedding = EmbeddingModel(
    model_name="openai/text-embedding-3-small",
    api_key="...",
    api_base=None,
    request_timeout=120,
    max_retries=3,
    max_concurrent_requests=10,
    dimensions=None,
    encoding_format=None,
    drop_unsupported_params=True,
    provider_options=None,
)
```

| 参数 | 说明 |
| --- | --- |
| `model_name` | 传给 LiteLLM 的 embedding 模型名。 |
| `api_key` | 模型服务 API key。也可以通过服务方支持的环境变量提供。 |
| `api_base` | 自定义 API endpoint，常用于 OpenAI-compatible 服务。 |
| `request_timeout` | 单次请求超时时间，单位为秒。 |
| `max_retries` | 底层请求失败后的重试次数。 |
| `max_concurrent_requests` | 当前模型实例允许的最大并发请求数。 |
| `dimensions` | 输出向量维度。只有支持维度裁剪的模型才会使用。 |
| `encoding_format` | 向量编码格式，例如 `float`。 |
| `drop_unsupported_params` | 让 LiteLLM 丢弃当前模型不支持的参数。 |
| `provider_options` | 长生命周期透传参数，适合放服务方固定选项。 |

## 调用方法

```python
result = await embedding.embed(request)
results = await embedding.embed_many([request_1, request_2])
```

| 方法 | 说明 |
| --- | --- |
| `embed` | 对一组文本执行一次向量化请求，返回 `EmbeddingResult`。 |
| `embed_many` | 并发执行多次向量化请求，返回顺序与输入顺序一致。 |

`embed` 接收的是一组文本，而不是单条文本。知识库构建时可以把多个 chunks 放在同一个请求里，减少请求次数。

## 请求格式

```python
from heta_framework.common.models import EmbeddingOptions, EmbeddingRequest

request = EmbeddingRequest(
    texts=["chunk one", "chunk two"],
    options=EmbeddingOptions(
        dimensions=1536,
        encoding_format="float",
        provider_options={"user": "kb-build-job-001"},
    ),
    trace_context={"stage": "chunk_embedding", "document_id": "doc-001"},
)
```

| 字段 | 说明 |
| --- | --- |
| `texts` | 本次要向量化的文本列表，不能为空。 |
| `options` | 本次请求的 embedding 参数，可选。 |
| `trace_context` | 调用追踪信息，不会发送给模型。 |

`EmbeddingOptions.provider_options` 会覆盖 `EmbeddingModel.provider_options` 中的同名字段，适合在单次请求中调整服务方专有参数。

## 返回结果

```python
result.vectors
result.model_name
result.usage
result.trace_context
result.raw_response
```

| 字段 | 说明 |
| --- | --- |
| `vectors` | 向量列表，顺序与 `EmbeddingRequest.texts` 一致。 |
| `model_name` | 当前 `EmbeddingModel` 配置的模型名。 |
| `usage` | token 消耗信息。 |
| `trace_context` | 请求携带的追踪信息。 |
| `raw_response` | LiteLLM 返回的原始响应字典。 |

## 错误处理

Embedding 层不会把失败请求转换为空向量。

| 错误 | 含义 |
| --- | --- |
| `EmbeddingError` | Embedding 层基础错误。 |
| `EmbeddingRequestError` | 请求模型服务失败，或请求参数不合法。 |
| `EmbeddingResponseError` | 响应格式错误，或返回向量数量与输入文本数量不一致。 |

错误对象会保留 `trace_context`，方便定位失败发生在哪个任务阶段、文档或 chunk。

## 能力范围

Embeddings 层只负责文本到向量：

- 统一 embedding 调用入口
- 异步请求和批量并发
- 输入顺序与向量顺序对齐
- token usage、原始响应和追踪上下文保留
- 向 LiteLLM 透传模型服务专有参数

Embeddings 不负责文档解析、chunk 切分、向量入库、相似度搜索、rerank 或知识库生命周期管理。
