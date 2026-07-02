# Rerankers

Rerankers 是 Heta 的重排序模型入口。它根据用户 query 重新评估候选文档、chunk 或图谱证据的相关性，常用于提高检索结果排序质量。

当前实现使用 LiteLLM 的 rerank endpoint。上层检索逻辑只依赖 `RerankModel`、`RerankRequest` 和 `RerankResult`，不需要直接处理 Cohere、Voyage、Jina、Together 或私有 rerank 服务的请求差异。

## Quick Start

```python
from heta_framework.common.models import RerankModel, RerankRequest

reranker = RerankModel(
    model_name="cohere/rerank-english-v3.0",
    api_key="...",
    top_n=5,
    max_concurrent_requests=10,
)

result = await reranker.rerank(
    RerankRequest(
        query="marine biodiversity in oceans",
        documents=[
            "Marine biodiversity is essential to ocean ecosystems.",
            "This page describes aircraft maintenance procedures.",
            "Coral reefs support a wide range of marine species.",
        ],
        trace_context={"stage": "search_rerank", "kb": "papers"},
    )
)

ranked_indices = [item.index for item in result.rankings]
```

`model_name` 遵循 LiteLLM 的 rerank 命名规则。LiteLLM 当前支持 Cohere、Together AI、Azure AI、DeepInfra、Nvidia NIM、Infinity、Fireworks AI、Voyage AI、watsonx.ai 等 rerank providers。

## Core Objects

| 对象 | 说明 |
| --- | --- |
| `RerankModel` | 长生命周期 rerank 模型客户端，负责请求执行、并发限制和 LiteLLM 调用。 |
| `RerankModelProtocol` | Rerank 能力协议，用于 Recipe、query engines 和自定义模型。 |
| `RerankRequest` | 一次重排序请求，包含 query、候选文档、调用参数和 trace 信息。 |
| `RerankOptions` | 单次请求参数，例如返回数量、是否返回文档和服务方专有参数。 |
| `RerankResult` | 重排序结果，包含有序的 `RerankItem` 列表、原始响应和 trace 信息。 |
| `RerankItem` | 一个候选文档的排序结果，包含原始索引、相关性分数和可选文档文本。 |

`RerankModelProtocol` 是结构化协议，不要求用户继承某个父类。自定义 rerank 模型只要实现 `rerank` 和 `rerank_many`，就可以被 query engine 接收。

## Configuration

```python
reranker = RerankModel(
    model_name="cohere/rerank-english-v3.0",
    api_key="...",
    api_base=None,
    request_timeout=120,
    max_retries=3,
    max_concurrent_requests=10,
    top_n=None,
    drop_unsupported_params=True,
    provider_options=None,
)
```

| 参数 | 说明 |
| --- | --- |
| `model_name` | 传给 LiteLLM 的 rerank 模型名。 |
| `api_key` | 模型服务 API key，也可以通过服务方支持的环境变量提供。 |
| `api_base` | 自定义 API endpoint，常用于兼容 LiteLLM rerank 协议的私有服务。 |
| `request_timeout` | 单次请求超时时间，单位为秒。 |
| `max_retries` | 底层请求失败后的重试次数。 |
| `max_concurrent_requests` | 当前模型实例允许的最大并发请求数。 |
| `top_n` | 默认返回的候选数量。单次请求可以用 `RerankOptions.top_n` 覆盖。 |
| `drop_unsupported_params` | 让 LiteLLM 丢弃当前模型不支持的参数。 |
| `provider_options` | 长生命周期透传参数，适合放服务方固定选项。 |

## Calling The Model

```python
result = await reranker.rerank(request)
results = await reranker.rerank_many([request_1, request_2])
```

| 方法 | 说明 |
| --- | --- |
| `rerank` | 对一组候选文档执行一次重排序请求，返回 `RerankResult`。 |
| `rerank_many` | 并发执行多次重排序请求，返回顺序与输入顺序一致。 |

## Request Format

```python
from heta_framework.common.models import RerankOptions, RerankRequest

request = RerankRequest(
    query="海洋生物多样性的关键证据",
    documents=[
        "海洋生物多样性维持生态系统稳定。",
        "该段落讨论数据库连接池配置。",
    ],
    options=RerankOptions(
        top_n=1,
        return_documents=True,
        provider_options={"user": "kb-search-job-001"},
    ),
    trace_context={"stage": "rerank", "query_id": "q-001"},
)
```

| 字段 | 说明 |
| --- | --- |
| `query` | 用户查询或检索意图，不能为空。 |
| `documents` | 候选文本列表，不能为空。返回结果中的 `index` 指向这个列表。 |
| `options` | 本次请求的 rerank 参数，可选。 |
| `trace_context` | 调用追踪信息，不会发送给模型。 |

`RerankOptions.provider_options` 会覆盖 `RerankModel.provider_options` 中的同名字段，适合在单次请求中调整服务方专有参数。

## Result

```python
result.rankings
result.model_name
result.trace_context
result.raw_response
```

| 字段 | 说明 |
| --- | --- |
| `rankings` | 有序的 `RerankItem` 列表，通常从最相关到最不相关。 |
| `model_name` | 当前 `RerankModel` 配置的模型名。 |
| `trace_context` | 请求携带的追踪信息。 |
| `raw_response` | LiteLLM 返回的原始响应字典。 |

`RerankItem.index` 是原始候选文档在 `RerankRequest.documents` 中的位置。Query engine 应使用这个索引回到原始 `QueryResult`，而不是把 reranker 当成新的内容来源。

## Errors

Rerank 层不会把失败请求转换为空排序。

| 错误 | 含义 |
| --- | --- |
| `RerankError` | Rerank 层基础错误。 |
| `RerankRequestError` | 请求模型服务失败，或请求参数不合法。 |
| `RerankResponseError` | 响应格式错误，或返回的候选索引超出输入列表范围。 |

错误对象会保留 `trace_context`，方便定位失败发生在哪个检索阶段、query 或知识库。

## Scope

Rerankers 层只负责 query 到候选文本的相关性重排：

- 统一 rerank 调用入口。
- 异步请求和批量并发。
- 保留候选文档原始索引。
- 原始响应和追踪上下文保留。
- 向 LiteLLM 透传模型服务专有参数。

Rerankers 不负责召回候选、融合向量检索和关键词检索、读取数据库、生成答案或管理 `KnowledgeBase` 生命周期。这些能力属于 Search、Stores 或更上层的查询编排。
