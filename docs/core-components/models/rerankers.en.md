# Rerankers

Rerankers are Heta's interface for reranking retrieved candidates. They use a user query to re-score candidate documents, chunks, or graph evidence and are commonly used to improve result order.

The current implementation uses LiteLLM's rerank endpoint. Query logic depends on `RerankModel`, `RerankRequest`, and `RerankResult`, not provider-specific request formats.

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

`model_name` follows LiteLLM rerank naming. LiteLLM supports providers such as Cohere, Together AI, Azure AI, DeepInfra, Nvidia NIM, Infinity, Fireworks AI, Voyage AI, and watsonx.ai.

## Core Objects

| Object | Meaning |
| --- | --- |
| `RerankModel` | Long-lived rerank client that executes requests, limits concurrency, and calls LiteLLM. |
| `RerankModelProtocol` | Capability protocol used by recipes, query engines, and custom models. |
| `RerankRequest` | One rerank request with query, candidate documents, options, and trace context. |
| `RerankOptions` | Per-request options such as `top_n`, `return_documents`, and provider options. |
| `RerankResult` | Ordered rerank output with `RerankItem` objects, raw response, and trace context. |
| `RerankItem` | A single candidate result with original index, relevance score, and optional document text. |

Custom rerank models only need to implement `rerank` and `rerank_many`.

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

| Parameter | Meaning |
| --- | --- |
| `model_name` | Rerank model name passed to LiteLLM. |
| `api_key` | Provider API key. |
| `api_base` | Custom endpoint for private services compatible with LiteLLM rerank. |
| `request_timeout` | Timeout per request, in seconds. |
| `max_retries` | Retry count for failed provider calls. |
| `max_concurrent_requests` | Maximum concurrent requests for this model instance. |
| `top_n` | Default number of candidates to return. |
| `drop_unsupported_params` | Let LiteLLM drop unsupported parameters. |
| `provider_options` | Long-lived provider-specific options. |

## Calling The Model

```python
result = await reranker.rerank(request)
results = await reranker.rerank_many([request_1, request_2])
```

`rerank_many` preserves input order.

## Request Format

```python
from heta_framework.common.models import RerankOptions, RerankRequest

request = RerankRequest(
    query="key evidence about marine biodiversity",
    documents=[
        "Marine biodiversity keeps ecosystems stable.",
        "This paragraph discusses database connection pools.",
    ],
    options=RerankOptions(
        top_n=1,
        return_documents=True,
        provider_options={"user": "kb-search-job-001"},
    ),
    trace_context={"stage": "rerank", "query_id": "q-001"},
)
```

`RerankItem.index` points back to the original `documents` list. Query engines should use that index to reorder the original `QueryResult` objects instead of treating the reranker as a content source.

## Errors

The rerank layer does not convert failed requests into empty rankings.

| Error | Meaning |
| --- | --- |
| `RerankError` | Base rerank-layer error. |
| `RerankRequestError` | Provider request failed, or request parameters are invalid. |
| `RerankResponseError` | Response format is invalid, or a returned candidate index is out of range. |

Errors preserve `trace_context`.

## Scope

Rerankers only score candidate text against a query. They do not recall candidates, fuse vector and keyword search, read databases, generate answers, or manage the `KnowledgeBase` lifecycle.
