# Embeddings

Embeddings are Heta's vector model interface. They turn text into vectors for knowledge-base construction, vector search, entity alignment, relation alignment, and similarity calculations.

The current implementation uses LiteLLM underneath. Heta code depends on `EmbeddingModel`, `EmbeddingRequest`, and `EmbeddingResult`, not provider-specific request formats.

## Quick Start

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
        texts=["first chunk", "second chunk"],
        trace_context={"stage": "chunk_embedding", "document_id": "doc-001"},
    )
)

vectors = result.vectors
```

For OpenAI-compatible endpoints, set `api_base`:

```python
embedding = EmbeddingModel(
    model_name="openai/text-embedding-v4",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="...",
)
```

`model_name` follows LiteLLM naming. The `openai/` prefix means OpenAI-compatible protocol, not necessarily OpenAI hosting.

## Core Objects

| Object | Meaning |
| --- | --- |
| `EmbeddingModel` | Long-lived embedding client that executes requests, limits concurrency, and calls LiteLLM. |
| `EmbeddingModelProtocol` | Capability protocol used by recipes, steps, query engines, and custom models. |
| `EmbeddingRequest` | One embedding request with texts, options, and trace context. |
| `EmbeddingOptions` | Per-request options such as dimensions, encoding format, and provider options. |
| `EmbeddingResult` | Embedding result with vectors, usage, raw response, and trace context. |

`EmbeddingModelProtocol` is structural. Custom embedding models only need to implement `embed` and `embed_many`.

## Configuration

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

| Parameter | Meaning |
| --- | --- |
| `model_name` | Embedding model name passed to LiteLLM. |
| `api_key` | Provider API key. |
| `api_base` | Custom endpoint for OpenAI-compatible services. |
| `request_timeout` | Timeout per request, in seconds. |
| `max_retries` | Retry count for failed provider calls. |
| `max_concurrent_requests` | Maximum concurrent requests for this model instance. |
| `dimensions` | Output dimension, only used by models that support dimension trimming. |
| `encoding_format` | Vector encoding format, such as `float`. |
| `drop_unsupported_params` | Let LiteLLM drop unsupported parameters. |
| `provider_options` | Long-lived provider-specific options. |

## Calling The Model

```python
result = await embedding.embed(request)
results = await embedding.embed_many([request_1, request_2])
```

`embed` accepts a list of texts, not just one text. During KB build, multiple chunks can be batched into one request.

## Request Format

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

`EmbeddingOptions.provider_options` overrides same-name options on the model instance for that request.

## Result

```python
result.vectors
result.model_name
result.usage
result.trace_context
result.raw_response
```

Vectors are returned in the same order as `EmbeddingRequest.texts`.

## Errors

The embedding layer does not convert failed requests into empty vectors.

| Error | Meaning |
| --- | --- |
| `EmbeddingError` | Base embedding-layer error. |
| `EmbeddingRequestError` | Provider request failed, or request parameters are invalid. |
| `EmbeddingResponseError` | Response format is invalid, or vector count does not match input text count. |

Errors preserve `trace_context`.

## Scope

Embeddings only handle text-to-vector conversion: async calls, batching, concurrency, ordering, usage, raw responses, trace context, and provider options.

They do not parse documents, split chunks, write vectors to a database, search for similarity, rerank results, or manage the `KnowledgeBase` lifecycle.
