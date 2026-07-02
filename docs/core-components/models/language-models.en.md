# Language Models

Language Models are Heta's unified LLM interface. Build steps and query engines depend on `LanguageModel`, `ModelRequest`, and `ModelResult` instead of provider-specific request formats.

The current implementation uses LiteLLM underneath. Heta provides the stable interface; LiteLLM routes requests to OpenAI, DashScope OpenAI-compatible endpoints, Anthropic, Gemini, and other providers.

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
        prompt="Extract entities from the following text: ...",
        response_schema={"type": "object"},
        trace_context={"stage": "entity_extraction", "chunk_id": "chunk-001"},
    )
)

entities = result.parsed
```

For a DashScope OpenAI-compatible endpoint, set `api_base` and use LiteLLM's OpenAI routing prefix:

```python
llm = LanguageModel(
    model_name="openai/qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="...",
    provider_options={"enable_thinking": False},
)
```

`model_name` follows LiteLLM naming. The `openai/` prefix means the request is sent through the OpenAI-compatible protocol; it does not mean the model must be hosted by OpenAI.

## Core Objects

| Object | Meaning |
| --- | --- |
| `LanguageModel` | Long-lived model client that executes requests, limits concurrency, and calls LiteLLM. |
| `LanguageModelProtocol` | Capability protocol used by recipes, steps, query engines, and custom models. |
| `ModelRequest` | One model call, including prompt, system prompt, options, and trace context. |
| `TextPart` / `ImagePart` | Multimodal request parts for text and image inputs. |
| `ModelOptions` | Per-request options such as temperature, output length, stop sequences, and response format. |
| `ModelResult` | Non-streaming result with text, parsed output, token usage, and raw response. |
| `ModelChunk` | Streaming result with text delta, finish reason, and raw chunk. |

`LanguageModelProtocol` is structural. Custom language models do not need to inherit a base class; implementing `invoke`, `invoke_many`, and `stream` is enough.

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

| Parameter | Meaning |
| --- | --- |
| `model_name` | Model name passed to LiteLLM. |
| `api_key` | Provider API key. It can also come from provider-supported environment variables. |
| `api_base` | Custom endpoint, usually for OpenAI-compatible services. |
| `request_timeout` | Timeout per request, in seconds. |
| `max_retries` | Retry count for failed provider calls. |
| `max_concurrent_requests` | Maximum concurrent requests for this model instance. |
| `default_temperature` | Default temperature when a request does not specify one. |
| `drop_unsupported_params` | Let LiteLLM drop parameters unsupported by the target model. |
| `provider_options` | Long-lived provider-specific options. |

The same `LanguageModel` instance can be reused in a recipe. Batch chunk processing, image description, entity extraction, and relation extraction all respect `max_concurrent_requests`.

## Calling The Model

```python
result = await llm.invoke(request)
results = await llm.invoke_many([request_1, request_2])

async for chunk in llm.stream(request):
    print(chunk.text_delta, end="")
```

| Method | Meaning |
| --- | --- |
| `invoke` | Execute one non-streaming request and return a complete `ModelResult`. |
| `invoke_many` | Execute multiple requests concurrently and preserve input order. |
| `stream` | Execute one streaming request and return `AsyncIterator[ModelChunk]`. |

`stream` does not support `response_schema`. Use `invoke` for structured output.

## Request Format

Text tasks usually use `prompt`:

```python
from heta_framework.common.models import ModelOptions, ModelRequest

request = ModelRequest(
    prompt="Extract entities and relations from the text.",
    system_prompt="You are a knowledge graph extractor.",
    options=ModelOptions(
        temperature=0.1,
        max_output_tokens=4096,
        top_p=0.9,
        response_format={"type": "json_object"},
        provider_options={"enable_thinking": False},
    ),
    response_schema={"type": "object"},
    trace_context={"stage": "graph_extraction", "chunk_id": "chunk-001"},
)
```

Multimodal tasks use `content`. `prompt` and `content` are mutually exclusive.

```python
from heta_framework.common.models import ImagePart, ModelRequest, TextPart

request = ModelRequest(
    content=[
        TextPart(text="Describe this image and extract key information."),
        ImagePart.from_uri("https://example.com/image.png", detail="high"),
    ],
    trace_context={"stage": "image_description", "document_id": "doc-001"},
)
```

`ImagePart` supports three sources:

```python
ImagePart.from_uri("https://example.com/image.png")
ImagePart.from_file("./images/page-001.png")
ImagePart.from_bytes(image_bytes, mime_type="image/png")
```

`path` and `data` inputs are converted to `data:image/...;base64,...` before being sent to the model.

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

If `response_schema` is set, application code usually reads `parsed`; otherwise it reads `text`.

## Errors

The model layer does not convert failures into empty strings.

| Error | Meaning |
| --- | --- |
| `ModelError` | Base model-layer error. |
| `ModelRequestError` | Provider request failed, or request parameters are invalid for this call. |
| `ModelResponseError` | Response format is invalid, or structured output parsing failed. |

Errors preserve `trace_context` so that failures can be traced back to a stage, document, or chunk.

## Scope

Models handle model communication: async calls, concurrency, text and image inputs, streaming, structured JSON parsing, token usage, raw responses, and trace context.

Models do not parse documents, split chunks, design prompts for business tasks, write vectors, build graphs, or manage the `KnowledgeBase` lifecycle.
