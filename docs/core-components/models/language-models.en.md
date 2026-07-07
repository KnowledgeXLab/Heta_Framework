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
| `ToolCallingLanguageModel` | Explicit opt-in client for native tool calling. It keeps normal model behavior and adds `invoke_with_tools`. |
| `LanguageModelProtocol` | Capability protocol used by recipes, steps, query engines, and custom models. |
| `ToolCallingLanguageModelProtocol` | Enhanced capability protocol for components such as agentic query engines that require native tool calling. |
| `ModelRequest` | One model call, including prompt, system prompt, options, and trace context. |
| `ToolCallingModelRequest` | One tool-calling model call with chat messages, tools, tool choice, options, and trace context. |
| `TextPart` / `ImagePart` | Multimodal request parts for text and image inputs. |
| `ModelOptions` | Per-request options such as temperature, output length, stop sequences, and response format. |
| `ModelResult` | Non-streaming result with text, parsed output, token usage, and raw response. |
| `ToolDefinition` / `ToolCall` / `ToolMessage` | Tool schema, model-requested tool call, and tool-calling conversation message. |
| `ToolCallingModelResult` | Tool-calling result with assistant message, tool calls, token usage, and raw response. |
| `ModelChunk` | Streaming result with text delta, finish reason, and raw chunk. |

`LanguageModelProtocol` is structural. Custom language models do not need to inherit a base class; implementing `invoke`, `invoke_many`, and `stream` is enough.

`ToolCallingLanguageModelProtocol` is also structural, but it extends `LanguageModelProtocol`: a model must provide the normal language-model methods and `invoke_with_tools`. Plain `LanguageModel` does not satisfy this protocol. Use `ToolCallingLanguageModel` or a custom equivalent when native tool calling is required.

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

`ToolCallingLanguageModel` accepts the same base configuration and adds `validate_function_calling_support`:

```python
from heta_framework.common.models import ToolCallingLanguageModel

tool_llm = ToolCallingLanguageModel(
    model_name="openai/gpt-4o-mini",
    api_key="...",
    validate_function_calling_support=True,
)
```

| Parameter | Meaning |
| --- | --- |
| `validate_function_calling_support` | Whether to call LiteLLM `supports_function_calling(model=...)` before sending requests with tools. Enabled by default. |

LiteLLM recommends `supports_function_calling(model=...)` for checking whether a model supports function calling. `ToolCallingLanguageModel` follows that check by default. For custom providers or models not yet known to LiteLLM, set `validate_function_calling_support=False` and let the caller own compatibility.

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

`ToolCallingLanguageModel` adds:

```python
result = await tool_llm.invoke_with_tools(tool_request)
```

| Method | Meaning |
| --- | --- |
| `invoke_with_tools` | Execute one OpenAI-compatible native tool-calling request and return `ToolCallingModelResult`. |

`invoke_with_tools` is non-streaming. Agentic query engines should execute returned `ToolCall` objects, then append tool outputs as `role="tool"` messages in the next request.

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

## Tool Calling

Tool calling uses the separate `ToolCallingLanguageModel` class so that plain language models are not automatically marked as tool-capable. It reuses `LanguageModel` configuration, concurrency control, and normal `invoke`/`stream` behavior, but only this explicit class satisfies `ToolCallingLanguageModelProtocol`.

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

When the model requests tool calls, execute the tools and append outputs to the next turn:

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

| Object | Meaning |
| --- | --- |
| `ToolDefinition` | Tool schema exposed to the model. Heta converts it into LiteLLM/OpenAI-compatible `{"type": "function", "function": ...}` format. |
| `ToolMessage` | Tool-calling conversation message. `role="tool"` requires `tool_call_id` and `content`. |
| `ToolCall` | Tool call requested by the model, with `id`, tool name, and JSON arguments. |
| `ToolCallingModelRequest` | One turn with chat messages, tools, and tool choice. |
| `ToolCallingModelResult` | Assistant message, finish reason, token usage, and raw response. |

`tool_choice` supports:

| Value | Meaning |
| --- | --- |
| `"auto"` | Let the model decide whether to answer or call a tool. |
| `"none"` | Prevent tool calls. |
| `"required"` | Require a tool call. |
| Tool name string | Force a specific tool. Heta converts it into LiteLLM's function tool choice object. |

### LiteLLM Compatibility

`ToolCallingLanguageModel` uses LiteLLM Chat Completion parameters: `messages`, `tools`, and `tool_choice`. LiteLLM translates these OpenAI-compatible parameters across providers. See LiteLLM docs:

- [Function Calling](https://docs.litellm.ai/docs/completion/function_call)
- [Input Params](https://docs.litellm.ai/docs/completion/input)

By default, if the request includes tools, `ToolCallingLanguageModel` first calls `litellm.supports_function_calling(model=...)`. If LiteLLM reports that the model does not support function calling, Heta raises `ModelRequestError` instead of letting an agentic query fail implicitly.

For private OpenAI-compatible endpoints where you know the model supports tools but LiteLLM cannot identify it yet, disable the check:

```python
tool_llm = ToolCallingLanguageModel(
    model_name="openai/custom-tool-model",
    api_base="https://example.com/v1",
    api_key="...",
    validate_function_calling_support=False,
)
```

Disabling the check only skips LiteLLM capability detection. The provider still must support `tools` and `tool_choice`.

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

Tool calling uses the same model error family:

- `ModelRequestError`: provider request failed, the model does not support function calling, or the support check failed.
- `ModelResponseError`: assistant message is missing, tool call arguments are not a JSON object, or the tool-calling response cannot be parsed.

## Scope

Models handle model communication: async calls, concurrency, text and image inputs, streaming, structured JSON parsing, native tool-calling requests and response parsing, token usage, raw responses, and trace context.

Models do not parse documents, split chunks, design prompts for business tasks, write vectors, build graphs, or manage the `KnowledgeBase` lifecycle.
