import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import (
    EmbeddingModel,
    EmbeddingModelProtocol,
    LanguageModel,
    LanguageModelProtocol,
    ModelOptions,
    ModelRequest,
    ModelRequestError,
    ModelResult,
    RerankModel,
    RerankModelProtocol,
    ToolCall,
    ToolCallingLanguageModel,
    ToolCallingLanguageModelProtocol,
    ToolCallingModelRequest,
    ToolCallingModelResult,
    ToolDefinition,
    ToolMessage,
)


def test_default_models_satisfy_model_protocols():
    language = LanguageModel(model_name="openai/test-model")
    tool_language = ToolCallingLanguageModel(model_name="openai/test-tool-model")
    embedding = EmbeddingModel(model_name="openai/test-embedding")
    reranker = RerankModel(model_name="cohere/test-rerank")

    assert isinstance(language, LanguageModelProtocol)
    assert not isinstance(language, ToolCallingLanguageModelProtocol)
    assert isinstance(tool_language, LanguageModelProtocol)
    assert isinstance(tool_language, ToolCallingLanguageModelProtocol)
    assert isinstance(embedding, EmbeddingModelProtocol)
    assert isinstance(reranker, RerankModelProtocol)


class FakeToolCallingLanguageModel:
    model_name = "test/tool-calling"

    async def invoke(self, request: ModelRequest) -> ModelResult:
        return ModelResult(text="ok", model_name=self.model_name)

    async def invoke_many(self, requests):
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest):
        if False:
            yield

    async def invoke_with_tools(
        self,
        request: ToolCallingModelRequest,
    ) -> ToolCallingModelResult:
        return ToolCallingModelResult(
            message=ToolMessage(role="assistant", content="done"),
            model_name=self.model_name,
        )


def test_tool_calling_language_model_protocol_extends_language_model_protocol():
    model = FakeToolCallingLanguageModel()

    assert isinstance(model, LanguageModelProtocol)
    assert isinstance(model, ToolCallingLanguageModelProtocol)


def test_tool_calling_model_request_validates_messages_and_tools():
    search_tool = ToolDefinition(
        name="search_wiki",
        description="Search wiki pages.",
        parameters_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    request = ToolCallingModelRequest(
        messages=(ToolMessage(role="user", content="What is Heta?"),),
        tools=(search_tool,),
        tool_choice="required",
    )

    assert request.tools == (search_tool,)
    assert request.messages[0].content == "What is Heta?"

    with pytest.raises(ValueError, match="duplicate"):
        ToolCallingModelRequest(
            messages=(ToolMessage(role="user", content="What is Heta?"),),
            tools=(search_tool, search_tool),
        )

    with pytest.raises(ValueError, match="requires at least one tool"):
        ToolCallingModelRequest(
            messages=(ToolMessage(role="user", content="What is Heta?"),),
            tool_choice="required",
        )


def test_tool_messages_validate_tool_call_shape():
    tool_call = ToolCall(id="call_1", name="search_wiki", arguments={"query": "Heta"})
    assistant_message = ToolMessage(role="assistant", tool_calls=(tool_call,))
    tool_message = ToolMessage(role="tool", content="result", tool_call_id="call_1")

    assert assistant_message.tool_calls == (tool_call,)
    assert tool_message.tool_call_id == "call_1"

    with pytest.raises(ValueError, match="tool_call_id"):
        ToolMessage(role="tool", content="result")

    with pytest.raises(ValueError, match="assistant messages require"):
        ToolMessage(role="assistant")


def test_tool_calling_language_model_builds_and_parses_litellm_calls(monkeypatch):
    calls = []

    class FakeLiteLLM:
        def supports_function_calling(self, model):
            return model == "openai/test-tool-model"

        async def acompletion(self, **kwargs):
            calls.append(kwargs)
            return {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "search_wiki",
                                        "arguments": '{"query":"Heta"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            }

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = ToolCallingLanguageModel(
            model_name="openai/test-tool-model",
            api_key="test-key",
            api_base="https://example.test/v1",
        )
        return await model.invoke_with_tools(
            ToolCallingModelRequest(
                messages=(
                    ToolMessage(role="system", content="Use tools carefully."),
                    ToolMessage(role="user", content="Find Heta docs"),
                ),
                tools=(
                    ToolDefinition(
                        name="search_wiki",
                        description="Search wiki pages.",
                        parameters_schema={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    ),
                ),
                tool_choice="search_wiki",
                options=ModelOptions(temperature=0, max_output_tokens=128),
                trace_context={"stage": "tool_test"},
            )
        )

    result = asyncio.run(run())

    assert calls[0]["model"] == "openai/test-tool-model"
    assert calls[0]["api_key"] == "test-key"
    assert calls[0]["api_base"] == "https://example.test/v1"
    assert calls[0]["messages"] == [
        {"role": "system", "content": "Use tools carefully."},
        {"role": "user", "content": "Find Heta docs"},
    ]
    assert calls[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search_wiki",
                "description": "Search wiki pages.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    assert calls[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "search_wiki"},
    }
    assert calls[0]["temperature"] == 0
    assert calls[0]["max_tokens"] == 128
    assert result.message.tool_calls == (
        ToolCall(id="call_1", name="search_wiki", arguments={"query": "Heta"}),
    )
    assert result.finish_reason == "tool_calls"
    assert result.trace_context == {"stage": "tool_test"}
    assert result.token_usage is not None
    assert result.token_usage.total_tokens == 13


def test_tool_calling_language_model_rejects_unsupported_litellm_model(monkeypatch):
    class FakeLiteLLM:
        def supports_function_calling(self, model):
            return False

        async def acompletion(self, **kwargs):
            raise AssertionError("acompletion should not be called")

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = ToolCallingLanguageModel(model_name="ollama/llama2")
        return await model.invoke_with_tools(
            ToolCallingModelRequest(
                messages=(ToolMessage(role="user", content="Find Heta docs"),),
                tools=(ToolDefinition(name="search_wiki", description="Search wiki pages."),),
            )
        )

    with pytest.raises(ModelRequestError, match="does not support function calling"):
        asyncio.run(run())


def test_tool_calling_language_model_can_bypass_litellm_support_check(monkeypatch):
    calls = []

    class FakeLiteLLM:
        def supports_function_calling(self, model):
            return False

        async def acompletion(self, **kwargs):
            calls.append(kwargs)
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "done"},
                    }
                ]
            }

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = ToolCallingLanguageModel(
            model_name="custom/tool-model",
            validate_function_calling_support=False,
        )
        return await model.invoke_with_tools(
            ToolCallingModelRequest(
                messages=(ToolMessage(role="user", content="Find Heta docs"),),
                tools=(ToolDefinition(name="search_wiki", description="Search wiki pages."),),
                tool_choice="auto",
            )
        )

    result = asyncio.run(run())

    assert calls
    assert result.message.content == "done"
