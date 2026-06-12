import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import (
    EmbeddingModel,
    EmbeddingOptions,
    EmbeddingRequest,
    EmbeddingRequestError,
)


def test_embedding_model_invokes_litellm_and_preserves_input_order(monkeypatch):
    calls = []

    class FakeLiteLLM:
        async def aembedding(self, **kwargs):
            calls.append(kwargs)
            return {
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            }

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = EmbeddingModel(
            model_name="openai/text-embedding-3-small",
            api_key="test-key",
            api_base="https://example.test/v1",
            dimensions=2,
            provider_options={"custom_flag": True},
        )
        return await model.embed(
            EmbeddingRequest(
                texts=["first", "second"],
                trace_context={"stage": "unit_test"},
            )
        )

    result = asyncio.run(run())

    assert calls[0]["model"] == "openai/text-embedding-3-small"
    assert calls[0]["input"] == ["first", "second"]
    assert calls[0]["api_key"] == "test-key"
    assert calls[0]["api_base"] == "https://example.test/v1"
    assert calls[0]["dimensions"] == 2
    assert calls[0]["custom_flag"] is True
    assert result.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert result.model_name == "openai/text-embedding-3-small"
    assert result.usage is not None
    assert result.usage.total_tokens == 4
    assert result.trace_context == {"stage": "unit_test"}


def test_embedding_model_accepts_per_request_options(monkeypatch):
    calls = []

    class FakeLiteLLM:
        async def aembedding(self, **kwargs):
            calls.append(kwargs)
            return {"data": [{"index": 0, "embedding": [1, 2, 3]}]}

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = EmbeddingModel(
            model_name="openai/text-embedding-3-large",
            dimensions=1536,
            encoding_format="float",
            provider_options={"user": "default"},
        )
        return await model.embed(
            EmbeddingRequest(
                texts=["hello"],
                options=EmbeddingOptions(
                    dimensions=3,
                    encoding_format="float",
                    provider_options={"user": "request"},
                ),
            )
        )

    result = asyncio.run(run())

    assert calls[0]["dimensions"] == 3
    assert calls[0]["encoding_format"] == "float"
    assert calls[0]["user"] == "request"
    assert result.vectors == [[1.0, 2.0, 3.0]]


def test_embedding_model_embed_many_preserves_order(monkeypatch):
    class FakeLiteLLM:
        async def aembedding(self, **kwargs):
            text = kwargs["input"][0]
            return {"data": [{"index": 0, "embedding": [float(text[-1])]}]}

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = EmbeddingModel(model_name="openai/test-embedding")
        return await model.embed_many(
            [
                EmbeddingRequest(texts=["item-1"], trace_context={"index": 1}),
                EmbeddingRequest(texts=["item-2"], trace_context={"index": 2}),
            ]
        )

    results = asyncio.run(run())

    assert [result.vectors for result in results] == [[[1.0]], [[2.0]]]
    assert [result.trace_context["index"] for result in results] == [1, 2]


def test_embedding_model_accepts_litellm_object_responses(monkeypatch):
    class FakeResponse(SimpleNamespace):
        def model_dump(self):
            return {"data": [{"index": 0, "embedding": [0.5]}]}

    class FakeLiteLLM:
        async def aembedding(self, **kwargs):
            return FakeResponse()

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = EmbeddingModel(model_name="openai/test-embedding")
        return await model.embed(EmbeddingRequest(texts=["hello"]))

    result = asyncio.run(run())

    assert result.vectors == [[0.5]]
    assert result.raw_response is not None


def test_embedding_model_rejects_empty_texts():
    async def run():
        model = EmbeddingModel(model_name="openai/test-embedding")
        await model.embed(EmbeddingRequest(texts=["  "]))

    with pytest.raises(EmbeddingRequestError):
        asyncio.run(run())
