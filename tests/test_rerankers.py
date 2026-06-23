import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import (
    RerankModel,
    RerankOptions,
    RerankRequest,
    RerankRequestError,
    RerankResponseError,
)


def test_rerank_model_invokes_litellm_and_preserves_rank_order(monkeypatch):
    calls = []

    class FakeLiteLLM:
        async def arerank(self, **kwargs):
            calls.append(kwargs)
            return {
                "results": [
                    {"index": 2, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.42},
                ],
                "meta": {"api_version": "test"},
            }

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = RerankModel(
            model_name="cohere/rerank-english-v3.0",
            api_key="test-key",
            api_base="https://example.test/v1",
            top_n=2,
            provider_options={"custom_flag": True},
        )
        return await model.rerank(
            RerankRequest(
                query="capital of the united states",
                documents=[
                    "Washington, D.C. is the capital of the United States.",
                    "Carson City is the capital of Nevada.",
                    "The United States has many state capitals.",
                ],
                trace_context={"stage": "unit_test"},
            )
        )

    result = asyncio.run(run())

    assert calls[0]["model"] == "cohere/rerank-english-v3.0"
    assert calls[0]["query"] == "capital of the united states"
    assert calls[0]["documents"][0].startswith("Washington")
    assert calls[0]["api_key"] == "test-key"
    assert calls[0]["api_base"] == "https://example.test/v1"
    assert calls[0]["top_n"] == 2
    assert calls[0]["custom_flag"] is True
    assert [item.index for item in result.rankings] == [2, 0]
    assert [item.score for item in result.rankings] == [0.91, 0.42]
    assert result.model_name == "cohere/rerank-english-v3.0"
    assert result.trace_context == {"stage": "unit_test"}
    assert result.raw_response is not None


def test_rerank_model_accepts_per_request_options(monkeypatch):
    calls = []

    class FakeLiteLLM:
        async def arerank(self, **kwargs):
            calls.append(kwargs)
            return {
                "results": [
                    {
                        "index": 1,
                        "score": 0.8,
                        "document": {"text": "beta document", "id": "doc-2"},
                    }
                ]
            }

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = RerankModel(
            model_name="voyage/rerank-2",
            top_n=5,
            provider_options={"user": "default"},
        )
        return await model.rerank(
            RerankRequest(
                query="beta",
                documents=["alpha document", "beta document"],
                options=RerankOptions(
                    top_n=1,
                    return_documents=True,
                    provider_options={"user": "request"},
                ),
            )
        )

    result = asyncio.run(run())

    assert calls[0]["top_n"] == 1
    assert calls[0]["return_documents"] is True
    assert calls[0]["user"] == "request"
    assert result.rankings[0].text == "beta document"
    assert result.rankings[0].metadata == {"document": {"id": "doc-2"}}


def test_rerank_model_rerank_many_preserves_order(monkeypatch):
    class FakeLiteLLM:
        async def arerank(self, **kwargs):
            score = float(kwargs["query"][-1])
            return {"results": [{"index": 0, "relevance_score": score}]}

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = RerankModel(model_name="cohere/test-rerank")
        return await model.rerank_many(
            [
                RerankRequest(query="query-1", documents=["document"]),
                RerankRequest(query="query-2", documents=["document"]),
            ]
        )

    results = asyncio.run(run())

    assert [result.rankings[0].score for result in results] == [1.0, 2.0]


def test_rerank_model_accepts_litellm_object_responses(monkeypatch):
    class FakeResponse(SimpleNamespace):
        def model_dump(self):
            return {"results": [{"index": 0, "relevance_score": 0.5}]}

    class FakeLiteLLM:
        async def arerank(self, **kwargs):
            return FakeResponse()

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = RerankModel(model_name="cohere/test-rerank")
        return await model.rerank(RerankRequest(query="hello", documents=["hello world"]))

    result = asyncio.run(run())

    assert result.rankings[0].score == 0.5
    assert result.raw_response is not None


def test_rerank_model_rejects_empty_query():
    async def run():
        model = RerankModel(model_name="cohere/test-rerank")
        await model.rerank(RerankRequest(query=" ", documents=["document"]))

    with pytest.raises(RerankRequestError):
        asyncio.run(run())


def test_rerank_model_rejects_out_of_range_response_index(monkeypatch):
    class FakeLiteLLM:
        async def arerank(self, **kwargs):
            return {"results": [{"index": 2, "relevance_score": 0.7}]}

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = RerankModel(model_name="cohere/test-rerank")
        await model.rerank(RerankRequest(query="hello", documents=["document"]))

    with pytest.raises(RerankResponseError):
        asyncio.run(run())
