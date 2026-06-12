import asyncio
import base64
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import ImagePart, LanguageModel, ModelRequest, TextPart


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def test_language_model_invokes_litellm_and_parses_json(monkeypatch):
    calls = []

    class FakeLiteLLM:
        async def acompletion(self, **kwargs):
            calls.append(kwargs)
            return {
                "choices": [
                    {
                        "message": {"content": '{"entities": ["Heta"]}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            }

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = LanguageModel(
            model_name="openai/test-model",
            api_key="test-key",
            api_base="https://example.test/v1",
            max_concurrent_requests=2,
        )
        return await model.invoke(
            ModelRequest(
                prompt="extract",
                response_schema={"type": "object"},
                trace_context={"stage": "unit_test"},
            )
        )

    result = asyncio.run(run())

    assert calls[0]["model"] == "openai/test-model"
    assert calls[0]["api_key"] == "test-key"
    assert calls[0]["api_base"] == "https://example.test/v1"
    assert calls[0]["messages"][-1]["content"] == "extract"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert result.text == '{"entities": ["Heta"]}'
    assert result.parsed == {"entities": ["Heta"]}
    assert result.model_name == "openai/test-model"
    assert result.token_usage is not None
    assert result.token_usage.total_tokens == 5
    assert result.trace_context == {"stage": "unit_test"}


def test_language_model_streams_chunks(monkeypatch):
    calls = []

    async def fake_stream():
        yield {"choices": [{"delta": {"content": "He"}, "finish_reason": None}]}
        yield {"choices": [{"delta": {"content": "ta"}, "finish_reason": None}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    class FakeLiteLLM:
        async def acompletion(self, **kwargs):
            calls.append(kwargs)
            return fake_stream()

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = LanguageModel(model_name="openai/test-model")
        return [chunk async for chunk in model.stream(ModelRequest(prompt="hello"))]

    chunks = asyncio.run(run())

    assert calls[0]["stream"] is True
    assert calls[0]["messages"][-1]["content"] == "hello"
    assert [chunk.text_delta for chunk in chunks] == ["He", "ta", ""]
    assert chunks[-1].finish_reason == "stop"


def test_language_model_accepts_litellm_object_responses(monkeypatch):
    class FakeResponse(SimpleNamespace):
        def model_dump(self):
            return {
                "choices": [
                    {
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            }

    class FakeLiteLLM:
        async def acompletion(self, **kwargs):
            return FakeResponse()

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = LanguageModel(model_name="openai/test-model")
        return await model.invoke(ModelRequest(prompt="hello"))

    result = asyncio.run(run())

    assert result.text == "ok"
    assert result.raw_response is not None


def test_language_model_accepts_multimodal_content(monkeypatch):
    calls = []

    class FakeLiteLLM:
        async def acompletion(self, **kwargs):
            calls.append(kwargs)
            return {
                "choices": [
                    {
                        "message": {"content": "image description"},
                        "finish_reason": "stop",
                    }
                ]
            }

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = LanguageModel(model_name="openai/test-vision-model")
        return await model.invoke(
            ModelRequest(
                content=[
                    TextPart("Describe this image."),
                    ImagePart.from_uri(
                        "https://example.test/image.png",
                        detail="high",
                        format="image/png",
                    ),
                ]
            )
        )

    result = asyncio.run(run())

    content = calls[0]["messages"][-1]["content"]
    assert content == [
        {"type": "text", "text": "Describe this image."},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.test/image.png",
                "detail": "high",
                "format": "image/png",
            },
        },
    ]
    assert result.text == "image description"


def test_language_model_accepts_image_path(monkeypatch, tmp_path):
    calls = []
    image_path = tmp_path / "pixel.png"
    image_path.write_bytes(PNG_1X1)

    class FakeLiteLLM:
        async def acompletion(self, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = LanguageModel(model_name="openai/test-vision-model")
        return await model.invoke(
            ModelRequest(
                content=[
                    TextPart("Describe this image."),
                    ImagePart.from_file(image_path),
                ]
            )
        )

    asyncio.run(run())

    image_url = calls[0]["messages"][-1]["content"][1]["image_url"]
    assert image_url["url"].startswith("data:image/png;base64,")
    assert image_url["format"] == "image/png"


def test_language_model_accepts_image_bytes(monkeypatch):
    calls = []

    class FakeLiteLLM:
        async def acompletion(self, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM())

    async def run():
        model = LanguageModel(model_name="openai/test-vision-model")
        return await model.invoke(
            ModelRequest(
                content=[
                    TextPart("Describe this uploaded image."),
                    ImagePart.from_bytes(PNG_1X1, mime_type="image/png"),
                ]
            )
        )

    asyncio.run(run())

    image_url = calls[0]["messages"][-1]["content"][1]["image_url"]
    assert image_url["url"].startswith("data:image/png;base64,")
    assert image_url["format"] == "image/png"
