import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import (
    LanguageModel,
    ModelOptions,
    ModelRequest,
)


PROVIDERS = [
    pytest.param(
        {
            "provider": "openai",
            "api_key_env": "HETA_LIVE_OPENAI_API_KEY",
            "model_name_env": "HETA_LIVE_OPENAI_MODEL_NAME",
            "base_url": "https://api.openai.com/v1",
            "model_name": "openai/gpt-4o-mini",
            "provider_options": {},
            "supports_json_mode": True,
            "supports_stop": True,
        },
        id="openai",
    ),
    pytest.param(
        {
            "provider": "dashscope",
            "api_key_env": "HETA_LIVE_QWEN_API_KEY",
            "model_name_env": "HETA_LIVE_QWEN_MODEL_NAME",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model_name": "openai/qwen-plus",
            "provider_options": {"enable_thinking": False},
            "supports_json_mode": True,
            "supports_stop": True,
        },
        id="dashscope-qwen",
    ),
]


def _require_model(provider_config: dict) -> LanguageModel:
    api_key = os.getenv(provider_config["api_key_env"])
    if not api_key:
        pytest.skip(f"set {provider_config['api_key_env']} to run live model tests")
    model_name = _model_name(provider_config)
    return LanguageModel(
        model_name=model_name,
        api_base=provider_config["base_url"],
        api_key=api_key,
        request_timeout=60,
        max_retries=1,
        max_concurrent_requests=2,
        default_temperature=_temperature(provider_config),
    )


def _options(provider_config: dict, **overrides) -> ModelOptions:
    provider_options = dict(provider_config.get("provider_options") or {})
    provider_options.update(overrides.pop("provider_options", {}) or {})
    return ModelOptions(
        temperature=overrides.pop("temperature", _temperature(provider_config)),
        max_output_tokens=overrides.pop("max_output_tokens", 64),
        top_p=overrides.pop("top_p", 1),
        stop_sequences=overrides.pop("stop_sequences", None),
        response_format=overrides.pop("response_format", None),
        provider_options=provider_options or None,
    )


def _model_name(provider_config: dict) -> str:
    model_name = os.getenv(provider_config["model_name_env"], provider_config["model_name"])
    if "/" not in model_name:
        return f"openai/{model_name}"
    return model_name


def _is_openai_gpt5(provider_config: dict) -> bool:
    model_name = _model_name(provider_config).removeprefix("openai/")
    return provider_config["provider"] == "openai" and model_name.startswith("gpt-5")


def _temperature(provider_config: dict) -> float:
    return 1 if _is_openai_gpt5(provider_config) else 0


def _supports_stop(provider_config: dict) -> bool:
    return provider_config["supports_stop"] and not _is_openai_gpt5(provider_config)


@pytest.mark.parametrize("provider_config", PROVIDERS)
def test_live_invoke_with_all_model_options(provider_config):
    async def run():
        model = _require_model(provider_config)
        try:
            result = await model.invoke(
                ModelRequest(
                    system_prompt="You are a precise test assistant.",
                    prompt="Reply with exactly: Heta live invoke ok",
                    options=_options(
                        provider_config,
                        stop_sequences=["\n"] if _supports_stop(provider_config) else None,
                    ),
                    trace_context={
                        "test": "live_invoke_with_all_model_options",
                        "provider": provider_config["provider"],
                    },
                )
            )
        finally:
            await model.aclose()

        assert result.model_name == _model_name(provider_config)
        assert result.trace_context is not None
        assert result.trace_context["test"] == "live_invoke_with_all_model_options"
        assert "Heta" in result.text

    asyncio.run(run())


@pytest.mark.parametrize("provider_config", PROVIDERS)
def test_live_invoke_parses_json_response_schema(provider_config):
    async def run():
        model = _require_model(provider_config)
        try:
            result = await model.invoke(
                ModelRequest(
                    system_prompt="Return only valid JSON. Do not include markdown fences.",
                    prompt='Return JSON exactly matching this object: {"status":"ok","value":7}',
                    options=_options(
                        provider_config,
                        response_format=(
                            {"type": "json_object"}
                            if provider_config["supports_json_mode"]
                            else None
                        ),
                    ),
                    response_schema={"type": "object"},
                    trace_context={
                        "test": "live_invoke_parses_json_response_schema",
                        "provider": provider_config["provider"],
                    },
                )
            )
        finally:
            await model.aclose()

        assert isinstance(result.parsed, dict)
        assert result.parsed.get("status") == "ok"
        assert int(result.parsed.get("value")) == 7

    asyncio.run(run())


@pytest.mark.parametrize("provider_config", PROVIDERS)
def test_live_invoke_many_preserves_order(provider_config):
    async def run():
        model = _require_model(provider_config)
        try:
            results = await model.invoke_many(
                [
                        ModelRequest(
                            prompt=f"Reply with exactly: item-{index}",
                            options=_options(provider_config, max_output_tokens=64),
                            trace_context={"test": "live_invoke_many_preserves_order", "index": index},
                        )
                    for index in range(2)
                ]
            )
        finally:
            await model.aclose()

        assert len(results) == 2
        assert [result.trace_context["index"] for result in results] == [0, 1]
        assert "item-0" in results[0].text
        assert "item-1" in results[1].text

    asyncio.run(run())


@pytest.mark.parametrize("provider_config", PROVIDERS)
def test_live_stream_returns_text_chunks(provider_config):
    async def run():
        model = _require_model(provider_config)
        try:
            chunks = [
                chunk
                async for chunk in model.stream(
                    ModelRequest(
                        prompt="Reply with exactly: Heta stream ok",
                        options=_options(provider_config, max_output_tokens=32),
                        trace_context={
                            "test": "live_stream_returns_text_chunks",
                            "provider": provider_config["provider"],
                        },
                    )
                )
            ]
        finally:
            await model.aclose()

        text = "".join(chunk.text_delta for chunk in chunks)
        assert chunks
        assert "Heta" in text
        assert chunks[0].trace_context is not None
        assert chunks[0].trace_context["test"] == "live_stream_returns_text_chunks"

    asyncio.run(run())
