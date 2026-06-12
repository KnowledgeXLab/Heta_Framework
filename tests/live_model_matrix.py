"""Run live model compatibility checks from provider model lists.

This script is intentionally not named ``test_*.py`` because it calls real
provider APIs and can be slow or costly. It discovers model IDs from official
OpenAI-compatible ``/models`` endpoints, filters text chat candidates for the
current ``LanguageModel`` interface, runs a small compatibility matrix, and
writes a JSON report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import LanguageModel, ModelOptions, ModelRequest


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    base_url: str
    api_key_env: str
    prefix: str


@dataclass
class ModelCheck:
    provider: str
    model_name: str
    official_source: str
    invoke: bool
    structured_output: bool
    invoke_many: bool
    stream: bool
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.invoke and self.structured_output and self.invoke_many and self.stream


PROVIDERS = [
    ProviderConfig(
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="HETA_LIVE_OPENAI_API_KEY",
        prefix="gpt-5",
    ),
    ProviderConfig(
        provider="dashscope",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="HETA_LIVE_QWEN_API_KEY",
        prefix="qwen3",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "dashscope", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="Maximum models per provider; 0 means no limit.")
    parser.add_argument(
        "--output",
        default="tests/live_reports/model_matrix.json",
        help="JSON report path.",
    )
    args = parser.parse_args()

    selected = [p for p in PROVIDERS if args.provider in {"all", p.provider}]
    report = asyncio.run(run_matrix(selected, limit=args.limit))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_summary(report), ensure_ascii=False, indent=2))
    print(f"Wrote {output}")


async def run_matrix(providers: list[ProviderConfig], *, limit: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "Provider /models endpoints",
        "providers": {},
    }
    for provider in providers:
        api_key = os.getenv(provider.api_key_env)
        if not api_key:
            report["providers"][provider.provider] = {
                "error": f"missing environment variable: {provider.api_key_env}",
                "models": [],
            }
            continue
        official_models = await list_models(provider, api_key)
        candidates = [
            model_id
            for model_id in official_models
            if model_id.startswith(provider.prefix) and is_text_chat_candidate(provider.provider, model_id)
        ]
        if limit:
            candidates = candidates[:limit]
        checks: list[ModelCheck] = []
        for model_name in candidates:
            check = await check_model(provider, api_key, model_name)
            checks.append(check)
            status = "PASS" if check.passed else "FAIL"
            print(f"{status} {provider.provider} {model_name} {check.error or ''}", flush=True)
        report["providers"][provider.provider] = {
            "official_models_count": len(official_models),
            "candidate_models_count": len(candidates),
            "models": [asdict(check) | {"passed": check.passed} for check in checks],
        }
    return report


async def list_models(provider: ProviderConfig, api_key: str) -> list[str]:
    async with httpx.AsyncClient(
        base_url=provider.base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    ) as client:
        response = await client.get("/models")
        response.raise_for_status()
        data = response.json().get("data", [])
    return sorted(str(item.get("id", "")) for item in data if item.get("id"))


def is_text_chat_candidate(provider: str, model_name: str) -> bool:
    lowered = model_name.lower()
    excluded = (
        "asr",
        "tts",
        "realtime",
        "livetranslate",
        "omni",
        "vl",
        "ocr",
        "s2s",
        "search-api",
        "codex",
        "-pro",
    )
    if any(token in lowered for token in excluded):
        return False
    return True


async def check_model(provider: ProviderConfig, api_key: str, model_name: str) -> ModelCheck:
    model = LanguageModel(
        model_name=litellm_model_name(model_name),
        api_base=provider.base_url,
        api_key=api_key,
        request_timeout=90,
        max_retries=0,
        max_concurrent_requests=2,
        default_temperature=temperature(provider.provider, model_name),
    )
    try:
        opts = options(provider.provider, model_name)
        invoke = await check_invoke(model, opts)
        structured = await check_structured_output(model, opts)
        many = await check_invoke_many(model, opts)
        stream = await check_stream(model, opts)
        return ModelCheck(
            provider=provider.provider,
            model_name=model_name,
            official_source=f"{provider.base_url}/models",
            invoke=invoke,
            structured_output=structured,
            invoke_many=many,
            stream=stream,
        )
    except Exception as exc:
        return ModelCheck(
            provider=provider.provider,
            model_name=model_name,
            official_source=f"{provider.base_url}/models",
            invoke=False,
            structured_output=False,
            invoke_many=False,
            stream=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        await model.aclose()


def temperature(provider: str, model_name: str) -> float:
    if provider == "openai" and model_name.startswith("gpt-5"):
        return 1
    return 0


def options(provider: str, model_name: str) -> ModelOptions:
    provider_options = None
    if provider == "dashscope" and "thinking" not in model_name and "preview" not in model_name:
        provider_options = {"enable_thinking": False}
    return ModelOptions(
        temperature=temperature(provider, model_name),
        top_p=1,
        max_output_tokens=256,
        provider_options=provider_options,
    )


def litellm_model_name(model_name: str) -> str:
    return model_name if "/" in model_name else f"openai/{model_name}"


async def check_invoke(model: LanguageModel, opts: ModelOptions) -> bool:
    result = await model.invoke(
        ModelRequest(prompt="Reply with exactly: ok", options=opts, trace_context={"case": "invoke"})
    )
    return bool(result.text.strip())


async def check_structured_output(model: LanguageModel, opts: ModelOptions) -> bool:
    json_opts = ModelOptions(
        temperature=opts.temperature,
        top_p=opts.top_p,
        max_output_tokens=opts.max_output_tokens,
        response_format={"type": "json_object"},
        provider_options=opts.provider_options,
    )
    result = await model.invoke(
        ModelRequest(
            system_prompt="Return only valid JSON. Do not include markdown fences.",
            prompt='Return JSON exactly matching this object: {"status":"ok","value":7}',
            options=json_opts,
            response_schema={"type": "object"},
            trace_context={"case": "structured_output"},
        )
    )
    return isinstance(result.parsed, dict) and result.parsed.get("status") == "ok"


async def check_invoke_many(model: LanguageModel, opts: ModelOptions) -> bool:
    results = await model.invoke_many(
        [
            ModelRequest(
                prompt=f"Reply with exactly: item-{index}",
                options=opts,
                trace_context={"case": "invoke_many", "index": index},
            )
            for index in range(2)
        ]
    )
    return (
        len(results) == 2
        and bool(results[0].text.strip())
        and bool(results[1].text.strip())
        and results[0].trace_context == {"case": "invoke_many", "index": 0}
        and results[1].trace_context == {"case": "invoke_many", "index": 1}
    )


async def check_stream(model: LanguageModel, opts: ModelOptions) -> bool:
    chunks = [
        chunk
        async for chunk in model.stream(
            ModelRequest(
                prompt="Reply with exactly: stream-ok",
                options=opts,
                trace_context={"case": "stream"},
            )
        )
    ]
    text = "".join(chunk.text_delta for chunk in chunks).lower()
    return bool(text.strip())


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for provider, payload in report["providers"].items():
        models = payload.get("models", [])
        summary[provider] = {
            "candidates": payload.get("candidate_models_count", 0),
            "passed": sum(1 for model in models if model.get("passed")),
            "failed": sum(1 for model in models if not model.get("passed")),
        }
    return summary


if __name__ == "__main__":
    main()
