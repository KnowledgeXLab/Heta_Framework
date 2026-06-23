"""Language-model helpers shared by built-in query engines."""

from __future__ import annotations

import json
import re
from typing import Any

from heta_framework.common.models import ModelOptions, ModelRequest
from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.kb.search.types import QueryResult
from heta_framework.kb.steps.types import model_ref

try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover - optional runtime dependency.
    repair_json = None


def require_language_model(component: object) -> LanguageModelProtocol:
    """Return a language model component or raise a clear type error."""
    if not isinstance(component, LanguageModelProtocol):
        raise TypeError("models.language must satisfy LanguageModelProtocol")
    return component


def language_model_from_context(context: object, name: str | None) -> LanguageModelProtocol:
    """Load a language model from a query context."""
    recipe = getattr(context, "recipe")
    return require_language_model(recipe.get_component(model_ref("language", name)))


async def invoke_json(
    language_model: LanguageModelProtocol,
    *,
    prompt: str,
    trace_context: dict[str, Any],
    max_output_tokens: int = 1024,
) -> dict[str, Any]:
    """Invoke a language model and parse a JSON-object response."""
    result = await language_model.invoke(
        ModelRequest(
            prompt=prompt,
            options=ModelOptions(
                temperature=0,
                max_output_tokens=max_output_tokens,
                response_format={"type": "json_object"},
            ),
            response_schema={"type": "object"},
            trace_context=trace_context,
        )
    )
    if isinstance(result.parsed, dict):
        return result.parsed
    parsed = parse_json_object(result.text)
    return parsed if isinstance(parsed, dict) else {}


async def answer_from_results(
    language_model: LanguageModelProtocol,
    *,
    query: str,
    results: tuple[QueryResult, ...],
    trace_context: dict[str, Any],
) -> str:
    """Generate a concise answer from retrieved results."""
    if not results:
        context_text = "No relevant context was retrieved."
    else:
        context_text = "\n\n".join(
            f"[{index}] {result.text}" for index, result in enumerate(results, start=1)
        )
    result = await language_model.invoke(
        ModelRequest(
            prompt=(
                "Answer the user question using only the retrieved context. "
                "If the context is insufficient, say that the available context "
                "is insufficient.\n\n"
                f"Retrieved context:\n{context_text}\n\n"
                f"User question: {query}"
            ),
            options=ModelOptions(temperature=0.1),
            trace_context=trace_context,
        )
    )
    return result.text


def parse_json_object(text: str) -> Any:
    """Parse a JSON object from LLM text, tolerating markdown fences."""
    cleaned = _strip_thinking(text).strip()
    cleaned = _strip_code_fence(cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        if repair_json is not None:
            try:
                return json.loads(repair_json(cleaned))
            except Exception:
                pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text
