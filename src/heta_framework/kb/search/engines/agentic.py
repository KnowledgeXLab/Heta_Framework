"""Tool-calling query engine with bounded retrieval and grounded synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from heta_framework.common.models import (
    EmbeddingModelProtocol,
    ModelOptions,
    ModelRequest,
    TokenUsage,
    ToolCallingLanguageModelProtocol,
    ToolCallingModelRequest,
    ToolMessage,
)
from heta_framework.kb.search.assets import SearchAssetRef
from heta_framework.kb.search.engines._language import parse_json_object, should_generate_answer
from heta_framework.kb.search.engines.agentic_prompts import (
    AGENTIC_RETRIEVAL_SYSTEM_PROMPT,
    GROUNDED_SYNTHESIS_SYSTEM_PROMPT,
    retrieval_user_prompt,
    synthesis_user_prompt,
)
from heta_framework.kb.search.protocols import QueryContext
from heta_framework.kb.search.tools import (
    QueryEvidenceLedger,
    QueryEvidenceRecord,
    QueryToolBudget,
    QueryToolContext,
    QueryToolProtocol,
    QueryToolRegistry,
    QueryToolResult,
)
from heta_framework.kb.search.types import (
    QueryCitation,
    QueryInsight,
    QueryRequest,
    QueryResponse,
    QueryResult,
    QueryTraceEvent,
)
from heta_framework.kb.search.wiki_tools import (
    ReadRawObjectTool,
    ReadWikiIndexTool,
    ReadWikiPageTool,
    SearchWikiTool,
)
from heta_framework.kb.steps.types import ComponentRef, model_ref

if TYPE_CHECKING:
    from heta_framework.kb.recipe import KnowledgeRecipe

_FINAL_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "insights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "uniqueItems": True,
                    },
                },
                "required": ["text", "evidence_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "insights"],
    "additionalProperties": False,
}


def _default_wiki_tools() -> tuple[QueryToolProtocol, ...]:
    return (
        ReadWikiIndexTool(),
        SearchWikiTool(),
        ReadWikiPageTool(),
        ReadRawObjectTool(),
    )


@dataclass(frozen=True)
class AgenticQueryEngine:
    """Dynamically retrieve wiki evidence and synthesize a grounded answer."""

    mode: str = "wiki_agent_search"
    tools: tuple[QueryToolProtocol, ...] = field(default_factory=_default_wiki_tools)
    language_model: str | None = None
    max_steps: int = 8
    max_synthesis_attempts: int = 2
    planning_max_output_tokens: int = 1024
    answer_max_output_tokens: int = 1536
    tool_budget: QueryToolBudget = field(default_factory=QueryToolBudget)

    def __post_init__(self) -> None:
        if self.mode.strip() == "":
            raise ValueError("mode must not be empty")
        if not self.tools:
            raise ValueError("tools must not be empty")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        if self.max_synthesis_attempts <= 0:
            raise ValueError("max_synthesis_attempts must be greater than zero")
        if self.planning_max_output_tokens <= 0:
            raise ValueError("planning_max_output_tokens must be greater than zero")
        if self.answer_max_output_tokens <= 0:
            raise ValueError("answer_max_output_tokens must be greater than zero")
        registry = QueryToolRegistry(self.tools)
        object.__setattr__(self, "mode", self.mode.strip())
        object.__setattr__(self, "tools", registry.tools)

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        """Return the union of search assets required by configured tools."""
        return QueryToolRegistry(self.tools).required_assets

    @property
    def required_components(self) -> frozenset[ComponentRef]:
        """Return the tool-calling model and components required by configured tools."""
        tool_components = QueryToolRegistry(self.tools).required_components
        return frozenset(
            {*tool_components, model_ref("language", self.language_model)}
        )

    def is_available_for(self, recipe: "KnowledgeRecipe") -> bool:
        """Return whether the recipe provides model capabilities used by the tools."""
        try:
            component = recipe.get_component(model_ref("language", self.language_model))
        except LookupError:
            return False
        if not isinstance(component, ToolCallingLanguageModelProtocol):
            return False
        for tool in self.tools:
            if not isinstance(tool, SearchWikiTool):
                continue
            try:
                embedding = recipe.get_component(
                    model_ref("embedding", tool.embedding_model)
                )
            except LookupError:
                return False
            if not isinstance(embedding, EmbeddingModelProtocol):
                return False
        return True

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        """Run a bounded tool loop followed by grounded answer synthesis."""
        language_model = _tool_calling_model(context, self.language_model)
        tool_registry = QueryToolRegistry(self.tools)
        ledger = QueryEvidenceLedger()
        tool_context = QueryToolContext(
            query_context=context,
            request=request,
            ledger=ledger,
            budget=self.tool_budget,
        )
        messages = [
            ToolMessage(role="system", content=AGENTIC_RETRIEVAL_SYSTEM_PROMPT),
            ToolMessage(role="user", content=retrieval_user_prompt(request.text)),
        ]
        trace_events: list[QueryTraceEvent] = []
        issues: list[dict[str, object]] = []
        usage = _UsageAccumulator()
        tool_call_count = 0
        executed_tool_calls = 0
        model_steps = 0
        termination_reason = "max_steps"

        for step_index in range(1, _max_steps(request, self.max_steps) + 1):
            model_steps = step_index
            model_result = await language_model.invoke_with_tools(
                ToolCallingModelRequest(
                    messages=tuple(messages),
                    tools=tool_registry.model_tool_definitions(),
                    tool_choice="auto",
                    options=ModelOptions(
                        temperature=0,
                        max_output_tokens=self.planning_max_output_tokens,
                    ),
                    trace_context={
                        "query_mode": self.mode,
                        "stage": "retrieval_planning",
                        "step": step_index,
                    },
                )
            )
            usage.add(model_result.token_usage)
            messages.append(model_result.message)
            tool_calls = model_result.message.tool_calls
            if not tool_calls:
                termination_reason = "model_completed"
                _trace(
                    trace_events,
                    request=request,
                    stage="agentic_retrieval",
                    message="The retrieval model completed evidence gathering.",
                    metadata={"step": step_index},
                )
                break

            budget_exhausted = False
            for tool_call in tool_calls:
                tool_call_count += 1
                if tool_call_count > self.tool_budget.max_tool_calls:
                    result = QueryToolResult.error(
                        "error: tool-call budget exhausted",
                        metadata={"citable": False, "budget_exhausted": True},
                    )
                    issues.append(
                        _issue(
                            "tool_call_budget_exhausted",
                            "A model-requested tool call was rejected by the execution budget.",
                            tool=tool_call.name,
                            tool_call_id=tool_call.id,
                        )
                    )
                    budget_exhausted = True
                else:
                    result, executed = await _run_tool(
                        tool_call.name,
                        tool_call.arguments,
                        registry=tool_registry,
                        context=tool_context,
                    )
                    executed_tool_calls += int(executed)
                    if result.is_error:
                        issues.append(
                            _issue(
                                "tool_call_failed",
                                result.content,
                                tool=tool_call.name,
                                tool_call_id=tool_call.id,
                            )
                        )

                bounded_result = _bounded_tool_result(result, ledger, self.tool_budget)
                if bounded_result is None:
                    tool_content = "error: evidence budget exhausted"
                    budget_exhausted = True
                    issues.append(
                        _issue(
                            "evidence_budget_exhausted",
                            (
                                "A tool result could not be recorded because the "
                                "evidence budget was full."
                            ),
                            tool=tool_call.name,
                            tool_call_id=tool_call.id,
                        )
                    )
                else:
                    ledger.add_result(
                        tool_name=tool_call.name,
                        tool_call_id=tool_call.id,
                        result=bounded_result,
                    )
                    tool_content = bounded_result.content

                messages.append(
                    ToolMessage(
                        role="tool",
                        content=tool_content,
                        tool_call_id=tool_call.id,
                    )
                )
                _trace(
                    trace_events,
                    request=request,
                    stage="agentic_tool",
                    message="Executed a model-requested query tool.",
                    metadata={
                        "step": step_index,
                        "tool": tool_call.name,
                        "tool_call_id": tool_call.id,
                        "is_error": result.is_error,
                    },
                )

            if (
                tool_call_count >= self.tool_budget.max_tool_calls
                or ledger.total_content_chars >= self.tool_budget.max_evidence_chars
            ):
                budget_exhausted = True
            if budget_exhausted:
                termination_reason = "budget_exhausted"
                break

        generate_answer = should_generate_answer(request, default=True)
        selected_records: tuple[QueryEvidenceRecord, ...]
        insights: tuple[QueryInsight, ...]
        if generate_answer:
            answer, insights, selected_records, synthesis_issues = await _synthesize_answer(
                language_model,
                question=request.text,
                ledger=ledger,
                max_attempts=self.max_synthesis_attempts,
                max_output_tokens=self.answer_max_output_tokens,
                query_mode=self.mode,
                usage=usage,
            )
            issues.extend(synthesis_issues)
            answer_generation = "generated" if not synthesis_issues else "fallback"
        else:
            answer = None
            insights = ()
            selected_records = _citable_records(ledger.evidence_records)
            answer_generation = "disabled"

        results = _results_from_records(selected_records, top_k=request.top_k)
        citations = _citations_from_records(selected_records, result_ids={r.id for r in results})
        _trace(
            trace_events,
            request=request,
            stage="agentic_synthesis",
            message="Produced the grounded query response.",
            metadata={
                "answer_generation": answer_generation,
                "selected_evidence_ids": tuple(r.evidence_id for r in selected_records),
            },
        )
        return QueryResponse(
            mode=self.mode,
            results=results,
            answer=answer,
            insights=insights,
            citations=citations,
            trace=tuple(trace_events),
            metadata={
                "model": language_model.model_name,
                "model_steps": model_steps,
                "tool_calls": tool_call_count,
                "executed_tool_calls": executed_tool_calls,
                "evidence_count": len(ledger.records),
                "evidence_chars": ledger.total_content_chars,
                "insight_count": len(insights),
                "selected_evidence_ids": tuple(r.evidence_id for r in selected_records),
                "termination_reason": termination_reason,
                "answer_generation": answer_generation,
                "token_usage": usage.to_dict(),
                "issues": tuple(issues),
            },
        )


async def _run_tool(
    name: str,
    arguments: Mapping[str, Any],
    *,
    registry: QueryToolRegistry,
    context: QueryToolContext,
) -> tuple[QueryToolResult, bool]:
    tool = registry.find(name)
    if tool is None:
        return (
            QueryToolResult.error(
                f"error: query tool is not registered: {name}",
                metadata={"citable": False},
            ),
            False,
        )
    try:
        _validate_tool_arguments(arguments, tool.parameters_schema)
    except (TypeError, ValueError) as exc:
        return (
            QueryToolResult.error(
                f"error: invalid tool arguments: {exc}",
                metadata={"citable": False},
            ),
            False,
        )
    try:
        result = await tool.run(arguments, context)
    except Exception as exc:  # noqa: BLE001
        return (
            QueryToolResult.error(
                f"error: query tool failed: {type(exc).__name__}: {exc}",
                metadata={"citable": False},
            ),
            True,
        )
    if not isinstance(result, QueryToolResult):
        return (
            QueryToolResult.error(
                "error: query tool returned an invalid result",
                metadata={"citable": False},
            ),
            True,
        )
    return result, True


def _validate_tool_arguments(
    arguments: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> None:
    if schema.get("type", "object") != "object":
        raise ValueError("tool parameter schema root type must be object")
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ValueError("tool parameter schema properties must be an object")
    required = schema.get("required", ())
    if not isinstance(required, (list, tuple)) or any(
        not isinstance(name, str) for name in required
    ):
        raise ValueError("tool parameter schema required must be a string list")
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError(f"missing required argument(s): {', '.join(missing)}")
    if schema.get("additionalProperties") is False:
        unexpected = [name for name in arguments if name not in properties]
        if unexpected:
            raise ValueError(f"unexpected argument(s): {', '.join(unexpected)}")
    for name, value in arguments.items():
        field_schema = properties.get(name)
        if isinstance(field_schema, Mapping):
            _validate_argument_value(name, value, field_schema)


def _validate_argument_value(
    name: str,
    value: object,
    schema: Mapping[str, Any],
) -> None:
    expected_type = schema.get("type")
    validators = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "object": lambda item: isinstance(item, Mapping),
        "array": lambda item: isinstance(item, (list, tuple)),
    }
    validator = validators.get(expected_type)
    if expected_type is not None and validator is None:
        raise ValueError(f"unsupported schema type for {name}: {expected_type}")
    if validator is not None and not validator(value):
        raise TypeError(f"argument {name} must be {expected_type}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            raise ValueError(f"argument {name} must be at least {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            raise ValueError(f"argument {name} must be at most {maximum}")
    if isinstance(value, str):
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        if isinstance(min_length, int) and len(value) < min_length:
            raise ValueError(f"argument {name} must contain at least {min_length} characters")
        if isinstance(max_length, int) and len(value) > max_length:
            raise ValueError(f"argument {name} must contain at most {max_length} characters")


def _bounded_tool_result(
    result: QueryToolResult,
    ledger: QueryEvidenceLedger,
    budget: QueryToolBudget,
) -> QueryToolResult | None:
    remaining = budget.max_evidence_chars - ledger.total_content_chars
    allowed = min(budget.max_tool_result_chars, remaining)
    if allowed <= 0:
        return None
    content, truncated = _truncate(result.content, allowed)
    metadata = dict(result.metadata)
    if truncated:
        metadata["engine_truncated"] = True
    return QueryToolResult(
        content=content,
        results=result.results,
        citations=result.citations,
        metadata=metadata,
        is_error=result.is_error,
    )


async def _synthesize_answer(
    language_model: ToolCallingLanguageModelProtocol,
    *,
    question: str,
    ledger: QueryEvidenceLedger,
    max_attempts: int,
    max_output_tokens: int,
    query_mode: str,
    usage: "_UsageAccumulator",
) -> tuple[
    str,
    tuple[QueryInsight, ...],
    tuple[QueryEvidenceRecord, ...],
    list[dict[str, object]],
]:
    validation_error: str | None = None
    records = ledger.evidence_records
    for attempt in range(1, max_attempts + 1):
        result = await language_model.invoke(
            ModelRequest(
                system_prompt=GROUNDED_SYNTHESIS_SYSTEM_PROMPT,
                prompt=synthesis_user_prompt(
                    question,
                    records,
                    validation_error=validation_error,
                ),
                options=ModelOptions(
                    temperature=0,
                    max_output_tokens=max_output_tokens,
                    response_format={"type": "json_object"},
                ),
                response_schema=_FINAL_RESPONSE_SCHEMA,
                trace_context={
                    "query_mode": query_mode,
                    "stage": "grounded_synthesis",
                    "attempt": attempt,
                },
            )
        )
        usage.add(result.token_usage)
        data = result.parsed if isinstance(result.parsed, dict) else parse_json_object(result.text)
        try:
            answer, insights, selected = _validated_synthesis(data, records)
        except ValueError as exc:
            validation_error = str(exc)
            continue
        return answer, insights, selected, []

    issue = _issue(
        "invalid_grounded_answer",
        validation_error or "The language model did not return a valid grounded answer.",
    )
    return (
        "The available evidence is insufficient to produce a grounded answer.",
        (),
        _citable_records(records),
        [issue],
    )


def _validated_synthesis(
    data: object,
    records: tuple[QueryEvidenceRecord, ...],
) -> tuple[str, tuple[QueryInsight, ...], tuple[QueryEvidenceRecord, ...]]:
    if not isinstance(data, dict):
        raise ValueError("response must be a JSON object")
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    raw_insights = data.get("insights")
    if not isinstance(raw_insights, list):
        raise ValueError("insights must be a list")
    by_id = {record.evidence_id: record for record in records}
    insights: list[QueryInsight] = []
    selected_ids: list[str] = []
    for index, raw_insight in enumerate(raw_insights, start=1):
        insight = _validated_insight(raw_insight, index=index, evidence=by_id)
        insights.append(insight)
        for evidence_id in insight.evidence_ids:
            if evidence_id not in selected_ids:
                selected_ids.append(evidence_id)
    selected = tuple(by_id[evidence_id] for evidence_id in selected_ids)
    return answer.strip(), tuple(insights), selected


def _validated_insight(
    data: object,
    *,
    index: int,
    evidence: Mapping[str, QueryEvidenceRecord],
) -> QueryInsight:
    if not isinstance(data, dict):
        raise ValueError(f"insight {index} must be an object")
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"insight {index} text must be a non-empty string")
    raw_ids = data.get("evidence_ids")
    if not isinstance(raw_ids, list) or any(not isinstance(item, str) for item in raw_ids):
        raise ValueError(f"insight {index} evidence_ids must be a list of strings")
    evidence_ids = tuple(item.strip() for item in raw_ids)
    if not evidence_ids or any(not evidence_id for evidence_id in evidence_ids):
        raise ValueError(f"insight {index} evidence_ids must not be empty")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError(f"insight {index} evidence_ids must be unique")
    unknown = [evidence_id for evidence_id in evidence_ids if evidence_id not in evidence]
    if unknown:
        raise ValueError(f"unknown evidence ids: {', '.join(unknown)}")
    non_citable = [
        evidence_id for evidence_id in evidence_ids if not _is_citable(evidence[evidence_id])
    ]
    if non_citable:
        raise ValueError(f"non-citable evidence ids: {', '.join(non_citable)}")
    return QueryInsight(text=text, evidence_ids=evidence_ids)


def _results_from_records(
    records: tuple[QueryEvidenceRecord, ...],
    *,
    top_k: int,
) -> tuple[QueryResult, ...]:
    unique: dict[str, QueryResult] = {}
    for record in records:
        for result in record.results:
            unique.setdefault(result.id, result)
    return tuple(unique.values())[:top_k]


def _citations_from_records(
    records: tuple[QueryEvidenceRecord, ...],
    *,
    result_ids: set[str],
) -> tuple[QueryCitation, ...]:
    unique: dict[str, QueryCitation] = {}
    for record in records:
        for citation in record.citations:
            if citation.result_id is None or citation.result_id in result_ids:
                unique.setdefault(citation.id, citation)
    return tuple(unique.values())


def _citable_records(
    records: tuple[QueryEvidenceRecord, ...],
) -> tuple[QueryEvidenceRecord, ...]:
    return tuple(record for record in records if _is_citable(record))


def _is_citable(record: QueryEvidenceRecord) -> bool:
    return record.metadata.get("citable") is True and not record.is_error


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    marker = "\n[truncated]"
    if max_chars <= len(marker):
        return text[:max_chars], True
    return text[: max_chars - len(marker)].rstrip() + marker, True


def _max_steps(request: QueryRequest, configured_maximum: int) -> int:
    value = request.options.get("max_steps")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return min(value, configured_maximum)
    return configured_maximum


def _tool_calling_model(
    context: QueryContext,
    name: str | None,
) -> ToolCallingLanguageModelProtocol:
    component = context.recipe.get_component(model_ref("language", name))
    if not isinstance(component, ToolCallingLanguageModelProtocol):
        raise TypeError(
            "models.language must satisfy ToolCallingLanguageModelProtocol "
            "for agentic query modes"
        )
    return component


def _issue(code: str, message: str, **details: object) -> dict[str, object]:
    return {"code": code, "message": message, **details}


def _trace(
    events: list[QueryTraceEvent],
    *,
    request: QueryRequest,
    stage: str,
    message: str,
    metadata: Mapping[str, object],
) -> None:
    if request.trace:
        events.append(QueryTraceEvent(stage=stage, message=message, metadata=metadata))


@dataclass
class _UsageAccumulator:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reported: bool = False

    def add(self, usage: TokenUsage | None) -> None:
        if usage is None:
            return
        self.reported = True
        self.prompt_tokens += usage.prompt_tokens or 0
        self.completion_tokens += usage.completion_tokens or 0
        self.total_tokens += usage.total_tokens or 0

    def to_dict(self) -> dict[str, int] | None:
        if not self.reported:
            return None
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
