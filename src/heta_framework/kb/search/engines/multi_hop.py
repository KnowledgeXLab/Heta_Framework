"""Multi-hop retrieval and reasoning search engine."""

from __future__ import annotations

from dataclasses import dataclass

from heta_framework.kb.search.assets import SearchAssetRef
from heta_framework.kb.search.engines._language import (
    answer_from_results,
    invoke_json,
    language_model_from_context,
    should_generate_answer,
)
from heta_framework.kb.search.engines._provenance import citations_from_results
from heta_framework.kb.search.engines._ranking import deduplicate_results
from heta_framework.kb.search.protocols import QueryContext
from heta_framework.kb.search.types import QueryRequest, QueryResponse, QueryResult, QueryTraceEvent
from heta_framework.kb.steps.types import ComponentRef, model_ref


@dataclass(frozen=True)
class MultiHopSearchEngine:
    """Iteratively retrieve evidence until the language model can answer."""

    mode: str = "heta_multihop_search"
    base_mode: str = "heta_rerank_search"
    language_model: str | None = None
    max_rounds: int = 3
    required_asset_refs: frozenset[SearchAssetRef] = frozenset(
        {
            SearchAssetRef(kind="chunk_vector_index"),
            SearchAssetRef(kind="chunk_full_text_index"),
            SearchAssetRef(kind="graph_tables"),
            SearchAssetRef(kind="graph_vector_index"),
        }
    )

    @property
    def required_assets(self) -> frozenset[SearchAssetRef]:
        """Return assets required by the configured base search mode."""
        return self.required_asset_refs

    @property
    def required_components(self) -> frozenset[ComponentRef]:
        """Return components required by multi-hop reasoning."""
        return frozenset({model_ref("language", self.language_model)})

    async def query(self, request: QueryRequest, context: QueryContext) -> QueryResponse:
        """Run a HetaDB-style retrieve, extract, and judge loop."""
        language_model = language_model_from_context(context, self.language_model)
        current_query = request.text
        gathered_info: list[str] = []
        gathered_results: list[QueryResult] = []
        trace_events: list[QueryTraceEvent] = []
        round_reports: list[dict[str, object]] = []
        issues: list[dict[str, object]] = []
        max_rounds = _max_rounds(request, self.max_rounds)
        generate_answer = should_generate_answer(request, default=True)

        for round_index in range(1, max_rounds + 1):
            base_response = await context.query(
                self.base_mode,
                QueryRequest(
                    text=current_query,
                    mode=self.base_mode,
                    top_k=request.top_k,
                    filters=request.filters,
                    options=request.options,
                    trace=request.trace,
                ),
            )
            gathered_results.extend(base_response.results)
            observation = _observation_text(base_response.results)
            useful_info = await _extract_useful_information(
                language_model,
                original_query=request.text,
                observation=observation,
                round_index=round_index,
            )
            if useful_info:
                gathered_info.append(useful_info)
            elif not base_response.results:
                issues.append(
                    {
                        "code": "round_no_results",
                        "message": "Base search returned no results for this round.",
                        "round": round_index,
                        "action": "continued",
                    }
                )
            else:
                issues.append(
                    {
                        "code": "round_no_useful_information",
                        "message": (
                            "The language model found no useful information "
                            "in the retrieved results."
                        ),
                        "round": round_index,
                        "action": "continued",
                    }
                )

            judgment = await _judge_answer(
                language_model,
                original_query=request.text,
                gathered_info=gathered_info,
                round_index=round_index,
            )
            round_reports.append(
                {
                    "round": round_index,
                    "query": current_query,
                    "result_count": len(base_response.results),
                    "extracted_information": bool(useful_info),
                    "answered": bool(judgment.get("judge")),
                    "next_query": judgment.get("next_query"),
                }
            )
            if request.trace:
                trace_events.extend(base_response.trace)
                trace_events.append(
                    QueryTraceEvent(
                        stage="heta_multihop_search",
                        message="Completed one retrieval and reasoning round.",
                        metadata={
                            "round": round_index,
                            "query": current_query,
                            "useful": bool(useful_info),
                            "answered": bool(judgment.get("judge")),
                        },
                    )
                )
            if judgment.get("judge") is True and isinstance(judgment.get("answer"), str):
                final_results = deduplicate_results(gathered_results)[: request.top_k]
                answer = str(judgment["answer"]).strip() if generate_answer else None
                return QueryResponse(
                    mode=self.mode,
                    results=final_results,
                    answer=answer,
                    citations=citations_from_results(final_results),
                    trace=tuple(trace_events),
                    metadata={
                        "base_mode": self.base_mode,
                        "rounds": round_index,
                        "round_reports": tuple(round_reports),
                        "issues": tuple(issues),
                        "answer_generation": "generated" if answer else "disabled",
                    },
                )

            next_query = judgment.get("next_query")
            if isinstance(next_query, str) and next_query.strip():
                current_query = next_query.strip()

        final_results = deduplicate_results(gathered_results)[: request.top_k]
        fallback_answer = ""
        if generate_answer:
            fallback_answer = await answer_from_results(
                language_model,
                query=request.text,
                results=final_results,
                trace_context={
                    "query_mode": "heta_multihop_search",
                    "stage": "fallback_answer",
                },
            )
        issues.append(
            {
                "code": "answer_not_confirmed",
                "message": "Maximum rounds completed without a confirmed answer.",
                "round": max_rounds,
                "action": "generated_fallback_answer" if generate_answer else "returned_results",
            }
        )
        return QueryResponse(
            mode=self.mode,
            results=final_results,
            answer=fallback_answer or None,
            citations=citations_from_results(final_results),
            trace=tuple(trace_events),
            metadata={
                "base_mode": self.base_mode,
                "rounds": max_rounds,
                "round_reports": tuple(round_reports),
                "issues": tuple(issues),
                "fallback": True,
                "answer_generation": "generated" if fallback_answer else "disabled",
            },
        )


async def _extract_useful_information(
    language_model: object,
    *,
    original_query: str,
    observation: str,
    round_index: int,
) -> str | None:
    data = await invoke_json(
        language_model,
        prompt=(
            "Decide whether the observation contains information useful for answering "
            "the original question. Return JSON only.\n\n"
            "Expected shape when useful: {\"usefulness\": true, \"information\": \"...\"}\n"
            "Expected shape when not useful: {\"usefulness\": false}\n\n"
            f"Original question: {original_query}\n\n"
            f"Observation:\n{observation}"
        ),
        trace_context={
            "query_mode": "heta_multihop_search",
            "round": round_index,
            "stage": "extract",
        },
    )
    if data.get("usefulness") is True and isinstance(data.get("information"), str):
        information = data["information"].strip()
        return information or None
    return None


async def _judge_answer(
    language_model: object,
    *,
    original_query: str,
    gathered_info: list[str],
    round_index: int,
) -> dict[str, object]:
    data = await invoke_json(
        language_model,
        prompt=(
            "Judge whether the accumulated information is sufficient to answer the "
            "original question. Return JSON only.\n\n"
            "If sufficient: {\"judge\": true, \"answer\": \"...\"}\n"
            "If insufficient: "
            "{\"judge\": false, \"next_query\": \"a better follow-up search query\"}\n\n"
            f"Original question: {original_query}\n\n"
            f"Accumulated information:\n{_joined_info(gathered_info)}"
        ),
        trace_context={
            "query_mode": "heta_multihop_search",
            "round": round_index,
            "stage": "judge",
        },
    )
    return data


def _observation_text(results: tuple[QueryResult, ...]) -> str:
    if not results:
        return "No relevant context was retrieved."
    return "\n\n".join(
        f"[{index}] {result.text}" for index, result in enumerate(results, start=1)
    )


def _joined_info(gathered_info: list[str]) -> str:
    if not gathered_info:
        return "No useful information has been extracted yet."
    return "\n".join(f"- {item}" for item in gathered_info)


def _max_rounds(request: QueryRequest, default: int) -> int:
    value = request.options.get("max_rounds")
    if isinstance(value, int) and value > 0:
        return value
    return default
