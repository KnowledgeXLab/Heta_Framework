"""Answer prompts used by built-in query engines."""

from __future__ import annotations

from heta_framework.kb.search.engines._language import numbered_context
from heta_framework.kb.search.types import QueryResult


def vector_answer_prompt(query: str, results: tuple[QueryResult, ...]) -> str:
    """Return the answer prompt for chunk vector search."""
    return _prompt(
        instruction=(
            "Answer the user question using only the retrieved text chunks. "
            "Use inline citation markers like [1] when referring to evidence. "
            "If the evidence is insufficient, say so."
        ),
        context_label="Retrieved evidence",
        query=query,
        results=results,
    )


def keyword_answer_prompt(query: str, results: tuple[QueryResult, ...]) -> str:
    """Return the answer prompt for keyword search."""
    return _prompt(
        instruction=(
            "Answer the user question using only the keyword-matched text chunks. "
            "Use inline citation markers like [1] when referring to evidence. "
            "If the evidence is insufficient, say so."
        ),
        context_label="Retrieved evidence",
        query=query,
        results=results,
    )


def graph_answer_prompt(query: str, results: tuple[QueryResult, ...]) -> str:
    """Return the answer prompt for Heta graph search."""
    return _prompt(
        instruction=(
            "Answer the user question using only the graph facts and their evidence. "
            "Explain relevant entities and relations when they matter. "
            "Use inline citation markers like [1] when referring to evidence. "
            "If the graph evidence is insufficient, say so."
        ),
        context_label="Retrieved graph evidence",
        query=query,
        results=results,
    )


def hybrid_answer_prompt(query: str, results: tuple[QueryResult, ...]) -> str:
    """Return the answer prompt for hybrid vector and graph search."""
    return _prompt(
        instruction=(
            "Answer the user question using only the retrieved chunk evidence and graph facts. "
            "Prefer evidence that is directly relevant and reconcile overlapping evidence. "
            "Use inline citation markers like [1] when referring to evidence. "
            "If the evidence is insufficient, say so."
        ),
        context_label="Retrieved evidence",
        query=query,
        results=results,
    )


def rerank_answer_prompt(query: str, results: tuple[QueryResult, ...]) -> str:
    """Return the answer prompt for reranked evidence search."""
    return _prompt(
        instruction=(
            "Answer the user question using only the reranked evidence. "
            "Prioritize higher-ranked evidence and use inline citation markers like [1]. "
            "If the evidence is insufficient, say so."
        ),
        context_label="Retrieved evidence",
        query=query,
        results=results,
    )


def rewrite_answer_prompt(query: str, results: tuple[QueryResult, ...]) -> str:
    """Return the answer prompt for query rewrite search."""
    return _prompt(
        instruction=(
            "Answer the user question using only the evidence retrieved from rewritten "
            "query variants. Synthesize the evidence without mentioning internal query "
            "rewriting. Use inline citation markers like [1] when referring to evidence. "
            "If the evidence is insufficient, say so."
        ),
        context_label="Retrieved evidence",
        query=query,
        results=results,
    )


def _prompt(
    *,
    instruction: str,
    context_label: str,
    query: str,
    results: tuple[QueryResult, ...],
) -> str:
    return (
        f"{instruction}\n\n"
        f"{context_label}:\n{numbered_context(results)}\n\n"
        f"User question: {query}"
    )
