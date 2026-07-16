"""Helpers for built-in query result provenance."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from heta_framework.kb.search.types import QueryCitation, QueryResult


def chunk_source(
    *,
    document_id: object | None = None,
    document_ids: Iterable[object] = (),
    object_key: object | None = None,
    object_keys: Iterable[object] = (),
    object_name: object | None = None,
    object_type: object | None = None,
    chunk_ids: Iterable[object] = (),
    page_index: object | None = None,
    chunk_index: object | None = None,
    token_start: object | None = None,
    token_end: object | None = None,
    evidence_count: object | None = None,
) -> dict[str, Any]:
    """Return the canonical source shape used by built-in query engines."""
    source: dict[str, Any] = {}
    _set_if_present(source, "document_id", document_id)
    doc_ids = tuple(str(item) for item in document_ids if item is not None and str(item).strip())
    if doc_ids:
        source["document_ids"] = doc_ids
    _set_if_present(source, "object_key", object_key)
    keys = tuple(str(item) for item in object_keys if item is not None and str(item).strip())
    if keys:
        source["object_keys"] = keys
        source["source_keys"] = keys
    _set_if_present(source, "object_name", object_name)
    _set_if_present(source, "object_type", object_type)
    ids = tuple(str(item) for item in chunk_ids if item is not None and str(item).strip())
    if ids:
        source["chunk_ids"] = ids
    _set_if_present(source, "page_index", page_index)
    _set_if_present(source, "chunk_index", chunk_index)
    _set_if_present(source, "token_start", token_start)
    _set_if_present(source, "token_end", token_end)
    _set_if_present(source, "evidence_count", evidence_count)
    return source


def chunk_source_from_metadata(metadata: Mapping[str, object], *, chunk_id: str) -> dict[str, Any]:
    """Build canonical chunk source metadata from vector-store metadata."""
    return chunk_source(
        document_id=metadata.get("document_id"),
        object_key=metadata.get("source_key"),
        object_name=metadata.get("source_name"),
        object_type=metadata.get("source_file_type"),
        chunk_ids=(chunk_id,),
        page_index=metadata.get("page_index"),
        chunk_index=metadata.get("chunk_index"),
        token_start=metadata.get("token_start"),
        token_end=metadata.get("token_end"),
    )


def citations_from_results(results: Iterable[QueryResult]) -> tuple[QueryCitation, ...]:
    """Create stable citations from query results without changing result semantics."""
    citations: list[QueryCitation] = []
    seen: set[tuple[str, str]] = set()
    for index, result in enumerate(results, start=1):
        key = (result.kind, result.id)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            QueryCitation(
                id=f"citation_{index}",
                result_id=result.id,
                source=result.source,
                text=result.text,
                metadata={"result_kind": result.kind},
            )
        )
    return tuple(citations)


def _set_if_present(target: dict[str, Any], key: str, value: object | None) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    target[key] = value
