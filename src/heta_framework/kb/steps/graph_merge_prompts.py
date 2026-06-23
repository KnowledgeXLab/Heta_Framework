"""LLM prompt helpers for Heta-style graph merging."""

from __future__ import annotations

import json
from typing import Any

from heta_framework.common.models import ModelOptions, ModelRequest
from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.kb.steps.types import IssueResolution, IssueSubject, StepIssue


async def invoke_graph_merge_json(
    language_model: LanguageModelProtocol,
    prompt: str,
    *,
    step_name: str,
    temperature: float,
    max_retries: int,
    subject: IssueSubject,
    issues: list[StepIssue],
) -> dict[str, Any]:
    """Invoke an LLM and return a JSON object, recording non-fatal failures."""
    last_error = ""
    for _ in range(max_retries):
        try:
            result = await language_model.invoke(
                ModelRequest(
                    prompt=prompt,
                    options=ModelOptions(
                        response_format={"type": "json_object"},
                        temperature=temperature,
                    ),
                    trace_context={"step": step_name},
                )
            )
            if isinstance(result.parsed, dict):
                return result.parsed
            if result.text:
                parsed = json.loads(result.text)
                if isinstance(parsed, dict):
                    return parsed
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
    issues.append(
        StepIssue(
            step=step_name,
            subject=subject,
            code="llm_merge_failed",
            message="LLM did not return a valid graph merge decision.",
            resolution=IssueResolution(
                action="kept_new_record",
                outcome="The current graph fact was inserted without historical merge.",
            ),
            details={"error": last_error or "invalid_json"},
        )
    )
    return {}


def entity_merge_prompt(records: list[dict[str, Any]]) -> str:
    """Return the LLM prompt for entity merge decisions."""
    return (
        "You merge duplicate knowledge graph entities.\n"
        "Return strict JSON with keys entity_list and mapping_table.\n"
        "Only set merge_tag=true when records refer to the same real entity.\n"
        "mapping_table maps the canonical NodeName to original NodeName values.\n"
        "Input entities:\n"
        f"{json.dumps(records, ensure_ascii=False, indent=2)}"
    )


def relation_merge_prompt(records: list[dict[str, Any]]) -> str:
    """Return the LLM prompt for relation merge decisions."""
    return (
        "You merge duplicate knowledge graph relations.\n"
        "Return strict JSON with keys relation_list and mapping_table.\n"
        "Only set merge_tag=true when records express the same relation.\n"
        "mapping_table maps canonical 'Node1||Node2' to original relation ids or 'Node1||Node2' values.\n"
        "Input relations:\n"
        f"{json.dumps(records, ensure_ascii=False, indent=2)}"
    )


def mapping_values(mapping: dict[Any, Any]) -> set[str]:
    """Return normalized values mentioned by a HetaDB-style mapping_table."""
    values: set[str] = set()
    for key, raw_values in mapping.items():
        values.add(normalize_merge_key(str(key)))
        if not isinstance(raw_values, list):
            continue
        for value in raw_values:
            values.add(normalize_merge_key(str(value)))
    return values


def normalize_merge_key(value: str) -> str:
    """Normalize an LLM mapping key for matching."""
    return " ".join(value.strip().lower().split())
