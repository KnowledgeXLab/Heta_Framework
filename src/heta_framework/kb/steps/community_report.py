"""Shared community report generation utilities."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from heta_framework.common.models import ModelOptions, ModelRequest
from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.common.stores.graph import GraphEdge, GraphNode
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommunitySchema:
    """Graph community structure shared by GraphRAG and HiRAG."""

    community_id: str
    level: int
    title: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    chunk_ids: tuple[str, ...]
    occurrence: float
    sub_communities: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommunityReport:
    """One generated community report."""

    community_id: str
    level: int
    title: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    chunk_ids: tuple[str, ...]
    occurrence: float
    report: str
    sub_communities: tuple[str, ...] = ()
    report_json: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CommunityReportGenerationConfig:
    """Options for shared community report generation."""

    prompt_template: str
    step_name: str
    report_context_max_tokens: int = 12000
    max_nodes_per_report: int = 80
    max_edges_per_report: int = 120
    report_max_output_tokens: int = 800
    temperature: float = 0.0
    response_format: Mapping[str, Any] | None = None
    provider_options: Mapping[str, Any] | None = None


class CommunityReportGraphAdapter(Protocol):
    """Graph data access needed by community report packing."""

    async def get_node(self, node_id: str) -> GraphNode | None:
        """Return one graph node by id."""

    async def get_edge(self, source_id: str, target_id: str) -> GraphEdge | None:
        """Return one graph edge by endpoints."""


async def generate_community_reports(
    communities: tuple[CommunitySchema, ...],
    *,
    graph_adapter: CommunityReportGraphAdapter,
    language_model: LanguageModelProtocol,
    config: CommunityReportGenerationConfig,
) -> tuple[CommunityReport, ...]:
    """Generate reports level by level so parent communities can use child reports."""
    community_datas: dict[str, dict[str, Any]] = {}
    for level in sorted({community.level for community in communities}, reverse=True):
        level_communities = [
            community for community in communities if community.level == level
        ]
        level_reports = await asyncio.gather(
            *(
                _form_single_community_report(
                    community,
                    already_reports=community_datas,
                    graph_adapter=graph_adapter,
                    language_model=language_model,
                    config=config,
                )
                for community in level_communities
            )
        )
        for community, report_json in zip(level_communities, level_reports, strict=True):
            community_datas[community.community_id] = {
                "report_string": community_report_json_to_str(report_json),
                "report_json": report_json,
                "level": community.level,
                "title": str(report_json.get("title") or community.title),
                "nodes": list(community.nodes),
                "edges": [list(edge) for edge in community.edges],
                "chunk_ids": list(community.chunk_ids),
                "occurrence": community.occurrence,
                "sub_communities": list(community.sub_communities),
            }

    return tuple(
        CommunityReport(
            community_id=community_id,
            level=int(data["level"]),
            title=str(data["title"]),
            nodes=tuple(str(node) for node in data["nodes"]),
            edges=tuple((str(edge[0]), str(edge[1])) for edge in data["edges"]),
            chunk_ids=tuple(str(chunk_id) for chunk_id in data["chunk_ids"]),
            occurrence=float(data["occurrence"]),
            sub_communities=tuple(str(item) for item in data["sub_communities"]),
            report_json=data["report_json"],
            report=str(data["report_string"]),
        )
        for community_id, data in sorted(community_datas.items())
    )


async def put_community_report(
    object_store: ObjectStoreProtocol,
    *,
    prefix: str,
    report: CommunityReport,
) -> str:
    """Persist one community report to object storage."""
    key = join_object_key(prefix, f"{_stable_object_id(report.community_id)}.json")
    await object_store.put(
        key,
        json.dumps(
            community_report_to_dict(report),
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8"),
    )
    return key


def community_report_to_dict(report: CommunityReport) -> dict[str, Any]:
    """Convert a community report to a stable artifact payload."""
    return {
        "community_id": report.community_id,
        "level": report.level,
        "title": report.title,
        "nodes": list(report.nodes),
        "edges": [list(edge) for edge in report.edges],
        "chunk_ids": list(report.chunk_ids),
        "occurrence": report.occurrence,
        "sub_communities": list(report.sub_communities),
        "report_json": dict(report.report_json or {}),
        "report": report.report,
    }


def community_schema_from_mapping(
    communities: Mapping[str, Mapping[str, Any]],
) -> tuple[CommunitySchema, ...]:
    """Convert mapping-based community schemas to the shared dataclass."""
    schemas: list[CommunitySchema] = []
    for community_id, data in sorted(communities.items()):
        schemas.append(
            CommunitySchema(
                community_id=community_id,
                level=int(data.get("level", 0)),
                title=str(data.get("title", f"Community {community_id}")),
                nodes=tuple(str(node) for node in data.get("nodes", ())),
                edges=tuple(
                    (str(edge[0]), str(edge[1]))
                    for edge in data.get("edges", ())
                    if isinstance(edge, list | tuple) and len(edge) == 2
                ),
                chunk_ids=tuple(str(chunk_id) for chunk_id in data.get("chunk_ids", ())),
                occurrence=float(data.get("occurrence", 0.0)),
                sub_communities=tuple(
                    str(community_id) for community_id in data.get("sub_communities", ())
                ),
            )
        )
    return tuple(schemas)


def community_schema_to_dict(schema: CommunitySchema) -> dict[str, Any]:
    """Convert a community schema to a JSON-compatible dict."""
    return {
        "community_id": schema.community_id,
        "level": schema.level,
        "title": schema.title,
        "nodes": list(schema.nodes),
        "edges": [list(edge) for edge in schema.edges],
        "chunk_ids": list(schema.chunk_ids),
        "occurrence": schema.occurrence,
        "sub_communities": list(schema.sub_communities),
    }


async def _form_single_community_report(
    community: CommunitySchema,
    *,
    already_reports: Mapping[str, Mapping[str, Any]],
    graph_adapter: CommunityReportGraphAdapter,
    language_model: LanguageModelProtocol,
    config: CommunityReportGenerationConfig,
) -> dict[str, Any]:
    prompt_overhead = len(_encode_tokens(str(config.prompt_template.format(input_text=""))))
    description = str(
        await _pack_single_community_describe(
            community,
            graph_adapter=graph_adapter,
            max_token_size=max(0, config.report_context_max_tokens - prompt_overhead - 200),
            already_reports=already_reports,
            config=config,
        )
    )
    result = await language_model.invoke(
        ModelRequest(
            prompt=str(config.prompt_template.format(input_text=description)),
            options=ModelOptions(
                temperature=config.temperature,
                max_output_tokens=config.report_max_output_tokens,
                response_format=config.response_format,
                provider_options=config.provider_options,
            ),
            trace_context={
                "step": config.step_name,
                "stage": "community_report",
                "community_id": community.community_id,
            },
        )
    )
    return parse_community_report_json(result.text)


async def _pack_single_community_describe(
    community: CommunitySchema,
    *,
    graph_adapter: CommunityReportGraphAdapter,
    max_token_size: int,
    already_reports: Mapping[str, Mapping[str, Any]],
    config: CommunityReportGenerationConfig,
) -> str:
    nodes_in_order = sorted(community.nodes)
    edges_in_order = sorted(community.edges, key=lambda edge: edge[0] + edge[1])
    nodes_data = await asyncio.gather(
        *(graph_adapter.get_node(node_id) for node_id in nodes_in_order)
    )
    edges_data = await asyncio.gather(
        *(graph_adapter.get_edge(source, target) for source, target in edges_in_order)
    )

    final_template = (
        "-----Reports-----\n"
        "```csv\n{reports}\n```\n"
        "-----Entities-----\n"
        "```csv\n{entities}\n```\n"
        "-----Relationships-----\n"
        "```csv\n{relationships}\n```"
    )
    base_tokens = len(
        _encode_tokens(final_template.format(reports="", entities="", relationships=""))
    )
    remaining_budget = max(0, max_token_size - base_tokens)

    report_describe = ""
    contain_nodes: set[str] = set()
    contain_edges: set[tuple[str, str]] = set()
    truncated = (
        len(nodes_in_order) > config.max_nodes_per_report
        or len(edges_in_order) > config.max_edges_per_report
    )
    if truncated and community.sub_communities and already_reports:
        report_describe, report_size, contain_nodes, contain_edges = (
            _pack_single_community_by_sub_communities(
                community,
                remaining_budget,
                already_reports,
            )
        )
        remaining_budget = max(0, remaining_budget - report_size)

    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    node_degrees = _node_degrees(nodes_in_order, edges_in_order)
    edge_degrees = _edge_degrees(edges_in_order, node_degrees)

    nodes_list_data = [
        [
            index,
            name,
            _node_entity_type(data),
            _node_description(data),
            node_degrees.get(name, 0),
        ]
        for index, (name, data) in enumerate(zip(nodes_in_order, nodes_data))
        if name not in contain_nodes
    ]
    edges_list_data = [
        [
            index,
            edge[0],
            edge[1],
            _edge_description(data),
            edge_degrees.get(edge, 0),
        ]
        for index, (edge, data) in enumerate(zip(edges_in_order, edges_data))
        if edge not in contain_edges
    ]
    nodes_list_data.sort(key=lambda row: row[-1], reverse=True)
    edges_list_data.sort(key=lambda row: row[-1], reverse=True)

    header_tokens = len(
        _encode_tokens(
            _list_of_list_to_csv([node_fields])
            + "\n"
            + _list_of_list_to_csv([edge_fields])
        )
    )
    data_budget = max(0, remaining_budget - header_tokens)
    total_items = len(nodes_list_data) + len(edges_list_data)
    node_ratio = len(nodes_list_data) / max(1, total_items)
    edge_ratio = 1 - node_ratio
    nodes_final = _truncate_list_by_token_size(
        nodes_list_data,
        key=_csv_row,
        max_token_size=int(data_budget * node_ratio),
    )[: config.max_nodes_per_report]
    edges_final = _truncate_list_by_token_size(
        edges_list_data,
        key=_csv_row,
        max_token_size=int(data_budget * edge_ratio),
    )[: config.max_edges_per_report]

    return final_template.format(
        reports=report_describe,
        entities=_list_of_list_to_csv([node_fields] + nodes_final),
        relationships=_list_of_list_to_csv([edge_fields] + edges_final),
    )


def _pack_single_community_by_sub_communities(
    community: CommunitySchema,
    max_token_size: int,
    already_reports: Mapping[str, Mapping[str, Any]],
) -> tuple[str, int, set[str], set[tuple[str, str]]]:
    sub_communities = [
        already_reports[key]
        for key in community.sub_communities
        if key in already_reports
    ]
    sub_communities = sorted(
        sub_communities,
        key=lambda item: float(item.get("occurrence", 0.0)),
        reverse=True,
    )
    truncated = _truncate_list_by_token_size(
        sub_communities,
        key=lambda item: str(item.get("report_string", "")),
        max_token_size=max_token_size,
    )
    fields = ["id", "report", "rating", "importance"]
    description = _list_of_list_to_csv(
        [fields]
        + [
            [
                index,
                community_data.get("report_string", ""),
                community_data.get("report_json", {}).get("rating", -1),
                community_data.get("occurrence", 0.0),
            ]
            for index, community_data in enumerate(truncated)
        ]
    )

    already_nodes: list[str] = []
    already_edges: list[tuple[str, str]] = []
    for community_data in truncated:
        already_nodes.extend(str(node) for node in community_data.get("nodes", ()))
        already_edges.extend(
            (str(edge[0]), str(edge[1]))
            for edge in community_data.get("edges", ())
            if isinstance(edge, list | tuple) and len(edge) == 2
        )
    return (
        description,
        len(_encode_tokens(description)),
        set(already_nodes),
        set(already_edges),
    )


def parse_community_report_json(text: str) -> dict[str, Any]:
    """Parse an LLM community report response."""
    value = _convert_response_to_json(text)
    if not isinstance(value, dict):
        return {
            "title": "Community Report",
            "summary": str(value),
            "rating": -1,
            "rating_explanation": "",
            "findings": [],
        }
    return value


def _extract_first_complete_json(text: str) -> Any | None:
    """Extract the first complete JSON object from text."""
    stack: list[int] = []
    first_json_start: int | None = None

    for index, char in enumerate(text):
        if char == "{":
            stack.append(index)
            if first_json_start is None:
                first_json_start = index
        elif char == "}" and stack:
            stack.pop()
            if not stack and first_json_start is not None:
                first_json = text[first_json_start : index + 1]
                try:
                    return json.loads(first_json.replace("\n", ""))
                except json.JSONDecodeError as exc:
                    logger.debug(
                        "JSON decoding failed while parsing community report: %s. "
                        "Attempted string: %s...",
                        exc,
                        first_json[:50],
                    )
                    return None
                finally:
                    first_json_start = None
    logger.debug("No complete JSON object found in community report response.")
    return None


def _parse_relaxed_json_value(value: str) -> Any:
    """Convert a relaxed JSON scalar string into a Python value."""
    value = value.strip()
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"')


def _extract_values_from_relaxed_json(text: str) -> dict[str, Any]:
    """Extract key values from a non-standard or malformed JSON object string."""
    extracted_values: dict[str, Any] = {}
    regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'

    for match in re.finditer(regex_pattern, text, re.DOTALL):
        key = match.group("key").strip('"')
        value = match.group("value").strip()
        if value.startswith("{") and value.endswith("}"):
            extracted_values[key] = _extract_values_from_relaxed_json(value)
        else:
            extracted_values[key] = _parse_relaxed_json_value(value)

    if not extracted_values:
        logger.debug("No values could be extracted from community report response.")
    return extracted_values


def _convert_response_to_json(response: str) -> dict[str, Any]:
    """Convert response text to JSON with fallback relaxed extraction."""
    parsed = _extract_first_complete_json(response)
    if parsed is None:
        logger.debug("Attempting relaxed JSON extraction for community report response.")
        parsed = _extract_values_from_relaxed_json(response)
    if not isinstance(parsed, dict):
        logger.debug("Unable to extract meaningful JSON data from community report response.")
        return {}
    return parsed


def community_report_json_to_str(parsed_output: Mapping[str, Any]) -> str:
    """Render parsed community report JSON as a plain report string."""
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: Any) -> str:
        if isinstance(finding, str):
            return finding
        if isinstance(finding, Mapping):
            return str(finding.get("summary", ""))
        return ""

    def finding_explanation(finding: Any) -> str:
        if isinstance(finding, str):
            return ""
        if isinstance(finding, Mapping):
            return str(finding.get("explanation", ""))
        return ""

    report_sections = "\n\n".join(
        f"## {finding_summary(finding)}\n\n{finding_explanation(finding)}"
        for finding in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}".strip()


def _node_degrees(
    node_ids: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> dict[str, int]:
    degrees = {node_id: 0 for node_id in node_ids}
    for source, target in edges:
        degrees[source] = degrees.get(source, 0) + 1
        degrees[target] = degrees.get(target, 0) + 1
    return degrees


def _edge_degrees(
    edges: tuple[tuple[str, str], ...],
    node_degrees: Mapping[str, int],
) -> dict[tuple[str, str], int]:
    return {
        edge: node_degrees.get(edge[0], 0) + node_degrees.get(edge[1], 0)
        for edge in edges
    }


def _node_entity_type(node: GraphNode | None) -> str:
    if node is None:
        return "UNKNOWN"
    return str(node.properties.get("entity_type") or "UNKNOWN")


def _node_description(node: GraphNode | None) -> str:
    if node is None:
        return "UNKNOWN"
    return str(node.properties.get("description") or "UNKNOWN")


def _edge_description(edge: GraphEdge | None) -> str:
    if edge is None:
        return "UNKNOWN"
    return str(edge.properties.get("description") or "UNKNOWN")


def _csv_row(row: list[Any]) -> str:
    return ",".join(str(item) for item in row)


def _list_of_list_to_csv(rows: list[list[Any]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(rows)
    return output.getvalue().strip()


def _truncate_list_by_token_size(
    items: list[Any],
    *,
    key: Any,
    max_token_size: int,
) -> list[Any]:
    if max_token_size <= 0:
        return []
    result: list[Any] = []
    total_tokens = 0
    for item in items:
        token_count = len(_encode_tokens(str(key(item))))
        if total_tokens + token_count > max_token_size:
            break
        result.append(item)
        total_tokens += token_count
    return result


def _encode_tokens(text: str) -> list[str]:
    return text.split()


def _stable_object_id(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest[:32]
