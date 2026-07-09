"""Generate community reports from a clusterable graph store."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
from dataclasses import dataclass
from typing import Any, Mapping

from heta_framework.common.models import ModelOptions, ModelRequest
from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.common.stores.graph import (
    ClusterableGraphStoreProtocol,
    GraphEdge,
    GraphNode,
)
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb.cleanup import StepCleanupPlan, object_key_targets
from heta_framework.kb.graphing.prompts import GRAPH_RAG_COMMUNITY_REPORT_PROMPT
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref


@dataclass(frozen=True)
class GraphCommunityConfig:
    """Configuration for GraphCommunity."""

    object_store: str | None = None
    graph_store: str | None = None
    language_model: str | None = None
    graph_cluster_algorithm: str = "leiden"
    community_reports_prefix: str = "graph/community_reports"
    community_reports_artifact: str = "community_reports"
    community_report_keys_artifact: str = "community_report_keys"
    graph_community_result_artifact: str = "graph_community_result"
    report_context_max_tokens: int = 12000
    max_nodes_per_report: int = 80
    max_edges_per_report: int = 120
    report_max_output_tokens: int = 800
    temperature: float = 0.0

    def __post_init__(self) -> None:
        validate_object_prefix(self.community_reports_prefix)
        if self.graph_cluster_algorithm.strip() == "":
            raise ValueError("graph_cluster_algorithm must not be empty")
        if self.community_reports_artifact.strip() == "":
            raise ValueError("community_reports_artifact must not be empty")
        if self.community_report_keys_artifact.strip() == "":
            raise ValueError("community_report_keys_artifact must not be empty")
        if self.graph_community_result_artifact.strip() == "":
            raise ValueError("graph_community_result_artifact must not be empty")
        if self.report_context_max_tokens <= 0:
            raise ValueError("report_context_max_tokens must be greater than zero")
        if self.max_nodes_per_report <= 0:
            raise ValueError("max_nodes_per_report must be greater than zero")
        if self.max_edges_per_report <= 0:
            raise ValueError("max_edges_per_report must be greater than zero")
        if self.report_max_output_tokens <= 0:
            raise ValueError("report_max_output_tokens must be greater than zero")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")


@dataclass(frozen=True)
class CommunityReport:
    """One generated graph community report."""

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
class GraphCommunityResult:
    """Artifacts produced by GraphCommunity."""

    community_count: int
    report_ids: tuple[str, ...]
    report_keys: tuple[str, ...]


@dataclass(frozen=True)
class _CommunitySchema:
    community_id: str
    level: int
    title: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    chunk_ids: tuple[str, ...]
    occurrence: float
    sub_communities: tuple[str, ...]


class GraphCommunity:
    """Run graph clustering and generate LLM community reports."""

    name = "graph_community"

    def __init__(self, config: GraphCommunityConfig | None = None) -> None:
        self.config = config or GraphCommunityConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components required by this step."""
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("objects", self.config.object_store),
                    store_ref("graph", self.config.graph_store),
                    model_ref("language", self.config.language_model),
                }
            )
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts produced by this step."""
        return StepCapabilities(
            artifacts=frozenset(
                {
                    self.config.graph_community_result_artifact,
                    self.config.community_reports_artifact,
                    self.config.community_report_keys_artifact,
                }
            )
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        """Return persisted community report objects produced by this step."""
        return StepCleanupPlan(
            object_key_targets(
                artifacts,
                self.config.community_report_keys_artifact,
                component=store_ref("objects", self.config.object_store).key,
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Cluster the graph store and generate community reports."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        graph_store = _require_clusterable_graph_store(
            context.get_component(store_ref("graph", self.config.graph_store).key)
        )
        language_model = _require_language_model(
            context.get_component(model_ref("language", self.config.language_model).key)
        )

        await graph_store.clustering(self.config.graph_cluster_algorithm)
        communities = _community_schemas_from_mapping(await graph_store.community_schema())
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
                        graph_store=graph_store,
                        language_model=language_model,
                        config=self.config,
                    )
                    for community in level_communities
                )
            )
            for community, report_json in zip(level_communities, level_reports, strict=True):
                community_datas[community.community_id] = {
                    "report_string": _community_report_json_to_str(report_json),
                    "report_json": report_json,
                    "level": community.level,
                    "title": str(report_json.get("title") or community.title),
                    "nodes": list(community.nodes),
                    "edges": [list(edge) for edge in community.edges],
                    "chunk_ids": list(community.chunk_ids),
                    "occurrence": community.occurrence,
                    "sub_communities": list(community.sub_communities),
                }

        reports = tuple(
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
        report_keys = tuple(
            [
                await _put_community_report(object_store, self.config, report)
                for report in reports
            ]
        )

        result = GraphCommunityResult(
            community_count=len(reports),
            report_ids=tuple(report.community_id for report in reports),
            report_keys=report_keys,
        )
        context.set_artifact(
            self.config.community_reports_artifact,
            tuple(_community_report_to_dict(report) for report in reports),
        )
        context.set_artifact(self.config.community_report_keys_artifact, result.report_keys)
        context.set_artifact(self.config.graph_community_result_artifact, result)


def _community_schemas_from_mapping(
    communities: Mapping[str, Mapping[str, Any]],
) -> tuple[_CommunitySchema, ...]:
    schemas: list[_CommunitySchema] = []
    for community_id, data in sorted(communities.items()):
        schemas.append(
            _CommunitySchema(
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


async def _form_single_community_report(
    community: _CommunitySchema,
    *,
    already_reports: Mapping[str, Mapping[str, Any]],
    graph_store: ClusterableGraphStoreProtocol,
    language_model: LanguageModelProtocol,
    config: GraphCommunityConfig,
) -> dict[str, Any]:
    prompt_overhead = len(
        _encode_tokens(str(GRAPH_RAG_COMMUNITY_REPORT_PROMPT.format(input_text="")))
    )
    description = str(await _pack_single_community_describe(
        community,
        graph_store=graph_store,
        max_token_size=max(0, config.report_context_max_tokens - prompt_overhead - 200),
        already_reports=already_reports,
        config=config,
    ))
    result = await language_model.invoke(
        ModelRequest(
            prompt=str(GRAPH_RAG_COMMUNITY_REPORT_PROMPT.format(input_text=description)),
            options=ModelOptions(
                temperature=config.temperature,
                max_output_tokens=config.report_max_output_tokens,
                provider_options={
              # 按具体 provider 的参数写
              "extra_body": {
                  "enable_thinking": False,
              }}
            ),
            trace_context={
                "step": GraphCommunity.name,
                "community_id": community.community_id,
            },
        )
    )
    return _parse_community_report_json(result.text)


async def _pack_single_community_describe(
    community: _CommunitySchema,
    *,
    graph_store: ClusterableGraphStoreProtocol,
    max_token_size: int,
    already_reports: Mapping[str, Mapping[str, Any]],
    config: GraphCommunityConfig,
) -> str:
    nodes_in_order = sorted(community.nodes)
    edges_in_order = sorted(community.edges, key=lambda edge: edge[0] + edge[1])
    nodes_data = await asyncio.gather(
        *(graph_store.get_node(node_id) for node_id in nodes_in_order)
    )
    edges_data = await asyncio.gather(
        *(
            _get_edge_by_endpoints(graph_store, source, target)
            for source, target in edges_in_order
        )
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
    community: _CommunitySchema,
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


async def _get_edge_by_endpoints(
    graph_store: ClusterableGraphStoreProtocol,
    source_id: str,
    target_id: str,
) -> GraphEdge | None:
    direct = await graph_store.get_edge(f"{source_id}--RELATED--{target_id}")
    if direct is not None:
        return direct
    return await graph_store.get_edge(f"{target_id}--RELATED--{source_id}")


def _describe_node(node: GraphNode) -> str:
    return (
        "- "
        f"{node.id} | type={node.properties.get('entity_type', '')} | "
        f"description={node.properties.get('description', '')}"
    )


def _describe_edge(edge: GraphEdge) -> str:
    return (
        "- "
        f"{edge.source_id} -> {edge.target_id} | "
        f"weight={edge.properties.get('weight', '')} | "
        f"description={edge.properties.get('description', '')}"
    )


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


def _parse_community_report_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return {
            "title": "Community Report",
            "summary": stripped,
            "rating": -1,
            "rating_explanation": "",
            "findings": [],
        }
    if not isinstance(value, dict):
        return {
            "title": "Community Report",
            "summary": str(value),
            "rating": -1,
            "rating_explanation": "",
            "findings": [],
        }
    return value


def _community_report_json_to_str(parsed_output: Mapping[str, Any]) -> str:
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


async def _put_community_report(
    object_store: ObjectStoreProtocol,
    config: GraphCommunityConfig,
    report: CommunityReport,
) -> str:
    key = join_object_key(
        config.community_reports_prefix,
        f"{_stable_object_id(report.community_id)}.json",
    )
    await object_store.put(
        key,
        json.dumps(
            _community_report_to_dict(report),
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8"),
    )
    return key


def _community_report_to_dict(report: CommunityReport) -> dict[str, Any]:
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


def _stable_object_id(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest[:32]


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_clusterable_graph_store(component: object) -> ClusterableGraphStoreProtocol:
    if not isinstance(component, ClusterableGraphStoreProtocol):
        raise TypeError("stores.graph must satisfy ClusterableGraphStoreProtocol")
    return component


def _require_language_model(component: object) -> LanguageModelProtocol:
    if not isinstance(component, LanguageModelProtocol):
        raise TypeError("models.language must satisfy LanguageModelProtocol")
    return component
