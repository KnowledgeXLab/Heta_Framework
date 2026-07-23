"""Generate HiRAG community reports from a built HiRAG graph schema."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.common.stores.graph import GraphEdge, GraphNode
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import validate_object_prefix
from heta_framework.common.stores.sql import SQLStoreProtocol
from heta_framework.kb.cleanup import StepCleanupPlan, object_key_targets
from heta_framework.kb.steps.build_hirag_graph import (
    HiRAGTableNames,
    _community_row,
    _filter_edges_with_known_endpoints,
    _upsert_community_rows,
)
from heta_framework.kb.steps.community_report import (
    CommunityReportGenerationConfig,
    CommunitySchema,
    community_report_to_dict,
    generate_community_reports,
    put_community_report,
)
from heta_framework.kb.steps.extract_hirag_graph import HIRAG_PROMPTS
from heta_framework.kb.steps.graph_storage import batches
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref


@dataclass(frozen=True)
class HiRAGCommunityConfig:
    """Configuration for HiRAGCommunity."""

    table_names: HiRAGTableNames = field(default_factory=HiRAGTableNames)
    graph_node_keys_artifact: str = "hi_rag_graph_node_keys"
    graph_edge_keys_artifact: str = "hi_rag_graph_edge_keys"
    community_schema_artifact: str = "hi_rag_community_schema"
    community_reports_artifact: str = "hi_rag_community_reports"
    community_report_keys_artifact: str = "hi_rag_community_report_keys"
    result_artifact: str = "hi_rag_community_result"
    community_reports_prefix: str = "hi_rag/community_reports"
    report_context_max_tokens: int = 12000
    max_nodes_per_report: int = 80
    max_edges_per_report: int = 120
    report_max_output_tokens: int = 800
    temperature: float = 0.0
    batch_size: int = 128
    object_store: str | None = None
    sql_store: str | None = None
    language_model: str | None = None
    prompts: Mapping[str, Any] = field(default_factory=lambda: dict(HIRAG_PROMPTS))

    def __post_init__(self) -> None:
        validate_object_prefix(self.community_reports_prefix)
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
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        for name in (
            self.graph_node_keys_artifact,
            self.graph_edge_keys_artifact,
            self.community_schema_artifact,
            self.community_reports_artifact,
            self.community_report_keys_artifact,
            self.result_artifact,
        ):
            if name.strip() == "":
                raise ValueError("artifact names must not be empty")


@dataclass(frozen=True)
class HiRAGCommunityResult:
    """Artifacts produced by HiRAGCommunity."""

    community_count: int
    report_ids: tuple[str, ...]
    report_keys: tuple[str, ...]


class HiRAGCommunity:
    """Generate and persist HiRAG community reports."""

    name = "hirag_community"

    def __init__(self, config: HiRAGCommunityConfig | None = None) -> None:
        self.config = config or HiRAGCommunityConfig()

    @property
    def requirements(self) -> StepRequirements:
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("objects", self.config.object_store),
                    store_ref("sql", self.config.sql_store),
                    model_ref("language", self.config.language_model),
                }
            ),
            artifacts=frozenset(
                {
                    self.config.graph_node_keys_artifact,
                    self.config.graph_edge_keys_artifact,
                    self.config.community_schema_artifact,
                }
            ),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        return StepCapabilities(
            artifacts=frozenset(
                {
                    self.config.result_artifact,
                    self.config.community_reports_artifact,
                    self.config.community_report_keys_artifact,
                }
            )
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        return StepCleanupPlan(
            object_key_targets(
                artifacts,
                self.config.community_report_keys_artifact,
                component=store_ref("objects", self.config.object_store).key,
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        sql_store = _require_sql_store(
            context.get_component(store_ref("sql", self.config.sql_store).key)
        )
        language_model = _require_language_model(
            context.get_component(model_ref("language", self.config.language_model).key)
        )

        node_keys = tuple(context.get_artifact(self.config.graph_node_keys_artifact))
        edge_keys = tuple(context.get_artifact(self.config.graph_edge_keys_artifact))
        nodes = [json.loads((await object_store.get(key)).decode("utf-8")) for key in node_keys]
        edges = [json.loads((await object_store.get(key)).decode("utf-8")) for key in edge_keys]
        edges = _filter_edges_with_known_endpoints(nodes, edges)
        communities = tuple(
            _community_schema_from_artifact(item)
            for item in context.get_artifact(self.config.community_schema_artifact)
        )

        reports = await generate_community_reports(
            communities,
            graph_adapter=_HiRAGCommunityReportAdapter(nodes, edges),
            language_model=language_model,
            config=CommunityReportGenerationConfig(
                prompt_template=str(self.config.prompts["community_report"]),
                step_name=self.name,
                report_context_max_tokens=self.config.report_context_max_tokens,
                max_nodes_per_report=self.config.max_nodes_per_report,
                max_edges_per_report=self.config.max_edges_per_report,
                report_max_output_tokens=self.config.report_max_output_tokens,
                temperature=self.config.temperature,
                response_format={"type": "json_object"},
            ),
        )
        report_keys = tuple(
            [
                await put_community_report(
                    object_store,
                    prefix=self.config.community_reports_prefix,
                    report=report,
                )
                for report in reports
            ]
        )

        community_rows = [_community_row(community_report_to_dict(report)) for report in reports]
        async with sql_store.transaction() as tx:
            for batch in batches(community_rows, self.config.batch_size):
                await _upsert_community_rows(tx, self.config.table_names.communities, batch)

        result = HiRAGCommunityResult(
            community_count=len(reports),
            report_ids=tuple(report.community_id for report in reports),
            report_keys=report_keys,
        )
        context.set_artifact(
            self.config.community_reports_artifact,
            tuple(community_report_to_dict(report) for report in reports),
        )
        context.set_artifact(self.config.community_report_keys_artifact, result.report_keys)
        context.set_artifact(self.config.result_artifact, result)


class _HiRAGCommunityReportAdapter:
    def __init__(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
        self.nodes = {str(node["id"]): node for node in nodes}
        self.normalized_nodes = {_normalized_entity_id(node["id"]): node for node in nodes}
        self.edges = {
            tuple(sorted((_normalized_entity_id(edge["source_id"]), _normalized_entity_id(edge["target_id"])))): edge
            for edge in edges
        }

    async def get_node(self, node_id: str) -> GraphNode | None:
        node = self.nodes.get(node_id) or self.normalized_nodes.get(_normalized_entity_id(node_id))
        if node is None:
            return None
        return GraphNode(
            id=str(node["id"]),
            labels=tuple(str(label) for label in node.get("labels", ("Entity",))),
            properties=dict(node.get("properties") or {}),
        )

    async def get_edge(self, source_id: str, target_id: str) -> GraphEdge | None:
        edge = self.edges.get(
            tuple(sorted((_normalized_entity_id(source_id), _normalized_entity_id(target_id))))
        )
        if edge is None:
            return None
        return GraphEdge(
            id=str(edge["id"]),
            source_id=str(edge["source_id"]),
            target_id=str(edge["target_id"]),
            type=str(edge.get("type") or "RELATED"),
            properties=dict(edge.get("properties") or {}),
        )


def _community_schema_from_artifact(data: Mapping[str, Any]) -> CommunitySchema:
    return CommunitySchema(
        community_id=str(data["community_id"]),
        level=int(data["level"]),
        title=str(data["title"]),
        nodes=tuple(str(node) for node in data["nodes"]),
        edges=tuple(
            (str(edge[0]), str(edge[1]))
            for edge in data["edges"]
            if isinstance(edge, list | tuple) and len(edge) == 2
        ),
        chunk_ids=tuple(str(chunk_id) for chunk_id in data["chunk_ids"]),
        occurrence=float(data["occurrence"]),
        sub_communities=tuple(str(item) for item in data["sub_communities"]),
    )


def _normalized_entity_id(value: Any) -> str:
    return html.unescape(str(value).upper().strip())


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_sql_store(component: object) -> SQLStoreProtocol:
    if not isinstance(component, SQLStoreProtocol):
        raise TypeError("stores.sql must satisfy SQLStoreProtocol")
    return component


def _require_language_model(component: object) -> LanguageModelProtocol:
    if not isinstance(component, LanguageModelProtocol):
        raise TypeError("models.language must satisfy LanguageModelProtocol")
    return component
