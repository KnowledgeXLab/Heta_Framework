"""Generate community reports from a clusterable graph store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.common.stores.graph import (
    ClusterableGraphStoreProtocol,
    GraphEdge,
    GraphNode,
)
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import validate_object_prefix
from heta_framework.kb.cleanup import StepCleanupPlan, object_key_targets
from heta_framework.kb.graphing.prompts import GRAPH_RAG_COMMUNITY_REPORT_PROMPT
from heta_framework.kb.steps.community_report import (
    CommunityReport,
    CommunityReportGenerationConfig,
    CommunitySchema,
    community_report_to_dict,
    community_schema_from_mapping,
    generate_community_reports,
    put_community_report,
)
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
class GraphCommunityResult:
    """Artifacts produced by GraphCommunity."""

    community_count: int
    report_ids: tuple[str, ...]
    report_keys: tuple[str, ...]


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
        reports = await generate_community_reports(
            community_schema_from_mapping(await graph_store.community_schema()),
            graph_adapter=_GraphStoreCommunityReportAdapter(graph_store),
            language_model=language_model,
            config=CommunityReportGenerationConfig(
                prompt_template=str(GRAPH_RAG_COMMUNITY_REPORT_PROMPT),
                step_name=self.name,
                report_context_max_tokens=self.config.report_context_max_tokens,
                max_nodes_per_report=self.config.max_nodes_per_report,
                max_edges_per_report=self.config.max_edges_per_report,
                report_max_output_tokens=self.config.report_max_output_tokens,
                temperature=self.config.temperature,
                provider_options={
                    "extra_body": {
                        "enable_thinking": False,
                    }
                },
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

        result = GraphCommunityResult(
            community_count=len(reports),
            report_ids=tuple(report.community_id for report in reports),
            report_keys=report_keys,
        )
        context.set_artifact(
            self.config.community_reports_artifact,
            tuple(community_report_to_dict(report) for report in reports),
        )
        context.set_artifact(self.config.community_report_keys_artifact, result.report_keys)
        context.set_artifact(self.config.graph_community_result_artifact, result)


class _GraphStoreCommunityReportAdapter:
    def __init__(self, graph_store: ClusterableGraphStoreProtocol) -> None:
        self.graph_store = graph_store

    async def get_node(self, node_id: str) -> GraphNode | None:
        return await self.graph_store.get_node(node_id)

    async def get_edge(self, source_id: str, target_id: str) -> GraphEdge | None:
        direct = await self.graph_store.get_edge(f"{source_id}--RELATED--{target_id}")
        if direct is not None:
            return direct
        return await self.graph_store.get_edge(f"{target_id}--RELATED--{source_id}")


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
