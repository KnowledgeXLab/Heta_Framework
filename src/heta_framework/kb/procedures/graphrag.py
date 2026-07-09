"""GraphRAG-style graph extraction and community report procedure."""

from __future__ import annotations

from dataclasses import dataclass, field

from heta_framework.kb.steps import (
    BuildRAGGraph,
    BuildRAGGraphConfig,
    ExtractGraph,
    ExtractGraphConfig,
    GraphCommunity,
    GraphCommunityConfig,
    KnowledgeStepProtocol,
    RAGGraphTableNames,
    RAGGraphVectorCollections,
)


@dataclass(frozen=True)
class GraphRAGProcedure:
    """Static step composition for GraphRAG-style graph workflows."""

    chunk_keys_artifact: str = "chunk_keys"
    entity_keys_artifact: str = "entity_keys"
    graph_node_keys_artifact: str = "graph_node_keys"
    graph_edge_keys_artifact: str = "graph_edge_keys"
    community_reports_artifact: str = "community_reports"
    community_report_keys_artifact: str = "community_report_keys"
    graph_community_result_artifact: str = "graph_community_result"

    table_names: RAGGraphTableNames = field(default_factory=RAGGraphTableNames)
    vector_collections: RAGGraphVectorCollections = field(
        default_factory=RAGGraphVectorCollections
    )

    graph_cluster_algorithm: str = "leiden"
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    summary_llm_max_tokens: int = 1200
    report_context_max_tokens: int = 12000
    report_max_output_tokens: int = 800
    vector_metric: str = "cosine"
    batch_size: int = 128
    temperature: float = 0.0

    object_store: str | None = None
    graph_store: str | None = None
    sql_store: str | None = None
    vector_store: str | None = None
    language_model: str | None = None
    embedding_model: str | None = None

    @property
    def name(self) -> str:
        """Return the stable procedure name."""
        return "graphrag"

    def steps(self) -> tuple[KnowledgeStepProtocol, ...]:
        """Expand this procedure into executable build steps."""
        return (
            ExtractGraph(
                ExtractGraphConfig(
                    chunk_keys_artifact=self.chunk_keys_artifact,
                    entity_keys_artifact=self.entity_keys_artifact,
                    graph_node_keys_artifact=self.graph_node_keys_artifact,
                    graph_edge_keys_artifact=self.graph_edge_keys_artifact,
                    entity_extract_max_gleaning=self.entity_extract_max_gleaning,
                    entity_summary_to_max_tokens=self.entity_summary_to_max_tokens,
                    summary_llm_max_tokens=self.summary_llm_max_tokens,
                    temperature=self.temperature,
                    object_store=self.object_store,
                    graph_store=self.graph_store,
                    language_model=self.language_model,
                )
            ),
            GraphCommunity(
                GraphCommunityConfig(
                    graph_cluster_algorithm=self.graph_cluster_algorithm,
                    community_reports_artifact=self.community_reports_artifact,
                    community_report_keys_artifact=self.community_report_keys_artifact,
                    graph_community_result_artifact=self.graph_community_result_artifact,
                    report_context_max_tokens=self.report_context_max_tokens,
                    report_max_output_tokens=self.report_max_output_tokens,
                    temperature=self.temperature,
                    object_store=self.object_store,
                    graph_store=self.graph_store,
                    language_model=self.language_model,
                )
            ),
            BuildRAGGraph(
                BuildRAGGraphConfig(
                    table_names=self.table_names,
                    vector_collections=self.vector_collections,
                    graph_node_keys_artifact=self.graph_node_keys_artifact,
                    graph_edge_keys_artifact=self.graph_edge_keys_artifact,
                    community_report_keys_artifact=self.community_report_keys_artifact,
                    chunk_keys_artifact=self.chunk_keys_artifact,
                    vector_metric=self.vector_metric,
                    batch_size=self.batch_size,
                    object_store=self.object_store,
                    sql_store=self.sql_store,
                    vector_store=self.vector_store,
                    embedding_model=self.embedding_model,
                )
            ),
        )
