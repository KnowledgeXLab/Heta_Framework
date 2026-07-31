"""GraphRAG-style graph extraction and community report procedure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from heta_framework.kb.steps import (
    AdaptUniversalGraphForGraphRAG,
    AdaptUniversalGraphForGraphRAGConfig,
    BuildRAGGraph,
    BuildRAGGraphConfig,
    ConstrainGraphByOntology,
    ConstrainGraphByOntologyConfig,
    ExtractUniversalGraph,
    ExtractUniversalGraphConfig,
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
    schema_constraint_enabled: bool = False
    ontology_schema: Mapping[str, Any] = field(default_factory=dict)

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
        relation_keys_artifact = "graph_rag_relation_keys"
        constrained_entity_keys_artifact = "graph_rag_ontology_entity_keys"
        constrained_relation_keys_artifact = "graph_rag_ontology_relation_keys"
        entity_input = (
            constrained_entity_keys_artifact
            if self.schema_constraint_enabled
            else self.entity_keys_artifact
        )
        relation_input = (
            constrained_relation_keys_artifact
            if self.schema_constraint_enabled
            else relation_keys_artifact
        )
        steps: list[KnowledgeStepProtocol] = [
            ExtractUniversalGraph(
                ExtractUniversalGraphConfig(
                    chunk_keys_artifact=self.chunk_keys_artifact,
                    entity_keys_artifact=self.entity_keys_artifact,
                    relation_keys_artifact=relation_keys_artifact,
                    temperature=self.temperature,
                    object_store=self.object_store,
                    language_model=self.language_model,
                )
            )
        ]
        if self.schema_constraint_enabled:
            steps.append(
                ConstrainGraphByOntology(
                    ConstrainGraphByOntologyConfig(
                        schema=self.ontology_schema,
                        chunk_keys_artifact=self.chunk_keys_artifact,
                        entity_keys_artifact=self.entity_keys_artifact,
                        relation_keys_artifact=relation_keys_artifact,
                        constrained_entity_keys_artifact=constrained_entity_keys_artifact,
                        constrained_relation_keys_artifact=constrained_relation_keys_artifact,
                        object_store=self.object_store,
                        language_model=self.language_model,
                    )
                )
            )
        steps.extend(
            [
                AdaptUniversalGraphForGraphRAG(
                    AdaptUniversalGraphForGraphRAGConfig(
                        chunk_keys_artifact=self.chunk_keys_artifact,
                        entity_keys_artifact=entity_input,
                        relation_keys_artifact=relation_input,
                        graph_nodes_prefix="graph/nodes",
                        graph_edges_prefix="graph/edges",
                        graph_node_keys_artifact=self.graph_node_keys_artifact,
                        graph_edge_keys_artifact=self.graph_edge_keys_artifact,
                        object_store=self.object_store,
                        graph_store=self.graph_store,
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
            ]
        )
        return tuple(steps)
