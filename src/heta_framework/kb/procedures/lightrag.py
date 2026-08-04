"""LightRAG-style graph extraction and retrieval build procedure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from heta_framework.kb.steps import (
    AdaptUniversalGraphForLightRAG,
    AdaptUniversalGraphForLightRAGConfig,
    BuildLightRAGGraph,
    BuildLightRAGGraphConfig,
    ConstrainGraphByOntology,
    ConstrainGraphByOntologyConfig,
    ExtractUniversalGraph,
    ExtractUniversalGraphConfig,
    KnowledgeStepProtocol,
    LightRAGTableNames,
    LightRAGVectorCollections,
)


@dataclass(frozen=True)
class LightRAGProcedure:
    """Static step composition for LightRAG-style graph workflows."""

    chunk_keys_artifact: str = "chunk_keys"
    entity_keys_artifact: str = "light_rag_entity_keys"
    graph_node_keys_artifact: str = "light_rag_graph_node_keys"
    graph_edge_keys_artifact: str = "light_rag_graph_edge_keys"
    extract_result_artifact: str = "extract_light_rag_graph_result"
    build_result_artifact: str = "build_light_rag_graph_result"

    table_names: LightRAGTableNames = field(default_factory=LightRAGTableNames)
    vector_collections: LightRAGVectorCollections = field(
        default_factory=LightRAGVectorCollections
    )

    extraction_format: Literal["json", "tuple"] = "json"
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    summary_llm_max_tokens: int = 1200
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
        return "lightrag"

    def steps(self) -> tuple[KnowledgeStepProtocol, ...]:
        """Expand this procedure into executable build steps."""
        relation_keys_artifact = "light_rag_relation_keys"
        constrained_entity_keys_artifact = "light_rag_ontology_entity_keys"
        constrained_relation_keys_artifact = "light_rag_ontology_relation_keys"
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
                AdaptUniversalGraphForLightRAG(
                    AdaptUniversalGraphForLightRAGConfig(
                        chunk_keys_artifact=self.chunk_keys_artifact,
                        entity_keys_artifact=entity_input,
                        relation_keys_artifact=relation_input,
                        graph_nodes_prefix="light_rag/graph/nodes",
                        graph_edges_prefix="light_rag/graph/edges",
                        graph_node_keys_artifact=self.graph_node_keys_artifact,
                        graph_edge_keys_artifact=self.graph_edge_keys_artifact,
                        result_artifact=self.extract_result_artifact,
                        temperature=self.temperature,
                        object_store=self.object_store,
                        graph_store=self.graph_store,
                        language_model=self.language_model,
                    )
                ),
                BuildLightRAGGraph(
                    BuildLightRAGGraphConfig(
                        table_names=self.table_names,
                        vector_collections=self.vector_collections,
                        graph_node_keys_artifact=self.graph_node_keys_artifact,
                        graph_edge_keys_artifact=self.graph_edge_keys_artifact,
                        chunk_keys_artifact=self.chunk_keys_artifact,
                        result_artifact=self.build_result_artifact,
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
