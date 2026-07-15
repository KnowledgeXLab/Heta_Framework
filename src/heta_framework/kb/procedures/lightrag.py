"""LightRAG-style graph extraction and retrieval build procedure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from heta_framework.kb.steps import (
    BuildLightRAGGraph,
    BuildLightRAGGraphConfig,
    ExtractLightRAGGraph,
    ExtractLightRAGGraphConfig,
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
        return (
            ExtractLightRAGGraph(
                ExtractLightRAGGraphConfig(
                    extraction_format=self.extraction_format,
                    chunk_keys_artifact=self.chunk_keys_artifact,
                    entity_keys_artifact=self.entity_keys_artifact,
                    graph_node_keys_artifact=self.graph_node_keys_artifact,
                    graph_edge_keys_artifact=self.graph_edge_keys_artifact,
                    result_artifact=self.extract_result_artifact,
                    entity_extract_max_gleaning=self.entity_extract_max_gleaning,
                    entity_summary_to_max_tokens=self.entity_summary_to_max_tokens,
                    summary_llm_max_tokens=self.summary_llm_max_tokens,
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
        )
