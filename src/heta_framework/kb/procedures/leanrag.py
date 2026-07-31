"""LeanRAG procedure composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from heta_framework.kb.steps import (
    AdaptUniversalGraphForLeanRAG,
    AdaptUniversalGraphForLeanRAGConfig,
    BuildLeanRAGGraph,
    BuildLeanRAGGraphConfig,
    ConstrainGraphByOntology,
    ConstrainGraphByOntologyConfig,
    ExtractUniversalGraph,
    ExtractUniversalGraphConfig,
    KnowledgeStepProtocol,
    LeanRAGSemanticAggregation,
    LeanRAGSemanticAggregationConfig,
    LeanRAGTableNames,
    LeanRAGVectorCollections,
    ParseDocuments,
    ParseDocumentsConfig,
    SplitDocuments,
    SplitDocumentsConfig,
)


@dataclass(frozen=True)
class LeanRAGProcedure:
    """Static step composition for LeanRAG workflows."""

    chunk_token_size: int = 1024
    chunk_overlap_token_size: int = 128
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    summary_llm_max_tokens: int = 1200
    aggregation_reduction_dimension: int = 2
    aggregation_cluster_threshold: float = 0.1
    aggregation_cluster_size: int = 20
    aggregation_random_seed: int = 224
    aggregation_gmm_n_init: int = 5
    aggregation_gmm_init_params: str = "k-means++"
    aggregation_clustering_backend: str = "auto"
    vector_metric: str = "cosine"
    batch_size: int = 128
    temperature: float = 0.0
    top_k: int = 10
    level_mode: int = 2
    text_unit_k: int = 5
    response_type: str = "Multiple Paragraphs"
    only_need_context: bool = False
    generate_answer: bool = True
    schema_constraint_enabled: bool = False
    ontology_schema: Mapping[str, Any] = field(default_factory=dict)

    raw_prefix: str = "raw"
    parsed_prefix: str = "parsed"
    chunks_prefix: str = "chunks"
    chunk_keys_artifact: str = "chunk_keys"
    chunks_artifact: str = "lean_rag_chunks"
    entity_keys_artifact: str = "lean_rag_universal_entity_keys"
    relation_keys_artifact: str = "lean_rag_universal_relation_keys"
    base_entities_artifact: str = "lean_rag_base_entities"
    base_relations_artifact: str = "lean_rag_base_relations"
    extraction_trace_artifact: str = "lean_rag_extraction_trace"
    graph_node_keys_artifact: str = "lean_rag_graph_node_keys"
    graph_edge_keys_artifact: str = "lean_rag_graph_edge_keys"
    extract_result_artifact: str = "extract_lean_rag_graph_result"
    all_entities_layers_artifact: str = "lean_rag_all_entities_layers"
    aggregate_entities_artifact: str = "lean_rag_aggregate_entities"
    generated_relations_artifact: str = "lean_rag_generated_relations"
    communities_artifact: str = "lean_rag_communities"
    parent_edges_artifact: str = "lean_rag_parent_edges"
    semantic_trace_artifact: str = "lean_rag_semantic_aggregation_trace"
    semantic_result_artifact: str = "lean_rag_semantic_aggregation_result"
    build_result_artifact: str = "build_lean_rag_graph_result"

    table_names: LeanRAGTableNames = field(default_factory=LeanRAGTableNames)
    vector_collections: LeanRAGVectorCollections = field(default_factory=LeanRAGVectorCollections)

    object_store: str | None = None
    graph_store: str | None = None
    sql_store: str | None = None
    vector_store: str | None = None
    language_model: str | None = None
    embedding_model: str | None = None
    parser_registry: str | None = None

    @property
    def name(self) -> str:
        return "leanrag"

    def steps(self) -> tuple[KnowledgeStepProtocol, ...]:
        constrained_entity_keys_artifact = "lean_rag_ontology_entity_keys"
        constrained_relation_keys_artifact = "lean_rag_ontology_relation_keys"
        entity_input = (
            constrained_entity_keys_artifact
            if self.schema_constraint_enabled
            else self.entity_keys_artifact
        )
        relation_input = (
            constrained_relation_keys_artifact
            if self.schema_constraint_enabled
            else self.relation_keys_artifact
        )
        steps: list[KnowledgeStepProtocol] = [
            ParseDocuments(
                ParseDocumentsConfig(
                    raw_prefix=self.raw_prefix,
                    parsed_prefix=self.parsed_prefix,
                    object_store=self.object_store,
                    parser_registry=self.parser_registry,
                )
            ),
            SplitDocuments(
                SplitDocumentsConfig(
                    chunks_prefix=self.chunks_prefix,
                    chunk_size=self.chunk_token_size,
                    overlap=self.chunk_overlap_token_size,
                    encoding_name="cl100k_base",
                    object_store=self.object_store,
                )
            ),
            ExtractUniversalGraph(
                ExtractUniversalGraphConfig(
                    chunk_keys_artifact=self.chunk_keys_artifact,
                    entity_keys_artifact=self.entity_keys_artifact,
                    relation_keys_artifact=self.relation_keys_artifact,
                    temperature=self.temperature,
                    object_store=self.object_store,
                    language_model=self.language_model,
                )
            ),
        ]
        if self.schema_constraint_enabled:
            steps.append(
                ConstrainGraphByOntology(
                    ConstrainGraphByOntologyConfig(
                        schema=self.ontology_schema,
                        chunk_keys_artifact=self.chunk_keys_artifact,
                        entity_keys_artifact=self.entity_keys_artifact,
                        relation_keys_artifact=self.relation_keys_artifact,
                        constrained_entity_keys_artifact=constrained_entity_keys_artifact,
                        constrained_relation_keys_artifact=constrained_relation_keys_artifact,
                        object_store=self.object_store,
                        language_model=self.language_model,
                    )
                )
            )
        steps.extend(
            [
                AdaptUniversalGraphForLeanRAG(
                    AdaptUniversalGraphForLeanRAGConfig(
                        chunk_keys_artifact=self.chunk_keys_artifact,
                        entity_keys_artifact=entity_input,
                        relation_keys_artifact=relation_input,
                        graph_nodes_prefix="lean_rag/graph/nodes",
                        graph_edges_prefix="lean_rag/graph/edges",
                        graph_node_keys_artifact=self.graph_node_keys_artifact,
                        graph_edge_keys_artifact=self.graph_edge_keys_artifact,
                        chunks_artifact=self.chunks_artifact,
                        base_entities_artifact=self.base_entities_artifact,
                        base_relations_artifact=self.base_relations_artifact,
                        extraction_trace_artifact=self.extraction_trace_artifact,
                        result_artifact=self.extract_result_artifact,
                        object_store=self.object_store,
                        graph_store=self.graph_store,
                    )
                ),
                LeanRAGSemanticAggregation(
                    LeanRAGSemanticAggregationConfig(
                        base_entities_artifact=self.base_entities_artifact,
                        base_relations_artifact=self.base_relations_artifact,
                        all_entities_layers_artifact=self.all_entities_layers_artifact,
                        aggregate_entities_artifact=self.aggregate_entities_artifact,
                        generated_relations_artifact=self.generated_relations_artifact,
                        communities_artifact=self.communities_artifact,
                        parent_edges_artifact=self.parent_edges_artifact,
                        trace_artifact=self.semantic_trace_artifact,
                        result_artifact=self.semantic_result_artifact,
                        reduction_dimension=self.aggregation_reduction_dimension,
                        cluster_threshold=self.aggregation_cluster_threshold,
                        cluster_size=self.aggregation_cluster_size,
                        random_seed=self.aggregation_random_seed,
                        gmm_n_init=self.aggregation_gmm_n_init,
                        gmm_init_params=self.aggregation_gmm_init_params,
                        clustering_backend=(  # type: ignore[arg-type]
                            self.aggregation_clustering_backend
                        ),
                        temperature=self.temperature,
                        language_model=self.language_model,
                        embedding_model=self.embedding_model,
                    )
                ),
                BuildLeanRAGGraph(
                    BuildLeanRAGGraphConfig(
                        table_names=self.table_names,
                        vector_collections=self.vector_collections,
                        chunks_artifact=self.chunks_artifact,
                        base_relations_artifact=self.base_relations_artifact,
                        all_entities_layers_artifact=self.all_entities_layers_artifact,
                        generated_relations_artifact=self.generated_relations_artifact,
                        communities_artifact=self.communities_artifact,
                        parent_edges_artifact=self.parent_edges_artifact,
                        result_artifact=self.build_result_artifact,
                        vector_metric=self.vector_metric,
                        batch_size=self.batch_size,
                        graph_store=self.graph_store,
                        sql_store=self.sql_store,
                        vector_store=self.vector_store,
                        embedding_model=self.embedding_model,
                    )
                ),
            ]
        )
        return tuple(steps)
