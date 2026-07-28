"""LeanRAG procedure composition."""

from __future__ import annotations

from dataclasses import dataclass, field

from heta_framework.kb.steps import (
    BuildLeanRAGGraph,
    BuildLeanRAGGraphConfig,
    ExtractLeanRAGGraph,
    ExtractLeanRAGGraphConfig,
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

    raw_prefix: str = "raw"
    parsed_prefix: str = "parsed"
    chunks_prefix: str = "chunks"
    chunk_keys_artifact: str = "chunk_keys"
    chunks_artifact: str = "lean_rag_chunks"
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
        return (
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
            ExtractLeanRAGGraph(
                ExtractLeanRAGGraphConfig(
                    chunk_keys_artifact=self.chunk_keys_artifact,
                    chunks_artifact=self.chunks_artifact,
                    base_entities_artifact=self.base_entities_artifact,
                    base_relations_artifact=self.base_relations_artifact,
                    extraction_trace_artifact=self.extraction_trace_artifact,
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
                    embedding_model=self.embedding_model,
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
                    clustering_backend=self.aggregation_clustering_backend,  # type: ignore[arg-type]
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
        )
