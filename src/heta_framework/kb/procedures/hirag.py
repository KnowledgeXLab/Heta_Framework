"""HiRAG procedure composition."""

from __future__ import annotations

from dataclasses import dataclass, field

from heta_framework.kb.steps import (
    BuildHiRAGGraph,
    BuildHiRAGGraphConfig,
    ExtractHiRAGGraph,
    ExtractHiRAGGraphConfig,
    HiRAGTableNames,
    HiRAGVectorCollections,
    KnowledgeStepProtocol,
    ParseDocuments,
    ParseDocumentsConfig,
    SplitDocuments,
    SplitDocumentsConfig,
)


@dataclass(frozen=True)
class HiRAGProcedure:
    """Static step composition for HiRAG workflows."""

    enable_hierarchical_mode: bool = True
    enable_hierachical_mode: bool | None = None
    enable_naive_rag: bool = False

    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    summary_llm_max_tokens: int = 1200
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    hierarchical_layers: int = 50
    hierarchical_max_length_in_cluster: int = 60000
    hierarchical_reduction_dimension: int = 2
    hierarchical_cluster_threshold: float = 0.1
    hierarchical_sparsity_threshold: float = 0.98
    hierarchical_sparsity_change_rate: float = 0.05
    clustering_backend: str = "auto"
    vector_metric: str = "cosine"
    batch_size: int = 128
    temperature: float = 0.0

    top_k: int = 20
    top_m: int = 10
    level: int = 2
    max_token_for_text_unit: int = 20000
    max_token_for_local_context: int = 20000
    max_token_for_bridge_knowledge: int = 12500
    max_token_for_community_report: int = 12500
    community_single_one: bool = False
    response_type: str = "Multiple Paragraphs"

    raw_prefix: str = "raw"
    parsed_prefix: str = "parsed"
    chunks_prefix: str = "chunks"
    chunk_keys_artifact: str = "chunk_keys"
    graph_node_keys_artifact: str = "hi_rag_graph_node_keys"
    graph_edge_keys_artifact: str = "hi_rag_graph_edge_keys"
    chunks_artifact: str = "hi_rag_chunks"
    extract_result_artifact: str = "extract_hi_rag_graph_result"
    build_result_artifact: str = "build_hi_rag_graph_result"

    table_names: HiRAGTableNames = field(default_factory=HiRAGTableNames)
    vector_collections: HiRAGVectorCollections = field(default_factory=HiRAGVectorCollections)

    object_store: str | None = None
    graph_store: str | None = None
    sql_store: str | None = None
    vector_store: str | None = None
    language_model: str | None = None
    embedding_model: str | None = None
    parser_registry: str | None = None

    @property
    def name(self) -> str:
        return "hirag"

    @property
    def hierarchical_mode_enabled(self) -> bool:
        if self.enable_hierachical_mode is not None:
            return self.enable_hierachical_mode
        return self.enable_hierarchical_mode

    def steps(self) -> tuple[KnowledgeStepProtocol, ...]:
        if not self.hierarchical_mode_enabled:
            raise ValueError("HiRAGProcedure MVP requires hierarchical mode to be enabled")
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
            ExtractHiRAGGraph(
                ExtractHiRAGGraphConfig(
                    chunk_keys_artifact=self.chunk_keys_artifact,
                    result_artifact=self.extract_result_artifact,
                    graph_node_keys_artifact=self.graph_node_keys_artifact,
                    graph_edge_keys_artifact=self.graph_edge_keys_artifact,
                    chunks_artifact=self.chunks_artifact,
                    entity_extract_max_gleaning=self.entity_extract_max_gleaning,
                    entity_summary_to_max_tokens=self.entity_summary_to_max_tokens,
                    summary_llm_max_tokens=self.summary_llm_max_tokens,
                    hierarchical_layers=self.hierarchical_layers,
                    hierarchical_max_length_in_cluster=self.hierarchical_max_length_in_cluster,
                    hierarchical_reduction_dimension=self.hierarchical_reduction_dimension,
                    hierarchical_cluster_threshold=self.hierarchical_cluster_threshold,
                    hierarchical_sparsity_threshold=self.hierarchical_sparsity_threshold,
                    hierarchical_sparsity_change_rate=self.hierarchical_sparsity_change_rate,
                    clustering_backend=self.clustering_backend,  # type: ignore[arg-type]
                    temperature=self.temperature,
                    object_store=self.object_store,
                    graph_store=self.graph_store,
                    language_model=self.language_model,
                    embedding_model=self.embedding_model,
                )
            ),
            BuildHiRAGGraph(
                BuildHiRAGGraphConfig(
                    table_names=self.table_names,
                    vector_collections=self.vector_collections,
                    graph_node_keys_artifact=self.graph_node_keys_artifact,
                    graph_edge_keys_artifact=self.graph_edge_keys_artifact,
                    chunks_artifact=self.chunks_artifact,
                    result_artifact=self.build_result_artifact,
                    vector_metric=self.vector_metric,
                    graph_cluster_algorithm=self.graph_cluster_algorithm,
                    max_graph_cluster_size=self.max_graph_cluster_size,
                    graph_cluster_seed=self.graph_cluster_seed,
                    batch_size=self.batch_size,
                    temperature=self.temperature,
                    object_store=self.object_store,
                    graph_store=self.graph_store,
                    sql_store=self.sql_store,
                    vector_store=self.vector_store,
                    language_model=self.language_model,
                    embedding_model=self.embedding_model,
                )
            ),
        )
