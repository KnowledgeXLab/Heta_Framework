import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.kb.procedures import (  # noqa: E402
    HiRAGProcedure,
    HetaGraphProcedure,
    KnowledgeProcedureProtocol,
    LightRAGProcedure,
)
from heta_framework.kb.search import QueryEngineRegistry  # noqa: E402
from heta_framework.kb.steps import (  # noqa: E402
    BuildLightRAGGraph,
    BuildHiRAGGraph,
    BuildGraph,
    DeduplicateEntities,
    DeduplicateRelations,
    ExtractLightRAGGraph,
    ExtractHiRAGGraph,
    ExtractEntities,
    ExtractRelations,
    GraphTableNames,
    LightRAGTableNames,
    HiRAGTableNames,
    MergeGraphIntoStore,
    ParseDocuments,
    SplitDocuments,
)


def test_heta_graph_build_procedure_expands_to_deduplicated_build_steps():
    procedure = HetaGraphProcedure.build(
        chunk_keys_artifact="custom_chunk_keys",
        table_names=GraphTableNames(
            entities="paper_entities",
            relations="paper_relations",
            evidence="paper_graph_evidence",
        ),
        object_store="main",
        sql_store="pg",
        vector_store="milvus",
        language_model="reasoner",
        embedding_model="embedder",
    )

    steps = procedure.steps()

    assert isinstance(procedure, KnowledgeProcedureProtocol)
    assert procedure.name == "heta_graph"
    assert [type(step) for step in steps] == [
        ExtractEntities,
        ExtractRelations,
        DeduplicateEntities,
        DeduplicateRelations,
        BuildGraph,
    ]
    assert steps[0].config.chunk_keys_artifact == "custom_chunk_keys"
    assert steps[0].config.entity_keys_artifact == "entity_keys"
    assert steps[1].config.entity_keys_artifact == "entity_keys"
    assert steps[1].config.relation_keys_artifact == "relation_keys"
    assert steps[2].config.deduplicated_entity_keys_artifact == "deduplicated_entity_keys"
    assert steps[3].config.deduplicated_relation_keys_artifact == "deduplicated_relation_keys"
    assert steps[4].config.entity_keys_artifact == "deduplicated_entity_keys"
    assert steps[4].config.relation_keys_artifact == "deduplicated_relation_keys"
    assert steps[4].config.table_names.entities == "paper_entities"
    assert steps[4].config.sql_store == "pg"
    assert steps[4].config.vector_store == "milvus"


def test_heta_graph_procedure_can_skip_deduplication():
    steps = HetaGraphProcedure.build(deduplicate=False).steps()

    assert [type(step) for step in steps] == [
        ExtractEntities,
        ExtractRelations,
        BuildGraph,
    ]
    assert steps[-1].config.entity_keys_artifact == "entity_keys"
    assert steps[-1].config.relation_keys_artifact == "relation_keys"


def test_heta_graph_merge_procedure_uses_merge_graph_into_store():
    steps = HetaGraphProcedure.merge_into_store().steps()

    assert [type(step) for step in steps] == [
        ExtractEntities,
        ExtractRelations,
        DeduplicateEntities,
        DeduplicateRelations,
        MergeGraphIntoStore,
    ]
    assert steps[-1].config.entity_keys_artifact == "deduplicated_entity_keys"
    assert steps[-1].config.relation_keys_artifact == "deduplicated_relation_keys"


def test_lightrag_procedure_expands_to_extract_and_build_steps():
    procedure = LightRAGProcedure(
        extraction_format="tuple",
        chunk_keys_artifact="custom_chunk_keys",
        table_names=LightRAGTableNames(
            entities="lr_entities",
            relations="lr_relations",
            chunks="lr_chunks",
        ),
        object_store="main",
        graph_store="graph",
        sql_store="sqlite",
        vector_store="vectors",
        language_model="reasoner",
        embedding_model="embedder",
    )

    steps = procedure.steps()

    assert isinstance(procedure, KnowledgeProcedureProtocol)
    assert procedure.name == "lightrag"
    assert [type(step) for step in steps] == [ExtractLightRAGGraph, BuildLightRAGGraph]
    assert steps[0].config.extraction_format == "tuple"
    assert steps[0].config.chunk_keys_artifact == "custom_chunk_keys"
    assert steps[0].config.graph_store == "graph"
    assert steps[1].config.graph_node_keys_artifact == "light_rag_graph_node_keys"
    assert steps[1].config.table_names.entities == "lr_entities"
    assert steps[1].config.sql_store == "sqlite"
    assert steps[1].config.vector_store == "vectors"


def test_lightrag_query_modes_registered_by_default_registry():
    modes = QueryEngineRegistry.defaults().modes

    assert "light_rag_local_query" in modes
    assert "light_rag_global_query" in modes
    assert "light_rag_hybrid_query" in modes
    assert "light_rag_mix_query" in modes


def test_hirag_procedure_expands_to_parse_split_extract_and_build_steps():
    procedure = HiRAGProcedure(
        chunk_token_size=256,
        chunk_overlap_token_size=32,
        table_names=HiRAGTableNames(
            entities="hi_entities",
            relations="hi_relations",
            communities="hi_communities",
            chunks="hi_chunks",
        ),
        object_store="main",
        graph_store="graph",
        sql_store="sqlite",
        vector_store="vectors",
        language_model="reasoner",
        embedding_model="embedder",
        max_graph_cluster_size=7,
        graph_cluster_seed=123,
    )

    steps = procedure.steps()

    assert isinstance(procedure, KnowledgeProcedureProtocol)
    assert procedure.name == "hirag"
    assert [type(step) for step in steps] == [
        ParseDocuments,
        SplitDocuments,
        ExtractHiRAGGraph,
        BuildHiRAGGraph,
    ]
    assert steps[1].config.chunk_size == 256
    assert steps[1].config.overlap == 32
    assert steps[2].config.graph_store == "graph"
    assert steps[3].config.table_names.entities == "hi_entities"
    assert steps[3].config.graph_cluster_algorithm == "leiden"
    assert steps[3].config.max_graph_cluster_size == 7
    assert steps[3].config.graph_cluster_seed == 123
    assert steps[3].config.sql_store == "sqlite"
    assert steps[3].config.vector_store == "vectors"


def test_hirag_procedure_keeps_original_hierachical_typo_alias():
    procedure = HiRAGProcedure(enable_hierarchical_mode=True, enable_hierachical_mode=False)

    assert procedure.hierarchical_mode_enabled is False


def test_hirag_query_modes_registered_by_default_registry():
    modes = QueryEngineRegistry.defaults().modes

    assert "hi_rag_query" in modes
    assert "hi_rag_nobridge_query" in modes
    assert "hi_rag_local_query" in modes
    assert "hi_rag_global_query" in modes
    assert "hi_rag_bridge_query" in modes
