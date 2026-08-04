import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.kb import (  # noqa: E402
    KnowledgeModels,
    KnowledgeRecipe,
    KnowledgeStores,
)
from heta_framework.kb.procedures import (  # noqa: E402
    HetaGraphProcedure,
    HetaWikiProcedure,
    HiRAGProcedure,
    KnowledgeProcedureProtocol,
    LightRAGProcedure,
)
from heta_framework.kb.search import QueryEngineRegistry  # noqa: E402
from heta_framework.kb.steps import (  # noqa: E402
    AdaptUniversalGraphForHiRAG,
    AdaptUniversalGraphForLightRAG,
    BuildGraph,
    BuildHiRAGGraph,
    BuildLightRAGGraph,
    BuildWikiPages,
    BuildWikiPagesConfig,
    DeduplicateEntities,
    DeduplicateRelations,
    EmbedChunks,
    EmbedChunksConfig,
    ExtractUniversalGraph,
    GraphTableNames,
    HiRAGCommunity,
    HiRAGHierarchicalAggregation,
    HiRAGTableNames,
    IndexFullText,
    IndexFullTextConfig,
    IndexVectors,
    IndexVectorsConfig,
    LightRAGTableNames,
    MergeGraphIntoStore,
    ParseDocuments,
    SplitDocuments,
    SplitWikiPages,
    SplitWikiPagesConfig,
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
        ExtractUniversalGraph,
        DeduplicateEntities,
        DeduplicateRelations,
        BuildGraph,
    ]
    assert steps[0].config.chunk_keys_artifact == "custom_chunk_keys"
    assert steps[0].config.entity_keys_artifact == "entity_keys"
    assert steps[0].config.relation_keys_artifact == "relation_keys"
    assert steps[1].config.deduplicated_entity_keys_artifact == "deduplicated_entity_keys"
    assert steps[2].config.deduplicated_relation_keys_artifact == "deduplicated_relation_keys"
    assert steps[3].config.entity_keys_artifact == "deduplicated_entity_keys"
    assert steps[3].config.relation_keys_artifact == "deduplicated_relation_keys"
    assert steps[3].config.table_names.entities == "paper_entities"
    assert steps[3].config.sql_store == "pg"
    assert steps[3].config.vector_store == "milvus"


def test_heta_graph_procedure_can_skip_deduplication():
    steps = HetaGraphProcedure.build(deduplicate=False).steps()

    assert [type(step) for step in steps] == [
        ExtractUniversalGraph,
        BuildGraph,
    ]
    assert steps[-1].config.entity_keys_artifact == "entity_keys"
    assert steps[-1].config.relation_keys_artifact == "relation_keys"


def test_heta_graph_merge_procedure_uses_merge_graph_into_store():
    steps = HetaGraphProcedure.merge_into_store().steps()

    assert [type(step) for step in steps] == [
        ExtractUniversalGraph,
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
    assert [type(step) for step in steps] == [
        ExtractUniversalGraph,
        AdaptUniversalGraphForLightRAG,
        BuildLightRAGGraph,
    ]
    assert steps[0].config.chunk_keys_artifact == "custom_chunk_keys"
    assert steps[1].config.graph_store == "graph"
    assert steps[1].config.graph_node_keys_artifact == "light_rag_graph_node_keys"
    assert steps[2].config.table_names.entities == "lr_entities"
    assert steps[2].config.sql_store == "sqlite"
    assert steps[2].config.vector_store == "vectors"


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
        ExtractUniversalGraph,
        AdaptUniversalGraphForHiRAG,
        HiRAGHierarchicalAggregation,
        HiRAGCommunity,
        BuildHiRAGGraph,
    ]
    assert steps[1].config.chunk_size == 256
    assert steps[1].config.overlap == 32
    assert steps[3].config.graph_store == "graph"
    assert steps[4].config.graph_store == "graph"
    assert steps[4].config.embedding_model == "embedder"
    assert steps[5].config.graph_cluster_algorithm == "leiden"
    assert steps[5].config.max_graph_cluster_size == 7
    assert steps[5].config.graph_cluster_seed == 123
    assert steps[5].config.language_model == "reasoner"
    assert steps[6].config.table_names.entities == "hi_entities"
    assert steps[6].config.graph_cluster_algorithm == "leiden"
    assert steps[6].config.max_graph_cluster_size == 7
    assert steps[6].config.graph_cluster_seed == 123
    assert steps[6].config.sql_store == "sqlite"
    assert steps[6].config.vector_store == "vectors"


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


def test_heta_wiki_procedure_expands_to_connected_wiki_steps():
    procedure = HetaWikiProcedure()

    steps = procedure.steps()

    assert isinstance(procedure, KnowledgeProcedureProtocol)
    assert procedure.name == "heta_wiki"
    assert [type(step) for step in steps] == [
        BuildWikiPages,
        SplitWikiPages,
        EmbedChunks,
        IndexVectors,
        IndexFullText,
    ]
    assert steps[1].config.wiki_page_keys_artifact == "wiki_page_keys"
    assert steps[2].config.preset == "wiki"
    assert steps[2].config.batch_size == 10
    assert steps[3].config.preset == "wiki"
    assert steps[4].config.preset == "wiki"


def test_heta_wiki_procedure_accepts_named_components_and_external_input():
    procedure = HetaWikiProcedure(
        build_pages_config=BuildWikiPagesConfig(
            summary_mode="extractive",
            object_store="wiki",
            parsed_document_keys_artifact="new_parsed_document_keys",
        ),
        split_pages_config=SplitWikiPagesConfig(object_store="wiki"),
        embed_chunks_config=EmbedChunksConfig(
            preset="wiki",
            object_store="wiki",
            embedding_model="wiki",
        ),
        index_vectors_config=IndexVectorsConfig(
            preset="wiki",
            object_store="wiki",
            vector_store="wiki",
        ),
        index_full_text_config=IndexFullTextConfig(
            preset="wiki",
            object_store="wiki",
            text_index_store="wiki",
        ),
    )

    steps = procedure.steps()

    assert steps[0].config.parsed_document_keys_artifact == "new_parsed_document_keys"
    assert all(step.config.object_store == "wiki" for step in steps)
    assert steps[2].config.embedding_model == "wiki"
    assert steps[3].config.vector_store == "wiki"
    assert steps[4].config.text_index_store == "wiki"


def test_heta_wiki_procedure_rejects_broken_step_wiring():
    with pytest.raises(ValueError, match="embed_chunks_config.preset"):
        HetaWikiProcedure(embed_chunks_config=EmbedChunksConfig())

    with pytest.raises(ValueError, match="same object store"):
        HetaWikiProcedure(
            build_pages_config=BuildWikiPagesConfig(object_store="pages"),
            split_pages_config=SplitWikiPagesConfig(object_store="chunks"),
        )


def test_heta_wiki_procedure_forms_a_valid_recipe_dataflow():
    recipe = KnowledgeRecipe(
        models=KnowledgeModels(language=object(), embedding=object()),
        stores=KnowledgeStores(
            objects=object(),
            vector=object(),
            text_index=object(),
        ),
        steps=(HetaWikiProcedure(),),
    )

    validation = recipe.validate(initial_artifacts={"parsed_document_keys"})

    assert validation.valid
    assert [type(step) for step in recipe.expanded_steps()] == [
        BuildWikiPages,
        SplitWikiPages,
        EmbedChunks,
        IndexVectors,
        IndexFullText,
    ]
