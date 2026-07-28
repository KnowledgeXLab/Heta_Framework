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
    KnowledgeProcedureProtocol,
)
from heta_framework.kb.steps import (  # noqa: E402
    BuildGraph,
    BuildWikiPages,
    BuildWikiPagesConfig,
    DeduplicateEntities,
    DeduplicateRelations,
    EmbedChunks,
    EmbedChunksConfig,
    ExtractEntities,
    ExtractRelations,
    GraphTableNames,
    IndexFullText,
    IndexFullTextConfig,
    IndexVectors,
    IndexVectorsConfig,
    MergeGraphIntoStore,
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
