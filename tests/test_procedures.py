import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.kb.procedures import (  # noqa: E402
    HetaGraphProcedure,
    KnowledgeProcedureProtocol,
)
from heta_framework.kb.steps import (  # noqa: E402
    BuildGraph,
    DeduplicateEntities,
    DeduplicateRelations,
    ExtractEntities,
    ExtractRelations,
    GraphTableNames,
    MergeGraphIntoStore,
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
