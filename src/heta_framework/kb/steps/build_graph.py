"""Build Heta-style graph tables from extracted entities and relations."""

from __future__ import annotations

from dataclasses import dataclass, field

from heta_framework.common.models.protocols import EmbeddingModelProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.sql import SQLStoreProtocol
from heta_framework.common.stores.vector import VectorStoreProtocol
from heta_framework.kb.chunking import ParsedChunk
from heta_framework.kb.graphing import ExtractedEntity, ExtractedRelation
from heta_framework.kb.search import SearchAsset
from heta_framework.kb.steps.graph_storage import (
    GraphStorageConfig,
    GraphTableNames,
    GraphVectorCollections,
    batches,
    embed_entity_records,
    embed_relation_records,
    ensure_graph_tables,
    entity_row,
    evidence_rows_for_entity,
    evidence_rows_for_relation,
    graph_vector_dimension,
    relation_row,
    upsert_entity_rows,
    upsert_evidence_rows,
    upsert_graph_vectors,
    upsert_relation_rows,
)
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import (
    StepCapabilities,
    StepIssue,
    StepRequirements,
    model_ref,
    store_ref,
)


@dataclass(frozen=True)
class BuildGraphConfig:
    """Configuration for BuildGraph."""

    table_names: GraphTableNames = field(default_factory=GraphTableNames)
    vector_collections: GraphVectorCollections = field(default_factory=GraphVectorCollections)
    entity_keys_artifact: str = "deduplicated_entity_keys"
    relation_keys_artifact: str = "deduplicated_relation_keys"
    chunk_keys_artifact: str = "chunk_keys"
    vector_metric: str = "cosine"
    batch_size: int = 128
    object_store: str | None = None
    sql_store: str | None = None
    vector_store: str | None = None
    embedding_model: str | None = None

    def __post_init__(self) -> None:
        if self.entity_keys_artifact.strip() == "":
            raise ValueError("entity_keys_artifact must not be empty")
        if self.relation_keys_artifact.strip() == "":
            raise ValueError("relation_keys_artifact must not be empty")
        if self.chunk_keys_artifact.strip() == "":
            raise ValueError("chunk_keys_artifact must not be empty")
        if self.vector_metric not in {"cosine", "dot", "l2"}:
            raise ValueError("vector_metric must be one of: cosine, dot, l2")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")


@dataclass(frozen=True)
class BuildGraphResult:
    """Artifacts produced by BuildGraph."""

    entity_count: int
    relation_count: int
    evidence_count: int
    entity_vector_count: int
    relation_vector_count: int
    vector_dimension: int
    skipped_evidence_count: int
    issues: tuple[StepIssue, ...] = ()


class BuildGraph:
    """Write extracted graph facts into Heta-style PostgreSQL tables."""

    name = "build_graph"

    def __init__(self, config: BuildGraphConfig | None = None) -> None:
        self.config = config or BuildGraphConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components and artifacts required by this step."""
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("objects", self.config.object_store),
                    store_ref("sql", self.config.sql_store),
                    store_ref("vector", self.config.vector_store),
                    model_ref("embedding", self.config.embedding_model),
                }
            ),
            artifacts=frozenset(
                {
                    self.config.entity_keys_artifact,
                    self.config.relation_keys_artifact,
                    self.config.chunk_keys_artifact,
                }
            ),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts and query modes produced by this step."""
        sql_store_ref = store_ref("sql", self.config.sql_store)
        vector_store_ref = store_ref("vector", self.config.vector_store)
        return StepCapabilities(
            artifacts=frozenset({"build_graph_result"}),
            queries=frozenset({"heta_graph_search"}),
            search_assets=(
                SearchAsset(
                    kind="graph_tables",
                    name=self.config.table_names.entities,
                    store=sql_store_ref.key,
                    metadata={
                        "entities_table": self.config.table_names.entities,
                        "relations_table": self.config.table_names.relations,
                        "evidence_table": self.config.table_names.evidence,
                    },
                ),
                SearchAsset(
                    kind="graph_vector_index",
                    name=self.config.vector_collections.entities,
                    store=vector_store_ref.key,
                    metadata={
                        "entity_collection": self.config.vector_collections.entities,
                        "relation_collection": self.config.vector_collections.relations,
                    },
                ),
            ),
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Run the graph build step and write SQL graph tables."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        sql_store = _require_sql_store(
            context.get_component(store_ref("sql", self.config.sql_store).key)
        )
        vector_store = _require_vector_store(
            context.get_component(store_ref("vector", self.config.vector_store).key)
        )
        embedding_model = _require_embedding_model(
            context.get_component(model_ref("embedding", self.config.embedding_model).key)
        )
        entity_keys = tuple(context.get_artifact(self.config.entity_keys_artifact))
        relation_keys = tuple(context.get_artifact(self.config.relation_keys_artifact))
        chunk_keys = tuple(context.get_artifact(self.config.chunk_keys_artifact))

        entities = [ExtractedEntity.from_json(await object_store.get(key)) for key in entity_keys]
        relations = [
            ExtractedRelation.from_json(await object_store.get(key)) for key in relation_keys
        ]
        chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]
        chunk_sources = {chunk.chunk_id: chunk for chunk in chunks}

        entity_by_id = {entity.entity_id: entity for entity in entities}
        if len(entity_by_id) != len(entities):
            raise ValueError("entity keys must not contain duplicate entity ids")

        storage_config = _storage_config(self.config)
        issues: list[StepIssue] = []
        evidence_rows: list[dict[str, object]] = []
        for entity in entities:
            evidence_rows.extend(
                evidence_rows_for_entity(entity, chunk_sources, issues, step_name=self.name)
            )
        for relation in relations:
            evidence_rows.extend(
                evidence_rows_for_relation(relation, chunk_sources, issues, step_name=self.name)
            )

        entity_vectors = await embed_entity_records(embedding_model, entities, storage_config)
        relation_vectors = await embed_relation_records(embedding_model, relations, storage_config)
        vector_dimension = graph_vector_dimension(entity_vectors, relation_vectors)

        async with sql_store.transaction() as tx:
            await ensure_graph_tables(tx, storage_config)
            for batch in batches(
                [entity_row(entity) for entity in entities],
                self.config.batch_size,
            ):
                await upsert_entity_rows(tx, self.config.table_names.entities, batch)
            for batch in batches(
                [relation_row(relation) for relation in relations],
                self.config.batch_size,
            ):
                await upsert_relation_rows(tx, self.config.table_names.relations, batch)
            for batch in batches(evidence_rows, self.config.batch_size):
                await upsert_evidence_rows(tx, self.config.table_names.evidence, batch)

        await upsert_graph_vectors(vector_store, entity_vectors, relation_vectors, storage_config)

        result = BuildGraphResult(
            entity_count=len(entities),
            relation_count=len(relations),
            evidence_count=len(evidence_rows),
            entity_vector_count=len(entity_vectors),
            relation_vector_count=len(relation_vectors),
            vector_dimension=vector_dimension,
            skipped_evidence_count=len(issues),
            issues=tuple(issues),
        )
        context.set_artifact("build_graph_result", result)


def _storage_config(config: BuildGraphConfig) -> GraphStorageConfig:
    return GraphStorageConfig(
        table_names=config.table_names,
        vector_collections=config.vector_collections,
        vector_metric=config.vector_metric,
        batch_size=config.batch_size,
        trace_step=BuildGraph.name,
    )


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_sql_store(component: object) -> SQLStoreProtocol:
    if not isinstance(component, SQLStoreProtocol):
        raise TypeError("stores.sql must satisfy SQLStoreProtocol")
    return component


def _require_vector_store(component: object) -> VectorStoreProtocol:
    if not isinstance(component, VectorStoreProtocol):
        raise TypeError("stores.vector must satisfy VectorStoreProtocol")
    return component


def _require_embedding_model(component: object) -> EmbeddingModelProtocol:
    if not isinstance(component, EmbeddingModelProtocol):
        raise TypeError("models.embedding must satisfy EmbeddingModelProtocol")
    return component
