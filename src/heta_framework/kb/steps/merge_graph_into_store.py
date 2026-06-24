"""Merge extracted graph facts into an existing Heta-style graph store."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, TypeVar

from heta_framework.common.models.protocols import EmbeddingModelProtocol, LanguageModelProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.sql import SQLStoreProtocol
from heta_framework.common.stores.vector import (
    VectorCollectionConfig,
    VectorQuery,
    VectorRecord,
    VectorStoreProtocol,
)
from heta_framework.kb.chunking import ParsedChunk
from heta_framework.kb.cleanup import CleanupTarget, StepCleanupPlan
from heta_framework.kb.graphing import ExtractedEntity, ExtractedRelation
from heta_framework.kb.search import SearchAsset
from heta_framework.kb.steps.graph_merge_prompts import (
    entity_merge_prompt,
    invoke_graph_merge_json,
    mapping_values,
    relation_merge_prompt,
)
from heta_framework.kb.steps.graph_storage import (
    GraphStorageConfig,
    GraphTableNames,
    GraphVectorCollections,
    available_evidence_rows_for_entity,
    available_evidence_rows_for_relation,
    batches,
    embed_entity_records,
    embed_relation_records,
    ensure_graph_tables,
    entity_row,
    relation_row,
    upsert_entity_rows,
    upsert_evidence_rows,
    upsert_relation_rows,
)
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import (
    IssueResolution,
    IssueSubject,
    StepCapabilities,
    StepIssue,
    StepRequirements,
    model_ref,
    store_ref,
)

_T = TypeVar("_T")


@dataclass(frozen=True)
class MergeGraphIntoStoreConfig:
    """Configuration for MergeGraphIntoStore."""

    table_names: GraphTableNames = field(default_factory=GraphTableNames)
    vector_collections: GraphVectorCollections = field(default_factory=GraphVectorCollections)
    entity_keys_artifact: str = "deduplicated_entity_keys"
    relation_keys_artifact: str = "deduplicated_relation_keys"
    chunk_keys_artifact: str = "chunk_keys"
    vector_metric: str = "cosine"
    similarity_threshold: float | None = None
    top_k: int = 8
    batch_size: int = 64
    llm_max_retries: int = 2
    temperature: float = 0
    object_store: str | None = None
    sql_store: str | None = None
    vector_store: str | None = None
    embedding_model: str | None = None
    language_model: str | None = None

    def __post_init__(self) -> None:
        if self.entity_keys_artifact.strip() == "":
            raise ValueError("entity_keys_artifact must not be empty")
        if self.relation_keys_artifact.strip() == "":
            raise ValueError("relation_keys_artifact must not be empty")
        if self.chunk_keys_artifact.strip() == "":
            raise ValueError("chunk_keys_artifact must not be empty")
        if self.vector_metric not in {"cosine", "dot", "l2"}:
            raise ValueError("vector_metric must be one of: cosine, dot, l2")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.llm_max_retries <= 0:
            raise ValueError("llm_max_retries must be greater than zero")
        if self.similarity_threshold is not None and self.similarity_threshold < 0:
            raise ValueError("similarity_threshold must be non-negative")


@dataclass(frozen=True)
class MergeGraphIntoStoreResult:
    """Summary produced by MergeGraphIntoStore."""

    input_entity_count: int
    input_relation_count: int
    inserted_entity_count: int
    inserted_relation_count: int
    merged_entity_count: int
    merged_relation_count: int
    deleted_entity_count: int
    deleted_relation_count: int
    evidence_count: int
    issues: tuple[StepIssue, ...] = ()


@dataclass(frozen=True)
class _StoredEntity:
    entity_id: str
    name: str
    type: str
    subtype: str | None
    description: str
    attributes: dict[str, str]
    source_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class _StoredRelation:
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    source_entity_name: str
    target_entity_name: str
    type: str
    name: str
    description: str
    attributes: dict[str, str]
    source_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class _EntityDecision:
    record: ExtractedEntity
    candidates: tuple[_StoredEntity, ...]


@dataclass(frozen=True)
class _RelationDecision:
    record: ExtractedRelation
    candidates: tuple[_StoredRelation, ...]


@dataclass(frozen=True)
class _MergeStats:
    inserted: int = 0
    merged: int = 0
    deleted: int = 0


class MergeGraphIntoStore:
    """Merge current graph facts into existing SQL and vector graph stores."""

    name = "merge_graph_into_store"

    def __init__(self, config: MergeGraphIntoStoreConfig | None = None) -> None:
        self.config = config or MergeGraphIntoStoreConfig()

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
                    model_ref("language", self.config.language_model),
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
            artifacts=frozenset({"merge_graph_into_store_result"}),
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

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        """Return graph SQL tables and vector collections managed by this step."""
        sql_store_ref = store_ref("sql", self.config.sql_store).key
        vector_store_ref = store_ref("vector", self.config.vector_store).key
        return StepCleanupPlan(
            (
                CleanupTarget(
                    kind="sql_table",
                    value=self.config.table_names.entities,
                    component=sql_store_ref,
                ),
                CleanupTarget(
                    kind="sql_table",
                    value=self.config.table_names.relations,
                    component=sql_store_ref,
                ),
                CleanupTarget(
                    kind="sql_table",
                    value=self.config.table_names.evidence,
                    component=sql_store_ref,
                ),
                CleanupTarget(
                    kind="vector_collection",
                    value=self.config.vector_collections.entities,
                    component=vector_store_ref,
                ),
                CleanupTarget(
                    kind="vector_collection",
                    value=self.config.vector_collections.relations,
                    component=vector_store_ref,
                ),
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Merge current graph facts into SQL and vector stores."""
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
        language_model = _require_language_model(
            context.get_component(model_ref("language", self.config.language_model).key)
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

        issues: list[StepIssue] = []
        storage_config = _storage_config(self.config)
        async with sql_store.transaction() as tx:
            await ensure_graph_tables(tx, storage_config)

        entity_vectors = await embed_entity_records(embedding_model, entities, storage_config)
        entity_stats, entity_id_mapping, entity_name_mapping, entity_evidence_count = (
            await _merge_entities(
                entities=entities,
                entity_vectors=entity_vectors,
                chunk_sources=chunk_sources,
                sql_store=sql_store,
                vector_store=vector_store,
                embedding_model=embedding_model,
                language_model=language_model,
                config=self.config,
                storage_config=storage_config,
                issues=issues,
            )
        )

        normalized_relations, relation_mapping_issues = _apply_entity_mapping_to_relations(
            relations,
            entity_id_mapping,
            entity_name_mapping,
        )
        issues.extend(relation_mapping_issues)
        relation_vectors = await embed_relation_records(
            embedding_model,
            normalized_relations,
            storage_config,
        )
        relation_stats, relation_evidence_count = await _merge_relations(
            relations=normalized_relations,
            relation_vectors=relation_vectors,
            chunk_sources=chunk_sources,
            sql_store=sql_store,
            vector_store=vector_store,
            embedding_model=embedding_model,
            language_model=language_model,
            config=self.config,
            storage_config=storage_config,
            issues=issues,
        )

        result = MergeGraphIntoStoreResult(
            input_entity_count=len(entities),
            input_relation_count=len(relations),
            inserted_entity_count=entity_stats.inserted,
            inserted_relation_count=relation_stats.inserted,
            merged_entity_count=entity_stats.merged,
            merged_relation_count=relation_stats.merged,
            deleted_entity_count=entity_stats.deleted,
            deleted_relation_count=relation_stats.deleted,
            evidence_count=entity_evidence_count + relation_evidence_count,
            issues=tuple(issues),
        )
        context.set_artifact("merge_graph_into_store_result", result)


async def _merge_entities(
    *,
    entities: list[ExtractedEntity],
    entity_vectors: list[VectorRecord],
    chunk_sources: dict[str, ParsedChunk],
    sql_store: SQLStoreProtocol,
    vector_store: VectorStoreProtocol,
    embedding_model: EmbeddingModelProtocol,
    language_model: LanguageModelProtocol,
    config: MergeGraphIntoStoreConfig,
    storage_config: GraphStorageConfig,
    issues: list[StepIssue],
) -> tuple[_MergeStats, dict[str, str], dict[str, str], int]:
    collection = config.vector_collections.entities
    vector_by_id = {record.id: record for record in entity_vectors}
    dimension = _vector_dimension(entity_vectors)
    if dimension:
        await vector_store.create_collection(
            VectorCollectionConfig(
                name=collection,
                dimension=dimension,
                metric=config.vector_metric,  # type: ignore[arg-type]
            )
        )

    decisions: list[_EntityDecision] = []
    passthrough: list[ExtractedEntity] = []
    if not entity_vectors or not await vector_store.has_collection(collection):
        passthrough = entities
    else:
        for entity in entities:
            vector = vector_by_id[entity.entity_id]
            candidates = await _candidate_entities(
                entity=entity,
                vector=vector.vector,
                sql_store=sql_store,
                vector_store=vector_store,
                config=config,
            )
            if not candidates:
                passthrough.append(entity)
                continue
            selected = await _select_entity_candidates(
                language_model=language_model,
                entity=entity,
                candidates=candidates,
                config=config,
                issues=issues,
            )
            if selected:
                decisions.append(_EntityDecision(record=entity, candidates=tuple(selected)))
            else:
                passthrough.append(entity)

    groups = _entity_groups(decisions)
    final_entities: list[ExtractedEntity] = list(passthrough)
    ids_to_delete: set[str] = set()
    merged_old_ids: list[tuple[str, set[str]]] = []
    entity_id_mapping = {entity.entity_id: entity.entity_id for entity in entities}
    entity_name_mapping = {_normalize_name(entity.name): entity.name for entity in entities}
    merged_count = 0

    for group in groups:
        merged = await _merge_entity_group(
            language_model=language_model,
            group=group,
            config=config,
            issues=issues,
        )
        if merged is None:
            final_entities.extend(group.new_entities)
            continue
        merged_count += 1
        final_entities.append(merged)
        group_old_ids = {candidate.entity_id for candidate in group.candidates}
        merged_old_ids.append((merged.entity_id, group_old_ids))
        for entity in group.new_entities:
            entity_id_mapping[entity.entity_id] = merged.entity_id
            entity_name_mapping[_normalize_name(entity.name)] = merged.name
        for candidate in group.candidates:
            ids_to_delete.add(candidate.entity_id)
            entity_id_mapping[candidate.entity_id] = merged.entity_id
            entity_name_mapping[_normalize_name(candidate.name)] = merged.name

    final_vectors = await embed_entity_records(embedding_model, final_entities, storage_config)

    evidence_rows: list[dict[str, object]] = []
    for entity in final_entities:
        evidence_rows.extend(available_evidence_rows_for_entity(entity, chunk_sources))
    for merged_entity_id, old_ids in merged_old_ids:
        old_rows = await _fetch_evidence_rows(
            sql_store,
            config.table_names.evidence,
            old_ids,
            fact_type="entity",
        )
        evidence_rows.extend(_retarget_evidence_rows(old_rows, merged_entity_id))
    evidence_rows = _dedupe_evidence_rows(evidence_rows)

    async with sql_store.transaction() as tx:
        await _delete_graph_rows(
            tx,
            entity_table=config.table_names.entities,
            relation_table=None,
            evidence_table=config.table_names.evidence,
            entity_ids=ids_to_delete,
            relation_ids=(),
        )
        for batch in batches([entity_row(entity) for entity in final_entities], config.batch_size):
            await upsert_entity_rows(tx, config.table_names.entities, batch)
        for batch in batches(evidence_rows, config.batch_size):
            await upsert_evidence_rows(tx, config.table_names.evidence, batch)

    if ids_to_delete:
        await vector_store.delete(collection, sorted(ids_to_delete))
    if final_vectors:
        await vector_store.upsert(collection, final_vectors)

    return (
        _MergeStats(
            inserted=len(final_entities) - merged_count,
            merged=merged_count,
            deleted=len(ids_to_delete),
        ),
        entity_id_mapping,
        entity_name_mapping,
        len(evidence_rows),
    )


async def _merge_relations(
    *,
    relations: list[ExtractedRelation],
    relation_vectors: list[VectorRecord],
    chunk_sources: dict[str, ParsedChunk],
    sql_store: SQLStoreProtocol,
    vector_store: VectorStoreProtocol,
    embedding_model: EmbeddingModelProtocol,
    language_model: LanguageModelProtocol,
    config: MergeGraphIntoStoreConfig,
    storage_config: GraphStorageConfig,
    issues: list[StepIssue],
) -> tuple[_MergeStats, int]:
    collection = config.vector_collections.relations
    vector_by_id = {record.id: record for record in relation_vectors}
    dimension = _vector_dimension(relation_vectors)
    if dimension:
        await vector_store.create_collection(
            VectorCollectionConfig(
                name=collection,
                dimension=dimension,
                metric=config.vector_metric,  # type: ignore[arg-type]
            )
        )

    decisions: list[_RelationDecision] = []
    passthrough: list[ExtractedRelation] = []
    if not relation_vectors or not await vector_store.has_collection(collection):
        passthrough = relations
    else:
        for relation in relations:
            vector = vector_by_id[relation.relation_id]
            candidates = await _candidate_relations(
                relation=relation,
                vector=vector.vector,
                sql_store=sql_store,
                vector_store=vector_store,
                config=config,
            )
            if not candidates:
                passthrough.append(relation)
                continue
            selected = await _select_relation_candidates(
                language_model=language_model,
                relation=relation,
                candidates=candidates,
                config=config,
                issues=issues,
            )
            if selected:
                decisions.append(_RelationDecision(record=relation, candidates=tuple(selected)))
            else:
                passthrough.append(relation)

    groups = _relation_groups(decisions)
    final_relations: list[ExtractedRelation] = list(passthrough)
    ids_to_delete: set[str] = set()
    merged_old_ids: list[tuple[str, set[str]]] = []
    merged_count = 0

    for group in groups:
        merged = await _merge_relation_group(
            language_model=language_model,
            group=group,
            config=config,
            issues=issues,
        )
        if merged is None:
            final_relations.extend(group.new_relations)
            continue
        merged_count += 1
        final_relations.append(merged)
        group_old_ids = {candidate.relation_id for candidate in group.candidates}
        merged_old_ids.append((merged.relation_id, group_old_ids))
        for candidate in group.candidates:
            ids_to_delete.add(candidate.relation_id)

    final_vectors = await embed_relation_records(embedding_model, final_relations, storage_config)

    evidence_rows: list[dict[str, object]] = []
    for relation in final_relations:
        evidence_rows.extend(available_evidence_rows_for_relation(relation, chunk_sources))
    for merged_relation_id, old_ids in merged_old_ids:
        old_rows = await _fetch_evidence_rows(
            sql_store,
            config.table_names.evidence,
            old_ids,
            fact_type="relation",
        )
        evidence_rows.extend(_retarget_evidence_rows(old_rows, merged_relation_id))
    evidence_rows = _dedupe_evidence_rows(evidence_rows)

    async with sql_store.transaction() as tx:
        await _delete_graph_rows(
            tx,
            entity_table=None,
            relation_table=config.table_names.relations,
            evidence_table=config.table_names.evidence,
            entity_ids=(),
            relation_ids=ids_to_delete,
        )
        for batch in batches([relation_row(relation) for relation in final_relations], config.batch_size):
            await upsert_relation_rows(tx, config.table_names.relations, batch)
        for batch in batches(evidence_rows, config.batch_size):
            await upsert_evidence_rows(tx, config.table_names.evidence, batch)

    if ids_to_delete:
        await vector_store.delete(collection, sorted(ids_to_delete))
    if final_vectors:
        await vector_store.upsert(collection, final_vectors)

    return (
        _MergeStats(
            inserted=len(final_relations) - merged_count,
            merged=merged_count,
            deleted=len(ids_to_delete),
        ),
        len(evidence_rows),
    )


@dataclass
class _EntityGroup:
    new_entities: list[ExtractedEntity]
    candidates: list[_StoredEntity]


@dataclass
class _RelationGroup:
    new_relations: list[ExtractedRelation]
    candidates: list[_StoredRelation]


def _entity_groups(decisions: list[_EntityDecision]) -> list[_EntityGroup]:
    groups: list[tuple[set[str], _EntityGroup]] = []
    for decision in decisions:
        ids = {decision.record.entity_id}
        ids.update(candidate.entity_id for candidate in decision.candidates)
        _add_entity_group(groups, ids, decision)
    return [group for _, group in groups]


def _add_entity_group(
    groups: list[tuple[set[str], _EntityGroup]],
    ids: set[str],
    decision: _EntityDecision,
) -> None:
    matches = [index for index, (existing, _) in enumerate(groups) if existing & ids]
    new_group = _EntityGroup([decision.record], list(decision.candidates))
    if not matches:
        groups.append((ids, new_group))
        return
    base_ids, base_group = groups[matches[0]]
    base_ids.update(ids)
    _append_unique(base_group.new_entities, new_group.new_entities, key=lambda item: item.entity_id)
    _append_unique(base_group.candidates, new_group.candidates, key=lambda item: item.entity_id)
    for index in reversed(matches[1:]):
        other_ids, other_group = groups.pop(index)
        base_ids.update(other_ids)
        _append_unique(base_group.new_entities, other_group.new_entities, key=lambda item: item.entity_id)
        _append_unique(base_group.candidates, other_group.candidates, key=lambda item: item.entity_id)


def _relation_groups(decisions: list[_RelationDecision]) -> list[_RelationGroup]:
    groups: list[tuple[set[str], _RelationGroup]] = []
    for decision in decisions:
        ids = {decision.record.relation_id}
        ids.update(candidate.relation_id for candidate in decision.candidates)
        _add_relation_group(groups, ids, decision)
    return [group for _, group in groups]


def _add_relation_group(
    groups: list[tuple[set[str], _RelationGroup]],
    ids: set[str],
    decision: _RelationDecision,
) -> None:
    matches = [index for index, (existing, _) in enumerate(groups) if existing & ids]
    new_group = _RelationGroup([decision.record], list(decision.candidates))
    if not matches:
        groups.append((ids, new_group))
        return
    base_ids, base_group = groups[matches[0]]
    base_ids.update(ids)
    _append_unique(base_group.new_relations, new_group.new_relations, key=lambda item: item.relation_id)
    _append_unique(base_group.candidates, new_group.candidates, key=lambda item: item.relation_id)
    for index in reversed(matches[1:]):
        other_ids, other_group = groups.pop(index)
        base_ids.update(other_ids)
        _append_unique(base_group.new_relations, other_group.new_relations, key=lambda item: item.relation_id)
        _append_unique(base_group.candidates, other_group.candidates, key=lambda item: item.relation_id)


async def _candidate_entities(
    *,
    entity: ExtractedEntity,
    vector: list[float],
    sql_store: SQLStoreProtocol,
    vector_store: VectorStoreProtocol,
    config: MergeGraphIntoStoreConfig,
) -> list[_StoredEntity]:
    if not await vector_store.has_collection(config.vector_collections.entities):
        return []
    hits = await vector_store.search(
        config.vector_collections.entities,
        VectorQuery(vector=vector, top_k=config.top_k),
    )
    candidates: list[_StoredEntity] = []
    for hit in hits:
        if hit.id == entity.entity_id:
            continue
        if config.similarity_threshold is not None and hit.score < config.similarity_threshold:
            continue
        candidate = await _fetch_entity(sql_store, config.table_names, hit.id)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


async def _candidate_relations(
    *,
    relation: ExtractedRelation,
    vector: list[float],
    sql_store: SQLStoreProtocol,
    vector_store: VectorStoreProtocol,
    config: MergeGraphIntoStoreConfig,
) -> list[_StoredRelation]:
    if not await vector_store.has_collection(config.vector_collections.relations):
        return []
    hits = await vector_store.search(
        config.vector_collections.relations,
        VectorQuery(vector=vector, top_k=config.top_k),
    )
    candidates: list[_StoredRelation] = []
    for hit in hits:
        if hit.id == relation.relation_id:
            continue
        if config.similarity_threshold is not None and hit.score < config.similarity_threshold:
            continue
        candidate = await _fetch_relation(sql_store, config.table_names, hit.id)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


async def _select_entity_candidates(
    *,
    language_model: LanguageModelProtocol,
    entity: ExtractedEntity,
    candidates: list[_StoredEntity],
    config: MergeGraphIntoStoreConfig,
    issues: list[StepIssue],
) -> list[_StoredEntity]:
    payload = [_entity_for_llm(entity), *[_stored_entity_for_llm(candidate) for candidate in candidates]]
    parsed = await invoke_graph_merge_json(
        language_model,
        entity_merge_prompt(payload),
        step_name=MergeGraphIntoStore.name,
        temperature=config.temperature,
        max_retries=config.llm_max_retries,
        subject=IssueSubject(type="entity", id=entity.entity_id),
        issues=issues,
    )
    mapping = parsed.get("mapping_table") if isinstance(parsed, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        return []
    required_names = mapping_values(mapping)
    selected = [
        candidate
        for candidate in candidates
        if _normalize_name(candidate.name) in required_names
    ]
    return selected


async def _select_relation_candidates(
    *,
    language_model: LanguageModelProtocol,
    relation: ExtractedRelation,
    candidates: list[_StoredRelation],
    config: MergeGraphIntoStoreConfig,
    issues: list[StepIssue],
) -> list[_StoredRelation]:
    payload = [
        _relation_for_llm(relation),
        *[_stored_relation_for_llm(candidate) for candidate in candidates],
    ]
    parsed = await invoke_graph_merge_json(
        language_model,
        relation_merge_prompt(payload),
        step_name=MergeGraphIntoStore.name,
        temperature=config.temperature,
        max_retries=config.llm_max_retries,
        subject=IssueSubject(type="relation", id=relation.relation_id),
        issues=issues,
    )
    mapping = parsed.get("mapping_table") if isinstance(parsed, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        return []
    required = mapping_values(mapping)
    selected = [
        candidate
        for candidate in candidates
        if candidate.relation_id in required or _relation_pair_key(candidate) in required
    ]
    return selected


async def _merge_entity_group(
    *,
    language_model: LanguageModelProtocol,
    group: _EntityGroup,
    config: MergeGraphIntoStoreConfig,
    issues: list[StepIssue],
) -> ExtractedEntity | None:
    payload = [
        *[_entity_for_llm(entity) for entity in group.new_entities],
        *[_stored_entity_for_llm(candidate) for candidate in group.candidates],
    ]
    parsed = await invoke_graph_merge_json(
        language_model,
        entity_merge_prompt(payload),
        step_name=MergeGraphIntoStore.name,
        temperature=config.temperature,
        max_retries=config.llm_max_retries,
        subject=IssueSubject(type="entity_group", id=_group_id([item["Id"] for item in payload])),
        issues=issues,
    )
    if not isinstance(parsed, dict):
        return None
    mapping = parsed.get("mapping_table")
    entity_list = parsed.get("entity_list")
    if not isinstance(mapping, dict) or not mapping or not isinstance(entity_list, list):
        return None
    merged_items = [item for item in entity_list if isinstance(item, dict) and item.get("merge_tag")]
    if not merged_items:
        return None
    item = merged_items[0]
    source_ids = tuple(
        sorted(
            {
                *[entity.entity_id for entity in group.new_entities],
                *[candidate.entity_id for candidate in group.candidates],
            }
        )
    )
    source_chunk_ids = _unique(
        [
            *[chunk for entity in group.new_entities for chunk in entity.source_chunk_ids],
            *[chunk for candidate in group.candidates for chunk in candidate.source_chunk_ids],
        ]
    )
    base = group.new_entities[0]
    name = _first_text(item, "NodeName", "name") or base.name
    description = _first_text(item, "Description", "description") or base.description
    return ExtractedEntity(
        entity_id=_fact_id("entity", name, description, source_ids),
        chunk_id=source_chunk_ids[0],
        document_id=base.document_id,
        name=name,
        type=_first_text(item, "Type", "type") or base.type,
        subtype=_first_text(item, "Subtype", "SubType", "subtype") or base.subtype,
        description=description,
        attributes=_string_mapping(item.get("Attr") or item.get("attributes") or {}),
        source_chunk_ids=tuple(source_chunk_ids),
    )


async def _merge_relation_group(
    *,
    language_model: LanguageModelProtocol,
    group: _RelationGroup,
    config: MergeGraphIntoStoreConfig,
    issues: list[StepIssue],
) -> ExtractedRelation | None:
    payload = [
        *[_relation_for_llm(relation) for relation in group.new_relations],
        *[_stored_relation_for_llm(candidate) for candidate in group.candidates],
    ]
    parsed = await invoke_graph_merge_json(
        language_model,
        relation_merge_prompt(payload),
        step_name=MergeGraphIntoStore.name,
        temperature=config.temperature,
        max_retries=config.llm_max_retries,
        subject=IssueSubject(type="relation_group", id=_group_id([item["Id"] for item in payload])),
        issues=issues,
    )
    if not isinstance(parsed, dict):
        return None
    mapping = parsed.get("mapping_table")
    relation_list = parsed.get("relation_list")
    if not isinstance(mapping, dict) or not mapping or not isinstance(relation_list, list):
        return None
    merged_items = [item for item in relation_list if isinstance(item, dict) and item.get("merge_tag")]
    if not merged_items:
        return None
    item = merged_items[0]
    base = group.new_relations[0]
    source_ids = tuple(
        sorted(
            {
                *[relation.relation_id for relation in group.new_relations],
                *[candidate.relation_id for candidate in group.candidates],
            }
        )
    )
    source_chunk_ids = _unique(
        [
            *[chunk for relation in group.new_relations for chunk in relation.source_chunk_ids],
            *[chunk for candidate in group.candidates for chunk in candidate.source_chunk_ids],
        ]
    )
    source_name = _first_text(item, "Node1", "source_entity_name") or base.source_entity_name
    target_name = _first_text(item, "Node2", "target_entity_name") or base.target_entity_name
    relation_name = _first_text(item, "Relation", "name") or base.name
    description = _first_text(item, "Description", "description") or base.description
    return ExtractedRelation(
        relation_id=_fact_id("relation", source_name, target_name, relation_name, description, source_ids),
        chunk_id=source_chunk_ids[0],
        document_id=base.document_id,
        source_entity_id=base.source_entity_id,
        target_entity_id=base.target_entity_id,
        source_entity_name=source_name,
        target_entity_name=target_name,
        type=_first_text(item, "Type", "type") or base.type,
        name=relation_name,
        description=description,
        attributes=_string_mapping(item.get("Attr") or item.get("attributes") or {}),
        source_chunk_ids=tuple(source_chunk_ids),
    )


def _apply_entity_mapping_to_relations(
    relations: list[ExtractedRelation],
    entity_id_mapping: dict[str, str],
    entity_name_mapping: dict[str, str],
) -> tuple[list[ExtractedRelation], list[StepIssue]]:
    normalized: list[ExtractedRelation] = []
    issues: list[StepIssue] = []
    for relation in relations:
        source_id = entity_id_mapping.get(relation.source_entity_id, relation.source_entity_id)
        target_id = entity_id_mapping.get(relation.target_entity_id, relation.target_entity_id)
        source_name = entity_name_mapping.get(_normalize_name(relation.source_entity_name), relation.source_entity_name)
        target_name = entity_name_mapping.get(_normalize_name(relation.target_entity_name), relation.target_entity_name)
        if source_id == target_id:
            issues.append(
                StepIssue(
                    step=MergeGraphIntoStore.name,
                    subject=IssueSubject(type="relation", id=relation.relation_id),
                    code="self_relation_after_entity_merge",
                    message="Relation endpoints resolved to the same entity after entity merge.",
                    resolution=IssueResolution(
                        action="skipped_relation",
                        outcome="The relation was not written to the graph store.",
                    ),
                )
            )
            continue
        normalized.append(
            ExtractedRelation(
                relation_id=relation.relation_id,
                chunk_id=relation.chunk_id,
                document_id=relation.document_id,
                source_entity_id=source_id,
                target_entity_id=target_id,
                source_entity_name=source_name,
                target_entity_name=target_name,
                type=relation.type,
                name=relation.name,
                description=relation.description,
                attributes=relation.attributes,
                source_chunk_ids=relation.source_chunk_ids,
            )
        )
    return normalized, issues


async def _fetch_entity(
    sql_store: SQLStoreProtocol,
    table_names: GraphTableNames,
    entity_id: str,
) -> _StoredEntity | None:
    row = await sql_store.fetch_one(
        f"SELECT * FROM {table_names.entities} WHERE entity_id = :entity_id",
        {"entity_id": entity_id},
    )
    if row is None:
        return None
    evidence = await _fetch_evidence_rows(sql_store, table_names.evidence, (entity_id,), fact_type="entity")
    return _StoredEntity(
        entity_id=str(row["entity_id"]),
        name=str(row["entity_name"]),
        type=str(row["entity_type"]),
        subtype=str(row["entity_subtype"]) if row.get("entity_subtype") else None,
        description=str(row["description"]),
        attributes=_parse_attributes(row.get("attributes")),
        source_chunk_ids=tuple(str(item["chunk_id"]) for item in evidence),
    )


async def _fetch_relation(
    sql_store: SQLStoreProtocol,
    table_names: GraphTableNames,
    relation_id: str,
) -> _StoredRelation | None:
    row = await sql_store.fetch_one(
        f"SELECT * FROM {table_names.relations} WHERE relation_id = :relation_id",
        {"relation_id": relation_id},
    )
    if row is None:
        return None
    evidence = await _fetch_evidence_rows(
        sql_store,
        table_names.evidence,
        (relation_id,),
        fact_type="relation",
    )
    return _StoredRelation(
        relation_id=str(row["relation_id"]),
        source_entity_id=str(row["source_entity_id"]),
        target_entity_id=str(row["target_entity_id"]),
        source_entity_name=str(row["source_entity_name"]),
        target_entity_name=str(row["target_entity_name"]),
        type=str(row["relation_type"]),
        name=str(row["relation_name"]),
        description=str(row["description"]),
        attributes=_parse_attributes(row.get("attributes")),
        source_chunk_ids=tuple(str(item["chunk_id"]) for item in evidence),
    )


async def _fetch_evidence_rows(
    sql_store: SQLStoreProtocol,
    table: str,
    fact_ids: set[str] | tuple[str, ...] | list[str],
    *,
    fact_type: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fact_id in fact_ids:
        rows.extend(
            await sql_store.fetch_all(
                f"SELECT * FROM {table} WHERE fact_id = :fact_id AND fact_type = :fact_type",
                {"fact_id": fact_id, "fact_type": fact_type},
            )
        )
    return rows


async def _delete_graph_rows(
    tx: SQLStoreProtocol,
    *,
    entity_table: str | None,
    relation_table: str | None,
    evidence_table: str,
    entity_ids: set[str] | tuple[str, ...],
    relation_ids: set[str] | tuple[str, ...],
) -> None:
    for entity_id in entity_ids:
        if entity_table is not None:
            await tx.execute(f"DELETE FROM {entity_table} WHERE entity_id = :id", {"id": entity_id})
        await tx.execute(
            f"DELETE FROM {evidence_table} WHERE fact_id = :id AND fact_type = 'entity'",
            {"id": entity_id},
        )
    for relation_id in relation_ids:
        if relation_table is not None:
            await tx.execute(f"DELETE FROM {relation_table} WHERE relation_id = :id", {"id": relation_id})
        await tx.execute(
            f"DELETE FROM {evidence_table} WHERE fact_id = :id AND fact_type = 'relation'",
            {"id": relation_id},
        )


def _retarget_evidence_rows(rows: list[dict[str, object]], fact_id: str) -> list[dict[str, object]]:
    return [
        {
            "fact_id": fact_id,
            "fact_type": row["fact_type"],
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "source_key": row["source_key"],
            "source_name": row["source_name"],
            "metadata": row["metadata"],
        }
        for row in rows
    ]


def _dedupe_evidence_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, object]] = []
    for row in rows:
        key = (str(row["fact_id"]), str(row["fact_type"]), str(row["chunk_id"]))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _entity_for_llm(entity: ExtractedEntity) -> dict[str, Any]:
    return {
        "Id": entity.entity_id,
        "NodeName": entity.name,
        "Type": entity.type,
        "SubType": entity.subtype,
        "Description": entity.description,
        "Attr": dict(entity.attributes),
        "chunk_id": list(entity.source_chunk_ids),
    }


def _stored_entity_for_llm(entity: _StoredEntity) -> dict[str, Any]:
    return {
        "Id": entity.entity_id,
        "NodeName": entity.name,
        "Type": entity.type,
        "SubType": entity.subtype,
        "Description": entity.description,
        "Attr": entity.attributes,
        "chunk_id": list(entity.source_chunk_ids),
    }


def _relation_for_llm(relation: ExtractedRelation) -> dict[str, Any]:
    return {
        "Id": relation.relation_id,
        "Node1": relation.source_entity_name,
        "Node2": relation.target_entity_name,
        "Relation": relation.name,
        "Type": relation.type,
        "Description": relation.description,
        "Attr": dict(relation.attributes),
        "chunk_id": list(relation.source_chunk_ids),
    }


def _stored_relation_for_llm(relation: _StoredRelation) -> dict[str, Any]:
    return {
        "Id": relation.relation_id,
        "Node1": relation.source_entity_name,
        "Node2": relation.target_entity_name,
        "Relation": relation.name,
        "Type": relation.type,
        "Description": relation.description,
        "Attr": relation.attributes,
        "chunk_id": list(relation.source_chunk_ids),
    }


def _relation_pair_key(relation: _StoredRelation) -> str:
    return f"{_normalize_name(relation.source_entity_name)}||{_normalize_name(relation.target_entity_name)}"


def _storage_config(config: MergeGraphIntoStoreConfig) -> GraphStorageConfig:
    return GraphStorageConfig(
        table_names=config.table_names,
        vector_collections=config.vector_collections,
        vector_metric=config.vector_metric,
        batch_size=config.batch_size,
        trace_step=MergeGraphIntoStore.name,
    )


def _vector_dimension(records: list[VectorRecord]) -> int:
    if not records:
        return 0
    return len(records[0].vector)


def _fact_id(prefix: str, *parts: object) -> str:
    payload = "\n".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:16]}"


def _group_id(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()[:16]


def _normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _first_text(data: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key).strip(): str(item).strip()
        for key, item in value.items()
        if str(key).strip() and str(item).strip()
    }


def _parse_attributes(value: object) -> dict[str, str]:
    if isinstance(value, dict):
        return _string_mapping(value)
    if isinstance(value, str) and value.strip():
        try:
            return _string_mapping(json.loads(value))
        except json.JSONDecodeError:
            return {}
    return {}


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _append_unique(target: list[_T], values: list[_T], *, key: Any) -> None:
    existing = {key(item) for item in target}
    for value in values:
        marker = key(value)
        if marker in existing:
            continue
        existing.add(marker)
        target.append(value)


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


def _require_language_model(component: object) -> LanguageModelProtocol:
    if not isinstance(component, LanguageModelProtocol):
        raise TypeError("models.language must satisfy LanguageModelProtocol")
    return component


__all__ = [
    "MergeGraphIntoStore",
    "MergeGraphIntoStoreConfig",
    "MergeGraphIntoStoreResult",
]
