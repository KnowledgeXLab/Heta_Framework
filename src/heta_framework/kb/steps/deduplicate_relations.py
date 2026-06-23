"""Deduplicate extracted relation artifacts within the current build batch."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from heta_framework.common.models import EmbeddingRequest, ModelOptions, ModelRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol, LanguageModelProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb.graphing import ExtractedRelation, make_deduplicated_relation_id
from heta_framework.kb.graphing.prompts import (
    RELATION_DEDUPLICATION_PROMPT,
    RELATION_DEDUPLICATION_RETRY_PROMPT,
    RELATION_DEDUPLICATION_SYSTEM_PROMPT,
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


RELATION_DEDUPLICATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relation": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "attributes": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["type", "name", "description", "attributes"],
        }
    },
    "required": ["relation"],
}


@dataclass(frozen=True)
class DeduplicateRelationsConfig:
    """Configuration for DeduplicateRelations."""

    deduplicated_relations_prefix: str = "deduplicated_relations"
    relation_keys_artifact: str = "relation_keys"
    deduplicated_relation_keys_artifact: str = "deduplicated_relation_keys"
    entity_id_mapping_artifact: str | None = "entity_id_mapping"
    semantic_merge: bool = True
    similarity_threshold: float = 0.9
    max_rounds: int = 10
    llm_batch_size: int = 20
    semantic_batch_size: int = 100
    semantic_batch_count: int = 4
    max_attempts: int = 3
    temperature: float = 0.0
    object_store: str | None = None
    language_model: str | None = None
    embedding_model: str | None = None

    def __post_init__(self) -> None:
        validate_object_prefix(self.deduplicated_relations_prefix)
        if self.relation_keys_artifact.strip() == "":
            raise ValueError("relation_keys_artifact must not be empty")
        if self.deduplicated_relation_keys_artifact.strip() == "":
            raise ValueError("deduplicated_relation_keys_artifact must not be empty")
        if (
            self.entity_id_mapping_artifact is not None
            and self.entity_id_mapping_artifact.strip() == ""
        ):
            raise ValueError("entity_id_mapping_artifact must not be empty")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be greater than zero")
        if self.llm_batch_size <= 0:
            raise ValueError("llm_batch_size must be greater than zero")
        if self.semantic_batch_size <= 0:
            raise ValueError("semantic_batch_size must be greater than zero")
        if self.semantic_batch_count <= 0:
            raise ValueError("semantic_batch_count must be greater than zero")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")


@dataclass(frozen=True)
class DeduplicateRelationsResult:
    """Artifacts produced by DeduplicateRelations."""

    relation_keys: tuple[str, ...]
    input_relation_count: int
    output_relation_count: int
    exact_merge_count: int
    semantic_merge_count: int
    failed_group_count: int
    exact_round_count: int
    semantic_round_count: int
    issues: tuple[StepIssue, ...]


@dataclass(frozen=True)
class _RelationGroup:
    relation: ExtractedRelation
    member_relation_ids: tuple[str, ...]


class DeduplicateRelations:
    """Merge duplicate ExtractedRelation artifacts while preserving the relation schema."""

    name = "deduplicate_relations"

    def __init__(self, config: DeduplicateRelationsConfig | None = None) -> None:
        self.config = config or DeduplicateRelationsConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components and artifacts required by this step."""
        components = {
            store_ref("objects", self.config.object_store),
            model_ref("language", self.config.language_model),
        }
        if self.config.semantic_merge:
            components.add(model_ref("embedding", self.config.embedding_model))
        artifacts = {self.config.relation_keys_artifact}
        if self.config.entity_id_mapping_artifact is not None:
            artifacts.add(self.config.entity_id_mapping_artifact)
        return StepRequirements(components=frozenset(components), artifacts=frozenset(artifacts))

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts produced by this step."""
        return StepCapabilities(
            artifacts=frozenset(
                {
                    "deduplicate_relations_result",
                    self.config.deduplicated_relation_keys_artifact,
                    "relation_id_mapping",
                }
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Run relation deduplication and store ExtractedRelation JSON objects."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        language_model = _require_language_model(
            context.get_component(model_ref("language", self.config.language_model).key)
        )
        embedding_model: EmbeddingModelProtocol | None = None
        if self.config.semantic_merge:
            embedding_model = _require_embedding_model(
                context.get_component(model_ref("embedding", self.config.embedding_model).key)
            )

        relation_keys = tuple(context.get_artifact(self.config.relation_keys_artifact))
        relations = [
            ExtractedRelation.from_json(await object_store.get(key)) for key in relation_keys
        ]
        entity_id_mapping = _entity_id_mapping(context, self.config.entity_id_mapping_artifact)
        relations = [_apply_entity_mapping(relation, entity_id_mapping) for relation in relations]
        relations = [_normalize_relation_endpoints(relation) for relation in relations]

        (
            groups,
            exact_merge_count,
            failed_exact,
            exact_round_count,
            exact_issues,
        ) = await _dedup_exact_relations(
            relations,
            language_model=language_model,
            config=self.config,
        )
        semantic_merge_count = 0
        failed_semantic = 0
        semantic_round_count = 0
        semantic_issues: tuple[StepIssue, ...] = ()
        if self.config.semantic_merge and embedding_model is not None:
            (
                groups,
                semantic_merge_count,
                failed_semantic,
                semantic_round_count,
                semantic_issues,
            ) = await _dedup_semantic_relations(
                groups,
                language_model=language_model,
                embedding_model=embedding_model,
                config=self.config,
            )

        output_keys: list[str] = []
        relation_id_mapping: dict[str, str] = {}
        for group in groups:
            key = join_object_key(
                self.config.deduplicated_relations_prefix,
                f"{group.relation.relation_id}.json",
            )
            await object_store.put(key, group.relation.to_json_bytes())
            output_keys.append(key)
            for member_id in group.member_relation_ids:
                relation_id_mapping[member_id] = group.relation.relation_id

        result = DeduplicateRelationsResult(
            relation_keys=tuple(output_keys),
            input_relation_count=len(relations),
            output_relation_count=len(output_keys),
            exact_merge_count=exact_merge_count,
            semantic_merge_count=semantic_merge_count,
            failed_group_count=failed_exact + failed_semantic,
            exact_round_count=exact_round_count,
            semantic_round_count=semantic_round_count,
            issues=tuple([*exact_issues, *semantic_issues]),
        )
        context.set_artifact("deduplicate_relations_result", result)
        context.set_artifact(self.config.deduplicated_relation_keys_artifact, result.relation_keys)
        context.set_artifact("relation_id_mapping", relation_id_mapping)


async def _dedup_exact_relations(
    relations: list[ExtractedRelation],
    *,
    language_model: LanguageModelProtocol,
    config: DeduplicateRelationsConfig,
) -> tuple[list[_RelationGroup], int, int, int, tuple[StepIssue, ...]]:
    current = [
        _RelationGroup(relation=relation, member_relation_ids=(relation.relation_id,))
        for relation in relations
    ]
    total_merged = 0
    total_failed = 0
    rounds = 0
    issues: list[StepIssue] = []
    suppressed_keys: set[tuple[str, str, str, str]] = set()
    while rounds < config.max_rounds:
        uniques, duplicates = _split_relation_uniques_duplicates(current)
        if not duplicates:
            break
        if all(key in suppressed_keys for key in duplicates):
            break
        rounds += 1
        occupied_keys = set(uniques)
        next_groups: list[_RelationGroup] = []
        for key, first_group in uniques.items():
            duplicate_group = duplicates.get(key)
            if duplicate_group is None:
                next_groups.append(first_group)
                continue
            if key in suppressed_keys:
                next_groups.extend(duplicate_group)
                continue
            merged_groups, reason = await _merge_exact_relation_group(
                tuple(duplicate_group),
                node1=first_group.relation.source_entity_name,
                node2=first_group.relation.target_entity_name,
                language_model=language_model,
                config=config,
            )
            if merged_groups is None:
                total_failed += 1
                suppressed_keys.add(key)
                issues.append(
                    _dedup_issue(
                        _relation_group_key(duplicate_group[0].relation),
                        reason or "relation deduplication failed",
                    )
                )
                next_groups.extend(duplicate_group)
                continue
            main, splits = merged_groups[0], merged_groups[1:]
            next_groups.append(main)
            occupied_keys.add(_exact_relation_key(main.relation))
            for split in splits:
                split_key = _exact_relation_key(split.relation)
                if split_key not in occupied_keys:
                    next_groups.append(split)
                    occupied_keys.add(split_key)
            total_merged += max(0, len(duplicate_group) - len(merged_groups))
        current = next_groups
    if rounds >= config.max_rounds:
        _, remaining_duplicates = _split_relation_uniques_duplicates(current)
        for groups in remaining_duplicates.values():
            issues.append(
                _dedup_issue(
                    _relation_group_key(groups[0].relation),
                    "max_rounds reached with duplicate relations remaining",
                    code="max_rounds_reached",
                )
            )

    return current, total_merged, total_failed, rounds, tuple(issues)


async def _merge_group_objects(
    grouped: list[tuple[_RelationGroup, ...]],
    *,
    language_model: LanguageModelProtocol,
    config: DeduplicateRelationsConfig,
) -> tuple[list[_RelationGroup], int, int]:
    merged: list[_RelationGroup] = []
    merge_count = 0
    failed_count = 0
    for group in grouped:
        if len(group) == 1:
            merged.append(group[0])
            continue
        merged_group, _ = await _merge_relation_group(
            group,
            language_model=language_model,
            config=config,
        )
        if merged_group is None:
            failed_count += 1
            merged.extend(group)
            continue
        merged_groups = merged_group if isinstance(merged_group, list) else [merged_group]
        merge_count += max(0, len(group) - len(merged_groups))
        merged.extend(merged_groups)
    return merged, merge_count, failed_count


async def _merge_exact_relation_group(
    group: tuple[_RelationGroup, ...],
    *,
    node1: str,
    node2: str,
    language_model: LanguageModelProtocol,
    config: DeduplicateRelationsConfig,
) -> tuple[list[_RelationGroup] | None, str | None]:
    split_groups: list[_RelationGroup] = []
    last_reason: str | None = None

    async def merge_batch(batch: tuple[_RelationGroup, ...]) -> _RelationGroup | None:
        nonlocal last_reason
        merged, reason = await _merge_relation_group(
            batch,
            language_model=language_model,
            config=config,
        )
        last_reason = reason
        if merged is None:
            return None
        main_group, extra_groups = _select_main_relation_group(
            merged,
            node1=node1,
            node2=node2,
        )
        split_groups.extend(extra_groups)
        return main_group

    if len(group) <= config.llm_batch_size:
        main = await merge_batch(group)
        if main is None:
            return None, last_reason or "relation batch merge failed"
        return [main, *split_groups], None

    accumulated: _RelationGroup | None = None
    for start in range(0, len(group), config.llm_batch_size):
        batch = group[start : start + config.llm_batch_size]
        if accumulated is not None:
            batch = (accumulated, *batch)
        accumulated = await merge_batch(tuple(batch))
        if accumulated is None:
            return None, last_reason or "relation batch merge failed"
    if accumulated is None:
        return None, last_reason or "relation batch merge failed"
    return [accumulated, *split_groups], None


async def _merge_relation_group(
    group: tuple[_RelationGroup, ...],
    *,
    language_model: LanguageModelProtocol,
    config: DeduplicateRelationsConfig,
) -> tuple[_RelationGroup | list[_RelationGroup] | None, str | None]:
    last_error = ""
    relations = tuple(item.relation for item in group)
    member_ids = tuple(member_id for item in group for member_id in item.member_relation_ids)
    for attempt in range(config.max_attempts):
        prompt = _build_relation_dedup_prompt(relations, error=last_error if attempt > 0 else None)
        try:
            result = await language_model.invoke(
                ModelRequest(
                    prompt=prompt,
                    system_prompt=RELATION_DEDUPLICATION_SYSTEM_PROMPT,
                    options=ModelOptions(
                        temperature=config.temperature,
                        response_format={"type": "json_object"},
                    ),
                    response_schema=RELATION_DEDUPLICATION_RESPONSE_SCHEMA,
                    trace_context={
                        "step": DeduplicateRelations.name,
                        "attempt": attempt + 1,
                        "member_relation_ids": member_ids,
                    },
                )
            )
            payload = result.parsed if result.parsed is not None else result.text
            return _relation_groups_from_dedup_payload(payload, relations, member_ids), None
        except Exception as exc:
            last_error = str(exc) or exc.__class__.__name__
    return None, last_error or "relation deduplication failed"


def _relation_groups_from_dedup_payload(
    payload: Any,
    relations: tuple[ExtractedRelation, ...],
    member_relation_ids: tuple[str, ...],
) -> _RelationGroup | list[_RelationGroup]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    raw_relations: list[dict[str, Any]]
    if isinstance(payload, list):
        raw_relations = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        if isinstance(payload.get("relation"), dict):
            raw_relations = [payload["relation"]]
        elif isinstance(payload.get("relations"), list):
            raw_relations = [item for item in payload["relations"] if isinstance(item, dict)]
        else:
            raw_relations = [payload]
    else:
        raise ValueError("relation deduplication response must be a JSON object or array")
    if not raw_relations:
        raise ValueError("relation deduplication response did not contain relations")

    representative_pair = (
        _normalize_name(relations[0].source_entity_name),
        _normalize_name(relations[0].target_entity_name),
    )
    main_index = next(
        (
            index
            for index, raw_relation in enumerate(raw_relations)
            if _raw_relation_pair(raw_relation) == representative_pair
        ),
        0,
    )
    groups = []
    for index, raw_relation in enumerate(raw_relations):
        group_member_ids = member_relation_ids
        is_main = index == main_index
        if not is_main:
            description = str(
                raw_relation.get("description") or raw_relation.get("Description") or ""
            )
            group_member_ids = (f"split:{index}:{description}",)
        groups.append(
            _RelationGroup(
                relation=_relation_from_raw_dedup_payload(
                    raw_relation,
                    relations,
                    group_member_ids,
                    aggregate_source_chunks=is_main,
                ),
                member_relation_ids=member_relation_ids if is_main else (),
            )
        )
    return groups[0] if len(groups) == 1 else groups


def _relation_from_raw_dedup_payload(
    raw_relation: dict[str, Any],
    relations: tuple[ExtractedRelation, ...],
    member_relation_ids: tuple[str, ...],
    *,
    aggregate_source_chunks: bool = True,
) -> ExtractedRelation:

    relation_type = _optional_string_any(raw_relation, ("type", "Type")) or relations[0].type
    relation_name = _required_string_any(raw_relation, ("name", "Relation"))
    description = _required_string_any(raw_relation, ("description", "Description"))
    attributes = raw_relation.get("attributes", raw_relation.get("Attr", {}))
    if not isinstance(attributes, dict):
        raise ValueError("attributes must be an object")

    if aggregate_source_chunks:
        source_chunk_ids = _unique_ordered(
            [
                *[chunk_id for relation in relations for chunk_id in relation.source_chunk_ids],
                *_raw_chunk_ids(raw_relation),
            ]
        )
    else:
        source_chunk_ids = _raw_chunk_ids(raw_relation) or _unique_ordered(
            chunk_id for relation in relations for chunk_id in relation.source_chunk_ids
        )
    representative = relations[0]
    source_name = _optional_string_any(raw_relation, ("source", "Node1", "source_name"))
    target_name = _optional_string_any(raw_relation, ("target", "Node2", "target_name"))
    source_entity_name = source_name or representative.source_entity_name
    target_entity_name = target_name or representative.target_entity_name
    source_entity_id = representative.source_entity_id
    target_entity_id = representative.target_entity_id
    if source_entity_name != representative.source_entity_name:
        source_entity_id = f"entity::{source_entity_name}"
    if target_entity_name != representative.target_entity_name:
        target_entity_id = f"entity::{target_entity_name}"
    return ExtractedRelation(
        relation_id=make_deduplicated_relation_id(
            member_relation_ids=member_relation_ids,
            name=relation_name,
        ),
        chunk_id=_dedup_chunk_id(source_chunk_ids),
        document_id=representative.document_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        source_entity_name=source_entity_name,
        target_entity_name=target_entity_name,
        type=relation_type,
        name=relation_name,
        description=description,
        attributes=_normalize_attributes(attributes),
        source_chunk_ids=source_chunk_ids,
    )


def _select_main_relation_group(
    result: _RelationGroup | list[_RelationGroup],
    *,
    node1: str,
    node2: str,
) -> tuple[_RelationGroup, list[_RelationGroup]]:
    groups = result if isinstance(result, list) else [result]
    main: _RelationGroup | None = None
    splits: list[_RelationGroup] = []
    normalized_pair = (_normalize_name(node1), _normalize_name(node2))
    for group in groups:
        pair = (
            _normalize_name(group.relation.source_entity_name),
            _normalize_name(group.relation.target_entity_name),
        )
        if main is None and pair == normalized_pair:
            main = group
        else:
            splits.append(group)
    if main is None:
        main = groups[0]
        splits = groups[1:]
    return main, splits


async def _dedup_semantic_relations(
    groups: list[_RelationGroup],
    *,
    language_model: LanguageModelProtocol,
    embedding_model: EmbeddingModelProtocol,
    config: DeduplicateRelationsConfig,
) -> tuple[list[_RelationGroup], int, int, int, tuple[StepIssue, ...]]:
    texts = [_relation_embedding_text(group.relation) for group in groups]
    embedding_result = await embedding_model.embed(
        EmbeddingRequest(
            texts=texts,
            trace_context={"step": DeduplicateRelations.name, "purpose": "semantic_merge"},
        )
    )
    embedded_groups = list(zip(groups, embedding_result.vectors, strict=True))
    current_batches = [
        embedded_groups[index : index + config.semantic_batch_size]
        for index in range(0, len(embedded_groups), config.semantic_batch_size)
    ]
    total_merged = 0
    total_failed = 0
    rounds = 0
    issues: list[StepIssue] = []
    while current_batches:
        rounds += 1
        processed_batches = []
        for batch in current_batches:
            (
                merged_groups,
                merged_count,
                failed_count,
                batch_issues,
            ) = await _merge_semantic_relation_records(
                [item[0] for item in batch],
                [item[1] for item in batch],
                language_model=language_model,
                config=config,
            )
            total_merged += merged_count
            total_failed += failed_count
            issues.extend(batch_issues)
            processed_batches.append(merged_groups)

        if len(processed_batches) == 1:
            return processed_batches[0], total_merged, total_failed, rounds, tuple(issues)

        flattened = [group for batch in processed_batches for group in batch]
        texts = [_relation_embedding_text(group.relation) for group in flattened]
        embedding_result = await embedding_model.embed(
            EmbeddingRequest(
                texts=texts,
                trace_context={
                    "step": DeduplicateRelations.name,
                    "purpose": "semantic_inter_batch_merge",
                    "round": rounds,
                },
            )
        )
        embedded_groups = list(zip(flattened, embedding_result.vectors, strict=True))
        batch_size = config.semantic_batch_size * config.semantic_batch_count
        current_batches = [
            embedded_groups[index : index + batch_size]
            for index in range(0, len(embedded_groups), batch_size)
        ]

    return [], total_merged, total_failed, rounds, tuple(issues)


async def _merge_semantic_relation_records(
    groups: list[_RelationGroup],
    vectors: list[list[float]],
    *,
    language_model: LanguageModelProtocol,
    config: DeduplicateRelationsConfig,
) -> tuple[list[_RelationGroup], int, int, tuple[StepIssue, ...]]:
    clusters = _cluster_vectors(vectors, threshold=config.similarity_threshold)
    output: list[_RelationGroup] = []
    merged_count = 0
    failed_count = 0
    issues: list[StepIssue] = []
    for cluster in clusters:
        cluster_groups = [groups[index] for index in cluster]
        if len(cluster_groups) == 1:
            output.append(cluster_groups[0])
            continue
        merged, reason = await _merge_relation_cluster_with_mapping(
            tuple(cluster_groups),
            language_model=language_model,
            config=config,
        )
        if merged is None:
            failed_count += 1
            issues.append(
                _dedup_issue(
                    _relation_cluster_key(cluster_groups),
                    reason or "relation semantic merge failed",
                )
            )
            output.extend(cluster_groups)
            continue
        merged_count += len(cluster_groups) - len(merged)
        output.extend(merged)
    return output, merged_count, failed_count, tuple(issues)


async def _merge_relation_cluster_with_mapping(
    cluster: tuple[_RelationGroup, ...],
    *,
    language_model: LanguageModelProtocol,
    config: DeduplicateRelationsConfig,
) -> tuple[list[_RelationGroup] | None, str | None]:
    last_error = ""
    for attempt in range(config.max_attempts):
        prompt = _build_relation_cluster_prompt(cluster, error=last_error if attempt > 0 else None)
        try:
            result = await language_model.invoke(
                ModelRequest(
                    prompt=prompt,
                    system_prompt=RELATION_DEDUPLICATION_SYSTEM_PROMPT,
                    options=ModelOptions(
                        temperature=config.temperature,
                        response_format={"type": "json_object"},
                    ),
                    trace_context={
                        "step": DeduplicateRelations.name,
                        "attempt": attempt + 1,
                        "phase": "semantic_mapping_merge",
                    },
                )
            )
            payload = result.parsed if result.parsed is not None else result.text
            return _relation_cluster_from_mapping_payload(payload, cluster), None
        except Exception as exc:
            last_error = str(exc) or exc.__class__.__name__
    return None, last_error or "relation semantic merge failed"


def _relation_cluster_from_mapping_payload(
    payload: Any,
    cluster: tuple[_RelationGroup, ...],
) -> list[_RelationGroup]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("relation cluster response must be a JSON object")
    mapping_table = payload.get("mapping_table", {}) or {}
    relation_list = payload.get("relation_list", [])
    if not isinstance(mapping_table, dict) or not mapping_table:
        return list(cluster)
    if not isinstance(relation_list, list):
        raise ValueError("relation_list must be a list")

    pair_to_groups: dict[tuple[str, str], list[_RelationGroup]] = {}
    for group in cluster:
        pair = (
            _normalize_name(group.relation.source_entity_name),
            _normalize_name(group.relation.target_entity_name),
        )
        pair_to_groups.setdefault(pair, []).append(group)

    llm_relation_by_pair = {
        pair: raw
        for raw in relation_list
        if isinstance(raw, dict)
        for pair in [_raw_relation_pair(raw)]
        if pair is not None
    }
    used_pairs: set[tuple[str, str]] = set()
    output: list[_RelationGroup] = []

    for canonical_key, original_values in mapping_table.items():
        canonical_pair = _parse_node_pair(canonical_key)
        if canonical_pair is None or not isinstance(original_values, list):
            continue
        related_groups: list[_RelationGroup] = []
        original_pairs: list[tuple[str, str]] = []
        for value in original_values:
            original_pair = _parse_node_pair(value)
            if original_pair is None:
                continue
            original_pairs.append(original_pair)
            related_groups.extend(pair_to_groups.get(original_pair, ()))
        if not related_groups:
            continue
        used_pairs.update(original_pairs)
        raw_relation = llm_relation_by_pair.get(canonical_pair)
        if raw_relation is None:
            raw_relation = {
                "Relation": related_groups[0].relation.name,
                "Type": related_groups[0].relation.type,
                "Description": related_groups[0].relation.description,
                "Attr": dict(related_groups[0].relation.attributes),
            }
        member_ids = tuple(
            member_id for group in related_groups for member_id in group.member_relation_ids
        )
        relation = _relation_from_raw_dedup_payload(
            raw_relation,
            tuple(group.relation for group in related_groups),
            member_ids,
        )
        output.append(_RelationGroup(relation=relation, member_relation_ids=member_ids))

    for pair, groups in pair_to_groups.items():
        if pair in used_pairs:
            continue
        output.extend(groups)

    return output or list(cluster)


def _build_relation_dedup_prompt(
    relations: tuple[ExtractedRelation, ...],
    *,
    error: str | None,
) -> str:
    template = RELATION_DEDUPLICATION_RETRY_PROMPT if error else RELATION_DEDUPLICATION_PROMPT
    relations_json = json.dumps(
        [
            {
                "relation_id": relation.relation_id,
                "source_entity_name": relation.source_entity_name,
                "target_entity_name": relation.target_entity_name,
                "type": relation.type,
                "name": relation.name,
                "description": relation.description,
                "attributes": dict(relation.attributes),
                "source_chunk_ids": relation.source_chunk_ids,
            }
            for relation in relations
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return template.format(error=error or "", relations_json=relations_json)


def _build_relation_cluster_prompt(
    cluster: tuple[_RelationGroup, ...],
    *,
    error: str | None,
) -> str:
    relations_json = json.dumps(
        [
            {
                "Node1": group.relation.source_entity_name,
                "Node2": group.relation.target_entity_name,
                "Relation": group.relation.name,
                "Type": group.relation.type,
                "Description": group.relation.description,
                "Attr": dict(group.relation.attributes),
            }
            for group in cluster
        ],
        ensure_ascii=False,
        indent=2,
    )
    retry = f"\nPrevious response was invalid: {error}\n" if error else ""
    return f"""Merge semantically duplicate knowledge graph relations.{retry}

Return only valid JSON with this shape:
{{
  "relation_list": [
    {{
      "Node1": "canonical source",
      "Node2": "canonical target",
      "Relation": "canonical relation name",
      "Type": "relation type",
      "Description": "merged factual description",
      "Attr": {{}},
      "merge_tag": true
    }}
  ],
  "mapping_table": {{
    "canonical source||canonical target": [
      "original source A||original target A",
      "original source B||original target B"
    ]
  }}
}}

Rules:
- Only include a mapping_table entry when relations should truly merge.
- If nothing should merge, return {{"relation_list": [], "mapping_table": {{}}}}.
- Original pairs listed in mapping_table must come from the input.
- Preserve only facts supported by the input relations.
- Do not invent new endpoints.

Relations:
{relations_json}
"""


def _apply_entity_mapping(
    relation: ExtractedRelation,
    entity_id_mapping: dict[str, str],
) -> ExtractedRelation:
    source_entity_id = entity_id_mapping.get(relation.source_entity_id, relation.source_entity_id)
    target_entity_id = entity_id_mapping.get(relation.target_entity_id, relation.target_entity_id)
    if (
        source_entity_id == relation.source_entity_id
        and target_entity_id == relation.target_entity_id
    ):
        return relation
    return ExtractedRelation(
        relation_id=relation.relation_id,
        chunk_id=relation.chunk_id,
        document_id=relation.document_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        source_entity_name=relation.source_entity_name,
        target_entity_name=relation.target_entity_name,
        type=relation.type,
        name=relation.name,
        description=relation.description,
        attributes=relation.attributes,
        source_chunk_ids=relation.source_chunk_ids,
    )


def _normalize_relation_endpoints(relation: ExtractedRelation) -> ExtractedRelation:
    source_name = _normalize_name(relation.source_entity_name)
    target_name = _normalize_name(relation.target_entity_name)
    source_id = relation.source_entity_id
    target_id = relation.target_entity_id
    if source_name != relation.source_entity_name:
        source_id = f"entity::{source_name}"
    if target_name != relation.target_entity_name:
        target_id = f"entity::{target_name}"
    if (
        source_name == relation.source_entity_name
        and target_name == relation.target_entity_name
    ):
        return relation
    return ExtractedRelation(
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


def _entity_id_mapping(context: StepContextProtocol, artifact_name: str | None) -> dict[str, str]:
    if artifact_name is None:
        return {}
    value = context.get_artifact(artifact_name)
    if not isinstance(value, dict):
        raise TypeError(f"{artifact_name} must be a dict[str, str]")
    return {str(key): str(mapped) for key, mapped in value.items()}


def _exact_relation_key(relation: ExtractedRelation) -> str:
    return (
        relation.source_entity_id,
        relation.target_entity_id,
        _normalize_name(relation.name),
        _normalize_name(relation.type),
    )


def _relation_group_key(relation: ExtractedRelation) -> str:
    return "|".join(
        [
            relation.source_entity_name,
            relation.target_entity_name,
            relation.name,
            relation.type,
        ]
    )


def _relation_cluster_key(groups: list[_RelationGroup]) -> str:
    return ",".join(_relation_group_key(group.relation) for group in groups)


def _dedup_issue(group_key: str, message: str, *, code: str = "deduplication_failed") -> StepIssue:
    return StepIssue(
        step=DeduplicateRelations.name,
        subject=IssueSubject(type="dedup_group", id=group_key),
        code=code,
        message=message,
        resolution=IssueResolution(
            action="kept_original_records",
            outcome="The group was not merged, and original records were kept.",
        ),
    )


def _split_relation_uniques_duplicates(
    groups: list[_RelationGroup],
) -> tuple[
    dict[tuple[str, str, str, str], _RelationGroup],
    dict[tuple[str, str, str, str], list[_RelationGroup]],
]:
    uniques: dict[tuple[str, str, str, str], _RelationGroup] = {}
    duplicates: dict[tuple[str, str, str, str], list[_RelationGroup]] = {}
    for group in groups:
        key = _exact_relation_key(group.relation)
        if key in uniques:
            duplicates.setdefault(key, [uniques[key]]).append(group)
        else:
            uniques[key] = group
    return uniques, duplicates


def _relation_embedding_text(relation: ExtractedRelation) -> str:
    attributes = " ".join(f"{key}:{value}" for key, value in sorted(relation.attributes.items()))
    return (
        f"{relation.source_entity_name} -> {relation.target_entity_name}\n"
        f"{relation.type}\n{relation.name}\n{relation.description}\n{attributes}"
    ).strip()


def _raw_relation_pair(raw_relation: dict[str, Any]) -> tuple[str, str] | None:
    node1 = (
        raw_relation.get("Node1")
        or raw_relation.get("source")
        or raw_relation.get("source_name")
    )
    node2 = (
        raw_relation.get("Node2")
        or raw_relation.get("target")
        or raw_relation.get("target_name")
    )
    if node1 is None or node2 is None:
        return None
    pair = (_normalize_name(str(node1)), _normalize_name(str(node2)))
    return pair if pair[0] and pair[1] else None


def _parse_node_pair(value: Any) -> tuple[str, str] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        pair = (_normalize_name(str(value[0])), _normalize_name(str(value[1])))
        return pair if pair[0] and pair[1] else None
    if not isinstance(value, str):
        return None
    if "||" in value:
        left, right = value.split("||", maxsplit=1)
        pair = (_normalize_name(left), _normalize_name(right))
        return pair if pair[0] and pair[1] else None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list) and len(parsed) == 2:
        pair = (_normalize_name(str(parsed[0])), _normalize_name(str(parsed[1])))
        return pair if pair[0] and pair[1] else None
    stripped = value.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        parts = [part.strip().strip("'\"") for part in stripped[1:-1].split(",", maxsplit=1)]
        if len(parts) == 2:
            pair = (_normalize_name(parts[0]), _normalize_name(parts[1]))
            return pair if pair[0] and pair[1] else None
    return None


def _raw_chunk_ids(data: dict[str, Any], *, key: str | None = None) -> tuple[str, ...]:
    value = data.get(key) if key is not None else data.get("chunk_id", data.get("ChunkId"))
    if value is None:
        return ()
    if isinstance(value, list):
        return _unique_ordered(str(item) for item in value if str(item).strip())
    text = str(value).strip()
    return (text,) if text else ()


def _cluster_vectors(vectors: list[list[float]], *, threshold: float) -> list[tuple[int, ...]]:
    if not vectors:
        return []
    normalized = [_normalize_vector(vector) for vector in vectors]
    clusters: list[tuple[int, ...]] = [(index,) for index in range(len(vectors))]
    distance_threshold = max(0.0, 1.0 - threshold)
    while True:
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                distance = _average_linkage_distance(
                    clusters[left_index],
                    clusters[right_index],
                    normalized,
                )
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (left_index, right_index)
        if best_pair is None or best_distance > distance_threshold:
            break
        left_index, right_index = best_pair
        merged = tuple(sorted((*clusters[left_index], *clusters[right_index])))
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left_index, right_index}
        ]
        clusters.append(merged)
    return clusters


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _average_linkage_distance(
    left: tuple[int, ...],
    right: tuple[int, ...],
    vectors: list[list[float]],
) -> float:
    distances = [
        _euclidean_distance(vectors[left_index], vectors[right_index])
        for left_index in left
        for right_index in right
    ]
    return sum(distances) / len(distances)


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have the same dimension")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _dedup_chunk_id(source_chunk_ids: tuple[str, ...]) -> str:
    payload = "\n".join(source_chunk_ids)
    import hashlib

    return f"dedup_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _normalize_name(value: str) -> str:
    return str(value).strip()


def _normalize_attributes(attributes: dict[Any, Any]) -> dict[str, str]:
    return {
        str(key).strip(): str(value).strip()
        for key, value in attributes.items()
        if str(key).strip() and str(value).strip()
    }


def _unique_ordered(values) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _required_string_any(data: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"{'/'.join(keys)} must contain a non-empty string")


def _optional_string_any(data: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_language_model(component: object) -> LanguageModelProtocol:
    if not isinstance(component, LanguageModelProtocol):
        raise TypeError("models.language must satisfy LanguageModelProtocol")
    return component


def _require_embedding_model(component: object) -> EmbeddingModelProtocol:
    if not isinstance(component, EmbeddingModelProtocol):
        raise TypeError("models.embedding must satisfy EmbeddingModelProtocol")
    return component
