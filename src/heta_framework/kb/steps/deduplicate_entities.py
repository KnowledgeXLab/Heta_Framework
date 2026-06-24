"""Deduplicate extracted entity artifacts within the current build batch."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from heta_framework.common.models import EmbeddingRequest, ModelOptions, ModelRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol, LanguageModelProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb.cleanup import StepCleanupPlan, object_key_targets
from heta_framework.kb.graphing import ExtractedEntity, make_deduplicated_entity_id
from heta_framework.kb.graphing.prompts import (
    ENTITY_DEDUPLICATION_PROMPT,
    ENTITY_DEDUPLICATION_RETRY_PROMPT,
    ENTITY_DEDUPLICATION_SYSTEM_PROMPT,
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


ENTITY_DEDUPLICATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entity": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "subtype": {"type": ["string", "null"]},
                "description": {"type": "string"},
                "attributes": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["name", "type", "description", "attributes"],
        }
    },
    "required": ["entity"],
}


@dataclass(frozen=True)
class DeduplicateEntitiesConfig:
    """Configuration for DeduplicateEntities."""

    deduplicated_entities_prefix: str = "deduplicated_entities"
    entity_keys_artifact: str = "entity_keys"
    deduplicated_entity_keys_artifact: str = "deduplicated_entity_keys"
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
        validate_object_prefix(self.deduplicated_entities_prefix)
        if self.entity_keys_artifact.strip() == "":
            raise ValueError("entity_keys_artifact must not be empty")
        if self.deduplicated_entity_keys_artifact.strip() == "":
            raise ValueError("deduplicated_entity_keys_artifact must not be empty")
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
class DeduplicateEntitiesResult:
    """Artifacts produced by DeduplicateEntities."""

    entity_keys: tuple[str, ...]
    input_entity_count: int
    output_entity_count: int
    exact_merge_count: int
    semantic_merge_count: int
    failed_group_count: int
    exact_round_count: int
    semantic_round_count: int
    issues: tuple[StepIssue, ...]


@dataclass(frozen=True)
class _EntityGroup:
    entity: ExtractedEntity
    member_entity_ids: tuple[str, ...]


class DeduplicateEntities:
    """Merge duplicate ExtractedEntity artifacts while preserving the entity schema."""

    name = "deduplicate_entities"

    def __init__(self, config: DeduplicateEntitiesConfig | None = None) -> None:
        self.config = config or DeduplicateEntitiesConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components and artifacts required by this step."""
        components = {
            store_ref("objects", self.config.object_store),
            model_ref("language", self.config.language_model),
        }
        if self.config.semantic_merge:
            components.add(model_ref("embedding", self.config.embedding_model))
        return StepRequirements(
            components=frozenset(components),
            artifacts=frozenset({self.config.entity_keys_artifact}),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts produced by this step."""
        return StepCapabilities(
            artifacts=frozenset(
                {
                    "deduplicate_entities_result",
                    self.config.deduplicated_entity_keys_artifact,
                    "entity_id_mapping",
                }
            )
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        """Return deduplicated entity objects produced by this step."""
        return StepCleanupPlan(
            object_key_targets(
                artifacts,
                self.config.deduplicated_entity_keys_artifact,
                component=store_ref("objects", self.config.object_store).key,
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Run entity deduplication and store ExtractedEntity JSON objects."""
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

        entity_keys = tuple(context.get_artifact(self.config.entity_keys_artifact))
        entities = [ExtractedEntity.from_json(await object_store.get(key)) for key in entity_keys]

        (
            groups,
            exact_merge_count,
            failed_exact,
            exact_round_count,
            exact_issues,
        ) = await _dedup_exact_entities(
            entities,
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
            ) = await _dedup_semantic_entities(
                groups,
                language_model=language_model,
                embedding_model=embedding_model,
                config=self.config,
            )

        output_keys: list[str] = []
        entity_id_mapping: dict[str, str] = {}
        for group in groups:
            key = join_object_key(
                self.config.deduplicated_entities_prefix,
                f"{group.entity.entity_id}.json",
            )
            await object_store.put(key, group.entity.to_json_bytes())
            output_keys.append(key)
            for member_id in group.member_entity_ids:
                entity_id_mapping[member_id] = group.entity.entity_id

        result = DeduplicateEntitiesResult(
            entity_keys=tuple(output_keys),
            input_entity_count=len(entities),
            output_entity_count=len(output_keys),
            exact_merge_count=exact_merge_count,
            semantic_merge_count=semantic_merge_count,
            failed_group_count=failed_exact + failed_semantic,
            exact_round_count=exact_round_count,
            semantic_round_count=semantic_round_count,
            issues=tuple([*exact_issues, *semantic_issues]),
        )
        context.set_artifact("deduplicate_entities_result", result)
        context.set_artifact(self.config.deduplicated_entity_keys_artifact, result.entity_keys)
        context.set_artifact("entity_id_mapping", entity_id_mapping)


async def _dedup_exact_entities(
    entities: list[ExtractedEntity],
    *,
    language_model: LanguageModelProtocol,
    config: DeduplicateEntitiesConfig,
) -> tuple[list[_EntityGroup], int, int, int, tuple[StepIssue, ...]]:
    current = [
        _EntityGroup(entity=entity, member_entity_ids=(entity.entity_id,))
        for entity in entities
    ]
    total_merged = 0
    total_failed = 0
    rounds = 0
    issues: list[StepIssue] = []
    suppressed_keys: set[str] = set()

    while rounds < config.max_rounds:
        uniques, duplicates = _split_entity_uniques_duplicates(current)
        if not duplicates:
            break
        if all(key in suppressed_keys for key in duplicates):
            break
        rounds += 1
        next_groups: list[_EntityGroup] = []
        for key, first_group in uniques.items():
            duplicate_group = duplicates.get(key)
            if duplicate_group is None:
                next_groups.append(first_group)
                continue
            if key in suppressed_keys:
                next_groups.extend(duplicate_group)
                continue
            merged_groups, reason = await _merge_exact_entity_group(
                tuple(duplicate_group),
                node_name=key,
                language_model=language_model,
                config=config,
            )
            if merged_groups is None:
                total_failed += 1
                suppressed_keys.add(key)
                issues.append(_dedup_issue(key, reason or "entity deduplication failed"))
                next_groups.extend(duplicate_group)
                continue
            total_merged += max(0, len(duplicate_group) - len(merged_groups))
            next_groups.extend(merged_groups)
        current = next_groups

    if rounds >= config.max_rounds:
        _, remaining_duplicates = _split_entity_uniques_duplicates(current)
        for key in remaining_duplicates:
            issues.append(
                _dedup_issue(
                    key,
                    "max_rounds reached with duplicate entities remaining",
                    code="max_rounds_reached",
                )
            )

    return current, total_merged, total_failed, rounds, tuple(issues)


async def _merge_group_objects(
    grouped: list[tuple[_EntityGroup, ...]],
    *,
    language_model: LanguageModelProtocol,
    config: DeduplicateEntitiesConfig,
) -> tuple[list[_EntityGroup], int, int]:
    merged: list[_EntityGroup] = []
    merge_count = 0
    failed_count = 0
    for group in grouped:
        if len(group) == 1:
            merged.append(group[0])
            continue
        merged_group, _ = await _merge_entity_group(
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


async def _merge_exact_entity_group(
    group: tuple[_EntityGroup, ...],
    *,
    node_name: str,
    language_model: LanguageModelProtocol,
    config: DeduplicateEntitiesConfig,
) -> tuple[list[_EntityGroup] | None, str | None]:
    split_groups: list[_EntityGroup] = []
    last_reason: str | None = None

    async def merge_batch(batch: tuple[_EntityGroup, ...]) -> _EntityGroup | None:
        nonlocal last_reason
        merged, reason = await _merge_entity_group(
            batch,
            language_model=language_model,
            config=config,
        )
        last_reason = reason
        if merged is None:
            return None
        main_group, extra_groups = _select_main_entity_group(merged, node_name=node_name)
        split_groups.extend(extra_groups)
        return main_group

    if len(group) <= config.llm_batch_size:
        main = await merge_batch(group)
        if main is None:
            return None, last_reason or "entity batch merge failed"
        return [main, *split_groups], None

    accumulated: _EntityGroup | None = None
    for start in range(0, len(group), config.llm_batch_size):
        batch = group[start : start + config.llm_batch_size]
        if accumulated is not None:
            batch = (accumulated, *batch)
        accumulated = await merge_batch(tuple(batch))
        if accumulated is None:
            return None, last_reason or "entity batch merge failed"
    if accumulated is None:
        return None, last_reason or "entity batch merge failed"
    return [accumulated, *split_groups], None


async def _merge_entity_group(
    group: tuple[_EntityGroup, ...],
    *,
    language_model: LanguageModelProtocol,
    config: DeduplicateEntitiesConfig,
) -> tuple[_EntityGroup | list[_EntityGroup] | None, str | None]:
    last_error = ""
    entities = tuple(item.entity for item in group)
    member_ids = tuple(member_id for item in group for member_id in item.member_entity_ids)
    for attempt in range(config.max_attempts):
        prompt = _build_entity_dedup_prompt(entities, error=last_error if attempt > 0 else None)
        try:
            result = await language_model.invoke(
                ModelRequest(
                    prompt=prompt,
                    system_prompt=ENTITY_DEDUPLICATION_SYSTEM_PROMPT,
                    options=ModelOptions(
                        temperature=config.temperature,
                        response_format={"type": "json_object"},
                    ),
                    response_schema=ENTITY_DEDUPLICATION_RESPONSE_SCHEMA,
                    trace_context={
                        "step": DeduplicateEntities.name,
                        "attempt": attempt + 1,
                        "member_entity_ids": member_ids,
                    },
                )
            )
            payload = result.parsed if result.parsed is not None else result.text
            return _entity_groups_from_dedup_payload(payload, entities, member_ids), None
        except Exception as exc:
            last_error = str(exc) or exc.__class__.__name__
    return None, last_error or "entity deduplication failed"


def _entity_groups_from_dedup_payload(
    payload: Any,
    entities: tuple[ExtractedEntity, ...],
    member_entity_ids: tuple[str, ...],
) -> _EntityGroup | list[_EntityGroup]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    raw_entities: list[dict[str, Any]]
    if isinstance(payload, list):
        raw_entities = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        if isinstance(payload.get("entity"), dict):
            raw_entities = [payload["entity"]]
        elif isinstance(payload.get("entities"), list):
            raw_entities = [item for item in payload["entities"] if isinstance(item, dict)]
        else:
            raw_entities = [payload]
    else:
        raise ValueError("entity deduplication response must be a JSON object or array")
    if not raw_entities:
        raise ValueError("entity deduplication response did not contain entities")

    normalized_node_name = _normalize_name(entities[0].name)
    main_index = next(
        (
            index
            for index, raw_entity in enumerate(raw_entities)
            if _normalize_name(_raw_entity_name(raw_entity)) == normalized_node_name
        ),
        0,
    )
    groups = []
    for index, raw_entity in enumerate(raw_entities):
        group_member_ids = member_entity_ids
        is_main = index == main_index
        if not is_main:
            description = str(raw_entity.get("description") or raw_entity.get("Description") or "")
            group_member_ids = (f"split:{index}:{description}",)
        groups.append(
            _EntityGroup(
                entity=_entity_from_raw_dedup_payload(
                    raw_entity,
                    entities,
                    group_member_ids,
                    aggregate_source_chunks=is_main,
                ),
                member_entity_ids=member_entity_ids if is_main else (),
            )
        )
    return groups[0] if len(groups) == 1 else groups


def _entity_from_raw_dedup_payload(
    raw_entity: dict[str, Any],
    entities: tuple[ExtractedEntity, ...],
    member_entity_ids: tuple[str, ...],
    *,
    aggregate_source_chunks: bool = True,
) -> ExtractedEntity:

    name = _required_string_any(raw_entity, ("name", "NodeName", "Nodename"))
    entity_type = _optional_string_any(raw_entity, ("type", "Type")) or entities[0].type
    description = _required_string_any(raw_entity, ("description", "Description"))
    subtype = raw_entity.get("subtype", raw_entity.get("Subtype"))
    if subtype is not None:
        if not isinstance(subtype, str):
            raise ValueError("subtype must be a string or null")
        subtype = subtype.strip() or None
    attributes = raw_entity.get("attributes", raw_entity.get("Attr", {}))
    if not isinstance(attributes, dict):
        raise ValueError("attributes must be an object")

    if aggregate_source_chunks:
        source_chunk_ids = _unique_ordered(
            [
                *[chunk_id for entity in entities for chunk_id in entity.source_chunk_ids],
                *_raw_chunk_ids(raw_entity, key="ChunkId"),
            ]
        )
    else:
        source_chunk_ids = _raw_chunk_ids(raw_entity) or _unique_ordered(
            chunk_id for entity in entities for chunk_id in entity.source_chunk_ids
        )
    representative = entities[0]
    return ExtractedEntity(
        entity_id=make_deduplicated_entity_id(
            member_entity_ids=member_entity_ids,
            name=name,
        ),
        chunk_id=_dedup_chunk_id(source_chunk_ids),
        document_id=representative.document_id,
        name=name,
        type=entity_type,
        subtype=subtype,
        description=description,
        attributes=_normalize_attributes(attributes),
        source_chunk_ids=source_chunk_ids,
    )


def _select_main_entity_group(
    result: _EntityGroup | list[_EntityGroup],
    *,
    node_name: str,
) -> tuple[_EntityGroup, list[_EntityGroup]]:
    groups = result if isinstance(result, list) else [result]
    main: _EntityGroup | None = None
    splits: list[_EntityGroup] = []
    normalized_node_name = _normalize_name(node_name)
    for group in groups:
        if main is None and _normalize_name(group.entity.name) == normalized_node_name:
            main = group
        else:
            splits.append(group)
    if main is None:
        main = groups[0]
        splits = groups[1:]
    return main, splits


async def _dedup_semantic_entities(
    groups: list[_EntityGroup],
    *,
    language_model: LanguageModelProtocol,
    embedding_model: EmbeddingModelProtocol,
    config: DeduplicateEntitiesConfig,
) -> tuple[list[_EntityGroup], int, int, int, tuple[StepIssue, ...]]:
    texts = [_entity_embedding_text(group.entity) for group in groups]
    embedding_result = await embedding_model.embed(
        EmbeddingRequest(
            texts=texts,
            trace_context={"step": DeduplicateEntities.name, "purpose": "semantic_merge"},
        )
    )
    embedded_groups = [
        (group, vector)
        for group, vector in zip(groups, embedding_result.vectors, strict=True)
    ]
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
            ) = await _merge_semantic_entity_records(
                [item[0] for item in batch],
                [item[1] for item in batch],
                language_model=language_model,
                embedding_model=embedding_model,
                config=config,
            )
            total_merged += merged_count
            total_failed += failed_count
            issues.extend(batch_issues)
            processed_batches.append(merged_groups)

        if len(processed_batches) == 1:
            return processed_batches[0], total_merged, total_failed, rounds, tuple(issues)

        next_batches: list[list[tuple[_EntityGroup, list[float]]]] = []
        flattened = [group for batch in processed_batches for group in batch]
        texts = [_entity_embedding_text(group.entity) for group in flattened]
        embedding_result = await embedding_model.embed(
            EmbeddingRequest(
                texts=texts,
                trace_context={
                    "step": DeduplicateEntities.name,
                    "purpose": "semantic_inter_batch_merge",
                    "round": rounds,
                },
            )
        )
        embedded_groups = list(zip(flattened, embedding_result.vectors, strict=True))
        batch_size = config.semantic_batch_size * config.semantic_batch_count
        for index in range(0, len(embedded_groups), batch_size):
            next_batches.append(embedded_groups[index : index + batch_size])
        current_batches = next_batches

    return [], total_merged, total_failed, rounds, tuple(issues)


async def _merge_semantic_entity_records(
    groups: list[_EntityGroup],
    vectors: list[list[float]],
    *,
    language_model: LanguageModelProtocol,
    embedding_model: EmbeddingModelProtocol,
    config: DeduplicateEntitiesConfig,
) -> tuple[list[_EntityGroup], int, int, tuple[StepIssue, ...]]:
    clusters = _cluster_vectors(vectors, threshold=config.similarity_threshold)
    output: list[_EntityGroup] = []
    merged_count = 0
    failed_count = 0
    issues: list[StepIssue] = []
    for cluster in clusters:
        cluster_groups = [groups[index] for index in cluster]
        if len(cluster_groups) == 1:
            output.append(cluster_groups[0])
            continue
        merged, reason = await _merge_entity_cluster_with_mapping(
            tuple(cluster_groups),
            language_model=language_model,
            embedding_model=embedding_model,
            config=config,
        )
        if merged is None:
            failed_count += 1
            issues.append(
                _dedup_issue(
                    _entity_cluster_key(cluster_groups),
                    reason or "entity semantic merge failed",
                )
            )
            output.extend(cluster_groups)
            continue
        merged_count += len(cluster_groups) - len(merged)
        output.extend(merged)
    return output, merged_count, failed_count, tuple(issues)


async def _merge_entity_cluster_with_mapping(
    cluster: tuple[_EntityGroup, ...],
    *,
    language_model: LanguageModelProtocol,
    embedding_model: EmbeddingModelProtocol,
    config: DeduplicateEntitiesConfig,
) -> tuple[list[_EntityGroup] | None, str | None]:
    del embedding_model
    last_error = ""
    for attempt in range(config.max_attempts):
        prompt = _build_entity_cluster_prompt(cluster, error=last_error if attempt > 0 else None)
        try:
            result = await language_model.invoke(
                ModelRequest(
                    prompt=prompt,
                    system_prompt=ENTITY_DEDUPLICATION_SYSTEM_PROMPT,
                    options=ModelOptions(
                        temperature=config.temperature,
                        response_format={"type": "json_object"},
                    ),
                    trace_context={
                        "step": DeduplicateEntities.name,
                        "attempt": attempt + 1,
                        "phase": "semantic_mapping_merge",
                    },
                )
            )
            payload = result.parsed if result.parsed is not None else result.text
            return _entity_cluster_from_mapping_payload(payload, cluster), None
        except Exception as exc:
            last_error = str(exc) or exc.__class__.__name__
    return None, last_error or "entity semantic merge failed"


def _entity_cluster_from_mapping_payload(
    payload: Any,
    cluster: tuple[_EntityGroup, ...],
) -> list[_EntityGroup]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("entity cluster response must be a JSON object")
    mapping_table = payload.get("mapping_table", {}) or {}
    entity_list = payload.get("entity_list", [])
    if not isinstance(mapping_table, dict) or not mapping_table:
        return list(cluster)
    if not isinstance(entity_list, list):
        raise ValueError("entity_list must be a list")

    name_to_groups: dict[str, list[_EntityGroup]] = {}
    for group in cluster:
        name_to_groups.setdefault(_normalize_name(group.entity.name), []).append(group)

    llm_entity_by_name = {
        _normalize_name(_raw_entity_name(raw)): raw
        for raw in entity_list
        if isinstance(raw, dict) and _raw_entity_name(raw)
    }
    used_names: set[str] = set()
    output: list[_EntityGroup] = []

    for canonical_name, original_names in mapping_table.items():
        if not isinstance(original_names, list):
            continue
        related_groups: list[_EntityGroup] = []
        for original_name in original_names:
            normalized = _normalize_name(str(original_name))
            related_groups.extend(name_to_groups.get(normalized, ()))
            used_names.add(normalized)
        if not related_groups:
            continue

        raw_entity = llm_entity_by_name.get(_normalize_name(str(canonical_name)))
        if raw_entity is None:
            raw_entity = {
                "name": str(canonical_name),
                "type": related_groups[0].entity.type,
                "subtype": related_groups[0].entity.subtype,
                "description": related_groups[0].entity.description,
                "attributes": dict(related_groups[0].entity.attributes),
            }
        member_ids = tuple(
            member_id for group in related_groups for member_id in group.member_entity_ids
        )
        entity = _entity_from_raw_dedup_payload(
            raw_entity,
            tuple(group.entity for group in related_groups),
            member_ids,
        )
        output.append(_EntityGroup(entity=entity, member_entity_ids=member_ids))

    for normalized_name, groups in name_to_groups.items():
        if normalized_name in used_names:
            continue
        output.extend(groups)

    return output or list(cluster)


def _build_entity_dedup_prompt(entities: tuple[ExtractedEntity, ...], *, error: str | None) -> str:
    template = ENTITY_DEDUPLICATION_RETRY_PROMPT if error else ENTITY_DEDUPLICATION_PROMPT
    entities_json = json.dumps(
        [
            {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "type": entity.type,
                "subtype": entity.subtype,
                "description": entity.description,
                "attributes": dict(entity.attributes),
                "source_chunk_ids": entity.source_chunk_ids,
            }
            for entity in entities
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return template.format(error=error or "", entities_json=entities_json)


def _build_entity_cluster_prompt(
    cluster: tuple[_EntityGroup, ...],
    *,
    error: str | None,
) -> str:
    entities_json = json.dumps(
        [
            {
                "NodeName": group.entity.name,
                "Description": group.entity.description,
                "Type": group.entity.type,
                "Subtype": group.entity.subtype,
                "Attr": dict(group.entity.attributes),
            }
            for group in cluster
        ],
        ensure_ascii=False,
        indent=2,
    )
    retry = f"\nPrevious response was invalid: {error}\n" if error else ""
    return f"""Merge semantically duplicate knowledge graph entities.{retry}

Return only valid JSON with this shape:
{{
  "entity_list": [
    {{
      "NodeName": "canonical entity name",
      "Description": "merged factual description",
      "Type": "entity type",
      "Subtype": null,
      "Attr": {{}},
      "merge_tag": true
    }}
  ],
  "mapping_table": {{
    "canonical entity name": ["original entity name A", "original entity name B"]
  }}
}}

Rules:
- Only include a mapping_table entry when entities should truly merge.
- If nothing should merge, return {{"entity_list": [], "mapping_table": {{}}}}.
- Original names listed in mapping_table must come from the input.
- Preserve only facts supported by the input entities.

Entities:
{entities_json}
"""


def _exact_entity_key(entity: ExtractedEntity) -> str:
    return _normalize_name(entity.name)


def _split_entity_uniques_duplicates(
    groups: list[_EntityGroup],
) -> tuple[dict[str, _EntityGroup], dict[str, list[_EntityGroup]]]:
    uniques: dict[str, _EntityGroup] = {}
    duplicates: dict[str, list[_EntityGroup]] = {}
    for group in groups:
        key = _exact_entity_key(group.entity)
        if key in uniques:
            duplicates.setdefault(key, [uniques[key]]).append(group)
        else:
            uniques[key] = group
    return uniques, duplicates


def _entity_embedding_text(entity: ExtractedEntity) -> str:
    subtype = f"/{entity.subtype}" if entity.subtype else ""
    attributes = " ".join(f"{key}:{value}" for key, value in sorted(entity.attributes.items()))
    return f"{entity.name}\n{entity.type}{subtype}\n{entity.description}\n{attributes}".strip()


def _entity_cluster_key(groups: list[_EntityGroup]) -> str:
    return ",".join(group.entity.name for group in groups)


def _dedup_issue(group_key: str, message: str, *, code: str = "deduplication_failed") -> StepIssue:
    return StepIssue(
        step=DeduplicateEntities.name,
        subject=IssueSubject(type="dedup_group", id=group_key),
        code=code,
        message=message,
        resolution=IssueResolution(
            action="kept_original_records",
            outcome="The group was not merged, and original records were kept.",
        ),
    )


def _raw_entity_name(raw_entity: dict[str, Any]) -> str:
    value = raw_entity.get("name") or raw_entity.get("NodeName") or raw_entity.get("Nodename")
    return str(value).strip() if value is not None else ""


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
