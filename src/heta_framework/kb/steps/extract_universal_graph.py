"""Universal entity and relation extraction step."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from heta_framework.common.models import ModelOptions, ModelRequest
from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb.chunking import ParsedChunk
from heta_framework.kb.cleanup import StepCleanupPlan, object_key_targets
from heta_framework.kb.graphing import (
    ExtractedEntity,
    ExtractedRelation,
    make_entity_id,
    make_relation_id,
)
from heta_framework.kb.graphing.prompts import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_EXTRACTION_RETRY_PROMPT,
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    RELATION_EXTRACTION_PROMPT,
    RELATION_EXTRACTION_RETRY_PROMPT,
    RELATION_EXTRACTION_SYSTEM_PROMPT,
)
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref
from heta_framework.kb.steps.universal_graph_common import (
    _require_language_model,
    _require_object_store,
)


UNIVERSAL_ENTITY_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
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
            },
        }
    },
    "required": ["entities"],
}

UNIVERSAL_RELATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "attributes": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["source", "target", "type", "name", "description", "attributes"],
            },
        }
    },
    "required": ["relations"],
}


@dataclass(frozen=True)
class ExtractUniversalGraphConfig:
    """Configuration for ExtractUniversalGraph."""

    entities_prefix: str = "entities"
    relations_prefix: str = "relations"
    max_attempts: int = 3
    temperature: float = 0.0
    object_store: str | None = None
    language_model: str | None = None
    chunk_keys_artifact: str = "chunk_keys"
    entity_keys_artifact: str = "entity_keys"
    relation_keys_artifact: str = "relation_keys"
    result_artifact: str = "extract_universal_graph_result"

    def __post_init__(self) -> None:
        validate_object_prefix(self.entities_prefix)
        validate_object_prefix(self.relations_prefix)
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        for name in (
            self.chunk_keys_artifact,
            self.entity_keys_artifact,
            self.relation_keys_artifact,
            self.result_artifact,
        ):
            if name.strip() == "":
                raise ValueError("artifact names must not be empty")


@dataclass(frozen=True)
class ExtractUniversalGraphResult:
    """Artifacts produced by ExtractUniversalGraph."""

    entity_keys: tuple[str, ...]
    relation_keys: tuple[str, ...]
    chunk_count: int
    entity_count: int
    relation_count: int
    failed_entity_chunk_ids: tuple[str, ...]
    failed_relation_chunk_ids: tuple[str, ...]
    skipped_relation_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class _ChunkUniversalGraphResult:
    entity_keys: tuple[str, ...]
    relation_keys: tuple[str, ...]
    failed_entity_chunk_id: str | None = None
    failed_relation_chunk_id: str | None = None
    skipped_relation_chunk_id: str | None = None


class ExtractUniversalGraph:
    """Extract entities and relations in one recipe-visible universal graph step."""

    name = "extract_universal_graph"

    def __init__(self, config: ExtractUniversalGraphConfig | None = None) -> None:
        self.config = config or ExtractUniversalGraphConfig()

    @property
    def requirements(self) -> StepRequirements:
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("objects", self.config.object_store),
                    model_ref("language", self.config.language_model),
                }
            ),
            artifacts=frozenset({self.config.chunk_keys_artifact}),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        return StepCapabilities(
            artifacts=frozenset(
                {
                    self.config.result_artifact,
                    self.config.entity_keys_artifact,
                    self.config.relation_keys_artifact,
                }
            )
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        component = store_ref("objects", self.config.object_store).key
        return StepCleanupPlan(
            object_key_targets(artifacts, self.config.entity_keys_artifact, component=component)
            + object_key_targets(
                artifacts,
                self.config.relation_keys_artifact,
                component=component,
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        language_model = _require_language_model(
            context.get_component(model_ref("language", self.config.language_model).key)
        )
        chunk_keys = tuple(context.get_artifact(self.config.chunk_keys_artifact))
        chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]

        chunk_results = await asyncio.gather(
            *(
                self._process_chunk(
                    chunk,
                    object_store=object_store,
                    language_model=language_model,
                )
                for chunk in chunks
            )
        )

        entity_keys: list[str] = []
        relation_keys: list[str] = []
        failed_entity_chunk_ids: list[str] = []
        failed_relation_chunk_ids: list[str] = []
        skipped_relation_chunk_ids: list[str] = []

        for chunk_result in chunk_results:
            entity_keys.extend(chunk_result.entity_keys)
            relation_keys.extend(chunk_result.relation_keys)
            if chunk_result.failed_entity_chunk_id is not None:
                failed_entity_chunk_ids.append(chunk_result.failed_entity_chunk_id)
            if chunk_result.failed_relation_chunk_id is not None:
                failed_relation_chunk_ids.append(chunk_result.failed_relation_chunk_id)
            if chunk_result.skipped_relation_chunk_id is not None:
                skipped_relation_chunk_ids.append(chunk_result.skipped_relation_chunk_id)

        context.set_artifact(
            self.config.result_artifact,
            ExtractUniversalGraphResult(
                entity_keys=tuple(entity_keys),
                relation_keys=tuple(relation_keys),
                chunk_count=len(chunks),
                entity_count=len(entity_keys),
                relation_count=len(relation_keys),
                failed_entity_chunk_ids=tuple(failed_entity_chunk_ids),
                failed_relation_chunk_ids=tuple(failed_relation_chunk_ids),
                skipped_relation_chunk_ids=tuple(skipped_relation_chunk_ids),
            ),
        )
        context.set_artifact(self.config.entity_keys_artifact, tuple(entity_keys))
        context.set_artifact(self.config.relation_keys_artifact, tuple(relation_keys))

    async def _process_chunk(
        self,
        chunk: ParsedChunk,
        *,
        object_store: ObjectStoreProtocol,
        language_model: LanguageModelProtocol,
    ) -> _ChunkUniversalGraphResult:
        entity_keys: list[str] = []
        relation_keys: list[str] = []
        chunk_entity_keys = await _existing_chunk_keys(
            object_store,
            prefix=self.config.entities_prefix,
            chunk_id=chunk.chunk_id,
        )
        if chunk_entity_keys:
            entities = [
                ExtractedEntity.from_json(await object_store.get(key)) for key in chunk_entity_keys
            ]
            entity_keys.extend(chunk_entity_keys)
        else:
            entities = await _extract_universal_entities(
                chunk,
                language_model=language_model,
                config=self.config,
            )
            if entities is None:
                return _ChunkUniversalGraphResult(
                    entity_keys=(),
                    relation_keys=(),
                    failed_entity_chunk_id=chunk.chunk_id,
                )
            for entity in entities:
                key = join_object_key(
                    self.config.entities_prefix,
                    f"{chunk.chunk_id}/{entity.entity_id}.json",
                )
                await object_store.put(key, entity.to_json_bytes())
                entity_keys.append(key)

        if len(entities) < 2:
            return _ChunkUniversalGraphResult(
                entity_keys=tuple(entity_keys),
                relation_keys=(),
                skipped_relation_chunk_id=chunk.chunk_id,
            )

        chunk_relation_keys = await _existing_chunk_keys(
            object_store,
            prefix=self.config.relations_prefix,
            chunk_id=chunk.chunk_id,
        )
        if chunk_relation_keys:
            relation_keys.extend(chunk_relation_keys)
            return _ChunkUniversalGraphResult(
                entity_keys=tuple(entity_keys),
                relation_keys=tuple(relation_keys),
            )

        relations = await _extract_universal_relations(
            chunk,
            tuple(entities),
            language_model=language_model,
            config=self.config,
        )
        if relations is None:
            return _ChunkUniversalGraphResult(
                entity_keys=tuple(entity_keys),
                relation_keys=(),
                failed_relation_chunk_id=chunk.chunk_id,
            )
        for relation in relations:
            key = join_object_key(
                self.config.relations_prefix,
                f"{chunk.chunk_id}/{relation.relation_id}.json",
            )
            await object_store.put(key, relation.to_json_bytes())
            relation_keys.append(key)
        return _ChunkUniversalGraphResult(
            entity_keys=tuple(entity_keys),
            relation_keys=tuple(relation_keys),
        )


async def _existing_chunk_keys(
    object_store: ObjectStoreProtocol,
    *,
    prefix: str,
    chunk_id: str,
) -> tuple[str, ...]:
    chunk_prefix = join_object_key(prefix, chunk_id)
    marker = f"{chunk_prefix}/"
    return tuple(
        item.key
        for item in sorted(await object_store.list(chunk_prefix), key=lambda value: value.key)
        if item.key.startswith(marker) and item.key.endswith(".json")
    )


async def _extract_universal_entities(
    chunk: ParsedChunk,
    *,
    language_model: LanguageModelProtocol,
    config: ExtractUniversalGraphConfig,
) -> list[ExtractedEntity] | None:
    last_error = ""
    for attempt in range(config.max_attempts):
        prompt = _entity_prompt(chunk, error=last_error if attempt > 0 else None)
        try:
            result = await language_model.invoke(
                ModelRequest(
                    prompt=prompt,
                    system_prompt=ENTITY_EXTRACTION_SYSTEM_PROMPT,
                    options=ModelOptions(
                        temperature=config.temperature,
                        response_format={"type": "json_object"},
                    ),
                    response_schema=UNIVERSAL_ENTITY_RESPONSE_SCHEMA,
                    trace_context={
                        "step": ExtractUniversalGraph.name,
                        "stage": "entity_extraction",
                        "chunk_id": chunk.chunk_id,
                        "attempt": attempt + 1,
                    },
                )
            )
            payload = result.parsed if result.parsed is not None else result.text
            return _entities_from_payload(payload, chunk)
        except Exception as exc:
            last_error = str(exc) or exc.__class__.__name__
    return None


async def _extract_universal_relations(
    chunk: ParsedChunk,
    entities: tuple[ExtractedEntity, ...],
    *,
    language_model: LanguageModelProtocol,
    config: ExtractUniversalGraphConfig,
) -> list[ExtractedRelation] | None:
    last_error = ""
    for attempt in range(config.max_attempts):
        prompt = _relation_prompt(chunk, entities, error=last_error if attempt > 0 else None)
        try:
            result = await language_model.invoke(
                ModelRequest(
                    prompt=prompt,
                    system_prompt=RELATION_EXTRACTION_SYSTEM_PROMPT,
                    options=ModelOptions(
                        temperature=config.temperature,
                        response_format={"type": "json_object"},
                    ),
                    response_schema=UNIVERSAL_RELATION_RESPONSE_SCHEMA,
                    trace_context={
                        "step": ExtractUniversalGraph.name,
                        "stage": "relation_extraction",
                        "chunk_id": chunk.chunk_id,
                        "attempt": attempt + 1,
                    },
                )
            )
            payload = result.parsed if result.parsed is not None else result.text
            return _relations_from_payload(payload, chunk, entities)
        except Exception as exc:
            last_error = str(exc) or exc.__class__.__name__
    return None


def _entity_prompt(chunk: ParsedChunk, *, error: str | None) -> str:
    template = ENTITY_EXTRACTION_RETRY_PROMPT if error else ENTITY_EXTRACTION_PROMPT
    return template.format(
        error=error or "",
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source_name=chunk.source.name,
        chunk_text=chunk.text,
    )


def _relation_prompt(
    chunk: ParsedChunk,
    entities: tuple[ExtractedEntity, ...],
    *,
    error: str | None,
) -> str:
    template = RELATION_EXTRACTION_RETRY_PROMPT if error else RELATION_EXTRACTION_PROMPT
    entities_json = json.dumps(
        [
            {
                "name": entity.name,
                "type": entity.type,
                "subtype": entity.subtype,
                "description": entity.description,
                "attributes": dict(entity.attributes),
            }
            for entity in entities
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return template.format(
        error=error or "",
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source_name=chunk.source.name,
        entities_json=entities_json,
        chunk_text=chunk.text,
    )


def _entities_from_payload(payload: Any, chunk: ParsedChunk) -> list[ExtractedEntity]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("entity response must be a JSON object")
    raw_entities = payload.get("entities")
    if not isinstance(raw_entities, list):
        raise ValueError("entities must be a list")

    entities: list[ExtractedEntity] = []
    for index, raw_entity in enumerate(raw_entities):
        if not isinstance(raw_entity, dict):
            raise ValueError(f"entities[{index}] must be an object")
        entity = _entity_from_raw(raw_entity, chunk)
        if entity is not None:
            entities.append(entity)
    return entities


def _entity_from_raw(raw_entity: Mapping[str, Any], chunk: ParsedChunk) -> ExtractedEntity | None:
    name = _required_string(raw_entity, "name")
    entity_type = _required_string(raw_entity, "type")
    description = _required_string(raw_entity, "description")
    subtype = raw_entity.get("subtype")
    if subtype is not None:
        if not isinstance(subtype, str):
            raise ValueError("subtype must be a string or null")
        subtype = subtype.strip() or None
    attributes = raw_entity.get("attributes")
    if not isinstance(attributes, dict):
        raise ValueError("attributes must be an object")
    normalized_attributes = {
        str(key).strip(): str(value).strip()
        for key, value in attributes.items()
        if str(key).strip() and str(value).strip()
    }
    source_chunk_ids = chunk.parent_chunk_ids or (chunk.chunk_id,)
    return ExtractedEntity(
        entity_id=make_entity_id(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            name=name,
            type=entity_type,
            subtype=subtype,
            description=description,
        ),
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        name=name,
        type=entity_type,
        subtype=subtype,
        description=description,
        attributes=normalized_attributes,
        source_chunk_ids=source_chunk_ids,
    )


def _relations_from_payload(
    payload: Any,
    chunk: ParsedChunk,
    entities: tuple[ExtractedEntity, ...],
) -> list[ExtractedRelation]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("relation response must be a JSON object")
    raw_relations = payload.get("relations")
    if not isinstance(raw_relations, list):
        raise ValueError("relations must be a list")

    entity_by_name = _entity_by_normalized_name(entities)
    relations: list[ExtractedRelation] = []
    for index, raw_relation in enumerate(raw_relations):
        if not isinstance(raw_relation, dict):
            raise ValueError(f"relations[{index}] must be an object")
        relations.append(_relation_from_raw(raw_relation, chunk, entity_by_name))
    return relations


def _relation_from_raw(
    raw_relation: Mapping[str, Any],
    chunk: ParsedChunk,
    entity_by_name: Mapping[str, ExtractedEntity],
) -> ExtractedRelation:
    source_name = _required_string(raw_relation, "source")
    target_name = _required_string(raw_relation, "target")
    relation_type = _required_string(raw_relation, "type")
    relation_name = _required_string(raw_relation, "name")
    description = _required_string(raw_relation, "description")
    source_key = _normalize_name(source_name)
    target_key = _normalize_name(target_name)
    if source_key == target_key:
        raise ValueError("source and target must be different entities")
    source_entity = entity_by_name.get(source_key)
    target_entity = entity_by_name.get(target_key)
    if source_entity is None:
        raise ValueError(f"source entity not found in current chunk: {source_name}")
    if target_entity is None:
        raise ValueError(f"target entity not found in current chunk: {target_name}")
    attributes = raw_relation.get("attributes")
    if not isinstance(attributes, dict):
        raise ValueError("attributes must be an object")
    normalized_attributes = {
        str(key).strip(): str(value).strip()
        for key, value in attributes.items()
        if str(key).strip() and str(value).strip()
    }
    source_chunk_ids = chunk.parent_chunk_ids or (chunk.chunk_id,)
    return ExtractedRelation(
        relation_id=make_relation_id(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            source_entity_id=source_entity.entity_id,
            target_entity_id=target_entity.entity_id,
            type=relation_type,
            name=relation_name,
            description=description,
        ),
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source_entity_id=source_entity.entity_id,
        target_entity_id=target_entity.entity_id,
        source_entity_name=source_entity.name,
        target_entity_name=target_entity.name,
        type=relation_type,
        name=relation_name,
        description=description,
        attributes=normalized_attributes,
        source_chunk_ids=source_chunk_ids,
    )


def _entity_by_normalized_name(
    entities: tuple[ExtractedEntity, ...],
) -> dict[str, ExtractedEntity]:
    result: dict[str, ExtractedEntity] = {}
    for entity in entities:
        result.setdefault(_normalize_name(entity.name), entity)
    return result


def _normalize_name(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()
