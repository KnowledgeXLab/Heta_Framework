"""Extract graph relations from chunk and entity artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from heta_framework.common.models import ModelOptions, ModelRequest
from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb.chunking import ParsedChunk
from heta_framework.kb.graphing import ExtractedEntity, ExtractedRelation, make_relation_id
from heta_framework.kb.graphing.prompts import (
    RELATION_EXTRACTION_PROMPT,
    RELATION_EXTRACTION_RETRY_PROMPT,
    RELATION_EXTRACTION_SYSTEM_PROMPT,
)
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref


RELATION_RESPONSE_SCHEMA: dict[str, Any] = {
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
class ExtractRelationsConfig:
    """Configuration for ExtractRelations."""

    relations_prefix: str = "relations"
    max_attempts: int = 3
    temperature: float = 0.0
    object_store: str | None = None
    language_model: str | None = None
    chunk_keys_artifact: str = "chunk_keys"
    entity_keys_artifact: str = "entity_keys"
    relation_keys_artifact: str = "relation_keys"

    def __post_init__(self) -> None:
        validate_object_prefix(self.relations_prefix)
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if self.chunk_keys_artifact.strip() == "":
            raise ValueError("chunk_keys_artifact must not be empty")
        if self.entity_keys_artifact.strip() == "":
            raise ValueError("entity_keys_artifact must not be empty")
        if self.relation_keys_artifact.strip() == "":
            raise ValueError("relation_keys_artifact must not be empty")


@dataclass(frozen=True)
class ExtractRelationsResult:
    """Artifacts produced by ExtractRelations."""

    relation_keys: tuple[str, ...]
    chunk_count: int
    relation_count: int
    skipped_chunk_ids: tuple[str, ...]
    failed_chunk_ids: tuple[str, ...]


class ExtractRelations:
    """Extract typed relation artifacts from ParsedChunk and ExtractedEntity objects."""

    name = "extract_relations"

    def __init__(self, config: ExtractRelationsConfig | None = None) -> None:
        self.config = config or ExtractRelationsConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components and artifacts required by this step."""
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("objects", self.config.object_store),
                    model_ref("language", self.config.language_model),
                }
            ),
            artifacts=frozenset(
                {
                    self.config.chunk_keys_artifact,
                    self.config.entity_keys_artifact,
                }
            ),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts produced by this step."""
        return StepCapabilities(
            artifacts=frozenset({"extract_relations_result", self.config.relation_keys_artifact})
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Run relation extraction and store ExtractedRelation JSON objects."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        language_model = _require_language_model(
            context.get_component(model_ref("language", self.config.language_model).key)
        )
        chunk_keys = tuple(context.get_artifact(self.config.chunk_keys_artifact))
        entity_keys = tuple(context.get_artifact(self.config.entity_keys_artifact))
        chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]
        entities = [ExtractedEntity.from_json(await object_store.get(key)) for key in entity_keys]
        entities_by_chunk_id = _group_entities_by_chunk_id(entities)

        relation_keys: list[str] = []
        skipped_chunk_ids: list[str] = []
        failed_chunk_ids: list[str] = []
        for chunk in chunks:
            chunk_entities = entities_by_chunk_id.get(chunk.chunk_id, ())
            if len(chunk_entities) < 2:
                skipped_chunk_ids.append(chunk.chunk_id)
                continue
            relations = await _extract_relations_from_chunk(
                chunk,
                chunk_entities,
                language_model=language_model,
                config=self.config,
            )
            if relations is None:
                failed_chunk_ids.append(chunk.chunk_id)
                continue
            for relation in relations:
                relation_key = join_object_key(
                    self.config.relations_prefix,
                    f"{chunk.chunk_id}/{relation.relation_id}.json",
                )
                await object_store.put(relation_key, relation.to_json_bytes())
                relation_keys.append(relation_key)

        result = ExtractRelationsResult(
            relation_keys=tuple(relation_keys),
            chunk_count=len(chunks),
            relation_count=len(relation_keys),
            skipped_chunk_ids=tuple(skipped_chunk_ids),
            failed_chunk_ids=tuple(failed_chunk_ids),
        )
        context.set_artifact("extract_relations_result", result)
        context.set_artifact(self.config.relation_keys_artifact, result.relation_keys)


async def _extract_relations_from_chunk(
    chunk: ParsedChunk,
    entities: tuple[ExtractedEntity, ...],
    *,
    language_model: LanguageModelProtocol,
    config: ExtractRelationsConfig,
) -> list[ExtractedRelation] | None:
    last_error = ""
    for attempt in range(config.max_attempts):
        prompt = _build_prompt(chunk, entities, error=last_error if attempt > 0 else None)
        try:
            result = await language_model.invoke(
                ModelRequest(
                    prompt=prompt,
                    system_prompt=RELATION_EXTRACTION_SYSTEM_PROMPT,
                    options=ModelOptions(
                        temperature=config.temperature,
                        response_format={"type": "json_object"},
                    ),
                    response_schema=RELATION_RESPONSE_SCHEMA,
                    trace_context={
                        "step": ExtractRelations.name,
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


def _build_prompt(
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


def _relations_from_payload(
    payload: Any,
    chunk: ParsedChunk,
    entities: tuple[ExtractedEntity, ...],
) -> list[ExtractedRelation]:
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
    raw_relation: dict[str, Any],
    chunk: ParsedChunk,
    entity_by_name: dict[str, ExtractedEntity],
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


def _group_entities_by_chunk_id(
    entities: list[ExtractedEntity],
) -> dict[str, tuple[ExtractedEntity, ...]]:
    grouped: dict[str, list[ExtractedEntity]] = {}
    for entity in entities:
        grouped.setdefault(entity.chunk_id, []).append(entity)
    return {chunk_id: tuple(items) for chunk_id, items in grouped.items()}


def _entity_by_normalized_name(
    entities: tuple[ExtractedEntity, ...],
) -> dict[str, ExtractedEntity]:
    result: dict[str, ExtractedEntity] = {}
    for entity in entities:
        result.setdefault(_normalize_name(entity.name), entity)
    return result


def _normalize_name(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_language_model(component: object) -> LanguageModelProtocol:
    if not isinstance(component, LanguageModelProtocol):
        raise TypeError("models.language must satisfy LanguageModelProtocol")
    return component
