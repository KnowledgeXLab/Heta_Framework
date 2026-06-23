"""Extract graph entities from chunk artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from heta_framework.common.models import ModelOptions, ModelRequest
from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb.chunking import ParsedChunk
from heta_framework.kb.graphing import ExtractedEntity, make_entity_id
from heta_framework.kb.graphing.prompts import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_EXTRACTION_RETRY_PROMPT,
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
)
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref


ENTITY_RESPONSE_SCHEMA: dict[str, Any] = {
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


@dataclass(frozen=True)
class ExtractEntitiesConfig:
    """Configuration for ExtractEntities."""

    entities_prefix: str = "entities"
    max_attempts: int = 3
    temperature: float = 0.0
    object_store: str | None = None
    language_model: str | None = None
    chunk_keys_artifact: str = "chunk_keys"
    entity_keys_artifact: str = "entity_keys"

    def __post_init__(self) -> None:
        validate_object_prefix(self.entities_prefix)
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if self.chunk_keys_artifact.strip() == "":
            raise ValueError("chunk_keys_artifact must not be empty")
        if self.entity_keys_artifact.strip() == "":
            raise ValueError("entity_keys_artifact must not be empty")


@dataclass(frozen=True)
class ExtractEntitiesResult:
    """Artifacts produced by ExtractEntities."""

    entity_keys: tuple[str, ...]
    chunk_count: int
    entity_count: int
    failed_chunk_ids: tuple[str, ...]


class ExtractEntities:
    """Extract typed entity artifacts from ParsedChunk JSON objects."""

    name = "extract_entities"

    def __init__(self, config: ExtractEntitiesConfig | None = None) -> None:
        self.config = config or ExtractEntitiesConfig()

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
            artifacts=frozenset({self.config.chunk_keys_artifact}),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts produced by this step."""
        return StepCapabilities(
            artifacts=frozenset({"extract_entities_result", self.config.entity_keys_artifact})
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Run entity extraction and store ExtractedEntity JSON objects."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        language_model = _require_language_model(
            context.get_component(model_ref("language", self.config.language_model).key)
        )
        chunk_keys = tuple(context.get_artifact(self.config.chunk_keys_artifact))
        chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]

        entity_keys: list[str] = []
        failed_chunk_ids: list[str] = []
        for chunk in chunks:
            entities = await _extract_entities_from_chunk(
                chunk,
                language_model=language_model,
                config=self.config,
            )
            if entities is None:
                failed_chunk_ids.append(chunk.chunk_id)
                continue
            for entity in entities:
                entity_key = join_object_key(
                    self.config.entities_prefix,
                    f"{chunk.chunk_id}/{entity.entity_id}.json",
                )
                await object_store.put(entity_key, entity.to_json_bytes())
                entity_keys.append(entity_key)

        result = ExtractEntitiesResult(
            entity_keys=tuple(entity_keys),
            chunk_count=len(chunks),
            entity_count=len(entity_keys),
            failed_chunk_ids=tuple(failed_chunk_ids),
        )
        context.set_artifact("extract_entities_result", result)
        context.set_artifact(self.config.entity_keys_artifact, result.entity_keys)


async def _extract_entities_from_chunk(
    chunk: ParsedChunk,
    *,
    language_model: LanguageModelProtocol,
    config: ExtractEntitiesConfig,
) -> list[ExtractedEntity] | None:
    last_error = ""
    for attempt in range(config.max_attempts):
        prompt = _build_prompt(chunk, error=last_error if attempt > 0 else None)
        try:
            result = await language_model.invoke(
                ModelRequest(
                    prompt=prompt,
                    system_prompt=ENTITY_EXTRACTION_SYSTEM_PROMPT,
                    options=ModelOptions(
                        temperature=config.temperature,
                        response_format={"type": "json_object"},
                    ),
                    response_schema=ENTITY_RESPONSE_SCHEMA,
                    trace_context={
                        "step": ExtractEntities.name,
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


def _build_prompt(chunk: ParsedChunk, *, error: str | None) -> str:
    template = ENTITY_EXTRACTION_RETRY_PROMPT if error else ENTITY_EXTRACTION_PROMPT
    return template.format(
        error=error or "",
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source_name=chunk.source.name,
        chunk_text=chunk.text,
    )


def _entities_from_payload(payload: Any, chunk: ParsedChunk) -> list[ExtractedEntity]:
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


def _entity_from_raw(raw_entity: dict[str, Any], chunk: ParsedChunk) -> ExtractedEntity | None:
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
