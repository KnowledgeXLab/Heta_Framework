"""Ontology constraint step for universal graph artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from heta_framework.common.models import ModelOptions, ModelRequest
from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb.chunking import ParsedChunk
from heta_framework.kb.cleanup import StepCleanupPlan, object_key_targets
from heta_framework.kb.graphing import ExtractedEntity, ExtractedRelation
from heta_framework.kb.graphing.prompts import (
    ONTOLOGY_CONSTRAINT_SYSTEM_PROMPT,
    ONTOLOGY_ENTITY_CONSTRAINT_PROMPT,
    ONTOLOGY_ENTITY_CONSTRAINT_RETRY_PROMPT,
    ONTOLOGY_RELATION_CONSTRAINT_PROMPT,
    ONTOLOGY_RELATION_CONSTRAINT_RETRY_PROMPT,
)
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref
from heta_framework.kb.steps.universal_graph_common import (
    _entities_by_chunk,
    _json,
    _relations_by_chunk,
    _require_language_model,
    _require_object_store,
)


@dataclass(frozen=True)
class ConstrainGraphByOntologyConfig:
    """Configuration for optional ontology/schema graph constraint."""

    schema: Mapping[str, Any]
    entities_prefix: str = "ontology/entities"
    relations_prefix: str = "ontology/relations"
    max_attempts: int = 3
    temperature: float = 0.0
    object_store: str | None = None
    language_model: str | None = None
    chunk_keys_artifact: str = "chunk_keys"
    entity_keys_artifact: str = "entity_keys"
    relation_keys_artifact: str = "relation_keys"
    constrained_entity_keys_artifact: str = "ontology_entity_keys"
    constrained_relation_keys_artifact: str = "ontology_relation_keys"
    result_artifact: str = "constrain_graph_by_ontology_result"

    def __post_init__(self) -> None:
        validate_object_prefix(self.entities_prefix)
        validate_object_prefix(self.relations_prefix)
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if not self.schema:
            raise ValueError("schema must not be empty")


@dataclass(frozen=True)
class ConstrainGraphByOntologyResult:
    """Artifacts produced by ConstrainGraphByOntology."""

    entity_keys: tuple[str, ...]
    relation_keys: tuple[str, ...]
    input_entity_count: int
    input_relation_count: int
    entity_count: int
    relation_count: int
    failed_chunk_ids: tuple[str, ...]


class ConstrainGraphByOntology:
    """Optionally constrain universal graph entities and relations with an ontology schema."""

    name = "constrain_graph_by_ontology"

    def __init__(self, config: ConstrainGraphByOntologyConfig) -> None:
        self.config = config

    @property
    def requirements(self) -> StepRequirements:
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
                    self.config.relation_keys_artifact,
                }
            ),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        return StepCapabilities(
            artifacts=frozenset(
                {
                    self.config.result_artifact,
                    self.config.constrained_entity_keys_artifact,
                    self.config.constrained_relation_keys_artifact,
                }
            )
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        component = store_ref("objects", self.config.object_store).key
        return StepCleanupPlan(
            object_key_targets(
                artifacts,
                self.config.constrained_entity_keys_artifact,
                component=component,
            )
            + object_key_targets(
                artifacts,
                self.config.constrained_relation_keys_artifact,
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
        entity_keys = tuple(context.get_artifact(self.config.entity_keys_artifact))
        relation_keys = tuple(context.get_artifact(self.config.relation_keys_artifact))
        chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]
        entities = [ExtractedEntity.from_json(await object_store.get(key)) for key in entity_keys]
        relations = [
            ExtractedRelation.from_json(await object_store.get(key)) for key in relation_keys
        ]
        entities_by_chunk = _entities_by_chunk(entities)
        relations_by_chunk = _relations_by_chunk(relations)
        schema_json = json.dumps(self.config.schema, ensure_ascii=False, separators=(",", ":"))

        output_entity_keys: list[str] = []
        output_relation_keys: list[str] = []
        failed_chunk_ids: list[str] = []
        for chunk in chunks:
            chunk_entities = entities_by_chunk.get(chunk.chunk_id, ())
            chunk_relations = relations_by_chunk.get(chunk.chunk_id, ())
            constrained_entities = await _constrain_entities(
                chunk,
                chunk_entities,
                schema_json,
                language_model=language_model,
                config=self.config,
            )
            if constrained_entities is None:
                failed_chunk_ids.append(chunk.chunk_id)
                continue
            constrained_relations = await _constrain_relations(
                chunk,
                constrained_entities,
                chunk_relations,
                schema_json,
                language_model=language_model,
                config=self.config,
            )
            if constrained_relations is None:
                failed_chunk_ids.append(chunk.chunk_id)
                continue
            for entity in constrained_entities:
                key = join_object_key(
                    self.config.entities_prefix,
                    f"{chunk.chunk_id}/{entity.entity_id}.json",
                )
                await object_store.put(key, entity.to_json_bytes())
                output_entity_keys.append(key)
            for relation in constrained_relations:
                key = join_object_key(
                    self.config.relations_prefix,
                    f"{chunk.chunk_id}/{relation.relation_id}.json",
                )
                await object_store.put(key, relation.to_json_bytes())
                output_relation_keys.append(key)

        context.set_artifact(
            self.config.constrained_entity_keys_artifact,
            tuple(output_entity_keys),
        )
        context.set_artifact(
            self.config.constrained_relation_keys_artifact,
            tuple(output_relation_keys),
        )
        context.set_artifact(
            self.config.result_artifact,
            ConstrainGraphByOntologyResult(
                entity_keys=tuple(output_entity_keys),
                relation_keys=tuple(output_relation_keys),
                input_entity_count=len(entities),
                input_relation_count=len(relations),
                entity_count=len(output_entity_keys),
                relation_count=len(output_relation_keys),
                failed_chunk_ids=tuple(failed_chunk_ids),
            ),
        )


async def _constrain_entities(
    chunk: ParsedChunk,
    entities: tuple[ExtractedEntity, ...],
    schema_json: str,
    *,
    language_model: LanguageModelProtocol,
    config: ConstrainGraphByOntologyConfig,
) -> tuple[ExtractedEntity, ...] | None:
    if not entities:
        return ()
    last_error = ""
    for attempt in range(config.max_attempts):
        template = (
            ONTOLOGY_ENTITY_CONSTRAINT_RETRY_PROMPT
            if attempt > 0
            else ONTOLOGY_ENTITY_CONSTRAINT_PROMPT
        )
        prompt = template.format(
            error=last_error,
            schema_json=schema_json,
            entities_json=_json([entity.to_dict() for entity in entities]),
            chunk_text=chunk.text,
        )
        try:
            result = await language_model.invoke(
                ModelRequest(
                    prompt=prompt,
                    system_prompt=ONTOLOGY_CONSTRAINT_SYSTEM_PROMPT,
                    options=ModelOptions(
                        temperature=config.temperature,
                        response_format={"type": "json_object"},
                    ),
                    trace_context={
                        "step": ConstrainGraphByOntology.name,
                        "stage": "entities",
                        "chunk_id": chunk.chunk_id,
                        "attempt": attempt + 1,
                    },
                )
            )
            payload = result.parsed if result.parsed is not None else json.loads(result.text)
            raw_entities = payload.get("entities") if isinstance(payload, dict) else None
            if not isinstance(raw_entities, list):
                raise ValueError("entities must be a list")
            return tuple(ExtractedEntity.from_dict(dict(item)) for item in raw_entities)
        except Exception as exc:
            last_error = str(exc) or exc.__class__.__name__
    return None


async def _constrain_relations(
    chunk: ParsedChunk,
    entities: tuple[ExtractedEntity, ...],
    relations: tuple[ExtractedRelation, ...],
    schema_json: str,
    *,
    language_model: LanguageModelProtocol,
    config: ConstrainGraphByOntologyConfig,
) -> tuple[ExtractedRelation, ...] | None:
    if not relations:
        return ()
    entity_ids = {entity.entity_id for entity in entities}
    relation_input = [
        relation
        for relation in relations
        if relation.source_entity_id in entity_ids and relation.target_entity_id in entity_ids
    ]
    if not relation_input:
        return ()
    last_error = ""
    for attempt in range(config.max_attempts):
        template = (
            ONTOLOGY_RELATION_CONSTRAINT_RETRY_PROMPT
            if attempt > 0
            else ONTOLOGY_RELATION_CONSTRAINT_PROMPT
        )
        prompt = template.format(
            error=last_error,
            schema_json=schema_json,
            entities_json=_json([entity.to_dict() for entity in entities]),
            relations_json=_json([relation.to_dict() for relation in relation_input]),
            chunk_text=chunk.text,
        )
        try:
            result = await language_model.invoke(
                ModelRequest(
                    prompt=prompt,
                    system_prompt=ONTOLOGY_CONSTRAINT_SYSTEM_PROMPT,
                    options=ModelOptions(
                        temperature=config.temperature,
                        response_format={"type": "json_object"},
                    ),
                    trace_context={
                        "step": ConstrainGraphByOntology.name,
                        "stage": "relations",
                        "chunk_id": chunk.chunk_id,
                        "attempt": attempt + 1,
                    },
                )
            )
            payload = result.parsed if result.parsed is not None else json.loads(result.text)
            raw_relations = payload.get("relations") if isinstance(payload, dict) else None
            if not isinstance(raw_relations, list):
                raise ValueError("relations must be a list")
            constrained = tuple(ExtractedRelation.from_dict(dict(item)) for item in raw_relations)
            return tuple(
                relation
                for relation in constrained
                if (
                    relation.source_entity_id in entity_ids
                    and relation.target_entity_id in entity_ids
                )
            )
        except Exception as exc:
            last_error = str(exc) or exc.__class__.__name__
    return None
