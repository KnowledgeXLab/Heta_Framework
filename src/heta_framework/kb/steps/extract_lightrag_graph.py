"""Extract LightRAG-style graph records from chunk artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import html
import importlib.util
import json
import re
import sys
import time
import types
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from heta_framework.common.models import ModelOptions, ModelRequest
from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.common.stores.graph import GraphEdge, GraphNode, GraphStoreProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb.chunking import ParsedChunk
from heta_framework.kb.cleanup import StepCleanupPlan, object_key_targets
from heta_framework.kb.graphing.prompts import GRAPH_SUMMARY_PROMPT
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref


LIGHTRAG_FIELD_SEP = "<SEP>"
LIGHTRAG_TUPLE_DELIMITER = "<|#|>"
LIGHTRAG_COMPLETION_DELIMITER = "<|COMPLETE|>"
_INVALID_ENTITY_TYPE_CHARS = frozenset("'()<>|/\\")

NodeRecord = dict[str, Any]
EdgeRecord = dict[str, Any]


@dataclass(frozen=True)
class ExtractLightRAGGraphConfig:
    """Configuration for ExtractLightRAGGraph."""

    extraction_format: Literal["json", "tuple"] = "json"
    max_attempts: int = 3
    temperature: float = 0.0
    object_store: str | None = None
    graph_store: str | None = None
    language_model: str | None = None
    chunk_keys_artifact: str = "chunk_keys"
    entity_keys_artifact: str = "light_rag_entity_keys"
    graph_nodes_prefix: str = "light_rag/graph/nodes"
    graph_edges_prefix: str = "light_rag/graph/edges"
    graph_node_keys_artifact: str = "light_rag_graph_node_keys"
    graph_edge_keys_artifact: str = "light_rag_graph_edge_keys"
    result_artifact: str = "extract_light_rag_graph_result"
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    summary_llm_max_tokens: int = 1200
    language: str = "English"
    max_total_records: int = 1000
    max_entity_records: int = 1000
    entity_types_guidance: str | None = None
    json_system_prompt: str | None = None
    json_user_prompt: str | None = None
    json_continue_prompt: str | None = None
    tuple_system_prompt: str | None = None
    tuple_user_prompt: str | None = None
    tuple_continue_prompt: str | None = None
    summary_prompt: str | None = None

    def __post_init__(self) -> None:
        validate_object_prefix(self.graph_nodes_prefix)
        validate_object_prefix(self.graph_edges_prefix)
        if self.extraction_format not in {"json", "tuple"}:
            raise ValueError("extraction_format must be one of: json, tuple")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if self.entity_extract_max_gleaning < 0:
            raise ValueError("entity_extract_max_gleaning must not be negative")
        if self.entity_summary_to_max_tokens <= 0:
            raise ValueError("entity_summary_to_max_tokens must be greater than zero")
        if self.summary_llm_max_tokens <= 0:
            raise ValueError("summary_llm_max_tokens must be greater than zero")
        for field_name in (
            "chunk_keys_artifact",
            "entity_keys_artifact",
            "graph_node_keys_artifact",
            "graph_edge_keys_artifact",
            "result_artifact",
        ):
            if str(getattr(self, field_name)).strip() == "":
                raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True)
class ExtractLightRAGGraphResult:
    """Artifacts produced by ExtractLightRAGGraph."""

    node_keys: tuple[str, ...]
    edge_keys: tuple[str, ...]
    chunk_count: int
    entity_count: int
    relation_count: int
    failed_chunk_ids: tuple[str, ...]
    extraction_format: Literal["json", "tuple"]


@dataclass(frozen=True)
class _ChunkGraphExtraction:
    nodes: Mapping[str, list[NodeRecord]]
    edges: Mapping[tuple[str, str], list[EdgeRecord]]
    failed: bool = False


class ExtractLightRAGGraph:
    """Extract LightRAG graph nodes and edges from parsed chunks."""

    name = "extract_light_rag_graph"

    def __init__(self, config: ExtractLightRAGGraphConfig | None = None) -> None:
        self.config = config or ExtractLightRAGGraphConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components and artifacts required by this step."""
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("objects", self.config.object_store),
                    store_ref("graph", self.config.graph_store),
                    model_ref("language", self.config.language_model),
                }
            ),
            artifacts=frozenset({self.config.chunk_keys_artifact}),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts produced by this step."""
        return StepCapabilities(
            artifacts=frozenset(
                {
                    self.config.result_artifact,
                    self.config.entity_keys_artifact,
                    self.config.graph_node_keys_artifact,
                    self.config.graph_edge_keys_artifact,
                }
            )
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        """Return graph node and edge object artifacts produced by this step."""
        component = store_ref("objects", self.config.object_store).key
        return StepCleanupPlan(
            object_key_targets(
                artifacts,
                self.config.graph_node_keys_artifact,
                component=component,
            )
            + object_key_targets(
                artifacts,
                self.config.graph_edge_keys_artifact,
                component=component,
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Run LightRAG graph extraction and persist graph artifacts."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        graph_store = _require_graph_store(
            context.get_component(store_ref("graph", self.config.graph_store).key)
        )
        language_model = _require_language_model(
            context.get_component(model_ref("language", self.config.language_model).key)
        )

        chunk_keys = tuple(context.get_artifact(self.config.chunk_keys_artifact))
        chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]

        extractions = await asyncio.gather(
            *(
                self._process_single_content(chunk, language_model=language_model)
                for chunk in chunks
            )
        )

        maybe_nodes: defaultdict[str, list[NodeRecord]] = defaultdict(list)
        maybe_edges: defaultdict[tuple[str, str], list[EdgeRecord]] = defaultdict(list)
        failed_chunk_ids: list[str] = []

        for chunk, extraction in zip(chunks, extractions, strict=True):
            if extraction.failed:
                failed_chunk_ids.append(chunk.chunk_id)
                continue
            for name, records in extraction.nodes.items():
                maybe_nodes[name].extend(records)
            for endpoints, records in extraction.edges.items():
                maybe_edges[tuple(sorted(endpoints))].extend(records)

        graph_nodes = [
            await _merge_node_then_upsert(
                entity_name,
                records,
                graph_store,
                language_model=language_model,
                config=self.config,
            )
            for entity_name, records in maybe_nodes.items()
        ]
        graph_edges = [
            edge
            for endpoints, records in maybe_edges.items()
            if (
                edge := await _merge_edge_then_upsert(
                    endpoints[0],
                    endpoints[1],
                    records,
                    graph_store,
                    language_model=language_model,
                    config=self.config,
                )
            )
            is not None
        ]

        node_keys = tuple(
            [await _put_graph_node(object_store, self.config, node) for node in graph_nodes]
        )
        edge_keys = tuple(
            [await _put_graph_edge(object_store, self.config, edge) for edge in graph_edges]
        )

        result = ExtractLightRAGGraphResult(
            node_keys=node_keys,
            edge_keys=edge_keys,
            chunk_count=len(chunks),
            entity_count=len(graph_nodes),
            relation_count=len(graph_edges),
            failed_chunk_ids=tuple(failed_chunk_ids),
            extraction_format=self.config.extraction_format,
        )
        context.set_artifact(self.config.result_artifact, result)
        context.set_artifact(self.config.entity_keys_artifact, result.node_keys)
        context.set_artifact(self.config.graph_node_keys_artifact, result.node_keys)
        context.set_artifact(self.config.graph_edge_keys_artifact, result.edge_keys)

    async def _process_single_content(
        self,
        chunk: ParsedChunk,
        *,
        language_model: LanguageModelProtocol,
    ) -> _ChunkGraphExtraction:
        prompts = _lightrag_prompts()
        prompt_set = _build_prompt_set(self.config, prompts, chunk)
        response_format = {"type": "json_object"} if self.config.extraction_format == "json" else None
        final_result = await _invoke_lightrag_extraction(
            language_model,
            prompt=prompt_set.user_prompt,
            system_prompt=prompt_set.system_prompt,
            history_messages=[],
            chunk=chunk,
            config=self.config,
            response_format=response_format,
        )
        if final_result is None:
            return _ChunkGraphExtraction(nodes={}, edges={}, failed=True)

        extraction = _parse_lightrag_records(
            final_result,
            chunk,
            self.config,
            timestamp=int(time.time()),
        )
        if extraction.failed:
            return extraction

        history = _pack_user_assistant_messages(prompt_set.user_prompt, final_result)
        for _ in range(self.config.entity_extract_max_gleaning):
            glean_result = await _invoke_lightrag_extraction(
                language_model,
                prompt=prompt_set.continue_prompt,
                system_prompt=prompt_set.system_prompt,
                history_messages=history,
                chunk=chunk,
                config=self.config,
                response_format=response_format,
            )
            if glean_result is None:
                break
            history += _pack_user_assistant_messages(prompt_set.continue_prompt, glean_result)
            glean_extraction = _parse_lightrag_records(
                glean_result,
                chunk,
                self.config,
                timestamp=int(time.time()),
            )
            if glean_extraction.failed:
                break
            extraction = _merge_gleaning_extraction(extraction, glean_extraction)

        return extraction


@dataclass(frozen=True)
class _PromptSet:
    system_prompt: str
    user_prompt: str
    continue_prompt: str


def _build_prompt_set(
    config: ExtractLightRAGGraphConfig,
    prompts: Mapping[str, Any],
    chunk: ParsedChunk,
) -> _PromptSet:
    entity_types_guidance = (
        config.entity_types_guidance
        or str(prompts["default_entity_types_guidance"]).rstrip()
    )
    heading_context_block = ""
    if config.extraction_format == "json":
        examples = "\n".join(
            str(example).rstrip()
            for example in prompts.get("entity_extraction_json_examples", [])
        )
        context_base = {
            "entity_types_guidance": entity_types_guidance,
            "examples": examples,
            "language": config.language,
            "max_total_records": config.max_total_records,
            "max_entity_records": config.max_entity_records,
        }
        system_prompt = (
            config.json_system_prompt
            or str(prompts["entity_extraction_json_system_prompt"])
        ).format(**context_base)
        user_prompt = (
            config.json_user_prompt
            or str(prompts["entity_extraction_json_user_prompt"])
        ).format(
            **context_base,
            input_text=chunk.text,
            heading_context_block=heading_context_block,
        )
        continue_prompt = (
            config.json_continue_prompt
            or str(prompts["entity_continue_extraction_json_user_prompt"])
        ).format(**context_base)
    else:
        raw_examples = "\n".join(
            str(example).rstrip() for example in prompts.get("entity_extraction_examples", [])
        )
        example_context_base = {
            "tuple_delimiter": LIGHTRAG_TUPLE_DELIMITER,
            "completion_delimiter": LIGHTRAG_COMPLETION_DELIMITER,
            "entity_types_guidance": entity_types_guidance,
            "language": config.language,
        }
        examples = raw_examples.format(**example_context_base)
        context_base = {
            **example_context_base,
            "examples": examples,
            "max_total_records": config.max_total_records,
            "max_entity_records": config.max_entity_records,
        }
        system_prompt = (
            config.tuple_system_prompt or str(prompts["entity_extraction_system_prompt"])
        ).format(**context_base)
        user_prompt = (
            config.tuple_user_prompt or str(prompts["entity_extraction_user_prompt"])
        ).format(
            **context_base,
            input_text=chunk.text,
            heading_context_block=heading_context_block,
        )
        continue_prompt = (
            config.tuple_continue_prompt
            or str(prompts["entity_continue_extraction_user_prompt"])
        ).format(**context_base, input_text=chunk.text)
    return _PromptSet(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        continue_prompt=continue_prompt,
    )


async def _invoke_lightrag_extraction(
    language_model: LanguageModelProtocol,
    *,
    prompt: str,
    system_prompt: str,
    history_messages: list[dict[str, str]],
    chunk: ParsedChunk,
    config: ExtractLightRAGGraphConfig,
    response_format: dict[str, str] | None,
) -> str | None:
    history_text = _messages_to_prompt(history_messages)
    request_prompt = f"{history_text}\n\nuser:\n{prompt}".strip() if history_text else prompt
    last_error = ""
    for attempt in range(config.max_attempts):
        current_prompt = request_prompt
        if last_error:
            current_prompt = (
                f"{request_prompt}\n\n"
                f"Previous response was invalid or failed with this error:\n{last_error}\n\n"
                "Return the corrected answer using the requested format."
            )
        try:
            result = await language_model.invoke(
                ModelRequest(
                    prompt=current_prompt,
                    system_prompt=system_prompt,
                    options=ModelOptions(
                        temperature=config.temperature,
                        response_format=response_format,
                    ),
                    trace_context={
                        "step": ExtractLightRAGGraph.name,
                        "chunk_id": chunk.chunk_id,
                        "attempt": attempt + 1,
                        "extraction_format": config.extraction_format,
                    },
                )
            )
            text = result.text.strip()
            if text:
                return text
            if result.parsed is not None:
                return json.dumps(result.parsed, ensure_ascii=False)
            raise ValueError("model returned empty text")
        except Exception as exc:
            last_error = str(exc) or exc.__class__.__name__
    return None


def _parse_lightrag_records(
    content: str,
    chunk: ParsedChunk,
    config: ExtractLightRAGGraphConfig,
    *,
    timestamp: int,
) -> _ChunkGraphExtraction:
    if config.extraction_format == "json":
        return _parse_json_records(content, chunk, config, timestamp=timestamp)
    return _parse_tuple_records(content, chunk, timestamp=timestamp)


def _parse_json_records(
    content: str,
    chunk: ParsedChunk,
    config: ExtractLightRAGGraphConfig,
    *,
    timestamp: int,
) -> _ChunkGraphExtraction:
    text = _strip_json_fence(content).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return _ChunkGraphExtraction(nodes={}, edges={}, failed=True)
    if not isinstance(parsed, dict):
        return _ChunkGraphExtraction(nodes={}, edges={}, failed=True)

    nodes: defaultdict[str, list[NodeRecord]] = defaultdict(list)
    edges: defaultdict[tuple[str, str], list[EdgeRecord]] = defaultdict(list)
    file_path = _chunk_file_path(chunk)

    entities = parsed.get("entities", [])
    relationships = parsed.get("relationships", [])
    if not isinstance(entities, list) or not isinstance(relationships, list):
        return _ChunkGraphExtraction(nodes={}, edges={}, failed=True)

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        name = _clean_text(entity.get("name"), remove_inner_quotes=True)
        entity_type = _clean_entity_type(entity.get("type"))
        description = _clean_text(entity.get("description"))
        if not name or entity_type is None or not description:
            continue
        record = {
            "entity_name": name,
            "entity_type": entity_type,
            "description": description,
            "source_id": chunk.chunk_id,
            "file_path": file_path,
            "timestamp": timestamp,
            "extraction_format": config.extraction_format,
            "raw_entity_type": _clean_text(entity.get("type"), remove_inner_quotes=True),
        }
        nodes[name].append(record)

    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        source = _clean_text(relationship.get("source"), remove_inner_quotes=True)
        target = _clean_text(relationship.get("target"), remove_inner_quotes=True)
        description = _clean_text(relationship.get("description"))
        keywords = _clean_text(relationship.get("keywords"), remove_inner_quotes=True).replace(
            "，", ","
        )
        if not source or not target or source == target or not description:
            continue
        weight = _float_or_default(relationship.get("weight"), 1.0)
        record = {
            "src_id": source,
            "tgt_id": target,
            "description": description,
            "keywords": keywords,
            "weight": weight,
            "source_id": chunk.chunk_id,
            "file_path": file_path,
            "timestamp": timestamp,
            "extraction_format": config.extraction_format,
        }
        edges[(source, target)].append(record)

    return _ChunkGraphExtraction(nodes=dict(nodes), edges=dict(edges))


def _parse_tuple_records(
    content: str,
    chunk: ParsedChunk,
    *,
    timestamp: int,
) -> _ChunkGraphExtraction:
    records = _split_by_markers(
        content,
        ["\n", LIGHTRAG_COMPLETION_DELIMITER, LIGHTRAG_COMPLETION_DELIMITER.lower()],
    )
    nodes: defaultdict[str, list[NodeRecord]] = defaultdict(list)
    edges: defaultdict[tuple[str, str], list[EdgeRecord]] = defaultdict(list)
    file_path = _chunk_file_path(chunk)

    for record in records:
        record = record.strip().strip("()")
        if not record:
            continue
        attributes = [_clean_text(part, remove_inner_quotes=True) for part in record.split(LIGHTRAG_TUPLE_DELIMITER)]
        attributes = [part for part in attributes if part != ""]
        kind = attributes[0].lower() if attributes else ""
        if "entity" in kind and len(attributes) == 4:
            name = attributes[1]
            entity_type = _clean_entity_type(attributes[2])
            description = _clean_text(attributes[3])
            if not name or entity_type is None or not description:
                continue
            nodes[name].append(
                {
                    "entity_name": name,
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": chunk.chunk_id,
                    "file_path": file_path,
                    "timestamp": timestamp,
                    "extraction_format": "tuple",
                    "raw_entity_type": attributes[2],
                }
            )
            continue
        if "relation" in kind and len(attributes) == 5:
            source = attributes[1]
            target = attributes[2]
            keywords = _clean_text(attributes[3]).replace("，", ",")
            description = _clean_text(attributes[4])
            if not source or not target or source == target or not description:
                continue
            edges[(source, target)].append(
                {
                    "src_id": source,
                    "tgt_id": target,
                    "description": description,
                    "keywords": keywords,
                    "weight": 1.0,
                    "source_id": chunk.chunk_id,
                    "file_path": file_path,
                    "timestamp": timestamp,
                    "extraction_format": "tuple",
                }
            )

    return _ChunkGraphExtraction(nodes=dict(nodes), edges=dict(edges))


def _merge_gleaning_extraction(
    original: _ChunkGraphExtraction,
    gleaned: _ChunkGraphExtraction,
) -> _ChunkGraphExtraction:
    nodes: defaultdict[str, list[NodeRecord]] = defaultdict(list)
    edges: defaultdict[tuple[str, str], list[EdgeRecord]] = defaultdict(list)
    for name, records in original.nodes.items():
        nodes[name].extend(records)
    for endpoints, records in original.edges.items():
        edges[endpoints].extend(records)

    for name, records in gleaned.nodes.items():
        if name in nodes:
            if _record_text_score(records[0], include_keywords=False) > _record_text_score(
                nodes[name][0], include_keywords=False
            ):
                nodes[name] = list(records)
        else:
            nodes[name].extend(records)
    for endpoints, records in gleaned.edges.items():
        if endpoints in edges:
            if _record_text_score(records[0], include_keywords=True) > _record_text_score(
                edges[endpoints][0], include_keywords=True
            ):
                edges[endpoints] = list(records)
        else:
            edges[endpoints].extend(records)
    return _ChunkGraphExtraction(nodes=dict(nodes), edges=dict(edges))


async def _merge_node_then_upsert(
    entity_name: str,
    nodes_data: list[NodeRecord],
    graph_store: GraphStoreProtocol,
    *,
    language_model: LanguageModelProtocol,
    config: ExtractLightRAGGraphConfig,
) -> GraphNode:
    existing_types: list[str] = []
    existing_source_ids: list[str] = []
    existing_descriptions: list[str] = []
    existing_file_paths: list[str] = []

    existing_node = await graph_store.get_node(entity_name)
    if existing_node is not None:
        properties = dict(existing_node.properties)
        existing_types.extend(_property_values(properties.get("entity_type")))
        existing_source_ids.extend(_property_values(properties.get("source_ids")))
        existing_source_ids.extend(_property_values(properties.get("source_id")))
        existing_descriptions.extend(_property_values(properties.get("description")))
        existing_file_paths.extend(_property_values(properties.get("file_paths")))
        existing_file_paths.extend(_property_values(properties.get("file_path")))

    entity_type = _most_common_nonempty(
        [str(record.get("entity_type") or "") for record in nodes_data] + existing_types,
        default="unknown",
    )
    descriptions = _ordered_unique_descriptions(nodes_data, existing_descriptions)
    description = await _summarize_if_needed(
        entity_name,
        descriptions,
        language_model=language_model,
        config=config,
    )
    source_ids = _unique_nonempty(
        [str(record.get("source_id") or "") for record in nodes_data] + existing_source_ids
    )
    file_paths = _unique_nonempty(
        [str(record.get("file_path") or "") for record in nodes_data] + existing_file_paths
    )
    extraction_formats = _unique_nonempty(
        [str(record.get("extraction_format") or "") for record in nodes_data]
    )

    node = GraphNode(
        id=entity_name,
        labels=("Entity", entity_type),
        properties={
            "name": entity_name,
            "entity_name": entity_name,
            "entity_type": entity_type,
            "description": description,
            "source_ids": source_ids,
            "source_id": LIGHTRAG_FIELD_SEP.join(source_ids),
            "file_paths": file_paths,
            "file_path": LIGHTRAG_FIELD_SEP.join(file_paths),
            "extraction_format": extraction_formats[0] if extraction_formats else config.extraction_format,
            "raw_entity_types": _unique_nonempty(
                [str(record.get("raw_entity_type") or "") for record in nodes_data]
            ),
        },
    )
    await graph_store.upsert_nodes([node])
    return node


async def _merge_edge_then_upsert(
    source_id: str,
    target_id: str,
    edges_data: list[EdgeRecord],
    graph_store: GraphStoreProtocol,
    *,
    language_model: LanguageModelProtocol,
    config: ExtractLightRAGGraphConfig,
) -> GraphEdge | None:
    if source_id == target_id or not edges_data:
        return None

    edge_id = _edge_id(source_id, target_id)
    existing_weights: list[float] = []
    existing_source_ids: list[str] = []
    existing_descriptions: list[str] = []
    existing_keywords: list[str] = []
    existing_file_paths: list[str] = []

    existing_edge = await graph_store.get_edge(edge_id)
    if existing_edge is not None:
        properties = dict(existing_edge.properties)
        existing_weights.extend(
            [_float_or_default(value, 0.0) for value in _property_values(properties.get("weight"))]
        )
        existing_source_ids.extend(_property_values(properties.get("source_ids")))
        existing_source_ids.extend(_property_values(properties.get("source_id")))
        existing_descriptions.extend(_property_values(properties.get("description")))
        existing_keywords.extend(_property_values(properties.get("keywords")))
        existing_file_paths.extend(_property_values(properties.get("file_paths")))
        existing_file_paths.extend(_property_values(properties.get("file_path")))

    descriptions = _ordered_unique_descriptions(edges_data, existing_descriptions)
    if not descriptions:
        return None
    description = await _summarize_if_needed(
        f"({source_id}, {target_id})",
        descriptions,
        language_model=language_model,
        config=config,
    )
    source_ids = _unique_nonempty(
        [str(record.get("source_id") or "") for record in edges_data] + existing_source_ids
    )
    file_paths = _unique_nonempty(
        [str(record.get("file_path") or "") for record in edges_data] + existing_file_paths
    )
    keywords = ",".join(
        sorted(
            {
                keyword.strip()
                for value in [str(record.get("keywords") or "") for record in edges_data]
                + existing_keywords
                for keyword in value.split(",")
                if keyword.strip()
            }
        )
    )
    weight = sum(
        [_float_or_default(record.get("weight"), 1.0) for record in edges_data]
        + existing_weights
    )
    extraction_formats = _unique_nonempty(
        [str(record.get("extraction_format") or "") for record in edges_data]
    )

    for node_id in (source_id, target_id):
        if await graph_store.get_node(node_id) is None:
            await graph_store.upsert_nodes(
                [
                    GraphNode(
                        id=node_id,
                        labels=("Entity", "unknown"),
                        properties={
                            "name": node_id,
                            "entity_name": node_id,
                            "entity_type": "unknown",
                            "description": "",
                            "source_ids": source_ids,
                            "source_id": LIGHTRAG_FIELD_SEP.join(source_ids),
                            "file_paths": file_paths,
                            "file_path": LIGHTRAG_FIELD_SEP.join(file_paths),
                            "extraction_format": config.extraction_format,
                        },
                    )
                ]
            )

    edge = GraphEdge(
        id=edge_id,
        source_id=source_id,
        target_id=target_id,
        type="RELATED",
        properties={
            "src_id": source_id,
            "tgt_id": target_id,
            "description": description,
            "keywords": keywords,
            "weight": weight,
            "source_ids": source_ids,
            "source_id": LIGHTRAG_FIELD_SEP.join(source_ids),
            "file_paths": file_paths,
            "file_path": LIGHTRAG_FIELD_SEP.join(file_paths),
            "extraction_format": extraction_formats[0] if extraction_formats else config.extraction_format,
        },
    )
    await graph_store.upsert_edges([edge])
    return edge


async def _summarize_if_needed(
    entity_name: str,
    descriptions: list[str],
    *,
    language_model: LanguageModelProtocol,
    config: ExtractLightRAGGraphConfig,
) -> str:
    descriptions = _unique_nonempty(descriptions)
    if not descriptions:
        return ""
    description = LIGHTRAG_FIELD_SEP.join(descriptions)
    if len(description.split()) <= config.entity_summary_to_max_tokens:
        return description
    prompt_template = config.summary_prompt or GRAPH_SUMMARY_PROMPT
    result = await language_model.invoke(
        ModelRequest(
            prompt=prompt_template.format(
                entity_name=entity_name,
                description_list=json.dumps(descriptions, ensure_ascii=False),
            ),
            options=ModelOptions(
                temperature=config.temperature,
                max_output_tokens=config.summary_llm_max_tokens,
            ),
            trace_context={"step": ExtractLightRAGGraph.name, "stage": "summary"},
        )
    )
    return result.text.strip() or description


async def _put_graph_node(
    object_store: ObjectStoreProtocol,
    config: ExtractLightRAGGraphConfig,
    node: GraphNode,
) -> str:
    key = join_object_key(config.graph_nodes_prefix, f"{_stable_object_id(node.id)}.json")
    await object_store.put(
        key,
        json.dumps(_graph_node_to_dict(node), ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        ),
    )
    return key


async def _put_graph_edge(
    object_store: ObjectStoreProtocol,
    config: ExtractLightRAGGraphConfig,
    edge: GraphEdge,
) -> str:
    key = join_object_key(config.graph_edges_prefix, f"{_stable_object_id(edge.id)}.json")
    await object_store.put(
        key,
        json.dumps(_graph_edge_to_dict(edge), ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        ),
    )
    return key


def _graph_node_to_dict(node: GraphNode) -> dict[str, Any]:
    return {"id": node.id, "labels": list(node.labels), "properties": dict(node.properties)}


def _graph_edge_to_dict(edge: GraphEdge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "type": edge.type,
        "properties": dict(edge.properties),
    }


def _edge_id(source_id: str, target_id: str) -> str:
    return f"{source_id}--RELATED--{target_id}"


def _stable_object_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordered_unique_descriptions(
    records: list[Mapping[str, Any]],
    existing_descriptions: list[str],
) -> list[str]:
    unique: dict[str, Mapping[str, Any]] = {}
    for record in records:
        description = str(record.get("description") or "").strip()
        if description and description not in unique:
            unique[description] = record
    sorted_records = sorted(
        unique.values(),
        key=lambda record: (int(record.get("timestamp") or 0), -len(str(record.get("description") or ""))),
    )
    return _unique_nonempty(existing_descriptions + [str(record["description"]) for record in sorted_records])


def _property_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.split(LIGHTRAG_FIELD_SEP) if part]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _unique_nonempty(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _most_common_nonempty(values: list[str], *, default: str) -> str:
    counts = Counter(value for value in values if value)
    if not counts:
        return default
    return counts.most_common(1)[0][0]


def _record_text_score(record: Mapping[str, Any], *, include_keywords: bool) -> int:
    score = len(str(record.get("description") or ""))
    if include_keywords:
        score += len(str(record.get("keywords") or ""))
    return score


def _clean_entity_type(value: Any) -> str | None:
    entity_type = _clean_text(value, remove_inner_quotes=True)
    if "," in entity_type:
        entity_type = next((part.strip() for part in entity_type.split(",") if part.strip()), "")
    if not entity_type or any(char in _INVALID_ENTITY_TYPE_CHARS for char in entity_type):
        return None
    return entity_type.replace(" ", "").lower()


def _clean_text(value: Any, *, remove_inner_quotes: bool = False) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value).strip())
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    text = text.strip().strip('"').strip("'").strip()
    if remove_inner_quotes:
        text = text.replace('"', "").replace("'", "")
    return text


def _strip_json_fence(content: str) -> str:
    text = content.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else text


def _split_by_markers(content: str, markers: list[str]) -> list[str]:
    return [
        part.strip()
        for part in re.split("|".join(re.escape(marker) for marker in markers), content)
        if part.strip()
    ]


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _chunk_file_path(chunk: ParsedChunk) -> str:
    return chunk.source.name or chunk.source.key or "unknown_source"


def _pack_user_assistant_messages(prompt: str, generated_content: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": generated_content},
    ]


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip()
        content = str(message.get("content") or "").strip()
        if content:
            parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def _lightrag_prompts() -> Mapping[str, Any]:
    if not hasattr(_lightrag_prompts, "_cache"):
        setattr(_lightrag_prompts, "_cache", _load_lightrag_prompts())
    return getattr(_lightrag_prompts, "_cache")


def _load_lightrag_prompts() -> Mapping[str, Any]:
    try:
        from lightrag.prompt import PROMPTS  # type: ignore

        return PROMPTS
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[5]
    prompt_path = repo_root / "LightRAG" / "lightrag" / "prompt.py"
    if not prompt_path.exists():
        raise RuntimeError(
            "LightRAG prompts are required for ExtractLightRAGGraph but could not be found"
        )
    spec = importlib.util.spec_from_file_location("_heta_lightrag_prompt", prompt_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load LightRAG prompt module from {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    if "yaml" not in sys.modules:
        yaml_stub = types.ModuleType("yaml")
        yaml_stub.YAMLError = ValueError
        yaml_stub.safe_load = lambda content: None
        sys.modules["yaml"] = yaml_stub
    spec.loader.exec_module(module)
    prompts = getattr(module, "PROMPTS", None)
    if not isinstance(prompts, dict):
        raise RuntimeError("LightRAG prompt module does not expose PROMPTS")
    return prompts


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_graph_store(component: object) -> GraphStoreProtocol:
    if not isinstance(component, GraphStoreProtocol):
        raise TypeError("stores.graph must satisfy GraphStoreProtocol")
    return component


def _require_language_model(component: object) -> LanguageModelProtocol:
    if not isinstance(component, LanguageModelProtocol):
        raise TypeError("models.language must satisfy LanguageModelProtocol")
    return component
