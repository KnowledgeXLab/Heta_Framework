"""Extract GraphRAG-style graph records from chunk artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Mapping

from heta_framework.common.models import ModelOptions, ModelRequest
from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.common.stores.graph import GraphEdge, GraphNode, GraphStoreProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb.chunking import ParsedChunk
from heta_framework.kb.cleanup import StepCleanupPlan, object_key_targets
from heta_framework.kb.graphing.prompts import (
    GRAPH_SUMMARY_PROMPT,
    GRAPH_RAG_ENTITY_CONTINUE_EXTRACTION_PROMPT,
    GRAPH_RAG_ENTITY_EXTRACTION_PROMPT,
    GRAPH_RAG_ENTITY_IF_LOOP_EXTRACTION_PROMPT,
)
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref


Message = dict[str, str]
NodeRecord = dict[str, Any]
EdgeRecord = dict[str, Any]
GRAPH_FIELD_SEP = "<SEP>"


@dataclass(frozen=True)
class ExtractGraphConfig:
    """Configuration for ExtractGraph."""

    entities_prefix: str = "entities"
    max_attempts: int = 3
    temperature: float = 0.0
    object_store: str | None = None
    graph_store: str | None = None
    language_model: str | None = None
    chunk_keys_artifact: str = "chunk_keys"
    entity_keys_artifact: str = "entity_keys"
    graph_nodes_prefix: str = "graph/nodes"
    graph_edges_prefix: str = "graph/edges"
    graph_node_keys_artifact: str = "graph_node_keys"
    graph_edge_keys_artifact: str = "graph_edge_keys"
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    summary_llm_max_tokens: int = 1200
    entity_extract_prompt: str = GRAPH_RAG_ENTITY_EXTRACTION_PROMPT
    context_base: Mapping[str, str] = field(
        default_factory=lambda: {
            "tuple_delimiter": "<|>",
            "record_delimiter": "##",
            "completion_delimiter": "<|COMPLETE|>",
            "entity_types": "organization,person,geo,event",
        }
    )
    continue_prompt: str = GRAPH_RAG_ENTITY_CONTINUE_EXTRACTION_PROMPT
    if_loop_prompt: str = GRAPH_RAG_ENTITY_IF_LOOP_EXTRACTION_PROMPT

    def __post_init__(self) -> None:
        validate_object_prefix(self.entities_prefix)
        validate_object_prefix(self.graph_nodes_prefix)
        validate_object_prefix(self.graph_edges_prefix)
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
        if self.chunk_keys_artifact.strip() == "":
            raise ValueError("chunk_keys_artifact must not be empty")
        if self.entity_keys_artifact.strip() == "":
            raise ValueError("entity_keys_artifact must not be empty")
        if self.graph_node_keys_artifact.strip() == "":
            raise ValueError("graph_node_keys_artifact must not be empty")
        if self.graph_edge_keys_artifact.strip() == "":
            raise ValueError("graph_edge_keys_artifact must not be empty")


@dataclass(frozen=True)
class ExtractGraphResult:
    """Artifacts produced by ExtractGraph."""

    node_keys: tuple[str, ...]
    edge_keys: tuple[str, ...]
    chunk_count: int
    entity_count: int
    relation_count: int
    failed_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class _ChunkGraphExtraction:
    nodes: Mapping[str, list[NodeRecord]]
    edges: Mapping[tuple[str, str], list[EdgeRecord]]
    failed: bool = False


class ExtractGraph:
    """Extract graph nodes and edges from parsed chunks into a GraphStore."""

    name = "extract_graph"

    def __init__(self, config: ExtractGraphConfig | None = None) -> None:
        self.config = config or ExtractGraphConfig()

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
                    "extract_graph_result",
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
        """Run graph extraction and store graph nodes and edges."""
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

        result = ExtractGraphResult(
            node_keys=node_keys,
            edge_keys=edge_keys,
            chunk_count=len(chunks),
            entity_count=len(graph_nodes),
            relation_count=len(graph_edges),
            failed_chunk_ids=tuple(failed_chunk_ids),
        )
        context.set_artifact("extract_graph_result", result)
        context.set_artifact(self.config.entity_keys_artifact, result.node_keys)
        context.set_artifact(self.config.graph_node_keys_artifact, result.node_keys)
        context.set_artifact(self.config.graph_edge_keys_artifact, result.edge_keys)

    async def _process_single_content(
        self,
        chunk: ParsedChunk,
        *,
        language_model: LanguageModelProtocol,
    ) -> _ChunkGraphExtraction:
        chunk_key = chunk.chunk_id
        hint_prompt = str(
            self.config.entity_extract_prompt.format(
                **self.config.context_base,
                input_text=str(chunk.text),
            )
        )
        final_result = await _extract_graph_from_processed_prompt(
            prompt=str(hint_prompt),
            history_messages=[],
            chunk=chunk,
            language_model=language_model,
            config=self.config,
        )
        if final_result is None:
            return _ChunkGraphExtraction(nodes={}, edges={}, failed=True)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for glean_index in range(self.config.entity_extract_max_gleaning):
            glean_result = await _extract_graph_from_processed_prompt(
                prompt=str(self.config.continue_prompt),
                history_messages=history,
                chunk=chunk,
                language_model=language_model,
                config=self.config,
            )
            if glean_result is None:
                break

            history += pack_user_ass_to_openai_messages(self.config.continue_prompt, glean_result)
            final_result += glean_result
            if glean_index == self.config.entity_extract_max_gleaning - 1:
                break

            if_loop_result = await _extract_graph_from_processed_prompt(
                prompt=str(self.config.if_loop_prompt),
                history_messages=history,
                chunk=chunk,
                language_model=language_model,
                config=self.config,
            )
            if if_loop_result is None:
                break
            if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
                break

        return _parse_graph_rag_records(final_result, chunk_key, self.config)


def pack_user_ass_to_openai_messages(prompt: str, generated_content: str) -> list[Message]:
    """Pack one user/assistant turn in OpenAI-compatible message shape."""
    return [
        {"role": "user", "content": str(prompt)},
        {"role": "assistant", "content": str(generated_content)},
    ]


async def _extract_graph_from_processed_prompt(
    prompt: str,
    history_messages: list[Message],
    chunk: ParsedChunk,
    *,
    language_model: LanguageModelProtocol,
    config: ExtractGraphConfig,
) -> str | None:
    last_error = ""
    for attempt in range(config.max_attempts):
        current_prompt = str(prompt)
        if last_error:
            current_prompt = (
                f"{str(prompt)}\n\n"
                f"Previous response was invalid or failed with this error:\n{last_error}\n\n"
                "Return the corrected answer using the requested format."
            )

        messages: list[Message] = []
        messages.extend(history_messages)
        messages.append({"role": "user", "content": str(current_prompt)})
        request_prompt = str(_messages_to_prompt(messages))

        try:
            result = await language_model.invoke(
                ModelRequest(
                    prompt=str(request_prompt),
                    options=ModelOptions(temperature=config.temperature),
                    trace_context={
                        "step": ExtractGraph.name,
                        "chunk_id": chunk.chunk_id,
                        "attempt": attempt + 1,
                    },
                )
            )
            text = result.text.strip()
            if text:
                return text
            raise ValueError("model returned empty text")
        except Exception as exc:
            last_error = str(exc) or exc.__class__.__name__
    return None


def _messages_to_prompt(messages: list[Message]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip() or "user"
        content = str(message.get("content", "")).strip()
        if content:
            parts.append(f"{role}:\n{content}")
    return str("\n\n".join(parts))


def _parse_graph_rag_records(
    content: str,
    chunk_key: str,
    config: ExtractGraphConfig,
) -> _ChunkGraphExtraction:
    records = split_string_by_multi_markers(
        content,
        [
            config.context_base["record_delimiter"],
            config.context_base["completion_delimiter"],
        ],
    )

    maybe_nodes: defaultdict[str, list[NodeRecord]] = defaultdict(list)
    maybe_edges: defaultdict[tuple[str, str], list[EdgeRecord]] = defaultdict(list)
    for record in records:
        match = re.search(r"\((.*)\)", record)
        if match is None:
            continue
        attributes = split_string_by_multi_markers(
            match.group(1),
            [config.context_base["tuple_delimiter"]],
        )

        entity = _handle_single_entity_extraction(attributes, chunk_key)
        if entity is not None:
            maybe_nodes[entity["entity_name"]].append(entity)
            continue

        relation = _handle_single_relationship_extraction(attributes, chunk_key)
        if relation is not None:
            maybe_edges[(relation["src_id"], relation["tgt_id"])].append(relation)

    return _ChunkGraphExtraction(nodes=dict(maybe_nodes), edges=dict(maybe_edges))


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple literal markers."""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [result.strip() for result in results if result.strip()]


def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
) -> NodeRecord | None:
    if len(record_attributes) < 4 or _record_kind(record_attributes[0]) != "entity":
        return None

    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name:
        return None
    return {
        "entity_name": entity_name,
        "entity_type": clean_str(record_attributes[2].upper()) or "ENTITY",
        "description": clean_str(record_attributes[3]),
        "source_id": chunk_key,
    }


def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
) -> EdgeRecord | None:
    if len(record_attributes) < 5 or _record_kind(record_attributes[0]) != "relationship":
        return None

    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    if not source or not target or source == target:
        return None

    weight_text = record_attributes[-1]
    return {
        "src_id": source,
        "tgt_id": target,
        "weight": float(weight_text) if is_float_regex(weight_text) else 1.0,
        "description": clean_str(record_attributes[3]),
        "source_id": chunk_key,
    }


def _record_kind(value: str) -> str:
    return clean_str(value).strip('"').strip("'").lower()


def clean_str(value: Any) -> str:
    """Clean a string by removing HTML escapes and control characters."""
    if not isinstance(value, str):
        return ""
    result = html.unescape(value.strip())
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def is_float_regex(value: str) -> bool:
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value.strip()))


async def _merge_node_then_upsert(
    entity_name: str,
    nodes_data: list[NodeRecord],
    graph_store: GraphStoreProtocol,
    *,
    language_model: LanguageModelProtocol,
    config: ExtractGraphConfig,
) -> GraphNode:
    existing_types: list[str] = []
    existing_source_ids: list[str] = []
    existing_descriptions: list[str] = []

    existing_node = await graph_store.get_node(entity_name)
    if existing_node is not None:
        properties = existing_node.properties
        existing_types.extend(_property_values(properties.get("entity_type")))
        existing_source_ids.extend(_property_values(properties.get("source_ids")))
        existing_source_ids.extend(_property_values(properties.get("source_id")))
        existing_descriptions.extend(_property_values(properties.get("description")))

    entity_type = _most_common_nonempty(
        [record.get("entity_type", "") for record in nodes_data] + existing_types,
        default="ENTITY",
    )
    description = GRAPH_FIELD_SEP.join(
        _unique_nonempty(
            [record.get("description", "") for record in nodes_data] + existing_descriptions
        )
    )
    description = await _handle_entity_relation_summary(
        entity_name,
        description,
        language_model=language_model,
        config=config,
    )
    source_ids = _unique_nonempty(
        [record.get("source_id", "") for record in nodes_data] + existing_source_ids
    )

    node = GraphNode(
        id=entity_name,
        labels=("Entity", entity_type),
        properties={
            "name": entity_name,
            "entity_type": entity_type,
            "description": description,
            "source_ids": source_ids,
            "source_id": GRAPH_FIELD_SEP.join(source_ids),
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
    config: ExtractGraphConfig,
) -> GraphEdge | None:
    if source_id == target_id or not edges_data:
        return None

    edge_id = _edge_id(source_id, target_id)
    existing_weights: list[float] = []
    existing_source_ids: list[str] = []
    existing_descriptions: list[str] = []
    existing_orders: list[int] = []

    existing_edge = await graph_store.get_edge(edge_id)
    if existing_edge is not None:
        properties = existing_edge.properties
        weight = properties.get("weight")
        if isinstance(weight, int | float):
            existing_weights.append(float(weight))
        existing_source_ids.extend(_property_values(properties.get("source_ids")))
        existing_source_ids.extend(_property_values(properties.get("source_id")))
        existing_descriptions.extend(_property_values(properties.get("description")))
        order = properties.get("order")
        if isinstance(order, int):
            existing_orders.append(order)

    source_ids = _unique_nonempty(
        [record.get("source_id", "") for record in edges_data] + existing_source_ids
    )
    description = GRAPH_FIELD_SEP.join(
        _unique_nonempty(
            [record.get("description", "") for record in edges_data] + existing_descriptions
        )
    )
    description = await _handle_entity_relation_summary(
        f"{source_id} -> {target_id}",
        description,
        language_model=language_model,
        config=config,
    )
    weights = [
        float(record.get("weight", 1.0))
        for record in edges_data
        if is_float_regex(str(record.get("weight", "")))
    ]
    weight = sum(weights + existing_weights) if weights or existing_weights else 1.0
    order = min([_record_order(record) for record in edges_data] + existing_orders)

    for node_id in (source_id, target_id):
        if await graph_store.get_node(node_id) is None:
            await graph_store.upsert_nodes(
                [
                    GraphNode(
                        id=node_id,
                        labels=("Entity", "UNKNOWN"),
                        properties={
                            "name": node_id,
                            "entity_type": "UNKNOWN",
                            "description": description,
                            "source_ids": source_ids,
                            "source_id": GRAPH_FIELD_SEP.join(source_ids),
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
            "weight": weight,
            "description": description,
            "source_ids": source_ids,
            "source_id": GRAPH_FIELD_SEP.join(source_ids),
            "order": order,
        },
    )
    await graph_store.upsert_edges([edge])
    return edge


def _edge_id(source_id: str, target_id: str) -> str:
    return f"{source_id}--RELATED--{target_id}"


async def _put_graph_node(
    object_store: ObjectStoreProtocol,
    config: ExtractGraphConfig,
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
    config: ExtractGraphConfig,
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
    return {
        "id": node.id,
        "labels": list(node.labels),
        "properties": dict(node.properties),
    }


def _graph_edge_to_dict(edge: GraphEdge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "type": edge.type,
        "properties": dict(edge.properties),
    }


def _stable_object_id(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest[:32]


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    *,
    language_model: LanguageModelProtocol,
    config: ExtractGraphConfig,
) -> str:
    tokens = _encode_summary_tokens(description)
    if len(tokens) <= config.entity_summary_to_max_tokens:
        return description

    use_description = _decode_summary_tokens(tokens[: config.summary_llm_max_tokens])
    prompt = str(GRAPH_SUMMARY_PROMPT.format(
        entity_name=str(entity_or_relation_name),
        description_list="\n".join(
            f"- {item}"
            for item in split_string_by_multi_markers(use_description, [GRAPH_FIELD_SEP])
        ),
    ))
    result = await language_model.invoke(
        ModelRequest(
            prompt=str(prompt),
            options=ModelOptions(
                temperature=config.temperature,
                max_output_tokens=config.entity_summary_to_max_tokens,
            ),
            trace_context={
                "step": ExtractGraph.name,
                "summary_target": str(entity_or_relation_name),
            },
        )
    )
    summary = result.text.strip()
    return summary or description


def _encode_summary_tokens(text: str) -> list[str]:
    return text.split()


def _decode_summary_tokens(tokens: list[str]) -> str:
    return " ".join(tokens)


def _property_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return split_string_by_multi_markers(value, [GRAPH_FIELD_SEP])
    if isinstance(value, Iterable):
        return [clean_str(item) for item in value if clean_str(item)]
    return [clean_str(value)]


def _most_common_nonempty(values: Iterable[Any], *, default: str) -> str:
    cleaned = [clean_str(value) for value in values if clean_str(value)]
    if not cleaned:
        return default
    return Counter(cleaned).most_common(1)[0][0]


def _record_order(record: EdgeRecord) -> int:
    value = record.get("order", 1)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return 1


def _unique_nonempty(values: Iterable[Any]) -> list[str]:
    cleaned = [clean_str(value) for value in values]
    return sorted({value for value in cleaned if value})


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
