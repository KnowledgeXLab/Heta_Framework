"""Extract HiRAG-style hierarchical graph records from chunk artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import html
import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from heta_framework.common.models import EmbeddingRequest, ModelOptions, ModelRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol, LanguageModelProtocol
from heta_framework.common.stores.graph import GraphEdge, GraphNode, GraphStoreProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb.chunking import ParsedChunk
from heta_framework.kb.cleanup import StepCleanupPlan, object_key_targets
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref


Message = dict[str, str]
NodeRecord = dict[str, Any]
EdgeRecord = dict[str, Any]
TraceRecord = dict[str, Any]


def _load_original_hirag_prompts() -> tuple[str, dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[5]
    prompt_path = repo_root / "HiRAG" / "hirag" / "prompt.py"
    if prompt_path.exists():
        spec = importlib.util.spec_from_file_location("_heta_original_hirag_prompt", prompt_path)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return str(module.GRAPH_FIELD_SEP), dict(module.PROMPTS)
    return "<SEP>", {
        "DEFAULT_TUPLE_DELIMITER": "<|>",
        "DEFAULT_RECORD_DELIMITER": "##",
        "DEFAULT_COMPLETION_DELIMITER": "<|COMPLETE|>",
        "META_ENTITY_TYPES": ["organization", "person", "location", "event"],
        "hi_entity_extraction": "{input_text}",
        "hi_relation_extraction": "{entities}\n{input_text}",
        "entiti_continue_extraction": "MANY entities were missed in the last extraction. Add them below using the same format.",
        "entiti_if_loop_extraction": "It appears some entities may have still been missed. Answer yes or no.",
        "summary_clusters": "{entity_description_list}",
        "summarize_entity_descriptions": "{entity_name}\n{description_list}",
    }


GRAPH_FIELD_SEP, HIRAG_PROMPTS = _load_original_hirag_prompts()


@dataclass(frozen=True)
class ExtractHiRAGGraphConfig:
    """Configuration for HiRAG hierarchical graph extraction."""

    temperature: float = 0.0
    max_attempts: int = 3
    object_store: str | None = None
    graph_store: str | None = None
    language_model: str | None = None
    embedding_model: str | None = None
    chunk_keys_artifact: str = "chunk_keys"
    result_artifact: str = "extract_hi_rag_graph_result"
    chunks_artifact: str = "hi_rag_chunks"
    base_entities_artifact: str = "hi_rag_base_entities"
    base_relations_artifact: str = "hi_rag_base_relations"
    hierarchical_layers_artifact: str = "hi_rag_hierarchical_layers"
    summary_entities_artifact: str = "hi_rag_summary_entities"
    summary_relations_artifact: str = "hi_rag_summary_relations"
    graph_node_keys_artifact: str = "hi_rag_graph_node_keys"
    graph_edge_keys_artifact: str = "hi_rag_graph_edge_keys"
    extraction_trace_artifact: str = "hi_rag_extraction_trace"
    graph_nodes_prefix: str = "hi_rag/graph/nodes"
    graph_edges_prefix: str = "hi_rag/graph/edges"
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    summary_llm_max_tokens: int = 1200
    hierarchical_layers: int = 50
    hierarchical_max_length_in_cluster: int = 60000
    hierarchical_reduction_dimension: int = 2
    hierarchical_cluster_threshold: float = 0.1
    hierarchical_sparsity_threshold: float = 0.99
    hierarchical_sparsity_change_rate: float = 0.03
    hierarchical_tokenizer_encoding: str = "cl100k_base"
    clustering_backend: Literal["auto", "deterministic"] = "auto"
    random_seed: int = 224
    prompts: Mapping[str, Any] = field(default_factory=lambda: dict(HIRAG_PROMPTS))

    def __post_init__(self) -> None:
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
        if self.hierarchical_layers < 0:
            raise ValueError("hierarchical_layers must not be negative")
        for name in (
            self.chunk_keys_artifact,
            self.result_artifact,
            self.chunks_artifact,
            self.base_entities_artifact,
            self.base_relations_artifact,
            self.hierarchical_layers_artifact,
            self.summary_entities_artifact,
            self.summary_relations_artifact,
            self.graph_node_keys_artifact,
            self.graph_edge_keys_artifact,
            self.extraction_trace_artifact,
        ):
            if name.strip() == "":
                raise ValueError("artifact names must not be empty")


@dataclass(frozen=True)
class ExtractHiRAGGraphResult:
    """Artifacts produced by ExtractHiRAGGraph."""

    chunk_count: int
    base_entity_count: int
    base_relation_count: int
    summary_entity_count: int
    summary_relation_count: int
    merged_entity_count: int
    merged_relation_count: int
    hierarchical_layer_count: int
    failed_chunk_ids: tuple[str, ...]
    node_keys: tuple[str, ...]
    edge_keys: tuple[str, ...]


@dataclass(frozen=True)
class _ChunkExtraction:
    nodes: Mapping[str, list[NodeRecord]]
    edges: Mapping[tuple[str, str], list[EdgeRecord]]
    trace: TraceRecord
    failed: bool = False


class ExtractHiRAGGraph:
    """Run HiRAG two-stage extraction and hierarchical attribute-entity construction."""

    name = "extract_hirag_graph"

    def __init__(self, config: ExtractHiRAGGraphConfig | None = None) -> None:
        self.config = config or ExtractHiRAGGraphConfig()

    @property
    def requirements(self) -> StepRequirements:
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("objects", self.config.object_store),
                    store_ref("graph", self.config.graph_store),
                    model_ref("language", self.config.language_model),
                    model_ref("embedding", self.config.embedding_model),
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
                    self.config.chunks_artifact,
                    self.config.base_entities_artifact,
                    self.config.base_relations_artifact,
                    self.config.hierarchical_layers_artifact,
                    self.config.summary_entities_artifact,
                    self.config.summary_relations_artifact,
                    self.config.graph_node_keys_artifact,
                    self.config.graph_edge_keys_artifact,
                    self.config.extraction_trace_artifact,
                }
            )
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
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
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        graph_store = _require_graph_store(
            context.get_component(store_ref("graph", self.config.graph_store).key)
        )
        language_model = _require_language_model(
            context.get_component(model_ref("language", self.config.language_model).key)
        )
        embedding_model = _require_embedding_model(
            context.get_component(model_ref("embedding", self.config.embedding_model).key)
        )

        chunk_keys = tuple(context.get_artifact(self.config.chunk_keys_artifact))
        chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]

        entity_results = await asyncio.gather(
            *(
                self._process_single_content_entity(chunk, language_model=language_model)
                for chunk in chunks
            )
        )

        context_entities = {
            chunk.chunk_id: list(result.nodes.keys())
            for chunk, result in zip(chunks, entity_results, strict=True)
            if not result.failed
        }
        all_base_entities = _first_records_by_name(
            record
            for result in entity_results
            for records in result.nodes.values()
            for record in records
        )
        await _attach_embeddings(
            all_base_entities,
            embedding_model,
            purpose="hi_rag_base_entity_clustering",
        )

        relation_results = await asyncio.gather(
            *(
                self._process_single_content_relation(
                    chunk,
                    entities=context_entities.get(chunk.chunk_id, ()),
                    language_model=language_model,
                )
                for chunk in chunks
            )
        )

        base_nodes: defaultdict[str, list[NodeRecord]] = defaultdict(list)
        base_edges: defaultdict[tuple[str, str], list[EdgeRecord]] = defaultdict(list)
        failed_chunk_ids: list[str] = []
        trace: list[TraceRecord] = []

        for chunk, entity_result, relation_result in zip(
            chunks, entity_results, relation_results, strict=True
        ):
            if entity_result.failed or relation_result.failed:
                failed_chunk_ids.append(chunk.chunk_id)
            trace.extend([entity_result.trace, relation_result.trace])
            for name, records in entity_result.nodes.items():
                base_nodes[name].extend(records)
            for endpoints, records in relation_result.edges.items():
                base_edges[tuple(sorted(endpoints))].extend(records)

        hierarchical = await _perform_hierarchical_clustering(
            list(all_base_entities.values()),
            language_model=language_model,
            embedding_model=embedding_model,
            config=self.config,
        )

        maybe_nodes: defaultdict[str, list[NodeRecord]] = defaultdict(list)
        maybe_edges: defaultdict[tuple[str, str], list[EdgeRecord]] = defaultdict(list)
        for name, records in base_nodes.items():
            maybe_nodes[name].extend(records)
        for endpoints, records in base_edges.items():
            maybe_edges[endpoints].extend(records)
        for record in hierarchical["summary_entities"]:
            maybe_nodes[str(record["entity_name"])].append(record)
        for record in hierarchical["summary_relations"]:
            maybe_edges[tuple(sorted((str(record["src_id"]), str(record["tgt_id"]))))].append(record)

        graph_nodes = await asyncio.gather(
            *(
                _merge_node_then_upsert(
                    entity_name,
                    records,
                    graph_store,
                    language_model=language_model,
                    config=self.config,
                )
                for entity_name, records in maybe_nodes.items()
            )
        )
        maybe_graph_edges = await asyncio.gather(
            *(
                _merge_edge_then_upsert(
                    endpoints[0],
                    endpoints[1],
                    records,
                    graph_store,
                    language_model=language_model,
                    config=self.config,
                )
                for endpoints, records in maybe_edges.items()
            )
        )
        graph_edges = [edge for edge in maybe_graph_edges if edge is not None]

        node_keys = tuple(
            await asyncio.gather(
                *(_put_graph_node(object_store, self.config, node) for node in graph_nodes)
            )
        )
        edge_keys = tuple(
            await asyncio.gather(
                *(_put_graph_edge(object_store, self.config, edge) for edge in graph_edges)
            )
        )

        result = ExtractHiRAGGraphResult(
            chunk_count=len(chunks),
            base_entity_count=sum(len(records) for records in base_nodes.values()),
            base_relation_count=sum(len(records) for records in base_edges.values()),
            summary_entity_count=len(hierarchical["summary_entities"]),
            summary_relation_count=len(hierarchical["summary_relations"]),
            merged_entity_count=len(graph_nodes),
            merged_relation_count=len(graph_edges),
            hierarchical_layer_count=len(hierarchical["layers"]),
            failed_chunk_ids=tuple(dict.fromkeys(failed_chunk_ids)),
            node_keys=node_keys,
            edge_keys=edge_keys,
        )

        context.set_artifact(self.config.result_artifact, result)
        context.set_artifact(self.config.chunks_artifact, [_chunk_trace(chunk) for chunk in chunks])
        context.set_artifact(
            self.config.base_entities_artifact,
            [record for records in base_nodes.values() for record in records],
        )
        context.set_artifact(
            self.config.base_relations_artifact,
            [record for records in base_edges.values() for record in records],
        )
        context.set_artifact(self.config.hierarchical_layers_artifact, hierarchical["layers"])
        context.set_artifact(self.config.summary_entities_artifact, hierarchical["summary_entities"])
        context.set_artifact(self.config.summary_relations_artifact, hierarchical["summary_relations"])
        context.set_artifact(self.config.graph_node_keys_artifact, node_keys)
        context.set_artifact(self.config.graph_edge_keys_artifact, edge_keys)
        context.set_artifact(self.config.extraction_trace_artifact, trace + hierarchical["trace"])

    async def _process_single_content_entity(
        self,
        chunk: ParsedChunk,
        *,
        language_model: LanguageModelProtocol,
    ) -> _ChunkExtraction:
        prompts = self.config.prompts
        context_base = _context_base(prompts, entity_types=",".join(prompts["META_ENTITY_TYPES"]))
        prompt = str(prompts["hi_entity_extraction"].format(**context_base, input_text=chunk.text))
        final_result, glean_count, failed = await _run_hirag_gleaning(
            prompt,
            chunk=chunk,
            stage="hi_entity_extraction",
            language_model=language_model,
            config=self.config,
        )
        if failed:
            return _ChunkExtraction(
                nodes={},
                edges={},
                failed=True,
                trace=_flow_trace(chunk, "hi_entity_extraction", prompt, final_result, [], [], glean_count, True),
            )
        parsed = _parse_hirag_records(final_result, chunk.chunk_id, self.config, layer=0)
        return _ChunkExtraction(
            nodes=parsed.nodes,
            edges=parsed.edges,
            trace=_flow_trace(
                chunk,
                "hi_entity_extraction",
                prompt,
                final_result,
                [record for records in parsed.nodes.values() for record in records],
                [record for records in parsed.edges.values() for record in records],
                glean_count,
                False,
            ),
        )

    async def _process_single_content_relation(
        self,
        chunk: ParsedChunk,
        *,
        entities: Iterable[str],
        language_model: LanguageModelProtocol,
    ) -> _ChunkExtraction:
        prompts = self.config.prompts
        context_base = _context_base(prompts, entities=",".join(entities))
        prompt = str(prompts["hi_relation_extraction"].format(**context_base, input_text=chunk.text))
        final_result, glean_count, failed = await _run_hirag_gleaning(
            prompt,
            chunk=chunk,
            stage="hi_relation_extraction",
            language_model=language_model,
            config=self.config,
        )
        if failed:
            return _ChunkExtraction(
                nodes={},
                edges={},
                failed=True,
                trace=_flow_trace(chunk, "hi_relation_extraction", prompt, final_result, [], [], glean_count, True),
            )
        parsed = _parse_hirag_records(final_result, chunk.chunk_id, self.config, layer=0)
        return _ChunkExtraction(
            nodes=parsed.nodes,
            edges=parsed.edges,
            trace={
                **_flow_trace(
                    chunk,
                    "hi_relation_extraction",
                    prompt,
                    final_result,
                    [record for records in parsed.nodes.values() for record in records],
                    [record for records in parsed.edges.values() for record in records],
                    glean_count,
                    False,
                ),
                "entity_list": list(entities),
            },
        )


async def _run_hirag_gleaning(
    prompt: str,
    *,
    chunk: ParsedChunk,
    stage: str,
    language_model: LanguageModelProtocol,
    config: ExtractHiRAGGraphConfig,
) -> tuple[str, int, bool]:
    final_result = await _invoke_text(
        prompt,
        (),
        chunk=chunk,
        stage=stage,
        language_model=language_model,
        config=config,
    )
    if final_result is None:
        return "", 0, True

    history = pack_user_ass_to_openai_messages(prompt, final_result)
    if_loop_result = await _invoke_text(
        str(config.prompts["entiti_if_loop_extraction"]),
        history,
        chunk=chunk,
        stage=f"{stage}:if_loop",
        language_model=language_model,
        config=config,
    )
    if if_loop_result is None:
        return final_result, 0, False
    if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
        return final_result, 0, False

    glean_count = 0
    for glean_index in range(config.entity_extract_max_gleaning):
        continue_prompt = str(config.prompts["entiti_continue_extraction"])
        glean_result = await _invoke_text(
            continue_prompt,
            history,
            chunk=chunk,
            stage=f"{stage}:glean",
            language_model=language_model,
            config=config,
        )
        if glean_result is None:
            break
        glean_count += 1
        history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
        final_result += glean_result
        if glean_index == config.entity_extract_max_gleaning - 1:
            break
        if_loop_result = await _invoke_text(
            str(config.prompts["entiti_if_loop_extraction"]),
            history,
            chunk=chunk,
            stage=f"{stage}:if_loop",
            language_model=language_model,
            config=config,
        )
        if if_loop_result is None:
            break
        if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
            break
    return final_result, glean_count, False


async def _invoke_text(
    prompt: str,
    history_messages: Iterable[Message],
    *,
    chunk: ParsedChunk,
    stage: str,
    language_model: LanguageModelProtocol,
    config: ExtractHiRAGGraphConfig,
) -> str | None:
    last_error = ""
    for attempt in range(config.max_attempts):
        current_prompt = prompt
        if last_error:
            current_prompt = (
                f"{prompt}\n\nPrevious response was invalid or failed with this error:\n"
                f"{last_error}\n\nReturn the corrected answer using the requested format."
            )
        messages = [*history_messages, {"role": "user", "content": current_prompt}]
        try:
            result = await language_model.invoke(
                ModelRequest(
                    prompt=_messages_to_prompt(messages),
                    options=ModelOptions(temperature=config.temperature),
                    trace_context={
                        "step": ExtractHiRAGGraph.name,
                        "stage": stage,
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


def pack_user_ass_to_openai_messages(prompt: str, generated_content: str) -> list[Message]:
    return [
        {"role": "user", "content": str(prompt)},
        {"role": "assistant", "content": str(generated_content)},
    ]


def _messages_to_prompt(messages: Iterable[Message]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip() or "user"
        content = str(message.get("content", "")).strip()
        if content:
            parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def _parse_hirag_records(
    content: str,
    chunk_key: str,
    config: ExtractHiRAGGraphConfig,
    *,
    layer: int,
    cluster_id: str | None = None,
    is_summary: bool = False,
    parent_entity_ids: Iterable[str] = (),
) -> _ChunkExtraction:
    prompts = config.prompts
    records = split_string_by_multi_markers(
        _strip_fences(content),
        [prompts["DEFAULT_RECORD_DELIMITER"], prompts["DEFAULT_COMPLETION_DELIMITER"]],
    )
    maybe_nodes: defaultdict[str, list[NodeRecord]] = defaultdict(list)
    maybe_edges: defaultdict[tuple[str, str], list[EdgeRecord]] = defaultdict(list)
    skipped: list[str] = []
    for record in records:
        match = re.search(r"\((.*)\)", record, flags=re.DOTALL)
        if match is None:
            skipped.append(record)
            continue
        attributes = split_string_by_multi_markers(
            match.group(1),
            [prompts["DEFAULT_TUPLE_DELIMITER"]],
        )
        entity = _handle_single_entity_extraction(
            attributes,
            chunk_key,
            layer=layer,
            cluster_id=cluster_id,
            is_summary=is_summary,
            parent_entity_ids=parent_entity_ids,
        )
        if entity is not None:
            maybe_nodes[entity["entity_name"]].append(entity)
            continue
        relation = _handle_single_relationship_extraction(
            attributes,
            chunk_key,
            layer=layer,
            cluster_id=cluster_id,
            is_summary=is_summary,
            parent_entity_ids=parent_entity_ids,
        )
        if relation is not None:
            maybe_edges[(relation["src_id"], relation["tgt_id"])].append(relation)
            continue
        skipped.append(record)
    return _ChunkExtraction(
        nodes=dict(maybe_nodes),
        edges=dict(maybe_edges),
        trace={"skipped_records": skipped},
    )


def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    *,
    layer: int = 0,
    cluster_id: str | None = None,
    is_summary: bool = False,
    parent_entity_ids: Iterable[str] = (),
) -> NodeRecord | None:
    if len(record_attributes) < 4 or _record_kind(record_attributes[0]) != "entity":
        return None
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    description = clean_str(record_attributes[3])
    parent_ids = tuple(dict.fromkeys(str(parent) for parent in parent_entity_ids if str(parent)))
    return {
        "entity_name": entity_name,
        "entity_type": entity_type,
        "raw_entity_type": clean_str(record_attributes[2]),
        "description": description,
        "source_id": chunk_key,
        "source_ids": [chunk_key] if chunk_key else [],
        "layer": layer,
        "cluster_id": cluster_id,
        "is_summary": is_summary,
        "parent_entity_ids": list(parent_ids),
    }


def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    *,
    layer: int = 0,
    cluster_id: str | None = None,
    is_summary: bool = False,
    parent_entity_ids: Iterable[str] = (),
) -> EdgeRecord | None:
    if len(record_attributes) < 5 or _record_kind(record_attributes[0]) != "relationship":
        return None
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    if not source or not target or source == target:
        return None
    weight = float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    parent_ids = tuple(dict.fromkeys(str(parent) for parent in parent_entity_ids if str(parent)))
    return {
        "src_id": source,
        "tgt_id": target,
        "weight": weight,
        "description": clean_str(record_attributes[3]),
        "source_id": chunk_key,
        "source_ids": [chunk_key] if chunk_key else [],
        "order": _record_order({"order": record_attributes[4] if len(record_attributes) > 5 else 1}),
        "layer": layer,
        "cluster_id": cluster_id,
        "is_summary": is_summary,
        "parent_entity_ids": list(parent_ids),
    }


async def _perform_hierarchical_clustering(
    entities: list[NodeRecord],
    *,
    language_model: LanguageModelProtocol,
    embedding_model: EmbeddingModelProtocol,
    config: ExtractHiRAGGraphConfig,
) -> dict[str, list[dict[str, Any]]]:
    if not entities or config.hierarchical_layers == 0:
        return {"layers": [], "summary_entities": [], "summary_relations": [], "trace": []}

    random.seed(config.random_seed)
    current_nodes = [dict(entity) for entity in entities]
    layers: list[dict[str, Any]] = []
    summary_entities: list[NodeRecord] = []
    summary_relations: list[EdgeRecord] = []
    trace: list[TraceRecord] = []
    previous_sparsity = 0.01

    for layer_index in range(config.hierarchical_layers):
        if len(current_nodes) <= 2:
            trace.append(
                {
                    "stage": "hierarchical_clustering",
                    "layer": layer_index,
                    "entity_count": len(current_nodes),
                    "stop_reason": "entity_count_le_2",
                }
            )
            break

        cluster_assignments, backend, cluster_trace = _cluster_assignments(current_nodes, config)
        flat_labels = [label for labels in cluster_assignments for label in labels]
        cluster_sizes = Counter(flat_labels)
        if len(current_nodes) > 1:
            sparsity = 1 - sum(size * (size - 1) for size in cluster_sizes.values()) / (
                len(current_nodes) * (len(current_nodes) - 1)
            )
        else:
            sparsity = 1.0
        change_rate = abs(sparsity - previous_sparsity) / (previous_sparsity + 1e-8)
        previous_sparsity = sparsity
        layer_trace: TraceRecord = {
            "stage": "hierarchical_clustering",
            "layer": layer_index,
            "entity_count": len(current_nodes),
            "embedding_shape": [len(current_nodes), len(current_nodes[0].get("embedding", []))],
            "backend": backend,
            "labels": cluster_assignments,
            "cluster_sizes": dict(cluster_sizes),
            "cluster_sparsity": sparsity,
            "cluster_sparsity_change_rate": change_rate,
            **cluster_trace,
        }
        if sparsity >= config.hierarchical_sparsity_threshold:
            layer_trace["stop_reason"] = "cluster_sparsity_threshold"
            trace.append(layer_trace)
            break
        if change_rate <= config.hierarchical_sparsity_change_rate:
            layer_trace["stop_reason"] = "cluster_sparsity_change_rate"
            trace.append(layer_trace)
            break

        next_nodes: list[NodeRecord] = []
        clusters: list[dict[str, Any]] = []
        unique_labels = sorted(cluster_sizes)
        for label in unique_labels:
            cluster_nodes = [
                node
                for node, labels in zip(current_nodes, cluster_assignments, strict=True)
                if label in labels
            ]
            clusters.append(
                {
                    "cluster_id": str(label),
                    "entity_ids": [str(node["entity_name"]) for node in cluster_nodes],
                    "size": len(cluster_nodes),
                }
            )
            if len(cluster_nodes) == 1:
                next_nodes.extend(cluster_nodes)
                continue
            sampled_nodes, sampling_trace = _fit_cluster_token_budget(cluster_nodes, config)
            parsed = await _summarize_cluster(
                sampled_nodes,
                layer=layer_index + 1,
                cluster_id=str(label),
                language_model=language_model,
                embedding_model=embedding_model,
                config=config,
            )
            layer_trace.setdefault("summary_clusters", []).append(
                {
                    "cluster_id": str(label),
                    "input_entity_ids": [str(node["entity_name"]) for node in sampled_nodes],
                    **sampling_trace,
                    "summary_entity_count": sum(len(records) for records in parsed.nodes.values()),
                    "summary_relation_count": sum(len(records) for records in parsed.edges.values()),
                }
            )
            for records in parsed.nodes.values():
                next_nodes.extend(records)
                summary_entities.extend(records)
            for records in parsed.edges.values():
                summary_relations.extend(records)
        layers.append({"layer": layer_index + 1, "clusters": clusters})
        trace.append(layer_trace)
        current_nodes = _dedupe_nodes(node for node in next_nodes if "entity_name" in node)
        await _attach_embeddings(
            current_nodes,
            embedding_model,
            purpose=f"hi_rag_summary_entity_layer_{layer_index + 1}",
        )
    return {
        "layers": layers,
        "summary_entities": summary_entities,
        "summary_relations": summary_relations,
        "trace": trace,
    }


def _cluster_assignments(
    nodes: list[NodeRecord],
    config: ExtractHiRAGGraphConfig,
) -> tuple[list[list[int]], str, dict[str, Any]]:
    if config.clustering_backend == "deterministic":
        return _deterministic_cluster_assignments(nodes), "deterministic", {}
    try:
        import numpy as np
        import umap
    except Exception as exc:
        return _deterministic_cluster_assignments(nodes), "deterministic_fallback", {"fallback_reason": str(exc)}

    embeddings = np.array([node.get("embedding", []) for node in nodes], dtype=float)
    if embeddings.ndim != 2 or embeddings.shape[0] <= 2 or embeddings.shape[1] == 0:
        return _deterministic_cluster_assignments(nodes), "deterministic_fallback", {"fallback_reason": "invalid_embedding_shape"}

    try:
        n_neighbors = int((len(nodes) - 1) ** 0.5)
        if n_neighbors <= 1:
            n_neighbors = 2
        n_components = max(1, min(config.hierarchical_reduction_dimension, len(nodes) - 2))
        reduced = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric="cosine",
            random_state=config.random_seed,
        ).fit_transform(embeddings)
        n_clusters, model, bics = _optimal_gmm_cluster_count(
            reduced,
            max_clusters=50,
            random_state=config.random_seed,
        )
        if model is None:
            return _deterministic_cluster_assignments(nodes), "deterministic_fallback", {
                "fallback_reason": "no_gmm_candidates"
            }
        probabilities = model.predict_proba(reduced)
        assignments = [
            [int(label) for label in np.where(probability > config.hierarchical_cluster_threshold)[0]]
            for probability in probabilities
        ]
        assignments = [
            labels if labels else [int(np.argmax(probability))]
            for labels, probability in zip(assignments, probabilities, strict=True)
        ]
        return assignments, "hirag_native_umap_gmm", {
            "n_clusters": n_clusters,
            "umap_n_neighbors": n_neighbors,
            "gmm_bics": bics,
            "cluster_threshold": config.hierarchical_cluster_threshold,
        }
    except Exception as exc:
        return _deterministic_cluster_assignments(nodes), "deterministic_fallback", {"fallback_reason": str(exc)}


def _optimal_gmm_cluster_count(
    embeddings: Any,
    *,
    max_clusters: int,
    random_state: int,
    rel_tol: float = 1e-3,
) -> tuple[int, Any | None, list[float]]:
    import numpy as np
    from sklearn.mixture import GaussianMixture

    max_count = min(len(embeddings), max_clusters)
    candidates = np.arange(1, max_count)
    bics: list[float] = []
    best_bic = float("inf")
    best_model: Any | None = None
    best_count = 1
    previous_bic = float("inf")
    for count in candidates:
        model = GaussianMixture(
            n_components=int(count),
            random_state=random_state,
            n_init=5,
            init_params="k-means++",
        )
        model.fit(embeddings)
        bic = float(model.bic(embeddings))
        bics.append(bic)
        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_count = int(count)
        if previous_bic != float("inf") and abs(previous_bic - bic) / abs(previous_bic) < rel_tol:
            break
        previous_bic = bic
    if not bics:
        return 1, None, []
    return best_count, best_model, bics


def _deterministic_cluster_assignments(nodes: list[NodeRecord]) -> list[list[int]]:
    if len(nodes) <= 2:
        return [[0] for _ in nodes]
    halfway = max(2, (len(nodes) + 1) // 2)
    return [[0] if index < halfway else [1] for index, _ in enumerate(nodes)]


async def _summarize_cluster(
    cluster_nodes: list[NodeRecord],
    *,
    layer: int,
    cluster_id: str,
    language_model: LanguageModelProtocol,
    embedding_model: EmbeddingModelProtocol,
    config: ExtractHiRAGGraphConfig,
) -> _ChunkExtraction:
    prompts = config.prompts
    context_base = _context_base(
        prompts,
        meta_attribute_list=prompts["META_ENTITY_TYPES"],
        entity_description_list=",".join(
            f"({node['entity_name']}, {node['description']})" for node in cluster_nodes
        ),
    )
    prompt = str(prompts["summary_clusters"].format(**context_base))
    result = await _invoke_summary(
        prompt,
        language_model=language_model,
        config=config,
        layer=layer,
        cluster_id=cluster_id,
    )
    parent_ids = [str(node["entity_name"]) for node in cluster_nodes]
    parsed = _parse_hirag_records(
        result,
        "",
        config,
        layer=layer,
        cluster_id=cluster_id,
        is_summary=True,
        parent_entity_ids=parent_ids,
    )
    summary_nodes = [record for records in parsed.nodes.values() for record in records]
    await _attach_embeddings(
        summary_nodes,
        embedding_model,
        purpose=f"hi_rag_summary_cluster_{layer}_{cluster_id}",
    )
    return parsed


async def _invoke_summary(
    prompt: str,
    *,
    language_model: LanguageModelProtocol,
    config: ExtractHiRAGGraphConfig,
    layer: int,
    cluster_id: str,
) -> str:
    result = await language_model.invoke(
        ModelRequest(
            prompt=prompt,
            options=ModelOptions(temperature=config.temperature),
            trace_context={
                "step": ExtractHiRAGGraph.name,
                "stage": "summary_clusters",
                "layer": layer,
                "cluster_id": cluster_id,
            },
        )
    )
    return result.text.strip()


async def _attach_embeddings(
    nodes: Iterable[NodeRecord] | Mapping[str, NodeRecord],
    embedding_model: EmbeddingModelProtocol,
    *,
    purpose: str,
) -> None:
    node_list = list(nodes.values()) if isinstance(nodes, Mapping) else list(nodes)
    if not node_list:
        return
    result = await embedding_model.embed(
        EmbeddingRequest(
            texts=[str(node.get("description") or "") for node in node_list],
            trace_context={"step": ExtractHiRAGGraph.name, "purpose": purpose},
        )
    )
    if len(result.vectors) != len(node_list):
        raise ValueError("embedding result count must match node count")
    for node, vector in zip(node_list, result.vectors, strict=True):
        node["embedding"] = [float(value) for value in vector]


async def _merge_node_then_upsert(
    entity_name: str,
    nodes_data: list[NodeRecord],
    graph_store: GraphStoreProtocol,
    *,
    language_model: LanguageModelProtocol,
    config: ExtractHiRAGGraphConfig,
) -> GraphNode:
    existing_types: list[str] = []
    existing_source_ids: list[str] = []
    existing_descriptions: list[str] = []
    existing = await graph_store.get_node(entity_name)
    if existing is not None:
        existing_types.extend(_property_values(existing.properties.get("entity_type")))
        existing_source_ids.extend(_property_values(existing.properties.get("source_ids")))
        existing_source_ids.extend(_property_values(existing.properties.get("source_id")))
        existing_descriptions.extend(_property_values(existing.properties.get("description")))

    entity_type = _most_common_nonempty(
        [record.get("entity_type", "") for record in nodes_data] + existing_types,
        default="ENTITY",
    )
    descriptions = _unique_nonempty([record.get("description", "") for record in nodes_data] + existing_descriptions)
    description = await _handle_entity_relation_summary(
        entity_name,
        GRAPH_FIELD_SEP.join(descriptions),
        language_model=language_model,
        config=config,
    )
    source_ids = _unique_nonempty(
        [
            source_id
            for record in nodes_data
            for source_id in [record.get("source_id", ""), *record.get("source_ids", [])]
        ]
        + existing_source_ids
    )
    layers = [int(record.get("layer", 0)) for record in nodes_data if str(record.get("layer", "")).isdigit()]
    is_summary = any(bool(record.get("is_summary")) for record in nodes_data)
    parent_ids = _unique_nonempty(
        parent for record in nodes_data for parent in record.get("parent_entity_ids", [])
    )
    cluster_ids = _unique_nonempty(record.get("cluster_id") for record in nodes_data)
    node = GraphNode(
        id=entity_name,
        labels=("Entity", entity_type),
        properties={
            "name": entity_name,
            "entity_type": entity_type,
            "raw_entity_type": _most_common_nonempty(
                [record.get("raw_entity_type", "") for record in nodes_data],
                default=entity_type,
            ),
            "description": description,
            "source_id": GRAPH_FIELD_SEP.join(source_ids),
            "source_ids": source_ids,
            "layer": min(layers) if layers else 0,
            "cluster_id": cluster_ids[0] if cluster_ids else None,
            "cluster_ids": cluster_ids,
            "is_summary": is_summary,
            "parent_entity_ids": parent_ids,
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
    config: ExtractHiRAGGraphConfig,
) -> GraphEdge | None:
    if source_id == target_id or not edges_data:
        return None
    edge_id = _edge_id(source_id, target_id)
    existing_weights: list[float] = []
    existing_source_ids: list[str] = []
    existing_descriptions: list[str] = []
    existing_orders: list[int] = []
    existing = await graph_store.get_edge(edge_id)
    if existing is not None:
        weight = existing.properties.get("weight")
        if isinstance(weight, int | float):
            existing_weights.append(float(weight))
        existing_source_ids.extend(_property_values(existing.properties.get("source_ids")))
        existing_source_ids.extend(_property_values(existing.properties.get("source_id")))
        existing_descriptions.extend(_property_values(existing.properties.get("description")))
        order = existing.properties.get("order")
        if isinstance(order, int):
            existing_orders.append(order)

    source_ids = _unique_nonempty(
        [
            source
            for record in edges_data
            for source in [record.get("source_id", ""), *record.get("source_ids", [])]
        ]
        + existing_source_ids
    )
    description = GRAPH_FIELD_SEP.join(
        _unique_nonempty([record.get("description", "") for record in edges_data] + existing_descriptions)
    )
    description = await _handle_entity_relation_summary(
        f"{source_id} -> {target_id}",
        description,
        language_model=language_model,
        config=config,
    )
    weights = [float(record.get("weight", 1.0)) for record in edges_data]
    orders = [_record_order(record) for record in edges_data] + existing_orders
    layers = [int(record.get("layer", 0)) for record in edges_data if str(record.get("layer", "")).isdigit()]
    is_summary = any(bool(record.get("is_summary")) for record in edges_data)
    parent_ids = _unique_nonempty(parent for record in edges_data for parent in record.get("parent_entity_ids", []))
    cluster_ids = _unique_nonempty(record.get("cluster_id") for record in edges_data)

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
                            "source_id": GRAPH_FIELD_SEP.join(source_ids),
                            "source_ids": source_ids,
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
            "description": description,
            "weight": sum(weights + existing_weights) if weights or existing_weights else 1.0,
            "order": min(orders) if orders else 1,
            "source_id": GRAPH_FIELD_SEP.join(source_ids),
            "source_ids": source_ids,
            "layer": min(layers) if layers else 0,
            "cluster_id": cluster_ids[0] if cluster_ids else None,
            "cluster_ids": cluster_ids,
            "is_summary": is_summary,
            "parent_entity_ids": parent_ids,
        },
    )
    await graph_store.upsert_edges([edge])
    return edge


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    *,
    language_model: LanguageModelProtocol,
    config: ExtractHiRAGGraphConfig,
) -> str:
    tokens = description.split()
    if len(tokens) <= config.entity_summary_to_max_tokens:
        return description
    prompt = str(
        config.prompts["summarize_entity_descriptions"].format(
            entity_name=entity_or_relation_name,
            description_list=GRAPH_FIELD_SEP.join(tokens[: config.summary_llm_max_tokens]).split(GRAPH_FIELD_SEP),
        )
    )
    result = await language_model.invoke(
        ModelRequest(
            prompt=prompt,
            options=ModelOptions(
                temperature=config.temperature,
                max_output_tokens=config.entity_summary_to_max_tokens,
            ),
            trace_context={
                "step": ExtractHiRAGGraph.name,
                "stage": "merge_summary",
                "summary_target": str(entity_or_relation_name),
            },
        )
    )
    return result.text.strip() or description


async def _put_graph_node(
    object_store: ObjectStoreProtocol,
    config: ExtractHiRAGGraphConfig,
    node: GraphNode,
) -> str:
    key = join_object_key(config.graph_nodes_prefix, f"{_stable_object_id(node.id)}.json")
    await object_store.put(
        key,
        json.dumps(_graph_node_to_dict(node), ensure_ascii=False, sort_keys=True).encode("utf-8"),
    )
    return key


async def _put_graph_edge(
    object_store: ObjectStoreProtocol,
    config: ExtractHiRAGGraphConfig,
    edge: GraphEdge,
) -> str:
    key = join_object_key(config.graph_edges_prefix, f"{_stable_object_id(edge.id)}.json")
    await object_store.put(
        key,
        json.dumps(_graph_edge_to_dict(edge), ensure_ascii=False, sort_keys=True).encode("utf-8"),
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


def _stable_object_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _edge_id(source_id: str, target_id: str) -> str:
    return f"{source_id}--RELATED--{target_id}"


def _context_base(prompts: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "tuple_delimiter": prompts["DEFAULT_TUPLE_DELIMITER"],
        "record_delimiter": prompts["DEFAULT_RECORD_DELIMITER"],
        "completion_delimiter": prompts["DEFAULT_COMPLETION_DELIMITER"],
        **extra,
    }


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [result.strip() for result in results if result.strip()]


def clean_str(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    result = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", html.unescape(value.strip()))
    return result.strip().strip('"').strip("'")


def is_float_regex(value: str) -> bool:
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value.strip()))


def _record_kind(value: str) -> str:
    return clean_str(value).strip('"').strip("'").lower()


def _record_order(record: Mapping[str, Any]) -> int:
    value = record.get("order", 1)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return 1


def _strip_fences(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _first_records_by_name(records: Iterable[NodeRecord]) -> dict[str, NodeRecord]:
    result: dict[str, NodeRecord] = {}
    for record in records:
        result.setdefault(str(record["entity_name"]), dict(record))
    return result


def _dedupe_nodes(nodes: Iterable[NodeRecord]) -> list[NodeRecord]:
    seen: set[str] = set()
    result: list[NodeRecord] = []
    for node in nodes:
        name = str(node.get("entity_name") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(node)
    return result


def _fit_cluster_token_budget(
    cluster_nodes: list[NodeRecord],
    config: ExtractHiRAGGraphConfig,
) -> tuple[list[NodeRecord], dict[str, Any]]:
    sampled = list(cluster_nodes)
    total_length = _cluster_text_length(sampled)
    discount_times = 0
    while total_length > config.hierarchical_max_length_in_cluster and len(sampled) > 1:
        selected = max(1, int(len(sampled) * 0.8))
        sampled = random.sample(sampled, selected)
        total_length = _cluster_text_length(sampled)
        discount_times += 1
    return sampled, {
        "sampled": len(sampled) != len(cluster_nodes),
        "sampled_count": len(sampled),
        "original_count": len(cluster_nodes),
        "discount_times": discount_times,
    }


def _cluster_text_length(nodes: Iterable[NodeRecord]) -> int:
    try:
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")
        return sum(
            len(tokenizer.encode(str(node.get("entity_name", ""))))
            + len(tokenizer.encode(str(node.get("description", ""))))
            for node in nodes
        )
    except Exception:
        return sum(
            len(str(node.get("entity_name", "")).split())
            + len(str(node.get("description", "")).split())
            for node in nodes
        )


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


def _unique_nonempty(values: Iterable[Any]) -> list[str]:
    cleaned = [clean_str(value) for value in values]
    return sorted({value for value in cleaned if value})


def _flow_trace(
    chunk: ParsedChunk,
    stage: str,
    prompt: str,
    raw_response: str,
    entities: list[NodeRecord],
    relations: list[EdgeRecord],
    glean_count: int,
    failed: bool,
) -> TraceRecord:
    return {
        "stage": stage,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "prompt": prompt,
        "raw_response": raw_response,
        "entities": entities,
        "relationships": relations,
        "gleaning_count": glean_count,
        "failed": failed,
    }


def _chunk_trace(chunk: ParsedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "content": chunk.text,
        "chunk_order_index": chunk.chunk_index,
        "tokens": max(0, chunk.token_end - chunk.token_start),
        "full_doc_id": chunk.document_id,
        "source_key": chunk.source.key,
        "file_path": chunk.source.name,
    }


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


def _require_embedding_model(component: object) -> EmbeddingModelProtocol:
    if not isinstance(component, EmbeddingModelProtocol):
        raise TypeError("models.embedding must satisfy EmbeddingModelProtocol")
    return component
