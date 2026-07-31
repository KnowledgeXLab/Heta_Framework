"""LeanRAG graph extraction and semantic aggregation steps."""

from __future__ import annotations

import hashlib
import csv
import io
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from heta_framework.common.models import EmbeddingRequest, ModelOptions, ModelRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol, LanguageModelProtocol
from heta_framework.kb.chunking import ParsedChunk
from heta_framework.kb.cleanup import StepCleanupPlan
from heta_framework.kb.graphing.prompts import LEANRAG_PROMPTS
from heta_framework.kb.steps.graph_storage import compact_json
from heta_framework.kb.steps.hirag_hierarchical_aggregation import (
    HIRAG_PROMPTS,
    ExtractHiRAGBaseGraph,
    ExtractHiRAGBaseGraphConfig,
    ExtractHiRAGBaseGraphResult,
)
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref


NodeRecord = dict[str, Any]
EdgeRecord = dict[str, Any]
ChunkRecord = dict[str, Any]
LEANRAG_FIELD_SEP = "<SEP>"


@dataclass(frozen=True)
class ExtractLeanRAGGraphConfig:
    """Configuration for LeanRAG base graph extraction."""

    temperature: float = 0.0
    max_attempts: int = 3
    object_store: str | None = None
    graph_store: str | None = None
    language_model: str | None = None
    embedding_model: str | None = None
    chunk_keys_artifact: str = "chunk_keys"
    chunks_artifact: str = "lean_rag_chunks"
    base_entities_artifact: str = "lean_rag_base_entities"
    base_relations_artifact: str = "lean_rag_base_relations"
    extraction_trace_artifact: str = "lean_rag_extraction_trace"
    graph_node_keys_artifact: str = "lean_rag_graph_node_keys"
    graph_edge_keys_artifact: str = "lean_rag_graph_edge_keys"
    result_artifact: str = "extract_lean_rag_graph_result"
    graph_nodes_prefix: str = "lean_rag/graph/nodes"
    graph_edges_prefix: str = "lean_rag/graph/edges"
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    summary_llm_max_tokens: int = 1200
    prompts: Mapping[str, Any] = field(default_factory=lambda: dict(HIRAG_PROMPTS))

    def __post_init__(self) -> None:
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
        for name in (
            self.chunk_keys_artifact,
            self.chunks_artifact,
            self.base_entities_artifact,
            self.base_relations_artifact,
            self.extraction_trace_artifact,
            self.graph_node_keys_artifact,
            self.graph_edge_keys_artifact,
            self.result_artifact,
        ):
            if name.strip() == "":
                raise ValueError("artifact names must not be empty")


@dataclass(frozen=True)
class ExtractLeanRAGGraphResult:
    """Artifacts produced by ExtractLeanRAGGraph."""

    chunk_count: int
    base_entity_count: int
    base_relation_count: int
    failed_chunk_ids: tuple[str, ...]
    node_keys: tuple[str, ...]
    edge_keys: tuple[str, ...]


class ExtractLeanRAGGraph:
    """Extract LeanRAG base entities and relations with a thin HiRAG adapter."""

    name = "extract_leanrag_graph"

    def __init__(self, config: ExtractLeanRAGGraphConfig | None = None) -> None:
        self.config = config or ExtractLeanRAGGraphConfig()

    @property
    def requirements(self) -> StepRequirements:
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
        return StepCapabilities(
            artifacts=frozenset(
                {
                    self.config.result_artifact,
                    self.config.chunks_artifact,
                    self.config.base_entities_artifact,
                    self.config.base_relations_artifact,
                    self.config.extraction_trace_artifact,
                    self.config.graph_node_keys_artifact,
                    self.config.graph_edge_keys_artifact,
                }
            )
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        delegate = ExtractHiRAGBaseGraph(_hirag_config(self.config))
        return delegate.cleanup_plan(artifacts)

    async def run(self, context: StepContextProtocol) -> None:
        """Run shared HiRAG base extraction and publish LeanRAG artifacts."""
        object_store = context.get_component(store_ref("objects", self.config.object_store).key)
        chunk_keys = tuple(context.get_artifact(self.config.chunk_keys_artifact))
        parsed_chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]

        delegate = ExtractHiRAGBaseGraph(_hirag_config(self.config))
        await delegate.run(context)

        hirag_result = context.get_artifact(self.config.result_artifact)
        if not isinstance(hirag_result, ExtractHiRAGBaseGraphResult):
            raise TypeError("ExtractHiRAGBaseGraph did not produce the expected result type")

        chunks = [_leanrag_chunk_record(chunk) for chunk in parsed_chunks]
        entities = adapt_hirag_entities_to_leanrag(
            context.get_artifact(self.config.base_entities_artifact),
            context.get_artifact(self.config.base_relations_artifact),
        )
        relations = adapt_hirag_relations_to_leanrag(
            context.get_artifact(self.config.base_relations_artifact)
        )
        trace = [
            {
                **dict(item),
                "leanrag_adapter": {
                    "base_graph_source": "hirag_two_stage_extraction",
                    "chunk_hash_field": "hash_code",
                },
            }
            for item in context.get_artifact(self.config.extraction_trace_artifact)
        ]

        result = ExtractLeanRAGGraphResult(
            chunk_count=len(chunks),
            base_entity_count=len(entities),
            base_relation_count=len(relations),
            failed_chunk_ids=hirag_result.failed_chunk_ids,
            node_keys=hirag_result.node_keys,
            edge_keys=hirag_result.edge_keys,
        )
        context.set_artifact(self.config.chunks_artifact, chunks)
        context.set_artifact(self.config.base_entities_artifact, entities)
        context.set_artifact(self.config.base_relations_artifact, relations)
        context.set_artifact(self.config.extraction_trace_artifact, trace)
        context.set_artifact(self.config.result_artifact, result)


def compute_leanrag_hash(text: str) -> str:
    """Return the LeanRAG-compatible md5 hash for chunk text."""
    return hashlib.md5(text.encode()).hexdigest()


def adapt_hirag_entities_to_leanrag(
    entities: Iterable[Mapping[str, Any]],
    relations: Iterable[Mapping[str, Any]] = (),
) -> list[NodeRecord]:
    """Map HiRAG entity records to merged LeanRAG base entity records."""
    relation_degrees = Counter(
        endpoint
        for relation in relations
        for endpoint in (str(relation.get("src_id", "")), str(relation.get("tgt_id", "")))
        if endpoint
    )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entity in entities:
        name = str(entity.get("entity_name", "")).strip()
        if name:
            grouped[name].append(entity)

    records: list[NodeRecord] = []
    for name, items in grouped.items():
        source_ids = _unique(
            source_id
            for item in items
            for source_id in _record_source_ids(item)
        )
        descriptions = _unique(str(item.get("description", "")).strip() for item in items)
        entity_type = _most_common(
            str(item.get("entity_type", "")).strip() for item in items
        ) or "UNKNOWN"
        records.append(
            {
                "entity_name": name,
                "entity_type": entity_type,
                "description": LEANRAG_FIELD_SEP.join(descriptions),
                "source_id": "|".join(source_ids),
                "source_ids": source_ids,
                "degree": int(relation_degrees.get(name, 0)),
                "parent": "",
                "level": 0,
                "is_aggregate": False,
                "properties": {"base_graph_source": "hirag_two_stage_extraction"},
            }
        )
    return records


def adapt_hirag_relations_to_leanrag(
    relations: Iterable[Mapping[str, Any]],
) -> list[EdgeRecord]:
    """Map HiRAG relation records to merged LeanRAG base relation records."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for relation in relations:
        source = str(relation.get("src_id", "")).strip()
        target = str(relation.get("tgt_id", "")).strip()
        if source and target and source != target:
            grouped[tuple(sorted((source, target)))].append(relation)

    records: list[EdgeRecord] = []
    for (source, target), items in grouped.items():
        source_ids = _unique(
            source_id
            for item in items
            for source_id in _record_source_ids(item)
        )
        descriptions = _unique(str(item.get("description", "")).strip() for item in items)
        weight = sum(float(item.get("weight", 1.0) or 1.0) for item in items)
        records.append(
            {
                "src_tgt": source,
                "tgt_src": target,
                "description": LEANRAG_FIELD_SEP.join(descriptions),
                "weight": weight,
                "source_id": "|".join(source_ids),
                "source_ids": source_ids,
                "level": 0,
                "is_generated": False,
                "properties": {"base_graph_source": "hirag_two_stage_extraction"},
            }
        )
    return records


def _hirag_config(config: ExtractLeanRAGGraphConfig) -> ExtractHiRAGBaseGraphConfig:
    return ExtractHiRAGBaseGraphConfig(
        temperature=config.temperature,
        max_attempts=config.max_attempts,
        object_store=config.object_store,
        graph_store=config.graph_store,
        language_model=config.language_model,
        chunk_keys_artifact=config.chunk_keys_artifact,
        result_artifact=config.result_artifact,
        chunks_artifact=config.chunks_artifact,
        base_entities_artifact=config.base_entities_artifact,
        base_relations_artifact=config.base_relations_artifact,
        graph_node_keys_artifact=config.graph_node_keys_artifact,
        graph_edge_keys_artifact=config.graph_edge_keys_artifact,
        extraction_trace_artifact=config.extraction_trace_artifact,
        graph_nodes_prefix=config.graph_nodes_prefix,
        graph_edges_prefix=config.graph_edges_prefix,
        entity_extract_max_gleaning=config.entity_extract_max_gleaning,
        entity_summary_to_max_tokens=config.entity_summary_to_max_tokens,
        summary_llm_max_tokens=config.summary_llm_max_tokens,
        prompts=config.prompts,
    )


def _leanrag_chunk_record(chunk: Any) -> ChunkRecord:
    if isinstance(chunk, ParsedChunk):
        parsed = chunk
    elif isinstance(chunk, Mapping):
        parsed = ParsedChunk.from_dict(dict(chunk))
    else:
        raise TypeError("lean_rag_chunks must contain ParsedChunk-compatible data")
    return {
        "chunk_id": parsed.chunk_id,
        "hash_code": compute_leanrag_hash(parsed.text),
        "document_id": parsed.document_id,
        "text": parsed.text,
        "content": parsed.text,
        "source_key": parsed.source.key,
        "source_name": parsed.source.name,
        "source_file_type": parsed.source.file_type,
        "page_index": parsed.page_index,
        "chunk_index": parsed.chunk_index,
        "chunk_order_index": parsed.chunk_index,
        "token_start": parsed.token_start,
        "token_end": parsed.token_end,
        "token_count": parsed.token_end - parsed.token_start,
        "metadata": {
            "parent_chunk_ids": list(parsed.parent_chunk_ids),
            "leanrag_hash_code": compute_leanrag_hash(parsed.text),
        },
    }


def _record_source_ids(record: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    source_id = str(record.get("source_id", "")).strip()
    if source_id:
        values.extend(source_id.split("|"))
    source_ids = record.get("source_ids", ())
    if isinstance(source_ids, str):
        values.extend(source_ids.split("|"))
    elif isinstance(source_ids, Iterable):
        values.extend(str(item) for item in source_ids)
    return _unique(value.strip() for value in values)


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _most_common(values: Iterable[str]) -> str:
    counts = Counter(value for value in values if value)
    if not counts:
        return ""
    return counts.most_common(1)[0][0]
TraceRecord = dict[str, Any]
ClusterBackend = Literal["auto", "deterministic"]


@dataclass(frozen=True)
class LeanRAGSemanticAggregationConfig:
    """Configuration for LeanRAG semantic aggregation."""

    base_entities_artifact: str = "lean_rag_base_entities"
    base_relations_artifact: str = "lean_rag_base_relations"
    all_entities_layers_artifact: str = "lean_rag_all_entities_layers"
    aggregate_entities_artifact: str = "lean_rag_aggregate_entities"
    generated_relations_artifact: str = "lean_rag_generated_relations"
    communities_artifact: str = "lean_rag_communities"
    parent_edges_artifact: str = "lean_rag_parent_edges"
    trace_artifact: str = "lean_rag_semantic_aggregation_trace"
    result_artifact: str = "lean_rag_semantic_aggregation_result"
    reduction_dimension: int = 2
    cluster_threshold: float = 0.1
    cluster_size: int = 20
    random_seed: int = 224
    gmm_n_init: int = 5
    gmm_init_params: str = "k-means++"
    max_length_in_cluster: int = 60000
    relation_context_max_tokens: int = 12000
    clustering_backend: ClusterBackend = "auto"
    temperature: float = 0.0
    aggregate_max_output_tokens: int = 800
    relation_max_output_tokens: int = 240
    language_model: str | None = None
    embedding_model: str | None = None
    prompts: Mapping[str, Any] = field(default_factory=lambda: dict(LEANRAG_PROMPTS))

    def __post_init__(self) -> None:
        if self.reduction_dimension <= 0:
            raise ValueError("reduction_dimension must be greater than zero")
        if self.cluster_size <= 1:
            raise ValueError("cluster_size must be greater than one")
        if self.gmm_n_init <= 0:
            raise ValueError("gmm_n_init must be greater than zero")
        if self.max_length_in_cluster <= 0:
            raise ValueError("max_length_in_cluster must be greater than zero")
        if self.relation_context_max_tokens <= 0:
            raise ValueError("relation_context_max_tokens must be greater than zero")
        if self.clustering_backend not in {"auto", "deterministic"}:
            raise ValueError("clustering_backend must be one of: auto, deterministic")
        for name in (
            self.base_entities_artifact,
            self.base_relations_artifact,
            self.all_entities_layers_artifact,
            self.aggregate_entities_artifact,
            self.generated_relations_artifact,
            self.communities_artifact,
            self.parent_edges_artifact,
            self.trace_artifact,
            self.result_artifact,
        ):
            if name.strip() == "":
                raise ValueError("artifact names must not be empty")


@dataclass(frozen=True)
class LeanRAGSemanticAggregationResult:
    """Artifacts produced by LeanRAGSemanticAggregation."""

    base_entity_count: int
    aggregate_entity_count: int
    generated_relation_count: int
    community_count: int
    layer_count: int
    root_entity_name: str | None


class LeanRAGSemanticAggregation:
    """Build LeanRAG aggregate entities, generated relations, and parent tree."""

    name = "leanrag_semantic_aggregation"

    def __init__(self, config: LeanRAGSemanticAggregationConfig | None = None) -> None:
        self.config = config or LeanRAGSemanticAggregationConfig()

    @property
    def requirements(self) -> StepRequirements:
        return StepRequirements(
            components=frozenset(
                {
                    model_ref("language", self.config.language_model),
                    model_ref("embedding", self.config.embedding_model),
                }
            ),
            artifacts=frozenset(
                {
                    self.config.base_entities_artifact,
                    self.config.base_relations_artifact,
                }
            ),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        return StepCapabilities(
            artifacts=frozenset(
                {
                    self.config.all_entities_layers_artifact,
                    self.config.aggregate_entities_artifact,
                    self.config.generated_relations_artifact,
                    self.config.communities_artifact,
                    self.config.parent_edges_artifact,
                    self.config.trace_artifact,
                    self.config.result_artifact,
                }
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        language_model = _require_language_model(
            context.get_component(model_ref("language", self.config.language_model).key)
        )
        embedding_model = _require_embedding_model(
            context.get_component(model_ref("embedding", self.config.embedding_model).key)
        )
        base_entities = [dict(item) for item in context.get_artifact(self.config.base_entities_artifact)]
        base_relations = [dict(item) for item in context.get_artifact(self.config.base_relations_artifact)]

        result = await perform_semantic_aggregation(
            base_entities,
            base_relations,
            language_model=language_model,
            embedding_model=embedding_model,
            config=self.config,
        )
        for artifact_name, value in result.artifacts.items():
            context.set_artifact(artifact_name, value)
        context.set_artifact(self.config.result_artifact, result.summary)


@dataclass(frozen=True)
class _AggregationOutput:
    artifacts: Mapping[str, Any]
    summary: LeanRAGSemanticAggregationResult


async def perform_semantic_aggregation(
    base_entities: list[NodeRecord],
    base_relations: list[EdgeRecord],
    *,
    language_model: LanguageModelProtocol,
    embedding_model: EmbeddingModelProtocol,
    config: LeanRAGSemanticAggregationConfig,
) -> _AggregationOutput:
    nodes = [_base_node(entity) for entity in base_entities]
    relations_by_pair = _relations_by_pair(base_relations)
    generated_relations_by_pair: dict[tuple[str, str], EdgeRecord] = {}
    communities: dict[str, NodeRecord] = {}
    aggregate_entities: list[NodeRecord] = []
    parent_edges: list[EdgeRecord] = []
    trace: list[TraceRecord] = []
    all_layers: list[Any] = [nodes]
    root_entity_name: str | None = None

    await _attach_embeddings(nodes, embedding_model, purpose="leanrag_base_entities")
    max_depth = _max_depth(len(nodes), config.cluster_size)

    current_nodes = nodes
    for layer in range(max_depth):
        if len(current_nodes) <= 2:
            trace.append(_stop_trace(layer, current_nodes, "entity_count_le_2", config, max_depth))
            break

        assignments, backend, cluster_trace = _cluster_assignments(current_nodes, config)
        unique_labels = sorted({label for labels in assignments for label in labels})
        layer_trace: TraceRecord = {
            "stage": "semantic_aggregation",
            "layer": layer,
            "entity_count": len(current_nodes),
            "embedding_shape": _embedding_shape(current_nodes),
            "backend": backend,
            "cluster_labels": assignments,
            "unique_cluster_count": len(unique_labels),
            "cluster_sizes": dict(Counter(label for labels in assignments for label in labels)),
            "params": _cluster_params(config, max_depth),
            **cluster_trace,
        }
        if len(unique_labels) <= 4:
            layer_trace["stop_reason"] = "unique_cluster_count_le_4"
            trace.append(layer_trace)
            break

        next_nodes: list[NodeRecord] = []
        layer_communities: list[str] = []
        for label in unique_labels:
            indexes = [index for index, labels in enumerate(assignments) if label in labels]
            cluster_nodes = [current_nodes[index] for index in indexes]
            if len(cluster_nodes) == 1:
                cluster_nodes[0]["parent"] = cluster_nodes[0]["entity_name"]
                next_nodes.append(dict(cluster_nodes[0]))
                parent_edges.append(_parent_edge(cluster_nodes[0]["entity_name"], cluster_nodes[0]["parent"], layer))
                continue

            community, temp_node, prompt, response = await _aggregate_cluster(
                cluster_nodes,
                base_relations_by_pair=relations_by_pair,
                generated_relations_by_pair=generated_relations_by_pair,
                layer=layer,
                language_model=language_model,
                embedding_model=embedding_model,
                config=config,
            )
            communities[community["entity_name"]] = community
            aggregate_entities.append(temp_node)
            layer_communities.append(community["entity_name"])
            for child in cluster_nodes:
                child["parent"] = temp_node["entity_name"]
                parent_edges.append(_parent_edge(child["entity_name"], temp_node["entity_name"], layer))
            next_nodes.append(temp_node)
            layer_trace.setdefault("aggregate_entities", []).append(
                {
                    "cluster_label": label,
                    "children": community["children"],
                    "aggregate_entity": temp_node["entity_name"],
                    "prompt": prompt,
                    "raw_response": response,
                }
            )

        generated = await _generate_layer_relations(
            layer_communities,
            communities,
            relations_by_pair,
            generated_relations_by_pair,
            layer=layer,
            max_depth=max_depth,
            language_model=language_model,
            config=config,
            trace=layer_trace,
        )
        generated_relations_by_pair.update({tuple(sorted((r["src_tgt"], r["tgt_src"]))): r for r in generated})

        current_nodes = _dedupe_nodes(next_nodes)
        all_layers.append(current_nodes)
        trace.append(layer_trace)
        await _attach_embeddings(current_nodes, embedding_model, purpose=f"leanrag_layer_{layer + 1}")

    if all_layers:
        last_layer = all_layers[-1]
        if isinstance(last_layer, list) and len(last_layer) == 1:
            last_layer[0]["parent"] = "root"
            parent_edges.append(_parent_edge(last_layer[0]["entity_name"], "root", len(all_layers) - 1))
            root_entity_name = last_layer[0]["entity_name"]
        elif isinstance(last_layer, list) and len(last_layer) > 1:
            root_community, root_node, prompt, response = await _aggregate_cluster(
                last_layer,
                base_relations_by_pair={},
                generated_relations_by_pair=generated_relations_by_pair,
                layer=max(0, len(all_layers) - 1),
                language_model=language_model,
                embedding_model=embedding_model,
                config=config,
                entity_type="community",
            )
            root_node["parent"] = "root"
            root_community["level"] = max(0, len(all_layers) - 1)
            communities[root_community["entity_name"]] = root_community
            aggregate_entities.append(root_node)
            for child in last_layer:
                child["parent"] = root_node["entity_name"]
                parent_edges.append(_parent_edge(child["entity_name"], root_node["entity_name"], len(all_layers) - 1))
            parent_edges.append(_parent_edge(root_node["entity_name"], "root", len(all_layers)))
            all_layers.append(root_node)
            root_entity_name = root_node["entity_name"]
            trace.append(
                {
                    "stage": "semantic_aggregation_root",
                    "layer": root_community["level"],
                    "children": root_community["children"],
                    "aggregate_entity": root_node["entity_name"],
                    "prompt": prompt,
                    "raw_response": response,
                }
            )

    generated_relations = list(generated_relations_by_pair.values())
    artifacts = {
        config.all_entities_layers_artifact: strip_embeddings(all_layers),
        config.aggregate_entities_artifact: strip_embeddings(aggregate_entities),
        config.generated_relations_artifact: generated_relations,
        config.communities_artifact: list(communities.values()),
        config.parent_edges_artifact: parent_edges,
        config.trace_artifact: trace,
    }
    summary = LeanRAGSemanticAggregationResult(
        base_entity_count=len(base_entities),
        aggregate_entity_count=len(aggregate_entities),
        generated_relation_count=len(generated_relations),
        community_count=len(communities),
        layer_count=len(all_layers),
        root_entity_name=root_entity_name,
    )
    return _AggregationOutput(artifacts=artifacts, summary=summary)


async def _aggregate_cluster(
    cluster_nodes: list[NodeRecord],
    *,
    base_relations_by_pair: Mapping[tuple[str, str], EdgeRecord],
    generated_relations_by_pair: Mapping[tuple[str, str], EdgeRecord],
    layer: int,
    language_model: LanguageModelProtocol,
    embedding_model: EmbeddingModelProtocol,
    config: LeanRAGSemanticAggregationConfig,
    entity_type: str = "aggregate entity",
) -> tuple[NodeRecord, NodeRecord, str, str]:
    child_names = [str(node["entity_name"]) for node in cluster_nodes]
    relations = _direct_relations(child_names, child_names, base_relations_by_pair, generated_relations_by_pair)
    describe = pack_single_community_describe(cluster_nodes, relations, max_token_size=config.max_length_in_cluster)
    prompt = str(config.prompts["aggregate_entities"].format(input_text=describe))
    response = await _invoke(
        prompt,
        language_model=language_model,
        max_output_tokens=config.aggregate_max_output_tokens,
        temperature=config.temperature,
        trace_context={"step": LeanRAGSemanticAggregation.name, "stage": "aggregate_entities", "layer": layer},
    )
    data = parse_aggregate_response(response)
    source_ids = _unique(source_id for node in cluster_nodes for source_id in _source_ids(node))
    community = {
        "entity_name": str(data.get("entity_name", "")).strip() or f"Aggregate Layer {layer}",
        "entity_description": str(data.get("entity_description", "")).strip(),
        "findings": data.get("findings", []),
        "level": layer,
        "children": child_names,
        "source_id": "|".join(source_ids),
        "source_ids": source_ids,
    }
    temp_node = {
        "entity_name": community["entity_name"],
        "description": community["entity_description"],
        "source_id": community["source_id"],
        "source_ids": source_ids,
        "entity_type": entity_type,
        "degree": 1,
        "parent": "",
        "level": layer + 1,
        "children": child_names,
        "findings": community["findings"],
        "is_aggregate": True,
    }
    await _attach_embeddings([temp_node], embedding_model, purpose=f"leanrag_aggregate_{layer}")
    return community, temp_node, prompt, response


async def _generate_layer_relations(
    aggregate_names: list[str],
    communities: Mapping[str, NodeRecord],
    base_relations_by_pair: Mapping[tuple[str, str], EdgeRecord],
    generated_relations_by_pair: Mapping[tuple[str, str], EdgeRecord],
    *,
    layer: int,
    max_depth: int,
    language_model: LanguageModelProtocol,
    config: LeanRAGSemanticAggregationConfig,
    trace: TraceRecord,
) -> list[EdgeRecord]:
    generated: list[EdgeRecord] = []
    for index, source in enumerate(aggregate_names):
        for target in aggregate_names[index + 1 :]:
            source_children = [str(item) for item in communities[source]["children"]]
            target_children = [str(item) for item in communities[target]["children"]]
            evidence = _direct_relations(
                source_children,
                target_children,
                base_relations_by_pair,
                generated_relations_by_pair,
            )
            if not evidence:
                continue
            relation_information = [
                f"relationship<|>{item['src_tgt']}<|>{item['tgt_src']}<|>{item['description']} "
                for item in evidence.values()
            ]
            token_count = token_count_for_text("\n".join(relation_information))
            gene_tokens = (layer + 1) * 40
            allowed_tokens = (max_depth - layer) * 40 * 2
            if token_count > allowed_tokens:
                prompt = str(
                    config.prompts["cluster_cluster_relation"].format(
                        entity_a=source,
                        entity_b=target,
                        entity_a_description=communities[source].get("findings", []),
                        entity_b_description=communities[target].get("findings", []),
                        relation_information="\n".join(relation_information),
                        tokens=gene_tokens,
                    )
                )
                description = await _invoke(
                    prompt,
                    language_model=language_model,
                    max_output_tokens=config.relation_max_output_tokens,
                    temperature=config.temperature,
                    trace_context={
                        "step": LeanRAGSemanticAggregation.name,
                        "stage": "cluster_cluster_relation",
                        "layer": layer,
                    },
                )
                called_llm = True
            else:
                prompt = ""
                description = "\n".join(relation_information)
                called_llm = False
            relation = {
                "src_tgt": source,
                "tgt_src": target,
                "description": description,
                "weight": 1,
                "level": layer + 1,
                "source_id": "|".join(_unique(source_id for item in evidence.values() for source_id in _source_ids(item))),
                "source_ids": _unique(source_id for item in evidence.values() for source_id in _source_ids(item)),
                "is_generated": True,
                "evidence_relation_ids": [compact_json([item["src_tgt"], item["tgt_src"], item.get("level", 0)]) for item in evidence.values()],
            }
            generated.append(relation)
            trace.setdefault("aggregate_relations", []).append(
                {
                    "src_tgt": source,
                    "tgt_src": target,
                    "evidence_count": len(evidence),
                    "token_count": token_count,
                    "allowed_tokens": allowed_tokens,
                    "gene_tokens": gene_tokens,
                    "called_llm": called_llm,
                    "prompt": prompt,
                    "description": description,
                }
            )
    return generated


def pack_single_community_describe(
    entities: list[Mapping[str, Any]],
    relations: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    max_token_size: int = 12000,
) -> str:
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    nodes_list = [
        [
            index,
            entity.get("entity_name"),
            entity.get("entity_type", "UNKNOWN"),
            entity.get("description", "UNKNOWN"),
            entity.get("degree", 1),
        ]
        for index, entity in enumerate(entities)
    ]
    nodes_list = sorted(nodes_list, key=lambda row: row[-1], reverse=True)
    nodes_list = truncate_list_by_token_size(nodes_list, key=lambda row: str(row[3]), max_token_size=max_token_size // 2)
    edges_list = [
        [index, relation.get("src_tgt"), relation.get("tgt_src"), relation.get("description", "UNKNOWN")]
        for index, relation in enumerate(relations.values())
    ]
    edges_list = sorted(edges_list, key=lambda row: row[-1], reverse=True)
    edges_list = truncate_list_by_token_size(edges_list, key=lambda row: str(row[3]), max_token_size=max_token_size // 2)
    return f"""
-----Entities-----
```csv
{_list_of_list_to_csv([node_fields] + nodes_list)}
```
-----Relationships-----
```csv
{_list_of_list_to_csv([edge_fields] + edges_list)}
```"""


def parse_aggregate_response(response: str) -> dict[str, Any]:
    match = _first_json_object(response)
    if match is not None:
        return match
    name_match = re.search(r"Aggregate Entity Name:\s*(.+)", response)
    description_match = re.search(r"Aggregate Entity Description:\s*(.+?)(?:\n\nFindings:|\Z)", response, re.DOTALL)
    findings = [
        {"summary": summary.strip(), "explanation": explanation.strip()}
        for _, summary, explanation in re.findall(
            r"<summary_(\d+)>:\s*(.*?)\s*<explanation_\1>:\s*(.*?)(?=\n<summary_\d+>:|\Z)",
            response,
            re.DOTALL,
        )
    ]
    return {
        "entity_name": name_match.group(1).strip() if name_match else "Aggregate Entity",
        "entity_description": description_match.group(1).strip() if description_match else response.strip(),
        "findings": findings,
    }


def strip_embeddings(value: Any) -> Any:
    if isinstance(value, list):
        return [strip_embeddings(item) for item in value]
    if isinstance(value, dict):
        return {key: strip_embeddings(item) for key, item in value.items() if key != "embedding"}
    return value


def export_all_entities_json_lines(all_entities_layers: list[Any]) -> str:
    return "\n".join(json.dumps(layer, ensure_ascii=False) for layer in all_entities_layers)


def export_generated_relations_json_lines(generated_relations: list[EdgeRecord]) -> str:
    return "\n".join(json.dumps(relation, ensure_ascii=False) for relation in generated_relations)


def export_communities_json_lines(communities: list[NodeRecord]) -> str:
    return "\n".join(json.dumps(community, ensure_ascii=False) for community in communities)


def _cluster_assignments(
    nodes: list[NodeRecord],
    config: LeanRAGSemanticAggregationConfig,
) -> tuple[list[list[int]], str, dict[str, Any]]:
    if config.clustering_backend == "deterministic":
        return _deterministic_cluster_assignments(nodes, config.cluster_size), "deterministic", {}
    try:
        import numpy as np
        import umap
        from sklearn.mixture import GaussianMixture
    except Exception as exc:
        return _deterministic_cluster_assignments(nodes, config.cluster_size), "deterministic_fallback", {"fallback_reason": str(exc)}

    embeddings = np.array([node.get("embedding", []) for node in nodes], dtype=float)
    if embeddings.ndim != 2 or embeddings.shape[0] <= 2 or embeddings.shape[1] == 0:
        return _deterministic_cluster_assignments(nodes, config.cluster_size), "deterministic_fallback", {"fallback_reason": "invalid_embedding_shape"}
    try:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
        if n_neighbors <= 1:
            n_neighbors = 2
        reduced = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=min(config.reduction_dimension, len(embeddings) - 2),
            metric="cosine",
            random_state=config.random_seed,
        ).fit_transform(embeddings)
        n_clusters = max(len(embeddings) // config.cluster_size, _optimal_clusters(reduced, config, GaussianMixture))
        model = GaussianMixture(
            n_components=n_clusters,
            random_state=config.random_seed,
            n_init=config.gmm_n_init,
            init_params=config.gmm_init_params,
        )
        model.fit(reduced)
        probabilities = model.predict_proba(reduced)
        labels = [[int(prob.argmax())] for prob in probabilities]
        return labels, "leanrag_umap_gmm", {"n_clusters": int(n_clusters), "umap_n_neighbors": n_neighbors}
    except Exception as exc:
        return _deterministic_cluster_assignments(nodes, config.cluster_size), "deterministic_fallback", {"fallback_reason": str(exc)}


def _optimal_clusters(embeddings: Any, config: LeanRAGSemanticAggregationConfig, gaussian_mixture: Any) -> int:
    import numpy as np

    max_clusters = min(len(embeddings), 50)
    candidates = np.arange(1, max_clusters)
    if len(candidates) == 0:
        return 1
    bics: list[float] = []
    previous_bic = float("inf")
    for count in candidates:
        model = gaussian_mixture(
            n_components=int(count),
            random_state=config.random_seed,
            n_init=config.gmm_n_init,
            init_params=config.gmm_init_params,
        )
        model.fit(embeddings)
        bic = float(model.bic(embeddings))
        bics.append(bic)
        if previous_bic != float("inf") and abs(previous_bic - bic) / abs(previous_bic) < 1e-3:
            break
        previous_bic = bic
    return int(candidates[int(np.argmin(bics))])


def _deterministic_cluster_assignments(nodes: list[NodeRecord], cluster_size: int) -> list[list[int]]:
    return [[index // cluster_size] for index, _ in enumerate(nodes)]


async def _attach_embeddings(
    nodes: Iterable[NodeRecord],
    embedding_model: EmbeddingModelProtocol,
    *,
    purpose: str,
) -> None:
    node_list = list(nodes)
    if not node_list:
        return
    result = await embedding_model.embed(
        EmbeddingRequest(
            texts=[str(node.get("description") or node.get("entity_description") or "") for node in node_list],
            trace_context={"step": LeanRAGSemanticAggregation.name, "purpose": purpose},
        )
    )
    if len(result.vectors) != len(node_list):
        raise ValueError("embedding result count must match node count")
    for node, vector in zip(node_list, result.vectors, strict=True):
        node["embedding"] = [float(value) for value in vector]


async def _invoke(
    prompt: str,
    *,
    language_model: LanguageModelProtocol,
    max_output_tokens: int,
    temperature: float,
    trace_context: Mapping[str, Any],
) -> str:
    result = await language_model.invoke(
        ModelRequest(
            prompt=prompt,
            options=ModelOptions(temperature=temperature, max_output_tokens=max_output_tokens),
            trace_context=dict(trace_context),
        )
    )
    return result.text.strip()


def _base_node(entity: Mapping[str, Any]) -> NodeRecord:
    return {
        **dict(entity),
        "parent": str(entity.get("parent") or ""),
        "level": int(entity.get("level", 0) or 0),
        "is_aggregate": bool(entity.get("is_aggregate", False)),
    }


def _relations_by_pair(relations: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], EdgeRecord]:
    return {
        tuple(sorted((str(relation["src_tgt"]), str(relation["tgt_src"])))): dict(relation)
        for relation in relations
        if relation.get("src_tgt") and relation.get("tgt_src")
    }


def _direct_relations(
    set1: Iterable[str],
    set2: Iterable[str],
    base_relations: Mapping[tuple[str, str], EdgeRecord],
    generated_relations: Mapping[tuple[str, str], EdgeRecord],
) -> dict[tuple[str, str], EdgeRecord]:
    left = set(set1)
    right = set(set2)
    return {
        key: value
        for source in (base_relations, generated_relations)
        for key, value in source.items()
        if (key[0] in left and key[1] in right) or (key[0] in right and key[1] in left)
    }


def _parent_edge(child: str, parent: str, layer: int) -> EdgeRecord:
    return {
        "src_tgt": child,
        "tgt_src": parent,
        "description": "LeanRAG parent assignment",
        "weight": 1,
        "level": layer,
        "is_generated": True,
        "relation_type": "leanrag_parent",
    }


def _dedupe_nodes(nodes: Iterable[NodeRecord]) -> list[NodeRecord]:
    by_name: dict[str, NodeRecord] = {}
    for node in nodes:
        name = str(node.get("entity_name", ""))
        if name and name not in by_name:
            by_name[name] = node
    return list(by_name.values())


def _max_depth(entity_count: int, cluster_size: int) -> int:
    if entity_count <= 1:
        return 1
    return round(math.log(entity_count, cluster_size)) + 1


def _stop_trace(
    layer: int,
    nodes: list[NodeRecord],
    reason: str,
    config: LeanRAGSemanticAggregationConfig,
    max_depth: int,
) -> TraceRecord:
    return {
        "stage": "semantic_aggregation",
        "layer": layer,
        "entity_count": len(nodes),
        "embedding_shape": _embedding_shape(nodes),
        "params": _cluster_params(config, max_depth),
        "stop_reason": reason,
    }


def _cluster_params(config: LeanRAGSemanticAggregationConfig, max_depth: int) -> dict[str, Any]:
    return {
        "reduction_dimension": config.reduction_dimension,
        "cluster_threshold": config.cluster_threshold,
        "cluster_size": config.cluster_size,
        "random_seed": config.random_seed,
        "gmm_n_init": config.gmm_n_init,
        "gmm_init_params": config.gmm_init_params,
        "max_depth": max_depth,
    }


def _embedding_shape(nodes: list[NodeRecord]) -> list[int]:
    if not nodes:
        return [0, 0]
    first = nodes[0].get("embedding", [])
    return [len(nodes), len(first) if isinstance(first, list) else 0]


def _source_ids(record: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    source_ids = record.get("source_ids", ())
    if isinstance(source_ids, str):
        values.extend(source_ids.split("|"))
    elif isinstance(source_ids, Iterable):
        values.extend(str(item) for item in source_ids)
    source_id = str(record.get("source_id", "")).strip()
    if source_id:
        values.extend(source_id.split("|"))
    return _unique(value.strip() for value in values)


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _list_of_list_to_csv(data: list[list[Any]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    for row in data:
        writer.writerow(row)
    return output.getvalue().strip()


def truncate_list_by_token_size(
    list_data: list[Any],
    *,
    key: Any,
    max_token_size: int,
) -> list[Any]:
    if max_token_size <= 0:
        return []
    tokens = 0
    for index, item in enumerate(list_data):
        tokens += token_count_for_text(str(key(item)))
        if tokens > max_token_size:
            return list_data[:index]
    return list_data


def token_count_for_text(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(text.split())


def _first_json_object(text: str) -> dict[str, Any] | None:
    stack: list[int] = []
    start: int | None = None
    for index, char in enumerate(text):
        if char == "{":
            if start is None:
                start = index
            stack.append(index)
        elif char == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                try:
                    return json.loads(text[start : index + 1].replace("\n", ""))
                except json.JSONDecodeError:
                    return None
    return None


def _require_language_model(component: object) -> LanguageModelProtocol:
    if not isinstance(component, LanguageModelProtocol):
        raise TypeError("models.language must satisfy LanguageModelProtocol")
    return component


def _require_embedding_model(component: object) -> EmbeddingModelProtocol:
    if not isinstance(component, EmbeddingModelProtocol):
        raise TypeError("models.embedding must satisfy EmbeddingModelProtocol")
    return component
