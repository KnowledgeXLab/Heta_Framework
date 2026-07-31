"""Adapt universal graph artifacts to RAG-specific graph artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from heta_framework.common.models import ModelOptions, ModelRequest
from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.common.stores.graph import GraphEdge, GraphNode, GraphStoreProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb.chunking import ParsedChunk
from heta_framework.kb.cleanup import StepCleanupPlan, object_key_targets
from heta_framework.kb.graphing import ExtractedEntity, ExtractedRelation
from heta_framework.kb.graphing.prompts import LIGHTRAG_RELATION_KEYWORDS_PROMPT
from heta_framework.kb.steps.leanrag_semantic_aggregation import compute_leanrag_hash
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref
from heta_framework.kb.steps.universal_graph_common import (
    RAGAdapterTarget,
    _json,
    _json_bytes,
    _most_common,
    _relation_weight,
    _require_graph_store,
    _require_language_model,
    _require_object_store,
    _safe_key,
    _stable_id,
    _unique,
)


GRAPH_FIELD_SEP = "<SEP>"
LEANRAG_FIELD_SEP = "<SEP>"


@dataclass(frozen=True)
class AdaptUniversalGraphConfig:
    """Configuration for adapting universal graph artifacts to a RAG graph format."""

    target: RAGAdapterTarget
    object_store: str | None = None
    graph_store: str | None = None
    language_model: str | None = None
    chunk_keys_artifact: str = "chunk_keys"
    entity_keys_artifact: str = "entity_keys"
    relation_keys_artifact: str = "relation_keys"
    graph_nodes_prefix: str = "universal_rag/graph/nodes"
    graph_edges_prefix: str = "universal_rag/graph/edges"
    graph_node_keys_artifact: str = "graph_node_keys"
    graph_edge_keys_artifact: str = "graph_edge_keys"
    chunks_artifact: str = "rag_chunks"
    base_entities_artifact: str = "rag_base_entities"
    base_relations_artifact: str = "rag_base_relations"
    extraction_trace_artifact: str = "rag_extraction_trace"
    document_token_counts_artifact: str = "document_token_counts"
    result_artifact: str = "adapt_universal_graph_result"
    lightrag_extract_keywords: bool = True
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.target not in {"graphrag", "lightrag", "hirag", "leanrag"}:
            raise ValueError("target must be one of: graphrag, lightrag, hirag, leanrag")
        validate_object_prefix(self.graph_nodes_prefix)
        validate_object_prefix(self.graph_edges_prefix)
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        for name in (
            self.chunk_keys_artifact,
            self.entity_keys_artifact,
            self.relation_keys_artifact,
            self.graph_node_keys_artifact,
            self.graph_edge_keys_artifact,
            self.result_artifact,
            self.document_token_counts_artifact,
        ):
            if name.strip() == "":
                raise ValueError("artifact names must not be empty")


@dataclass(frozen=True)
class AdaptUniversalGraphResult:
    """Artifacts produced by AdaptUniversalGraph."""

    target: RAGAdapterTarget
    node_keys: tuple[str, ...]
    edge_keys: tuple[str, ...]
    entity_count: int
    relation_count: int
    chunk_count: int


@dataclass(frozen=True)
class _UniversalGraphAdapterConfig:
    """Shared configuration for universal graph adapter steps."""

    object_store: str | None = None
    graph_store: str | None = None
    chunk_keys_artifact: str = "chunk_keys"
    entity_keys_artifact: str = "entity_keys"
    relation_keys_artifact: str = "relation_keys"
    graph_nodes_prefix: str = "universal_rag/graph/nodes"
    graph_edges_prefix: str = "universal_rag/graph/edges"
    graph_node_keys_artifact: str = "graph_node_keys"
    graph_edge_keys_artifact: str = "graph_edge_keys"
    result_artifact: str = "adapt_universal_graph_result"

    def __post_init__(self) -> None:
        validate_object_prefix(self.graph_nodes_prefix)
        validate_object_prefix(self.graph_edges_prefix)
        for name in (
            self.chunk_keys_artifact,
            self.entity_keys_artifact,
            self.relation_keys_artifact,
            self.graph_node_keys_artifact,
            self.graph_edge_keys_artifact,
            self.result_artifact,
        ):
            if name.strip() == "":
                raise ValueError("artifact names must not be empty")


@dataclass(frozen=True)
class AdaptUniversalGraphForGraphRAGConfig(_UniversalGraphAdapterConfig):
    """Configuration for adapting universal graph artifacts to GraphRAG."""


@dataclass(frozen=True)
class AdaptUniversalGraphForLightRAGConfig(_UniversalGraphAdapterConfig):
    """Configuration for adapting universal graph artifacts to LightRAG."""

    language_model: str | None = None
    lightrag_extract_keywords: bool = True
    temperature: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")


@dataclass(frozen=True)
class _HierarchicalUniversalGraphAdapterConfig(_UniversalGraphAdapterConfig):
    """Shared configuration for hierarchical RAG universal graph adapters."""

    chunks_artifact: str = "rag_chunks"
    base_entities_artifact: str = "rag_base_entities"
    base_relations_artifact: str = "rag_base_relations"
    extraction_trace_artifact: str = "rag_extraction_trace"

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            self.chunks_artifact,
            self.base_entities_artifact,
            self.base_relations_artifact,
            self.extraction_trace_artifact,
        ):
            if name.strip() == "":
                raise ValueError("artifact names must not be empty")


@dataclass(frozen=True)
class AdaptUniversalGraphForHiRAGConfig(_HierarchicalUniversalGraphAdapterConfig):
    """Configuration for adapting universal graph artifacts to HiRAG."""


@dataclass(frozen=True)
class AdaptUniversalGraphForLeanRAGConfig(_HierarchicalUniversalGraphAdapterConfig):
    """Configuration for adapting universal graph artifacts to LeanRAG."""

    document_token_counts_artifact: str = "document_token_counts"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.document_token_counts_artifact.strip() == "":
            raise ValueError("document_token_counts_artifact must not be empty")


@dataclass(frozen=True)
class _UniversalGraphAdapterInputs:
    object_store: ObjectStoreProtocol
    graph_store: GraphStoreProtocol
    chunks: list[ParsedChunk]
    chunk_by_id: dict[str, ParsedChunk]
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


class _BaseUniversalGraphAdapter:
    """Shared runtime behavior for concrete universal graph adapter steps."""

    config: _UniversalGraphAdapterConfig

    @property
    def requirements(self) -> StepRequirements:
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("objects", self.config.object_store),
                    store_ref("graph", self.config.graph_store),
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
                    self.config.graph_node_keys_artifact,
                    self.config.graph_edge_keys_artifact,
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

    async def _load_inputs(self, context: StepContextProtocol) -> _UniversalGraphAdapterInputs:
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        graph_store = _require_graph_store(
            context.get_component(store_ref("graph", self.config.graph_store).key)
        )
        chunk_keys = tuple(context.get_artifact(self.config.chunk_keys_artifact))
        entity_keys = tuple(context.get_artifact(self.config.entity_keys_artifact))
        relation_keys = tuple(context.get_artifact(self.config.relation_keys_artifact))
        chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]
        entities = [ExtractedEntity.from_json(await object_store.get(key)) for key in entity_keys]
        relations = [
            ExtractedRelation.from_json(await object_store.get(key)) for key in relation_keys
        ]
        return _UniversalGraphAdapterInputs(
            object_store=object_store,
            graph_store=graph_store,
            chunks=chunks,
            chunk_by_id={chunk.chunk_id: chunk for chunk in chunks},
            entities=entities,
            relations=relations,
        )

    async def _persist_graph(
        self,
        context: StepContextProtocol,
        inputs: _UniversalGraphAdapterInputs,
        *,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        target: RAGAdapterTarget,
    ) -> None:
        await inputs.graph_store.upsert_nodes(nodes)
        if edges:
            await inputs.graph_store.upsert_edges(edges)

        node_keys = tuple(
            [
                await _put_graph_node(
                    inputs.object_store,
                    self.config.graph_nodes_prefix,
                    node,
                )
                for node in nodes
            ]
        )
        edge_keys = tuple(
            [
                await _put_graph_edge(
                    inputs.object_store,
                    self.config.graph_edges_prefix,
                    edge,
                )
                for edge in edges
            ]
        )
        context.set_artifact(self.config.graph_node_keys_artifact, node_keys)
        context.set_artifact(self.config.graph_edge_keys_artifact, edge_keys)
        context.set_artifact(
            self.config.result_artifact,
            AdaptUniversalGraphResult(
                target=target,
                node_keys=node_keys,
                edge_keys=edge_keys,
                entity_count=len(nodes),
                relation_count=len(edges),
                chunk_count=len(inputs.chunks),
            ),
        )


class AdaptUniversalGraphForGraphRAG(_BaseUniversalGraphAdapter):
    """Adapt universal graph artifacts to GraphRAG graph artifacts."""

    name = "adapt_universal_graph_for_graphrag"

    def __init__(self, config: AdaptUniversalGraphForGraphRAGConfig | None = None) -> None:
        self.config = config or AdaptUniversalGraphForGraphRAGConfig()

    async def run(self, context: StepContextProtocol) -> None:
        inputs = await self._load_inputs(context)
        await self._persist_graph(
            context,
            inputs,
            nodes=_graphrag_nodes(inputs.entities),
            edges=_graphrag_edges(inputs.relations),
            target="graphrag",
        )


class AdaptUniversalGraphForLightRAG(_BaseUniversalGraphAdapter):
    """Adapt universal graph artifacts to LightRAG graph artifacts."""

    name = "adapt_universal_graph_for_lightrag"

    def __init__(self, config: AdaptUniversalGraphForLightRAGConfig | None = None) -> None:
        self.config = config or AdaptUniversalGraphForLightRAGConfig()

    @property
    def requirements(self) -> StepRequirements:
        base = super().requirements
        if not self.config.lightrag_extract_keywords:
            return base
        return StepRequirements(
            components=base.components
            | frozenset({model_ref("language", self.config.language_model)}),
            artifacts=base.artifacts,
        )

    async def run(self, context: StepContextProtocol) -> None:
        inputs = await self._load_inputs(context)
        language_model = None
        if self.config.lightrag_extract_keywords:
            language_model = _require_language_model(
                context.get_component(model_ref("language", self.config.language_model).key)
            )
        await self._persist_graph(
            context,
            inputs,
            nodes=_lightrag_nodes(inputs.entities, chunk_by_id=inputs.chunk_by_id),
            edges=await _lightrag_edges(
                inputs.relations,
                chunk_by_id=inputs.chunk_by_id,
                language_model=language_model,
                temperature=self.config.temperature,
            ),
            target="lightrag",
        )


class _BaseHierarchicalUniversalGraphAdapter(_BaseUniversalGraphAdapter):
    config: _HierarchicalUniversalGraphAdapterConfig
    target: RAGAdapterTarget

    @property
    def capabilities(self) -> StepCapabilities:
        base = super().capabilities
        return StepCapabilities(
            artifacts=base.artifacts
            | frozenset(
                {
                    self.config.chunks_artifact,
                    self.config.base_entities_artifact,
                    self.config.base_relations_artifact,
                    self.config.extraction_trace_artifact,
                }
            )
        )

    def _set_base_artifacts(
        self,
        context: StepContextProtocol,
        inputs: _UniversalGraphAdapterInputs,
        *,
        document_token_counts: Mapping[str, int] | None = None,
    ) -> None:
        chunks = [_chunk_record(chunk) for chunk in inputs.chunks]
        base_entities = self._base_entities(inputs.entities, inputs.relations)
        if document_token_counts is not None:
            base_entities = _attach_document_provenance(
                base_entities,
                chunks=chunks,
                document_token_counts=document_token_counts,
            )
        context.set_artifact(
            self.config.chunks_artifact,
            chunks,
        )
        context.set_artifact(
            self.config.base_entities_artifact,
            base_entities,
        )
        context.set_artifact(
            self.config.base_relations_artifact,
            self._base_relations(inputs.relations),
        )
        context.set_artifact(
            self.config.extraction_trace_artifact,
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "stage": "universal_graph_adapter",
                    "target": self.target,
                }
                for chunk in inputs.chunks
            ],
        )

    def _base_entities(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _base_relations(self, relations: list[ExtractedRelation]) -> list[dict[str, Any]]:
        raise NotImplementedError


class AdaptUniversalGraphForHiRAG(_BaseHierarchicalUniversalGraphAdapter):
    """Adapt universal graph artifacts to HiRAG graph artifacts."""

    name = "adapt_universal_graph_for_hirag"
    target: RAGAdapterTarget = "hirag"

    def __init__(self, config: AdaptUniversalGraphForHiRAGConfig | None = None) -> None:
        self.config = config or AdaptUniversalGraphForHiRAGConfig()

    async def run(self, context: StepContextProtocol) -> None:
        inputs = await self._load_inputs(context)
        await self._persist_graph(
            context,
            inputs,
            nodes=_hirag_nodes(inputs.entities),
            edges=_hirag_edges(inputs.relations),
            target="hirag",
        )
        self._set_base_artifacts(context, inputs)

    def _base_entities(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> list[dict[str, Any]]:
        return _hirag_base_entities(entities, relations)

    def _base_relations(self, relations: list[ExtractedRelation]) -> list[dict[str, Any]]:
        return _hirag_base_relations(relations)


class AdaptUniversalGraphForLeanRAG(_BaseHierarchicalUniversalGraphAdapter):
    """Adapt universal graph artifacts to LeanRAG graph artifacts."""

    name = "adapt_universal_graph_for_leanrag"
    target: RAGAdapterTarget = "leanrag"

    def __init__(self, config: AdaptUniversalGraphForLeanRAGConfig | None = None) -> None:
        self.config = config or AdaptUniversalGraphForLeanRAGConfig()

    @property
    def requirements(self) -> StepRequirements:
        base = super().requirements
        return StepRequirements(
            components=base.components,
            artifacts=base.artifacts | frozenset({self.config.document_token_counts_artifact}),
        )

    async def run(self, context: StepContextProtocol) -> None:
        inputs = await self._load_inputs(context)
        await self._persist_graph(
            context,
            inputs,
            nodes=_leanrag_nodes(inputs.entities),
            edges=_leanrag_edges(inputs.relations),
            target="leanrag",
        )
        document_token_counts = _document_token_counts(
            context.get_artifact(self.config.document_token_counts_artifact)
        )
        self._set_base_artifacts(context, inputs, document_token_counts=document_token_counts)

    def _base_entities(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> list[dict[str, Any]]:
        return _leanrag_base_entities(entities, relations)

    def _base_relations(self, relations: list[ExtractedRelation]) -> list[dict[str, Any]]:
        return _leanrag_base_relations(relations)


class AdaptUniversalGraph:
    """Backward-compatible dispatcher for concrete universal graph adapter steps."""

    name = "adapt_universal_graph"

    def __init__(self, config: AdaptUniversalGraphConfig) -> None:
        self.config = config
        self._delegate = _adapter_from_legacy_config(config)

    @property
    def requirements(self) -> StepRequirements:
        return self._delegate.requirements

    @property
    def capabilities(self) -> StepCapabilities:
        return self._delegate.capabilities

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        return self._delegate.cleanup_plan(artifacts)

    async def run(self, context: StepContextProtocol) -> None:
        await self._delegate.run(context)


def _adapter_from_legacy_config(
    config: AdaptUniversalGraphConfig,
) -> _BaseUniversalGraphAdapter:
    kwargs = {
        "object_store": config.object_store,
        "graph_store": config.graph_store,
        "chunk_keys_artifact": config.chunk_keys_artifact,
        "entity_keys_artifact": config.entity_keys_artifact,
        "relation_keys_artifact": config.relation_keys_artifact,
        "graph_nodes_prefix": config.graph_nodes_prefix,
        "graph_edges_prefix": config.graph_edges_prefix,
        "graph_node_keys_artifact": config.graph_node_keys_artifact,
        "graph_edge_keys_artifact": config.graph_edge_keys_artifact,
        "result_artifact": config.result_artifact,
    }
    if config.target == "graphrag":
        return AdaptUniversalGraphForGraphRAG(AdaptUniversalGraphForGraphRAGConfig(**kwargs))
    if config.target == "lightrag":
        return AdaptUniversalGraphForLightRAG(
            AdaptUniversalGraphForLightRAGConfig(
                **kwargs,
                language_model=config.language_model,
                lightrag_extract_keywords=config.lightrag_extract_keywords,
                temperature=config.temperature,
            )
        )
    if config.target == "hirag":
        return AdaptUniversalGraphForHiRAG(
            AdaptUniversalGraphForHiRAGConfig(
                **kwargs,
                chunks_artifact=config.chunks_artifact,
                base_entities_artifact=config.base_entities_artifact,
                base_relations_artifact=config.base_relations_artifact,
                extraction_trace_artifact=config.extraction_trace_artifact,
            )
        )
    return AdaptUniversalGraphForLeanRAG(
        AdaptUniversalGraphForLeanRAGConfig(
            **kwargs,
            chunks_artifact=config.chunks_artifact,
            base_entities_artifact=config.base_entities_artifact,
            base_relations_artifact=config.base_relations_artifact,
            extraction_trace_artifact=config.extraction_trace_artifact,
            document_token_counts_artifact=config.document_token_counts_artifact,
        )
    )


def _group_entities_by_name(
    entities: list[ExtractedEntity],
) -> dict[str, list[ExtractedEntity]]:
    grouped: dict[str, list[ExtractedEntity]] = defaultdict(list)
    for entity in entities:
        grouped[entity.name].append(entity)
    return grouped


def _group_relations_by_endpoints(
    relations: list[ExtractedRelation],
) -> dict[tuple[str, str], list[ExtractedRelation]]:
    grouped: dict[tuple[str, str], list[ExtractedRelation]] = defaultdict(list)
    for relation in relations:
        source = relation.source_entity_name
        target_name = relation.target_entity_name
        if source and target_name and source != target_name:
            grouped[tuple(sorted((source, target_name)))].append(relation)
    return grouped


def _base_node_properties(
    name: str,
    entities: list[ExtractedEntity],
) -> tuple[str, list[str], dict[str, Any]]:
    entity_type = _most_common(entity.type for entity in entities) or "ENTITY"
    descriptions = _unique(entity.description for entity in entities)
    source_ids = _unique(source_id for entity in entities for source_id in entity.source_chunk_ids)
    properties: dict[str, Any] = {
        "name": name,
        "entity_name": name,
        "entity_type": entity_type,
        "description": GRAPH_FIELD_SEP.join(descriptions),
        "source_id": GRAPH_FIELD_SEP.join(source_ids),
        "source_ids": source_ids,
        "universal_entity_ids": [entity.entity_id for entity in entities],
    }
    return entity_type, source_ids, properties


def _base_edge_properties(
    relations: list[ExtractedRelation],
) -> tuple[list[str], dict[str, Any]] | None:
    descriptions = _unique(relation.description for relation in relations)
    if not descriptions:
        return None
    source_ids = _unique(
        source_id for relation in relations for source_id in relation.source_chunk_ids
    )
    properties: dict[str, Any] = {
        "description": GRAPH_FIELD_SEP.join(descriptions),
        "weight": sum(_relation_weight(relation) for relation in relations) or 1.0,
        "source_id": GRAPH_FIELD_SEP.join(source_ids),
        "source_ids": source_ids,
        "relation_type": _most_common(relation.type for relation in relations) or "RELATED",
        "relation_name": _most_common(relation.name for relation in relations) or "related",
        "universal_relation_ids": [relation.relation_id for relation in relations],
    }
    return source_ids, properties


def _edge_from_properties(source: str, target_name: str, properties: dict[str, Any]) -> GraphEdge:
    return GraphEdge(
        id=_stable_id("edge", source, target_name),
        source_id=source,
        target_id=target_name,
        type="RELATED",
        properties=properties,
    )


def _graphrag_nodes(entities: list[ExtractedEntity]) -> list[GraphNode]:
    nodes: list[GraphNode] = []
    for name, items in _group_entities_by_name(entities).items():
        entity_type, _, properties = _base_node_properties(name, items)
        nodes.append(GraphNode(id=name, labels=("Entity", entity_type), properties=properties))
    return nodes


def _graphrag_edges(relations: list[ExtractedRelation]) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for (source, target_name), items in _group_relations_by_endpoints(relations).items():
        edge_parts = _base_edge_properties(items)
        if edge_parts is not None:
            _, properties = edge_parts
            edges.append(_edge_from_properties(source, target_name, properties))
    return edges


def _lightrag_nodes(
    entities: list[ExtractedEntity],
    *,
    chunk_by_id: Mapping[str, ParsedChunk],
) -> list[GraphNode]:
    nodes: list[GraphNode] = []
    for name, items in _group_entities_by_name(entities).items():
        entity_type, source_ids, properties = _base_node_properties(name, items)
        file_paths = _file_paths_for_source_ids(source_ids, chunk_by_id=chunk_by_id)
        properties.update(
            {
                "file_path": GRAPH_FIELD_SEP.join(file_paths),
                "file_paths": file_paths,
                "extraction_format": "universal",
            }
        )
        nodes.append(GraphNode(id=name, labels=("Entity", entity_type), properties=properties))
    return nodes


async def _lightrag_edges(
    relations: list[ExtractedRelation],
    *,
    chunk_by_id: Mapping[str, ParsedChunk],
    language_model: LanguageModelProtocol | None,
    temperature: float,
) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for (source, target_name), items in _group_relations_by_endpoints(relations).items():
        edge_parts = _base_edge_properties(items)
        if edge_parts is None:
            continue
        source_ids, properties = edge_parts
        file_paths = _file_paths_for_source_ids(source_ids, chunk_by_id=chunk_by_id)
        properties.update(
            {
                "src_id": source,
                "tgt_id": target_name,
                "keywords": await _lightrag_keywords(
                    items,
                    chunk_by_id=chunk_by_id,
                    language_model=language_model,
                    temperature=temperature,
                ),
                "file_path": GRAPH_FIELD_SEP.join(file_paths),
                "file_paths": file_paths,
                "extraction_format": "universal",
            }
        )
        edges.append(_edge_from_properties(source, target_name, properties))
    return edges


def _hirag_nodes(entities: list[ExtractedEntity]) -> list[GraphNode]:
    nodes: list[GraphNode] = []
    for name, items in _group_entities_by_name(entities).items():
        entity_type, _, properties = _base_node_properties(name, items)
        properties.update(
            {
                "raw_entity_type": entity_type,
                "layer": 0,
                "cluster_id": None,
                "cluster_ids": [],
                "is_summary": False,
                "parent_entity_ids": [],
            }
        )
        nodes.append(GraphNode(id=name, labels=("Entity", entity_type), properties=properties))
    return nodes


def _hirag_edges(relations: list[ExtractedRelation]) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for (source, target_name), items in _group_relations_by_endpoints(relations).items():
        edge_parts = _base_edge_properties(items)
        if edge_parts is None:
            continue
        _, properties = edge_parts
        properties.update(
            {
                "order": 1,
                "layer": 0,
                "cluster_id": None,
                "cluster_ids": [],
                "is_summary": False,
                "parent_entity_ids": [],
            }
        )
        edges.append(_edge_from_properties(source, target_name, properties))
    return edges


def _leanrag_nodes(entities: list[ExtractedEntity]) -> list[GraphNode]:
    return _graphrag_nodes(entities)


def _leanrag_edges(relations: list[ExtractedRelation]) -> list[GraphEdge]:
    return _graphrag_edges(relations)


async def _lightrag_keywords(
    relations: list[ExtractedRelation],
    *,
    chunk_by_id: Mapping[str, ParsedChunk],
    language_model: LanguageModelProtocol | None,
    temperature: float,
) -> str:
    fallback = ",".join(
        sorted(
            {
                value
                for relation in relations
                for value in (relation.type, relation.name)
                if value.strip()
            }
        )
    )
    if language_model is None:
        return fallback
    relation = relations[0]
    chunk = chunk_by_id.get(relation.chunk_id)
    try:
        result = await language_model.invoke(
            ModelRequest(
                prompt=LIGHTRAG_RELATION_KEYWORDS_PROMPT.format(
                    relation_json=_json(relation.to_dict()),
                    chunk_text=chunk.text if chunk else "",
                ),
                options=ModelOptions(
                    temperature=temperature,
                    response_format={"type": "json_object"},
                ),
                trace_context={
                    "step": AdaptUniversalGraphForLightRAG.name,
                    "stage": "lightrag_relation_keywords",
                    "relation_id": relation.relation_id,
                },
            )
        )
        payload = result.parsed if result.parsed is not None else json.loads(result.text)
        if isinstance(payload, dict) and isinstance(payload.get("keywords"), str):
            keywords = payload["keywords"].strip()
            return keywords or fallback
    except Exception:
        return fallback
    return fallback


def _file_paths_for_source_ids(
    source_ids: list[str],
    *,
    chunk_by_id: Mapping[str, ParsedChunk],
) -> list[str]:
    return _unique(
        _chunk_file_path(chunk)
        for source_id in source_ids
        if (chunk := chunk_by_id.get(source_id)) is not None
    )


def _chunk_file_path(chunk: ParsedChunk) -> str:
    return chunk.source.name or chunk.source.key or "unknown_source"


def _base_entities(
    entities: list[ExtractedEntity],
    relations: list[ExtractedRelation],
    *,
    target: RAGAdapterTarget,
) -> list[dict[str, Any]]:
    degrees = Counter(
        endpoint
        for relation in relations
        for endpoint in (relation.source_entity_name, relation.target_entity_name)
    )
    grouped: dict[str, list[ExtractedEntity]] = defaultdict(list)
    for entity in entities:
        grouped[entity.name].append(entity)
    records: list[dict[str, Any]] = []
    for name, items in grouped.items():
        source_ids = _unique(source_id for item in items for source_id in item.source_chunk_ids)
        entity_type = _most_common(item.type for item in items) or "ENTITY"
        record = {
            "entity_name": name,
            "entity_type": entity_type,
            "raw_entity_type": entity_type,
            "description": GRAPH_FIELD_SEP.join(_unique(item.description for item in items)),
            "source_id": GRAPH_FIELD_SEP.join(source_ids),
            "source_ids": source_ids,
            "layer": 0,
            "cluster_id": None,
            "is_summary": False,
            "parent_entity_ids": [],
            "properties": {"base_graph_source": "universal_graph_extraction"},
        }
        if target == "leanrag":
            record.update(
                {
                    "degree": int(degrees.get(name, 0)),
                    "parent": "",
                    "level": 0,
                    "is_aggregate": False,
                    "children": [],
                    "description": LEANRAG_FIELD_SEP.join(
                        _unique(item.description for item in items)
                    ),
                }
            )
        records.append(record)
    return records


def _base_relations(
    relations: list[ExtractedRelation],
    *,
    target: RAGAdapterTarget,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[ExtractedRelation]] = defaultdict(list)
    for relation in relations:
        grouped[tuple(sorted((relation.source_entity_name, relation.target_entity_name)))].append(
            relation
        )
    records: list[dict[str, Any]] = []
    for (source, target_name), items in grouped.items():
        source_ids = _unique(source_id for item in items for source_id in item.source_chunk_ids)
        if target == "leanrag":
            records.append(
                {
                    "src_tgt": source,
                    "tgt_src": target_name,
                    "description": LEANRAG_FIELD_SEP.join(
                        _unique(item.description for item in items)
                    ),
                    "weight": sum(_relation_weight(item) for item in items) or 1.0,
                    "level": 0,
                    "source_id": "|".join(source_ids),
                    "source_ids": source_ids,
                    "is_generated": False,
                    "evidence_relation_ids": [item.relation_id for item in items],
                    "properties": {"base_graph_source": "universal_graph_extraction"},
                }
            )
        else:
            records.append(
                {
                    "src_id": source,
                    "tgt_id": target_name,
                    "description": GRAPH_FIELD_SEP.join(
                        _unique(item.description for item in items)
                    ),
                    "weight": sum(_relation_weight(item) for item in items) or 1.0,
                    "order": 1,
                    "source_id": GRAPH_FIELD_SEP.join(source_ids),
                    "source_ids": source_ids,
                    "layer": 0,
                    "cluster_id": None,
                    "is_summary": False,
                    "properties": {"base_graph_source": "universal_graph_extraction"},
                }
            )
    return records


def _hirag_base_entities(
    entities: list[ExtractedEntity],
    relations: list[ExtractedRelation],
) -> list[dict[str, Any]]:
    return _base_entities(entities, relations, target="hirag")


def _hirag_base_relations(relations: list[ExtractedRelation]) -> list[dict[str, Any]]:
    return _base_relations(relations, target="hirag")


def _leanrag_base_entities(
    entities: list[ExtractedEntity],
    relations: list[ExtractedRelation],
) -> list[dict[str, Any]]:
    return _base_entities(entities, relations, target="leanrag")


def _leanrag_base_relations(relations: list[ExtractedRelation]) -> list[dict[str, Any]]:
    return _base_relations(relations, target="leanrag")


def _attach_document_provenance(
    entities: list[dict[str, Any]],
    *,
    chunks: list[Mapping[str, Any]],
    document_token_counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    document_by_source_id: dict[str, str] = {}
    document_name_by_id: dict[str, str] = {}
    for chunk in chunks:
        document_id = str(chunk.get("document_id") or "")
        if not document_id:
            continue
        source_name = str(chunk.get("source_name") or "")
        if source_name and document_id not in document_name_by_id:
            document_name_by_id[document_id] = source_name
        for key_name in ("chunk_id", "hash_code"):
            source_id = str(chunk.get(key_name) or "")
            if source_id:
                document_by_source_id[source_id] = document_id

    records: list[dict[str, Any]] = []
    for entity in entities:
        record = dict(entity)
        documents = _unique(
            document_by_source_id[source_id]
            for source_id in _record_source_ids(record)
            if source_id in document_by_source_id
        )
        document_tokens = {
            document_id: int(document_token_counts.get(document_id, 0))
            for document_id in documents
        }
        document_names = {
            document_id: document_name_by_id.get(document_id, "")
            for document_id in documents
        }
        document_details = [
            {
                "document_id": document_id,
                "document_name": document_names.get(document_id, ""),
                "document_token_count": document_tokens.get(document_id, 0),
            }
            for document_id in documents
        ]
        record.update(
            {
                "documents": documents,
                "document_names": document_names,
                "document_details": document_details,
                "document_tokens": document_tokens,
                "document_token_count": sum(document_tokens.values()),
            }
        )
        properties = dict(record.get("properties") or {})
        properties.update(
            {
                "documents": documents,
                "document_names": document_names,
                "document_details": document_details,
                "document_tokens": document_tokens,
                "document_token_count": record["document_token_count"],
            }
        )
        record["properties"] = properties
        records.append(record)
    return records


def _record_source_ids(record: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    source_ids = record.get("source_ids", ())
    if isinstance(source_ids, str):
        values.extend(source_ids.split("|"))
    elif isinstance(source_ids, list | tuple | set):
        values.extend(str(item) for item in source_ids)
    source_id = str(record.get("source_id") or "")
    if source_id:
        values.extend(source_id.split("|"))
    return _unique(value.strip() for value in values)


def _document_token_counts(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("document_token_counts must be a mapping")
    return {str(key): int(count) for key, count in value.items()}


def _chunk_record(chunk: ParsedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "hash_code": compute_leanrag_hash(chunk.text),
        "document_id": chunk.document_id,
        "text": chunk.text,
        "content": chunk.text,
        "source_key": chunk.source.key,
        "source_name": chunk.source.name,
        "source_file_type": chunk.source.file_type,
        "page_index": chunk.page_index,
        "chunk_index": chunk.chunk_index,
        "chunk_order_index": chunk.chunk_index,
        "token_start": chunk.token_start,
        "token_end": chunk.token_end,
        "token_count": chunk.token_end - chunk.token_start,
        "metadata": {
            "parent_chunk_ids": list(chunk.parent_chunk_ids),
            "leanrag_hash_code": compute_leanrag_hash(chunk.text),
        },
    }


async def _put_graph_node(
    object_store: ObjectStoreProtocol,
    prefix: str,
    node: GraphNode,
) -> str:
    key = join_object_key(prefix, f"{_safe_key(node.id)}.json")
    await object_store.put(key, _json_bytes(_graph_node_dict(node)))
    return key


async def _put_graph_edge(
    object_store: ObjectStoreProtocol,
    prefix: str,
    edge: GraphEdge,
) -> str:
    key = join_object_key(prefix, f"{_safe_key(edge.id)}.json")
    await object_store.put(key, _json_bytes(_graph_edge_dict(edge)))
    return key


def _graph_node_dict(node: GraphNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "labels": list(node.labels),
        "properties": dict(node.properties),
    }


def _graph_edge_dict(edge: GraphEdge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "type": edge.type,
        "properties": dict(edge.properties),
    }
