"""Build HiRAG SQL tables, vector indexes, graph store data, and community reports."""

from __future__ import annotations

import asyncio
import html
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Mapping

from heta_framework.common.models import EmbeddingRequest, ModelOptions, ModelRequest
from heta_framework.common.models.protocols import EmbeddingModelProtocol, LanguageModelProtocol
from heta_framework.common.stores.graph import GraphEdge, GraphNode, GraphStoreProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.common.stores.sql import SQLStoreProtocol
from heta_framework.common.stores.vector import VectorCollectionConfig, VectorRecord, VectorStoreProtocol
from heta_framework.kb.cleanup import CleanupTarget, StepCleanupPlan, object_key_targets
from heta_framework.kb.search import SearchAsset
from heta_framework.kb.steps.extract_hirag_graph import HIRAG_PROMPTS
from heta_framework.kb.steps.graph_storage import batches, compact_json, validate_identifier
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, model_ref, store_ref


@dataclass(frozen=True)
class HiRAGTableNames:
    """SQL table names used by HiRAG storage."""

    entities: str = "hi_rag_entities"
    relations: str = "hi_rag_relations"
    communities: str = "hi_rag_communities"
    chunks: str = "hi_rag_chunks"

    def __post_init__(self) -> None:
        validate_identifier(self.entities, field_name="table_names.entities")
        validate_identifier(self.relations, field_name="table_names.relations")
        validate_identifier(self.communities, field_name="table_names.communities")
        validate_identifier(self.chunks, field_name="table_names.chunks")


@dataclass(frozen=True)
class HiRAGVectorCollections:
    """Vector collection names used by HiRAG storage."""

    entities: str = "hi_rag_entities"

    def __post_init__(self) -> None:
        validate_identifier(self.entities, field_name="vector_collections.entities")


@dataclass(frozen=True)
class BuildHiRAGGraphConfig:
    """Configuration for BuildHiRAGGraph."""

    table_names: HiRAGTableNames = field(default_factory=HiRAGTableNames)
    vector_collections: HiRAGVectorCollections = field(default_factory=HiRAGVectorCollections)
    graph_node_keys_artifact: str = "hi_rag_graph_node_keys"
    graph_edge_keys_artifact: str = "hi_rag_graph_edge_keys"
    chunks_artifact: str = "hi_rag_chunks"
    community_reports_artifact: str = "hi_rag_community_reports"
    community_report_keys_artifact: str = "hi_rag_community_report_keys"
    result_artifact: str = "build_hi_rag_graph_result"
    community_reports_prefix: str = "hi_rag/community_reports"
    vector_metric: str = "cosine"
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    batch_size: int = 128
    report_max_output_tokens: int = 800
    temperature: float = 0.0
    object_store: str | None = None
    graph_store: str | None = None
    sql_store: str | None = None
    vector_store: str | None = None
    embedding_model: str | None = None
    language_model: str | None = None
    prompts: Mapping[str, Any] = field(default_factory=lambda: dict(HIRAG_PROMPTS))

    def __post_init__(self) -> None:
        validate_object_prefix(self.community_reports_prefix)
        if self.vector_metric not in {"cosine", "dot", "l2"}:
            raise ValueError("vector_metric must be one of: cosine, dot, l2")
        if self.graph_cluster_algorithm not in {"leiden", "connected_components"}:
            raise ValueError("graph_cluster_algorithm must be one of: leiden, connected_components")
        if self.max_graph_cluster_size <= 0:
            raise ValueError("max_graph_cluster_size must be greater than zero")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.report_max_output_tokens <= 0:
            raise ValueError("report_max_output_tokens must be greater than zero")
        for name in (
            self.graph_node_keys_artifact,
            self.graph_edge_keys_artifact,
            self.chunks_artifact,
            self.community_reports_artifact,
            self.community_report_keys_artifact,
            self.result_artifact,
        ):
            if name.strip() == "":
                raise ValueError("artifact names must not be empty")


@dataclass(frozen=True)
class BuildHiRAGGraphResult:
    """Artifacts produced by BuildHiRAGGraph."""

    entity_count: int
    relation_count: int
    community_count: int
    chunk_count: int
    entity_vector_count: int
    vector_dimension: int


class BuildHiRAGGraph:
    """Write HiRAG graph artifacts into HetaFramework stores."""

    name = "build_hirag_graph"

    def __init__(self, config: BuildHiRAGGraphConfig | None = None) -> None:
        self.config = config or BuildHiRAGGraphConfig()

    @property
    def requirements(self) -> StepRequirements:
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("objects", self.config.object_store),
                    store_ref("graph", self.config.graph_store),
                    store_ref("sql", self.config.sql_store),
                    store_ref("vector", self.config.vector_store),
                    model_ref("embedding", self.config.embedding_model),
                    model_ref("language", self.config.language_model),
                }
            ),
            artifacts=frozenset(
                {
                    self.config.graph_node_keys_artifact,
                    self.config.graph_edge_keys_artifact,
                    self.config.chunks_artifact,
                }
            ),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        sql_store_ref = store_ref("sql", self.config.sql_store)
        vector_store_ref = store_ref("vector", self.config.vector_store)
        return StepCapabilities(
            artifacts=frozenset(
                {
                    self.config.result_artifact,
                    self.config.community_reports_artifact,
                    self.config.community_report_keys_artifact,
                }
            ),
            queries=frozenset(
                {
                    "hi_rag_query",
                    "hi_rag_nobridge_query",
                    "hi_rag_local_query",
                    "hi_rag_global_query",
                    "hi_rag_bridge_query",
                }
            ),
            search_assets=(
                SearchAsset(
                    kind="hi_rag_tables",
                    name=self.config.table_names.entities,
                    store=sql_store_ref.key,
                    metadata={
                        "entities_table": self.config.table_names.entities,
                        "relations_table": self.config.table_names.relations,
                        "communities_table": self.config.table_names.communities,
                        "chunks_table": self.config.table_names.chunks,
                    },
                ),
                SearchAsset(
                    kind="hi_rag_vector_index",
                    name=self.config.vector_collections.entities,
                    store=vector_store_ref.key,
                    metadata={"entity_collection": self.config.vector_collections.entities},
                ),
            ),
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        sql_store_ref = store_ref("sql", self.config.sql_store).key
        vector_store_ref = store_ref("vector", self.config.vector_store).key
        return StepCleanupPlan(
            (
                CleanupTarget("sql_table", self.config.table_names.entities, sql_store_ref),
                CleanupTarget("sql_table", self.config.table_names.relations, sql_store_ref),
                CleanupTarget("sql_table", self.config.table_names.communities, sql_store_ref),
                CleanupTarget("sql_table", self.config.table_names.chunks, sql_store_ref),
                CleanupTarget("vector_collection", self.config.vector_collections.entities, vector_store_ref),
            )
            + tuple(
                object_key_targets(
                    artifacts,
                    self.config.community_report_keys_artifact,
                    component=store_ref("objects", self.config.object_store).key,
                )
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        graph_store = _require_graph_store(
            context.get_component(store_ref("graph", self.config.graph_store).key)
        )
        sql_store = _require_sql_store(
            context.get_component(store_ref("sql", self.config.sql_store).key)
        )
        vector_store = _require_vector_store(
            context.get_component(store_ref("vector", self.config.vector_store).key)
        )
        embedding_model = _require_embedding_model(
            context.get_component(model_ref("embedding", self.config.embedding_model).key)
        )
        language_model = _require_language_model(
            context.get_component(model_ref("language", self.config.language_model).key)
        )

        node_keys = tuple(context.get_artifact(self.config.graph_node_keys_artifact))
        edge_keys = tuple(context.get_artifact(self.config.graph_edge_keys_artifact))
        chunks = list(context.get_artifact(self.config.chunks_artifact))
        nodes = [json.loads((await object_store.get(key)).decode("utf-8")) for key in node_keys]
        edges = [json.loads((await object_store.get(key)).decode("utf-8")) for key in edge_keys]
        edges = _filter_edges_with_known_endpoints(nodes, edges)

        await _upsert_graph_store(graph_store, nodes, edges)
        reports = await _generate_community_reports(
            nodes,
            edges,
            language_model=language_model,
            config=self.config,
        )
        report_keys = tuple(
            [
                await _put_community_report(object_store, self.config, report)
                for report in reports
            ]
        )

        node_rows = [_entity_row(node) for node in nodes]
        edge_rows = [_relation_row(edge) for edge in edges]
        community_rows = [_community_row(report) for report in reports]
        chunk_rows = [_chunk_row(chunk) for chunk in chunks]
        vectors = await _embed_entities(
            embedding_model,
            nodes,
            batch_size=self.config.batch_size,
        )
        vector_dimension = len(vectors[0].vector) if vectors else 0

        async with sql_store.transaction() as tx:
            await _ensure_tables(tx, self.config.table_names)
            for batch in batches(node_rows, self.config.batch_size):
                await _upsert_entity_rows(tx, self.config.table_names.entities, batch)
            for batch in batches(edge_rows, self.config.batch_size):
                await _upsert_relation_rows(tx, self.config.table_names.relations, batch)
            for batch in batches(community_rows, self.config.batch_size):
                await _upsert_community_rows(tx, self.config.table_names.communities, batch)
            for batch in batches(chunk_rows, self.config.batch_size):
                await _upsert_chunk_rows(tx, self.config.table_names.chunks, batch)

        if vectors:
            await vector_store.create_collection(
                VectorCollectionConfig(
                    name=self.config.vector_collections.entities,
                    dimension=vector_dimension,
                    metric=self.config.vector_metric,  # type: ignore[arg-type]
                )
            )
            for batch in batches(vectors, self.config.batch_size):
                await vector_store.upsert(self.config.vector_collections.entities, batch)

        result = BuildHiRAGGraphResult(
            entity_count=len(nodes),
            relation_count=len(edges),
            community_count=len(reports),
            chunk_count=len(chunks),
            entity_vector_count=len(vectors),
            vector_dimension=vector_dimension,
        )
        context.set_artifact(self.config.result_artifact, result)
        context.set_artifact(self.config.community_reports_artifact, reports)
        context.set_artifact(self.config.community_report_keys_artifact, report_keys)


class HiRAGGraphIndexAdapter:
    """Graph helper used by HiRAG query code when only indexed rows are available."""

    def __init__(self, entities: list[dict[str, Any]], relations: list[dict[str, Any]]) -> None:
        self.entities = {str(row["entity_id"]): row for row in entities}
        self.relations = {str(row["relation_id"]): row for row in relations}
        self._adjacency: defaultdict[str, set[str]] = defaultdict(set)
        self._edge_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for row in relations:
            source = str(row["source_entity_id"])
            target = str(row["target_entity_id"])
            self._adjacency[source].add(target)
            self._adjacency[target].add(source)
            self._edge_by_pair[tuple(sorted((source, target)))] = row

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        return self.entities.get(node_id)

    def get_edge(self, source_id: str, target_id: str) -> dict[str, Any] | None:
        return self._edge_by_pair.get(tuple(sorted((source_id, target_id))))

    def get_node_edges(self, node_id: str) -> list[tuple[str, str]]:
        return [(node_id, target) for target in sorted(self._adjacency.get(node_id, ()))]

    def node_degree(self, node_id: str) -> int:
        return len(self._adjacency.get(node_id, ()))

    def shortest_path(self, source: str, target: str) -> list[str]:
        if source == target:
            return [source]
        queue: deque[tuple[str, list[str]]] = deque([(source, [source])])
        seen = {source}
        while queue:
            current, path = queue.popleft()
            for neighbor in sorted(self._adjacency.get(current, ())):
                if neighbor in seen:
                    continue
                if neighbor == target:
                    return [*path, neighbor]
                seen.add(neighbor)
                queue.append((neighbor, [*path, neighbor]))
        return [source, target]

    def subgraph_edges(self, nodes: list[str]) -> list[dict[str, Any]]:
        node_set = set(nodes)
        return [
            row
            for row in self.relations.values()
            if row["source_entity_id"] in node_set and row["target_entity_id"] in node_set
        ]


async def _ensure_tables(tx: SQLStoreProtocol, tables: HiRAGTableNames) -> None:
    await tx.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.entities} (
            entity_id TEXT PRIMARY KEY,
            entity_name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            raw_entity_type TEXT,
            description TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_ids TEXT NOT NULL,
            layer INTEGER NOT NULL,
            cluster_id TEXT,
            is_summary INTEGER NOT NULL,
            parent_entity_ids TEXT NOT NULL,
            properties TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await tx.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.relations} (
            relation_id TEXT PRIMARY KEY,
            source_entity_id TEXT NOT NULL,
            target_entity_id TEXT NOT NULL,
            description TEXT NOT NULL,
            weight REAL NOT NULL,
            "order" INTEGER NOT NULL,
            source_id TEXT NOT NULL,
            source_ids TEXT NOT NULL,
            layer INTEGER NOT NULL,
            cluster_id TEXT,
            is_summary INTEGER NOT NULL,
            properties TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await tx.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.communities} (
            community_id TEXT PRIMARY KEY,
            level INTEGER NOT NULL,
            title TEXT NOT NULL,
            report TEXT NOT NULL,
            report_json TEXT NOT NULL,
            nodes TEXT NOT NULL,
            edges TEXT NOT NULL,
            chunk_ids TEXT NOT NULL,
            occurrence REAL NOT NULL,
            sub_communities TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await tx.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.chunks} (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            content TEXT NOT NULL,
            source_key TEXT,
            source_name TEXT,
            chunk_order_index INTEGER NOT NULL,
            token_count INTEGER,
            full_doc_id TEXT,
            metadata TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


async def _upsert_entity_rows(tx: SQLStoreProtocol, table: str, rows: list[dict[str, object]]) -> None:
    for row in rows:
        await tx.execute(
            f"""
            INSERT INTO {table}
            (
                entity_id, entity_name, entity_type, raw_entity_type, description,
                source_id, source_ids, layer, cluster_id, is_summary,
                parent_entity_ids, properties, updated_at
            )
            VALUES (
                :entity_id, :entity_name, :entity_type, :raw_entity_type, :description,
                :source_id, :source_ids, :layer, :cluster_id, :is_summary,
                :parent_entity_ids, :properties, CURRENT_TIMESTAMP
            )
            ON CONFLICT (entity_id) DO UPDATE SET
                entity_name = excluded.entity_name,
                entity_type = excluded.entity_type,
                raw_entity_type = excluded.raw_entity_type,
                description = excluded.description,
                source_id = excluded.source_id,
                source_ids = excluded.source_ids,
                layer = excluded.layer,
                cluster_id = excluded.cluster_id,
                is_summary = excluded.is_summary,
                parent_entity_ids = excluded.parent_entity_ids,
                properties = excluded.properties,
                updated_at = CURRENT_TIMESTAMP
            """,
            row,
        )


async def _upsert_relation_rows(tx: SQLStoreProtocol, table: str, rows: list[dict[str, object]]) -> None:
    for row in rows:
        await tx.execute(
            f"""
            INSERT INTO {table}
            (
                relation_id, source_entity_id, target_entity_id, description, weight,
                "order", source_id, source_ids, layer, cluster_id, is_summary,
                properties, updated_at
            )
            VALUES (
                :relation_id, :source_entity_id, :target_entity_id, :description, :weight,
                :order, :source_id, :source_ids, :layer, :cluster_id, :is_summary,
                :properties, CURRENT_TIMESTAMP
            )
            ON CONFLICT (relation_id) DO UPDATE SET
                source_entity_id = excluded.source_entity_id,
                target_entity_id = excluded.target_entity_id,
                description = excluded.description,
                weight = excluded.weight,
                "order" = excluded."order",
                source_id = excluded.source_id,
                source_ids = excluded.source_ids,
                layer = excluded.layer,
                cluster_id = excluded.cluster_id,
                is_summary = excluded.is_summary,
                properties = excluded.properties,
                updated_at = CURRENT_TIMESTAMP
            """,
            row,
        )


async def _upsert_community_rows(tx: SQLStoreProtocol, table: str, rows: list[dict[str, object]]) -> None:
    for row in rows:
        await tx.execute(
            f"""
            INSERT INTO {table}
            (
                community_id, level, title, report, report_json, nodes, edges,
                chunk_ids, occurrence, sub_communities, updated_at
            )
            VALUES (
                :community_id, :level, :title, :report, :report_json, :nodes, :edges,
                :chunk_ids, :occurrence, :sub_communities, CURRENT_TIMESTAMP
            )
            ON CONFLICT (community_id) DO UPDATE SET
                level = excluded.level,
                title = excluded.title,
                report = excluded.report,
                report_json = excluded.report_json,
                nodes = excluded.nodes,
                edges = excluded.edges,
                chunk_ids = excluded.chunk_ids,
                occurrence = excluded.occurrence,
                sub_communities = excluded.sub_communities,
                updated_at = CURRENT_TIMESTAMP
            """,
            row,
        )


async def _upsert_chunk_rows(tx: SQLStoreProtocol, table: str, rows: list[dict[str, object]]) -> None:
    for row in rows:
        await tx.execute(
            f"""
            INSERT INTO {table}
            (
                chunk_id, document_id, content, source_key, source_name,
                chunk_order_index, token_count, full_doc_id, metadata, updated_at
            )
            VALUES (
                :chunk_id, :document_id, :content, :source_key, :source_name,
                :chunk_order_index, :token_count, :full_doc_id, :metadata, CURRENT_TIMESTAMP
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                document_id = excluded.document_id,
                content = excluded.content,
                source_key = excluded.source_key,
                source_name = excluded.source_name,
                chunk_order_index = excluded.chunk_order_index,
                token_count = excluded.token_count,
                full_doc_id = excluded.full_doc_id,
                metadata = excluded.metadata,
                updated_at = CURRENT_TIMESTAMP
            """,
            row,
        )


async def _embed_entities(
    embedding_model: EmbeddingModelProtocol,
    nodes: list[dict[str, Any]],
    *,
    batch_size: int,
) -> list[VectorRecord]:
    records: list[VectorRecord] = []
    for batch in batches(nodes, batch_size):
        texts = [_entity_vector_text(node) for node in batch]
        result = await embedding_model.embed(
            EmbeddingRequest(
                texts=texts,
                trace_context={"step": BuildHiRAGGraph.name, "purpose": "hi_rag_entity_index"},
            )
        )
        if len(result.vectors) != len(batch):
            raise ValueError("entity embedding result count must match batch size")
        for node, text, vector in zip(batch, texts, result.vectors, strict=True):
            properties = dict(node.get("properties") or {})
            records.append(
                VectorRecord(
                    id=str(node["id"]),
                    vector=[float(value) for value in vector],
                    text=text,
                    metadata={
                        "fact_type": "hi_rag_entity",
                        "entity_name": str(properties.get("name") or node["id"]),
                        "entity_type": str(properties.get("entity_type") or "ENTITY"),
                        "source_ids": _list_value(properties.get("source_ids")),
                        "layer": int(properties.get("layer") or 0),
                        "cluster_id": properties.get("cluster_id"),
                        "is_summary": bool(properties.get("is_summary")),
                        "embedding_model": result.model_name or embedding_model.model_name,
                    },
                )
            )
    return records


async def _upsert_graph_store(
    graph_store: GraphStoreProtocol,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> None:
    await graph_store.upsert_nodes(
        [
            GraphNode(
                id=str(node["id"]),
                labels=tuple(str(label) for label in node.get("labels") or ("Entity",)),
                properties=dict(node.get("properties") or {}),
            )
            for node in nodes
        ]
    )
    await graph_store.upsert_edges(
        [
            GraphEdge(
                id=str(edge["id"]),
                source_id=str(edge["source_id"]),
                target_id=str(edge["target_id"]),
                type=str(edge.get("type") or "RELATED"),
                properties=dict(edge.get("properties") or {}),
            )
            for edge in edges
        ]
    )


async def _generate_community_reports(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    language_model: LanguageModelProtocol,
    config: BuildHiRAGGraphConfig,
) -> list[dict[str, Any]]:
    communities = _community_schema(nodes, edges, config)
    return await asyncio.gather(
        *(
            _generate_single_community_report(
                community,
                nodes,
                edges,
                language_model=language_model,
                config=config,
            )
            for community in communities
        )
    )


async def _generate_single_community_report(
    community: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    language_model: LanguageModelProtocol,
    config: BuildHiRAGGraphConfig,
) -> dict[str, Any]:
    prompt = str(
        config.prompts["community_report"].format(
            input_text=_community_context(community, nodes, edges)
        )
    )
    result = await language_model.invoke(
        ModelRequest(
            prompt=prompt,
            options=ModelOptions(
                temperature=config.temperature,
                max_output_tokens=config.report_max_output_tokens,
                response_format={"type": "json_object"},
            ),
            trace_context={
                "step": BuildHiRAGGraph.name,
                "stage": "community_report",
                "community_id": community["community_id"],
            },
        )
    )
    report_json = _parse_json_object(result.text)
    if not report_json:
        report_json = {"title": community["title"], "summary": result.text, "findings": []}
    return {
        **community,
        "title": str(report_json.get("title") or community["title"]),
        "report_json": report_json,
        "report": _community_report_json_to_str(report_json),
    }


def _community_schema(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    config: BuildHiRAGGraphConfig,
) -> list[dict[str, Any]]:
    if config.graph_cluster_algorithm == "connected_components":
        return _connected_component_communities(nodes, edges)
    return _leiden_communities(nodes, edges, config)


def _leiden_communities(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    config: BuildHiRAGGraphConfig,
) -> list[dict[str, Any]]:
    try:
        import networkx as nx
        from graspologic.partition import hierarchical_leiden
        from graspologic.utils import largest_connected_component
    except Exception:
        return _connected_component_communities(nodes, edges)

    graph = nx.Graph()
    for node in nodes:
        graph.add_node(str(node["id"]), **dict(node.get("properties") or {}))
    for edge in edges:
        properties = dict(edge.get("properties") or {})
        graph.add_edge(
            str(edge["source_id"]),
            str(edge["target_id"]),
            **properties,
        )

    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        return _connected_component_communities(nodes, edges)

    try:
        stable_graph = _stable_largest_connected_component(graph, largest_connected_component, nx)
        partitions = hierarchical_leiden(
            stable_graph,
            max_cluster_size=config.max_graph_cluster_size,
            random_seed=config.graph_cluster_seed,
        )
    except Exception:
        return _connected_component_communities(nodes, edges)

    node_clusters: defaultdict[str, list[dict[str, int]]] = defaultdict(list)
    levels: defaultdict[int, set[int]] = defaultdict(set)
    for partition in partitions:
        level = int(partition.level)
        cluster_id = int(partition.cluster)
        node_id = str(partition.node)
        node_clusters[node_id].append({"level": level, "cluster": cluster_id})
        levels[level].add(cluster_id)

    if not node_clusters:
        return _connected_component_communities(nodes, edges)
    return _cluster_schema_from_node_clusters(nodes, edges, node_clusters, levels)


def _stable_largest_connected_component(graph: Any, largest_connected_component: Any, nx: Any) -> Any:
    graph = graph.copy()
    graph = largest_connected_component(graph)
    node_mapping = {node: html.unescape(str(node).upper().strip()) for node in graph.nodes()}
    graph = nx.relabel_nodes(graph, node_mapping)
    return _stabilize_graph(graph, nx)


def _stabilize_graph(graph: Any, nx: Any) -> Any:
    fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
    fixed_graph.add_nodes_from(sorted(graph.nodes(data=True), key=lambda item: item[0]))

    graph_edges = list(graph.edges(data=True))
    if not graph.is_directed():
        graph_edges = [
            (target, source, data) if source > target else (source, target, data)
            for source, target, data in graph_edges
        ]
    fixed_graph.add_edges_from(sorted(graph_edges, key=lambda item: f"{item[0]} -> {item[1]}"))
    return fixed_graph


def _cluster_schema_from_node_clusters(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    node_clusters: defaultdict[str, list[dict[str, int]]],
    levels: defaultdict[int, set[int]],
) -> list[dict[str, Any]]:
    normalized_node_by_id = {
        html.unescape(str(node["id"]).upper().strip()): node
        for node in nodes
    }
    communities: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "level": 0,
            "title": "",
            "nodes": set(),
            "edges": set(),
            "chunk_ids": set(),
            "occurrence": 0.0,
            "sub_communities": [],
        }
    )
    max_chunk_count = 1

    for node_id, clusters in node_clusters.items():
        node_edges = [
            edge
            for edge in edges
            if _normalized_entity_id(edge["source_id"]) == node_id
            or _normalized_entity_id(edge["target_id"]) == node_id
        ]
        node = normalized_node_by_id.get(node_id)
        for cluster_info in clusters:
            community_id = str(cluster_info["cluster"])
            community = communities[community_id]
            community["level"] = int(cluster_info["level"])
            community["title"] = f"Cluster {community_id}"
            community["nodes"].add(node_id)
            community["edges"].update(
                tuple(sorted((_normalized_entity_id(edge["source_id"]), _normalized_entity_id(edge["target_id"]))))
                for edge in node_edges
            )
            if node is not None:
                community["chunk_ids"].update(_node_chunk_ids(node))
            max_chunk_count = max(max_chunk_count, len(community["chunk_ids"]))

    ordered_levels = sorted(levels)
    for index, current_level in enumerate(ordered_levels[:-1]):
        next_level = ordered_levels[index + 1]
        for community_id in levels[current_level]:
            current = communities[str(community_id)]
            current["sub_communities"] = [
                str(child_id)
                for child_id in levels[next_level]
                if communities[str(child_id)]["nodes"].issubset(current["nodes"])
            ]

    return [
        {
            "community_id": community_id,
            "level": data["level"],
            "title": data["title"],
            "nodes": sorted(data["nodes"]),
            "edges": [list(edge) for edge in sorted(data["edges"])],
            "chunk_ids": sorted(data["chunk_ids"]),
            "occurrence": len(data["chunk_ids"]) / max_chunk_count,
            "sub_communities": list(data["sub_communities"]),
        }
        for community_id, data in sorted(communities.items(), key=lambda item: (item[1]["level"], item[0]))
    ]


def _connected_component_communities(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node_ids = sorted(str(node["id"]) for node in nodes)
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    for node_id in node_ids:
        adjacency[node_id]
    for edge in edges:
        source = str(edge["source_id"])
        target = str(edge["target_id"])
        adjacency[source].add(target)
        adjacency[target].add(source)

    node_by_id = {str(node["id"]): node for node in nodes}
    visited: set[str] = set()
    communities: list[dict[str, Any]] = []
    raw_components: list[tuple[list[str], list[list[str]], list[str]]] = []
    max_chunk_count = 1
    for node_id in node_ids:
        if node_id in visited:
            continue
        queue = deque([node_id])
        component: set[str] = set()
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            queue.extend(sorted(adjacency[current] - visited))
        component_edges = [
            [str(edge["source_id"]), str(edge["target_id"])]
            for edge in edges
            if str(edge["source_id"]) in component and str(edge["target_id"]) in component
        ]
        chunk_ids = sorted(
            {
                source_id
                for nid in component
                for source_id in _list_value(dict(node_by_id[nid].get("properties") or {}).get("source_ids"))
            }
        )
        max_chunk_count = max(max_chunk_count, len(chunk_ids))
        raw_components.append((sorted(component), component_edges, chunk_ids))

    for index, (component_nodes, component_edges, chunk_ids) in enumerate(raw_components):
        communities.append(
            {
                "community_id": f"community_{index}",
                "level": 0,
                "title": f"Community community_{index}",
                "nodes": component_nodes,
                "edges": component_edges,
                "chunk_ids": chunk_ids,
                "occurrence": len(chunk_ids) / max_chunk_count,
                "sub_communities": [],
            }
        )
    return communities


def _normalized_entity_id(value: Any) -> str:
    return html.unescape(str(value).upper().strip())


def _node_chunk_ids(node: dict[str, Any]) -> list[str]:
    properties = dict(node.get("properties") or {})
    return sorted(set(_list_value(properties.get("source_id")) + _list_value(properties.get("source_ids"))))


def _filter_edges_with_known_endpoints(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    node_ids = {str(node["id"]) for node in nodes}
    return [
        edge
        for edge in edges
        if str(edge.get("source_id")) in node_ids and str(edge.get("target_id")) in node_ids
    ]


def _community_context(community: dict[str, Any], nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    node_by_id = {str(node["id"]): node for node in nodes}
    normalized_node_by_id = {_normalized_entity_id(node["id"]): node for node in nodes}
    community_nodes = [
        node
        for node_id in community["nodes"]
        if (node := node_by_id.get(str(node_id)) or normalized_node_by_id.get(_normalized_entity_id(node_id)))
        is not None
    ]
    community_edge_keys = {
        tuple(sorted((_normalized_entity_id(edge[0]), _normalized_entity_id(edge[1]))))
        for edge in community["edges"]
    }
    community_edges = [
        edge
        for edge in edges
        if tuple(sorted((_normalized_entity_id(edge["source_id"]), _normalized_entity_id(edge["target_id"]))))
        in community_edge_keys
    ]
    node_lines = ["id,entity,type,description,degree"]
    degrees = CounterLike.from_edges(community["edges"])
    for index, node in enumerate(community_nodes):
        properties = dict(node.get("properties") or {})
        node_lines.append(
            ",".join(
                [
                    str(index),
                    _csv_cell(str(properties.get("name") or node["id"])),
                    _csv_cell(str(properties.get("entity_type") or "UNKNOWN")),
                    _csv_cell(str(properties.get("description") or "")),
                    str(degrees.get(_normalized_entity_id(node["id"]), 0)),
                ]
            )
        )
    edge_lines = ["id,source,target,description,rank"]
    for index, edge in enumerate(community_edges):
        properties = dict(edge.get("properties") or {})
        edge_lines.append(
            ",".join(
                [
                    str(index),
                    _csv_cell(str(edge["source_id"])),
                    _csv_cell(str(edge["target_id"])),
                    _csv_cell(str(properties.get("description") or "")),
                    str(
                        degrees.get(_normalized_entity_id(edge["source_id"]), 0)
                        + degrees.get(_normalized_entity_id(edge["target_id"]), 0)
                    ),
                ]
            )
        )
    return "-----Reports-----\n```csv\n\n```\n-----Entities-----\n```csv\n" + "\n".join(node_lines) + "\n```\n-----Relationships-----\n```csv\n" + "\n".join(edge_lines) + "\n```"


class CounterLike(defaultdict[str, int]):
    @classmethod
    def from_edges(cls, edges: list[list[str]]) -> "CounterLike":
        counter = cls(int)
        for source, target in edges:
            counter[_normalized_entity_id(source)] += 1
            counter[_normalized_entity_id(target)] += 1
        return counter


def _community_report_json_to_str(report_json: dict[str, Any]) -> str:
    title = report_json.get("title", "Report")
    summary = report_json.get("summary", "")
    findings = report_json.get("findings", [])
    sections = []
    for finding in findings:
        if isinstance(finding, str):
            sections.append(f"## {finding}\n")
        elif isinstance(finding, dict):
            sections.append(f"## {finding.get('summary', '')}\n\n{finding.get('explanation', '')}")
    return f"# {title}\n\n{summary}\n\n" + "\n\n".join(sections)


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


async def _put_community_report(
    object_store: ObjectStoreProtocol,
    config: BuildHiRAGGraphConfig,
    report: dict[str, Any],
) -> str:
    key = join_object_key(config.community_reports_prefix, f"{report['community_id']}.json")
    await object_store.put(key, json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return key


def _entity_row(node: dict[str, Any]) -> dict[str, object]:
    properties = dict(node.get("properties") or {})
    source_ids = _list_value(properties.get("source_ids"))
    return {
        "entity_id": str(node["id"]),
        "entity_name": str(properties.get("name") or node["id"]),
        "entity_type": str(properties.get("entity_type") or "ENTITY"),
        "raw_entity_type": properties.get("raw_entity_type"),
        "description": str(properties.get("description") or ""),
        "source_id": str(properties.get("source_id") or ""),
        "source_ids": compact_json(source_ids),
        "layer": int(properties.get("layer") or 0),
        "cluster_id": properties.get("cluster_id"),
        "is_summary": 1 if properties.get("is_summary") else 0,
        "parent_entity_ids": compact_json(_list_value(properties.get("parent_entity_ids"))),
        "properties": compact_json(properties),
    }


def _relation_row(edge: dict[str, Any]) -> dict[str, object]:
    properties = dict(edge.get("properties") or {})
    source_ids = _list_value(properties.get("source_ids"))
    return {
        "relation_id": str(edge["id"]),
        "source_entity_id": str(edge["source_id"]),
        "target_entity_id": str(edge["target_id"]),
        "description": str(properties.get("description") or ""),
        "weight": float(properties.get("weight") or 0.0),
        "order": int(properties.get("order") or 1),
        "source_id": str(properties.get("source_id") or ""),
        "source_ids": compact_json(source_ids),
        "layer": int(properties.get("layer") or 0),
        "cluster_id": properties.get("cluster_id"),
        "is_summary": 1 if properties.get("is_summary") else 0,
        "properties": compact_json(properties),
    }


def _community_row(report: dict[str, Any]) -> dict[str, object]:
    return {
        "community_id": str(report["community_id"]),
        "level": int(report["level"]),
        "title": str(report["title"]),
        "report": str(report["report"]),
        "report_json": compact_json(report["report_json"]),
        "nodes": compact_json(report["nodes"]),
        "edges": compact_json(report["edges"]),
        "chunk_ids": compact_json(report["chunk_ids"]),
        "occurrence": float(report["occurrence"]),
        "sub_communities": compact_json(report["sub_communities"]),
    }


def _chunk_row(chunk: Mapping[str, Any]) -> dict[str, object]:
    return {
        "chunk_id": str(chunk["chunk_id"]),
        "document_id": str(chunk.get("document_id") or chunk.get("full_doc_id") or ""),
        "content": str(chunk.get("content") or ""),
        "source_key": chunk.get("source_key"),
        "source_name": chunk.get("file_path"),
        "chunk_order_index": int(chunk.get("chunk_order_index") or 0),
        "token_count": int(chunk.get("tokens") or 0),
        "full_doc_id": chunk.get("full_doc_id"),
        "metadata": compact_json(dict(chunk)),
    }


def _entity_vector_text(node: dict[str, Any]) -> str:
    properties = dict(node.get("properties") or {})
    return "\n".join(
        value
        for value in (
            str(properties.get("name") or node["id"]),
            str(properties.get("entity_type") or "ENTITY"),
            str(properties.get("description") or ""),
        )
        if value
    )


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item for item in value.split("<SEP>") if item]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _csv_cell(value: str) -> str:
    if "," in value or "\n" in value or '"' in value:
        return '"' + value.replace('"', '""') + '"'
    return value


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_graph_store(component: object) -> GraphStoreProtocol:
    if not isinstance(component, GraphStoreProtocol):
        raise TypeError("stores.graph must satisfy GraphStoreProtocol")
    return component


def _require_sql_store(component: object) -> SQLStoreProtocol:
    if not isinstance(component, SQLStoreProtocol):
        raise TypeError("stores.sql must satisfy SQLStoreProtocol")
    return component


def _require_vector_store(component: object) -> VectorStoreProtocol:
    if not isinstance(component, VectorStoreProtocol):
        raise TypeError("stores.vector must satisfy VectorStoreProtocol")
    return component


def _require_embedding_model(component: object) -> EmbeddingModelProtocol:
    if not isinstance(component, EmbeddingModelProtocol):
        raise TypeError("models.embedding must satisfy EmbeddingModelProtocol")
    return component


def _require_language_model(component: object) -> LanguageModelProtocol:
    if not isinstance(component, LanguageModelProtocol):
        raise TypeError("models.language must satisfy LanguageModelProtocol")
    return component
