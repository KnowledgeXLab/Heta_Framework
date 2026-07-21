import asyncio
import json
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import (  # noqa: E402
    EmbeddingRequest,
    EmbeddingResult,
    ModelChunk,
    ModelRequest,
    ModelResult,
)
from heta_framework.common.stores import (  # noqa: E402
    InMemoryGraphStore,
    InMemoryVectorStore,
    LocalObjectStore,
    SQLStore,
    VectorQuery,
)
from heta_framework.kb.steps import (  # noqa: E402
    BuildHiRAGGraph,
    BuildHiRAGGraphConfig,
    HiRAGGraphIndexAdapter,
    HiRAGTableNames,
    HiRAGVectorCollections,
)
from heta_framework.kb.steps.build_hirag_graph import _community_schema  # noqa: E402


TEST_PROMPTS = {
    "community_report": "REPORT {input_text}",
}


class FakeContext:
    def __init__(self, components):
        self.components = components
        self.artifacts = {}

    def get_component(self, key):
        return self.components[key]

    def get_artifact(self, key):
        return self.artifacts[key]

    def set_artifact(self, key, value):
        self.artifacts[key] = value


class FakeEmbeddingModel:
    @property
    def model_name(self):
        return "fake-embedding"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        vectors = [[float(len(text)), float(text.count("ALICE")), 1.0] for text in request.texts]
        return EmbeddingResult(vectors=vectors, model_name=self.model_name)

    async def embed_many(self, requests: Sequence[EmbeddingRequest]) -> list[EmbeddingResult]:
        return [await self.embed(request) for request in requests]


class FakeLanguageModel:
    def __init__(self):
        self.requests = []

    @property
    def model_name(self):
        return "fake-language"

    async def invoke(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        return ModelResult(
            text=json.dumps(
                {
                    "title": "Alice Community",
                    "summary": "Alice and Bob are connected.",
                    "findings": [{"summary": "Collaboration", "explanation": "They work together."}],
                    "rating": 8,
                }
            ),
            model_name=self.model_name,
        )

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        if False:
            yield ModelChunk(text_delta="", model_name=self.model_name)


def _node(entity_id, entity_type="PERSON", *, source_ids=None, is_summary=False):
    source_ids = source_ids or ["chunk_1"]
    return {
        "id": entity_id,
        "labels": ["Entity", entity_type],
        "properties": {
            "name": entity_id,
            "entity_type": entity_type,
            "raw_entity_type": entity_type.lower(),
            "description": f"{entity_id} appears in the graph.",
            "source_id": "<SEP>".join(source_ids),
            "source_ids": source_ids,
            "layer": 1 if is_summary else 0,
            "cluster_id": "0" if is_summary else None,
            "is_summary": is_summary,
            "parent_entity_ids": ["ALICE", "BOB"] if is_summary else [],
        },
    }


def _edge(source="ALICE", target="BOB"):
    return {
        "id": f"{source}--RELATED--{target}",
        "source_id": source,
        "target_id": target,
        "type": "RELATED",
        "properties": {
            "description": f"{source} is related to {target}.",
            "weight": 2.5,
            "order": 1,
            "source_id": "chunk_1",
            "source_ids": ["chunk_1"],
            "layer": 0,
            "cluster_id": None,
            "is_summary": False,
        },
    }


async def _put_inputs(object_store, context):
    nodes = [_node("ALICE"), _node("BOB"), _node("COLLABORATION", "EVENT", is_summary=True)]
    edges = [
        _edge("ALICE", "BOB"),
        _edge("COLLABORATION", "ALICE"),
        _edge("ALICE", "MISSING_ENTITY"),
    ]
    node_keys = []
    edge_keys = []
    for node in nodes:
        key = f"hi_rag/graph/nodes/{node['id'].lower()}.json"
        await object_store.put(key, json.dumps(node).encode("utf-8"))
        node_keys.append(key)
    for edge in edges:
        key = f"hi_rag/graph/edges/{edge['id'].lower().replace('--', '_')}.json"
        await object_store.put(key, json.dumps(edge).encode("utf-8"))
        edge_keys.append(key)
    context.set_artifact("hi_rag_graph_node_keys", tuple(node_keys))
    context.set_artifact("hi_rag_graph_edge_keys", tuple(edge_keys))
    context.set_artifact(
        "hi_rag_chunks",
        [
            {
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "content": "Alice collaborates with Bob.",
                "source_key": "raw/alice.txt",
                "file_path": "alice.txt",
                "chunk_order_index": 0,
                "tokens": 5,
                "full_doc_id": "doc_1",
            }
        ],
    )


def _config():
    return BuildHiRAGGraphConfig(
        table_names=HiRAGTableNames(
            entities="test_hi_entities",
            relations="test_hi_relations",
            communities="test_hi_communities",
            chunks="test_hi_chunks",
        ),
        vector_collections=HiRAGVectorCollections(entities="test_hi_entity_vectors"),
        graph_cluster_algorithm="connected_components",
        prompts=TEST_PROMPTS,
    )


def test_build_hirag_graph_declares_capabilities():
    step = BuildHiRAGGraph()

    assert step.name == "build_hirag_graph"
    assert {ref.key for ref in step.requirements.components} == {
        "stores.objects",
        "stores.graph",
        "stores.sql",
        "stores.vector",
        "models.embedding",
        "models.language",
    }
    assert step.requirements.artifacts == frozenset(
        {"hi_rag_graph_node_keys", "hi_rag_graph_edge_keys", "hi_rag_chunks"}
    )
    assert "hi_rag_query" in step.capabilities.queries
    assert [asset.kind for asset in step.capabilities.search_assets] == [
        "hi_rag_tables",
        "hi_rag_vector_index",
    ]


def test_build_hirag_graph_writes_sql_vectors_graph_and_reports(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    graph_store = InMemoryGraphStore()
    sql_store = SQLStore("sqlite:///:memory:")
    vector_store = InMemoryVectorStore()
    context = FakeContext(
        {
            "stores.objects": object_store,
            "stores.graph": graph_store,
            "stores.sql": sql_store,
            "stores.vector": vector_store,
            "models.embedding": FakeEmbeddingModel(),
            "models.language": FakeLanguageModel(),
        }
    )

    async def run():
        await _put_inputs(object_store, context)
        await BuildHiRAGGraph(_config()).run(context)
        entity_rows = await sql_store.fetch_all("SELECT * FROM test_hi_entities ORDER BY entity_id")
        relation_rows = await sql_store.fetch_all("SELECT * FROM test_hi_relations ORDER BY relation_id")
        community_rows = await sql_store.fetch_all("SELECT * FROM test_hi_communities")
        chunk_rows = await sql_store.fetch_all("SELECT * FROM test_hi_chunks")
        vector_count = await vector_store.count("test_hi_entity_vectors")
        hits = await vector_store.search(
            "test_hi_entity_vectors",
            VectorQuery(vector=[20.0, 1.0, 1.0], top_k=1),
        )
        report_key = context.artifacts["hi_rag_community_report_keys"][0]
        report = json.loads((await object_store.get(report_key)).decode("utf-8"))
        return entity_rows, relation_rows, community_rows, chunk_rows, vector_count, hits, report

    (
        entity_rows,
        relation_rows,
        community_rows,
        chunk_rows,
        vector_count,
        hits,
        report,
    ) = asyncio.run(run())

    assert len(entity_rows) == 3
    assert len(relation_rows) == 2
    assert len(community_rows) == 1
    assert len(chunk_rows) == 1
    assert vector_count == 3
    assert hits[0].metadata["fact_type"] == "hi_rag_entity"
    assert hits[0].metadata["source_ids"] == ["chunk_1"]
    assert "ALICE" in graph_store.nodes
    assert "ALICE--RELATED--BOB" in graph_store.edges
    assert "ALICE--RELATED--MISSING_ENTITY" not in graph_store.edges
    assert report["report_json"]["title"] == "Alice Community"
    assert json.loads(entity_rows[0]["source_ids"]) == ["chunk_1"]
    assert chunk_rows[0]["source_key"] == "raw/alice.txt"
    assert chunk_rows[0]["document_id"] == "doc_1"


def test_build_hirag_graph_default_community_schema_uses_leiden(monkeypatch):
    calls = []
    graspologic_module = ModuleType("graspologic")
    partition_module = ModuleType("graspologic.partition")
    utils_module = ModuleType("graspologic.utils")

    class Partition:
        def __init__(self, node, level, cluster):
            self.node = node
            self.level = level
            self.cluster = cluster

    def hierarchical_leiden(graph, *, max_cluster_size, random_seed):
        calls.append((graph.number_of_nodes(), graph.number_of_edges(), max_cluster_size, random_seed))
        return [
            Partition("ALICE", 0, 10),
            Partition("BOB", 0, 10),
            Partition("ALICE", 1, 20),
        ]

    partition_module.hierarchical_leiden = hierarchical_leiden
    utils_module.largest_connected_component = lambda graph: graph
    monkeypatch.setitem(sys.modules, "graspologic", graspologic_module)
    monkeypatch.setitem(sys.modules, "graspologic.partition", partition_module)
    monkeypatch.setitem(sys.modules, "graspologic.utils", utils_module)

    communities = _community_schema(
        [_node("ALICE"), _node("BOB")],
        [_edge("ALICE", "BOB")],
        BuildHiRAGGraphConfig(
            max_graph_cluster_size=7,
            graph_cluster_seed=123,
            prompts=TEST_PROMPTS,
        ),
    )

    assert calls == [(2, 1, 7, 123)]
    assert [community["community_id"] for community in communities] == ["10", "20"]
    assert communities[0]["nodes"] == ["ALICE", "BOB"]
    assert communities[0]["edges"] == [["ALICE", "BOB"]]
    assert communities[0]["sub_communities"] == ["20"]


def test_hirag_graph_index_adapter_shortest_path_and_edges():
    entities = [
        {"entity_id": "ALICE"},
        {"entity_id": "BOB"},
        {"entity_id": "CAROL"},
    ]
    relations = [
        {
            "relation_id": "r1",
            "source_entity_id": "ALICE",
            "target_entity_id": "BOB",
            "description": "Alice to Bob",
        },
        {
            "relation_id": "r2",
            "source_entity_id": "BOB",
            "target_entity_id": "CAROL",
            "description": "Bob to Carol",
        },
    ]

    adapter = HiRAGGraphIndexAdapter(entities, relations)

    assert adapter.node_degree("BOB") == 2
    assert adapter.get_edge("BOB", "ALICE")["relation_id"] == "r1"
    assert adapter.shortest_path("ALICE", "CAROL") == ["ALICE", "BOB", "CAROL"]
    assert adapter.shortest_path("ALICE", "MISSING") == ["ALICE", "MISSING"]
    assert [edge["relation_id"] for edge in adapter.subgraph_edges(["ALICE", "BOB"])] == ["r1"]
