import asyncio
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import (  # noqa: E402
    EmbeddingRequest,
    EmbeddingResult,
    ModelChunk,
    ModelRequest,
    ModelResult,
)
from heta_framework.common.stores import InMemoryVectorStore, LocalObjectStore, SQLStore  # noqa: E402
from heta_framework.kb.chunking import ParsedChunk  # noqa: E402
from heta_framework.kb.graphing import ExtractedEntity, ExtractedRelation  # noqa: E402
from heta_framework.kb.parsing import ParsedSource  # noqa: E402
from heta_framework.kb.steps import (  # noqa: E402
    BuildGraph,
    BuildGraphConfig,
    GraphTableNames,
    MergeGraphIntoStore,
    MergeGraphIntoStoreConfig,
)


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
        vectors = []
        for text in request.texts:
            first = text.splitlines()[0].lower()
            if "徐汇" in first or "xuhui" in first:
                vectors.append([0.0, 1.0, 0.0])
            elif "上海" in first or "shanghai" in first:
                vectors.append([1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return EmbeddingResult(vectors=vectors, model_name=self.model_name)

    async def embed_many(self, requests: Sequence[EmbeddingRequest]) -> list[EmbeddingResult]:
        return [await self.embed(request) for request in requests]


class FakeLanguageModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    @property
    def model_name(self):
        return "fake-language"

    async def invoke(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        return ModelResult(text="", parsed=self.responses.pop(0), model_name=self.model_name)

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        if False:
            yield ModelChunk(text_delta="", model_name=self.model_name)


def _tables(prefix: str) -> GraphTableNames:
    return GraphTableNames(
        entities=f"{prefix}_entities",
        relations=f"{prefix}_relations",
        evidence=f"{prefix}_graph_evidence",
    )


def _chunk(chunk_id: str, text: str) -> ParsedChunk:
    return ParsedChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        source=ParsedSource(
            key=f"raw/{chunk_id}.pdf",
            name=f"{chunk_id}.pdf",
            file_type="pdf",
            content_sha256="a" * 64,
        ),
        page_index=0,
        chunk_index=0,
        text=text,
        token_start=0,
        token_end=len(text),
    )


def _entity(entity_id: str, name: str, description: str, chunk_id: str) -> ExtractedEntity:
    return ExtractedEntity(
        entity_id=entity_id,
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        name=name,
        type="城市",
        subtype="直辖市",
        description=description,
        attributes={},
        source_chunk_ids=(chunk_id,),
    )


def _relation(
    relation_id: str,
    source_id: str,
    target_id: str,
    source_name: str,
    target_name: str,
    chunk_id: str,
) -> ExtractedRelation:
    return ExtractedRelation(
        relation_id=relation_id,
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        source_entity_id=source_id,
        target_entity_id=target_id,
        source_entity_name=source_name,
        target_entity_name=target_name,
        type="空间关系",
        name="包含行政区",
        description=f"{target_name} 是 {source_name} 下辖区域。",
        attributes={},
        source_chunk_ids=(chunk_id,),
    )


async def _put_inputs(object_store, context, entities, relations, chunks):
    entity_keys = []
    for entity in entities:
        key = f"entities/{entity.entity_id}.json"
        await object_store.put(key, entity.to_json_bytes())
        entity_keys.append(key)
    relation_keys = []
    for relation in relations:
        key = f"relations/{relation.relation_id}.json"
        await object_store.put(key, relation.to_json_bytes())
        relation_keys.append(key)
    chunk_keys = []
    for chunk in chunks:
        key = f"chunks/{chunk.chunk_id}.json"
        await object_store.put(key, chunk.to_json_bytes())
        chunk_keys.append(key)
    context.set_artifact("deduplicated_entity_keys", tuple(entity_keys))
    context.set_artifact("deduplicated_relation_keys", tuple(relation_keys))
    context.set_artifact("chunk_keys", tuple(chunk_keys))


def test_merge_graph_into_store_declares_dynamic_graph_contract():
    step = MergeGraphIntoStore()

    assert step.name == "merge_graph_into_store"
    assert {ref.key for ref in step.requirements.components} == {
        "models.embedding",
        "models.language",
        "stores.objects",
        "stores.sql",
        "stores.vector",
    }
    assert step.requirements.artifacts == frozenset(
        {"deduplicated_entity_keys", "deduplicated_relation_keys", "chunk_keys"}
    )
    assert step.capabilities.artifacts == frozenset({"merge_graph_into_store_result"})
    assert step.capabilities.queries == frozenset({"heta_graph_search"})
    assert [asset.kind for asset in step.capabilities.search_assets] == [
        "graph_tables",
        "graph_vector_index",
    ]
    assert step.capabilities.search_assets[0].metadata["entities_table"] == "entities"
    assert step.capabilities.search_assets[1].metadata["entity_collection"] == "graph_entities"


def test_merge_graph_into_store_merges_against_existing_graph(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    sql_store = SQLStore("sqlite:///:memory:")
    vector_store = InMemoryVectorStore()
    language_model = FakeLanguageModel(
        [
            {
                "entity_list": [
                    {
                        "NodeName": "上海市",
                        "Type": "城市",
                        "Subtype": "直辖市",
                        "Description": "上海市是中国直辖市和重要城市。",
                        "Attr": {},
                        "merge_tag": True,
                    }
                ],
                "mapping_table": {"上海市": ["上海市", "Shanghai"]},
            },
            {
                "entity_list": [
                    {
                        "NodeName": "上海市",
                        "Type": "城市",
                        "Subtype": "直辖市",
                        "Description": "上海市是中国直辖市和重要城市。",
                        "Attr": {},
                        "merge_tag": True,
                    }
                ],
                "mapping_table": {"上海市": ["上海市", "Shanghai"]},
            },
            {
                "relation_list": [
                    {
                        "Node1": "上海市",
                        "Node2": "徐汇区",
                        "Relation": "包含行政区",
                        "Type": "空间关系",
                        "Description": "徐汇区是上海市下辖区域。",
                        "Attr": {},
                        "merge_tag": True,
                    }
                ],
                "mapping_table": {
                    "上海市||徐汇区": ["relation_old", "relation_new"],
                },
            },
            {
                "relation_list": [
                    {
                        "Node1": "上海市",
                        "Node2": "徐汇区",
                        "Relation": "包含行政区",
                        "Type": "空间关系",
                        "Description": "徐汇区是上海市下辖区域。",
                        "Attr": {},
                        "merge_tag": True,
                    }
                ],
                "mapping_table": {
                    "上海市||徐汇区": ["relation_old", "relation_new"],
                },
            },
        ]
    )
    context = FakeContext(
        {
            "stores.objects": object_store,
            "stores.sql": sql_store,
            "stores.vector": vector_store,
            "models.embedding": FakeEmbeddingModel(),
            "models.language": language_model,
        }
    )
    table_names = _tables("papers")

    async def run():
        old_entities = (
            _entity("entity_old_shanghai", "上海市", "上海市是中国直辖市。", "chunk_old"),
            _entity("entity_xuhui", "徐汇区", "徐汇区是上海市辖区。", "chunk_old"),
        )
        old_relations = (
            _relation(
                "relation_old",
                "entity_old_shanghai",
                "entity_xuhui",
                "上海市",
                "徐汇区",
                "chunk_old",
            ),
        )
        await _put_inputs(
            object_store,
            context,
            old_entities,
            old_relations,
            (_chunk("chunk_old", "上海市包含徐汇区。"),),
        )
        await BuildGraph(BuildGraphConfig(table_names=table_names)).run(context)

        new_entities = (
            _entity("entity_new_shanghai", "Shanghai", "Shanghai is a Chinese city.", "chunk_new"),
        )
        new_relations = (
            _relation(
                "relation_new",
                "entity_new_shanghai",
                "entity_xuhui",
                "Shanghai",
                "徐汇区",
                "chunk_new",
            ),
        )
        await _put_inputs(
            object_store,
            context,
            new_entities,
            new_relations,
            (_chunk("chunk_new", "Shanghai 包含徐汇区。"),),
        )
        await MergeGraphIntoStore(
            MergeGraphIntoStoreConfig(table_names=table_names, similarity_threshold=0.5)
        ).run(context)
        entity_rows = await sql_store.fetch_all(
            "SELECT entity_id, entity_name, description FROM papers_entities ORDER BY entity_name"
        )
        relation_rows = await sql_store.fetch_all(
            "SELECT relation_id, source_entity_name, target_entity_name FROM papers_relations"
        )
        evidence_rows = await sql_store.fetch_all(
            "SELECT fact_id, fact_type, chunk_id FROM papers_graph_evidence ORDER BY chunk_id"
        )
        return entity_rows, relation_rows, evidence_rows, context.artifacts[
            "merge_graph_into_store_result"
        ]

    try:
        entity_rows, relation_rows, evidence_rows, result = asyncio.run(run())
    finally:
        asyncio.run(sql_store.aclose())

    assert result.input_entity_count == 1
    assert result.input_relation_count == 1
    assert result.merged_entity_count == 1
    assert result.deleted_entity_count == 1
    assert result.merged_relation_count == 1
    assert result.deleted_relation_count == 1
    assert not result.issues
    assert {row["entity_id"] for row in entity_rows} != {"entity_old_shanghai", "entity_xuhui"}
    assert any(row["entity_name"] == "上海市" for row in entity_rows)
    assert relation_rows[0]["source_entity_name"] == "上海市"
    assert {row["chunk_id"] for row in evidence_rows} == {"chunk_old", "chunk_new"}


def test_merge_graph_into_store_does_not_merge_unmatched_relation_mapping(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    sql_store = SQLStore("sqlite:///:memory:")
    vector_store = InMemoryVectorStore()
    language_model = FakeLanguageModel(
        [
            {
                "relation_list": [],
                "mapping_table": {"无关||映射": ["missing_relation"]},
            }
        ]
    )
    context = FakeContext(
        {
            "stores.objects": object_store,
            "stores.sql": sql_store,
            "stores.vector": vector_store,
            "models.embedding": FakeEmbeddingModel(),
            "models.language": language_model,
        }
    )
    table_names = _tables("papers")

    async def run():
        old_entities = (
            _entity("entity_old_shanghai", "上海市", "上海市是中国直辖市。", "chunk_old"),
            _entity("entity_xuhui", "徐汇区", "徐汇区是上海市辖区。", "chunk_old"),
        )
        old_relations = (
            _relation(
                "relation_old",
                "entity_old_shanghai",
                "entity_xuhui",
                "上海市",
                "徐汇区",
                "chunk_old",
            ),
        )
        await _put_inputs(
            object_store,
            context,
            old_entities,
            old_relations,
            (_chunk("chunk_old", "上海市包含徐汇区。"),),
        )
        await BuildGraph(BuildGraphConfig(table_names=table_names)).run(context)

        new_relations = (
            _relation(
                "relation_new",
                "entity_old_shanghai",
                "entity_xuhui",
                "上海市",
                "徐汇区",
                "chunk_new",
            ),
        )
        await _put_inputs(
            object_store,
            context,
            (),
            new_relations,
            (_chunk("chunk_new", "上海市继续包含徐汇区。"),),
        )
        await MergeGraphIntoStore(
            MergeGraphIntoStoreConfig(table_names=table_names, similarity_threshold=0.5)
        ).run(context)
        rows = await sql_store.fetch_all(
            "SELECT relation_id FROM papers_relations ORDER BY relation_id"
        )
        return rows, context.artifacts["merge_graph_into_store_result"]

    try:
        rows, result = asyncio.run(run())
    finally:
        asyncio.run(sql_store.aclose())

    assert [row["relation_id"] for row in rows] == ["relation_new", "relation_old"]
    assert result.merged_relation_count == 0
    assert result.deleted_relation_count == 0
