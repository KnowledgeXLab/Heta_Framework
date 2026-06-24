import asyncio
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import ModelChunk, ModelRequest, ModelResult  # noqa: E402
from heta_framework.common.stores import LocalObjectStore  # noqa: E402
from heta_framework.kb.chunking import ParsedChunk  # noqa: E402
from heta_framework.kb.graphing import ExtractedEntity, ExtractedRelation  # noqa: E402
from heta_framework.kb.parsing import ParsedSource  # noqa: E402
from heta_framework.kb.steps import ExtractRelations, ExtractRelationsConfig  # noqa: E402


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


class FakeLanguageModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    @property
    def model_name(self):
        return "fake-language"

    async def invoke(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return ModelResult(text="", parsed=response, model_name=self.model_name)

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        if False:
            yield ModelChunk(text_delta="")


def _source() -> ParsedSource:
    return ParsedSource(
        key="raw/shanghai.txt",
        name="shanghai.txt",
        file_type="txt",
        content_sha256="b" * 64,
    )


def _chunk(**overrides) -> ParsedChunk:
    values = {
        "chunk_id": "chunk_1",
        "document_id": "doc_1",
        "source": _source(),
        "page_index": 0,
        "chunk_index": 0,
        "text": "上海市包含徐汇区。",
        "token_start": 0,
        "token_end": 9,
        "parent_chunk_ids": (),
    }
    values.update(overrides)
    return ParsedChunk(**values)


def _entity(entity_id: str, name: str, *, chunk_id: str = "chunk_1") -> ExtractedEntity:
    return ExtractedEntity(
        entity_id=entity_id,
        chunk_id=chunk_id,
        document_id="doc_1",
        name=name,
        type="客观实体",
        subtype="行政区划",
        description=f"{name} 是地理实体。",
        attributes={},
        source_chunk_ids=(chunk_id,),
    )


async def _put_chunk_and_entities(
    object_store: LocalObjectStore,
    context: FakeContext,
    chunk: ParsedChunk,
    entities: tuple[ExtractedEntity, ...],
) -> None:
    await object_store.put(f"chunks/{chunk.chunk_id}.json", chunk.to_json_bytes())
    entity_keys = []
    for entity in entities:
        key = f"entities/{entity.chunk_id}/{entity.entity_id}.json"
        await object_store.put(key, entity.to_json_bytes())
        entity_keys.append(key)
    context.set_artifact("chunk_keys", (f"chunks/{chunk.chunk_id}.json",))
    context.set_artifact("entity_keys", tuple(entity_keys))


def test_extract_relations_writes_relation_json(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel(
        [
            {
                "relations": [
                    {
                        "source": "上海市",
                        "target": "徐汇区",
                        "type": "空间关系",
                        "name": "包含行政区",
                        "description": "徐汇区是上海市下辖的市辖区。",
                        "attributes": {},
                    }
                ]
            }
        ]
    )
    context = FakeContext({"stores.objects": object_store, "models.language": model})

    async def run():
        chunk = _chunk()
        await _put_chunk_and_entities(
            object_store,
            context,
            chunk,
            (_entity("entity_shanghai", "上海市"), _entity("entity_xuhui", "徐汇区")),
        )
        await ExtractRelations().run(context)
        key = context.artifacts["relation_keys"][0]
        return key, ExtractedRelation.from_json(await object_store.get(key))

    key, relation = asyncio.run(run())

    assert key.startswith("relations/chunk_1/relation_")
    assert relation.source_entity_id == "entity_shanghai"
    assert relation.target_entity_id == "entity_xuhui"
    assert relation.source_entity_name == "上海市"
    assert relation.target_entity_name == "徐汇区"
    assert relation.type == "空间关系"
    assert relation.name == "包含行政区"
    assert relation.source_chunk_ids == ("chunk_1",)
    assert context.artifacts["extract_relations_result"].relation_count == 1


def test_extract_relations_reuses_existing_relation_json(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel([])
    context = FakeContext({"stores.objects": object_store, "models.language": model})

    async def run():
        chunk = _chunk()
        relation = ExtractedRelation(
            relation_id="relation_cached",
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source_entity_id="entity_shanghai",
            target_entity_id="entity_xuhui",
            source_entity_name="上海市",
            target_entity_name="徐汇区",
            type="空间关系",
            name="包含行政区",
            description="徐汇区是上海市下辖的市辖区。",
            attributes={},
            source_chunk_ids=(chunk.chunk_id,),
        )
        await _put_chunk_and_entities(
            object_store,
            context,
            chunk,
            (_entity("entity_shanghai", "上海市"), _entity("entity_xuhui", "徐汇区")),
        )
        await object_store.put(
            "relations/chunk_1/relation_cached.json",
            relation.to_json_bytes(),
        )
        await ExtractRelations().run(context)

    asyncio.run(run())

    assert model.requests == []
    assert context.artifacts["relation_keys"] == ("relations/chunk_1/relation_cached.json",)
    assert context.artifacts["extract_relations_result"].relation_count == 1


def test_extract_relations_uses_parent_chunk_ids_for_rechunked_input(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel(
        [
            {
                "relations": [
                    {
                        "source": "Heta",
                        "target": "KnowledgeBase",
                        "type": "组件关系",
                        "name": "构建",
                        "description": "Heta 可以构建 KnowledgeBase。",
                        "attributes": {"scope": "framework"},
                    }
                ]
            }
        ]
    )
    context = FakeContext({"stores.objects": object_store, "models.language": model})

    async def run():
        chunk = _chunk(
            chunk_id="chunk_re",
            text="Heta 可以构建 KnowledgeBase。",
            parent_chunk_ids=("chunk_a", "chunk_b"),
        )
        entities = (
            _entity("entity_heta", "Heta", chunk_id="chunk_re"),
            _entity("entity_kb", "KnowledgeBase", chunk_id="chunk_re"),
        )
        await object_store.put("rechunked/chunk_re.json", chunk.to_json_bytes())
        entity_keys = []
        for entity in entities:
            key = f"entities/chunk_re/{entity.entity_id}.json"
            await object_store.put(key, entity.to_json_bytes())
            entity_keys.append(key)
        context.set_artifact("rechunked_chunk_keys", ("rechunked/chunk_re.json",))
        context.set_artifact("entity_keys", tuple(entity_keys))
        await ExtractRelations(
            ExtractRelationsConfig(chunk_keys_artifact="rechunked_chunk_keys")
        ).run(context)
        key = context.artifacts["relation_keys"][0]
        return ExtractedRelation.from_json(await object_store.get(key))

    relation = asyncio.run(run())

    assert relation.chunk_id == "chunk_re"
    assert relation.source_chunk_ids == ("chunk_a", "chunk_b")
    assert relation.attributes == {"scope": "framework"}


def test_extract_relations_retries_unknown_entity(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel(
        [
            {
                "relations": [
                    {
                        "source": "上海市",
                        "target": "不存在实体",
                        "type": "空间关系",
                        "name": "包含",
                        "description": "无效关系。",
                        "attributes": {},
                    }
                ]
            },
            {
                "relations": [
                    {
                        "source": "上海市",
                        "target": "徐汇区",
                        "type": "空间关系",
                        "name": "包含行政区",
                        "description": "徐汇区是上海市下辖的市辖区。",
                        "attributes": {},
                    }
                ]
            },
        ]
    )
    context = FakeContext({"stores.objects": object_store, "models.language": model})

    async def run():
        chunk = _chunk(chunk_id="chunk_retry")
        await _put_chunk_and_entities(
            object_store,
            context,
            chunk,
            (
                _entity("entity_shanghai", "上海市", chunk_id="chunk_retry"),
                _entity("entity_xuhui", "徐汇区", chunk_id="chunk_retry"),
            ),
        )
        await ExtractRelations(ExtractRelationsConfig(max_attempts=2)).run(context)
        key = context.artifacts["relation_keys"][0]
        return ExtractedRelation.from_json(await object_store.get(key))

    relation = asyncio.run(run())

    assert relation.target_entity_name == "徐汇区"
    assert len(model.requests) == 2
    assert "previous relation extraction response was invalid" in model.requests[1].prompt


def test_extract_relations_skips_chunks_with_too_few_entities(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel([])
    context = FakeContext({"stores.objects": object_store, "models.language": model})

    async def run():
        chunk = _chunk(chunk_id="chunk_single")
        await _put_chunk_and_entities(
            object_store,
            context,
            chunk,
            (_entity("entity_shanghai", "上海市", chunk_id="chunk_single"),),
        )
        await ExtractRelations().run(context)

    asyncio.run(run())

    result = context.artifacts["extract_relations_result"]
    assert result.relation_count == 0
    assert result.skipped_chunk_ids == ("chunk_single",)
    assert result.failed_chunk_ids == ()
    assert context.artifacts["relation_keys"] == ()
    assert model.requests == []
