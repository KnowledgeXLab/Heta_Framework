import asyncio
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.models import ModelChunk, ModelRequest, ModelResult  # noqa: E402
from heta_framework.common.stores import LocalObjectStore  # noqa: E402
from heta_framework.kb.chunking import ParsedChunk  # noqa: E402
from heta_framework.kb.graphing import ExtractedEntity  # noqa: E402
from heta_framework.kb.parsing import ParsedSource  # noqa: E402
from heta_framework.kb.steps import ExtractEntities, ExtractEntitiesConfig  # noqa: E402


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
        content_sha256="a" * 64,
    )


def _chunk(**overrides) -> ParsedChunk:
    values = {
        "chunk_id": "chunk_1",
        "document_id": "doc_1",
        "source": _source(),
        "page_index": 0,
        "chunk_index": 0,
        "text": "上海市是中华人民共和国直辖市。",
        "token_start": 0,
        "token_end": 12,
        "parent_chunk_ids": (),
    }
    values.update(overrides)
    return ParsedChunk(**values)


def test_extract_entities_writes_entity_json(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel(
        [
            {
                "entities": [
                    {
                        "name": "上海市",
                        "type": "客观实体",
                        "subtype": "行政区划",
                        "description": "上海市是中华人民共和国直辖市。",
                        "attributes": {"所属国家": "中华人民共和国"},
                    }
                ]
            }
        ]
    )
    context = FakeContext(
        {
            "stores.objects": object_store,
            "models.language": model,
        }
    )

    async def run():
        chunk = _chunk()
        await object_store.put("chunks/chunk_1.json", chunk.to_json_bytes())
        context.set_artifact("chunk_keys", ("chunks/chunk_1.json",))
        await ExtractEntities().run(context)
        key = context.artifacts["entity_keys"][0]
        return key, ExtractedEntity.from_json(await object_store.get(key))

    key, entity = asyncio.run(run())

    assert key.startswith("entities/chunk_1/entity_")
    assert entity.name == "上海市"
    assert entity.type == "客观实体"
    assert entity.subtype == "行政区划"
    assert entity.attributes == {"所属国家": "中华人民共和国"}
    assert entity.source_chunk_ids == ("chunk_1",)
    assert context.artifacts["extract_entities_result"].entity_count == 1


def test_extract_entities_reuses_existing_entity_json(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel([])
    context = FakeContext(
        {
            "stores.objects": object_store,
            "models.language": model,
        }
    )

    async def run():
        chunk = _chunk()
        entity = ExtractedEntity(
            entity_id="entity_cached",
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            name="上海市",
            type="客观实体",
            subtype="行政区划",
            description="上海市是中华人民共和国直辖市。",
            attributes={},
            source_chunk_ids=(chunk.chunk_id,),
        )
        await object_store.put("chunks/chunk_1.json", chunk.to_json_bytes())
        await object_store.put("entities/chunk_1/entity_cached.json", entity.to_json_bytes())
        context.set_artifact("chunk_keys", ("chunks/chunk_1.json",))
        await ExtractEntities().run(context)

    asyncio.run(run())

    assert model.requests == []
    assert context.artifacts["entity_keys"] == ("entities/chunk_1/entity_cached.json",)
    assert context.artifacts["extract_entities_result"].entity_count == 1


def test_extract_entities_uses_parent_chunk_ids_for_rechunked_input(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel(
        [
            {
                "entities": [
                    {
                        "name": "Heta",
                        "type": "技术",
                        "subtype": None,
                        "description": "Heta 是知识库框架。",
                        "attributes": {},
                    }
                ]
            }
        ]
    )
    context = FakeContext({"stores.objects": object_store, "models.language": model})

    async def run():
        chunk = _chunk(
            chunk_id="chunk_re",
            text="Heta 是知识库框架。",
            parent_chunk_ids=("chunk_a", "chunk_b"),
        )
        await object_store.put("rechunked/chunk_re.json", chunk.to_json_bytes())
        context.set_artifact("rechunked_chunk_keys", ("rechunked/chunk_re.json",))
        await ExtractEntities(
            ExtractEntitiesConfig(chunk_keys_artifact="rechunked_chunk_keys")
        ).run(context)
        key = context.artifacts["entity_keys"][0]
        return ExtractedEntity.from_json(await object_store.get(key))

    entity = asyncio.run(run())

    assert entity.chunk_id == "chunk_re"
    assert entity.source_chunk_ids == ("chunk_a", "chunk_b")


def test_extract_entities_retries_invalid_model_output(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel(
        [
            {"entities": [{"name": "", "type": "技术", "description": "bad", "attributes": {}}]},
            {
                "entities": [
                    {
                        "name": "Heta",
                        "type": "技术",
                        "subtype": "框架",
                        "description": "Heta 是知识库框架。",
                        "attributes": {},
                    }
                ]
            },
        ]
    )
    context = FakeContext({"stores.objects": object_store, "models.language": model})

    async def run():
        chunk = _chunk(chunk_id="chunk_retry", text="Heta 是知识库框架。")
        await object_store.put("chunks/chunk_retry.json", chunk.to_json_bytes())
        context.set_artifact("chunk_keys", ("chunks/chunk_retry.json",))
        await ExtractEntities(ExtractEntitiesConfig(max_attempts=2)).run(context)
        key = context.artifacts["entity_keys"][0]
        return ExtractedEntity.from_json(await object_store.get(key))

    entity = asyncio.run(run())

    assert entity.name == "Heta"
    assert len(model.requests) == 2
    assert "previous entity extraction response was invalid" in model.requests[1].prompt


def test_extract_entities_records_failed_chunks(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel(
        [
            {"entities": "not-a-list"},
            {"entities": "still-not-a-list"},
        ]
    )
    context = FakeContext({"stores.objects": object_store, "models.language": model})

    async def run():
        chunk = _chunk(chunk_id="chunk_bad")
        await object_store.put("chunks/chunk_bad.json", chunk.to_json_bytes())
        context.set_artifact("chunk_keys", ("chunks/chunk_bad.json",))
        await ExtractEntities(ExtractEntitiesConfig(max_attempts=2)).run(context)

    asyncio.run(run())

    result = context.artifacts["extract_entities_result"]
    assert result.entity_count == 0
    assert result.failed_chunk_ids == ("chunk_bad",)
    assert context.artifacts["entity_keys"] == ()
