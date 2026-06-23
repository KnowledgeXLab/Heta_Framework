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
from heta_framework.common.stores import LocalObjectStore  # noqa: E402
from heta_framework.kb.graphing import ExtractedEntity, ExtractedRelation  # noqa: E402
from heta_framework.kb.steps import (  # noqa: E402
    DeduplicateEntities,
    DeduplicateEntitiesConfig,
    DeduplicateRelations,
    DeduplicateRelationsConfig,
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
            yield ModelChunk(text_delta="", model_name=self.model_name)


class FakeEmbeddingModel:
    @property
    def model_name(self):
        return "fake-embedding"

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        vectors = []
        for text in request.texts:
            lower = text.lower()
            first_line = lower.splitlines()[0]
            if "xuhui" in first_line or "徐汇" in first_line:
                vectors.append([0.0, 1.0, 0.0])
            elif "shanghai" in first_line or "上海" in first_line:
                vectors.append([1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return EmbeddingResult(vectors=vectors, model_name=self.model_name)

    async def embed_many(self, requests: Sequence[EmbeddingRequest]) -> list[EmbeddingResult]:
        return [await self.embed(request) for request in requests]


def _entity(entity_id: str, name: str, description: str, *, chunk_id: str) -> ExtractedEntity:
    return ExtractedEntity(
        entity_id=entity_id,
        chunk_id=chunk_id,
        document_id="doc_1",
        name=name,
        type="城市",
        subtype="直辖市",
        description=description,
        attributes={},
        source_chunk_ids=(chunk_id,),
    )


def _relation(
    relation_id: str,
    source_entity_id: str,
    target_entity_id: str,
    name: str,
    description: str,
    *,
    chunk_id: str,
) -> ExtractedRelation:
    return ExtractedRelation(
        relation_id=relation_id,
        chunk_id=chunk_id,
        document_id="doc_1",
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        source_entity_name="上海市",
        target_entity_name="徐汇区",
        type="空间关系",
        name=name,
        description=description,
        attributes={},
        source_chunk_ids=(chunk_id,),
    )


def test_deduplicate_entities_exact_merge_preserves_entity_protocol(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel(
        [
            {
                "entity": {
                    "name": "上海市",
                    "type": "城市",
                    "subtype": "直辖市",
                    "description": "上海市是中华人民共和国直辖市，也是重要城市。",
                    "attributes": {"所属国家": "中华人民共和国"},
                }
            }
        ]
    )
    context = FakeContext({"stores.objects": object_store, "models.language": model})

    async def run():
        entities = (
            _entity("entity_a", "上海市", "上海市是中华人民共和国直辖市。", chunk_id="chunk_a"),
            _entity("entity_b", " 上海市 ", "上海市是重要城市。", chunk_id="chunk_b"),
        )
        keys = []
        for entity in entities:
            key = f"entities/{entity.entity_id}.json"
            await object_store.put(key, entity.to_json_bytes())
            keys.append(key)
        context.set_artifact("entity_keys", tuple(keys))
        await DeduplicateEntities(DeduplicateEntitiesConfig(semantic_merge=False)).run(context)
        key = context.artifacts["deduplicated_entity_keys"][0]
        return ExtractedEntity.from_json(await object_store.get(key))

    entity = asyncio.run(run())

    assert entity.name == "上海市"
    assert entity.source_chunk_ids == ("chunk_a", "chunk_b")
    assert context.artifacts["entity_id_mapping"] == {
        "entity_a": entity.entity_id,
        "entity_b": entity.entity_id,
    }
    assert context.artifacts["deduplicate_entities_result"].output_entity_count == 1
    assert context.artifacts["deduplicate_entities_result"].exact_merge_count == 1


def test_deduplicate_entities_semantic_merge_is_default(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel(
        [
            {
                "entity_list": [
                    {
                        "NodeName": "上海市",
                        "Type": "城市",
                        "Subtype": "直辖市",
                        "Description": "上海市是中国直辖市。",
                        "Attr": {},
                        "merge_tag": True,
                    }
                ],
                "mapping_table": {"上海市": ["上海市", "Shanghai"]},
            }
        ]
    )
    context = FakeContext(
        {
            "stores.objects": object_store,
            "models.language": model,
            "models.embedding": FakeEmbeddingModel(),
        }
    )

    async def run():
        entities = (
            _entity("entity_a", "上海市", "上海市是中国城市。", chunk_id="chunk_a"),
            _entity(
                "entity_b",
                "Shanghai",
                "Shanghai is a Chinese municipality.",
                chunk_id="chunk_b",
            ),
            _entity("entity_c", "徐汇区", "徐汇区是上海市辖区。", chunk_id="chunk_c"),
        )
        keys = []
        for entity in entities:
            key = f"entities/{entity.entity_id}.json"
            await object_store.put(key, entity.to_json_bytes())
            keys.append(key)
        context.set_artifact("entity_keys", tuple(keys))
        await DeduplicateEntities().run(context)
        return [
            ExtractedEntity.from_json(await object_store.get(key))
            for key in context.artifacts["deduplicated_entity_keys"]
        ]

    entities = asyncio.run(run())

    assert len(entities) == 2
    assert context.artifacts["deduplicate_entities_result"].semantic_merge_count == 1
    assert (
        context.artifacts["entity_id_mapping"]["entity_a"]
        == context.artifacts["entity_id_mapping"]["entity_b"]
    )
    assert context.artifacts["entity_id_mapping"]["entity_c"] in {
        entity.entity_id for entity in entities
    }


def test_deduplicate_entities_keeps_llm_split_outputs(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel(
        [
            [
                {
                    "NodeName": "Apple",
                    "Type": "公司",
                    "Subtype": None,
                    "Description": "Apple 是一家科技公司。",
                    "Attr": {},
                },
                {
                    "NodeName": "Apple fruit",
                    "Type": "水果",
                    "Subtype": None,
                    "Description": "apple 也可以指水果。",
                    "Attr": {},
                },
            ]
        ]
    )
    context = FakeContext({"stores.objects": object_store, "models.language": model})

    async def run():
        entities = (
            _entity("entity_a", "Apple", "Apple 是一家科技公司。", chunk_id="chunk_a"),
            _entity("entity_b", "Apple", "apple 也可以指水果。", chunk_id="chunk_b"),
        )
        keys = []
        for entity in entities:
            key = f"entities/{entity.entity_id}.json"
            await object_store.put(key, entity.to_json_bytes())
            keys.append(key)
        context.set_artifact("entity_keys", tuple(keys))
        await DeduplicateEntities(DeduplicateEntitiesConfig(semantic_merge=False)).run(context)
        return [
            ExtractedEntity.from_json(await object_store.get(key))
            for key in context.artifacts["deduplicated_entity_keys"]
        ]

    entities = asyncio.run(run())

    assert len(entities) == 2
    assert {entity.type for entity in entities} == {"公司", "水果"}
    assert context.artifacts["deduplicate_entities_result"].exact_round_count == 1


def test_deduplicate_entities_records_issue_and_keeps_originals_on_bad_output(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel([{}, {}, {}])
    context = FakeContext({"stores.objects": object_store, "models.language": model})

    async def run():
        entities = (
            _entity("entity_a", "Bad", "first.", chunk_id="chunk_a"),
            _entity("entity_b", "Bad", "second.", chunk_id="chunk_b"),
        )
        keys = []
        for entity in entities:
            key = f"entities/{entity.entity_id}.json"
            await object_store.put(key, entity.to_json_bytes())
            keys.append(key)
        context.set_artifact("entity_keys", tuple(keys))
        await DeduplicateEntities(DeduplicateEntitiesConfig(semantic_merge=False)).run(context)
        return [
            ExtractedEntity.from_json(await object_store.get(key))
            for key in context.artifacts["deduplicated_entity_keys"]
        ]

    entities = asyncio.run(run())
    result = context.artifacts["deduplicate_entities_result"]

    assert {entity.entity_id for entity in entities} == {"entity_a", "entity_b"}
    assert result.failed_group_count == 1
    assert len(result.issues) == 1
    assert result.issues[0].step == "deduplicate_entities"
    assert result.issues[0].subject.type == "dedup_group"
    assert result.issues[0].subject.id == "Bad"
    assert "must contain a non-empty string" in result.issues[0].message
    assert result.issues[0].resolution.action == "kept_original_records"
    assert result.issues[0].severity == "warning"


def test_deduplicate_relations_applies_entity_mapping_and_merges(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel(
        [
            {
                "relation": {
                    "type": "空间关系",
                    "name": "包含行政区",
                    "description": "徐汇区是上海市下辖行政区。",
                    "attributes": {},
                }
            }
        ]
    )
    context = FakeContext({"stores.objects": object_store, "models.language": model})

    async def run():
        relations = (
            _relation(
                "relation_a",
                "entity_shanghai_old",
                "entity_xuhui",
                "包含行政区",
                "上海市包含徐汇区。",
                chunk_id="chunk_a",
            ),
            _relation(
                "relation_b",
                "entity_shanghai_new",
                "entity_xuhui",
                "包含行政区",
                "徐汇区是上海下辖区。",
                chunk_id="chunk_b",
            ),
        )
        keys = []
        for relation in relations:
            key = f"relations/{relation.relation_id}.json"
            await object_store.put(key, relation.to_json_bytes())
            keys.append(key)
        context.set_artifact("relation_keys", tuple(keys))
        context.set_artifact(
            "entity_id_mapping",
            {
                "entity_shanghai_old": "entity_shanghai",
                "entity_shanghai_new": "entity_shanghai",
                "entity_xuhui": "entity_xuhui",
            },
        )
        await DeduplicateRelations(DeduplicateRelationsConfig(semantic_merge=False)).run(context)
        key = context.artifacts["deduplicated_relation_keys"][0]
        return ExtractedRelation.from_json(await object_store.get(key))

    relation = asyncio.run(run())

    assert relation.source_entity_id == "entity_shanghai"
    assert relation.target_entity_id == "entity_xuhui"
    assert relation.source_chunk_ids == ("chunk_a", "chunk_b")
    assert context.artifacts["relation_id_mapping"] == {
        "relation_a": relation.relation_id,
        "relation_b": relation.relation_id,
    }
    assert context.artifacts["deduplicate_relations_result"].exact_merge_count == 1


def test_deduplicate_relations_records_issue_and_keeps_originals_on_bad_output(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel([{}, {}, {}])
    context = FakeContext({"stores.objects": object_store, "models.language": model})

    async def run():
        relations = (
            _relation(
                "relation_a",
                "entity_shanghai",
                "entity_xuhui",
                "Bad",
                "first.",
                chunk_id="chunk_a",
            ),
            _relation(
                "relation_b",
                "entity_shanghai",
                "entity_xuhui",
                "Bad",
                "second.",
                chunk_id="chunk_b",
            ),
        )
        keys = []
        for relation in relations:
            key = f"relations/{relation.relation_id}.json"
            await object_store.put(key, relation.to_json_bytes())
            keys.append(key)
        context.set_artifact("relation_keys", tuple(keys))
        context.set_artifact("entity_id_mapping", {})
        await DeduplicateRelations(DeduplicateRelationsConfig(semantic_merge=False)).run(context)
        return [
            ExtractedRelation.from_json(await object_store.get(key))
            for key in context.artifacts["deduplicated_relation_keys"]
        ]

    relations = asyncio.run(run())
    result = context.artifacts["deduplicate_relations_result"]

    assert {relation.relation_id for relation in relations} == {"relation_a", "relation_b"}
    assert result.failed_group_count == 1
    assert len(result.issues) == 1
    assert result.issues[0].step == "deduplicate_relations"
    assert result.issues[0].subject.type == "dedup_group"
    assert result.issues[0].subject.id == "上海市|徐汇区|Bad|空间关系"
    assert "must contain a non-empty string" in result.issues[0].message
    assert result.issues[0].resolution.action == "kept_original_records"
    assert result.issues[0].code == "deduplication_failed"


def test_deduplicate_relations_semantic_merge_uses_mapping_table(tmp_path):
    object_store = LocalObjectStore(tmp_path)
    model = FakeLanguageModel(
        [
            {
                "relation_list": [
                    {
                        "Node1": "上海市",
                        "Node2": "徐汇区",
                        "Relation": "包含行政区",
                        "Type": "空间关系",
                        "Description": "徐汇区是上海市下辖行政区。",
                        "Attr": {},
                        "merge_tag": True,
                    }
                ],
                "mapping_table": {
                    "上海市||徐汇区": [
                        "上海市||徐汇区",
                    ]
                },
            }
        ]
    )
    context = FakeContext(
        {
            "stores.objects": object_store,
            "models.language": model,
            "models.embedding": FakeEmbeddingModel(),
        }
    )

    async def run():
        relations = (
            _relation(
                "relation_a",
                "entity_shanghai",
                "entity_xuhui",
                "包含行政区",
                "上海市包含徐汇区。",
                chunk_id="chunk_a",
            ),
            _relation(
                "relation_b",
                "entity_shanghai",
                "entity_xuhui",
                "下辖",
                "徐汇区是上海市下辖行政区。",
                chunk_id="chunk_b",
            ),
        )
        keys = []
        for relation in relations:
            key = f"relations/{relation.relation_id}.json"
            await object_store.put(key, relation.to_json_bytes())
            keys.append(key)
        context.set_artifact("relation_keys", tuple(keys))
        context.set_artifact("entity_id_mapping", {})
        await DeduplicateRelations().run(context)
        return [
            ExtractedRelation.from_json(await object_store.get(key))
            for key in context.artifacts["deduplicated_relation_keys"]
        ]

    relations = asyncio.run(run())

    assert len(relations) == 1
    assert relations[0].name == "包含行政区"
    assert relations[0].source_chunk_ids == ("chunk_a", "chunk_b")
    assert context.artifacts["deduplicate_relations_result"].semantic_round_count == 1


def test_deduplicate_steps_require_embedding_only_when_semantic_merge_is_enabled():
    entity_step = DeduplicateEntities()
    assert {ref.key for ref in entity_step.requirements.components} == {
        "stores.objects",
        "models.language",
        "models.embedding",
    }

    relation_step = DeduplicateRelations(DeduplicateRelationsConfig(semantic_merge=False))
    assert {ref.key for ref in relation_step.requirements.components} == {
        "stores.objects",
        "models.language",
    }
    assert relation_step.requirements.artifacts == frozenset(
        {"relation_keys", "entity_id_mapping"}
    )
