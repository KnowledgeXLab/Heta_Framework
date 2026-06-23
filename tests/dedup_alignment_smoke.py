"""Smoke-check Heta Framework dedup steps against HetaDB dedup functions."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import types
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _install_hetadb_import_stubs() -> None:
    jieba = types.ModuleType("jieba")
    jieba.initialize = lambda: None
    jieba.cut = lambda text: text.split()
    sys.modules.setdefault("jieba", jieba)

    zhconv = types.ModuleType("zhconv")
    zhconv.convert = lambda text, _: text
    sys.modules.setdefault("zhconv", zhconv)

    openai = types.ModuleType("openai")
    openai.OpenAI = object
    sys.modules.setdefault("openai", openai)

    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda value, *_, **__: value
    sys.modules.setdefault("tqdm", tqdm)

    pymilvus = types.ModuleType("pymilvus")
    pymilvus.Collection = object
    sys.modules.setdefault("pymilvus", pymilvus)

    sklearn = types.ModuleType("sklearn")
    sklearn_cluster = types.ModuleType("sklearn.cluster")
    sklearn_preprocessing = types.ModuleType("sklearn.preprocessing")

    class AgglomerativeClustering:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def fit_predict(self, values: Any) -> list[int]:
            return list(range(len(values)))

    sklearn_cluster.AgglomerativeClustering = AgglomerativeClustering
    sklearn_preprocessing.normalize = lambda values: values
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules.setdefault("sklearn.cluster", sklearn_cluster)
    sys.modules.setdefault("sklearn.preprocessing", sklearn_preprocessing)

    sql_db = types.ModuleType("hetadb.core.db_build.sql_db.sql_db")
    for name in (
        "create_graph_tables",
        "delete_entities_from_pg",
        "insert_entities_to_pg",
        "insert_cluster_chunk_relations",
        "delete_cluster_chunk_relations_by_cluster_ids",
        "get_chunk_source_mapping",
        "insert_relations_to_pg",
        "delete_relations_from_pg",
    ):
        setattr(sql_db, name, lambda *args, **kwargs: None)
    sys.modules.setdefault("hetadb.core.db_build.sql_db.sql_db", sql_db)

    vector_db = types.ModuleType("hetadb.core.db_build.vector_db.vector_db")
    for name in (
        "ensure_nodes_collection",
        "connect_milvus",
        "insert_nodes_records_to_milvus",
        "delete_nodes_records_from_milvus",
        "search_similar_entities",
        "ensure_rel_collection",
        "rel_milvus_to_record_format",
        "insert_relations_to_milvus",
        "delete_relations_from_milvus",
        "search_similar_relations",
    ):
        setattr(vector_db, name, lambda *args, **kwargs: None)
    sys.modules.setdefault("hetadb.core.db_build.vector_db.vector_db", vector_db)


_install_hetadb_import_stubs()

from hetadb.core.db_build.graph_db.node_dedup_merge import dedup_nodes  # noqa: E402
from hetadb.core.db_build.graph_db.rel_dedup_merge import dedup_relations  # noqa: E402
from heta_framework.common.models import ModelChunk, ModelRequest, ModelResult  # noqa: E402
from heta_framework.common.stores import LocalObjectStore  # noqa: E402
from heta_framework.kb.graphing import ExtractedEntity, ExtractedRelation  # noqa: E402
from heta_framework.kb.steps import (  # noqa: E402
    DeduplicateEntities,
    DeduplicateEntitiesConfig,
    DeduplicateRelations,
    DeduplicateRelationsConfig,
)


class FakeContext:
    def __init__(self, components: dict[str, object]) -> None:
        self.components = components
        self.artifacts: dict[str, object] = {}

    def get_component(self, key: str) -> object:
        return self.components[key]

    def get_artifact(self, key: str) -> object:
        return self.artifacts[key]

    def set_artifact(self, key: str, value: object) -> None:
        self.artifacts[key] = value


class FakeLanguageModel:
    def __init__(self, responses: Sequence[Any]) -> None:
        self.responses = list(responses)

    @property
    def model_name(self) -> str:
        return "alignment-fake"

    async def invoke(self, request: ModelRequest) -> ModelResult:
        del request
        response = self.responses.pop(0)
        return ModelResult(text="", parsed=response, model_name=self.model_name)

    async def invoke_many(self, requests: Sequence[ModelRequest]) -> list[ModelResult]:
        return [await self.invoke(request) for request in requests]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        del request
        if False:
            yield ModelChunk(text_delta="", model_name=self.model_name)


def _node_record(name: str, desc: str, chunk_id: str, type_: str = "类型") -> dict[str, Any]:
    return {
        "NodeName": name,
        "Description": desc,
        "Type": type_,
        "chunk_id": chunk_id,
    }


def _entity(record: dict[str, Any], index: int) -> ExtractedEntity:
    chunk_value = record["chunk_id"]
    source_chunk_ids = (
        tuple(str(item) for item in chunk_value)
        if isinstance(chunk_value, list)
        else (str(chunk_value),)
    )
    return ExtractedEntity(
        entity_id=f"entity_{index}",
        chunk_id=str(record["chunk_id"]),
        document_id="doc_alignment",
        name=str(record["NodeName"]),
        type=str(record.get("Type") or "类型"),
        subtype=None,
        description=str(record["Description"]),
        attributes={},
        source_chunk_ids=source_chunk_ids,
    )


def _relation_record(
    node1: str,
    node2: str,
    relation: str,
    type_: str,
    desc: str,
    chunk_id: str,
) -> dict[str, Any]:
    return {
        "Node1": node1,
        "Node2": node2,
        "Relation": relation,
        "Type": type_,
        "Description": desc,
        "chunk_id": chunk_id,
    }


def _relation(record: dict[str, Any], index: int) -> ExtractedRelation:
    chunk_value = record["chunk_id"]
    source_chunk_ids = (
        tuple(str(item) for item in chunk_value)
        if isinstance(chunk_value, list)
        else (str(chunk_value),)
    )
    return ExtractedRelation(
        relation_id=f"relation_{index}",
        chunk_id=str(record["chunk_id"]),
        document_id="doc_alignment",
        source_entity_id=f"entity::{record['Node1']}",
        target_entity_id=f"entity::{record['Node2']}",
        source_entity_name=str(record["Node1"]),
        target_entity_name=str(record["Node2"]),
        type=str(record["Type"]),
        name=str(record["Relation"]),
        description=str(record["Description"]),
        attributes={},
        source_chunk_ids=source_chunk_ids,
    )


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _entity_signature_from_hetadb(record: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    chunks = _record_chunks(record)
    return (
        str(record.get("NodeName")),
        str(record.get("Description")),
        tuple(sorted(chunks)),
    )


def _entity_signature(entity: ExtractedEntity) -> tuple[str, str, tuple[str, ...]]:
    return (entity.name, entity.description, tuple(sorted(entity.source_chunk_ids)))


def _relation_signature_from_hetadb(
    record: dict[str, Any],
) -> tuple[str, str, str, str, tuple[str, ...]]:
    chunks = _record_chunks(record)
    return (
        str(record.get("Node1")),
        str(record.get("Node2")),
        str(record.get("Relation")),
        str(record.get("Description")),
        tuple(sorted(chunks)),
    )


def _record_chunks(record: dict[str, Any]) -> list[str]:
    chunks = []
    for key in ("chunk_id", "ChunkId"):
        value = record.get(key)
        if not value:
            continue
        if isinstance(value, list):
            chunks.extend(str(chunk) for chunk in value)
        else:
            chunks.append(str(value))
    deduplicated = []
    seen = set()
    for chunk in chunks:
        if chunk in seen:
            continue
        seen.add(chunk)
        deduplicated.append(chunk)
    return deduplicated


def _relation_signature(relation: ExtractedRelation) -> tuple[str, str, str, str, tuple[str, ...]]:
    return (
        relation.source_entity_name,
        relation.target_entity_name,
        relation.name,
        relation.description,
        tuple(sorted(relation.source_chunk_ids)),
    )


def _fake_use_llm(responses: Sequence[Any]):
    queue = list(responses)

    def use_llm(**_: Any) -> str:
        return json.dumps(queue.pop(0), ensure_ascii=False)

    return use_llm


async def _run_framework_entities(
    records: Sequence[dict[str, Any]],
    responses: Sequence[Any],
) -> list[ExtractedEntity]:
    with tempfile.TemporaryDirectory() as tmp:
        store = LocalObjectStore(tmp)
        context = FakeContext(
            {"stores.objects": store, "models.language": FakeLanguageModel(responses)}
        )
        keys = []
        for index, record in enumerate(records):
            if not str(record.get("NodeName", "")).strip():
                continue
            entity = _entity(record, index)
            key = f"entities/{index}.json"
            await store.put(key, entity.to_json_bytes())
            keys.append(key)
        context.set_artifact("entity_keys", tuple(keys))
        await DeduplicateEntities(DeduplicateEntitiesConfig(semantic_merge=False)).run(context)
        return [
            ExtractedEntity.from_json(await store.get(key))
            for key in context.artifacts["deduplicated_entity_keys"]
        ]


async def _run_framework_relations(
    records: Sequence[dict[str, Any]],
    responses: Sequence[Any],
    entity_id_mapping: dict[str, str] | None = None,
) -> list[ExtractedRelation]:
    with tempfile.TemporaryDirectory() as tmp:
        store = LocalObjectStore(tmp)
        context = FakeContext(
            {"stores.objects": store, "models.language": FakeLanguageModel(responses)}
        )
        keys = []
        for index, record in enumerate(records):
            if (
                not str(record.get("Node1", "")).strip()
                or not str(record.get("Node2", "")).strip()
            ):
                continue
            relation = _relation(record, index)
            key = f"relations/{index}.json"
            await store.put(key, relation.to_json_bytes())
            keys.append(key)
        context.set_artifact("relation_keys", tuple(keys))
        if entity_id_mapping is not None:
            context.set_artifact("entity_id_mapping", entity_id_mapping)
        await DeduplicateRelations(
            DeduplicateRelationsConfig(
                semantic_merge=False,
                entity_id_mapping_artifact="entity_id_mapping"
                if entity_id_mapping is not None
                else None,
            )
        ).run(context)
        return [
            ExtractedRelation.from_json(await store.get(key))
            for key in context.artifacts["deduplicated_relation_keys"]
        ]


def _run_hetadb_entities(
    records: Sequence[dict[str, Any]],
    responses: Sequence[Any],
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        input_path = base / "nodes.jsonl"
        output_path = base / "nodes_out.jsonl"
        _write_jsonl(input_path, records)
        dedup_nodes(
            use_llm=_fake_use_llm(responses),
            dedup_template="{entity_block}",
            input_path=input_path,
            output_path=output_path,
            workers=1,
            max_rounds=10,
            llm_batch_size=20,
        )
        return _read_jsonl(output_path)


def _run_hetadb_relations(
    records: Sequence[dict[str, Any]],
    responses: Sequence[Any],
    name_mapping: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        input_path = base / "rels.jsonl"
        mapping_path = base / "mapping.json"
        output_path = base / "rels_out.jsonl"
        _write_jsonl(input_path, records)
        mapping_path.write_text(
            json.dumps(name_mapping or {}, ensure_ascii=False),
            encoding="utf-8",
        )
        dedup_relations(
            use_llm=_fake_use_llm(responses),
            rel_dedup_prompt="{relation_block}",
            input_path=input_path,
            mapping_path=mapping_path,
            output_path=output_path,
            workers=1,
            max_rounds=10,
            llm_batch_size=20,
        )
        return _read_jsonl(output_path)


def _compare_entities(
    name: str,
    records: Sequence[dict[str, Any]],
    responses: Sequence[Any],
) -> bool:
    hetadb = _run_hetadb_entities(records, responses)
    framework = asyncio.run(_run_framework_entities(records, responses))
    left = sorted(_entity_signature_from_hetadb(record) for record in hetadb)
    right = sorted(_entity_signature(entity) for entity in framework)
    ok = left == right
    print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  hetadb   =", left)
        print("  framework=", right)
    return ok


def _compare_relations(
    name: str,
    records: Sequence[dict[str, Any]],
    responses: Sequence[Any],
    name_mapping: dict[str, str] | None = None,
    entity_id_mapping: dict[str, str] | None = None,
) -> bool:
    hetadb = _run_hetadb_relations(records, responses, name_mapping=name_mapping)
    framework = asyncio.run(
        _run_framework_relations(
            records,
            responses,
            entity_id_mapping=entity_id_mapping,
        )
    )
    left = sorted(_relation_signature_from_hetadb(record) for record in hetadb)
    right = sorted(_relation_signature(relation) for relation in framework)
    ok = left == right
    print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  hetadb   =", left)
        print("  framework=", right)
    return ok


def main() -> int:
    cases = [
        (
            "entity_no_duplicate",
            _compare_entities,
            [_node_record("北京", "北京是城市。", "c1")],
            [],
        ),
        (
            "entity_exact_merge",
            _compare_entities,
            [
                _node_record("上海", "上海是城市。", "c1"),
                _node_record("上海", "上海是直辖市。", "c2"),
            ],
            [{"NodeName": "上海", "Description": "上海是城市和直辖市。"}],
        ),
        (
            "entity_case_sensitive",
            _compare_entities,
            [
                _node_record("Apple", "Apple 是公司。", "c1"),
                _node_record("apple", "apple 是水果。", "c2"),
            ],
            [],
        ),
        (
            "entity_split",
            _compare_entities,
            [
                _node_record("Apple", "Apple 是公司。", "c1"),
                _node_record("Apple", "Apple 也可能是水果。", "c2"),
            ],
            [
                [
                    {"NodeName": "Apple", "Description": "Apple 是公司。"},
                    {
                        "NodeName": "Apple fruit",
                        "Description": "Apple 也可能是水果。",
                        "chunk_id": "c2",
                    },
                ]
            ],
        ),
        (
            "entity_multi_round",
            _compare_entities,
            [
                _node_record("A", "A1", "c1"),
                _node_record("A", "A2", "c2"),
                _node_record("A", "A3", "c3"),
            ],
            [{"NodeName": "A", "Description": "A123"}],
        ),
        (
            "relation_no_duplicate",
            _compare_relations,
            [_relation_record("上海", "徐汇", "包含", "空间", "上海包含徐汇。", "c1")],
            [],
        ),
        (
            "relation_exact_merge",
            _compare_relations,
            [
                _relation_record("上海", "徐汇", "包含", "空间", "上海包含徐汇。", "c1"),
                _relation_record("上海", "徐汇", "包含", "空间", "徐汇属于上海。", "c2"),
            ],
            [
                {
                    "Node1": "上海",
                    "Node2": "徐汇",
                    "Relation": "包含",
                    "Type": "空间",
                    "Description": "上海包含徐汇。",
                }
            ],
        ),
        (
            "relation_case_sensitive",
            _compare_relations,
            [
                _relation_record("A", "B", "uses", "依赖", "A uses B.", "c1"),
                _relation_record("A", "B", "Uses", "依赖", "A Uses B.", "c2"),
            ],
            [],
        ),
        (
            "relation_split",
            _compare_relations,
            [
                _relation_record("A", "B", "关联", "泛化", "A 与 B 有业务关联。", "c1"),
                _relation_record("A", "B", "关联", "泛化", "A 与 B 有技术关联。", "c2"),
            ],
            [
                [
                    {
                        "Node1": "A",
                        "Node2": "B",
                        "Relation": "业务关联",
                        "Type": "泛化",
                        "Description": "A 与 B 有业务关联。",
                    },
                    {
                        "Node1": "A",
                        "Node2": "C",
                        "Relation": "技术关联",
                        "Type": "泛化",
                        "Description": "A 与 C 有技术关联。",
                        "chunk_id": "c2",
                    },
                ]
            ],
        ),
        (
            "relation_multi_round",
            _compare_relations,
            [
                _relation_record("A", "B", "包含", "空间", "r1", "c1"),
                _relation_record("A", "B", "包含", "空间", "r2", "c2"),
                _relation_record("A", "B", "包含", "空间", "r3", "c3"),
            ],
            [{"Node1": "A", "Node2": "B", "Relation": "包含", "Type": "空间", "Description": "r123"}],
        ),
        (
            "entity_trimmed_name_merge",
            _compare_entities,
            [
                _node_record(" 深圳 ", "深圳是城市。", "c1"),
                _node_record("深圳", "深圳是经济特区。", "c2"),
            ],
            [{"NodeName": "深圳", "Description": "深圳是城市和经济特区。"}],
        ),
        (
            "entity_different_internal_spaces_do_not_merge",
            _compare_entities,
            [
                _node_record("New York", "New York 是城市。", "c1"),
                _node_record("New  York", "New  York 有额外空格。", "c2"),
            ],
            [],
        ),
        (
            "entity_batch_accumulated_merge",
            _compare_entities,
            [
                _node_record("Batch", "b1", "c1"),
                _node_record("Batch", "b2", "c2"),
                _node_record("Batch", "b3", "c3"),
            ],
            [
                {"NodeName": "Batch", "Description": "b12"},
                {"NodeName": "Batch", "Description": "b123"},
            ],
        ),
        (
            "entity_split_same_name_continues_rounds",
            _compare_entities,
            [
                _node_record("Topic", "t1", "c1"),
                _node_record("Topic", "t2", "c2"),
            ],
            [
                [
                    {"NodeName": "Topic", "Description": "t1"},
                    {"NodeName": "Topic", "Description": "t2", "chunk_id": "c2"},
                ],
                {"NodeName": "Topic", "Description": "t12"},
            ],
        ),
        (
            "entity_no_duplicate_mixed_names",
            _compare_entities,
            [
                _node_record("甲", "甲实体。", "c1"),
                _node_record("乙", "乙实体。", "c2"),
                _node_record("丙", "丙实体。", "c3"),
            ],
            [],
        ),
        (
            "relation_trimmed_endpoint_merge",
            _compare_relations,
            [
                _relation_record(" 上海 ", " 徐汇 ", "包含", "空间", "r1", "c1"),
                _relation_record("上海", "徐汇", "包含", "空间", "r2", "c2"),
            ],
            [{"Node1": "上海", "Node2": "徐汇", "Relation": "包含", "Type": "空间", "Description": "r12"}],
        ),
        (
            "relation_different_type_no_merge",
            _compare_relations,
            [
                _relation_record("A", "B", "包含", "空间", "r1", "c1"),
                _relation_record("A", "B", "包含", "组织", "r2", "c2"),
            ],
            [],
        ),
        (
            "relation_different_relation_no_merge",
            _compare_relations,
            [
                _relation_record("A", "B", "包含", "空间", "r1", "c1"),
                _relation_record("A", "B", "下辖", "空间", "r2", "c2"),
            ],
            [],
        ),
        (
            "relation_batch_accumulated_merge",
            _compare_relations,
            [
                _relation_record("A", "B", "包含", "空间", "r1", "c1"),
                _relation_record("A", "B", "包含", "空间", "r2", "c2"),
                _relation_record("A", "B", "包含", "空间", "r3", "c3"),
            ],
            [
                {"Node1": "A", "Node2": "B", "Relation": "包含", "Type": "空间", "Description": "r12"},
                {
                    "Node1": "A",
                    "Node2": "B",
                    "Relation": "包含",
                    "Type": "空间",
                    "Description": "r123",
                },
            ],
        ),
        (
            "relation_split_collision_filtered",
            _compare_relations,
            [
                _relation_record("A", "B", "关联", "泛化", "r1", "c1"),
                _relation_record("A", "B", "关联", "泛化", "r2", "c2"),
                _relation_record("A", "C", "技术关联", "泛化", "existing", "c3"),
            ],
            [
                [
                    {
                        "Node1": "A",
                        "Node2": "B",
                        "Relation": "业务关联",
                        "Type": "泛化",
                        "Description": "r12",
                    },
                    {
                        "Node1": "A",
                        "Node2": "C",
                        "Relation": "技术关联",
                        "Type": "泛化",
                        "Description": "split should be filtered",
                        "chunk_id": "c2",
                    },
                ]
            ],
        ),
        (
            "entity_chunk_list_preserved",
            _compare_entities,
            [
                {
                    **_node_record("ListChunk", "l1", ["c1", "c2"]),
                },
                _node_record("ListChunk", "l2", "c3"),
            ],
            [{"NodeName": "ListChunk", "Description": "l123"}],
        ),
        (
            "entity_llm_chunkid_field_preserved",
            _compare_entities,
            [
                _node_record("ChunkIdEntity", "x1", "c1"),
                _node_record("ChunkIdEntity", "x2", "c2"),
            ],
            [
                {
                    "NodeName": "ChunkIdEntity",
                    "Description": "x12",
                    "ChunkId": ["cx"],
                }
            ],
        ),
        (
            "entity_empty_name_skipped",
            _compare_entities,
            [
                _node_record("", "empty should skip", "c1"),
                _node_record("Valid", "valid remains", "c2"),
            ],
            [],
        ),
        (
            "entity_dict_response_without_wrapper",
            _compare_entities,
            [
                _node_record("Plain", "p1", "c1"),
                _node_record("Plain", "p2", "c2"),
            ],
            [{"NodeName": "Plain", "Description": "p12", "Extra": "kept"}],
        ),
        (
            "entity_split_chunkid_field_preserved",
            _compare_entities,
            [
                _node_record("SplitChunk", "s1", "c1"),
                _node_record("SplitChunk", "s2", "c2"),
            ],
            [
                [
                    {"NodeName": "SplitChunk", "Description": "s1"},
                    {
                        "NodeName": "SplitChunkOther",
                        "Description": "s2",
                        "ChunkId": ["c2x"],
                    },
                ]
            ],
        ),
        (
            "relation_chunk_list_preserved",
            _compare_relations,
            [
                {
                    **_relation_record("A", "B", "连接", "拓扑", "l1", ["c1", "c2"]),
                },
                _relation_record("A", "B", "连接", "拓扑", "l2", "c3"),
            ],
            [
                {
                    "Node1": "A",
                    "Node2": "B",
                    "Relation": "连接",
                    "Type": "拓扑",
                    "Description": "l123",
                }
            ],
        ),
        (
            "relation_llm_chunkid_field_preserved",
            _compare_relations,
            [
                _relation_record("A", "B", "依赖", "技术", "d1", "c1"),
                _relation_record("A", "B", "依赖", "技术", "d2", "c2"),
            ],
            [
                {
                    "Node1": "A",
                    "Node2": "B",
                    "Relation": "依赖",
                    "Type": "技术",
                    "Description": "d12",
                    "ChunkId": ["cx"],
                }
            ],
        ),
        (
            "relation_reverse_direction_no_merge",
            _compare_relations,
            [
                _relation_record("A", "B", "包含", "空间", "A 包含 B", "c1"),
                _relation_record("B", "A", "包含", "空间", "B 包含 A", "c2"),
            ],
            [],
        ),
        (
            "relation_empty_endpoint_skipped",
            _compare_relations,
            [
                _relation_record("", "B", "包含", "空间", "skip", "c1"),
                _relation_record("A", "B", "包含", "空间", "keep", "c2"),
            ],
            [],
        ),
        (
            "relation_entity_mapping_merge",
            _compare_relations,
            [
                _relation_record("上海市", "徐汇区", "包含", "空间", "r1", "c1"),
                _relation_record("上海", "徐汇区", "包含", "空间", "r2", "c2"),
            ],
            [
                {
                    "Node1": "上海市",
                    "Node2": "徐汇区",
                    "Relation": "包含",
                    "Type": "空间",
                    "Description": "r12",
                }
            ],
            {"上海": "上海市"},
            {"entity::上海": "entity::上海市"},
        ),
        (
            "entity_list_main_not_first",
            _compare_entities,
            [
                _node_record("Main", "m1", "c1"),
                _node_record("Main", "m2", "c2"),
            ],
            [
                [
                    {
                        "NodeName": "Other",
                        "Description": "split other",
                        "chunk_id": "c2",
                    },
                    {"NodeName": "Main", "Description": "merged main"},
                ]
            ],
        ),
        (
            "entity_list_no_matching_uses_first",
            _compare_entities,
            [
                _node_record("Original", "o1", "c1"),
                _node_record("Original", "o2", "c2"),
            ],
            [
                [
                    {"NodeName": "First", "Description": "first result"},
                    {
                        "NodeName": "Second",
                        "Description": "second split",
                        "chunk_id": "c2",
                    },
                ]
            ],
        ),
        (
            "entity_duplicate_chunk_ids_deduped",
            _compare_entities,
            [
                _node_record("DupChunk", "d1", "c1"),
                _node_record("DupChunk", "d2", "c1"),
            ],
            [{"NodeName": "DupChunk", "Description": "d12"}],
        ),
        (
            "entity_batch_size_21_accumulates",
            _compare_entities,
            [
                _node_record("LargeBatch", f"lb{i}", f"c{i}")
                for i in range(21)
            ],
            [
                {"NodeName": "LargeBatch", "Description": "lb0-19"},
                {"NodeName": "LargeBatch", "Description": "lb0-20"},
            ],
        ),
        (
            "entity_split_collides_then_next_round",
            _compare_entities,
            [
                _node_record("Alpha", "a1", "c1"),
                _node_record("Alpha", "a2", "c2"),
                _node_record("Beta", "b0", "c3"),
            ],
            [
                [
                    {"NodeName": "Alpha", "Description": "a12"},
                    {
                        "NodeName": "Beta",
                        "Description": "beta split",
                        "chunk_id": "c2",
                    },
                ],
                {"NodeName": "Beta", "Description": "beta merged"},
            ],
        ),
        (
            "relation_list_main_not_first",
            _compare_relations,
            [
                _relation_record("A", "B", "关联", "类型", "r1", "c1"),
                _relation_record("A", "B", "关联", "类型", "r2", "c2"),
            ],
            [
                [
                    {
                        "Node1": "A",
                        "Node2": "C",
                        "Relation": "拆分",
                        "Type": "类型",
                        "Description": "split",
                        "chunk_id": "c2",
                    },
                    {
                        "Node1": "A",
                        "Node2": "B",
                        "Relation": "关联",
                        "Type": "类型",
                        "Description": "main",
                    },
                ]
            ],
        ),
        (
            "relation_list_no_matching_uses_first",
            _compare_relations,
            [
                _relation_record("A", "B", "关系", "类型", "r1", "c1"),
                _relation_record("A", "B", "关系", "类型", "r2", "c2"),
            ],
            [
                [
                    {
                        "Node1": "X",
                        "Node2": "Y",
                        "Relation": "first",
                        "Type": "类型",
                        "Description": "first result",
                    },
                    {
                        "Node1": "P",
                        "Node2": "Q",
                        "Relation": "second",
                        "Type": "类型",
                        "Description": "second split",
                        "chunk_id": "c2",
                    },
                ]
            ],
        ),
        (
            "relation_duplicate_chunk_ids_deduped",
            _compare_relations,
            [
                _relation_record("A", "B", "重复", "类型", "d1", "c1"),
                _relation_record("A", "B", "重复", "类型", "d2", "c1"),
            ],
            [
                {
                    "Node1": "A",
                    "Node2": "B",
                    "Relation": "重复",
                    "Type": "类型",
                    "Description": "d12",
                }
            ],
        ),
        (
            "relation_batch_size_21_accumulates",
            _compare_relations,
            [
                _relation_record("A", "B", "大批量", "类型", f"r{i}", f"c{i}")
                for i in range(21)
            ],
            [
                {
                    "Node1": "A",
                    "Node2": "B",
                    "Relation": "大批量",
                    "Type": "类型",
                    "Description": "r0-19",
                },
                {
                    "Node1": "A",
                    "Node2": "B",
                    "Relation": "大批量",
                    "Type": "类型",
                    "Description": "r0-20",
                },
            ],
        ),
        (
            "relation_target_mapping_merge",
            _compare_relations,
            [
                _relation_record("上海市", "徐汇", "包含", "空间", "r1", "c1"),
                _relation_record("上海市", "徐汇区", "包含", "空间", "r2", "c2"),
            ],
            [
                {
                    "Node1": "上海市",
                    "Node2": "徐汇区",
                    "Relation": "包含",
                    "Type": "空间",
                    "Description": "r12",
                }
            ],
            {"徐汇": "徐汇区"},
            {"entity::徐汇": "entity::徐汇区"},
        ),
        (
            "entity_empty_list_response_falls_back",
            _compare_entities,
            [
                _node_record("EmptyList", "e1", "c1"),
                _node_record("EmptyList", "e2", "c2"),
            ],
            [[] for _ in range(10)],
        ),
        (
            "entity_non_dict_items_ignored",
            _compare_entities,
            [
                _node_record("MixedList", "m1", "c1"),
                _node_record("MixedList", "m2", "c2"),
            ],
            [
                [
                    "bad",
                    123,
                    {"NodeName": "MixedList", "Description": "mixed merged"},
                ]
            ],
        ),
        (
            "entity_missing_description_defaults_empty",
            _compare_entities,
            [
                _node_record("NoDesc", "n1", "c1"),
                _node_record("NoDesc", "n2", "c2"),
            ],
            [{"NodeName": "NoDesc"}],
        ),
        (
            "entity_none_chunk_ignored",
            _compare_entities,
            [
                _node_record("NoneChunk", "n1", None),
                _node_record("NoneChunk", "n2", "c2"),
            ],
            [{"NodeName": "NoneChunk", "Description": "n12"}],
        ),
        (
            "entity_duplicate_with_empty_split_name_skips_next_round",
            _compare_entities,
            [
                _node_record("SplitEmpty", "s1", "c1"),
                _node_record("SplitEmpty", "s2", "c2"),
            ],
            [
                [
                    {"NodeName": "SplitEmpty", "Description": "s12"},
                    {"NodeName": "", "Description": "empty split", "chunk_id": "c2"},
                ]
            ],
        ),
        (
            "relation_empty_list_response_falls_back",
            _compare_relations,
            [
                _relation_record("A", "B", "空列表", "类型", "e1", "c1"),
                _relation_record("A", "B", "空列表", "类型", "e2", "c2"),
            ],
            [[] for _ in range(10)],
        ),
        (
            "relation_non_dict_items_ignored",
            _compare_relations,
            [
                _relation_record("A", "B", "混合", "类型", "m1", "c1"),
                _relation_record("A", "B", "混合", "类型", "m2", "c2"),
            ],
            [
                [
                    "bad",
                    123,
                    {
                        "Node1": "A",
                        "Node2": "B",
                        "Relation": "混合",
                        "Type": "类型",
                        "Description": "mixed merged",
                    },
                ]
            ],
        ),
        (
            "relation_missing_description_defaults_empty",
            _compare_relations,
            [
                _relation_record("A", "B", "无描述", "类型", "n1", "c1"),
                _relation_record("A", "B", "无描述", "类型", "n2", "c2"),
            ],
            [{"Node1": "A", "Node2": "B", "Relation": "无描述", "Type": "类型"}],
        ),
        (
            "relation_split_missing_node_falls_back",
            _compare_relations,
            [
                _relation_record("A", "B", "缺节点", "类型", "s1", "c1"),
                _relation_record("A", "B", "缺节点", "类型", "s2", "c2"),
            ],
            [
                [
                    {
                        "Node1": "A",
                        "Node2": "B",
                        "Relation": "缺节点",
                        "Type": "类型",
                        "Description": "main",
                    },
                    {
                        "Relation": "split",
                        "Type": "类型",
                        "Description": "split missing nodes",
                        "chunk_id": "c2",
                    },
                ]
            ],
        ),
        (
            "relation_source_and_target_mapping_merge",
            _compare_relations,
            [
                _relation_record("沪", "徐汇", "包含", "空间", "r1", "c1"),
                _relation_record("上海市", "徐汇区", "包含", "空间", "r2", "c2"),
            ],
            [
                {
                    "Node1": "上海市",
                    "Node2": "徐汇区",
                    "Relation": "包含",
                    "Type": "空间",
                    "Description": "r12",
                }
            ],
            {"沪": "上海市", "徐汇": "徐汇区"},
            {
                "entity::沪": "entity::上海市",
                "entity::徐汇": "entity::徐汇区",
            },
        ),
    ]

    passed = 0
    for case in cases:
        name, compare, records, responses, *extra = case
        if compare(name, records, responses, *extra):
            passed += 1
    print(f"summary: {passed}/{len(cases)} aligned")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
