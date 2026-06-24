import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.stores import (  # noqa: E402
    InMemoryVectorStore,
    LocalObjectStore,
    SQLStore,
    VectorCollectionConfig,
)
from heta_framework.kb import (  # noqa: E402
    BuildGraph,
    BuildGraphConfig,
    ChunkTableNames,
    ChunkVectorCollections,
    EmbedChunks,
    GraphTableNames,
    GraphVectorCollections,
    IndexVectors,
    IndexVectorsConfig,
    KnowledgeBase,
    KnowledgeRecipe,
    KnowledgeStores,
    MergeChunks,
    MergeChunksConfig,
    ParseDocuments,
    PersistChunks,
    PersistChunksConfig,
    RecipeRunRecord,
    RechunkDocuments,
    SplitDocuments,
)


def test_knowledge_base_delete_removes_derived_resources_and_keeps_raw(tmp_path: Path):
    async def run():
        object_store = LocalObjectStore(tmp_path / "objects")
        sql_store = SQLStore(f"sqlite:///{tmp_path / 'kb.sqlite3'}")
        vector_store = InMemoryVectorStore()

        object_keys = (
            "parsed/doc.json",
            "chunks/chunk.json",
            "embeddings/chunk.json",
            "merged_chunks/chunk.json",
            "rechunked_chunks/chunk.json",
        )
        await object_store.put("raw/doc.txt", b"raw")
        for key in object_keys:
            await object_store.put(key, b"{}")
        await object_store.put(
            "_heta/knowledge_bases/delete_test/latest_run.json",
            b"{}",
        )
        await object_store.put(
            "_heta/knowledge_bases/delete_test/runs/run_1/state.json",
            b"{}",
        )

        table_names = GraphTableNames(
            entities="delete_test_entities",
            relations="delete_test_relations",
            evidence="delete_test_evidence",
        )
        chunk_tables = ChunkTableNames(chunks="delete_test_chunks")
        for table in (
            chunk_tables.chunks,
            table_names.entities,
            table_names.relations,
            table_names.evidence,
        ):
            await sql_store.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY)")

        chunk_collections = ChunkVectorCollections(chunks="delete_test_chunk_vectors")
        graph_collections = GraphVectorCollections(
            entities="delete_test_graph_entities",
            relations="delete_test_graph_relations",
        )
        merge_collection = "delete_test_merge_vectors"
        for collection in (
            chunk_collections.chunks,
            merge_collection,
            graph_collections.entities,
            graph_collections.relations,
        ):
            await vector_store.create_collection(
                VectorCollectionConfig(name=collection, dimension=2)
            )

        recipe = KnowledgeRecipe(
            stores=KnowledgeStores(
                objects=object_store,
                sql=sql_store,
                vector=vector_store,
            ),
            steps=(
                ParseDocuments(),
                SplitDocuments(),
                EmbedChunks(),
                IndexVectors(IndexVectorsConfig(collection_names=chunk_collections)),
                MergeChunks(MergeChunksConfig(merge_collection=merge_collection)),
                RechunkDocuments(),
                PersistChunks(PersistChunksConfig(table_names=chunk_tables)),
                BuildGraph(
                    BuildGraphConfig(
                        table_names=table_names,
                        vector_collections=graph_collections,
                    )
                ),
            ),
        )
        kb = KnowledgeBase(
            name="delete test",
            description=None,
            recipe=recipe,
            run_record=RecipeRunRecord(
                run_id="run_1",
                status="succeeded",
                started_at="2026-01-01T00:00:00+00:00",
                finished_at="2026-01-01T00:01:00+00:00",
                step_records=(),
                artifacts={
                    "parsed_document_keys": ("parsed/doc.json",),
                    "chunk_keys": ("chunks/chunk.json",),
                    "chunk_embedding_keys": ("embeddings/chunk.json",),
                    "merged_chunk_keys": ("merged_chunks/chunk.json",),
                    "rechunked_chunk_keys": ("rechunked_chunks/chunk.json",),
                },
            ),
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:01:00+00:00",
        )

        dry_run = await kb.delete(dry_run=True)
        assert dry_run.dry_run is True
        assert dry_run.issues == ()
        assert all(target.value != "raw/doc.txt" for target in dry_run.targets)
        assert await object_store.exists("parsed/doc.json")
        assert await vector_store.has_collection(chunk_collections.chunks)
        assert await _table_exists(sql_store, chunk_tables.chunks)

        result = await kb.delete()
        assert result.issues == ()
        assert set(result.deleted_object_keys) == set(object_keys)
        assert set(result.dropped_sql_tables) == {
            chunk_tables.chunks,
            table_names.entities,
            table_names.relations,
            table_names.evidence,
        }
        assert set(result.dropped_vector_collections) == {
            chunk_collections.chunks,
            merge_collection,
            graph_collections.entities,
            graph_collections.relations,
        }

        assert await object_store.exists("raw/doc.txt")
        for key in object_keys:
            assert not await object_store.exists(key)
        assert await object_store.list("_heta/knowledge_bases/delete_test") == []
        for table in (
            chunk_tables.chunks,
            table_names.entities,
            table_names.relations,
            table_names.evidence,
        ):
            assert not await _table_exists(sql_store, table)
        for collection in (
            chunk_collections.chunks,
            merge_collection,
            graph_collections.entities,
            graph_collections.relations,
        ):
            assert not await vector_store.has_collection(collection)

        await object_store.aclose()
        await sql_store.aclose()
        await vector_store.aclose()

    asyncio.run(run())


async def _table_exists(sql_store: SQLStore, table: str) -> bool:
    row = await sql_store.fetch_one(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = :name",
        {"name": table},
    )
    return row is not None
