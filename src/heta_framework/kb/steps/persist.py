"""Persist chunk artifacts into a SQL store."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.sql import SQLStoreProtocol
from heta_framework.kb.chunking import ParsedChunk
from heta_framework.kb.search import SearchAsset
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import StepCapabilities, StepRequirements, store_ref


SQLDialect = Literal["generic", "postgresql"]


@dataclass(frozen=True)
class ChunkTableNames:
    """SQL table names used by chunk persistence."""

    chunks: str = "chunks"

    def __post_init__(self) -> None:
        _validate_identifier(self.chunks, field_name="table_names.chunks")


@dataclass(frozen=True)
class PersistChunksConfig:
    """Configuration for PersistChunks."""

    table_names: ChunkTableNames = field(default_factory=ChunkTableNames)
    dialect: SQLDialect = "generic"
    object_store: str | None = None
    sql_store: str | None = None
    chunk_keys_artifact: str = "rechunked_chunk_keys"

    def __post_init__(self) -> None:
        if self.dialect not in {"generic", "postgresql"}:
            raise ValueError("dialect must be one of: generic, postgresql")
        if self.chunk_keys_artifact.strip() == "":
            raise ValueError("chunk_keys_artifact must not be empty")


@dataclass(frozen=True)
class PersistChunksResult:
    """Artifacts produced by PersistChunks."""

    table: str
    chunk_count: int


class PersistChunks:
    """Persist ParsedChunk JSON objects to a SQL chunk table."""

    name = "persist_chunks"

    def __init__(self, config: PersistChunksConfig | None = None) -> None:
        self.config = config or PersistChunksConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components and artifacts required by this step."""
        return StepRequirements(
            components=frozenset(
                {
                    store_ref("objects", self.config.object_store),
                    store_ref("sql", self.config.sql_store),
                }
            ),
            artifacts=frozenset({self.config.chunk_keys_artifact}),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts produced by this step."""
        sql_store_ref = store_ref("sql", self.config.sql_store)
        return StepCapabilities(
            artifacts=frozenset({"persist_chunks_result"}),
            queries=frozenset({"keyword_search"}),
            search_assets=(
                SearchAsset(
                    kind="chunk_text_index",
                    name=self.config.table_names.chunks,
                    store=sql_store_ref.key,
                    metadata={
                        "table": self.config.table_names.chunks,
                        "dialect": self.config.dialect,
                        "id_field": "chunk_id",
                        "text_field": "content_text",
                        "document_id_field": "document_id",
                        "source_id_field": "source_id",
                        "source_chunk_field": "source_chunk",
                        "metadata_field": "metadata_json",
                    },
                ),
            ),
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Create the chunk table if needed and insert chunks."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        sql_store = _require_sql_store(
            context.get_component(store_ref("sql", self.config.sql_store).key)
        )
        chunk_keys = tuple(context.get_artifact(self.config.chunk_keys_artifact))
        chunks = [ParsedChunk.from_json(await object_store.get(key)) for key in chunk_keys]

        async with sql_store.transaction() as tx:
            for statement in _create_table_statements(
                self.config.table_names.chunks,
                self.config.dialect,
            ):
                await tx.execute(statement)
            for chunk in chunks:
                await tx.execute(
                    _insert_statement(self.config.table_names.chunks, self.config.dialect),
                    _chunk_parameters(chunk),
                )

        result = PersistChunksResult(table=self.config.table_names.chunks, chunk_count=len(chunks))
        context.set_artifact("persist_chunks_result", result)


def _create_table_statements(table: str, dialect: SQLDialect) -> tuple[str, ...]:
    if dialect == "postgresql":
        return (
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id SERIAL PRIMARY KEY,
                chunk_id VARCHAR(128),
                document_id TEXT,
                content_text TEXT,
                content_tsv tsvector,
                source_id TEXT,
                source_chunk TEXT,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            f"CREATE INDEX IF NOT EXISTS idx_{table}_chunk_id ON {table}(chunk_id)",
            f"CREATE INDEX IF NOT EXISTS idx_{table}_tsv ON {table} USING GIN(content_tsv)",
        )
    return (
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            chunk_id VARCHAR(128),
            document_id VARCHAR(256),
            content_text TEXT,
            source_id TEXT,
            source_chunk TEXT,
            metadata_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    )


def _insert_statement(table: str, dialect: SQLDialect) -> str:
    if dialect == "postgresql":
        return f"""
        INSERT INTO {table}
        (chunk_id, document_id, content_text, content_tsv, source_id, source_chunk, metadata_json)
        VALUES
        (
            :chunk_id,
            :document_id,
            :content_text,
            to_tsvector('simple', :content_text),
            :source_id,
            :source_chunk,
            :metadata_json
        )
        """
    return f"""
    INSERT INTO {table}
    (chunk_id, document_id, content_text, source_id, source_chunk, metadata_json)
    VALUES
    (:chunk_id, :document_id, :content_text, :source_id, :source_chunk, :metadata_json)
    """


def _chunk_parameters(chunk: ParsedChunk) -> dict[str, str]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "content_text": chunk.text,
        "source_id": chunk.source.key,
        "source_chunk": json.dumps(list(chunk.parent_chunk_ids or (chunk.chunk_id,))),
        "metadata_json": json.dumps(
            {
                "source": asdict(chunk.source),
                "page_index": chunk.page_index,
                "chunk_index": chunk.chunk_index,
                "token_start": chunk.token_start,
                "token_end": chunk.token_end,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def _validate_identifier(value: str, *, field_name: str = "table") -> None:
    if value.strip() == "":
        raise ValueError(f"{field_name} must not be empty")
    if not value.replace("_", "").isalnum() or value[0].isdigit():
        raise ValueError(f"{field_name} must be a simple SQL identifier")


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_sql_store(component: object) -> SQLStoreProtocol:
    if not isinstance(component, SQLStoreProtocol):
        raise TypeError("stores.sql must satisfy SQLStoreProtocol")
    return component
