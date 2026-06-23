import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.common.stores import (  # noqa: E402
    LocalObjectStore,
    ObjectInfo,
    ObjectStoreProtocol,
    S3ObjectStore,
)
from heta_framework.common.stores.object.types import (  # noqa: E402
    join_object_key,
    validate_object_key,
    validate_object_prefix,
)


def test_local_object_store_satisfies_protocol(tmp_path):
    assert isinstance(LocalObjectStore(tmp_path), ObjectStoreProtocol)


def test_local_object_store_puts_gets_lists_and_deletes(tmp_path):
    async def run():
        store = LocalObjectStore(tmp_path)
        await store.put("parsed/doc1.md", b"hello")
        await store.put("parsed/nested/doc2.md", b"heta")
        await store.put("parsed2/doc3.md", b"other")
        await store.put("raw/doc1.pdf", b"%PDF")

        data = await store.get("parsed/doc1.md")
        exists = await store.exists("parsed/doc1.md")
        missing = await store.exists("parsed/missing.md")
        objects = await store.list("parsed/")

        await store.delete("parsed/doc1.md")
        deleted = await store.exists("parsed/doc1.md")
        await store.aclose()
        return data, exists, missing, objects, deleted

    data, exists, missing, objects, deleted = asyncio.run(run())

    assert data == b"hello"
    assert exists is True
    assert missing is False
    assert [item.key for item in objects] == ["parsed/doc1.md", "parsed/nested/doc2.md"]
    assert all(item.size is not None for item in objects)
    assert deleted is False


def test_local_object_store_rejects_unsafe_keys(tmp_path):
    async def run(key):
        store = LocalObjectStore(tmp_path)
        try:
            await store.put(key, b"bad")
        finally:
            await store.aclose()

    for key in ["", "/absolute.txt", "../escape.txt", "a/../b.txt", r"a\b.txt"]:
        with pytest.raises(ValueError):
            asyncio.run(run(key))


def test_object_key_helpers_validate_and_join():
    assert validate_object_key("parsed/doc1.md") == "parsed/doc1.md"
    assert validate_object_prefix("parsed/") == "parsed"
    assert validate_object_prefix("") == ""
    assert join_object_key("kb/papers", "parsed/doc1.md") == "kb/papers/parsed/doc1.md"


def test_object_info_validates_key():
    with pytest.raises(ValueError):
        ObjectInfo(key="../bad")


def test_s3_object_store_uses_prefixed_keys():
    async def run():
        client = FakeS3Client()
        store = S3ObjectStore(
            bucket="heta",
            prefix="kb/papers",
            endpoint_url="http://localhost:9000",
            access_key_id="access",
            secret_access_key="secret",
            addressing_style="path",
            client=client,
        )

        await store.put("parsed/doc1.md", b"hello")
        data = await store.get("parsed/doc1.md")
        exists = await store.exists("parsed/doc1.md")
        missing = await store.exists("parsed/missing.md")
        objects = await store.list("parsed/")
        await store.delete("parsed/doc1.md")
        deleted = await store.exists("parsed/doc1.md")
        await store.aclose()
        return data, exists, missing, objects, deleted, client.closed

    data, exists, missing, objects, deleted, closed = asyncio.run(run())

    assert data == b"hello"
    assert exists is True
    assert missing is False
    assert [item.key for item in objects] == ["parsed/doc1.md"]
    assert objects[0].etag == "abc"
    assert deleted is False
    assert closed is True


class FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3Error(Exception):
    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}


class FakePaginator:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects

    def paginate(self, *, Bucket: str, Prefix: str):
        del Bucket
        contents = []
        for key, data in self._objects.items():
            if key.startswith(Prefix):
                contents.append(
                    {
                        "Key": key,
                        "Size": len(data),
                        "LastModified": datetime(2026, 1, 1, tzinfo=UTC),
                        "ETag": '"abc"',
                    }
                )
        return [{"Contents": contents}]


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.closed = False

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        assert Bucket == "heta"
        self.objects[Key] = Body

    def get_object(self, *, Bucket: str, Key: str):
        assert Bucket == "heta"
        if Key not in self.objects:
            raise FakeS3Error("NoSuchKey")
        return {"Body": FakeBody(self.objects[Key])}

    def head_object(self, *, Bucket: str, Key: str) -> None:
        assert Bucket == "heta"
        if Key not in self.objects:
            raise FakeS3Error("404")

    def get_paginator(self, name: str) -> FakePaginator:
        assert name == "list_objects_v2"
        return FakePaginator(self.objects)

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        assert Bucket == "heta"
        self.objects.pop(Key, None)

    def close(self) -> None:
        self.closed = True
