# Object Stores

Object Stores are Heta's unified interface for file-like objects. They only handle `key -> bytes`; they do not understand business objects such as documents, chunks, entities, or relations.

Typical objects include:

```text
raw files
parsed documents
chunk JSON
images
model output artifacts
export files
runtime metadata
```

## Quick Start

```python
from heta_framework.common.stores import LocalObjectStore

store = LocalObjectStore("./heta")

await store.put("parsed/doc1.md", b"# Doc 1")
data = await store.get("parsed/doc1.md")

exists = await store.exists("parsed/doc1.md")
objects = await store.list("parsed/")
await store.delete("parsed/doc1.md")
```

Object keys use relative POSIX paths:

```text
raw/paper.pdf
parsed/paper.json
chunks/chunk_001.json
images/page_1.png
extract/entities.jsonl
```

ObjectStore rejects empty keys, absolute paths, backslashes, and `..` so local filesystem and object storage behavior stay consistent.

## Implementations

| Store | Use |
| --- | --- |
| `LocalObjectStore` | Local directory implementation for development, tests, and single-machine deployment. |
| `S3ObjectStore` | S3-compatible implementation for AWS S3, MinIO, Ceph, and private object stores. |

## S3-Compatible Storage

Install:

```bash
pip install "heta[s3]"
```

Example:

```python
from heta_framework.common.stores import S3ObjectStore

store = S3ObjectStore(
    bucket="heta",
    prefix="kb/papers",
    endpoint_url="http://10.6.8.115:9000",
    region="us-east-1",
    access_key_id="minioadmin",
    secret_access_key="minioadmin",
    addressing_style="path",
)

await store.put("parsed/doc1.md", b"# Doc 1")
data = await store.get("parsed/doc1.md")
```

Application code still uses logical keys such as:

```text
parsed/doc1.md
```

The actual backend key is prefixed:

```text
kb/papers/parsed/doc1.md
```

## Methods

```python
await store.put(key, data)
data = await store.get(key)
exists = await store.exists(key)
objects = await store.list(prefix)
await store.delete(key)
await store.aclose()
```

| Method | Meaning |
| --- | --- |
| `put` | Write bytes. |
| `get` | Read bytes; raises `FileNotFoundError` when missing. |
| `exists` | Check whether an object exists. |
| `list` | List objects by prefix and return `list[ObjectInfo]`. |
| `delete` | Delete an object; missing objects are ignored. |
| `aclose` | Release resources held by the store. |

## Scope

Object Stores handle object reads and writes, local/S3 adapters, key validation, and unified listing.

They do not handle document semantics, schemas, versioning, lineage, permissions, transactions, or the `KnowledgeBase` lifecycle.
