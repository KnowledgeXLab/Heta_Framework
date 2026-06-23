# Object Stores

Object Stores 提供 Heta 文件类对象的统一读写入口。它只处理 `key -> bytes`，不理解 Document、Chunk、Entity、Relation 等 Heta 业务语义。

典型对象包括原始文件、解析结果、图片、JSONL、CSV、chunk 中间文件、模型输出缓存和导出结果。

## 快速开始

```python
from heta_framework.common.stores import LocalObjectStore

store = LocalObjectStore("./heta")

await store.put("parsed/doc1.md", b"# Doc 1")
data = await store.get("parsed/doc1.md")

exists = await store.exists("parsed/doc1.md")
objects = await store.list("parsed/")
await store.delete("parsed/doc1.md")
```

`key` 使用相对 POSIX 路径：

```text
raw/paper.pdf
parsed/paper.jsonl
chunks/paper.jsonl
images/page_1.png
extract/entities.jsonl
```

ObjectStore 会拒绝空 key、绝对路径、反斜杠和 `..`，避免本地存储和对象存储行为不一致。

## S3 兼容存储

S3ObjectStore 使用 S3 兼容协议接入 AWS S3、MinIO、Ceph RGW 以及支持 S3 兼容模式的对象存储。

安装：

```bash
pip install "heta[s3]"
```

示例：

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

上层传入的 key 仍然是：

```text
parsed/doc1.md
```

实际后端 key 会变成：

```text
kb/papers/parsed/doc1.md
```

## 核心对象

| 对象 | 说明 |
| --- | --- |
| `ObjectStoreProtocol` | 对象存储能力协议。 |
| `LocalObjectStore` | 本地目录实现，适合开发、测试和单机部署。 |
| `S3ObjectStore` | S3 兼容实现，适合 AWS S3、MinIO、Ceph 和私有云对象存储。 |
| `ObjectInfo` | `list` 返回的对象基础信息。 |

## 方法

```python
await store.put(key, data)
data = await store.get(key)
await store.exists(key)
await store.list(prefix)
await store.delete(key)
```

| 方法 | 说明 |
| --- | --- |
| `put` | 写入 bytes。 |
| `get` | 读取 bytes；对象不存在时抛出 `FileNotFoundError`。 |
| `exists` | 判断对象是否存在。 |
| `list` | 按 prefix 列出对象，返回 `list[ObjectInfo]`。 |
| `delete` | 删除对象；对象不存在时不报错。 |
| `aclose` | 释放 store 持有的资源。 |

## 能力范围

Object Stores 负责：

- 文件类对象读写
- 本地目录和 S3 兼容对象存储适配
- 统一 key 校验
- 统一 list 返回格式

Object Stores 不负责文档语义、业务 schema、版本管理、血缘追踪、权限系统、事务系统或 KnowledgeBase 生命周期管理。这些能力应该由 Recipe、Manifest、Lineage 或更高层模块承担。
