# Quick Start

这个示例用一个本地文本文件跑通第一个知识库构建。

它覆盖完整的黄金路径：

```text
raw text
  -> ParseDocuments
  -> SplitDocuments
  -> EmbedChunks
  -> IndexVectors
  -> HetaGraphProcedure
  -> KnowledgeBase
```

第一遍建议使用本地 ObjectStore、内存 VectorStore 和 SQLite。
这能让用户先理解 Heta 的构建方式；生产环境再把这些组件替换成 S3、Milvus、PostgreSQL。

## 安装

在 `heta_framework` 目录下安装本地开发版本：

```bash
pip install -e ".[sql]"
```

如果要使用 OpenAI 模型，设置环境变量：

```bash
export OPENAI_API_KEY="sk-..."
```

Heta 的模型层由 LiteLLM 驱动，`model_name` 使用 LiteLLM 的模型命名方式，例如
`openai/gpt-4o-mini`、`openai/text-embedding-3-small`。

## 构建第一个知识库

创建 `quickstart.py`：

```python
import asyncio
import os
from pathlib import Path

from heta_framework.common.models import EmbeddingModel, LanguageModel
from heta_framework.common.stores import (
    InMemoryTextIndexStore,
    InMemoryVectorStore,
    LocalObjectStore,
    SQLStore,
)
from heta_framework.kb import (
    DocumentParserRegistry,
    EmbedChunks,
    HetaGraphProcedure,
    IndexFullText,
    IndexVectors,
    KnowledgeBase,
    KnowledgeModels,
    KnowledgeParsers,
    KnowledgeRecipe,
    KnowledgeStores,
    ParseDocuments,
    SplitDocuments,
    SplitDocumentsConfig,
    TextParser,
)


async def main() -> None:
    workspace = Path("heta-quickstart")
    workspace.mkdir(exist_ok=True)

    object_store = LocalObjectStore(workspace / "objects")
    vector_store = InMemoryVectorStore()
    text_index_store = InMemoryTextIndexStore()
    sql_store = SQLStore(f"sqlite:///{workspace / 'knowledge.db'}")

    llm = LanguageModel(
        model_name="openai/gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    embedding = EmbeddingModel(
        model_name="openai/text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    await object_store.put(
        "raw/heta.txt",
        (
            "Heta is a framework for building knowledge bases. "
            "It uses recipes to compose parsers, language models, embedding models, "
            "object stores, vector stores, SQL stores, and graph-building steps. "
            "The HetaGraphProcedure extracts entities and relations from chunks, "
            "deduplicates them, and writes graph facts into SQL and vector storage."
        ).encode("utf-8"),
    )

    recipe = KnowledgeRecipe(
        parsers=KnowledgeParsers(
            documents=DocumentParserRegistry([TextParser()]),
        ),
        models=KnowledgeModels(
            language=llm,
            embedding=embedding,
        ),
        stores=KnowledgeStores(
            objects=object_store,
            vector=vector_store,
            text_index=text_index_store,
            sql=sql_store,
        ),
        steps=(
            ParseDocuments(),
            SplitDocuments(),
            IndexFullText(),
            EmbedChunks(),
            IndexVectors(),
            *HetaGraphProcedure.build().steps(),
        ),
    )
    recipe.require_valid()

    kb = await KnowledgeBase.create(
        recipe=recipe,
        name="quickstart",
        description="A first Heta knowledge base.",
    )

    print("status:", kb.run_record.status)
    print("queries:", sorted(kb.available_queries))

    entity_count = await sql_store.fetch_one("SELECT COUNT(*) AS count FROM entities")
    relation_count = await sql_store.fetch_one("SELECT COUNT(*) AS count FROM relations")
    print("entities:", entity_count["count"] if entity_count else 0)
    print("relations:", relation_count["count"] if relation_count else 0)

    response = await kb.query(
        "How does Heta build a knowledge base?",
        mode="vector_search",
        top_k=3,
    )
    for result in response.results:
        print("hit:", round(result.score or 0, 4), result.text[:120])

    await llm.aclose()
    await embedding.aclose()
    await sql_store.aclose()
    await text_index_store.aclose()
    await vector_store.aclose()
    await object_store.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

运行：

```bash
python quickstart.py
```

成功后会看到类似输出：

```text
status: succeeded
queries: ['full_text_search', 'heta_graph_search', 'heta_multihop_search', 'heta_rerank_search', 'heta_rewrite_search', 'hybrid_search', 'vector_search']
entities: 3
relations: 2
hit: 0.72 Heta is a framework for building knowledge bases...
```

实体和关系数量会随模型输出略有变化。
稳定不变的是构建状态、`vector_search` 查询入口，以及本地目录中的构建产物。

`SplitDocuments` 默认使用 `cl100k_base` tokenizer。
如果运行环境完全离线，需要提前缓存 tokenizer 文件，或者临时改成
`SplitDocuments(SplitDocumentsConfig(encoding_name="unicode"))`。

## 产物位置

示例会生成：

```text
heta-quickstart/
  knowledge.db
  objects/
    raw/
      heta.txt
    parsed/
      ...
    chunks/
      ...
    embeddings/
      ...
    entities/
      ...
    relations/
      ...
    deduplicated_entities/
      ...
    deduplicated_relations/
      ...
```

其中：

- `objects/raw/` 保存原始文件。
- `objects/parsed/` 保存统一的 `ParsedDocument`。
- `objects/chunks/` 保存切分后的 `ParsedChunk`。
- `objects/embeddings/` 保存 chunk embedding 产物。
- `knowledge.db` 保存 Heta graph 的实体、关系和证据表。
- `vector_store` 在这个示例里是内存实现，进程结束后不会持久化。

## 替换生产组件

Recipe 不绑定具体存储实现。
把本地组件替换成生产组件时，steps 不需要改变：

```python
object_store = S3ObjectStore(...)
vector_store = MilvusVectorStore(...)
sql_store = SQLStore("postgresql+psycopg://postgres:postgres@host:5432/postgres")
```

同一份 Recipe 仍然通过：

```python
kb = await KnowledgeBase.create(recipe=recipe, name="production-kb")
```

完成构建。

## 最小向量知识库

如果第一阶段只需要向量检索，可以先不启用 Heta graph：

```python
steps=(
    ParseDocuments(),
    SplitDocuments(),
    EmbedChunks(),
    IndexVectors(),
)
```

这时构建完成后只会解锁 `vector_search`。
需要实体、关系和图谱证据时，再追加 `HetaGraphProcedure.build()` 或
`HetaGraphProcedure.merge_into_store()`。
