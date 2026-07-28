# Heta Wiki Procedure

`HetaWikiProcedure` 将 `ParsedDocument` 转换成可阅读、可检索的 Wiki，并建立向量与全文双索引。

```text
parsed_document_keys
  -> BuildWikiPages
  -> SplitWikiPages
  -> EmbedChunks(preset="wiki")
  -> IndexVectors(preset="wiki")
  -> IndexFullText(preset="wiki")
```

Procedure 只做静态 step composition。它不解析原始文件、不执行查询，也不包含 agent loop。

## Complete Recipe

`ParseDocuments` 位于 procedure 外部，因为不同文件类型可以使用不同 parser，但最终都提供相同的 `parsed_document_keys` artifact。

```python
from heta_framework.kb import (
    HetaWikiProcedure,
    KnowledgeModels,
    KnowledgeParsers,
    KnowledgeRecipe,
    KnowledgeStores,
    ParseDocuments,
)

recipe = KnowledgeRecipe(
    parsers=KnowledgeParsers(documents=document_parser_registry),
    models=KnowledgeModels(
        language=tool_calling_language_model,
        embedding=embedding_model,
    ),
    stores=KnowledgeStores(
        objects=object_store,
        vector=vector_store,
        text_index=text_index_store,
    ),
    steps=(
        ParseDocuments(),
        HetaWikiProcedure(),
    ),
)
```

`KnowledgeRecipe.expanded_steps()` 会把 procedure 展开为五个真实 steps。Recipe 校验、运行记录、断点续跑和 cleanup 仍以这些 steps 为准。

## Output Contract

默认 artifact 链是：

```text
wiki_page_keys
wiki_index_key
wiki_log_key
wiki_chunk_keys
wiki_chunk_embedding_keys
```

建立的检索资产和 query modes 是：

```text
wiki_chunk_vector_index    -> wiki_vector_search
wiki_chunk_full_text_index -> wiki_full_text_search
```

`SplitWikiPages` 生成的 chunk 保留两层来源：`source` 指向生成的 Wiki page，`origin_source` 指向最初解析的原始文档。两个索引都会传播这条来源链，避免派生产物失去原文身份。

当 Recipe 同时提供 `ToolCallingLanguageModelProtocol` 和 embedding model 时，默认 query registry 还会开放 `wiki_agent_search`。运行时使用的 Wiki tools 属于 `AgenticQueryEngine`，不属于构建 procedure。

## Configuration

Procedure 直接复用五个 step Config，不复制第二套配置字段：

```python
from heta_framework.kb import (
    BuildWikiPagesConfig,
    EmbedChunksConfig,
    HetaWikiProcedure,
    IndexFullTextConfig,
    IndexVectorsConfig,
    SplitWikiPagesConfig,
)

procedure = HetaWikiProcedure(
    build_pages_config=BuildWikiPagesConfig(
        summary_mode="model",
        max_document_pages=80,
    ),
    split_pages_config=SplitWikiPagesConfig(
        chunk_size=1024,
        overlap=50,
    ),
    embed_chunks_config=EmbedChunksConfig(
        preset="wiki",
        batch_size=10,
    ),
    index_vectors_config=IndexVectorsConfig(preset="wiki"),
    index_full_text_config=IndexFullTextConfig(preset="wiki"),
)
```

Procedure 构造时会保证：

- embedding、vector index 和 full-text index 都使用 `wiki` preset。
- `SplitWikiPages` 读取 `BuildWikiPages` 产生的 `wiki_page_keys`。
- 五个 steps 使用同一个 ObjectStore。

这些约束让展开后的数据流天然闭合，而不是运行时才发现接线错误。

## Safety And Lifecycle

- `BuildWikiPages` 限制单文档页数和 token 数。
- `SplitWikiPages` 限制每个 Wiki page 的 chunk 数。
- `EmbedChunks` 默认 `batch_size=10`；确认 provider 支持后可显式调高。
- 失败构建可以通过 run state 从失败 step 继续。
- 向已经成功的 KB 追加文档属于独立的增量更新生命周期，不由 procedure 隐式执行。
