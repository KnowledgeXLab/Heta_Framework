# Split Wiki Pages

`SplitWikiPages` 将 `wiki_page_keys` 指向的 Wiki Markdown 转换为符合现有 `ParsedChunk` 协议的检索 chunks。

```text
wiki_page_keys -> wiki_chunk_keys
```

它按 `###` 至 `######` 标题维护 `heading_path`，再在每个 section 内按 token budget 切分。代码块中的伪标题不会改变标题树。

每个 Wiki chunk 同时保留两层来源：`source` 指向可直接读取的生成 Wiki page，`origin_source` 指向最初进入知识库的原始文档。向量与全文索引会传播这条来源链，因此检索结果既能读取 Wiki page，也能审计并引用原文。

每个 chunk 文本包含：

```text
Page: <title>
Summary: <summary>
Section: <heading path>

<section content>
```

```python
SplitWikiPages(SplitWikiPagesConfig(
    chunk_size=1024,
    overlap=50,
    max_chunks_per_page=256,
    oversized_page_policy="fail",
))
```

默认输出 `wiki_chunk_keys` 和 `split_wiki_pages_result`，JSON 写入 `wiki_chunks/`。已有且内容一致的 chunk 会被复用。超过单页 chunk 限制时可以 `fail` 或记录 issue 后 `skip`。

标准后续链路是：

```text
EmbedChunks(preset="wiki")
IndexVectors(preset="wiki")
IndexFullText(preset="wiki")
```
