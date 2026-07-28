# Build Wiki Pages

`BuildWikiPages` 将 `parsed_document_keys` 指向的 `ParsedDocument` 转换成可阅读的 Wiki page。

```text
parsed_document_keys -> Wiki page Markdown + index + page log
```

每个页面包含稳定 frontmatter、`title`、`summary`、`content` 和原始 `source`。如果 `ParsedDocument.original_content` 是 Markdown，step 优先保留其标题和结构；否则回退到 `pages[].text`。

## Contract

默认输入 artifact：`parsed_document_keys`。默认输出：

```text
wiki_page_keys
wiki_index_key       # wiki/index.md
wiki_log_key         # wiki/page_log.json
build_wiki_pages_result
```

默认写入 `wiki/pages/{number}-{slug}.md`。`summary_mode="model"` 需要 `LanguageModelProtocol`；`summary_mode="extractive"` 不调用模型。

```python
BuildWikiPages(BuildWikiPagesConfig(
    summary_mode="model",
    max_document_pages=80,
    max_document_tokens=262_144,
    oversized_document_policy="fail",
))
```

step 会复用 source metadata 一致且可解析的已有页面，因此失败后续跑不会重复生成成功产物。超出页数或 token 限制时，可以选择 `fail` 或记录 issue 后 `skip`。

页面构建与检索切分是两个职责：本 step 不产生 `ParsedChunk`，后续由 `SplitWikiPages` 完成。
