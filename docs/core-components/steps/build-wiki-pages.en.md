# Build Wiki Pages

`BuildWikiPages` converts `ParsedDocument` records referenced by `parsed_document_keys` into readable Wiki pages.

```text
parsed_document_keys -> Wiki page Markdown + index + page log
```

Each page contains stable frontmatter, `title`, `summary`, `content`, and source provenance. Markdown in `ParsedDocument.original_content` is preferred; `pages[].text` is the fallback.

## Contract

Default outputs:

```text
wiki_page_keys
wiki_index_key       # wiki/index.md
wiki_log_key         # wiki/page_log.json
build_wiki_pages_result
```

Pages are written to `wiki/pages/{number}-{slug}.md`. `summary_mode="model"` requires `LanguageModelProtocol`; `summary_mode="extractive"` is offline.

```python
BuildWikiPages(BuildWikiPagesConfig(
    summary_mode="model",
    max_document_pages=80,
    max_document_tokens=262_144,
    oversized_document_policy="fail",
))
```

The step reuses an existing page when its parsed structure and source metadata match, enabling build resume. Oversized documents can fail or be skipped with a structured issue. Retrieval chunks are a separate responsibility of `SplitWikiPages`.
