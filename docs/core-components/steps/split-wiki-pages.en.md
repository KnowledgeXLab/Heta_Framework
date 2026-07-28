# Split Wiki Pages

`SplitWikiPages` converts generated Wiki Markdown into retrieval chunks that use the existing `ParsedChunk` contract.

```text
wiki_page_keys -> wiki_chunk_keys
```

It builds `heading_path` from `###` through `######` headings, then splits each section within a token budget. Headings inside fenced code blocks do not modify the heading tree.

Each Wiki chunk keeps two provenance levels: `source` identifies the generated Wiki page that can be read directly, while `origin_source` identifies the original document ingested into the knowledge base. Vector and full-text indexes propagate this chain, so query results can both open the Wiki page and audit or cite the original source.

Each chunk includes page, summary, section path, and section content.

```python
SplitWikiPages(SplitWikiPagesConfig(
    chunk_size=1024,
    overlap=50,
    max_chunks_per_page=256,
    oversized_page_policy="fail",
))
```

The default outputs are `wiki_chunk_keys` and `split_wiki_pages_result`, with JSON under `wiki_chunks/`. Matching existing chunks are reused. Oversized pages can fail or be skipped with a structured issue.

The standard downstream chain is:

```text
EmbedChunks(preset="wiki")
IndexVectors(preset="wiki")
IndexFullText(preset="wiki")
```
