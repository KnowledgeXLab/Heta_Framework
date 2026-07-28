# Changelog

All notable changes to Heta Framework are documented in this file.

## 0.1.1 - 2026-07-28

### Added

- Added an explicit `ToolCallingLanguageModelProtocol` and LiteLLM-backed
  `ToolCallingLanguageModel`.
- Added `BuildWikiPages`, `SplitWikiPages`, and `HetaWikiProcedure` for
  constructing readable and searchable Wiki knowledge bases.
- Added reusable Wiki query tools and `AgenticQueryEngine` for grounded,
  tool-driven retrieval.
- Added optional document-level original text content, Wiki heading paths,
  original-source provenance, and evidence-backed query insights.
- Added Wiki presets for chunk embedding, vector indexing, and full-text
  indexing.

### Changed

- Reduced the default `EmbedChunks` batch size from 64 to 10 so providers with
  smaller batch limits work without additional configuration.
- Propagated Wiki page and original-document provenance through vector search,
  full-text search, citations, and retrieval evaluation.

### Compatibility

- Existing 0.1.0 parsed documents and chunks remain readable without migration.
- Existing positional configuration constructors retain their 0.1.0 field
  order.
- Default retrieval recipes and query modes are unchanged; Wiki behavior is
  enabled only when its procedure and required components are selected.
- `QueryResponse.insights` is appended to the response contract without
  changing the position of existing fields.

### Known Limitations

- Adding documents to an already successful Wiki knowledge base remains an
  explicit update lifecycle rather than an implicit procedure operation.
- Build resume must use a recipe equivalent to the recipe used for the
  original run.
