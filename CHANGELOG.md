# Changelog

All notable changes to Heta Framework are documented in this file.

## 0.1.2 - 2026-08-05

### Added

- Added universal graph extraction, ontology constraints, and graph adapters
  shared by GraphRAG, LightRAG, HiRAG, and LeanRAG build procedures.
- Added built-in procedures and query engines for GraphRAG, LightRAG, HiRAG,
  and LeanRAG retrieval modes.
- Added clusterable graph-store protocols, in-memory graph clustering,
  hierarchical graph aggregation, and community report generation.
- Added a `graph` optional dependency group for graph construction, Leiden
  clustering, and hierarchical semantic aggregation.

### Changed

- Replaced the former entity-then-relation extraction path with universal graph
  extraction and per-method graph adaptation.
- Extended retrieval provenance and BEIR evaluation handling to support results
  associated with multiple source documents.
- Added Python-version-specific NetworkX and scikit-learn pins so Python 3.10
  remains supported while Python 3.11 and 3.12 reproduce the graph feature
  development environment.

### Fixed

- Bundled LightRAG and LeanRAG query prompts in Heta Framework so installed
  packages no longer depend on adjacent LightRAG or LeanRAG source repositories.
- Fixed evidence recall matching for aggregated query results carrying multiple
  document and source identifiers, including HiRAG results evaluated on MultiHop-RAG.
- Completed graph dependency declarations for NumPy, scikit-learn, UMAP,
  NetworkX, and graspologic.
- Added the object store required by LeanRAG query integration tests.

### Compatibility

- Heta Framework 0.1.2 supports Python 3.10 through 3.12.
- Existing Wiki and agentic query APIs from 0.1.1 remain available alongside
  the new RAG procedures and query engines.
- Install `heta-framework[graph]` to use GraphRAG, LightRAG, HiRAG, LeanRAG,
  and Leiden clustering dependencies.

### Known Limitations

- Graph construction and query integration have automated smoke coverage, but
  production deployments should validate their selected model and store
  implementations with representative documents.

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
