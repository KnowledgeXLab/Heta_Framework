# Heta Wiki Procedure

`HetaWikiProcedure` converts `ParsedDocument` records into a readable, searchable Wiki and builds vector and full-text indexes.

```text
parsed_document_keys
  -> BuildWikiPages
  -> SplitWikiPages
  -> EmbedChunks(preset="wiki")
  -> IndexVectors(preset="wiki")
  -> IndexFullText(preset="wiki")
```

The procedure is static step composition. It does not parse raw files, execute queries, or run an agent loop.

## Complete Recipe

`ParseDocuments` stays outside the procedure because each source type may use a different parser while exposing the same `parsed_document_keys` artifact.

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

`KnowledgeRecipe.expanded_steps()` expands the procedure into five real steps. Validation, run records, resume behavior, and cleanup continue to operate on those steps.

## Output Contract

The default artifact chain is:

```text
wiki_page_keys
wiki_index_key
wiki_log_key
wiki_chunk_keys
wiki_chunk_embedding_keys
```

The procedure produces:

```text
wiki_chunk_vector_index    -> wiki_vector_search
wiki_chunk_full_text_index -> wiki_full_text_search
```

Chunks produced by `SplitWikiPages` retain two provenance levels: `source` identifies the generated Wiki page and `origin_source` identifies the original parsed document. Both indexes propagate this chain so derived artifacts retain their original document identity.

When the recipe also provides `ToolCallingLanguageModelProtocol` and an embedding model, the default registry exposes `wiki_agent_search`. Its runtime tools belong to `AgenticQueryEngine`, not this build procedure.

## Configuration

The procedure reuses the five step config types instead of defining a second configuration surface:

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

Construction validates that all index steps use the `wiki` preset, `SplitWikiPages` consumes `wiki_page_keys`, and every step uses the same ObjectStore.

## Safety And Lifecycle

- `BuildWikiPages` bounds document pages and tokens.
- `SplitWikiPages` bounds chunks per Wiki page.
- `EmbedChunks` defaults to `batch_size=10`; raise it after confirming provider support.
- Failed builds can resume from the failed step through run state.
- Appending documents to an already successful KB is a separate update lifecycle and is not performed implicitly by the procedure.
