# Vector Search

`vector_search` is Heta's base semantic retrieval mode. It consumes the `chunk_vector_index` produced by `IndexVectors` and returns semantically similar chunks through `KnowledgeBase.query()`.

## Required Asset

`IndexVectors` declares:

```python
SearchAsset(
    kind="chunk_vector_index",
    name="chunks",
    store="stores.vector",
    metadata={
        "collection": "chunks",
        "id_field": "id",
        "text_field": "text",
        "metadata_field": "metadata",
    },
)
```

When this asset exists in the latest run record, the default query registry enables:

```text
vector_search
```

## Usage

```python
response = await kb.query(
    "How does Heta build a knowledge base?",
    mode="vector_search",
    top_k=5,
)

for result in response.results:
    print(result.score, result.text)
```

Each `QueryResult` represents one chunk:

```text
id
text
score
kind = "chunk"
source
metadata
```

`source` includes document id, source key, source name, page index, chunk index, and token offsets.

## Execution Flow

```text
query text
  -> models.embedding.embed()
  -> stores.vector.search(collection="chunks")
  -> QueryResponse
```

`VectorSearchEngine` does not read ObjectStore. It uses the `text` and `metadata` stored in vector records.

## Filters

`filters` are passed to `VectorStore.search()`:

```python
response = await kb.query(
    "Heta graph",
    mode="vector_search",
    filters={"document_id": "doc_123"},
)
```

Filter support depends on the vector store. The in-memory implementation uses exact metadata matching; the Milvus adapter converts filters into a Milvus expression.

## Scope

`vector_search` only performs vector recall. It does not do BM25, SQL text search, graph expansion, reranking, query rewriting, or answer generation.
