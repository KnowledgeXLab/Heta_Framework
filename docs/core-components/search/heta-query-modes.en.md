# Heta Query Modes

Heta Framework includes a set of composed query modes based on HetaDB experience. They are not new storage structures; they are query strategies built from existing retrieval assets.

Base query modes:

```text
vector_search
sql_text_search
full_text_search
heta_graph_search
```

Composed query modes:

| Mode | Purpose | Dependencies |
| --- | --- | --- |
| `hybrid_search` | Weighted RRF fusion of vector search and Heta graph search. | `chunk_vector_index`, `graph_tables`, `graph_vector_index` |
| `heta_rerank_search` | Fuses Heta hybrid search and full-text search, then optionally reranks candidates. | `chunk_vector_index`, `chunk_full_text_index`, `graph_tables`, `graph_vector_index` |
| `heta_rewrite_search` | Uses a language model to generate 3 query variants, runs Heta rerank search for each, then fuses results. | `models.language`, plus assets required by `heta_rerank_search` |
| `heta_multihop_search` | Runs up to 3 rounds of Heta rerank search, information extraction, and sufficiency checks. | `models.language`, plus assets required by `heta_rerank_search` |

Composed modes call base modes through `QueryContext.query(...)`, so recursion checks, asset checks, and trace handling all use the same path.

## hybrid_search

`hybrid_search` aligns with HetaDB's "vector + graph" hybrid retrieval path, but tightens scoring.

The framework uses weighted RRF:

1. Run `vector_search` for chunk candidates.
2. Run `heta_graph_search` for entities, relations, and evidence.
3. Use ranks from each result list instead of mixing raw scores.
4. Apply per-mode weights.
5. Reward results that appear in multiple sources.

```python
response = await kb.query(
    "marine biodiversity evidence",
    mode="hybrid_search",
    top_k=8,
    options={
        "hybrid_weights": {
            "vector_search": 1.0,
            "heta_graph_search": 1.2,
        }
    },
)
```

Options:

| `options` field | Default | Meaning |
| --- | --- | --- |
| `candidate_top_k` | `min(top_k * 3, 50)` | Candidates recalled from each base mode. |
| `rrf_k` | `60` | RRF smoothing parameter. |
| `hybrid_weights` | `{"vector_search": 1.0, "heta_graph_search": 1.0}` | Fusion weights. |

## heta_rerank_search

`heta_rerank_search` is the high-precision retrieval path.

Default flow:

1. Run `hybrid_search` for vector and graph candidates.
2. Run `full_text_search` for full-text chunk matches.
3. Merge candidates with Reciprocal Rank Fusion.
4. If `KnowledgeModels.reranker` exists, rerank candidates.
5. Otherwise keep the RRF order.

```python
response = await kb.query(
    "What loss function does the model use?",
    mode="heta_rerank_search",
    top_k=5,
)
```

Without a reranker, the mode still works and falls back to RRF ordering.

## heta_rewrite_search

`heta_rewrite_search` is useful when user phrasing is vague or one search may miss synonyms.

Default flow:

1. Use `models.language` to generate 3 query variants.
2. Run `heta_rerank_search` for each variant.
3. Fuse results with RRF.
4. If rewriting fails, fall back to one base search.

```python
response = await kb.query(
    "how does the thing handle sequences",
    mode="heta_rewrite_search",
    top_k=8,
)
```

Failures are recorded in `QueryResponse.metadata["issues"]` instead of being silently ignored.

## heta_multihop_search

`heta_multihop_search` is for questions that require multiple facts.

Default flow:

1. Run the base retrieval mode for the current query.
2. Ask the language model whether retrieved evidence is useful.
3. Ask whether accumulated evidence is enough to answer.
4. If enough, return an answer.
5. If not, generate the next query and continue, up to 3 rounds.
6. If still insufficient, generate a conservative answer from accumulated evidence.

```python
response = await kb.query(
    "How does the proposed method compare to the baseline across all datasets?",
    mode="heta_multihop_search",
    top_k=6,
    options={"max_rounds": 3},
)
```

`metadata["round_reports"]` records each round's query, result count, usefulness, answer sufficiency, and next query. Recoverable issues are recorded in `metadata["issues"]`.

## Recipe Requirements

Composed modes depend on existing components:

```python
from heta_framework.common.models import EmbeddingModel, LanguageModel, RerankModel
from heta_framework.kb import KnowledgeModels

models = KnowledgeModels(
    language=LanguageModel(model_name="openai/gpt-4o-mini", api_key="..."),
    embedding=EmbeddingModel(model_name="openai/text-embedding-3-small", api_key="..."),
    reranker=RerankModel(model_name="cohere/rerank-english-v3.0", api_key="..."),
)
```

Without `language`, `heta_rewrite_search` and `heta_multihop_search` are unavailable. Without `reranker`, `heta_rerank_search` still uses RRF ordering.
