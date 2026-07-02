# BEIR

`BeirBenchmark` integrates official preprocessed BEIR retrieval datasets.

BEIR evaluates retrieval quality: recall, ranking, and cross-domain robustness. It does not require PDF parsing, OCR, or answer generation.

## Data Layout

```text
corpus.jsonl
    document collection with _id, title, text

queries.jsonl
    query collection with _id, text

qrels/{split}.tsv
    query_id, document_id, relevance
```

Recommended subsets:

| dataset | Use |
| --- | --- |
| `scifact` | Small, stable scientific fact retrieval; good smoke test. |
| `nfcorpus` | Medical/biomedical retrieval. |
| `fiqa` | Financial QA retrieval. |
| `hotpotqa` | Retrieval task derived from multi-hop QA. |

## Usage

```python
from heta_framework.evaluation import BenchmarkRunner, BeirBenchmark

benchmark = BeirBenchmark(dataset="scifact")

result = await BenchmarkRunner().run(
    benchmark=benchmark,
    recipe=recipe,
    knowledge_base_name="beir_scifact_vector_v1",
    query_modes=("vector_search",),
)
```

By default, Heta downloads from:

```text
https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip
```

Use local data:

```python
benchmark = BeirBenchmark(
    dataset="scifact",
    data_root="/data/beir/scifact",
    download=False,
)
```

`data_root` should contain:

```text
corpus.jsonl
queries.jsonl
qrels/test.tsv
```

## Mapping

BEIR labels are document-level, while Heta results are usually chunk-level. The adapter writes each corpus item as one text document:

```text
raw/benchmarks/beir_{dataset}/{split}/{document_id}/{document_id}.txt
```

Evaluators map chunk hits back to benchmark document ids and deduplicate by document before computing metrics.

## Default Evaluators

`BeirBenchmark` uses standard IR metrics:

```text
beir_ndcg@1 / @3 / @5 / @10 / @100
beir_map@1 / @3 / @5 / @10 / @100
beir_recall@1 / @3 / @5 / @10 / @100
beir_precision@1 / @3 / @5 / @10 / @100
beir_mrr@1 / @3 / @5 / @10 / @100
```

For a lighter run:

```python
from heta_framework.evaluation import BeirRetrievalMetric

evaluators=(
    BeirRetrievalMetric(metric="ndcg", k=10),
    BeirRetrievalMetric(metric="recall", k=10),
)
```

## Scope

BEIR evaluates retrieval quality only. Use UDA-Benchmark or MultiHop-RAG when you need answer quality or multi-hop evidence evaluation.

## Sources

```text
GitHub: https://github.com/beir-cellar/beir
Dataset URL pattern: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip
Paper: https://arxiv.org/abs/2104.08663
```
