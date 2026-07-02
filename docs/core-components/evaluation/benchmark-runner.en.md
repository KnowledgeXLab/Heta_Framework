# BenchmarkRunner

`BenchmarkRunner` is the entry point for running a benchmark.

It receives a `KnowledgeRecipe` and a benchmark adapter, then builds KBs, runs queries, scores responses, and writes a report:

```text
benchmark.prepare()
benchmark.documents()
  -> write ObjectStore raw/benchmarks/...
benchmark.run_units()
  -> KnowledgeBase.create(recipe)
benchmark.cases()
  -> kb.query(...)
  -> evaluator.evaluate(...)
EvaluationReport
```

It evaluates the recipe's build strategy and query behavior, not a manually prepared KB.

## Basic Usage

```python
from heta_framework.evaluation import BenchmarkRunner, BenchmarkRunConfig

result = await BenchmarkRunner().run(
    benchmark=benchmark,
    recipe=recipe,
    knowledge_base_name="multihop_graph_v1",
    query_modes=("heta_multihop_search",),
    config=BenchmarkRunConfig(
        top_k=5,
        report_id="eval_multihop_graph_v1",
        max_concurrent_queries=8,
    ),
)

report = result.report
knowledge_bases = result.knowledge_bases
```

`BenchmarkRunResult` includes:

```text
knowledge_bases
report
benchmark_document_keys
report_key
```

`knowledge_bases` are intermediate KBs built for the run. `report` is the final evaluation artifact.

## ObjectStore Requirement

`BenchmarkRunner` requires `stores.objects` in the recipe because benchmark documents are written to:

```text
raw/benchmarks/{benchmark_name}/{split}/{document_id}/{name}
```

These raw keys become initial build artifacts:

```text
benchmark_document_keys
source_keys
```

## Run Units

The runner supports two run shapes.

Single KB:

```text
one benchmark
  -> one run unit
  -> one KB
  -> all cases
```

Multi-KB:

```text
one benchmark
  -> many run units
  -> one KB per run unit
  -> each KB runs its own cases
```

Multi-KB mode derives KB names as:

```text
{knowledge_base_name}-{unit_id}
```

The final result is still one aggregated `EvaluationReport`.

## Query Modes

`query_modes` declares which modes to call:

```python
query_modes=(
    "vector_search",
    "heta_graph_search",
    "heta_multihop_search",
)
```

For each case and mode, the runner calls:

```python
kb.query(
    case.query,
    mode=query_mode,
    top_k=config.top_k,
    options=config.query_options,
    trace=config.trace,
)
```

If a case fails, the error is written to `EvaluationCaseResult.error`; the report structure is preserved.

## Evaluators

By default, the runner uses:

```python
benchmark.evaluators()
```

You can override evaluators for experiments:

```python
result = await BenchmarkRunner().run(
    benchmark=benchmark,
    recipe=recipe,
    knowledge_base_name="kb",
    query_modes=("vector_search",),
    evaluators=(EvidenceRecallAtK(k=10),),
)
```

## Report Persistence

With `persist_report=True`, the report is written to:

```text
_heta/knowledge_bases/{knowledge_base_name}/evaluations/{report_id}/report.json
```

Disable persistence:

```python
BenchmarkRunConfig(persist_report=False)
```

## JsonlBenchmark

`JsonlBenchmark` is the smallest local benchmark adapter:

```python
from heta_framework.evaluation import BenchmarkManifest, JsonlBenchmark

benchmark = JsonlBenchmark(
    manifest=BenchmarkManifest(
        name="local_rag_eval",
        version="v1",
        split="test",
        task_type="rag_qa",
    ),
    documents_jsonl="documents.jsonl",
    cases_jsonl="cases.jsonl",
)
```

`documents.jsonl`:

```json
{"document_id":"doc_1","name":"doc.txt","media_type":"text/plain","text":"Marine biodiversity."}
```

`cases.jsonl`:

```json
{"case_id":"case_1","query":"What is discussed?","expected":{"answers":["Marine biodiversity"],"evidence":[{"locator":{"document_id":"doc_1"},"text":"Marine biodiversity."}]}}
```
