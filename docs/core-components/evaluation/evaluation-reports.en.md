# Evaluation Reports

`EvaluationReport` is the final product of Heta Evaluation.

It records one benchmark run:

```text
which recipe
which benchmark
which query modes
what query results
how each evaluator scored
what the summary is
```

`KnowledgeBase` is the intermediate artifact built from benchmark documents. `EvaluationReport` is the evaluation result.

## Report Structure

```python
EvaluationReport(
    report_id="eval_001",
    benchmark=benchmark.manifest,
    knowledge_base_name="multihop_graph_v1",
    knowledge_base_manifest=kb.manifest().to_dict(),
    recipe_manifest=recipe.manifest().to_dict(),
    query_modes=("heta_multihop_search",),
    score_summary={"evidence_recall@5": 0.82},
    case_results=(...),
    started_at="...",
    finished_at="...",
)
```

Important fields:

| Field | Meaning |
| --- | --- |
| `report_id` | Stable ID for this evaluation run. |
| `benchmark` | Benchmark name, version, split, and task type. |
| `knowledge_base_name` | Base KB name for this run. |
| `knowledge_base_manifest` | Single KB manifest or a list for multi-KB benchmarks. |
| `recipe_manifest` | Snapshot of the evaluated recipe. |
| `query_modes` | Query modes called in this run. |
| `score_summary` | Aggregated evaluator scores. |
| `case_results` | Detailed result per case and query mode. |
| `metadata` | Runtime config, environment info, or benchmark-specific data. |

The report stores both recipe and KB manifests so results are traceable and comparable.

## Case Results

Each `EvaluationCaseResult` corresponds to:

```text
one benchmark case
one query mode
one QueryResponse or one error
```

```python
EvaluationCaseResult(
    case_id="case_001",
    query="...",
    query_mode="heta_graph_search",
    response=response,
    scores=(
        EvaluationScore(name="evidence_recall@5", value=1.0),
    ),
    latency_ms=45.2,
)
```

`response` is the standard `QueryResponse` returned by `kb.query(...)`.

## EvaluationScore

```python
EvaluationScore(
    name="evidence_recall@5",
    value=0.8,
    passed=None,
    metadata={"matched": 4, "expected": 5},
)
```

| Field | Meaning |
| --- | --- |
| `name` | Evaluator name. |
| `value` | Numeric score, boolean result, or label. |
| `passed` | Optional pass/fail judgement. |
| `metadata` | Matched evidence, missing evidence, judge reason, or debugging data. |

## Default Location

When persisted through the KB ObjectStore:

```text
_heta/knowledge_bases/{knowledge_base_name}/evaluations/{report_id}/report.json
```

If no ObjectStore is available, the runner can still return an in-memory `EvaluationReport`.

## Design Boundary

`EvaluationReport` does not modify the KB. It records evaluation facts.

One recipe can run multiple benchmarks, and one benchmark can compare multiple recipes. This is why the report is a first-class artifact.
