# Benchmark Protocols

Heta Evaluation evaluates a `KnowledgeRecipe`, not an isolated `KnowledgeBase`.

A benchmark adapter turns an external benchmark into Heta inputs: documents, queries, expected answers, evidence labels, and default evaluators. `BenchmarkRunner` then builds KBs from the recipe, runs queries, scores responses, and writes an `EvaluationReport`.

```text
Benchmark
  -> documents
  -> ObjectStore raw/
  -> run_units
  -> KnowledgeBase.create(recipe)
  -> kb.query(...)
  -> evaluators
  -> EvaluationReport
```

`KnowledgeBase` is an intermediate build product. `EvaluationReport` is the final evaluation product.

## BenchmarkProtocol

Each benchmark adapter implements `BenchmarkProtocol`:

```python
class BenchmarkProtocol(Protocol):
    @property
    def manifest(self) -> BenchmarkManifest: ...

    def resources(self) -> tuple[BenchmarkResource, ...]: ...

    async def prepare(self, workspace: BenchmarkWorkspace) -> PreparedBenchmark: ...
    async def documents(self, prepared: PreparedBenchmark) -> AsyncIterator[BenchmarkDocument]: ...
    async def cases(self, prepared: PreparedBenchmark) -> AsyncIterator[BenchmarkCase]: ...
    async def run_units(self, prepared: PreparedBenchmark) -> AsyncIterator[BenchmarkRunUnit]: ...
    def evaluators(self) -> tuple[BenchmarkEvaluatorProtocol, ...]: ...
```

| Method | Responsibility |
| --- | --- |
| `manifest` | Benchmark name, version, split, task type, and citation metadata. |
| `resources()` | External resources needed to prepare the benchmark. |
| `prepare()` | Download, extract, validate, or locate local data. |
| `documents()` | Raw documents that should be written to KB `raw/`. |
| `cases()` | Query cases, expected answers, and evidence labels. |
| `run_units()` | How many KBs the runner should build and which documents/cases each uses. |
| `evaluators()` | Default scoring methods for this benchmark. |

The adapter does not build KBs, call query engines, or aggregate reports. That belongs to `BenchmarkRunner`.

## Run Units

`BenchmarkRunUnit` describes one independent build-and-evaluate unit:

```python
BenchmarkRunUnit(
    unit_id="ADI_2009",
    document_ids=("ADI_2009",),
    case_ids=(
        "ADI/2009/page_49.pdf-1",
        "ADI/2009/page_59.pdf-2",
    ),
)
```

This supports two real benchmark shapes:

```text
corpus unit
    one KB for all cases

many units
    many KBs, each with its own cases, aggregated into one report
```

## Build Scope

`BenchmarkManifest.build_scope` describes benchmark organization:

```python
BenchmarkManifest(
    name="multihop_rag",
    version="official",
    split="all",
    task_type="multi_hop_qa",
    build_scope="corpus",
)
```

`build_scope` is a semantic hint. The actual execution plan comes from `run_units()`.

| build_scope | Meaning |
| --- | --- |
| `corpus` | One corpus, one KB, many query cases. |
| `case` | Many small KB units, each bound to cases. |
| `group` | Reserved for explicit domain/topic/split groups. |

## Documents

`BenchmarkDocument` is the raw document input:

```python
BenchmarkDocument(
    document_id="doc_001",
    name="paper.pdf",
    media_type="application/pdf",
    path=Path("paper.pdf"),
)
```

`document_id` must be stable and unique within the benchmark. Exactly one of `data`, `path`, and `source_uri` must be set.

Default raw key:

```text
raw/benchmarks/{benchmark_name}/{split}/{document_id}/{name}
```

This key enters `ParsedSource` and later appears in query result provenance.

## Cases

`BenchmarkCase` is one query evaluation sample:

```python
BenchmarkCase(
    case_id="case_001",
    query="What ocean management topics are discussed?",
    expected=BenchmarkExpected(
        answers=("marine protected areas",),
        evidence=(
            BenchmarkEvidence(
                reference_id="doc_001_p4",
                locator={"document_id": "doc_001", "page_index": 4},
                text="...",
            ),
        ),
    ),
)
```

`BenchmarkEvidence.locator` is intentionally open. Built-in evaluators recognize common fields such as `document_id`, `source_key`, `page_index`, `chunk_id`, `table_id`, `row_index`, and `column`.

## Evaluators

Benchmark adapters declare their own default evaluators:

```python
def evaluators(self):
    return (
        EvidenceRecallAtK(k=5),
        AnswerContains(),
    )
```

Principle:

```text
Benchmark owns scoring policy.
Common evaluators are reusable building blocks.
```

Many benchmark labels are tightly coupled to scoring. Evaluators are the scoring methods exposed by a benchmark, while Heta provides reusable common evaluators.
