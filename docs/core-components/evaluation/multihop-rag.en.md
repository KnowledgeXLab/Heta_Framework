# MultiHop-RAG

`MultiHopRagBenchmark` integrates the official MultiHop-RAG benchmark.

MultiHop-RAG is useful for evaluating questions that require evidence across multiple facts. It can compare vector-only recipes, Heta graph recipes, rewrite recipes, and multihop recipes on complex questions.

## Data Layout

MultiHop-RAG is a corpus-level benchmark:

```text
corpus.json
    full news article corpus

MultiHopRAG.json
    query, answer, question_type, evidence_list
```

It builds one KB:

```text
corpus.json -> KnowledgeBase.create(recipe)
MultiHopRAG.json -> many query cases
```

## Usage

Use local files:

```python
from heta_framework.evaluation import BenchmarkRunner, MultiHopRagBenchmark

benchmark = MultiHopRagBenchmark(
    corpus_path="corpus.json",
    queries_path="MultiHopRAG.json",
)

result = await BenchmarkRunner().run(
    benchmark=benchmark,
    recipe=recipe,
    knowledge_base_name="multihop_rag_graph_v1",
    query_modes=("heta_multihop_search",),
)
```

Or allow the adapter to download official files:

```python
benchmark = MultiHopRagBenchmark(download=True)
```

Downloaded files are stored under `BenchmarkWorkspace.cache_dir / "multihop_rag"`.

## Document Mapping

Each article becomes a `BenchmarkDocument`:

```text
document_id = article_{sha256(url or title)[:16]}
name = {document_id}.txt
media_type = text/plain
```

The text includes title, source, published time, URL, and body.

Raw key:

```text
raw/benchmarks/multihop_rag/all/{document_id}/{document_id}.txt
```

## Case Mapping

Each row in `MultiHopRAG.json` becomes a `BenchmarkCase`:

```text
case_id = multihop_rag_{index}
query = row["query"]
expected.answers = row["answer"]
labels.question_type = row["question_type"]
```

Evidence uses raw `source_key` as the locator so `EvidenceRecallAtK` can match query result sources.

## Default Evaluators

```text
EvidenceRecallAtK(k=5)
AnswerContains()
```

Recommended modes:

```text
vector_search
heta_graph_search
heta_rerank_search
heta_rewrite_search
heta_multihop_search
```

## Sources

```text
GitHub: https://github.com/yixuantt/MultiHop-RAG
Hugging Face: https://huggingface.co/datasets/yixuantt/MultiHopRAG
```
