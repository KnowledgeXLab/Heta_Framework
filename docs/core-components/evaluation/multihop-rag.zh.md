# MultiHop-RAG

`MultiHopRagBenchmark` 接入官方 MultiHop-RAG benchmark。

MultiHop-RAG 是 corpus-level benchmark：

```text
corpus.json
    全量新闻文章语料

MultiHopRAG.json
    query、answer、question_type、evidence_list
```

因此它只需要建一次 KB：

```text
corpus.json -> KnowledgeBase.create(recipe)
MultiHopRAG.json -> 多 case query
```

## Usage

使用本地文件：

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

允许 adapter 下载官方文件：

```python
benchmark = MultiHopRagBenchmark(download=True)
```

下载文件会放在 `BenchmarkWorkspace.cache_dir / "multihop_rag"` 下。

## Document Mapping

`corpus.json` 中每篇文章会变成一个 `BenchmarkDocument`：

```text
document_id
    article_{sha256(url or title)[:16]}

name
    {document_id}.txt

media_type
    text/plain
```

文档内容会包含：

```text
Title
Source
Published at
URL
Body
```

Runner 写入 ObjectStore 后的 raw key：

```text
raw/benchmarks/multihop_rag/all/{document_id}/{document_id}.txt
```

## Case Mapping

`MultiHopRAG.json` 中每条 query 会变成一个 `BenchmarkCase`：

```text
case_id
    multihop_rag_{index}

query
    row["query"]

expected.answers
    row["answer"]

labels.question_type
    row["question_type"]
```

`evidence_list` 中每条 evidence 会变成 `BenchmarkEvidence`。

Heta query result 的 `source.document_id` 是解析后的内容 ID，不一定等于 benchmark article id。
因此 MultiHop-RAG adapter 使用 raw `source_key` 做 evidence locator：

```python
BenchmarkEvidence(
    reference_id=document_id,
    locator={
        "source_key": "raw/benchmarks/multihop_rag/all/...txt",
    },
    text=fact,
    metadata={
        "document_id": document_id,
        "title": title,
        "source": source,
        "url": url,
    },
)
```

这样 `EvidenceRecallAtK` 可以直接通过 query result 的 `source.object_key` / `source.source_key` 判断命中。

## Default Evaluators

默认 evaluators：

```text
EvidenceRecallAtK(k=5)
AnswerContains()
```

MultiHop-RAG 更适合测试：

```text
heta_graph_search
heta_rerank_search
heta_rewrite_search
heta_multihop_search
```

也可以用同一个 benchmark 对比不同 recipe：

```text
vector-only recipe
graph recipe
graph + rewrite recipe
graph + multihop recipe
```

## Sources

官方资源：

```text
GitHub: https://github.com/yixuantt/MultiHop-RAG
Hugging Face: https://huggingface.co/datasets/yixuantt/MultiHopRAG
```
