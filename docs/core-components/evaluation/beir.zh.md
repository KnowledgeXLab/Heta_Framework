# BEIR

`BeirBenchmark` 接入 BEIR 的官方预处理检索数据集。

BEIR 适合评估 retrieval 本身：召回是否准确、排序是否合理、不同领域下检索是否稳定。它不要求 PDF 解析、OCR 或答案生成。

## Data Layout

BEIR 的核心文件是：

```text
corpus.jsonl
    文档集合，包含 _id、title、text

queries.jsonl
    查询集合，包含 _id、text

qrels/{split}.tsv
    query_id、document_id、relevance
```

Heta 第一版推荐使用四个子集：

| dataset | 用途 |
| --- | --- |
| `scifact` | 科学事实检索，小而稳定，适合 smoke test。 |
| `nfcorpus` | 医学/生物医学检索，语义要求更强。 |
| `fiqa` | 金融问答检索，适合测试领域迁移。 |
| `hotpotqa` | 多跳问答来源的检索任务，适合测试复杂查询召回。 |

这四个子集覆盖科学、医学、金融和多跳问答，足够支撑 Heta 的标准检索评估，不需要一开始接完整 BEIR 全量集合。

## Usage

```python
from heta_framework.evaluation import BenchmarkRunner, BeirBenchmark

benchmark = BeirBenchmark(
    dataset="scifact",
)

result = await BenchmarkRunner().run(
    benchmark=benchmark,
    recipe=recipe,
    knowledge_base_name="beir_scifact_vector_v1",
    query_modes=("vector_search",),
)
```

默认会从 BEIR 官方公开地址下载：

```text
https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip
```

也可以使用本地已下载数据：

```python
benchmark = BeirBenchmark(
    dataset="scifact",
    data_root="/data/beir/scifact",
    download=False,
)
```

`data_root` 应该包含：

```text
corpus.jsonl
queries.jsonl
qrels/test.tsv
```

## Document Mapping

BEIR 的 qrels 是 document-level，而 Heta query result 通常是 chunk-level。因此 adapter 会把每个 BEIR corpus item 写成一个独立 text document：

```text
raw/benchmarks/beir_{dataset}/{split}/{document_id}/{document_id}.txt
```

例如：

```text
raw/benchmarks/beir_scifact/test/D1/D1.txt
```

`document_id` 来自 BEIR 原始 `_id` 的安全文件名形式。这样即使文档后续被 `SplitDocuments` 切成多个 chunk，评估器也能从 query result 的 `source_key` 或 `object_key` 反推出它属于哪个 BEIR document。

## Case Mapping

每个 query 会变成一个 `BenchmarkCase`：

```text
case_id
    BEIR query _id

query
    BEIR query text

expected.evidence
    qrels 中 relevance > 0 的 document labels
```

每条 qrel 会映射成：

```python
BenchmarkEvidence(
    reference_id="D1",
    locator={
        "source_key_prefix": "raw/benchmarks/beir_scifact/test/D1/",
    },
    metadata={
        "beir_doc_id": "D1",
        "beir_document_id": "D1",
        "relevance": 2,
    },
)
```

`reference_id` 使用安全后的 benchmark document id。`metadata.beir_doc_id` 保留原始 BEIR id，方便报告和调试。

## Default Evaluators

`BeirBenchmark` 默认使用标准 IR 指标：

```text
beir_ndcg@1 / @3 / @5 / @10 / @100
beir_map@1 / @3 / @5 / @10 / @100
beir_recall@1 / @3 / @5 / @10 / @100
beir_precision@1 / @3 / @5 / @10 / @100
beir_mrr@1 / @3 / @5 / @10 / @100
```

Heta 会先把 chunk-level hits 映射回 document id，并按文档去重，然后再计算 BEIR 指标。这避免同一个长文档命中多个 chunk 时影响 document-level 分数。

如果只想跑一组轻量指标，可以覆盖 evaluator：

```python
from heta_framework.evaluation import BeirRetrievalMetric

result = await BenchmarkRunner().run(
    benchmark=benchmark,
    recipe=recipe,
    knowledge_base_name="beir_scifact_vector_v1",
    query_modes=("vector_search",),
    evaluators=(
        BeirRetrievalMetric(metric="ndcg", k=10),
        BeirRetrievalMetric(metric="recall", k=10),
    ),
)
```

## Scope

BEIR 只评估检索质量，不评估生成答案。

如果要评估 RAG answer quality，使用 UDA-Benchmark 或 MultiHop-RAG 更合适。如果要评估纯检索召回、排序和跨领域泛化，使用 BEIR 更直接。

## Sources

官方资源：

```text
GitHub: https://github.com/beir-cellar/beir
Dataset URL pattern: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip
Paper: https://arxiv.org/abs/2104.08663
```
