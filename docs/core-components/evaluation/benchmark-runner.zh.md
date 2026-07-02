# BenchmarkRunner

`BenchmarkRunner` 是执行 benchmark 的入口。

它接收一套 `KnowledgeRecipe` 和一个 benchmark adapter，然后完成建库、查询、评分和报告生成：

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

因此它评估的是 recipe 的构建方案和查询效果，而不是一个已经手工建好的 KB。

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

`BenchmarkRunResult` 包含：

```text
knowledge_bases
report
benchmark_document_keys
report_key
```

`knowledge_bases` 是本次 benchmark run 构建出的中间 KB；`report` 是最终评估产物。

## ObjectStore Requirement

`BenchmarkRunner` 要求 recipe 配置 `stores.objects`。

原因是 benchmark documents 需要先写入 ObjectStore：

```text
raw/benchmarks/{benchmark_name}/{split}/{document_id}/{name}
```

这些 raw keys 会作为 initial artifacts 传入 build：

```text
benchmark_document_keys
source_keys
```

当前内置 `ParseDocuments` 仍然按 raw prefix 扫描对象。如果需要严格只消费 benchmark documents，可以在后续自定义 step 中读取 `source_keys`。

## Run Units

`BenchmarkRunner` 支持两种运行形态。

第一种是单 KB：

```text
one benchmark
  -> one run unit
  -> one KB
  -> all cases
```

适合 BEIR 这类全 corpus 检索任务。

第二种是多 KB：

```text
one benchmark
  -> many run units
  -> one KB per run unit
  -> each KB runs its own cases
```

适合 UDA-fin 这类每个问题绑定到具体 PDF 的任务。

Run unit 由 benchmark adapter 声明：

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

`document_ids` 和 `case_ids` 为空时表示使用全部文档和全部 cases。这就是 corpus-level benchmark 的默认形态。

多 KB 模式下，Runner 会为每个 unit 派生一个 KB 名称：

```text
{knowledge_base_name}-{unit_id}
```

最终仍然只生成一个聚合 `EvaluationReport`。

## Query Modes

`query_modes` 声明本次评估调用哪些查询方式：

```python
query_modes=(
    "vector_search",
    "heta_graph_search",
    "heta_multihop_search",
)
```

Runner 会对每个 case、每个 query mode 调用：

```python
kb.query(
    case.query,
    mode=query_mode,
    top_k=config.top_k,
    options=config.query_options,
    trace=config.trace,
)
```

如果某个 case 查询失败，Runner 会把错误写入 `EvaluationCaseResult.error`，不会让整个评估报告失去结构。

默认查询并发：

```text
max_concurrent_queries = 8
```

如果 provider 限流较严格，可以调低。如果本地 store 和模型服务吞吐更高，可以调高。

## Evaluators

默认使用 benchmark adapter 声明的评分方法：

```python
benchmark.evaluators()
```

也可以在 run 时覆盖：

```python
result = await BenchmarkRunner().run(
    benchmark=benchmark,
    recipe=recipe,
    knowledge_base_name="kb",
    query_modes=("vector_search",),
    evaluators=(EvidenceRecallAtK(k=10),),
)
```

覆盖适合临时实验。正式 benchmark adapter 应该在 `benchmark.evaluators()` 中声明默认评分策略。

## Report Persistence

默认情况下，`persist_report=True` 时，report 会写入 KB 的 ObjectStore：

```text
_heta/knowledge_bases/{knowledge_base_name}/evaluations/{report_id}/report.json
```

关闭持久化：

```python
BenchmarkRunConfig(persist_report=False)
```

这时 Runner 只返回内存中的 `EvaluationReport`。

## JsonlBenchmark

`JsonlBenchmark` 是最小的本地 benchmark adapter，适合团队快速把自己的测试集接进 Heta。

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

`documents.jsonl` 示例：

```json
{"document_id":"doc_1","name":"doc.txt","media_type":"text/plain","text":"Marine biodiversity."}
```

也可以使用本地路径：

```json
{"document_id":"doc_1","name":"paper.pdf","media_type":"application/pdf","path":"paper.pdf"}
```

`cases.jsonl` 示例：

```json
{"case_id":"case_1","query":"What is discussed?","expected":{"answers":["Marine biodiversity"],"evidence":[{"locator":{"document_id":"doc_1"},"text":"Marine biodiversity."}]}}
```

`JsonlBenchmark` 默认 evaluators：

```text
EvidenceRecallAtK(k=5)
AnswerContains()
```
