# Evaluation Reports

`EvaluationReport` 是 Heta Evaluation 的最终产物。

它记录一次 benchmark run 的完整结果：

```text
哪个 recipe
在哪个 benchmark 上
用哪些 query modes
得到什么查询结果
每个 evaluator 如何评分
整体 summary 是什么
```

`KnowledgeBase` 是 recipe 在 benchmark documents 上构建出的中间产物。`EvaluationReport` 才是最终评估结果。

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

主要字段：

| 字段 | 说明 |
| --- | --- |
| `report_id` | 本次评估运行的稳定 ID。 |
| `benchmark` | benchmark 的名称、版本、split 和任务类型。 |
| `knowledge_base_name` | 本次评估的 KB 基名。单 KB 模式下也是实际 KB 名称。 |
| `knowledge_base_manifest` | 单个 KB manifest，或多 KB 模式下的 manifest 列表。 |
| `recipe_manifest` | 被评估 recipe 的 manifest 快照。 |
| `query_modes` | 本次评估调用的 query modes。 |
| `score_summary` | evaluator 分数的聚合结果。 |
| `case_results` | 每个 case、每种 query mode 的详细结果。 |
| `metadata` | 运行配置、环境信息或 benchmark 特有信息。 |

报告同时保存 recipe 和 KB manifest，是为了让结果可追踪、可复现、可比较。

多 KB benchmark 会在 `metadata.run_units` 中记录每个执行单位：

```json
{
  "unit_id": "ADI_2009",
  "document_ids": ["ADI_2009"],
  "case_ids": ["ADI/2009/page_49.pdf-1"]
}
```

对应的 `EvaluationCaseResult.metadata` 会记录该 case 使用了哪个 unit 和哪个 KB。

## Case Results

每个 `EvaluationCaseResult` 对应：

```text
one benchmark case
one query mode
one QueryResponse or one error
```

结构：

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

`response` 直接使用 `kb.query(...)` 返回的 `QueryResponse`：

```text
results
answer
citations
trace
metadata
```

Evaluation 不定义另一套查询结果格式。它直接消费 Heta query layer 的标准输出。

## EvaluationScore

`EvaluationScore` 是单个 evaluator 的输出：

```python
EvaluationScore(
    name="evidence_recall@5",
    value=0.8,
    passed=None,
    metadata={
        "matched": 4,
        "expected": 5,
    },
)
```

字段含义：

| 字段 | 说明 |
| --- | --- |
| `name` | evaluator 名称。 |
| `value` | 分数、布尔结果或标签。 |
| `passed` | 可选 pass/fail 判断。 |
| `metadata` | 命中证据、缺失证据、judge reason 等调试信息。 |

## Default Location

当 `BenchmarkRunner` 使用 KB 的 ObjectStore 持久化报告时，默认位置是：

```text
_heta/knowledge_bases/{knowledge_base_name}/evaluations/{report_id}/report.json
```

这个位置和 KB runtime metadata 保持一致。

如果用户显式提供 output store 或 output prefix，Runner 可以写到用户指定位置。如果没有可用 ObjectStore，Runner 仍然可以只返回内存中的 `EvaluationReport`。

## Design Boundary

`EvaluationReport` 不改变 KB，只记录评估事实。

同一个 recipe 可以跑多个 benchmark：

```text
recipe -> UDA report
recipe -> MultiHop-RAG report
recipe -> LegalBench-RAG report
```

同一个 benchmark 可以比较多个 recipe：

```text
benchmark -> recipe A report
benchmark -> recipe B report
benchmark -> recipe C report
```

这也是 report 成为一等产物的原因。
