# Benchmark Protocols

Heta Evaluation 的评价对象是 `KnowledgeRecipe`。

Benchmark 提供数据和默认评估方法，Runner 后续会用指定 recipe 建库、查询、评分，并生成 `EvaluationReport`。

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

`KnowledgeBase` 是中间构建产物。
`EvaluationReport` 是最终评估产物。

## BenchmarkProtocol

Benchmark adapter 需要满足 `BenchmarkProtocol`：

```python
class BenchmarkProtocol(Protocol):
    @property
    def manifest(self) -> BenchmarkManifest: ...

    def resources(self) -> tuple[BenchmarkResource, ...]: ...

    async def prepare(
        self,
        workspace: BenchmarkWorkspace,
    ) -> PreparedBenchmark: ...

    async def documents(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkDocument]: ...

    async def cases(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkCase]: ...

    async def run_units(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkRunUnit]: ...

    def evaluators(self) -> tuple[BenchmarkEvaluatorProtocol, ...]: ...
```

职责划分：

| 方法 | 职责 |
| --- | --- |
| `manifest` | 声明 benchmark 名称、版本、split、任务类型和引用信息。 |
| `resources()` | 声明下载或准备 benchmark 所需的外部资源。 |
| `prepare()` | 下载、解压、校验或定位本地数据，返回 prepared state。 |
| `documents()` | 产出需要写入 KB `raw/` 的原始文档。 |
| `cases()` | 产出查询样本、标准答案和证据标签。 |
| `run_units()` | 声明 Runner 应该建几个 KB，以及每个 KB 使用哪些文档和 cases。 |
| `evaluators()` | 声明这个 benchmark 默认用哪些方法评分。 |

Benchmark 不负责建库，不调用 query engine，不聚合报告。

## Run Units

`BenchmarkRunUnit` 是一次独立建库和评估的执行单位：

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

字段含义：

| 字段 | 说明 |
| --- | --- |
| `unit_id` | 本次执行单位的稳定 ID。 |
| `document_ids` | 本 unit 要写入 KB 的文档 ID。为空表示使用全部文档。 |
| `case_ids` | 本 unit 要评估的 case ID。为空表示使用全部 cases。 |
| `metadata` | benchmark 特有信息，例如 `doc_name`、subset 或 source split。 |

因此 Runner 实际支持两种情况：

```text
corpus unit
    一个 KB 跑全部 cases。

many units
    多个 KB，各自跑对应 cases，最后汇总一个 report。
```

## Build Scope

`BenchmarkManifest` 声明 `build_scope`：

```python
BenchmarkManifest(
    name="multihop_rag",
    version="official",
    split="all",
    task_type="multi_hop_qa",
    build_scope="corpus",
)
```

`build_scope` 是 benchmark 的语义提示，不是 Runner 的执行开关。
真正的执行计划由 `run_units()` 决定。

推荐语义：

| build_scope | 语义 |
| --- | --- |
| `corpus` | 整个 benchmark 共用一个 corpus，一次建库，多 case 查询。 |
| `case` | benchmark 内存在多个小 KB 执行单位，每个 case 或 case 集合绑定自己的文档。 |
| `group` | 预留给显式 domain、topic 或 split group 场景。 |

BEIR 属于 `corpus`：

```text
corpus.jsonl
  -> 建一次 KB

queries.jsonl
  -> 多个 query case
```

UDA-fin 属于 `case`，但 Runner 会按 `doc_name` 聚合 run units：

```text
ADI_2009.pdf
  -> 建一个 KB
  -> 跑 ADI_2009 的 cases
```

## Documents

`BenchmarkDocument` 是 benchmark 输入给 KB 的原始文件：

```python
BenchmarkDocument(
    document_id="doc_001",
    name="paper.pdf",
    media_type="application/pdf",
    path=Path("paper.pdf"),
)
```

`document_id` 必须在 benchmark 内稳定唯一。
`name` 是文件名，不是路径。
`data`、`path`、`source_uri` 三者必须且只能设置一个。

默认 raw key 规则：

```text
raw/benchmarks/{benchmark_name}/{split}/{document_id}/{name}
```

这个 key 会进入 parser 的 `ParsedSource`，并最终出现在 query result 的 provenance 中。

## Cases

`BenchmarkCase` 是一条评估样本：

```python
BenchmarkCase(
    case_id="case_001",
    query="What ocean management topics are discussed?",
    expected=BenchmarkExpected(
        answers=("marine protected areas",),
        evidence=(
            BenchmarkEvidence(
                reference_id="doc_001_p4",
                locator={
                    "document_id": "doc_001",
                    "page_index": 4,
                },
                text="...",
            ),
        ),
    ),
)
```

`BenchmarkExpected` 支持：

| 字段 | 用途 |
| --- | --- |
| `answers` | QA 类 benchmark 的标准答案。 |
| `evidence` | 检索、引用、grounding 评估需要的证据标签。 |
| `value` | 数值、分类、结构化输出等非纯文本答案。 |
| `metadata` | benchmark 特有信息。 |

`BenchmarkEvidence.locator` 是开放结构。
内置 evaluator 会识别常见字段，例如：

```text
document_id
source_key
page_index
chunk_id
table_id
row_index
column
```

特殊 benchmark 可以在 `locator` 里加入自己的字段。

## Evaluators

Benchmark 自己声明默认评估方法：

```python
def evaluators(self):
    return (
        EvidenceRecallAtK(k=5),
        AnswerContains(),
    )
```

这体现一个原则：

```text
Benchmark owns scoring policy.
Common evaluators are reusable building blocks.
```

很多 benchmark 的评估方法和数据标签强耦合。
因此 evaluator 不是脱离 benchmark 的主角，而是 benchmark 声明出来的评分方法。

通用 evaluator 后续会放在：

```text
heta_framework.evaluation.evaluators
```

例如：

```text
EvidenceRecallAtK
EvidencePrecisionAtK
AnswerExactMatch
AnswerContains
FaithfulnessJudge
```

## Custom Benchmark

用户自定义 benchmark 只需要实现同一协议。

最小实现包括：

```text
manifest
resources()
prepare()
documents()
cases()
evaluators()
```

本地私有 benchmark 可以让 `resources()` 返回空 tuple，并在 `prepare()` 中直接返回已有目录。
