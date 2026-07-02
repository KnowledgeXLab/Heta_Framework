# Evaluators

Evaluator 是 benchmark 使用的评分方法。

Heta Evaluation 遵循一个简单原则：

```text
Benchmark owns scoring policy.
Common evaluators are reusable building blocks.
```

也就是说，benchmark adapter 决定默认怎么评分；Heta 提供一组常用 evaluator，避免每个 benchmark 都重复实现 recall、exact match 或 answer contains。

## Protocol

```python
class BenchmarkEvaluatorProtocol(Protocol):
    @property
    def name(self) -> str: ...

    async def evaluate(
        self,
        *,
        case: BenchmarkCase,
        response: QueryResponse,
    ) -> EvaluationScore: ...
```

输入：

```text
BenchmarkCase
    benchmark 的问题、标准答案和证据标签。

QueryResponse
    kb.query(...) 的标准输出。
```

输出：

```python
EvaluationScore(
    name="evidence_recall@5",
    value=0.8,
    passed=None,
    metadata={"matched": 4, "expected": 5},
)
```

`value` 可以是：

```text
float
bool
str
```

`passed` 用于表达可选的 pass/fail 判断。`metadata` 用于保留命中证据、缺失证据、judge reason 等调试信息。

## EvidenceRecallAtK

```python
from heta_framework.evaluation import EvidenceRecallAtK

EvidenceRecallAtK(k=5)
```

`EvidenceRecallAtK` 用 `BenchmarkCase.expected.evidence` 对比 `QueryResponse.results[:k]`。

匹配顺序：

```text
1. locator match
2. reference_id match
3. text match
```

`locator` 支持开放字段。内置匹配会识别常见字段：

```text
document_id
source_key
object_key
page_index
chunk_id
```

如果 query result 的 `source` 里带有这些字段，评估会更准确。如果没有，也可以通过 gold evidence 的 `text` 做文本匹配。

## BeirRetrievalMetric

```python
from heta_framework.evaluation import BeirRetrievalMetric

BeirRetrievalMetric(metric="ndcg", k=10)
BeirRetrievalMetric(metric="recall", k=10)
```

`BeirRetrievalMetric` 用于 BEIR 这类标准信息检索 benchmark。

支持的指标：

```text
ndcg
map
recall
precision
mrr
```

BEIR 的 qrels 是 document-level。Heta 的 query result 通常是 chunk-level。因此 evaluator 会先把命中的 chunk 映射回 benchmark document id，并按 document 去重，再计算指标。

默认 BEIR 指标可以直接使用：

```python
from heta_framework.evaluation import beir_default_metrics

evaluators = beir_default_metrics(
    k_values=(1, 3, 5, 10, 100),
)
```

## AnswerContains

```python
from heta_framework.evaluation import AnswerContains

AnswerContains()
```

判断 `QueryResponse.answer` 是否包含任一 `case.expected.answers`。

这个 evaluator 适合宽松 QA 评估，例如答案可以包含解释性文字，只要覆盖标准答案即可。

## AnswerExactMatch

```python
from heta_framework.evaluation import AnswerExactMatch

AnswerExactMatch()
```

判断 `QueryResponse.answer` 归一化后是否等于任一 `case.expected.answers`。

这个 evaluator 适合短答案、枚举值、分类结果。

## Custom Evaluator

用户可以实现自己的 evaluator：

```python
class MyEvaluator:
    name = "my_score"

    async def evaluate(self, *, case, response):
        return EvaluationScore(
            name=self.name,
            value=1.0,
        )
```

然后在 benchmark 中声明：

```python
def evaluators(self):
    return (MyEvaluator(),)
```

或者在 Runner 中临时覆盖：

```python
BenchmarkRunner().run(
    benchmark=benchmark,
    recipe=recipe,
    knowledge_base_name="kb",
    query_modes=("vector_search",),
    evaluators=(MyEvaluator(),),
)
```
