# Heta Query Modes

Heta Framework 内置一组对齐 HetaDB 经验的组合 query modes。它们不是新的存储结构，而是基于已有基础检索能力组合出来的查询策略。

基础 query modes：

```text
vector_search
sql_text_search
full_text_search
heta_graph_search
```

组合 query modes：

| 模式 | 作用 | 依赖 |
| --- | --- | --- |
| `hybrid_search` | 向量检索和 Heta 图谱检索做加权 RRF 融合。 | `chunk_vector_index`、`graph_tables`、`graph_vector_index` |
| `heta_rerank_search` | Heta 混合检索和全文检索做 RRF 融合，并在提供 reranker 时重排候选。 | `chunk_vector_index`、`chunk_full_text_index`、`graph_tables`、`graph_vector_index` |
| `heta_rewrite_search` | 语言模型生成 3 个查询变体，分别执行 Heta 重排检索后再融合结果。 | `models.language`、默认依赖 `heta_rerank_search` 的资产 |
| `heta_multihop_search` | 最多 3 轮 Heta 重排检索、信息抽取和充分性判断，适合多跳问题。 | `models.language`、默认依赖 `heta_rerank_search` 的资产 |

组合模式通过 `QueryContext.query(...)` 调用基础能力，因此递归检测、资产检查和 trace 会走同一套路径。

## hybrid_search

`hybrid_search` 对齐 HetaDB 原来的“向量 + 图谱”混合检索路径，但 score 计算做了收紧。

HetaDB 原实现会把 chunk 向量分数、图谱召回带来的 chunk 出现次数、手写权重混合到一起。这个思路能工作，但不同来源的分数尺度不一致：向量相似度、图谱命中和 occurrence boost 不是同一种量，直接相加容易让某个来源在不同数据集里过强或过弱。

Framework 中改为 weighted RRF：

1. 调用 `vector_search` 召回 chunk。
2. 调用 `heta_graph_search` 召回实体、关系和证据。
3. 只使用各自结果列表中的排名，不直接混加原始分数。
4. 使用 per-mode 权重控制偏好。
5. 同一结果被多个来源命中时，自然获得更高融合分。

```python
response = await kb.query(
    "marine biodiversity evidence",
    mode="hybrid_search",
    top_k=8,
    options={
        "hybrid_weights": {
            "vector_search": 1.0,
            "heta_graph_search": 1.2,
        }
    },
)
```

可选参数：

| `options` 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `candidate_top_k` | `min(top_k * 3, 50)` | 每个基础检索模式召回的候选数量。 |
| `rrf_k` | `60` | RRF 平滑参数。 |
| `hybrid_weights` | `{"vector_search": 1.0, "heta_graph_search": 1.0}` | 不同检索来源的融合权重。 |

## heta_rerank_search

`heta_rerank_search` 对齐 HetaDB 的高精度检索路径。

默认流程：

1. 调用 `hybrid_search` 召回向量和 Heta 图谱候选。
2. 调用 `full_text_search` 召回全文检索匹配 chunk。
3. 使用 Reciprocal Rank Fusion 合并候选。
4. 如果 `KnowledgeModels.reranker` 存在，则调用 rerank 模型重新排序。
5. 如果没有 reranker，则保留 RRF 排序。

```python
response = await kb.query(
    "What loss function does the model use?",
    mode="heta_rerank_search",
    top_k=5,
)
```

可选参数：

| `options` 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `candidate_top_k` | `min(top_k * 3, 50)` | 每个基础检索模式召回的候选数量。 |
| `rrf_k` | `60` | RRF 平滑参数。 |

没有 `reranker` 时，`heta_rerank_search` 仍然可用，会退化为 RRF 融合排序。

## heta_rewrite_search

`heta_rewrite_search` 适合用户问题表述模糊、同义词较多或单次检索可能漏召回的场景。

默认流程：

1. 调用 `models.language` 生成 3 个查询变体。
2. 每个查询变体调用 `heta_rerank_search`。
3. 对多个变体的结果做 RRF 融合。
4. 如果 query rewrite 失败，则退化为一次基础检索。

```python
response = await kb.query(
    "how does the thing handle sequences",
    mode="heta_rewrite_search",
    top_k=8,
)
```

`heta_rewrite_search` 默认以 `heta_rerank_search` 作为基础检索模式。需要切换时，可以注册自定义 `RewriteSearchEngine(base_mode="hybrid_search", ...)`。

当 query rewrite 失败或模型没有返回可用查询变体时，响应不会静默吞掉问题。`QueryResponse.metadata["issues"]` 会记录失败原因和降级动作：

```python
{
    "code": "rewrite_invalid_output",
    "message": "Language model did not return a queries list.",
    "action": "used_base_search",
}
```

## heta_multihop_search

`heta_multihop_search` 适合需要串联多个事实的问题。它不会暴露内部推理文本，只返回最终答案、去重后的证据结果和可选 trace。

默认流程：

1. 使用当前问题调用基础检索模式。
2. 语言模型判断检索结果是否包含有用信息。
3. 语言模型判断累计信息是否足以回答原问题。
4. 如果足够，返回答案。
5. 如果不够，生成下一轮查询，最多执行 3 轮。
6. 仍未得到明确答案时，用累计证据生成一个保守回答。

```python
response = await kb.query(
    "How does the proposed method compare to the baseline across all datasets?",
    mode="heta_multihop_search",
    top_k=6,
    options={"max_rounds": 3},
)
```

可选参数：

| `options` 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `max_rounds` | `3` | 最大检索和判断轮数。 |

`heta_multihop_search` 会在 `metadata["round_reports"]` 中记录每轮查询、召回数量、是否抽取到有用信息、是否已经作答和下一轮查询。异常或降级情况写入 `metadata["issues"]`，例如无结果、没有抽取到有用信息，或达到最大轮数后只能生成保守回答。

## Recipe Requirements

组合查询模式依赖已有组件：

```python
from heta_framework.common.models import EmbeddingModel, LanguageModel, RerankModel
from heta_framework.kb import KnowledgeModels

models = KnowledgeModels(
    language=LanguageModel(model_name="openai/gpt-4o-mini", api_key="..."),
    embedding=EmbeddingModel(model_name="openai/text-embedding-3-small", api_key="..."),
    reranker=RerankModel(model_name="cohere/rerank-english-v3.0", api_key="..."),
)
```

没有 `language` 时，`heta_rewrite_search` 和 `heta_multihop_search` 不可用。没有 `reranker` 时，`heta_rerank_search` 会继续使用 RRF 排序。
