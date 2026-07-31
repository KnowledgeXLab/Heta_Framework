# Unified Graph Extraction Implementation

本文档说明当前 5 种 RAG 图谱抽取统一化的具体实现。

## Overview

当前实现将 HetaGraph、GraphRAG、LightRAG、HiRAG、LeanRAG 的基础图谱抽取统一为：

```text
chunks
  -> ExtractUniversalGraph
  -> optional ConstrainGraphByOntology
  -> RAG-specific universal graph adapter
  -> each RAG-specific build / community / aggregation steps
```

其中：

- `ExtractUniversalGraph` 是统一基础抽取 step。
- `ConstrainGraphByOntology` 是可选 ontology/schema constraint step，默认关闭。
- RAG-specific universal graph adapter 是面向不同 RAG 的独立适配 step。

抽取、约束、适配已拆分为独立实现文件。
抽取、约束和适配逻辑不再放在同一个模块中。
`universal_graph_common.py` 只保存跨 step 共享的内部 helper，
例如组件校验、JSON 序列化、按 chunk 分组、稳定 ID 和权重计算。
`universal_graph.py` 只保留向后兼容导出，已有
`heta_framework.kb.steps.universal_graph` 导入路径仍然可用。

核心实现文件：

- `src/heta_framework/kb/steps/extract_universal_graph.py`
- `src/heta_framework/kb/steps/constrain_graph_by_ontology.py`
- `src/heta_framework/kb/steps/adapt_universal_graph.py`
- `src/heta_framework/kb/steps/universal_graph_common.py`
- `src/heta_framework/kb/steps/universal_graph.py` (compatibility exports)
- `src/heta_framework/kb/graphing/prompts.py`
- `src/heta_framework/kb/procedures/heta_graph.py`
- `src/heta_framework/kb/procedures/graphrag.py`
- `src/heta_framework/kb/procedures/lightrag.py`
- `src/heta_framework/kb/procedures/hirag.py`
- `src/heta_framework/kb/procedures/leanrag.py`

## Prompt Centralization

所有新增和统一后的 graph extraction prompt 都集中在：

```text
src/heta_framework/kb/graphing/prompts.py
```

新增的主要 prompt / prompt set：

| Name | Purpose |
|---|---|
| `HETA_ENTITY_EXTRACTION_PROMPT` | HetaGraph entity prompt 的统一别名。 |
| `HETA_RELATION_EXTRACTION_PROMPT` | HetaGraph relation prompt 的统一别名。 |
| `ONTOLOGY_ENTITY_CONSTRAINT_PROMPT` | 根据 ontology schema 约束实体。 |
| `ONTOLOGY_RELATION_CONSTRAINT_PROMPT` | 根据 ontology schema 约束关系。 |
| `LIGHTRAG_RELATION_KEYWORDS_PROMPT` | 为 LightRAG 关系补充 keywords。 |
| `LIGHTRAG_PROMPTS` | LightRAG legacy extraction step 的本地 prompt set。 |
| `HIRAG_PROMPTS` | HiRAG summary/community 等 prompt set。 |
| `LEANRAG_PROMPTS` | LeanRAG aggregation prompt set。 |

同时移除了以下 step 中从外部项目加载 prompt 的逻辑：

- `extract_lightrag_graph.py`
- `hirag_hierarchical_aggregation.py`
- `leanrag_semantic_aggregation.py`

这些 step 现在只从 `heta_framework.kb.graphing.prompts` 获取 prompt。旧的
`extract_hirag_graph.py` / `extract_leanrag_graph.py` 仅保留为兼容导出入口。

## Step 1: ExtractUniversalGraph

实现位置：

```text
src/heta_framework/kb/steps/extract_universal_graph.py
```

类：

```python
ExtractUniversalGraph
ExtractUniversalGraphConfig
ExtractUniversalGraphResult
```

该 step 对 recipe/procedure 暴露为一个 step，并且不再复用旧的 `ExtractEntities`
和 `ExtractRelations` step。实体抽取、关系抽取、解析、ID 生成、artifact 写入
都直接内聚在 `ExtractUniversalGraph` 自身内部。

内部顺序为：

1. 使用 HetaGraph entity prompt 抽取实体。
2. 将实体解析为 `ExtractedEntity`。
3. 使用 HetaGraph relation prompt，基于当前 chunk 的实体列表抽取关系。
4. 将关系解析为 `ExtractedRelation`。
5. 写入 entity / relation object artifacts。

也就是说，旧的两个 step 仍可作为兼容 API 存在。
新版 procedure 不再通过它们完成基础图谱抽取。

输入 artifact：

| Artifact | Description |
|---|---|
| `chunk_keys` | chunk object keys。 |

输出 artifact：

| Artifact | Description |
|---|---|
| `entity_keys` | `ExtractedEntity` JSON objects。 |
| `relation_keys` | `ExtractedRelation` JSON objects。 |
| `extract_universal_graph_result` | 汇总统计和失败 chunk。 |

输出数据形式仍然是 HetaGraph 的强类型中间表示：

- `ExtractedEntity`
- `ExtractedRelation`

LLM trace 标识：

| Stage | Trace |
|---|---|
| Entity extraction | `step=extract_universal_graph`, `stage=entity_extraction` |
| Relation extraction | `step=extract_universal_graph`, `stage=relation_extraction` |

与旧实现的差异：

| 项目 | 旧组合方式 | 当前实现 |
|---|---|---|
| Recipe 可见 step | `ExtractEntities` + `ExtractRelations` | `ExtractUniversalGraph` |
| 内部实现 | 调用旧 step | 直接实现 prompt 调用和解析 |
| Entity artifact | 由 `ExtractEntities` 写入 | 由 `ExtractUniversalGraph` 写入 |
| Relation artifact | 由 `ExtractRelations` 写入 | 由 `ExtractUniversalGraph` 写入 |
| Trace step | `extract_entities` / `extract_relations` | `extract_universal_graph` |

## Step 2: ConstrainGraphByOntology

实现位置：

```text
src/heta_framework/kb/steps/constrain_graph_by_ontology.py
```

类：

```python
ConstrainGraphByOntology
ConstrainGraphByOntologyConfig
ConstrainGraphByOntologyResult
```

该 step 是可选项，默认不启用。

它接收：

- `chunk_keys`
- `entity_keys`
- `relation_keys`
- `schema`

其中 `schema` 当前支持 Python dict / JSON-compatible 配置。

示例：

```json
{
  "entity_types": {
    "METHOD": {
      "description": "Algorithms, models, systems, or procedures.",
      "aliases": ["model", "approach", "framework"]
    }
  },
  "relation_types": {
    "EVALUATED_ON": {
      "description": "A method is evaluated on a dataset.",
      "source_types": ["METHOD"],
      "target_types": ["DATASET"]
    }
  }
}
```

行为：

- 对实体执行 schema 约束、删除或类型重映射。
- 对关系执行 schema 约束、删除或类型重映射。
- 删除端点实体不存在的关系。
- 输出仍保持 `ExtractedEntity` / `ExtractedRelation` compatible JSON。

默认关闭原因：

- 最终 benchmark 要测试统一抽取本身。
- 尽量减少 ontology/schema constraint 的额外影响。
- 与 `heta-scifact-procedure-runs_combined` 对比时保持变量更少。

## Step 3: RAG-specific Graph Adapters

实现位置：

```text
src/heta_framework/kb/steps/adapt_universal_graph.py
```

类：

```python
AdaptUniversalGraphForGraphRAG
AdaptUniversalGraphForGraphRAGConfig
AdaptUniversalGraphForLightRAG
AdaptUniversalGraphForLightRAGConfig
AdaptUniversalGraphForHiRAG
AdaptUniversalGraphForHiRAGConfig
AdaptUniversalGraphForLeanRAG
AdaptUniversalGraphForLeanRAGConfig
AdaptUniversalGraphResult
```

这些 step 将统一图谱结果转换为不同 RAG 后续 build step 需要的 artifact。
每种 RAG 使用独立 adapter class，procedure 不再通过 `target` 参数把不同适配逻辑
混在同一个 step 中。`AdaptUniversalGraph` / `AdaptUniversalGraphConfig` 仅作为
向后兼容 dispatcher 保留。

### GraphRAG Adaptation

Step class:

```python
AdaptUniversalGraphForGraphRAG
```

输入：

- `ExtractedEntity`
- `ExtractedRelation`

输出：

- `graph_node_keys`
- `graph_edge_keys`

Graph node properties 包括：

- `name`
- `entity_name`
- `entity_type`
- `description`
- `source_id`
- `source_ids`
- `universal_entity_ids`

Graph edge properties 包括：

- `description`
- `weight`
- `source_id`
- `source_ids`
- `relation_type`
- `relation_name`
- `universal_relation_ids`

后续继续使用：

- `GraphCommunity`
- `BuildRAGGraph`

### LightRAG Adaptation

Step class:

```python
AdaptUniversalGraphForLightRAG
```

输出：

- `light_rag_graph_node_keys`
- `light_rag_graph_edge_keys`

LightRAG 关系额外需要：

- `keywords`
- `src_id`
- `tgt_id`
- `extraction_format`

`keywords` 的生成方式：

1. 优先调用 `LIGHTRAG_RELATION_KEYWORDS_PROMPT`。
2. 如果调用失败，回退为 relation `type` / `name` 组合。

后续继续使用：

- `BuildLightRAGGraph`

### HiRAG Adaptation

Step class:

```python
AdaptUniversalGraphForHiRAG
```

输出：

- `hi_rag_graph_node_keys`
- `hi_rag_graph_edge_keys`
- `hi_rag_chunks`
- `hi_rag_base_entities`
- `hi_rag_base_relations`
- `hi_rag_extraction_trace`

HiRAG base entity 会补充：

- `raw_entity_type`
- `layer`
- `cluster_id`
- `is_summary`
- `parent_entity_ids`

HiRAG base relation 会补充：

- `src_id`
- `tgt_id`
- `weight`
- `order`
- `layer`
- `is_summary`

后续继续使用：

- `HiRAGHierarchicalAggregation`
- `HiRAGCommunity`
- `BuildHiRAGGraph`

### LeanRAG Adaptation

Step class:

```python
AdaptUniversalGraphForLeanRAG
```

输出：

- `lean_rag_graph_node_keys`
- `lean_rag_graph_edge_keys`
- `lean_rag_chunks`
- `lean_rag_base_entities`
- `lean_rag_base_relations`
- `lean_rag_extraction_trace`

LeanRAG base entity 会补充：

- `degree`
- `parent`
- `level`
- `is_aggregate`
- `children`

LeanRAG base relation 会补充：

- `src_tgt`
- `tgt_src`
- `weight`
- `level`
- `is_generated`
- `evidence_relation_ids`

后续继续使用：

- `LeanRAGSemanticAggregation`
- `BuildLeanRAGGraph`

## Procedure Integration

### HetaGraphProcedure

历史流程：

```text
ExtractEntities
  -> ExtractRelations
  -> optional dedup
  -> BuildGraph / MergeGraphIntoStore
```

新流程：

```text
ExtractUniversalGraph
  -> optional ConstrainGraphByOntology
  -> optional dedup
  -> BuildGraph / MergeGraphIntoStore
```

### GraphRAGProcedure

新流程：

```text
ExtractUniversalGraph
  -> optional ConstrainGraphByOntology
  -> AdaptUniversalGraphForGraphRAG
  -> GraphCommunity
  -> BuildRAGGraph
```

### LightRAGProcedure

新流程：

```text
ExtractUniversalGraph
  -> optional ConstrainGraphByOntology
  -> AdaptUniversalGraphForLightRAG
  -> BuildLightRAGGraph
```

### HiRAGProcedure

新流程：

```text
ParseDocuments
  -> SplitDocuments
  -> ExtractUniversalGraph
  -> optional ConstrainGraphByOntology
  -> AdaptUniversalGraphForHiRAG
  -> HiRAGHierarchicalAggregation
  -> HiRAGCommunity
  -> BuildHiRAGGraph
```

### LeanRAGProcedure

新流程：

```text
ParseDocuments
  -> SplitDocuments
  -> ExtractUniversalGraph
  -> optional ConstrainGraphByOntology
  -> AdaptUniversalGraphForLeanRAG
  -> LeanRAGSemanticAggregation
  -> BuildLeanRAGGraph
```

## Public Exports

新增 step 已导出到：

```text
src/heta_framework/kb/steps/__init__.py
src/heta_framework/kb/__init__.py
```

`src/heta_framework/kb/steps/__init__.py` 现在直接从三个独立实现模块导入：

```text
heta_framework.kb.steps.extract_universal_graph
heta_framework.kb.steps.constrain_graph_by_ontology
heta_framework.kb.steps.adapt_universal_graph
```

旧路径 `heta_framework.kb.steps.universal_graph` 仍可导入同一组 class / config /
result 类型，但该文件不再承载具体实现。

可直接导入：

```python
from heta_framework.kb import (
    ExtractUniversalGraph,
    ExtractUniversalGraphConfig,
    ConstrainGraphByOntology,
    ConstrainGraphByOntologyConfig,
    AdaptUniversalGraphForGraphRAG,
    AdaptUniversalGraphForGraphRAGConfig,
    AdaptUniversalGraphForLightRAG,
    AdaptUniversalGraphForLightRAGConfig,
    AdaptUniversalGraphForHiRAG,
    AdaptUniversalGraphForHiRAGConfig,
    AdaptUniversalGraphForLeanRAG,
    AdaptUniversalGraphForLeanRAGConfig,
)
```

## Benchmark Script

更新位置：

```text
scripts/benchmark_scifact_procedures.py
```

当前行为：

- 打开 5 种 RAG：
  - `lightrag`
  - `graphrag`
  - `hirag`
  - `leanrag`
  - `heta`
- 显式设置：

```python
schema_constraint_enabled=False
```

- 默认输出目录：

```text
heta-scifact-procedure-runs_unified
```

- baseline 目录：

```text
heta-scifact-procedure-runs_combined
```

如果 baseline 的 `summary.json` 存在，脚本会在新 summary 中写入：

```json
"baseline_comparison": {
  "...": {
    "current_status": "...",
    "baseline_status": "...",
    "current_case_error_count": 0,
    "baseline_case_error_count": 0,
    "score_delta": {}
  }
}
```

## Verification

拆分后已执行语法检查：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  src/heta_framework/kb/steps/universal_graph.py \
  src/heta_framework/kb/steps/universal_graph_common.py \
  src/heta_framework/kb/steps/extract_universal_graph.py \
  src/heta_framework/kb/steps/constrain_graph_by_ontology.py \
  src/heta_framework/kb/steps/adapt_universal_graph.py \
  src/heta_framework/kb/steps/__init__.py
```

已执行公共导入和兼容导入检查：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python - <<'PY'
from heta_framework.kb.steps import (
    AdaptUniversalGraphForGraphRAG,
    ConstrainGraphByOntology,
    ExtractUniversalGraph,
)
from heta_framework.kb.steps.universal_graph import ExtractUniversalGraph as CompatExtract

assert ExtractUniversalGraph.__module__ == "heta_framework.kb.steps.extract_universal_graph"
assert ConstrainGraphByOntology.__module__ == (
    "heta_framework.kb.steps.constrain_graph_by_ontology"
)
assert AdaptUniversalGraphForGraphRAG.__module__ == (
    "heta_framework.kb.steps.adapt_universal_graph"
)
assert CompatExtract is ExtractUniversalGraph
PY
```

已执行 procedure 展开测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_procedures.py
```

结果：

```text
8 passed
```

已执行基础行长扫描。

未执行：

- 未执行真实 SciFact 全量 benchmark。
- 当前环境缺少 `ruff`，因此未执行 Ruff lint / format 检查。

## Current Limitations

- `ConstrainGraphByOntology` 已实现，但默认关闭，尚未用真实 benchmark 验证。
- RAG-specific adapter 当前主要做结构适配和必要字段补齐。
- 它们不重新执行各 RAG 原始基础抽取 prompt。
- LightRAG keywords 是唯一默认会在 adapter 中额外调用 LLM 的 RAG-specific enrichment。
- HiRAG / LeanRAG 的层级、community、aggregation 仍复用原有后续 step。
- 统一抽取后，不同 RAG 的基础图更一致。
- 方法差异主要体现在后续结构构建、summary、community 和 aggregation。
- 查询阶段仍保留各 RAG procedure 的差异。
