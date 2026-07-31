# Unified Graph Extraction Plan

本文档记录 5 种 RAG 图谱抽取流程统一化的设计草案和实施清单。

## Goal

将 HetaGraph、GraphRAG、LightRAG、HiRAG、LeanRAG 的图谱抽取流程拆成统一的三阶段：

1. 通用抽取：使用 HetaGraph 的抽取 prompt，在一个 unified extraction step 中先抽 entity，再基于 entity 抽 relation。
2. Schema 限制：可选 step，根据 dict / JSON-compatible ontology schema 对实体和关系进行约束、删除或规范化。
3. RAG-specific 适配：根据各 RAG 的结构需要，补充专属 prompt 抽取或格式转换。

## Design Principles

- 所有 prompt 文本集中放在 `src/heta_framework/kb/graphing/prompts.py`。
- 不从 HiRAG、LeanRAG、LightRAG 等外部项目或其他模块导入 prompt 文本。
- 第 2 步 schema 限制是可选 step。
- Schema 限制不改变实体和关系的数据形式。
- 基础实体和关系尽量使用 HetaGraph 的 `ExtractedEntity` / `ExtractedRelation` 作为统一中间表示。
- Universal extraction 必须是一个 step。该 step 内部可以按顺序发起 entity extraction 和 relation extraction 两类 LLM 调用，但对 recipe/procedure 暴露为一个统一图谱抽取 step。
- RAG-specific 阶段只补充各 RAG 必需的额外语义属性或做格式适配，不重新绕过前两步抽取基础图。

## Target Pipeline

```text
chunk text
  -> universal graph extraction step
       -> entity extraction
       -> relation extraction based on extracted entities
  -> optional ontology/schema constraint
  -> RAG-specific enrichment/adaptation
  -> target RAG graph/table format
```

## Stage 1: Universal Extraction

使用 HetaGraph 的抽取方式作为统一基础抽取。该阶段实现为一个 recipe/procedure 可见的 step，
在 step 内部串联执行实体抽取和关系抽取。

### Entity Extraction

Entity extraction 是 Universal Extraction step 的第一个内部子过程。

输入：

- chunk text
- allowed/general entity types
- entity extraction prompt

输出：

- entity name
- entity type
- entity description

最终落到统一实体形式：

- `name`
- `type`
- `description`
- runtime-generated metadata such as IDs and source chunk references

### Relation Extraction

Relation extraction 是 Universal Extraction step 的第二个内部子过程。它必须消费同一个 step
中已经抽取到的实体列表，再基于这些实体抽取关系。

输入：

- chunk text
- extracted entities
- relation extraction prompt

输出：

- source entity name
- target entity name
- relation type
- relation name
- relation description

最终落到统一关系形式：

- `source_entity_name`
- `target_entity_name`
- `type`
- `name`
- `description`
- runtime-generated metadata such as IDs and source chunk references

## Stage 2: Optional Ontology / Schema Constraint

该阶段作为独立可选 step 存在。开启后，它接收第 1 步输出的实体和关系，返回相同形式的实体和关系。

### Supported Schema Format

先支持 Python dict / JSON-compatible 配置，后续再扩展文件加载。

示例：

```json
{
  "entity_types": {
    "METHOD": {
      "description": "Algorithms, models, systems, or procedures.",
      "aliases": ["model", "approach", "framework"]
    },
    "DATASET": {
      "description": "Benchmarks, corpora, or evaluation datasets.",
      "aliases": ["benchmark", "corpus"]
    }
  },
  "relation_types": {
    "EVALUATED_ON": {
      "description": "A method is evaluated on a dataset.",
      "source_types": ["METHOD"],
      "target_types": ["DATASET"]
    },
    "IMPROVES": {
      "description": "A method improves another method or baseline.",
      "source_types": ["METHOD"],
      "target_types": ["METHOD"]
    }
  }
}
```

### Expected Behavior

Schema constraint prompt 应支持：

- 保留符合 schema 的实体和关系。
- 删除无证据、无关或无法映射到 schema 的实体。
- 删除端点实体不存在的关系。
- 删除 source/target 类型不符合 schema 约束的关系。
- 将自由实体类型映射到 schema 中允许的实体类型。
- 将自由关系类型映射到 schema 中允许的关系类型。
- 必要时修正实体或关系描述。

### Output Contract

输出仍保持：

- `ExtractedEntity` compatible records
- `ExtractedRelation` compatible records

不新增 schema-specific entity/relation 类型。

## Final Benchmark Plan

所有开发工作完成后，最终测试不启用 Ontology / Schema Constraint。也就是说：

- `Ontology / Schema Constraint` 设置为关闭。
- 5 种 RAG procedure 均使用统一抽取后的基础图，再进入各自 RAG-specific enrichment/adaptation。
- 使用 `scripts/benchmark_scifact_procedures.py` 运行 SciFact benchmark。
- 将新结果与既有目录 `heta-scifact-procedure-runs_combined` 中的结果进行对比。

该测试用于确认：在不引入 schema 约束影响的情况下，统一图谱抽取流程本身是否能稳定支撑
HetaGraph、GraphRAG、LightRAG、HiRAG、LeanRAG，并观察与旧流程的指标差异。

## Stage 3: RAG-Specific Enrichment / Adaptation

第 3 步根据目标 RAG 的需要进行 prompt 适配抽取或格式转换。

| RAG | Required Action |
|---|---|
| HetaGraph | 基本复用通用抽取结果；可选经过 schema constraint。 |
| GraphRAG | 将通用实体/关系转换为 GraphRAG 所需的 node/edge 表示；通常不需要额外 LLM。 |
| LightRAG | 在通用关系基础上补充 `relationship_keywords`，然后转换为 LightRAG 格式。 |
| HiRAG | 在通用基础图上继续做层级聚类、summary entity、community report 等抽取。 |
| LeanRAG | 在通用基础图上继续做 aggregate entity、aggregate description、cluster relation description。 |

## Prompt Inventory To Centralize

所有以下 prompt 应集中定义在 `src/heta_framework/kb/graphing/prompts.py`：

| Prompt | Purpose |
|---|---|
| `HETA_ENTITY_EXTRACTION_PROMPT` | 通用实体抽取。 |
| `HETA_RELATION_EXTRACTION_PROMPT` | 基于实体列表进行通用关系抽取。 |
| `ONTOLOGY_ENTITY_CONSTRAINT_PROMPT` | 根据 ontology schema 约束、删除或重分类实体。 |
| `ONTOLOGY_RELATION_CONSTRAINT_PROMPT` | 根据 ontology schema 约束、删除或重分类关系。 |
| `LIGHTRAG_RELATION_KEYWORDS_PROMPT` | 为 LightRAG 补充关系关键词。 |
| `HIRAG_COMMUNITY_SUMMARY_PROMPT` | 为 HiRAG 生成 community report 或 summary。 |
| `HIRAG_ENTITY_SUMMARY_PROMPT` | 为 HiRAG 汇总实体描述。 |
| `LEANRAG_AGGREGATE_ENTITY_PROMPT` | 为 LeanRAG 生成聚合实体。 |
| `LEANRAG_AGGREGATE_RELATION_PROMPT` | 为 LeanRAG 生成聚合实体之间的关系描述。 |

## Implementation Checklist

| Step | Task | Location | Purpose |
|---|---|---|---|
| 1 | 整理并集中所有图谱抽取 prompt | `src/heta_framework/kb/graphing/prompts.py` | 确保 5 种 RAG 的抽取、约束、适配 prompt 都从同一个文件取。 |
| 2 | 明确通用抽取 prompt | `graphing/prompts.py` | 使用 HetaGraph 的 entity-first、relation-after-entity prompt 作为统一基础抽取。 |
| 3 | 新增 ontology/schema constraint prompt template | `graphing/prompts.py` | 支持根据 dict / JSON-compatible schema 对实体和关系做保留、删除、类型重映射、描述修正。 |
| 4 | 新增 Universal Graph Extraction step | new step module | 在一个 step 内完成 entity extraction 和 relation extraction，输出 `ExtractedEntity` / `ExtractedRelation` compatible records。 |
| 5 | 新增 schema constraint step | new step module | 作为可选 step 接收 `ExtractedEntity` / `ExtractedRelation`，输出形式保持不变。 |
| 6 | 定义 schema 配置结构 | recipe/config layer | 支持传入 `entity_types`、`relation_types`、`source_types`、`target_types`。 |
| 7 | 在 procedure/recipe 中允许插入 optional schema step | procedure/recipe builder | 默认关闭，开启后插在通用抽取之后、RAG-specific 适配之前。 |
| 8 | 新增 RAG-specific prompt 适配模板 | `graphing/prompts.py` | 给 LightRAG / HiRAG / LeanRAG 保留专属语义补充 prompt。 |
| 9 | 改造 GraphRAG 抽取路径 | GraphRAG extraction step | 消费通用实体/关系结果，再转 GraphRAG 格式。 |
| 10 | 改造 LightRAG 抽取路径 | LightRAG extraction step | 消费通用实体/关系，额外补 `relationship_keywords`。 |
| 11 | 改造 HiRAG 抽取路径 | HiRAG extraction/build step | 消费通用实体/关系，再做层级、summary、community 相关抽取。 |
| 12 | 改造 LeanRAG 抽取路径 | LeanRAG extraction/build step | 消费通用实体/关系，再做 aggregate entity 和 aggregate relation。 |
| 13 | 保持 HetaGraph 路径兼容 | HetaGraph procedure | HetaGraph 本身应基本等于通用抽取加可选 schema step。 |
| 14 | 加配置开关 | recipe/config and benchmark config | 控制是否启用统一抽取、schema constraint、RAG-specific enrichment。 |
| 15 | 加最小单元测试 | `tests/` | 测试 prompt 解析、Universal Extraction step 输出、schema 过滤、类型映射、关系端点删除。 |
| 16 | 加小批量 live smoke 测试入口 | `scripts/` or benchmark config | 用少量 SciFact 数据验证 5 种 RAG 都能跑通。 |
| 17 | 关闭 schema constraint 进行最终 benchmark | `scripts/benchmark_scifact_procedures.py` | 组装 5 种 RAG procedure，运行 SciFact，并与 `heta-scifact-procedure-runs_combined` 对比。 |

## Suggested Phases

| Phase | Scope | Validation Target |
|---|---|---|
| Phase 1 | prompt 集中化 + schema constraint prompt + schema step | 不改变现有 5 种 RAG 行为。 |
| Phase 2 | 实现 Universal Graph Extraction step，并让 HetaGraph 使用通用抽取 + optional schema step | 验证一个 step 内完成 entity/relation 抽取后的中间格式稳定。 |
| Phase 3 | 接入 GraphRAG / LightRAG | 验证通用图到不同 RAG 格式的转换。 |
| Phase 4 | 接入 HiRAG / LeanRAG | 处理层级、聚合、summary 等复杂适配。 |
| Phase 5 | benchmark 开关和小批量实测 | 确认 5 种 RAG procedure 都能在统一抽取流程下跑通。 |
| Phase 6 | 关闭 schema constraint 跑完整 SciFact benchmark | 与 `heta-scifact-procedure-runs_combined` 对比结果差异。 |

## Open Decisions

- Schema constraint 默认策略： LLM 重分类后删除不合规项。
- Schema 配置最终放在 recipe、benchmark 参数，还是单独 schema registry。
- RAG-specific enrichment 是允许复用第 1 步的原 chunk text
- 统一抽取作为默认路径启用
