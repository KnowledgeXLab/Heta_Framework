# Step Protocols

Build Steps 是 Heta 知识库构建流程中的可组合执行单元。每个 step 都描述一个稳定的构建动作，并声明运行前后可用的能力。

Step 协议解决三个问题：

- 这个 step 需要哪些模型、存储、parser 或中间产物
- 这个 step 完成后会产生哪些 artifacts
- 这个 step 是否会解锁新的 query mode

这种设计让 Recipe 可以用显式步骤表达知识库能力，而不是用一组难以扩展的布尔开关。

```python
steps = [
    ParseDocuments(),
    SplitDocuments(),
    EmbedChunks(),
    IndexVectors(),
    ExtractEntities(),
    ExtractRelations(),
    BuildGraph(),
]
```

当前 parser、chunk、embedding、vector、SQL 持久化和 Heta-style graph 构建步骤都复用同一套协议。

## Step Groups

Step 是原子执行单元，但文档中可以按常见构建目标分组理解。
这些分组不是新的执行协议；真正运行时仍然是一组展开后的 steps。

| 分组 | Steps | 说明 |
| --- | --- | --- |
| 基础文档索引 | `ParseDocuments`、`SplitDocuments`、`EmbedChunks`、`IndexVectors` | 从原始文件到 chunk 向量索引，完成后提供 `vector_search`。 |
| Heta graph build | `MergeChunks`、`RechunkDocuments`、`PersistChunks`、`ExtractEntities`、`ExtractRelations`、`DeduplicateEntities`、`DeduplicateRelations`、`BuildGraph` | `IndexVectors` 之后的 HetaDB-style 建图链路。`MergeChunks`、`RechunkDocuments`、`PersistChunks` 是可选准备步骤，服务于后续图谱抽取和溯源，不提供新的 Milvus 检索库。 |
| Heta graph merge | `MergeChunks`、`RechunkDocuments`、`PersistChunks`、`ExtractEntities`、`ExtractRelations`、`DeduplicateEntities`、`DeduplicateRelations`、`MergeGraphIntoStore` | 动态增量图谱链路，最终合并进已有 SQL/vector graph store。 |

`HetaGraphProcedure` 只做静态接线和 step 展开，不读取 context、不访问 store、不执行任务。
Recipe runner 仍然基于展开后的 steps 做依赖校验和调度。

## Design Principles

Step 是构建动作，不是组件容器。

模型、存储和 parser 应放在 Recipe 顶层，由 step 通过组件引用获取。这样 Recipe 可以清楚展示使用了哪些基础组件，step 也不会持有隐式资源。

```python
ExtractRelations(model="strong")
```

这个写法表示“使用 Recipe 中名为 `strong` 的 language model”，而不是把模型实例直接塞进 step。默认不写名称时，step 使用对应类型的默认组件。

这样做的好处是：

- 简单 Recipe 保持简洁
- 复杂 Recipe 可以使用多个同类组件
- 执行器可以在运行前统一校验依赖
- Recipe summary、trace 和序列化更稳定

## Step Contract

一个 step 需要满足 `KnowledgeStepProtocol`：

```python
class KnowledgeStepProtocol(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def requirements(self) -> StepRequirements: ...

    @property
    def capabilities(self) -> StepCapabilities: ...

    async def run(self, context: StepContextProtocol) -> None: ...

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan: ...
```

| 成员 | 说明 |
| --- | --- |
| `name` | 稳定名称，用于日志、trace 和 recipe summary。 |
| `requirements` | 运行前需要的组件、artifacts 或 query modes。 |
| `capabilities` | 完成后提供的 artifacts 或 query modes。 |
| `run` | 执行 step。 |
| `cleanup_plan` | 声明这个 step 创建的持久化资源，供 `KnowledgeBase.delete()` 统一删除。 |

`requirements` 和 `capabilities` 是执行器做依赖校验和能力开放的基础。
`cleanup_plan` 不执行删除，只返回由当前 step 负责声明的派生产物。
原始输入，例如 ObjectStore 中的 `raw/` 文件，不应该进入 cleanup plan。

## Component References

组件引用使用 `ComponentRef`，通过 helper 创建：

```python
from heta_framework.kb.steps import model_ref, parser_ref, store_ref

model_ref("embedding").key
# "models.embedding"

model_ref("language", "strong").key
# "models.language.strong"

store_ref("vector").key
# "stores.vector"

store_ref("graph", "private").key
# "stores.graph.private"

parser_ref().key
# "parsers.documents"
```

默认组件不带 `name`。命名组件带 `name`：

```python
model_ref("language")           # default language model
model_ref("language", "strong") # named language model
```

Recipe 侧可以对应地提供一个默认模型和多个命名模型。Step 只引用名称，不直接保存实例。

## Requirements

`StepRequirements` 声明 step 运行前必须具备的条件：

```python
StepRequirements(
    components=frozenset({
        model_ref("embedding"),
        store_ref("vector"),
    }),
    artifacts=frozenset({"chunk_keys"}),
    queries=frozenset(),
)
```

| 字段 | 说明 |
| --- | --- |
| `components` | 需要的 Recipe 组件，例如模型、store 或 parser registry。 |
| `artifacts` | 需要的中间产物，例如 `parsed_document_keys`、`chunk_keys`。 |
| `queries` | 需要已经可用的查询能力，例如 `vector_search`。 |

## Capabilities

`StepCapabilities` 声明 step 完成后提供的能力：

```python
StepCapabilities(
    artifacts=frozenset({"embeddings"}),
    queries=frozenset({"vector_search"}),
)
```

查询能力应该由明确的 step 提供。例如：

```text
IndexVectors  -> vector_search
BuildGraph    -> heta_graph_search
EnableHybrid  -> hybrid_search
```

Hybrid search 不建议由系统自动推导。它通常需要 fusion、rerank、score normalization 和 query rewrite 等策略，应该通过显式 step 表达。

## Issues

Step 可以把可恢复问题写入 result 的 `issues` 字段。Issue 是运行诊断信息，不是主产物；主产物仍应保持干净、可继续被后续 step 消费。

通用 issue 协议使用 `StepIssue`：

```python
StepIssue(
    step="deduplicate_entities",
    subject=IssueSubject(type="dedup_group", id="上海市"),
    code="invalid_llm_output",
    severity="warning",
    message="LLM output is missing a non-empty Description field.",
    resolution=IssueResolution(
        action="kept_original_records",
        outcome="The group was not merged, and original records were kept.",
    ),
    details={"attempt_count": "3"},
)
```

字段含义：

| 字段 | 说明 |
| --- | --- |
| `step` | 产生 issue 的 step 名称。 |
| `subject` | issue 影响的对象，例如 `document`、`chunk`、`dedup_group` 或 `vector_record`。 |
| `code` | 稳定错误码，用于统计、过滤和测试。 |
| `severity` | 问题等级，当前支持 `info`、`warning`、`error`。 |
| `message` | 面向开发者的可读说明。 |
| `resolution` | 框架采取的恢复动作和结果。 |
| `details` | 少量结构化补充信息。 |

可恢复问题应该记录 issue 并继续 pipeline；不可恢复问题才应该让 step fail。后续 Recipe runner 可以统一收集各 step 的 issues，生成 build report。

## Step Context

Step 通过 `StepContextProtocol` 访问组件和中间产物：

```python
class StepContextProtocol(Protocol):
    def get_component(self, key: str) -> Any: ...
    def get_artifact(self, key: str) -> Any: ...
    def set_artifact(self, key: str, value: Any) -> None: ...
```

Step 只依赖这个协议。真实执行器后续会负责：

- 解析 component refs
- 校验 step requirements
- 管理 artifacts
- 记录状态、trace 和错误
- 控制并发、重试和取消

## Example

一个向量索引 step 可以这样声明自己的依赖和能力：

```python
class IndexVectors:
    name = "index_vectors"

    @property
    def requirements(self):
        return StepRequirements(
            components=frozenset({store_ref("vector")}),
            artifacts=frozenset({"embeddings"}),
        )

    @property
    def capabilities(self):
        return StepCapabilities(
            queries=frozenset({"vector_search"}),
        )
```

这个 step 不需要知道 Recipe 如何保存 vector store，也不需要直接持有 store 实例。执行器会根据 `stores.vector` 从 Recipe context 中解析组件。
