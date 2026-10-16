# HetaGen Task Producer Graph

`HetaGenTaskProducerGraph` 从已有知识库中检索文献，复用配对的全文、实体、关系和来源证据，通过 Omega 的原生 Data Designer workflow 生产 Harbor 任务。

也可以直接传入 Object Store 中的正文和 KG，运行时不需要知识库。两种入口使用同一套生产流程。

## 安装与环境

使用 Python 3.12，在 Heta 所在环境安装：

```bash
pip install "heta-framework[generation]"
```

该额外依赖包含 Data Designer 和 Harbor。知识库模式还需安装所使用的知识库及存储组件依赖。

运行需要 Docker、Compose、Buildx 和 OpenAI-compatible Chat Completions 服务。在 `.env` 设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`；已导出的环境变量优先。`api_base_url` 可覆盖服务地址，模型名称放在配置类中。

## 从知识库生成任务

将示例中的 `your-model` 替换为模型服务提供的标识；可选推理参数按服务支持情况设置。

`kb` 是已经建好的 Heta `KnowledgeBase`，`objects` 是保存本次输入快照及产物的 `ObjectStoreProtocol`，可以与知识库的存储不同：

```python
import json
from uuid import uuid4

from heta_framework.generation import (
    HetaGenTaskProducerGraph,
    HetaGenTaskProducerGraphConfig,
    TaskRequest,
)

async def generate_tasks(kb, objects):
    producer = HetaGenTaskProducerGraph(
        HetaGenTaskProducerGraphConfig(
            query_mode="heta_graph_search",
            top_k=20,
            max_documents=2,
            model="your-model",
            reasoning_effort="low",
            max_concurrency=2,
            generation_guidance="侧重数值方法实现、边界条件处理与隐藏测试泛化",
        )
    )
    result = await producer.produce(
        TaskRequest(
            inputs={"query": "热传导的数值计算方法"},
            output_prefix=f"tasks/graph/{uuid4().hex}/",
        ),
        knowledge_base=kb,
        object_store=objects,
    )
    manifest = json.loads(await objects.get(result.tasks_manifest_key))
    report = json.loads(await objects.get(result.report_key))
    print(result.status, len(manifest["tasks"]))
    print(report["inputs_manifest_key"], report["native"]["stages"])
    return result
```

在异步应用内调用 `await generate_tasks(kb, objects)`。这个例子会调用模型及 Docker；`max_documents` 是选取文档上限，不保证每篇都产出任务。`query` 选择材料，`generation_guidance` 描述希望考察的能力，二者分别保存。

只检查取材和输入转换，可调用 `prepare_inputs()`，无需任务生成模型凭据：

```python
producer = HetaGenTaskProducerGraph(
    HetaGenTaskProducerGraphConfig(query_mode="heta_graph_search")
)
prepared = await producer.prepare_inputs(
    TaskRequest(
        inputs={"query": "热传导的数值计算方法"},
        output_prefix="prepared/heat-transfer/run-001/",
    ),
    knowledge_base=kb,
    object_store=objects,
)
print(prepared.manifest_key, prepared.document_count)
```

知识库 query 本身是否调用模型，取决于所选查询引擎；准备器请求关闭答案合成。每次使用新的输出前缀，已有内容不会覆盖。随后可按下一节把快照中的 KG/text 输入到一次新的生产运行。

## 从对象存储生成任务

```python
from heta_framework.generation import ObjectInput

producer = HetaGenTaskProducerGraph(
    HetaGenTaskProducerGraphConfig(
        input_mode="objects",
        model="your-model",
        reasoning_effort="low",
        max_documents=2,
    )
)
result = await producer.produce(
    TaskRequest(
        inputs={
            "kg": ObjectInput(prefix="prepared/heat-transfer/run-001/inputs/kg/"),
            "text": ObjectInput(prefix="prepared/heat-transfer/run-001/inputs/text/"),
        },
        output_prefix="tasks/graph/heat-transfer/run-001/",
    ),
    object_store=objects,
)
```

也可用 `ObjectInput(keys=(... ,))` 指定已有材料。`.json` 与 `.txt` 按相同文件名主体精确配对，如 `kg/paper-01.json` 和 `text/paper-01.txt`；不做标题模糊匹配。缺少配对、同槽位重名、超出 `max_documents` 都会报错。

## 知识库输入要求

准备器通过查询结果及引用的 `document_id` / `document_ids`，或原始对象 keys 识别文档，再读取 KB 构建记录中的以下产物：

| 配置字段 | 默认 artifact | 内容 |
| --- | --- | --- |
| `parsed_documents_artifact` | `parsed_document_keys` | `ParsedDocument` 全文及源对象信息 |
| `entities_artifact` | `entity_keys` | `ExtractedEntity` 原始实体与证据 chunk IDs |
| `relations_artifact` | `relation_keys` | `ExtractedRelation` 原始端点及证据 chunk IDs |
| `chunks_artifact` | `chunk_keys` | `ParsedChunk` 字面证据 |

这与 `ParseDocuments`、`HetaGraphProcedure` 默认产物衔接。其他 procedure 可通过上述字段绑定自己的 artifact 名称。每项产物的来源存储按其生产步骤的 object-store 声明解析；无生产步骤的初始产物使用 KB 默认 object store。存储声明不唯一时明确拒绝。

准备器读取这些 artifact 的对象列表再按文档筛选；大型知识库准备成本与列表规模有关，`max_documents` 不限制扫描对象数。检索命中的片段用于选文档，实际生产使用这些文档的完整配对材料。查询回答不会直接当作 KG，也不从任意图索引反向恢复论文。

实体默认映射为 `TASK → Tasks`、`METHOD → Methods`、`DATASET → Datasets`、`METRIC → Metrics`、`BASELINE → Baselines`、`IMPLEMENTATION → Implementation`、`THEOREM → Theorems`、`DOMAIN_TERM → Domain_Terms`，类型匹配忽略大小写。未知类型不会自动归为领域术语。已有本体类型可由用户显式声明，例如 `entity_type_mapping={"NumericalMethod": "Methods"}`。

同篇的重复实体提及保留各自 ID、描述、属性和证据；不同语义类型在原生规范化后同名则拒绝。关系端点必须存在于该篇实体中，证据 chunk 必须属于同篇文档并可在保存的全文中唯一定位。证据保存原文行号及逐字引文，不能用模型总结替代。跨文档同名实体不合并。

## 正文与图谱格式

KG 至少包含 `A_Meta_Factual_Entities.Paper.title`、`B_Textually_Mentioned_Entities` 和非空 `E_Relations`。实体组使用 Omega 原生名称和名称字段，每个实体与关系都需非空 `evidence`，其中每条包含一基的 `[起始行, 结束行]` `source_lines` 及与配对全文完全一致的 `quote`。关系包含 `head`、`tail`、`relation`，端点须存在于 B 组。对象输入不支持仅由 C 组推断概念组成的端点。

正文与 KG 的格式示例（仅展示字段；实际输入仍需满足结构筛选阈值）：

`paper-01.txt`，保留行末换行：

```text
Method A uses Method B.
```

`paper-01.json`:

```json
{
  "A_Meta_Factual_Entities": {
    "Paper": {
      "title": "Method comparison"
    }
  },
  "B_Textually_Mentioned_Entities": {
    "Methods": [
      {
        "name": "Method A",
        "evidence": [
          {
            "source_lines": [
              1,
              1
            ],
            "quote": "Method A uses Method B.\n"
          }
        ]
      },
      {
        "name": "Method B",
        "evidence": [
          {
            "source_lines": [
              1,
              1
            ],
            "quote": "Method A uses Method B.\n"
          }
        ]
      }
    ]
  },
  "E_Relations": [
    {
      "head": "Method A",
      "tail": "Method B",
      "relation": "USES",
      "evidence": [
        {
          "source_lines": [
            1,
            1
          ],
          "quote": "Method A uses Method B.\n"
        }
      ]
    }
  ]
}
```

准备层校验材料完整性，生产流程随后执行结构筛选与模型评审。结构筛选的默认要求为：默认最大连通分量至少 20 个节点、20 条边、3 种关系，节点比例至少 0.35，并有 B 类锚点。

## 配置与执行

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `input_mode` | `knowledge_base` | `knowledge_base` 或 `objects` |
| `query_mode` | `heta_graph_search` | 知识库查询模式，须与已有索引匹配 |
| `top_k` | 20 | 查询返回数量 |
| `max_documents` | 3 | 单次生产选取的文档上限 |
| `generation_guidance` | 空字符串 | 出题要求 |
| `model` | `bailian/glm-5.3` | 生成模型标识 |
| `review_model` | `None` | 评审模型；默认与 `model` 相同 |
| `api_base_url` | `None` | 模型服务地址；未设置时读取环境配置 |
| `reasoning_effort` | `None` | 可选推理参数，需模型服务支持 |
| `max_tokens` | 32000 | 单次模型输出预算 |
| `request_timeout_sec` | 600 | 单次模型请求时限，秒 |
| `max_concurrency` | 1 | Data Designer 并发上限 |
| `settings_file` | `None` | 外置生产策略 JSON |
| `env_file` | `.env` | 模型凭据配置文件 |
| `workspace` | `None` | 临时工作目录的父目录 |

`config` 是只读配置；结构筛选与评审门槛、候选数量、难度档位和修复轮数通过 `settings_file` 指向外置 JSON。例如，将以下内容保存为 `graph-settings.json`，并设置 `settings_file=Path("graph-settings.json")`（`Path` 从 `pathlib` 导入）：

```json
{
  "difficulty": {"tiers": ["medium", "complex"]},
  "repair": {"validation_attempts": 3},
  "validation": {"oracle_repeats": 2, "harbor_env": "docker"}
}
```

未提供的设置使用原生默认值。`generation_guidance` 由 Heta 配置传入候选生成、写题、实现及实现复核，作为出题要求，与来源证据分别传递。`model` 用于强模型角色，`review_model` 可单独选择证据准入评审模型；未设置时共用 `model`。`max_tokens=32000`、`request_timeout_sec=600` 是每次调用预算。`max_concurrency` 通过 Data Designer 行组、非推理 worker 和模型并发上限共同控制。

原生八阶段为 `screen → review → candidates → candidate_review → spec → implementation → validate_static → validate_runtime`。各阶段按配置执行内部修订。不同文档的任务目录按输入身份隔离，即使生成相同 slug 也不会覆盖。工作区执行完回收；输入快照、阶段状态、原生 checkpoint parquet、拒绝清单及验证证据保存到 Object Store。

## 输出与验证

`TaskResult` 返回 `tasks.json` 和 `report.json` 的 key。报告包含 `inputs_manifest_key`、`designer_evidence_key`、阶段状态、每阶段行数、拒绝原因和模型步骤失败；清单中每个任务包含文件 key、SHA-256、权限和验证报告 key。下载时恢复文件相对路径和权限。

`report["native"]["complete"]` 表示工作流阶段是否执行完成；`records_complete` 表示已完成阶段是否保留了全部请求行。`stage_record_losses` 记录缺失行所在阶段、请求数、实际输出数及可识别的文档和难度档位。存在缺失行时，运行按失败或部分成功处理，已通过验证的任务仍可交付。

发布同时要求原生 workflow 完成、`validate_runtime` 完成行的静态/数据物化/确定性/Harbor 门控为真，并与独立的 `validation_report.json` 一致。默认 Oracle 两次均为 1、Nop 为 0，且无试验异常。目录存在或单个 ready 字段不构成通过证据。

正常无命中、结构拒绝和语义拒绝可得到 `succeeded` 与零任务；模型步骤错误、执行失败或最终验证失败时，已交付部分任务则为 `partial`，否则为 `failed`。输入合同错误直接抛出 `GraphInputError`，不发布任务清单。连接对象和凭据不写入配置快照；原生验证进程不继承模型凭据。

输入快照记录 KB build run ID、实际读取对象的哈希及本次 query 响应；执行前重新校验快照哈希。它固定的是本次读取并保存的字节，不提供跨存储事务或同前缀的并发写锁。一次运行独占一个新的输出前缀。

## 使用边界

生产验证检查格式、数据生成的确定性和参考解/空解行为，不提供科学正确性或任务难度的自动保证。生成任务用于评测或训练前，应检查题意、数据、评分器与起始文件的一致性。

[查看任务输入与输出示例](hopf-task-example.md)。
