# HetaGen Task Producer Derivation

`HetaGenTaskProducerDerivation` 从已有 Harbor 任务派生新的独立任务。用户提供种子任务和配置；组件提炼可迁移模式，生成新的题目、数据、参考解与测试，并交付经过验证的 Harbor 任务包。

该组件实现 `TaskProducerProtocol[HetaGenTaskProducerDerivationConfig]`。它复用 TB Factory 的候选校验和物化函数，使用 Data Designer 原生模型列管理模式提炼、候选生成及来源相似性审核。

## 安装与环境

使用 Python 3.12，在同一环境安装：

```bash
pip install "heta-framework[generation]"
```

真实生产需要 Docker、Compose、Buildx，以及支持 Data Designer 结构化输出提示的 OpenAI-compatible Chat Completions 服务。模型直接生成文件内容。

在 `.env` 设置 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。模型通过配置类的 `model` 指定；`reasoning_effort` 等推理参数需与模型服务支持的参数一致。`api_base_url` 可覆盖地址；已导出的环境变量优先于 `.env`。不要将凭据放入种子任务。

## 使用示例

将示例中的 `your-model` 替换为模型服务提供的标识；可选推理参数按服务支持情况设置。

准备本地 `seed-tasks/`，其中每个子目录是一份已有 Harbor 任务：

```text
seed-tasks/
  example/
    instruction.md
    task.toml
    environment/
    solution/
    tests/
```

以下示例包含上传、运行和读取结果：

```python
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from heta_framework.common.stores import LocalObjectStore
from heta_framework.generation import (
    HetaGenTaskProducerDerivation,
    HetaGenTaskProducerDerivationConfig,
    ObjectInput,
    TaskRequest,
)

async def main():
    objects = LocalObjectStore("heta-workspace/objects")
    try:
        source = Path("seed-tasks")
        for file in sorted(source.rglob("*")):
            if file.is_file():
                await objects.put(
                    f"sources/harbor-tasks/{file.relative_to(source).as_posix()}",
                    file.read_bytes(),
                )

        producer = HetaGenTaskProducerDerivation(
            HetaGenTaskProducerDerivationConfig(
                model="your-model",
                reasoning_effort="low",
                tasks_per_seed=4,
                max_concurrency=4,
                generation_guidance="保留核心能力，改变场景和合成数据",
            )
        )
        result = await producer.produce(
            TaskRequest(
                inputs={"seed_tasks": ObjectInput(prefix="sources/harbor-tasks/")},
                output_prefix=f"tasks/derivation/{uuid4().hex}/",
            ),
            object_store=objects,
        )
        manifest = json.loads(await objects.get(result.tasks_manifest_key))
        report = json.loads(await objects.get(result.report_key))
        print(result.status, len(manifest["tasks"]))
        print(report["native_results"])
        print(report["patterns_key"])
    finally:
        await objects.aclose()

asyncio.run(main())
```

示例会调用模型及 Docker。只验证交接流程时将配置改为 `mode="mock"`；模拟执行使用真实 Data Designer 调度，但不会调用模型、Docker，也不会发布正式科学任务。模拟成功时仍返回空任务清单，报告中的 `simulation_completed` 为 true。

已有种子在 Object Store 时可跳过上传。`tasks_per_seed=4` 是每个种子的请求数量；实际交付以清单为准。每个任务的 `files` 包含对象 key、SHA-256 和文件权限，下载后按记录恢复可执行权限。

## 输入要求

必填槽位 `seed_tasks` 可选择一份或多份任务。以 `task.toml` 所在目录识别任务根，每个根必须包含 `instruction.md`；不支持嵌套任务根或夹杂无关文件。输出位置必须为空。

模型读取公开题目和 `environment/` 文件。二进制材料只提供文件名、大小和摘要；公开文本超过 `max_public_bytes` 时拒绝输入。种子的 `task.toml`、`solution/`、`tests/` 不送入模型提示。这个边界不能替代用户对公开环境是否包含答案的检查。

## 生产流程

第一阶段，每个种子一行：

```text
公开任务 → task_pattern → pattern_result
```

`task_pattern` 是结构化模型列，包含工作流程、能力、交付物、变化空间和公开来源依据。`source_evidence` 每项分别记录 `file_path` 与 `finding`，路径必须精确对应公开文件，说明独立存放。`pattern_result` 检查来源路径是否存在。每个通过检查的模式按 `tasks_per_seed` 展开为独立变体，避免重复提炼。

第二阶段，每个派生请求一行：

```text
模式 + variant → candidate → novelty
                         ↓       ↓
                      candidate_check
                            ↓
                       task_package
                            ↓
                       task_result
```

`candidate` 和 `novelty` 使用原生结构化模型列。其余使用 Custom Column，执行确定性打包、TB 校验与物化及 Harbor。请求标识和变体进入提示词，目录按请求隔离。跨任务并发由 Data Designer 管理；每个任务的三项验证依次执行。

模型负责题目、公开数据、参考解、测试和输出路径。内部候选中，`public_files` 声明公开数据的相对路径与原始文本，`files` 仅声明 `solution/` 和 `tests/` 中的代码。框架负责稳定标识、`task.toml` 和两个 Dockerfile。

同一份公开数据原样物化到 `environment/inputs/` 和 `tests/inputs/`，在两边容器中均位于 `/workspace/inputs/`。独立 verifier 从自己的镜像读取可信输入，主容器修改输入不会改变判卷依据。公开数据与做题者提交产物分开保存。参考解与隐藏测试不会进入做题环境。

输出路径必须位于 `/workspace/` 或 `/app/`，并避开 `/workspace/inputs/`。参考解入口是 `solution/solve.sh`，判卷入口是 `tests/test.sh`。代码和数据均按声明保存，模型代码仍需通过实际 Harbor 验证。

来源相似性审核检查能力保留和独立性；同时拒绝直接复用实质性公开文件，并在发布时拒绝批次内相同的公开任务内容。这些检查不保证语义去重、难度校准或科研质量。

## 配置

配置是独立 frozen dataclass，producer 的 `config` 属性只读。

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `tasks_per_seed` | 1 | 每个种子派生数量，1–100 |
| `max_concurrency` | 1 | 模型请求及非模型行任务并发上限 |
| `model` | `bailian/glm-5.3` | Chat Completions 模型标识 |
| `generation_guidance` | 空文本 | 生成偏好 |
| `temperature`、`reasoning_effort` | 0.2、None | 可选模型参数；None 表示不发送 |
| `max_tokens` | 32000 | 单次模型输出预算 |
| `request_timeout_sec` | 300 | 单次模型请求时限 |
| `max_model_restarts` | 1 | Data Designer 模型会话重启上限，0–3 |
| `validation_timeout_sec` | 1800 | 单次 Harbor 验证预算 |
| `environment_cpus`、`environment_memory_mb` | 2、4096 | 主环境与独立 verifier 资源上限 |
| `environment_image` | `python:3.12-slim` | 两边共用的基础镜像；依赖预装于该镜像 |
| `max_public_bytes` | 128000 | 每个种子的公开文本总字节上限 |
| `env_file` | `.env` | 可选凭据文件 |
| `workspace` | None | 临时工作区父目录 |

`docker_builder`、`job_overhead_timeout_sec` 和 `worker_stop_grace_sec` 调整本地 Docker 作业执行与清理。临时工作区在返回后回收；该入口不提供跨调用恢复 API。

## 输出与验证

只有真实 Harbor 验证通过的任务进入 `tasks.json`：Oracle 得分 1，空提交得分 0，代表性错误提交得分 0，且三个 trial 均无执行异常。错误提交脚本崩溃不算通过。验证容器不会获得生成模型凭据。

独立 verifier 在整个运行期间保持无网络，Docker 适配器通过 `network_mode=none` 执行该约束，不依赖宿主机防火墙，也不支持动态切换网络策略。

`patterns.json` 保留提炼结果。`report.json` 记录阶段数量、请求与交付数量、配置、候选检查、来源相似性审核和各任务状态。缺失的生成行也记录为失败。验证证据与任务文件分开存储。

失败检查还记录 `failure_reason` 和有长度上限的执行日志：`reference_solution_rejected` 表示参考解未通过，`invalid_submission_accepted` 表示负例被误判为正确，`execution_error` 表示执行异常。日志保留具体缺失文件或数值不一致信息，便于区分打包错误与算法错误。

返回 `succeeded`、`partial` 或 `failed`；部分成功不会发布失败候选。该组件不执行内容修订循环、独立 Solver 求解或难度校准。种子提炼失败、候选非法、来源相似性审核拒绝、验证失败和重复内容均可能减少交付数量。

与 [Exploration](exploration-task-producer.zh.md) 的区别：Exploration 从混合原始资料探索出题；Derivation 从已有任务提炼能力模式，再派生新任务。公共协议与 Object Store 交接方式相同。
