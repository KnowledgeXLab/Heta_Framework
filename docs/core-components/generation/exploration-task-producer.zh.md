# HetaGen Task Producer Exploration

`HetaGenTaskProducerExploration` 从论文、代码、数据或任务说明组成的混合资料包中探索可验证的问题，产出 Harbor 任务集合。输入无需预先抽取 KG 或提炼工作流模式；出题 Agent 读取资料、探索选题并编写候选，Builder 验证，Reviewer 独立审核。

该组件源自 Sci-SKT，实现 `TaskProducerProtocol[HetaGenTaskProducerExplorationConfig]`，使用同一个 Python 环境中的 Data Designer 和 Harbor。

生产流程为 Author → Builder → Reviewer，支持按审核反馈进行有界修订。Web Search、批量多资料包和独立 Solver 不在这个入口中。

## 安装与环境

使用 Python 3.12：

```bash
pip install "heta-framework[generation]"
```

`generation` 安装 Data Designer、Harbor 及配置依赖。运行还需要支持 Linux 容器的 Docker、Compose、Buildx、角色镜像和模型服务。

## 使用示例

准备一个本地 `source-package/` 目录，其中 `PACKAGE.md` 说明资料来源和希望生成的任务，其余文件可以是论文、代码或数据。配置 `.env` 中的 `OPENAI_API_KEY`、`SCI_PIPELINE_MODEL`，并准备兼容的角色镜像与 Docker 环境，即可运行下面的完整示例。

```python
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from heta_framework.common.stores import LocalObjectStore
from heta_framework.generation import (
    HetaGenTaskProducerExploration,
    HetaGenTaskProducerExplorationConfig,
    ObjectInput,
    TaskRequest,
)

async def main():
    objects = LocalObjectStore("heta-workspace/objects")
    try:
        # 1. 将资料包放入 Object Store。
        source = Path("source-package")
        for file in sorted(source.rglob("*")):
            if file.is_file():
                key = f"sources/package-001/{file.relative_to(source).as_posix()}"
                await objects.put(key, file.read_bytes())

        # 2. 配置并运行出题，每次使用新的输出位置。
        producer = HetaGenTaskProducerExploration(
            HetaGenTaskProducerExplorationConfig(tasks_per_package=2)
        )
        result = await producer.produce(
            TaskRequest(
                inputs={"materials": ObjectInput(prefix="sources/package-001/")},
                output_prefix=f"tasks/{uuid4().hex}/",
            ),
            object_store=objects,
        )

        # 3. 读取实际交付的任务清单与生产报告。
        tasks = json.loads(await objects.get(result.tasks_manifest_key))
        report = json.loads(await objects.get(result.report_key))
        print(result.status, len(tasks["tasks"]))
        print(result.tasks_manifest_key, result.report_key)
        print(report["native_results"])
    finally:
        await objects.aclose()

asyncio.run(main())
```

示例默认执行真实生产，会调用模型。只检查交接流程时，将配置改为 `HetaGenTaskProducerExplorationConfig(mode="mock", tasks_per_package=2)`，不需要模型或 Docker；mock 不交付正式科学任务。

已有资料在 Object Store 中时，跳过上传循环，直接填写对应 `materials` 前缀。`tasks_per_package=2` 表示请求生成两个任务，实际交付数量以清单为准；清单中每个任务的 `files` 记录文件对象 key，可继续通过同一个 store 下载。

## 输入要求

唯一必填输入 `materials` 指向一个包含 `PACKAGE.md` 的资料包。文档内描述材料和来源，其余文件按资料包目录布局保留。

选中恰好一个 `PACKAGE.md`，全部文件在同一个资料包目录内。输出前缀必须为空，不能覆盖输入或已发布结果。

## 配置

公开配置类位于 `heta_framework.generation.exploration.config`，是 frozen dataclass，创建时校验参数；producer 的 `config` 属性只读。默认采用 `harbor` 模式。

| 配置 | 用途 | 默认值 |
| --- | --- | --- |
| `tasks_per_package` | 每包任务数，1–100 | 1 |
| `max_concurrency` | NeMo 行并发 | 1 |
| `max_revisions` | 审核反馈后的修订上限，0–10 | 1 |
| `max_role_attempts` | 单角色交付修复次数，1–3 | 2 |
| `generation_guidance`、`difficulty_guidance` | 自由文本生成要求与挑战引导 | 空字符串 |
| `role_timeout_sec`、`validation_timeout_sec` | 角色与验证时限 | 1800 秒 |
| `environment_cpus`、`environment_memory_mb` | 执行资源 | 2 CPU / 4096 MiB |
| `role_image`、`codex_version`、`docker_builder` | 角色执行环境 | 见下方镜像要求 |
| `job_overhead_timeout_sec`、`worker_stop_grace_sec` | 启动/回收预算与停止宽限 | 1200 / 30 秒 |
| `mode`、`mock_revise_once` | 正式生产或模拟交接测试 | `harbor` / `False` |
| `env_file`、`workspace` | 凭据文件及临时工作区父目录 | `.env` / 系统临时目录 |

模型和凭据通过指定文件或环境变量提供，例如 `OPENAI_API_KEY`、`SCI_PIPELINE_MODEL`、可选 `OPENAI_BASE_URL`。

`workspace` 可指定临时工作区的父目录。输入下载、Data Designer 执行及产物上传在 Heta 所在环境中完成；Harbor worker 使用同一 Python 解释器的子进程执行角色。

## 输出与验证

`result.tasks_manifest_key` 指向任务清单，`result.report_key` 指向生产报告。正式任务必须满足原生 accepted/harbor/validated 条件并通过 Harbor 配置和必要文件检查。

任务文件记录对象 key、SHA-256 和权限；验证/审核证据单独归档。shell 入口在清单中记为可执行权限，消费者下载时需按记录恢复。原生作业日志不自动全部上传。

该入口不提供跨运行恢复接口。

## 角色镜像

`role_image` 指向部署环境中已安装的镜像，默认标签为 `sci-pipeline/role:codex-0.154.0`。该标签不是框架提供的公共镜像下载地址；安装 Python 额外依赖不会创建角色镜像。

镜像必须提供 `codex` 命令和 `node` 用户，支持在 `/workspace` 中运行角色任务。`codex --version` 必须与配置中的 `codex_version` 一致，默认要求输出 `codex-cli 0.154.0`。使用自有镜像时显式设置 `role_image` 与 `codex_version`。

## 模拟运行

`mode="mock"` 使用 Data Designer 执行模拟角色交接，不调用模型或 Docker。可配合 `mock_revise_once=True` 检查修订流程。模拟运行不发布任务，任务清单为空，交付状态为 `failed`；模拟诊断保存在报告中。
