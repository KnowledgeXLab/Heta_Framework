# 当前运行支持的 Harbor 任务格式

运行器是 Harbor 0.23.0。本文由管线提供，说明当前支持的子集，不是科学资料。

作者的完整交付外层是 `/workspace/output/tasks/<task_id>/`：其下只有 task/（实际任务包）和 authoring/（来源与设计）。**solution/ 和 tests/ 必须放在 task/ 内，不能与 task/ 并列。authoring/ 不能放在 output/ 顶层。** 交接索引单独放 `/workspace/output/author-result.json`，package_path 指向 `tasks/<task_id>`。结束前逐一检查这些路径存在。

```text
task/
  instruction.md
  task.toml
  environment/Dockerfile
  environment/...公开代码与数据
  solution/solve.sh
  solution/...参考实现
  tests/test.sh
  tests/...测试实现与期望数据
```

最小 task.toml 示例（按实际运行需求调整超时和资源）：

```toml
schema_version = "1.4"

[metadata]
description = "任务描述"

[agent]
timeout_sec = 600

[verifier]
timeout_sec = 120

[environment]
build_timeout_sec = 600
cpus = 2
memory_mb = 2048
```

当前管线只接单步、Linux、CPU、Dockerfile 构建任务，不接 compose、预构建 docker_image、MCP、多步骤或远程环境。pipeline 配置还有限定资源预算；不要随意要求 GPU 或大规模内存。

Docker 构建上下文是 environment/。`COPY` 的来源相对于 environment/；不要 COPY solution/、tests/、authoring/。WORKDIR 自行明确，公开题意说明所有输入输出路径。Oracle 与后续正式 Solver 使用相同的初始环境定义。

Oracle 阶段，Harbor 注入 solution/ 到 /solution 并执行 /solution/solve.sh。文件不会自动处于当前工作目录，入口脚本应使用明确路径。Oracle 只能求解，不写 reward。

`solve.sh` 必须在公开题目指定的位置完成交付。若题目要求 `/app/process.py`，应在 Oracle 执行时创建该文件（例如从 `/solution/process.py` 复制实现），仅运行私有参考程序并生成结果不够。交付程序不能依赖 `/solution` 才能运行。只要求结果文件的任务则按题目交付结果，不额外要求程序。参考解和正式提交使用同一测试入口；答案不能预置于公开镜像。

验证阶段，Harbor 注入 tests/ 到 /tests 并运行 /tests/test.sh。评分脚本负责 `mkdir -p /logs/verifier`，写 `/logs/verifier/reward.txt`，文件只包含数值。当前合同：完全满足要求为 1，否则为 0。必须保证空提交、错误输出也能留下 reward=0。可在测试脚本里捕获测试框架正常的断言失败，然后正常结束评分；环境/解释器崩溃不应伪装成完整验证。

科学验证必须根据公开要求编写，不能仅为使参考解通过而放宽。提供一个合理的错误提交用于独立容器反例测试；空提交和错误提交均不得修改隐藏评分器。

authoring/ 与 task/ 并列，保存设计依据与必要的来源快照。这里不是 task 环境的一部分。来源记录应保留实际版本与引用位置，避免保留整个开发工具仓库或 .git。
