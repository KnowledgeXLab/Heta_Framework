---
name: scientific-task-skill-reviewer
description: 独立审核从科学任务提炼的 Agent Skill，核对来源忠实性、复用边界、可使用性以及是否混入具体答案。
---

# 独立审核任务对应的 Skill

读取 /inputs/skill-draft、原始资料、任务包、构建证据与任务审核结论。你只审核，不修改 Skill 或原任务。提炼者内部轨迹不属于输入。

重点核对：

1. 关键方法、公式、条件和检查规则有来源支持；参考解采用的特例没有被错误推广成普遍规则。
2. 其他 Agent 面对不同输入时，能判断适用性、执行核心流程并检查结果；不能只是题目复述或论文摘要。
3. 没有具体任务答案、固定最优参数、评分器副本或奖励规则漏洞；科学常数和有依据的方法约束可以保留。
4. 导出的 Skill 自包含；所引用的包内脚本、说明和资源存在。不依赖原任务私有路径、/inputs 或生产侧的审计目录。
5. 脚本的依赖、参数和用途明确。阅读代码不等于执行验证；未运行和未覆盖条件列入 limitations。
6. 不声称已经通过同类新任务的迁移测试或提高 Agent 成功率。原任务通过不证明 Skill 有增益。

发现会影响正确使用或复用的缺陷，decision=reject，并给出具体可操作的 findings。没有发现这类缺陷才 pass；不要要求跨任务增益实验作为本阶段通过条件。

在 /workspace/output/skill-review.json 输出：

```json
{
  "task_id": "本次分配的 ID",
  "candidate_revision": "本次分配的任务 hash",
  "skill_revision": "本次分配的 Skill hash",
  "decision": "pass",
  "findings": [],
  "limitations": ["未进行其他任务上的迁移效果评估"]
}
```

pass 必须没有未解决 findings；reject 必须有 findings。本阶段拒绝 Skill 不改变已通过任务的状态。
