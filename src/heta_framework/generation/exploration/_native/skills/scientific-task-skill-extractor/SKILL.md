---
name: scientific-task-skill-extractor
description: 从已通过构建验证和独立审核的科学任务中，提炼一份对应任务且可用于同类问题的 Agent Skill，并记录来源依据。
---

# 任务对应的 Skill 提炼

你的输入是已验收任务及其来源、参考解、测试、构建记录和审核结论。你的产物是一份供其他 Agent 使用的通用操作能力。任务与 Skill 一一关联；不生成 Recipe，不重新出题，不修改原任务。

## 提炼原则

- 先理解公开题意、方法选择和验收依据，再按需读取参考解、科学资料与执行记录。区分来源给出的方法、参考实现采用的一种做法和测试覆盖的有限性质。
- 提炼方法选择依据、适用条件、输入前提、关键工作流、科学约束、结果检查及必要的失败恢复。写清何时该用，以及何时不适用。
- 将任务专属路径、样本、最优参数与具体答案改为输入参数或使用者需要判断的条件。保留有科学依据的常数和方法约束，不机械删除所有数值。
- 不复制任务评分器、隐藏答案或整份参考解。确有独立复用价值的算法可整理成参数化辅助脚本；说明依赖与调用方式，仅在实际执行后声称脚本已验证。
- 不把简单的打开文件、复制目录等操作拆成独立科学技能。一份 Skill 可以覆盖解决该类任务所需的连贯流程。
- 长篇公式和方法说明按需放 references/；稳定辅助程序放 scripts/；模板放 assets/。SKILL.md 说明资源用途与读取时机，引用仅使用 Skill 包内相对路径或公开来源，不依赖 /inputs、原任务工作目录或生产侧审计文件。
- 当前没有正式 Solver 轨迹。不能编造 Agent 的失败经验、成功率、迁移测试或对其他任务的效果。
- 若任务无法支持有实际复用价值的 Skill，输出 not_applicable 并说明原因，不为满足数量要求写空泛内容。

## 输出合同

所有交付写入 /workspace/output：

```text
skill-draft.json
skills/<skill-name>/SKILL.md
skills/<skill-name>/references/   # 按需
skills/<skill-name>/scripts/      # 按需
skills/<skill-name>/assets/       # 按需
```

SKILL.md 使用 YAML frontmatter，至少含 name、description；name 为与目录一致的小写字母、数字和连字符名称，最多 64 字符。description 描述能力和使用时机。正文不需要固定标题，但必须足以指导同类新任务。

skill-draft.json：

```json
{
  "task_id": "本次分配的 ID",
  "candidate_revision": "本次分配的 hash",
  "status": "ready",
  "skill_path": "skills/<skill-name>",
  "reason": "",
  "evidence": ["candidate/task/solution/solve.sh: 核心方法实现", "source/PACKAGE.md: 来源入口"]
}
```

证据应指出具体文件或公开来源及其支持的关键方法；此记录属于生产侧，不要求 Skill 使用者访问原任务。status=not_applicable 时 skill_path=null，reason 必须说明缺少什么复用依据，可不生成 skills/。
