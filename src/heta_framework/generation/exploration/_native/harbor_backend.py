"""Production and optional skill roles use Harbor's existing Codex adapter."""

import shutil
from pathlib import Path

from harbor.models.task.config import TaskConfig

from .artifacts import candidate_from_output, copy_tree, read_json, safe_path, write_json
from .config import ApiSettings, RunConfig
from .contracts import BuildResult, ReviewResult, TaskRequest
from .harbor_runner import HarborRunner


class HarborBackend:
    mode = "harbor"

    def __init__(self, config: RunConfig, api: ApiSettings, skills_root: Path):
        self.config = config
        self.runner = HarborRunner(config, api)
        self.skills_root = skills_root

    def _role(
        self, role: str, output: Path, inputs: dict[str, Path], instructions: str, runtime: Path
    ) -> None:
        envelope = runtime / "role-task"
        environment = envelope / "environment"
        environment.mkdir(parents=True)
        for name, source in inputs.items():
            destination = environment / "inputs" / name
            if source.is_dir():
                copy_tree(source, destination)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        reference = Path(__file__).parent / "resources" / "harbor-task-format.md"
        shutil.copy2(reference, environment / "inputs" / "harbor-task-format.md")
        (environment / "Dockerfile").write_text(
            f"FROM {self.config.role_image}\n"
            "COPY inputs /inputs\n"
            "RUN chmod -R a-w /inputs && mkdir -p /workspace/output && chown -R node:node /workspace\n"
            "WORKDIR /workspace\n"
        )
        (envelope / "task.toml").write_text(
            'schema_version = "1.4"\n[agent]\nuser = "node"\n'
            f"timeout_sec = {self.config.role_timeout_sec}\n"
            f"[environment]\ncpus = {self.config.environment_cpus}\nmemory_mb = {self.config.environment_memory_mb}\n"
        )
        skill = (self.skills_root / f"scientific-task-{role}" / "SKILL.md").read_text()
        (envelope / "instruction.md").write_text(
            skill
            + "\n\n# 本次运行合同\n\n"
            + instructions
            + "\n\n将所有交付文件写到 /workspace/output。输入在 /inputs，只读。"
            "不要操作凭据、进程认证目录、评分日志或宿主机。"
            "本角色生产作业没有 Verifier，不要编造环境运行成功。"
            "联网资料可通过 curl/Python 获取；未提供搜索工具。完成文件后直接结束。\n"
            "Harbor 格式说明已提供在 /inputs/harbor-task-format.md；无需克隆 Harbor 仓库。"
            "产物仅保留必要来源快照，不携带完整工具仓库、.git、缓存或符号链接。\n"
        )
        result = self.runner.run(
            envelope,
            runtime / "execution",
            self.runner.codex(),
            verify=False,
            timeout=self.config.role_timeout_sec,
            artifacts=True,
            credentials=True,
        )
        self.runner.collect(result, output)

    def _deliver(self, role, output, inputs, instructions, validate):
        """Retry incomplete handoffs, not infrastructure errors or scientific decisions."""
        current_inputs = dict(inputs)
        correction = ""
        for attempt in range(self.config.max_role_attempts):
            runtime = output.parent / f"{role}-runtime" / f"attempt-{attempt:03d}"
            temporary = runtime / "output"
            try:
                self._role(role, temporary, current_inputs, instructions + correction, runtime)
                validate(temporary)
            except (ValueError, FileNotFoundError, KeyError) as exc:
                write_json(
                    runtime / "contract-error.json",
                    {"error_type": type(exc).__name__, "error": str(exc)},
                )
                if attempt + 1 == self.config.max_role_attempts:
                    raise ValueError(
                        f"{role} did not meet the output contract after {attempt + 1} attempts; see {runtime}"
                    ) from exc
                if temporary.is_dir():
                    current_inputs["partial-output"] = temporary
                correction = (
                    "\n上次交付格式检查失败，错误："
                    + str(exc)
                    + "。若存在 /inputs/partial-output，基于已有文件继续完善，避免重新获取已有材料。"
                    "新交付仍必须完整写入 /workspace/output，不直接结束为一段文字。"
                )
                continue
            if output.exists():
                if any(output.iterdir()):
                    raise ValueError("Final role output directory is not empty")
                output.rmdir()
            copy_tree(temporary, output)
            return

    def author(
        self,
        request: TaskRequest,
        source: Path,
        output: Path,
        revision: int,
        previous: Path | None,
        feedback: dict | None,
    ) -> None:
        inputs = {"source": source}
        if previous:
            inputs["previous"] = previous
            # Candidate is revision/author/tasks/<id>; return complete evidence for repairs.
            previous_revision = previous.parents[2]
            for role in ("builder", "reviewer"):
                evidence = previous_revision / role
                if evidence.is_dir():
                    inputs[f"previous-{role}"] = evidence
        if feedback:
            path = output.parent / "author-feedback.json"
            write_json(path, feedback)
            inputs["feedback.json"] = path

        def validate(path):
            package = candidate_from_output(path, request.task_id)
            TaskConfig.model_validate_toml((package / "task/task.toml").read_text())

        self._deliver(
            "author",
            output,
            inputs,
            f"只生成一个任务，task_id={request.task_id}，revision={revision}，variant={request.variant}。"
            f"最终 Harbor 任务根目录必须是 /workspace/output/tasks/{request.task_id}/task/，"
            "instruction.md、task.toml、environment/、solution/、tests/ 都必须在这个 task/ 内。"
            f"设计说明和来源放 /workspace/output/tasks/{request.task_id}/authoring/。"
            "交接索引必须是 /workspace/output/author-result.json。"
            "读取 /inputs/source/PACKAGE.md 及其材料。若有 previous 和 feedback.json，按反馈修订。"
            f"\n额外生成要求：{self.config.generation_guidance or '无。'}\n"
            f"难度引导（difficulty_guidance）：\n{self.config.difficulty_guidance or '未指定；按来源设计可验证的问题，不设难度配额。'}\n"
            "以上难度引导是出题偏好，不是已校准难度或通过率承诺，不改变正确性与交付合同。"
            "在 authoring/DESIGN.md 中说明实际选择的挑战、来源依据、验收方式及与目标的差距。"
            f"任务容器上限为 {self.config.environment_cpus} CPU、"
            f"{self.config.environment_memory_mb} MiB 内存，当前后端不支持 GPU；"
            f"单次自测预算 {self.config.validation_timeout_sec} 秒。"
            "应在预算内保留核心科学难点，不能自行增加资源或照搬其他 benchmark 的时限。\n"
            "使用 Linux 单容器 Dockerfile 格式，Harbor 0.23.0，task.toml schema_version=1.4。"
            "不要使用 compose、远程执行环境、多步骤任务、额外 MCP 或自定义 agent 配置。"
            "参考解入口 solution/solve.sh，测试入口 tests/test.sh，标量 reward=1 表示完全通过，"
            "否则 reward=0。测试应始终写 /logs/verifier/reward.txt；错误提交也正常结束评分，"
            "测试框架失败信息可以保存，但不要因正常拒绝错误答案而漏写 reward。"
            "本轮需验证空提交会失败，请选择需要产生或修正明确交付物的问题。"
            "solution/solve.sh 必须在公开题目指定路径创建或修改全部交付物；"
            "Harbor 不会自动把 /solution 中的参考程序安装到工作目录。"
            "按 skill 的参考解交付示例检查题目、solve.sh 与 tests 的路径一致性。"
            "环境不得包含参考解、隐藏测试或已完成答案。使用固定版本依赖。"
            "输出按 skill 的目录合同，并写 author-result.json。",
            validate,
        )

    def build(
        self, request: TaskRequest, candidate: Path, output: Path, candidate_revision: str
    ) -> BuildResult:
        def validate(path):
            raw = read_json(path / "builder-result.json")
            if not isinstance(raw, dict):
                raise ValueError("builder-result.json must be an object")
            issues = raw.get("issues", [])
            if not isinstance(issues, list) or any(not isinstance(issue, dict) for issue in issues):
                raise ValueError(
                    "Builder issues must be a list of objects with severity, target_role and finding"
                )
            if (
                raw.get("candidate_revision") != candidate_revision
                or raw.get("task_id") != request.task_id
            ):
                raise ValueError("Builder must return the assigned task_id and candidate_revision")
            for name in ("build-report.md", "validation.json"):
                if not (path / name).is_file():
                    raise ValueError(f"Builder missing {name}")
            probe = read_json(path / "probes/probe.json")
            if (
                not safe_path(path, probe["script"]).is_file()
                or probe.get("expected_reward") != 0
                or not probe.get("rationale")
            ):
                raise ValueError("Builder negative probe contract missing")

        self._deliver(
            "builder",
            output,
            {"candidate": candidate},
            f"task_id={request.task_id}；candidate_revision={candidate_revision}。"
            "候选包位于 /inputs/candidate。控制层会在你结束后用 Harbor 实际运行 Oracle、"
            "空提交和你提供的错误提交。本容器不提供 Docker runner，不自行运行候选脚本。"
            "先检查可构建性、评分合同和公开镜像有无答案泄露，输出 builder-result.json、"
            "build-report.md、validation.json；未运行项写 not_run，status=validation_pending。"
            "另外必须编写 probes/incorrect.sh：在全新任务环境运行，生成格式合法但关键数值或"
            "逻辑明显错误的交付物，不修改评分器或参考解、不伪造 reward，不只简单退出或留空。"
            "该脚本自包含，用 sh 可执行，不依赖本角色容器文件。"
            "写 probes/probe.json，字段 script='probes/incorrect.sh'、rationale（为何是错误答案）、"
            "expected_reward=0。发现任务内容缺陷记录 issues 并 target_role=author，不改任务。",
            validate,
        )
        raw = read_json(output / "builder-result.json")
        if raw.get("candidate_revision") != candidate_revision:
            raise ValueError("Builder returned stale candidate_revision")
        probe = read_json(output / "probes/probe.json")
        probe_path = safe_path(output, probe["script"])
        if (
            not probe_path.is_file()
            or probe.get("expected_reward") != 0
            or not probe.get("rationale")
        ):
            raise ValueError("Builder did not supply an executable negative-submission contract")
        task = candidate / "task"
        # Keep native task semantics; this initial adapter only handles ordinary single-container tasks.
        task_config = TaskConfig.model_validate_toml((task / "task.toml").read_text())
        if (
            task_config.steps
            or task_config.environment.docker_image
            or list((task / "environment").glob("*compose*"))
        ):
            raise ValueError("This pipeline currently supports only single-step Dockerfile tasks")
        checks = []
        for name, agent, expected in (
            ("oracle", {"name": "oracle"}, 1),
            ("empty_submission", {"name": "nop"}, 0),
            (
                "wrong_submission",
                {
                    "import_path": "heta_framework.generation.exploration._native.probe:IncorrectSubmissionAgent",
                    "kwargs": {"probe_path": str(probe_path)},
                },
                0,
            ),
        ):
            agent["override_timeout_sec"] = self.config.validation_timeout_sec
            trial = self.runner.run(
                task,
                output / "checks" / name,
                agent,
                verify=True,
                timeout=self.config.validation_timeout_sec,
            )
            reward = ((trial.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            exception = trial.get("exception_info")
            checks.append(
                {
                    "name": name,
                    "expected_reward": expected,
                    "reward": reward,
                    "status": "passed" if reward == expected and not exception else "failed",
                    "exception_type": exception.get("exception_type") if exception else None,
                    "trial_dir": trial["local_trial_dir"],
                }
            )
        issues = raw.get("issues", [])
        passed = all(check["status"] == "passed" for check in checks)
        if any(issue.get("severity") == "blocking" for issue in issues):
            passed = False
        if not passed:
            issues.append(
                {
                    "severity": "blocking",
                    "target_role": "author",
                    "finding": "Build/verification checks failed; inspect validation.json and checks logs.",
                    "requested_change": (
                        "读取失败检查的 checks/<name>/ 日志，按实际错误修订。"
                        "若 Oracle 报缺少题目要求的程序文件，请对照公开交付路径和 solution/solve.sh："
                        "仅运行 /solution 中的实现不会创建工作目录中的程序，"
                        "需在参考解执行时把实现安装或写入题目指定位置。"
                        "反馈中的路径和原因须以日志及脚本为依据；其他失败按其实际原因处理。"
                        "不得把答案预置到公开镜像或放宽测试；修订后重新验证。"
                    ),
                }
            )
        write_json(
            output / "validation.json",
            {
                "candidate_revision": candidate_revision,
                "execution_mode": "harbor",
                "checks": checks,
                "validated": passed,
                "limitations": [
                    "One representative wrong submission; no formal Solver or difficulty calibration."
                ],
            },
        )
        result = BuildResult(
            task_id=request.task_id,
            candidate_revision=candidate_revision,
            status="self_test_passed" if passed else "validation_failed",
            execution_mode="harbor",
            issues=issues,
        )
        write_json(output / "builder-agent-result.json", raw)
        write_json(output / "builder-result.json", result.model_dump())
        return result

    def review(
        self,
        request: TaskRequest,
        source: Path,
        candidate: Path,
        build_output: Path,
        output: Path,
        candidate_revision: str,
        revision: int,
    ) -> ReviewResult:
        def validate(path):
            data = read_json(path / "review.json")
            data["execution_mode"] = "harbor"
            result = ReviewResult.model_validate(data)
            if result.candidate_revision != candidate_revision or result.task_id != request.task_id:
                raise ValueError("Review must return the assigned task_id and candidate_revision")
            if not (path / "review.md").is_file():
                raise ValueError("Reviewer missing review.md")

        self._deliver(
            "reviewer",
            output,
            {"source": source, "candidate": candidate, "builder": build_output},
            f"task_id={request.task_id}；candidate_revision={candidate_revision}。"
            "来源在 /inputs/source，候选包在 /inputs/candidate，实际执行证据在 /inputs/builder。"
            "validation.json 是控制层真实 Harbor 回执汇总；checks 中有完整原生日志。"
            "candidate_revision 是整个交付包（包含 authoring）的内容标识；"
            "Harbor 原生 task_checksum 只标识 task 且采用自身算法，二者不能直接比较相等。"
            "跨角色副本一致性由控制层哈希检查负责，不要猜测哈希算法来增加审核阻断项。"
            "对照来源独立检查科学含义、题目和测试是否一致、参考解与测试是否犯同一错误、"
            "错误提交是否有代表性、公开环境是否泄露答案。需要时查阅原始论文和代码。"
            "任务文件禁止修改。输出 review.md 和 review.json。满足要求可 decision=pass，"
            "存在缺陷 decision=revise 并给作者可操作的反馈。不要凭空增加原题未要求的功能。",
            validate,
        )
        data = read_json(output / "review.json")
        data["execution_mode"] = "harbor"
        result = ReviewResult.model_validate(data)
        write_json(output / "review.json", result.model_dump())
        return result

    def extract_skill(self, request, inputs, output, revision):
        from .skill_extraction import validate_draft

        self._deliver(
            "skill-extractor",
            output,
            {name: inputs / name for name in ("source", "candidate", "builder", "reviewer")},
            f"task_id={request.task_id}；candidate_revision={revision}。"
            "从已通过验收的任务中提炼一份可用于同类新任务的 Skill。"
            "原始资料 /inputs/source，完整任务及设计 /inputs/candidate，"
            "构建自测证据 /inputs/builder，任务审核 /inputs/reviewer。"
            "交付 skill-draft.json 与 skills/<name>/SKILL.md 及必要资源。"
            "没有正式 Solver 轨迹，不声称已经验证迁移增益。",
            lambda path: validate_draft(path, request, revision),
        )

    def review_skill(self, request, inputs, draft, output, revision, skill_revision):
        from .skill_extraction import validate_review

        self._deliver(
            "skill-reviewer",
            output,
            {
                **{name: inputs / name for name in ("source", "candidate", "builder", "reviewer")},
                "skill-draft": draft,
            },
            f"task_id={request.task_id}；candidate_revision={revision}；skill_revision={skill_revision}。"
            "独立审核 /inputs/skill-draft，与原始资料、已验证任务和证据核对。"
            "只输出 skill-review.json，不修改 Skill 或任务；不接收提炼者的内部轨迹。",
            lambda path: validate_review(path, request, revision, skill_revision),
        )
