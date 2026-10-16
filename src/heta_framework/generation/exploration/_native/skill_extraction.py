"""Optional, independently reviewed skill artifacts derived from accepted tasks."""

import re
from pathlib import Path
from typing import Literal
from uuid import uuid4

import yaml
from pydantic import Field

from .artifacts import copy_tree, fingerprint, read_json, safe_path, write_json
from .contracts import Record, RoleBackend, TaskRequest, TaskResult


class SkillDraft(Record):
    task_id: str
    candidate_revision: str
    status: Literal["ready", "not_applicable"]
    skill_path: str | None = None
    reason: str = ""
    evidence: list[str] = Field(default_factory=list)


class SkillReview(Record):
    task_id: str
    candidate_revision: str
    skill_revision: str
    decision: Literal["pass", "reject"]
    findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def validate_draft(output: Path, request: TaskRequest, revision: str) -> SkillDraft:
    draft = SkillDraft.model_validate(read_json(output / "skill-draft.json"))
    if draft.task_id != request.task_id or draft.candidate_revision != revision:
        raise ValueError("Skill draft belongs to another task or candidate")
    if draft.status == "not_applicable":
        if not draft.reason or draft.skill_path is not None:
            raise ValueError("not_applicable requires a reason and no skill_path")
        return draft
    if not draft.skill_path or not draft.evidence:
        raise ValueError("Ready skill requires skill_path and evidence locators")
    skill = safe_path(output, draft.skill_path)
    if skill.parent != output / "skills":
        raise ValueError("Skill must be in skills/<name>")
    fingerprint(skill)
    text = (skill / "SKILL.md").read_text()
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError("SKILL.md requires YAML frontmatter")
    header, body = text[4:].split("\n---\n", 1)
    try:
        metadata = yaml.safe_load(header)
    except yaml.YAMLError as exc:
        raise ValueError("Invalid skill frontmatter") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Skill frontmatter must be a mapping")
    name = metadata.get("name")
    description = metadata.get("description")
    if (
        not isinstance(name, str)
        or len(name) > 64
        or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
        or name != skill.name
    ):
        raise ValueError("Skill name must match its lowercase-hyphen directory")
    if not isinstance(description, str) or not description.strip() or not body.strip():
        raise ValueError("Skill requires a description and nonempty instructions")
    return draft


def validate_review(
    output: Path, request: TaskRequest, revision: str, skill_hash: str
) -> SkillReview:
    review = SkillReview.model_validate(read_json(output / "skill-review.json"))
    if (
        review.task_id != request.task_id
        or review.candidate_revision != revision
        or review.skill_revision != skill_hash
    ):
        raise ValueError("Skill review belongs to another task/candidate/skill")
    if review.decision == "pass" and review.findings:
        raise ValueError("Skill approval has unresolved findings")
    if review.decision == "reject" and not review.findings:
        raise ValueError("Rejected skill needs actionable findings")
    return review


def extract_task_skill(
    request: TaskRequest, run_dir: Path, result: TaskResult, backend: RoleBackend
) -> TaskResult:
    """Never alter task acceptance; publish only a separately approved skill."""
    if result.status not in {"accepted", "mock_completed"}:
        result.skill_status = "not_applicable"
        write_json(Path(result.report_uri), result.model_dump())
        return result
    if result.skill_status not in {"disabled", "pending"}:
        return result
    result.skill_status = "pending"
    write_json(Path(result.report_uri), result.model_dump())
    task_dir = run_dir / "tasks" / request.task_id
    attempt = task_dir / f"revision-{result.revision:03d}"
    branch = attempt / "skill-extraction"
    result.skill_report_uri = str(branch / "result.json")

    def reset(path: Path) -> None:
        if path.exists():
            if backend.mode == "harbor":
                from .cleanup import cleanup_job

                for job in path.glob("**/jobs/*"):
                    if job.is_dir() and not cleanup_job(job):
                        raise RuntimeError("Cannot confirm interrupted skill job cleanup")
            archive = (
                run_dir / "history" / ("interrupted-" + uuid4().hex) / path.relative_to(run_dir)
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            path.rename(archive)

    try:
        branch.mkdir(exist_ok=True)
        candidate = Path(result.task_package_uri).parent
        if fingerprint(candidate) != result.candidate_revision:
            raise ValueError("Accepted candidate changed before skill extraction")
        # Roles get isolated copies; optional failures cannot damage the accepted task.
        inputs = branch / "inputs"
        reset(inputs)
        for name, source in {
            "source": Path(request.package_uri),
            "candidate": candidate,
            "builder": attempt / "builder",
            "reviewer": attempt / "reviewer",
        }.items():
            copy_tree(source, inputs / name)
        input_hash = fingerprint(inputs)
        draft_output = branch / "draft"
        checkpoint = branch / "draft-checkpoint.json"
        if checkpoint.exists():
            saved = read_json(checkpoint)
            if saved != {"input_hash": input_hash, "draft_hash": fingerprint(draft_output)}:
                raise ValueError("Skill extraction inputs or completed draft changed")
        else:
            reset(branch / "skill-extractor-runtime")
            reset(draft_output)
            backend.extract_skill(request, inputs, draft_output, result.candidate_revision)
            if fingerprint(inputs) != input_hash:
                raise ValueError("Skill extractor changed its inputs")
            validate_draft(draft_output, request, result.candidate_revision)
            write_json(
                checkpoint, {"input_hash": input_hash, "draft_hash": fingerprint(draft_output)}
            )
        draft = validate_draft(draft_output, request, result.candidate_revision)
        if draft.status == "not_applicable":
            result.skill_status = "not_applicable"
            result.skill_error = draft.reason
        else:
            skill_hash = fingerprint(safe_path(draft_output, draft.skill_path))
            review_output = branch / "review"
            review_input = branch / "review-input"
            reset(review_input)
            copy_tree(draft_output, review_input)
            draft_hash = fingerprint(review_input)
            reset(branch / "skill-reviewer-runtime")
            reset(review_output)
            backend.review_skill(
                request, inputs, review_input, review_output, result.candidate_revision, skill_hash
            )
            if fingerprint(inputs) != input_hash or fingerprint(review_input) != draft_hash:
                raise ValueError("Skill reviewer changed its inputs")
            review = validate_review(review_output, request, result.candidate_revision, skill_hash)
            result.skill_status = (
                ("mock_completed" if backend.mode == "mock" else "accepted")
                if review.decision == "pass"
                else "rejected"
            )
            if review.decision == "pass":
                result.skill_package_uri = str(safe_path(draft_output, draft.skill_path))
                write_json(
                    branch / "skill-link.json",
                    {
                        "task_id": request.task_id,
                        "candidate_revision": result.candidate_revision,
                        "skill_revision": skill_hash,
                        "skill_package_uri": result.skill_package_uri,
                        "source_package_uri": request.package_uri,
                        "source_sha256": fingerprint(Path(request.package_uri)),
                        "task_package_uri": result.task_package_uri,
                        "evidence": draft.evidence,
                        "execution_mode": backend.mode,
                        "review_decision": review.decision,
                        "transfer_evaluated": False,
                        "solver_access": False,
                    },
                )
    except Exception as exc:
        result.skill_status = "failed"
        result.skill_package_uri = None
        result.skill_error = f"{type(exc).__name__}: {exc}"
    write_json(
        branch / "result.json",
        {
            "task_id": result.task_id,
            "status": result.skill_status,
            "skill_package_uri": result.skill_package_uri,
            "error": result.skill_error,
            "transfer_evaluated": False,
            "execution_mode": backend.mode,
        },
    )
    write_json(Path(result.report_uri), result.model_dump())
    return result
