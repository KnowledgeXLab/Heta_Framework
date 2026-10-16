"""Bounded author → builder → reviewer loop, independent of Data Designer."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .artifacts import (
    NeedsMaterialsError,
    candidate_from_output,
    copy_tree,
    fingerprint,
    safe_path,
    write_json,
)
from .config import RunConfig
from .contracts import BuildResult, ReviewResult, RoleBackend, TaskRequest, TaskResult


def produce_task(
    request: TaskRequest, run_dir: Path, config: RunConfig, backend: RoleBackend
) -> TaskResult:
    task_dir = run_dir / "tasks" / request.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    result_path = task_dir / "result.json"
    if result_path.exists():
        result = TaskResult.model_validate_json(result_path.read_text())
        if config.generate_skills:
            from .skill_extraction import extract_task_skill

            return extract_task_skill(request, run_dir, result, backend)
        return result

    def reset(path: Path) -> None:
        # Preserve interrupted outputs for diagnosis; never overwrite them.
        if path.exists():
            if backend.mode == "harbor":
                from .cleanup import cleanup_job

                for job in path.glob("**/jobs/*"):
                    if job.is_dir() and not cleanup_job(job):
                        raise RuntimeError("Cannot confirm cleanup of interrupted Harbor job")
            archive = (
                run_dir / "history" / ("interrupted-" + uuid4().hex) / path.relative_to(run_dir)
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            path.rename(archive)

    def checkpoint(path: Path, outputs: list[Path]) -> dict | None:
        if not path.exists():
            return None
        saved = json.loads(path.read_text())
        if saved["hashes"] != [fingerprint(output) for output in outputs]:
            raise ValueError(f"Completed stage artifacts changed: {path}")
        return saved

    def save(path: Path, outputs: list[Path], value: dict | None = None) -> None:
        write_json(path, {"hashes": [fingerprint(output) for output in outputs], "value": value})

    result = TaskResult(
        task_id=request.task_id,
        status="failed",
        execution_mode=backend.mode,
        report_uri=str(task_dir / "result.json"),
    )
    previous = None
    feedback = None
    source = Path(request.package_uri)
    events_path = task_dir / "events.json"
    events: list[dict] = json.loads(events_path.read_text()) if events_path.exists() else []

    def event(stage: str, revision: int, **details: object) -> None:
        logging.getLogger(__name__).info("[%s] %s (revision %s)", request.task_id, stage, revision)
        events.append(
            {
                "stage": stage,
                "revision": revision,
                "at": datetime.now(timezone.utc).isoformat(),
                **details,
            }
        )
        write_json(task_dir / "events.json", events)

    try:
        source_hash = fingerprint(source)
        for revision in range(config.max_revisions + 1):
            result.revision = revision
            attempt = task_dir / f"revision-{revision:03d}"
            author_output = attempt / "author"
            attempt.mkdir(parents=True, exist_ok=True)
            author_checkpoint = attempt / "author-checkpoint.json"
            if checkpoint(author_checkpoint, [author_output]) is None:
                reset(attempt / "author-runtime")
                reset(author_output)
                author_output.mkdir(parents=True)
                event("author_started", revision)
                backend.author(request, source, author_output, revision, previous, feedback)
            else:
                event("author_reused", revision)
            if fingerprint(source) != source_hash:
                raise ValueError("Author changed the input package snapshot")
            candidate = candidate_from_output(author_output, request.task_id)
            save(author_checkpoint, [author_output])
            content_hash = fingerprint(candidate)
            result.candidate_revision = content_hash
            result.task_package_uri = str(candidate / "task")
            event("candidate_created", revision, candidate_revision=content_hash)

            # Each observing role receives its own copy. Hashes catch mutations, not just paths.
            build_input = attempt / "builder-input"
            build_output = attempt / "builder"
            build_checkpoint = attempt / "builder-checkpoint.json"
            saved_build = checkpoint(build_checkpoint, [build_input, build_output])
            if saved_build is None:
                reset(build_input)
                reset(attempt / "builder-runtime")
                reset(build_output)
                copy_tree(candidate, build_input)
                build = backend.build(request, build_input, build_output, content_hash)
            else:
                build = BuildResult.model_validate(saved_build["value"])
                event("builder_reused", revision)
            if fingerprint(build_input) != content_hash or fingerprint(candidate) != content_hash:
                raise ValueError("Builder changed the candidate package")
            if (
                build.candidate_revision != content_hash
                or build.task_id != request.task_id
                or build.execution_mode != backend.mode
            ):
                raise ValueError("Builder evidence belongs to another candidate/task/backend")
            for name in (build.build_report_path, build.validation_path):
                if not safe_path(build_output, name).is_file():
                    raise ValueError("Builder report artifact missing")
            save(build_checkpoint, [build_input, build_output], build.model_dump())
            build_status = build.simulated_status if backend.mode == "mock" else build.status
            event(
                "builder_finished",
                revision,
                status=build.status,
                simulated_status=build.simulated_status,
            )
            if build_status in {"build_failed", "validation_failed"}:
                feedback = build.model_dump()
            elif build_status != "self_test_passed":
                result.status = "validation_pending"
                break
            else:
                review_input = attempt / "reviewer-input"
                review_output = attempt / "reviewer"
                review_checkpoint = attempt / "reviewer-checkpoint.json"
                saved_review = checkpoint(review_checkpoint, [review_input, review_output])
                evidence_hash = fingerprint(build_output)
                if saved_review is None:
                    reset(review_input)
                    reset(attempt / "reviewer-runtime")
                    reset(review_output)
                    copy_tree(candidate, review_input)
                    review = backend.review(
                        request,
                        source,
                        review_input,
                        build_output,
                        review_output,
                        content_hash,
                        revision,
                    )
                else:
                    review = ReviewResult.model_validate(saved_review["value"])
                    event("reviewer_reused", revision)
                if (
                    fingerprint(review_input) != content_hash
                    or fingerprint(candidate) != content_hash
                ):
                    raise ValueError("Reviewer changed the candidate package")
                if fingerprint(build_output) != evidence_hash or fingerprint(source) != source_hash:
                    raise ValueError("Reviewer changed source or builder evidence")
                if (
                    review.candidate_revision != content_hash
                    or review.task_id != request.task_id
                    or review.execution_mode != backend.mode
                ):
                    raise ValueError("Review evidence belongs to another candidate/task/backend")
                save(review_checkpoint, [review_input, review_output], review.model_dump())
                decision = review.simulated_decision if backend.mode == "mock" else review.decision
                event(
                    "reviewer_finished",
                    revision,
                    decision=review.decision,
                    simulated_decision=review.simulated_decision,
                )
                if decision == "pass":
                    if any(issue.get("severity") == "blocking" for issue in review.issues):
                        raise ValueError("Pass decision has unresolved blocking issues")
                    result.status = "mock_completed" if backend.mode == "mock" else "accepted"
                    result.validated = backend.mode != "mock"
                    break
                if decision in {"validation_pending", "needs_materials"}:
                    result.status = decision
                    break
                if decision != "revise":
                    raise ValueError("Missing or invalid review decision")
                feedback = review.model_dump()
            previous = candidate
            write_json(attempt / "feedback.json", feedback)
            event("returned_to_author", revision)
            if revision == config.max_revisions:
                result.status = "revision_limit"
    except NeedsMaterialsError as exc:
        result.status = "needs_materials"
        result.error = str(exc)
        event("needs_materials", result.revision)
    except Exception as exc:
        # Backends must sanitize errors before returning them.
        result.status = "failed"
        result.validated = False
        result.error = f"{type(exc).__name__}: {exc}"
        event("failed", result.revision, error=result.error)
    if config.generate_skills and result.status in {"accepted", "mock_completed"}:
        result.skill_status = "pending"
    write_json(task_dir / "result.json", result.model_dump())
    if config.generate_skills:
        from .skill_extraction import extract_task_skill

        return extract_task_skill(request, run_dir, result, backend)
    return result
