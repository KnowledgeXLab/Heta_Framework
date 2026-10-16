"""Deterministic test doubles. No network, subprocesses, LLM calls or Docker."""

from pathlib import Path

from .artifacts import write_json
from .contracts import BuildResult, ReviewResult, TaskRequest


class MockBackend:
    mode = "mock"

    def __init__(self, revise_once: bool = False):
        self.revise_once = revise_once

    def author(
        self,
        request: TaskRequest,
        source: Path,
        output: Path,
        revision: int,
        previous: Path | None,
        feedback: dict | None,
    ) -> None:
        package = output / "tasks" / request.task_id
        files = {
            "task/instruction.md": (
                "# Mock arithmetic fixture\n\n"
                "This is a pipeline test fixture, not a scientific task derived from the source package.\n"
                "Read /workspace/input.json containing integer a and b; write their sum as an integer "
                "to /workspace/answer.txt.\n"
            ),
            "task/task.toml": 'schema_version = "1.4"\n[metadata]\nmock = true\n[agent]\ntimeout_sec = 60\n[verifier]\ntimeout_sec = 60\n',
            "task/environment/Dockerfile": (
                "FROM python:3.12-slim\nWORKDIR /workspace\nCOPY input.json /workspace/input.json\n"
            ),
            "task/environment/input.json": '{"a": 2, "b": ' + str(request.variant + 3) + "}\n",
            "task/solution/solve.sh": (
                "#!/bin/sh\nset -eu\npython - <<'PY'\nimport json\nfrom pathlib import Path\n"
                "d = json.loads(Path('/workspace/input.json').read_text())\n"
                "Path('/workspace/answer.txt').write_text(str(d['a'] + d['b']))\nPY\n"
            ),
            "task/tests/test.sh": (
                "#!/bin/sh\nset -eu\nmkdir -p /logs/verifier\nprintf '0' > /logs/verifier/reward.txt\n"
                "python - <<'PY'\nimport json\nfrom pathlib import Path\n"
                "d = json.loads(Path('/workspace/input.json').read_text())\n"
                "assert int(Path('/workspace/answer.txt').read_text()) == d['a'] + d['b']\nPY\n"
                "printf '1' > /logs/verifier/reward.txt\n"
            ),
            "authoring/DESIGN.md": (
                f"# MOCK ONLY\n\nRevision {revision}. Deterministic arithmetic fixture.\n"
                "The supplied scientific source is preserved for handoff testing, not used to derive this fixture.\n"
                f"Previous candidate supplied: {previous is not None}. Feedback supplied: {feedback is not None}.\n"
            ),
        }
        for name, content in files.items():
            path = package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        write_json(
            package / "authoring/sources.json",
            [
                {
                    "source_id": "mock-fixture",
                    "locator": "synthetic arithmetic",
                    "retrieved_at": None,
                    "revision": None,
                    "snapshot_path": None,
                    "evidence_locations": [],
                    "input_package_used_for_science": False,
                }
            ],
        )
        write_json(
            output / "author-result.json",
            {
                "schema_version": "1.0",
                "status": "ready",
                "execution_mode": "mock",
                "tasks": [{"task_id": request.task_id, "package_path": f"tasks/{request.task_id}"}],
                "issues": [],
            },
        )

    def build(
        self, request: TaskRequest, candidate: Path, output: Path, candidate_revision: str
    ) -> BuildResult:
        output.mkdir(parents=True)
        write_json(
            output / "validation.json",
            {
                "candidate_revision": candidate_revision,
                "execution_mode": "mock",
                "validated": False,
                "checks": [
                    {"name": name, "status": "not_run"}
                    for name in (
                        "environment_build",
                        "oracle",
                        "empty_submission",
                        "wrong_submission",
                    )
                ],
            },
        )
        (output / "build-report.md").write_text(
            "# MOCK ONLY\n\nSimulated successful handoff. No build, Oracle or tests were executed.\n"
        )
        result = BuildResult(
            task_id=request.task_id,
            candidate_revision=candidate_revision,
            status="validation_pending",
            execution_mode="mock",
            simulated_status="self_test_passed",
        )
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
        output.mkdir(parents=True)
        revise = self.revise_once and revision == 0
        result = ReviewResult(
            task_id=request.task_id,
            candidate_revision=candidate_revision,
            decision="validation_pending",
            execution_mode="mock",
            simulated_decision="revise" if revise else "pass",
            issues=[
                {
                    "id": "MOCK-R1",
                    "severity": "blocking",
                    "target_role": "author",
                    "finding": "Simulated revision request to exercise author-only repair.",
                }
            ]
            if revise
            else [],
            limitations=[
                "Mock review; no scientific correctness or task executability established."
            ],
        )
        write_json(output / "review.json", result.model_dump())
        (output / "review.md").write_text(
            f"# MOCK ONLY\n\nSimulated decision: {result.simulated_decision}.\n"
        )
        return result

    def extract_skill(self, request, inputs, output, revision):
        skill = output / "skills" / "sum-integer-fields"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: sum-integer-fields\n"
            "description: Sum specified integer fields in a JSON record and check the result.\n"
            "---\n\n# MOCK fixture\n\n"
            "Use when a record contains integer operands. Confirm the requested field names, "
            "reject non-integer values, add with integer arithmetic, and write the requested output. "
            "Check negative and zero operands. This mock does not establish scientific transfer.\n"
        )
        write_json(
            output / "skill-draft.json",
            {
                "task_id": request.task_id,
                "candidate_revision": revision,
                "status": "ready",
                "skill_path": "skills/sum-integer-fields",
                "evidence": ["candidate/task/instruction.md", "candidate/task/solution/solve.sh"],
            },
        )

    def review_skill(self, request, inputs, draft, output, revision, skill_revision):
        write_json(
            output / "skill-review.json",
            {
                "task_id": request.task_id,
                "candidate_revision": revision,
                "skill_revision": skill_revision,
                "decision": "pass",
                "findings": [],
                "limitations": ["Mock review only; no scientific validity or transfer verified."],
            },
        )
