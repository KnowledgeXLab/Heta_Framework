# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic domain checks for the OmegaDesigner pipeline.

This module re-implements, as pure functions, the validation half of Omega's
``omega/science/task_generation/pipeline.py`` (plus the static path resolution of
``path_contract.py``). Every check returns a
:class:`~omegadesigner.domain.schemas.CheckResult` instead of raising, so a
domain-rule violation becomes a repairable ``ok=False`` result.

The two deliberate exceptions are :func:`safe_relative_path` and
:func:`text_files`: they run inside Data Designer parsers where a ``ValueError``
is the intended repair signal (the exception text is fed back to the model).

Nothing in this module imports or executes Omega.
"""

from __future__ import annotations

import ast
import posixpath
import re
from pathlib import PurePosixPath
from typing import Any

from heta_framework.generation.graph._native._compat import tomllib
from heta_framework.generation.graph._native.config import DIFFICULTY_TIER_BANDS, Settings
from heta_framework.generation.graph._native.domain.schemas import (
    CANDIDATE_SCORE_FIELDS,
    EVALUATION_MODES,
    KEBAB,
    PINNED_REQUIREMENT,
    PYTHON_VERSIONS,
    REQUIRED_EVIDENCE_INVENTORY,
    REQUIRED_FEASIBILITY_CHECKS,
    REQUIRED_IMPLEMENTATION_FILES,
    REQUIRED_SPEC_FILES,
    REQUIRED_TASK_SPEC_FIELDS,
    TBS_CANARY,
    Affordance,
    Candidate,
    CandidateReview,
    CheckResult,
    DataContract,
    DifficultyContract,
    EvaluationTopology,
    RuntimeSpec,
    SemanticReview,
)

__all__ = [
    "check_candidate_list",
    "check_candidate_review",
    "check_implementation_draft",
    "check_semantic_review",
    "check_spec_draft",
    "safe_relative_path",
    "select_variants",
    "text_files",
    "validate_absolute_runtime_path",
    "validate_evaluation_topology",
    "validate_implementation_check",
    "validate_implementation_topology",
    "validate_instruction_absolute_paths",
    "validate_python_source",
    "validate_python_verifier",
    "validate_runtime_spec",
    "validate_task_toml",
]

# Required-field sets are derived from the pydantic models in ``schemas.py`` so
# the contract has exactly one source of truth. ``_REQUIRED_RUNTIME_FIELDS``
# drops ``agent_timeout_sec`` because Omega derives it from the difficulty
# contract instead of requiring the model to supply it.
_REQUIRED_SEMANTIC_REVIEW_FIELDS: frozenset[str] = frozenset(SemanticReview.model_fields)
_AFFORDANCE_FIELDS: frozenset[str] = frozenset(Affordance.model_fields)
_REQUIRED_CANDIDATE_FIELDS: frozenset[str] = frozenset(Candidate.model_fields) - {"authoring_warnings"}
_REQUIRED_CANDIDATE_REVIEW_FIELDS: frozenset[str] = frozenset(CandidateReview.model_fields)
_REQUIRED_DATA_CONTRACT_FIELDS: frozenset[str] = frozenset(DataContract.model_fields)
_REQUIRED_DIFFICULTY_FIELDS: frozenset[str] = frozenset(DifficultyContract.model_fields)
_REQUIRED_RUNTIME_FIELDS: frozenset[str] = frozenset(RuntimeSpec.model_fields) - {"agent_timeout_sec"}
_REQUIRED_TOPOLOGY_FIELDS: frozenset[str] = frozenset(EvaluationTopology.model_fields)

_REQUIRED_TASK_TOML_METADATA: frozenset[str] = frozenset(
    {
        "author_name",
        "author_email",
        "author_organization",
        "author_profile",
        "conflicts_of_interest",
        "relevant_experience",
        "domain",
        "field",
        "subfield",
        "tags",
        "expert_time_estimate_hours",
    }
)
_TASK_TOML_NAME = re.compile(r"terminal-bench-science/[a-z0-9]+(?:-[a-z0-9]+)*")
_RUNTIME_PATH = re.compile(r"/[A-Za-z0-9_./-]+")
_SHELL_SET_E = re.compile(r"(?m)^\s*set\s+-[^\n]*e")
_VERIFIER_SOLUTION_REFERENCE = re.compile(r"(?<![A-Za-z0-9_])/solution(?:/|\b)")
_RUNTIME_LIMITS: dict[str, tuple[int, int]] = {
    "verifier_timeout_sec": (1, 7_200),
    "build_timeout_sec": (1, 7_200),
    "cpus": (1, 16),
    "memory_mb": (256, 65_536),
    "storage_mb": (256, 102_400),
}
_INSTRUCTION_PATH_EXTENSIONS = (
    "txt|json|csv|toml|yaml|yml|py|sh|out|log|xml|html|gz|bin|tar|zip|npy|npz|pkl|pt|jsonl|cfg|conf|ini|dat|db|sqlite"
)
_INSTRUCTION_RELATIVE_PATH = re.compile(
    rf"(?<![/A-Za-z0-9_.-])"
    rf"([A-Za-z0-9_-]+/[A-Za-z0-9_./-]+\.(?:{_INSTRUCTION_PATH_EXTENSIONS}))\b"
)


# --------------------------------------------------------------------------- #
# internal helpers
# --------------------------------------------------------------------------- #
def _check_result(errors: list[str], warnings: list[str] | None = None) -> CheckResult:
    return CheckResult(ok=not errors, errors=list(errors), warnings=list(warnings or []))


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_equal(left: Any, right: Any) -> bool:
    left_value = _as_float(left)
    right_value = _as_float(right)
    return left_value is not None and right_value is not None and left_value == right_value


def _resolve_declared_path(node: ast.AST | None, names: dict[str, str]) -> str | None:
    """Conservative static path resolution (Omega ``path_contract.resolve``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
        left = _resolve_declared_path(node.left, names)
        right = _resolve_declared_path(node.right, names)
        if left is not None and right is not None:
            return posixpath.join(left, right) if isinstance(node.op, ast.Div) else left + right
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            label = func.id
        elif isinstance(func, ast.Attribute):
            label = func.attr
        else:
            label = ""
        args = [_resolve_declared_path(arg, names) for arg in node.args]
        if label in ("Path", "PurePath", "PurePosixPath", "join") and args and all(arg is not None for arg in args):
            return posixpath.join(*args)  # type: ignore[arg-type]
        if label in ("glob", "rglob", "joinpath") and args and all(arg is not None for arg in args):
            base = _resolve_declared_path(func.value, names) if isinstance(func, ast.Attribute) else None
            if base is not None:
                return posixpath.join(base, *args)  # type: ignore[arg-type]
            if label == "glob":
                return args[0]
    return None


def _resolved_paths(source: str) -> set[str]:
    """Return absolute string paths a source file statically references.

    Re-implementation of Omega ``path_contract.resolved_paths``. Generated source
    is never executed; only module-level constants are resolved.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: dict[str, str] = {}
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = _resolve_declared_path(node.value, names)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is not None:
                for target in targets:
                    if isinstance(target, ast.Name):
                        names[target.id] = value
    for node in ast.walk(tree):
        value = _resolve_declared_path(node, names)
        if value and value.startswith("/"):
            found.add(posixpath.normpath(value))
    return found


def _runtime_permission_warnings(files: dict[str, str]) -> list[str]:
    """Report authoring patterns; actual execution and TBS checks decide validity.

    Omega ``_validate_runtime_permissions``. Warnings are returned instead of
    being written back onto a draft, because this function never mutates inputs.
    """
    warnings: list[str] = []
    for name, source in files.items():
        if not name.endswith(".py"):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        assignments: dict[str, ast.AST] = {
            node.targets[0].id: node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        }
        aliases = {"run_submitted"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("omega_runtime"):
                aliases.update(alias.asname or alias.name for alias in node.names if alias.name == "run_submitted")

        def literals(node: ast.AST, seen: set[str] | None = None) -> Any:
            seen = set() if seen is None else seen
            if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
                yield from literals(assignments[node.id], seen | {node.id})
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                yield node.value
            else:
                for child in ast.iter_child_nodes(node):
                    yield from literals(child, seen)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
                "runtime_controls",
                "run_runtime_controls",
            }:
                warnings.append(
                    f"{name}: generic runtime self-tests belong in Omega regression tests; remove custom "
                    "runtime_controls; retain task-specific scientific checks"
                )
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if isinstance(node.func, ast.Name):
                called = node.func.id
            elif isinstance(node.func, ast.Attribute):
                called = node.func.attr
            else:
                called = ""
            if called in aliases and any(
                value == "/tests" or value.startswith("/tests/") for value in literals(node.args[0])
            ):
                warnings.append(
                    f"{name}: run_submitted cannot access protected /tests paths; "
                    "stage explicitly listed public programs/dependencies with private_program "
                    "and inputs with private_case"
                )
    return warnings


# --------------------------------------------------------------------------- #
# stage 2: semantic review
# --------------------------------------------------------------------------- #
def check_semantic_review(draft: dict, settings: Settings) -> CheckResult:
    """Omega ``_validate_review`` plus the deterministic re-gate at pipeline.py:241-252.

    ``ok`` reflects structural validity only. The re-gate outcome is reported in
    ``warnings`` as ``deterministic_regate:true`` / ``deterministic_regate:false``
    so a column can read it without re-deriving it.
    """
    if not isinstance(draft, dict):
        return _check_result(["Cheap model review must be an object"])

    errors: list[str] = []
    warnings: list[str] = []

    missing = sorted(key for key in _REQUIRED_SEMANTIC_REVIEW_FIELDS if key not in draft)
    if missing:
        errors.append(f"Cheap model review is missing fields: {', '.join(missing)}")
    if "paper_summary" in draft and not _nonempty_str(draft["paper_summary"]):
        errors.append("paper_summary must be a non-empty string")
    if "component_summary" in draft and not _nonempty_str(draft["component_summary"]):
        errors.append("component_summary must be a non-empty string")
    if "suitable_for_task_generation" in draft and not isinstance(draft["suitable_for_task_generation"], bool):
        errors.append("suitable_for_task_generation must be a boolean")
    if "blockers" in draft and not isinstance(draft["blockers"], list):
        errors.append("blockers must be an array")

    score: float | None = None
    if "suitability_score" in draft:
        try:
            score = float(draft["suitability_score"])
        except (TypeError, ValueError):
            errors.append("suitability_score must be numeric")
            score = None
        else:
            if not 0.0 <= score <= 1.0:
                errors.append("suitability_score must be in [0, 1]")

    if "evidence_inventory" in draft:
        inventory = draft["evidence_inventory"]
        if not isinstance(inventory, dict):
            errors.append("evidence_inventory must be an object")
        else:
            missing_inventory = sorted(REQUIRED_EVIDENCE_INVENTORY - set(inventory))
            if missing_inventory:
                errors.append(f"evidence_inventory is missing fields: {', '.join(missing_inventory)}")
            invalid_inventory = sorted(
                key for key in REQUIRED_EVIDENCE_INVENTORY if key in inventory and not isinstance(inventory[key], list)
            )
            if invalid_inventory:
                errors.append(f"evidence_inventory fields must be arrays: {', '.join(invalid_inventory)}")

    affordances: Any = None
    if "computational_affordances" in draft:
        affordances = draft["computational_affordances"]
        if not isinstance(affordances, list):
            errors.append("computational_affordances must be an array")
        else:
            for index, affordance in enumerate(affordances):
                if not isinstance(affordance, dict):
                    errors.append(f"computational_affordances[{index}] must be an object")
                    continue
                missing_affordance = sorted(_AFFORDANCE_FIELDS - set(affordance))
                if missing_affordance:
                    errors.append(
                        f"computational_affordances[{index}] is missing fields: " + ", ".join(missing_affordance)
                    )
            if draft.get("suitable_for_task_generation") is True and not affordances:
                errors.append("A suitable paper must expose at least one computational affordance")

    checks: Any = None
    if "constraint_checks" in draft:
        checks = draft["constraint_checks"]
        if not isinstance(checks, dict):
            errors.append("constraint_checks must be an object")
        else:
            missing_checks = sorted(REQUIRED_FEASIBILITY_CHECKS - set(checks))
            if missing_checks:
                errors.append(f"constraint_checks is missing fields: {', '.join(missing_checks)}")
            non_boolean = sorted(
                name for name in REQUIRED_FEASIBILITY_CHECKS if name in checks and not isinstance(checks[name], bool)
            )
            if non_boolean:
                errors.append(f"constraint_checks fields must be boolean: {', '.join(non_boolean)}")

    minimum = settings.review.minimum_suitability_score
    declared = draft.get("suitable_for_task_generation")
    gate = (
        declared is True
        and score is not None
        and 0.0 <= score <= 1.0
        and score >= minimum
        and isinstance(checks, dict)
        and all(bool(value) for value in checks.values())
    )
    warnings.append(f"deterministic_regate:{'true' if gate else 'false'}")
    if declared is not True:
        warnings.append("model_declared_suitable:false")
    if score is not None and score < minimum:
        warnings.append(f"suitability_score_below_minimum:{score}")
    if isinstance(checks, dict):
        for name in sorted(name for name in REQUIRED_FEASIBILITY_CHECKS if not bool(checks.get(name))):
            warnings.append(f"failed_constraint_check:{name}")
    return _check_result(errors, warnings)


# --------------------------------------------------------------------------- #
# stage 3: candidate generation
# --------------------------------------------------------------------------- #
def check_candidate_list(draft: dict, settings: Settings) -> CheckResult:
    """Omega ``_validate_candidate_draft``.

    Like Omega, this mutates each candidate in place by writing the
    ``authoring_warnings`` heuristics (pipeline.py:1080-1087).
    """
    if not isinstance(draft, dict) or set(draft) != {"candidates"}:
        return _check_result(["Candidate-generation output must contain exactly candidates"])

    errors: list[str] = []
    candidates = draft["candidates"]
    expected_count = settings.candidates.candidate_count
    if not isinstance(candidates, list) or len(candidates) != expected_count:
        errors.append(f"candidates must contain exactly {expected_count} items")
    if not isinstance(candidates, list):
        return _check_result(errors)

    seen_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            errors.append(f"candidates[{index}] must be an object")
            continue
        missing = sorted(_REQUIRED_CANDIDATE_FIELDS - set(candidate))
        if missing:
            errors.append(f"candidates[{index}] is missing fields: {', '.join(missing)}")
            continue
        candidate_id = candidate["candidate_id"]
        valid_id = isinstance(candidate_id, str) and KEBAB.fullmatch(candidate_id) is not None
        if not valid_id:
            errors.append(f"candidates[{index}].candidate_id must be lowercase kebab-case")
        elif candidate_id in seen_ids:
            errors.append(f"Duplicate candidate_id: {candidate_id}")

        arrays_ok = True
        for key in ("concepts_used", "evidence_basis", "required_reasoning_steps"):
            value = candidate[key]
            if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
                errors.append(f"candidates[{index}].{key} must be a non-empty string array")
                arrays_ok = False

        difficulty = candidate["estimated_difficulty"]
        difficulty_ok = not isinstance(difficulty, bool) and isinstance(difficulty, int) and 1 <= difficulty <= 5
        if not difficulty_ok:
            errors.append(f"candidates[{index}].estimated_difficulty must be an integer in [1, 5]")

        # Omega raises before this point, so a candidate with malformed concepts
        # or difficulty never receives authoring_warnings. Mirror that here.
        if arrays_ok and difficulty_ok:
            suggestions: list[str] = []
            if len(candidate["concepts_used"]) < (1 if difficulty <= 2 else 2):
                suggestions.append("Omega difficulty heuristic: few dependent concepts")
            if len(candidate["required_reasoning_steps"]) < (2 if difficulty <= 2 else 3):
                suggestions.append("Omega difficulty heuristic: short implementation path")
            if not candidate["weak_baselines"]:
                suggestions.append("No comparator analysis supplied; this is not a TBS rejection criterion")
            candidate["authoring_warnings"] = suggestions

        baselines = candidate["weak_baselines"]
        if not isinstance(baselines, list):
            errors.append(f"candidates[{index}].weak_baselines must be an array")
        else:
            for baseline in baselines:
                if not isinstance(baseline, dict) or not all(
                    isinstance(baseline.get(key), str) and baseline[key].strip()
                    for key in ("name", "strategy", "expected_failure")
                ):
                    errors.append(f"candidates[{index}].weak_baselines entries require name/strategy/expected_failure")

        shortcuts = candidate["shortcut_analysis"]
        if not isinstance(shortcuts, list):
            errors.append(f"candidates[{index}].shortcut_analysis must be an array")
        else:
            for shortcut in shortcuts:
                if not isinstance(shortcut, dict) or not all(
                    isinstance(shortcut.get(key), str) and shortcut[key].strip() for key in ("shortcut", "blocked_by")
                ):
                    errors.append(f"candidates[{index}].shortcut_analysis entries require shortcut/blocked_by")

        textual = _REQUIRED_CANDIDATE_FIELDS - {
            "concepts_used",
            "evidence_basis",
            "required_reasoning_steps",
            "weak_baselines",
            "shortcut_analysis",
            "estimated_difficulty",
        }
        invalid_text = sorted(
            key for key in textual if not isinstance(candidate[key], str) or not candidate[key].strip()
        )
        if invalid_text:
            errors.append(f"candidates[{index}] fields must be non-empty strings: {', '.join(invalid_text)}")

        if valid_id:
            seen_ids.add(candidate_id)

    return _check_result(errors)


# --------------------------------------------------------------------------- #
# stage 4: candidate review and tier selection
# --------------------------------------------------------------------------- #
def check_candidate_review(
    draft: dict,
    candidates_by_id: dict[str, Any],
    settings: Settings,
) -> CheckResult:
    """Omega ``_validate_candidate_review`` including both review thresholds."""
    if not isinstance(draft, dict):
        return _check_result(["Candidate-review output must be an object"])

    errors: list[str] = []
    missing = sorted(_REQUIRED_CANDIDATE_REVIEW_FIELDS - set(draft))
    if missing:
        errors.append(f"Candidate-review output is missing fields: {', '.join(missing)}")

    reviews = draft.get("reviews")
    reviews_valid = isinstance(reviews, list)
    reviews_by_id: dict[str, dict[str, Any]] = {}
    if not reviews_valid:
        errors.append("reviews must cover every candidate exactly once")
    else:
        if len(reviews) != len(candidates_by_id):
            errors.append("reviews must cover every candidate exactly once")
        for index, review in enumerate(reviews):
            if not isinstance(review, dict):
                errors.append(f"reviews[{index}] must be an object")
                continue
            candidate_id = review.get("candidate_id")
            if candidate_id not in candidates_by_id or candidate_id in reviews_by_id:
                errors.append(f"reviews[{index}] has an unknown or duplicate candidate_id")
                continue
            scores = review.get("scores")
            if not isinstance(scores, dict) or set(scores) != CANDIDATE_SCORE_FIELDS:
                errors.append(f"reviews[{index}].scores must contain the exact scoring rubric")
            else:
                for key, value in scores.items():
                    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                        errors.append(f"reviews[{index}].scores.{key} must be an integer in [1, 5]")
            overall = _as_float(review.get("overall_score"))
            if overall is None:
                errors.append(f"reviews[{index}].overall_score must be numeric")
            elif not 0.0 <= overall <= 1.0:
                errors.append(f"reviews[{index}].overall_score must be in [0, 1]")
            for key in ("hard_failures", "strengths", "weaknesses"):
                if not isinstance(review.get(key), list):
                    errors.append(f"reviews[{index}].{key} must be an array")
            if review.get("decision") not in {"select", "reject"}:
                errors.append(f"reviews[{index}].decision must be select or reject")
            reviews_by_id[candidate_id] = review
        if set(reviews_by_id) != set(candidates_by_id):
            errors.append("reviews must cover every candidate exactly once")

    if "selection_reason" in draft and not _nonempty_str(draft["selection_reason"]):
        errors.append("selection_reason must be a non-empty string")
    rejected_all = draft.get("rejected_all")
    global_blockers = draft.get("global_blockers")
    if ("rejected_all" in draft or "global_blockers" in draft) and (
        not isinstance(rejected_all, bool) or not isinstance(global_blockers, list)
    ):
        errors.append("rejected_all must be boolean and global_blockers must be an array")

    selected_id = draft.get("selected_candidate_id")
    selected_reviews = (
        [item for item in reviews if isinstance(item, dict) and item.get("decision") == "select"]
        if reviews_valid
        else []
    )
    if rejected_all is True:
        if selected_id is not None or selected_reviews:
            errors.append("rejected_all requires no selected candidate")
    elif rejected_all is False:
        if selected_id not in candidates_by_id or len(selected_reviews) != 1:
            errors.append("Exactly one known candidate must be selected")
        else:
            selected_review = reviews_by_id.get(selected_id)
            if selected_review is None:
                errors.append("Exactly one known candidate must be selected")
            else:
                if selected_review.get("decision") != "select" or selected_review.get("hard_failures"):
                    errors.append("Selected candidate must have decision=select and no hard failures")
                minimum_overall = settings.candidates.minimum_candidate_score
                minimum_dimension = settings.candidates.minimum_candidate_dimension_score
                overall = _as_float(selected_review.get("overall_score"))
                if overall is not None and overall < minimum_overall:
                    errors.append("Selected candidate is below minimum_candidate_score")
                scores = selected_review.get("scores")
                if isinstance(scores, dict):
                    below = sorted(
                        key
                        for key, value in scores.items()
                        if not isinstance(value, bool) and isinstance(value, int) and value < minimum_dimension
                    )
                    if below:
                        errors.append("Selected candidate is below minimum dimension score for: " + ", ".join(below))

    return _check_result(errors)


def _review_index(review_draft: dict) -> tuple[dict[str, dict], list[str]]:
    """Index ``reviews`` by candidate id, reporting structural problems once."""
    if not isinstance(review_draft, dict):
        return {}, ["Candidate-review output must be an object"]
    reviews_raw = review_draft.get("reviews")
    if not isinstance(reviews_raw, list):
        return {}, ["reviews must cover every candidate exactly once"]
    reviews: dict[str, dict] = {}
    for item in reviews_raw:
        if isinstance(item, dict) and isinstance(item.get("candidate_id"), str):
            reviews[item["candidate_id"]] = item
    return reviews, []


def _eligible_candidates_for_tier(
    candidates_by_id: dict[str, Any],
    reviews: dict[str, dict],
    tier: str,
    settings: Settings,
) -> tuple[list[tuple[float, float, str, dict[str, Any]]], list[str]]:
    """Eligibility filter for one tier: band, hard failures, score floors."""
    minimum_overall = settings.candidates.minimum_candidate_score
    minimum_dimension = settings.candidates.minimum_candidate_dimension_score
    band = DIFFICULTY_TIER_BANDS[tier]
    eligible: list[tuple[float, float, str, dict[str, Any]]] = []
    errors: list[str] = []
    for candidate_id, candidate in candidates_by_id.items():
        if not isinstance(candidate, dict):
            errors.append(f"candidates_by_id[{candidate_id}] must be an object")
            continue
        item = reviews.get(candidate_id)
        if item is None:
            errors.append(f"reviews must cover every candidate exactly once: {candidate_id}")
            continue
        if candidate.get("estimated_difficulty") not in band:
            continue
        if item.get("hard_failures"):
            continue
        overall = _as_float(item.get("overall_score"))
        if overall is None:
            errors.append(f"candidate {candidate_id}: overall_score must be numeric")
            continue
        if overall < minimum_overall:
            continue
        scores = item.get("scores")
        if not isinstance(scores, dict) or not scores:
            errors.append(f"candidate {candidate_id}: scores must contain the exact scoring rubric")
            continue
        try:
            minimum_dimension_value = min(scores.values())
            mean_score = sum(scores.values()) / len(scores)
        except TypeError:
            errors.append(f"candidate {candidate_id}: scores must contain the exact scoring rubric")
            continue
        if not isinstance(minimum_dimension_value, int) or isinstance(minimum_dimension_value, bool):
            errors.append(f"candidate {candidate_id}: scores must contain the exact scoring rubric")
            continue
        if minimum_dimension_value < minimum_dimension:
            continue
        eligible.append((overall, float(mean_score), candidate_id, candidate))
    return eligible, errors


def select_variants(
    candidates_by_id: dict[str, Any],
    review_draft: dict,
    tiers: tuple[str, ...] | list[str],
    settings: Settings,
) -> tuple[dict[str, dict[str, Any]], list[str], CheckResult]:
    """Omega ``_select_difficulty_variants`` (pipeline.py:1198-1237).

    One selection per requested tier: the highest-scoring eligible candidate whose
    ``estimated_difficulty`` falls in that tier's band, tie-broken by mean dimension score
    then candidate id, overridable per tier by ``preferred_candidate_ids``. A tier with no
    eligible candidate is reported as *missing* (Omega's ``missing_tiers``) instead of
    failing the paper; an ineligible *preferred* candidate is a hard error, as in Omega.

    Returns ``(selections, missing_tiers, result)``. ``selections`` preserves the requested
    tier order, which is the order the fan-out emits rows in.
    """
    reviews, errors = _review_index(review_draft)
    if errors:
        return {}, [], _check_result(errors)
    warnings: list[str] = []
    selections: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for tier in tiers:
        if tier not in DIFFICULTY_TIER_BANDS:
            errors.append(f"Unknown difficulty tier: {tier}")
            continue
        eligible, tier_errors = _eligible_candidates_for_tier(candidates_by_id, reviews, tier, settings)
        errors.extend(tier_errors)
        if not eligible:
            if tier_errors:
                continue
            preferred = settings.candidates.preferred_candidate_ids.get(tier)
            if preferred is not None:
                errors.append(f"no eligible preferred candidate for {tier}")
                continue
            warnings.append(f"no_eligible_candidate:{tier}")
            missing.append(tier)
            continue
        eligible.sort(key=lambda value: (-value[0], -value[1], value[2]))
        preferred = settings.candidates.preferred_candidate_ids.get(tier)
        if preferred is not None:
            matches = [item for item in eligible if item[2] == preferred]
            if not matches:
                errors.append(f"preferred candidate {preferred!r} is not eligible for {tier}")
                continue
            chosen = matches[0]
        else:
            chosen = eligible[0]
        warnings.append(f"difficulty_tier:{tier}")
        warnings.append(f"selected_candidate:{chosen[2]}")
        warnings.append(f"selected_overall_score:{chosen[0]}")
        selections[tier] = chosen[3]
    return selections, missing, _check_result(errors, warnings)


# --------------------------------------------------------------------------- #
# stage 5: task spec (already canonicalized)
# --------------------------------------------------------------------------- #
def check_spec_draft(
    draft: dict,
    settings: Settings,
    *,
    selected_candidate: dict | None = None,
) -> CheckResult:
    """Omega ``_validate_spec_draft`` for an ALREADY-CANONICALIZED spec draft.

    Canonicalization (time-budget rewrite, ``fill_environment_fields``,
    ``absolute_instruction_paths``, canary injection) is owned by another module
    and runs before this function. This function only validates and never
    rewrites files or mutates the draft.
    """
    if not isinstance(draft, dict):
        return _check_result(["Task-spec model output must be an object"])

    errors: list[str] = []
    warnings: list[str] = []

    package_name = str(draft.get("package_name") or "")
    if KEBAB.fullmatch(package_name) is None:
        errors.append("package_name must be lowercase kebab-case")
    elif len(package_name.split("-")) > 3:
        errors.append("package_name must contain at most three hyphen-separated tokens")

    task_spec = draft.get("task_spec")
    if not isinstance(task_spec, dict):
        errors.append("Strong model output must include a task_spec object")
    else:
        missing_spec = sorted(key for key in REQUIRED_TASK_SPEC_FIELDS if not task_spec.get(key))
        if missing_spec:
            errors.append(f"Generated task_spec is missing fields: {', '.join(missing_spec)}")

    data_contract = draft.get("data_contract")
    if not isinstance(data_contract, dict):
        errors.append("Spec output must include a data_contract object")
    else:
        missing_data = sorted(_REQUIRED_DATA_CONTRACT_FIELDS - set(data_contract))
        if missing_data:
            errors.append(f"data_contract is missing fields: {', '.join(missing_data)}")
        if not isinstance(data_contract.get("public_files"), list) or not data_contract.get("public_files"):
            errors.append("data_contract.public_files must be an array")
        if not isinstance(data_contract.get("test_files"), list) or not data_contract.get("test_files"):
            errors.append("data_contract.test_files must be an array")

    difficulty = draft.get("difficulty_contract")
    if not isinstance(difficulty, dict):
        errors.append("Spec output must include a difficulty_contract object")
        difficulty = {}
    else:
        missing_difficulty = sorted(_REQUIRED_DIFFICULTY_FIELDS - set(difficulty))
        if missing_difficulty:
            errors.append(f"difficulty_contract is missing fields: {', '.join(missing_difficulty)}")
        for key in ("concepts_used", "required_reasoning_steps", "weak_baselines", "shortcut_defenses"):
            if key in difficulty and not isinstance(difficulty[key], list):
                errors.append(f"difficulty_contract.{key} must be an array")

    expert_hours = difficulty.get("expert_time_estimate_hours")
    expert_hours_valid = (
        not isinstance(expert_hours, bool)
        and isinstance(expert_hours, (int, float))
        and 0.25 <= float(expert_hours) <= 8.0
        and float(expert_hours * 3600).is_integer()
    )
    if not expert_hours_valid:
        errors.append(
            "difficulty_contract.expert_time_estimate_hours must produce an integral "
            "timeout in seconds and be in [0.25, 8]"
        )
    expected_agent_timeout = int(float(expert_hours) * 3600) if expert_hours_valid else None

    runtime_raw = draft.get("runtime_spec")
    errors.extend(validate_runtime_spec(runtime_raw, agent_timeout_sec=expected_agent_timeout))
    runtime: dict[str, Any] = dict(runtime_raw) if isinstance(runtime_raw, dict) else {}
    if expected_agent_timeout is not None and "agent_timeout_sec" in runtime:
        stored = runtime["agent_timeout_sec"]
        if not isinstance(stored, bool) and isinstance(stored, int) and stored != expected_agent_timeout:
            errors.append(
                "runtime_spec.agent_timeout_sec must equal difficulty_contract.expert_time_estimate_hours * 3600"
            )
    if expected_agent_timeout is not None:
        runtime["agent_timeout_sec"] = expected_agent_timeout

    hidden_generalization = difficulty.get("hidden_generalization")
    if not _nonempty_str(hidden_generalization):
        errors.append("difficulty_contract.hidden_generalization must be non-empty")

    if selected_candidate is not None:
        estimated = selected_candidate.get("estimated_difficulty")
        estimated = estimated if isinstance(estimated, int) and not isinstance(estimated, bool) else 3
        if difficulty.get("selected_candidate_id") != selected_candidate.get("candidate_id"):
            errors.append("difficulty_contract must identify the selected candidate")
        candidate_concepts = selected_candidate.get("concepts_used")
        contract_concepts = difficulty.get("concepts_used")
        if isinstance(candidate_concepts, list) and isinstance(contract_concepts, list):
            missing_concepts = sorted(set(candidate_concepts) - set(contract_concepts))
            if missing_concepts:
                errors.append(
                    "difficulty_contract dropped selected candidate concepts: "
                    + ", ".join(str(concept) for concept in missing_concepts)
                )
        timeout_bounds = (900, 7_200) if estimated <= 2 else (7_200, 18_000) if estimated == 3 else (18_000, 28_800)
        timeout_value = runtime.get("agent_timeout_sec")
        if (
            not isinstance(timeout_value, int)
            or isinstance(timeout_value, bool)
            or not timeout_bounds[0] <= timeout_value <= timeout_bounds[1]
        ):
            warnings.append("Omega difficulty estimate differs from suggested time band " + str(timeout_bounds))

    topology = draft.get("evaluation_topology")
    errors.extend(validate_evaluation_topology(topology, runtime))

    files: dict[str, str] = {}
    try:
        files = text_files(
            draft.get("files"),
            stage="task spec",
            max_files=settings.limits.max_generated_files,
            max_chars=settings.limits.max_generated_file_chars,
        )
    except ValueError as error:
        errors.append(str(error))
    if set(files) != REQUIRED_SPEC_FILES:
        errors.append("Task-spec files must contain exactly instruction.md and task.toml")

    for path, content in files.items():
        if TBS_CANARY not in content:
            errors.append(f"{path} must contain the Terminal-Bench-Science canary comment")

    instruction = files.get("instruction.md", "")
    if "task.toml" in files:
        errors.extend(validate_task_toml(files["task.toml"], runtime))

    if expected_agent_timeout is not None and "instruction.md" in files:
        canonical_suffix = (
            f"\n\nYou have {expected_agent_timeout} seconds to complete this task. "
            "Do not cheat by using online solutions or hints specific to this task.\n"
        )
        if not instruction.endswith(canonical_suffix):
            errors.append("instruction.md must end with the canonical timeout/anti-cheat sentence")

    if (
        isinstance(topology, dict)
        and topology.get("mode") == "submitted_program_on_hidden_cases"
        and "instruction.md" in files
        and topology.get("program_artifact") not in instruction
    ):
        errors.append("instruction.md must name evaluation_topology.program_artifact")

    if "instruction.md" in files:
        errors.extend(validate_instruction_absolute_paths(instruction))

    return _check_result(errors, warnings)


# --------------------------------------------------------------------------- #
# stage 6: implementation
# --------------------------------------------------------------------------- #
def check_implementation_draft(
    draft: dict,
    spec_draft: dict,
    settings: Settings,
) -> CheckResult:
    """Omega ``_validate_implementation_draft``.

    The caller has already applied ``add_canary`` and
    ``fix_reward_initialization``; this function validates and never rewrites
    files. Omega's ``authoring_warnings`` (runtime-permission heuristics) are
    returned in ``warnings`` instead of being written back onto the draft.
    """
    if not isinstance(draft, dict):
        return _check_result(["Implementation model output must be an object"])

    errors: list[str] = []
    warnings: list[str] = []

    blockers = draft.get("implementation_blockers")
    if not isinstance(blockers, list):
        errors.append("implementation_blockers must be an array")
    elif blockers:
        errors.append(f"Implementation model reported blockers: {blockers}")
    if not _nonempty_str(draft.get("implementation_summary")):
        errors.append("implementation_summary must be a non-empty string")

    files: dict[str, str] = {}
    try:
        files = text_files(
            draft.get("files"),
            stage="implementation",
            max_files=settings.limits.max_generated_files,
            max_chars=settings.limits.max_generated_file_chars,
        )
    except ValueError as error:
        errors.append(str(error))

    missing = sorted(REQUIRED_IMPLEMENTATION_FILES - set(files))
    if missing:
        errors.append(f"Implementation is missing files: {', '.join(missing)}")
    unsupported = sorted(
        path
        for path in files
        if path != "generation_data.py"
        and not path.startswith(("solution/", "tests/", "environment/starter/", "generation/support/"))
    )
    if unsupported:
        errors.append(f"Implementation returned unsupported paths: {', '.join(unsupported)}")
    forbidden = sorted(
        path
        for path in files
        if path.endswith("Dockerfile")
        or path
        in {
            "instruction.md",
            "task.toml",
            "execution.md",
            "tests/support/omega_runtime.py",
            "tests/omega_test_body.sh",
            "tests/support/omega_reports.py",
        }
    )
    if forbidden:
        errors.append(f"Implementation must not replace contract/scaffold files: {', '.join(forbidden)}")
    missing_canary = sorted(
        path
        for path, content in files.items()
        if path.startswith(("solution/", "tests/")) and TBS_CANARY not in content
    )
    if missing_canary:
        errors.append("Terminal-Bench-Science canary comment is missing from: " + ", ".join(missing_canary))

    solution_sources = [path for path in files if path.startswith("solution/") and path != "solution/solve.sh"]
    if not solution_sources:
        errors.append("Implementation must include at least one solution source file")

    solve = files.get("solution/solve.sh")
    if isinstance(solve, str):
        if not solve.lstrip().startswith("#!"):
            errors.append("solution/solve.sh must contain a shebang")
        if not _SHELL_SET_E.search(solve):
            errors.append("solution/solve.sh must enable shell exit-on-error")
    test_sh = files.get("tests/test.sh")
    if isinstance(test_sh, str):
        if not test_sh.lstrip().startswith("#!"):
            errors.append("tests/test.sh must contain a shebang")
        if "/logs/verifier/reward.txt" not in test_sh:
            errors.append("tests/test.sh must write /logs/verifier/reward.txt")

    generator = files.get("generation_data.py")
    if isinstance(generator, str):
        errors.extend(validate_python_source(generator, "generation_data.py"))
        flags = ("--public-output-dir", "--test-output-dir", "--public-seed", "--hidden-seeds")
        if any(flag not in generator for flag in flags):
            errors.append(f"generation_data.py must implement {', '.join(flags)}")

    for path, source in files.items():
        if path.endswith(".py") and path != "generation_data.py":
            errors.extend(validate_python_source(source, path))

    if "tests/test_outputs.py" in files:
        errors.extend(validate_python_verifier(files["tests/test_outputs.py"]))

    verifier_files = {
        path: source for path, source in files.items() if path.startswith("tests/") and path.endswith((".py", ".sh"))
    }
    warnings.extend(_runtime_permission_warnings(verifier_files))

    solution_references = sorted(
        path for path, source in verifier_files.items() if _VERIFIER_SOLUTION_REFERENCE.search(source)
    )
    if solution_references:
        errors.append(
            "Separate verifier files must not read or execute /solution; reference truth and "
            "calibration must be verifier-owned: " + ", ".join(solution_references)
        )

    manual_demotions: list[str] = []
    for path, source in verifier_files.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"setuid", "setgid"}
            for node in ast.walk(tree)
        ):
            manual_demotions.append(path)
    if manual_demotions:
        errors.append(
            "Use Omega private_case/run_submitted instead of manual setuid/setgid: " + ", ".join(manual_demotions)
        )

    topology = spec_draft.get("evaluation_topology") if isinstance(spec_draft, dict) else None
    if isinstance(topology, dict):
        errors.extend(validate_implementation_topology(files, topology))
    else:
        warnings.append("evaluation_topology_missing:implementation topology not checked")

    return _check_result(errors, warnings)


def validate_implementation_check(
    draft: dict,
    spec_draft: dict,
    settings: Settings,
) -> CheckResult:
    """Omega's implementation-check stage (``_check_implementation``).

    Validates the corrections map shape and re-runs
    :func:`check_implementation_draft` on the returned files. Any
    ``implementation_blockers`` entry is an error (Omega only accepts blockers
    for an irreconcilable contract).
    """
    if not isinstance(draft, dict):
        return _check_result(["Implementation-check output must be an object"])

    errors: list[str] = []
    warnings: list[str] = []

    corrections = draft.get("corrections")
    if corrections is None:
        warnings.append("corrections_missing")
    elif not isinstance(corrections, dict):
        errors.append("corrections must be an object with added_files/modified_files/removed_files")
    else:
        expected_keys = {"added_files", "modified_files", "removed_files"}
        missing = sorted(expected_keys - set(corrections))
        extra = sorted(set(corrections) - expected_keys)
        if missing:
            errors.append("corrections is missing fields: " + ", ".join(missing))
        if extra:
            errors.append("corrections contains unsupported fields: " + ", ".join(extra))
        for key in sorted(expected_keys & set(corrections)):
            value = corrections[key]
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                errors.append(f"corrections.{key} must be a string array")

    result = check_implementation_draft(draft, spec_draft, settings)
    errors.extend(result.errors)
    warnings.extend(result.warnings)
    return _check_result(errors, warnings)


# --------------------------------------------------------------------------- #
# lower-level helpers (also used by the static-check layer)
# --------------------------------------------------------------------------- #
def validate_task_toml(task_toml: str, runtime: dict) -> list[str]:
    """Omega ``_validate_task_toml``. Returns error strings (``[]`` == ok)."""
    errors: list[str] = []
    try:
        parsed = tomllib.loads(task_toml)
    except Exception as error:
        return [f"task.toml does not parse: {error}"]
    runtime_value: dict[str, Any] = runtime if isinstance(runtime, dict) else {}

    if parsed.get("schema_version") != "1.0":
        errors.append('task.toml schema_version must be "1.0"')
    if parsed.get("artifacts") != runtime_value.get("artifact_paths"):
        errors.append("task.toml artifacts must exactly equal runtime_spec.artifact_paths")

    task = parsed.get("task")
    if not isinstance(task, dict):
        errors.append("task.toml must contain [task]")
    else:
        name = str(task.get("name") or "")
        if _TASK_TOML_NAME.fullmatch(name) is None:
            errors.append("task.toml [task].name must use terminal-bench-science/<kebab-case>")
        if not _nonempty_str(task.get("description")):
            errors.append("task.toml [task].description must be a non-empty string")
        authors = task.get("authors")
        if (
            not isinstance(authors, list)
            or not authors
            or any(not isinstance(author, dict) or not _nonempty_str(author.get("name")) for author in authors)
        ):
            errors.append("task.toml [task].authors must be an array of tables with a non-empty name")
        keywords = task.get("keywords")
        if (
            not isinstance(keywords, list)
            or not keywords
            or any(not isinstance(keyword, str) or not keyword.strip() for keyword in keywords)
        ):
            errors.append("task.toml [task].keywords must be a non-empty string array")

    metadata = parsed.get("metadata")
    expert_hours: float | None = None
    if not isinstance(metadata, dict):
        errors.append("task.toml must contain [metadata]")
    else:
        missing_metadata = sorted(
            key for key in _REQUIRED_TASK_TOML_METADATA if key not in metadata or metadata[key] in (None, "", [])
        )
        if missing_metadata:
            errors.append(f"task.toml [metadata] is missing non-empty fields: {', '.join(missing_metadata)}")
        raw_hours = metadata.get("expert_time_estimate_hours")
        if (
            isinstance(raw_hours, bool)
            or not isinstance(raw_hours, (int, float))
            or not 0.25 <= float(raw_hours) <= 8.0
        ):
            errors.append("task.toml [metadata].expert_time_estimate_hours must be in [0.25, 8]")
        else:
            expert_hours = float(raw_hours)
    if expert_hours is not None and not _numeric_equal(expert_hours * 3600.0, runtime_value.get("agent_timeout_sec")):
        errors.append(
            "runtime_spec.agent_timeout_sec must equal task.toml [metadata].expert_time_estimate_hours * 3600"
        )

    verifier = parsed.get("verifier")
    if not isinstance(verifier, dict):
        errors.append('task.toml [verifier].environment_mode must be "separate"')
    else:
        if verifier.get("environment_mode") != "separate":
            errors.append('task.toml [verifier].environment_mode must be "separate"')
        if not _numeric_equal(verifier.get("timeout_sec", -1), runtime_value.get("verifier_timeout_sec", -1)):
            errors.append("task.toml verifier timeout must equal runtime_spec.verifier_timeout_sec")
        verifier_environment = verifier.get("environment")
        if not isinstance(verifier_environment, dict) or verifier_environment.get("network_mode") != "no-network":
            errors.append('task.toml [verifier.environment].network_mode must be "no-network"')

    agent = parsed.get("agent")
    if not isinstance(agent, dict) or not _numeric_equal(
        agent.get("timeout_sec", -1), runtime_value.get("agent_timeout_sec", -1)
    ):
        errors.append("task.toml [agent].timeout_sec must equal runtime_spec.agent_timeout_sec")

    environment = parsed.get("environment")
    if not isinstance(environment, dict) or environment.get("network_mode") != "public":
        errors.append('task.toml [environment].network_mode must be "public"')
    if isinstance(environment, dict):
        numeric_contract: dict[str, Any] = {
            "build_timeout_sec": runtime_value.get("build_timeout_sec"),
            "cpus": runtime_value.get("cpus"),
            "memory_mb": runtime_value.get("memory_mb"),
            "storage_mb": runtime_value.get("storage_mb"),
            "gpus": 0,
        }
        for key, expected in numeric_contract.items():
            if not _numeric_equal(environment.get(key, -1), expected):
                errors.append(f"task.toml [environment].{key} must equal runtime_spec value {expected}")
        allowed_environment = set(numeric_contract) | {"network_mode", "validate_env", "gpu_types"}
        unsupported = sorted(set(environment) - allowed_environment)
        if unsupported:
            errors.append("task.toml [environment] contains Omega-only/unsupported fields: " + ", ".join(unsupported))
    return errors


def validate_runtime_spec(runtime: dict, *, agent_timeout_sec: int | None = None) -> list[str]:
    """Omega ``_validate_runtime_spec``. Returns error strings (``[]`` == ok).

    When ``agent_timeout_sec`` is supplied, Omega owns the field and skips its
    range check (the difficulty contract already fixed the value).
    """
    if not isinstance(runtime, dict):
        return ["Strong model output must include a runtime_spec object"]

    errors: list[str] = []
    missing = sorted(_REQUIRED_RUNTIME_FIELDS - set(runtime))
    if missing:
        errors.append(f"runtime_spec is missing fields: {', '.join(missing)}")
    if "profile" in runtime and runtime["profile"] != "python_scientific":
        errors.append("runtime_spec.profile must be python_scientific")
    if "python_version" in runtime and str(runtime["python_version"]) not in PYTHON_VERSIONS:
        errors.append("runtime_spec.python_version must be 3.11, 3.12, or 3.13")

    if "dependencies" in runtime:
        dependencies = runtime["dependencies"]
        if not isinstance(dependencies, list) or any(
            not isinstance(item, str) or PINNED_REQUIREMENT.fullmatch(item) is None for item in dependencies
        ):
            errors.append("runtime_spec.dependencies must contain only exact name==version pins")
        elif len(dependencies) != len({item.casefold() for item in dependencies}):
            errors.append("runtime_spec.dependencies contains duplicates")

    public_seed = runtime.get("public_seed")
    hidden_seeds = runtime.get("hidden_seeds")
    if "public_seed" in runtime and (isinstance(public_seed, bool) or not isinstance(public_seed, int)):
        errors.append("runtime_spec.public_seed must be an integer")
    if "hidden_seeds" in runtime:
        if (
            not isinstance(hidden_seeds, list)
            or len(hidden_seeds) < 2
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in hidden_seeds)
            or len(set(hidden_seeds)) != len(hidden_seeds)
            or public_seed in hidden_seeds
        ):
            errors.append("runtime_spec.hidden_seeds must contain at least two distinct non-public integer seeds")

    if "public_data_dir" in runtime:
        errors.extend(validate_absolute_runtime_path(str(runtime["public_data_dir"]), directory=True))

    if "artifact_paths" in runtime:
        artifacts = runtime["artifact_paths"]
        if not isinstance(artifacts, list) or not artifacts:
            errors.append("runtime_spec.artifact_paths must be an array")
        else:
            for artifact in artifacts:
                errors.extend(validate_absolute_runtime_path(str(artifact), directory=False))
            try:
                duplicated = len(artifacts) != len(set(artifacts))
            except TypeError:
                duplicated = True
            if duplicated:
                errors.append("runtime_spec.artifact_paths contains duplicates")

    for name, (lower, upper) in _RUNTIME_LIMITS.items():
        if name not in runtime:
            continue
        item = runtime[name]
        if isinstance(item, bool) or not isinstance(item, int) or not lower <= item <= upper:
            errors.append(f"runtime_spec.{name} must be an integer in [{lower}, {upper}]")
    if "gpus" in runtime and runtime["gpus"] != 0:
        errors.append("python_scientific currently requires runtime_spec.gpus=0")

    if agent_timeout_sec is None:
        supplied_timeout = runtime.get("agent_timeout_sec")
        if (
            isinstance(supplied_timeout, bool)
            or not isinstance(supplied_timeout, int)
            or not 1 <= supplied_timeout <= 28_800
        ):
            errors.append("runtime_spec.agent_timeout_sec must be an integer in [1, 28800]")
    return errors


def validate_evaluation_topology(topology: dict, runtime: dict) -> list[str]:
    """Omega ``_validate_evaluation_topology``. Returns error strings."""
    if not isinstance(topology, dict):
        return ["Spec output must include an evaluation_topology object"]

    errors: list[str] = []
    mode = topology.get("mode")
    if mode not in EVALUATION_MODES:
        errors.append(f"evaluation_topology.mode must be one of: {', '.join(sorted(EVALUATION_MODES))}")
    missing = sorted(_REQUIRED_TOPOLOGY_FIELDS - set(topology))
    if missing:
        errors.append(f"evaluation_topology is missing fields: {', '.join(missing)}")

    if "hidden_case_glob" in topology:
        hidden_glob = str(topology["hidden_case_glob"])
        if not hidden_glob.startswith("/tests/data/") or ".." in PurePosixPath(hidden_glob).parts:
            errors.append("evaluation_topology.hidden_case_glob must be under /tests/data/")
    if "agent_entrypoint" in topology:
        entrypoint = topology["agent_entrypoint"]
        if (
            not isinstance(entrypoint, list)
            or not entrypoint
            or any(not isinstance(item, str) or not item for item in entrypoint)
        ):
            errors.append("evaluation_topology.agent_entrypoint must be a non-empty string array")

    runtime_artifacts = runtime.get("artifact_paths") if isinstance(runtime, dict) else None
    artifact = topology.get("program_artifact")
    if mode == "submitted_program_on_hidden_cases":
        if not isinstance(artifact, str):
            errors.append("submitted_program_on_hidden_cases requires program_artifact")
        else:
            errors.extend(validate_absolute_runtime_path(artifact, directory=False))
            if not isinstance(runtime_artifacts, list) or artifact not in runtime_artifacts:
                errors.append("program_artifact must be one of runtime_spec.artifact_paths")
            entrypoint = topology.get("agent_entrypoint")
            if not isinstance(entrypoint, list) or artifact not in entrypoint:
                errors.append("agent_entrypoint must execute program_artifact")
            if runtime_artifacts != [artifact]:
                errors.append("program-on-hidden-cases mode must upload only the reusable program artifact")
    elif artifact is not None:
        errors.append("fixed_artifacts_against_hidden_reference requires program_artifact=null")
    return errors


def validate_implementation_topology(files: dict[str, str], topology: dict) -> list[str]:
    """Omega ``_validate_implementation_topology``. Returns error strings."""
    errors: list[str] = []
    if not isinstance(topology, dict) or topology.get("mode") != "submitted_program_on_hidden_cases":
        return errors
    if not isinstance(topology.get("program_artifact"), str):
        return errors

    artifact = str(topology["program_artifact"])
    relative = artifact.lstrip("/")
    for prefix in ("app/", "root/"):
        if relative.startswith(prefix):
            relative = relative[len(prefix) :]
            break
    reference_path = f"solution/{relative}"
    starter_path = f"environment/starter/{relative}"
    missing = [path for path in (reference_path, starter_path) if path not in files]
    if missing:
        errors.append("Program-on-hidden-cases implementation is missing: " + ", ".join(missing))

    solve = files.get("solution/solve.sh")
    if isinstance(solve, str) and (artifact not in solve or f"/solution/{relative}" not in solve):
        errors.append("solution/solve.sh must install the reference program from /solution into program_artifact")

    verifier_source = "\n".join(
        content for path, content in files.items() if path.startswith("tests/") and path.endswith((".py", ".sh"))
    )
    hidden_glob = topology.get("hidden_case_glob")
    hidden_root = str(PurePosixPath(str(hidden_glob)).parent) if hidden_glob is not None else ""
    resolved: set[str] = set()
    for path, content in files.items():
        if path.startswith("tests/") and path.endswith(".py"):
            resolved |= _resolved_paths(content)
    verifier_source += "\n" + "\n".join(sorted(resolved))
    if "run_submitted" in verifier_source:
        verifier_source += "\nsubprocess"
    required_tokens = (artifact, hidden_root, "subprocess")
    missing_tokens = [token for token in required_tokens if token not in verifier_source]
    if missing_tokens:
        errors.append(
            "Hidden-case verifier must execute the submitted program over private cases; missing tokens: "
            + ", ".join(missing_tokens)
        )
    return errors


def validate_absolute_runtime_path(value: str, *, directory: bool) -> list[str]:
    """Omega ``_validate_absolute_runtime_path``. Returns error strings."""
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or len(path.parts) < 3
        or path.parts[1] not in {"app", "root"}
        or any(part in {"", ".", ".."} for part in path.parts)
        or _RUNTIME_PATH.fullmatch(value) is None
    ):
        kind = "directory" if directory else "artifact"
        return [f"Unsafe runtime {kind} path: {value!r}"]
    return []


def validate_instruction_absolute_paths(instruction: str) -> list[str]:
    """Omega ``_validate_instruction_absolute_paths``. Returns error strings."""
    offenders = sorted(set(_INSTRUCTION_RELATIVE_PATH.findall(instruction)))
    if offenders:
        return ["instruction.md uses relative runtime paths; use absolute /app or /root paths: " + ", ".join(offenders)]
    return []


def validate_python_source(source: str, name: str) -> list[str]:
    """Omega ``_validate_python_source``. Returns error strings."""
    try:
        ast.parse(source, filename=name)
    except SyntaxError as error:
        return [f"{name} is not valid Python: {error}"]
    return []


def validate_python_verifier(source: str) -> list[str]:
    """Omega ``_validate_python_verifier``. Returns error strings."""
    try:
        tree = ast.parse(source, filename="tests/test_outputs.py")
    except SyntaxError as error:
        return [f"tests/test_outputs.py is not valid Python: {error}"]
    assertions = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    assertion_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.startswith("assert")
    ]
    guards = [
        node.test
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(isinstance(child, ast.Raise) for statement in node.body for child in ast.walk(statement))
    ]
    if not assertions and not assertion_calls and not guards:
        return ["tests/test_outputs.py must contain executable assertions"]
    meaningful_nodes = (ast.Name, ast.Call, ast.Attribute, ast.Subscript)
    expressions = [node.test for node in assertions] + assertion_calls + guards
    if not any(
        any(isinstance(child, meaningful_nodes) for child in ast.walk(expression)) for expression in expressions
    ):
        return ["tests/test_outputs.py assertions must inspect computed output, not only literals"]
    return []


def safe_relative_path(value: str, *, stage: str) -> str:
    """Omega ``_safe_relative_path``.

    Raises ``ValueError`` on traversal, absolute paths or backslashes; this is
    called from parsers where the exception is the intended repair signal.
    """
    raw = str(value or "").strip()
    if not raw or "\\" in raw:
        raise ValueError(f"Unsafe generated path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe generated path: {raw!r}")
    return path.as_posix()


def text_files(raw: object, *, stage: str, max_files: int, max_chars: int) -> dict[str, str]:
    """Omega ``_text_files`` with explicit budgets.

    Raises ``ValueError`` (the Data Designer repair signal) for malformed or
    over-budget payloads. Duplicate, empty and oversized files are rejected.
    """
    if isinstance(raw, dict):
        entries: list[tuple[Any, Any]] = list(raw.items())
    elif isinstance(raw, list):
        entries = [(item.get("path"), item.get("content")) for item in raw if isinstance(item, dict)]
    else:
        raise ValueError(f"{stage} output must include files as an object or path/content list")
    if not entries or len(entries) > max_files:
        raise ValueError(f"Generated file count must be in [1, {max_files}]")
    files: dict[str, str] = {}
    for raw_path, raw_content in entries:
        relative = safe_relative_path(raw_path, stage=stage)
        if relative in files:
            raise ValueError(f"Duplicate generated file path: {relative}")
        if not isinstance(raw_content, str) or not raw_content.strip():
            raise ValueError(f"Generated file is empty or non-text: {relative}")
        if len(raw_content) > max_chars:
            raise ValueError(f"Generated file exceeds {max_chars} characters: {relative}")
        files[relative] = raw_content
    return files
