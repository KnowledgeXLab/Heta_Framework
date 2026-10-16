# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom column generators: the Data Designer form of Omega's pipeline steps.

Design rules:

* ``LLMStructuredColumnConfig`` is deliberately NOT used: it validates against JSON
  Schema only, never instantiates pydantic, and its own correction default is 0 —
  strictly weaker than Omega. Every LLM step here is a custom column that calls the
  model facade with ``parser=<domain validator>`` and ``max_correction_steps``, which
  reproduces Omega's in-step validation retry (assistant draft + validation error fed
  back) while still going through Data Designer's throttling and tracing.
* Deterministic steps are plain Python columns. Every domain rule lives in
  ``omegadesigner.domain.checks`` so the same rule is used for gating and for repair.
* Prompts are the vendored Omega prompt files; the user payload keeps Omega's exact
  JSON shape (``indent=2, sort_keys=True``) with the JSON Schema appended, which is
  what Data Designer's built-in structured columns would have injected for us.
"""

from __future__ import annotations

import functools
import json
import os
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import data_designer.config as dd
from data_designer.engine.models.parsers.errors import ParserException
from pydantic import ValidationError

from heta_framework.generation.graph._native.config import DIFFICULTY_TIER_BANDS, Settings
from heta_framework.generation.graph._native.domain import canonicalize, checks, graph_screen, scaffold, static_checks
from heta_framework.generation.graph._native.domain.graph_screen import ScreenResult
from heta_framework.generation.graph._native.domain.schemas import (
    KEBAB,
    REQUIRED_EVIDENCE_INVENTORY,
    REQUIRED_FEASIBILITY_CHECKS,
    REQUIRED_IMPLEMENTATION_FILES,
    REQUIRED_PACKAGE_FILES,
    REQUIRED_TASK_SPEC_FIELDS,
    TBS_CANARY,
    CandidateList,
    CandidateReview,
    CheckResult,
    ImplementationDraft,
    SemanticReview,
    SpecDraft,
    schema_text,
)
from heta_framework.generation.graph._native.io_utils import sha256_text

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
PROMPTS_DIR = ASSETS_DIR / "prompts"
TBS_REFERENCE_DIR = ASSETS_DIR / "tbs_reference"
CHECK_SCRIPTS_DIR = ASSETS_DIR / "checks"

REPAIR_INSTRUCTION = (
    "Return a complete corrected JSON object for the same stage. Preserve the binding "
    "scientific contract and fix every reported structural error."
)

# The ``@custom_column_generator`` decorator is evaluated at import time, so the model
# aliases a column may use must be literals here. ``cli.py`` builds ModelConfigs with
# exactly these aliases; ``omegadesigner.config.ModelAliases`` documents the contract.
CHEAP_ALIAS = "cheap"
STRONG_ALIAS = "strong"
STRONG_CREATIVE_ALIAS = "strong_creative"
STRONG_DETERMINISTIC_ALIAS = "strong_deterministic"
STRONG_NO_RETRY_ALIAS = "strong_no_retry"
MODEL_ALIASES: tuple[str, ...] = (
    CHEAP_ALIAS,
    STRONG_ALIAS,
    STRONG_CREATIVE_ALIAS,
    STRONG_DETERMINISTIC_ALIAS,
    STRONG_NO_RETRY_ALIAS,
)

# Payload keys are Omega's; keep them verbatim so prompts behave identically.
CANDIDATE_GENERATION_PROMPT = "paper_task_candidate_generation.md"
CANDIDATE_REVIEW_PROMPT = "paper_task_candidate_review.md"
SEMANTIC_REVIEW_PROMPT = "paper_task_semantic_review.md"
SPEC_GENERATION_PROMPT = "paper_task_spec_generation.md"
IMPLEMENTATION_GENERATION_PROMPT = "paper_task_implementation_generation.md"
IMPLEMENTATION_CHECK_PROMPT = "paper_task_implementation_check.md"


# --------------------------------------------------------------------------------------
# prompt / payload helpers
# --------------------------------------------------------------------------------------


def prompt_text(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def user_payload(payload: dict[str, Any], model: Any | None = None) -> str:
    """Omega's user message plus the JSON Schema Data Designer would have injected."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if model is not None:
        text += (
            "\n\nYour response must be in JSON format inside a ```json code fence."
            "\nThe JSON format is given as a JSON Schema description within <response_schema> markup tags."
            "\nReturn an INSTANCE of that schema — a JSON object holding the described fields."
            "\nDo NOT return the schema itself, and do not include $schema, title, definitions or"
            " properties keys at the top level."
            f"\n\n<response_schema>\n{schema_text(model)}\n</response_schema>"
        )
    return text


def _extract_json(text: str) -> Any:
    """Extract the SINGLE top-level JSON object from a model response.

    Two failure modes drove this implementation:

    * drafts legitimately contain markdown fences inside ``instruction.md`` and file
      contents, so every fenced block is tried rather than a greedy slice;
    * a previous version recovered "the object with the most keys" when no block parsed
      cleanly, which silently returned a nested object (``runtime_spec`` has more keys
      than the spec root) and surfaced as a misleading "package_name: Field required".

    So recovery never mines nested objects: either a whole block parses, or the first
    ``{`` of a block decodes with trailing prose after it, or this raises a parse error
    that the repair loop can act on.
    """
    stripped = text.strip()
    fenced = [block.strip() for block in re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)]
    candidates = [*fenced, stripped]

    parsed_objects: list[dict] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed_objects.append(parsed)
    if parsed_objects:
        # Several whole blocks parsed: the payload is the structurally richest one.
        return max(parsed_objects, key=len)

    decoder = json.JSONDecoder()
    decoded: list[dict] = []
    for candidate in candidates:
        match = re.search(r"\{", candidate)
        if not match:
            continue
        try:
            obj, _end = decoder.raw_decode(candidate[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            decoded.append(obj)
    if decoded:
        return max(decoded, key=len)

    raise ValueError(
        "Response was not a single valid JSON object; it may be truncated or contain "
        f"unescaped characters (head: {stripped[:160]!r})"
    )


def validated_parser(
    model: type,
    tag: str,
    *,
    post: Callable[[dict[str, Any]], CheckResult | list[str]] | None = None,
    capture: list[str] | None = None,
) -> Callable[[str], dict[str, Any]]:
    """JSON -> pydantic (``extra='forbid'``) -> domain rules, raising ``ParserException``.

    The exception type is load-bearing. ``ModelFacade.agenerate`` only routes
    **``ParserException``** back to the model as a correction round
    (``engine/models/facade.py``: ``except ParserException``); any other exception aborts
    after the first attempt. Raising ``ValueError`` here silently disabled every repair
    round this project relies on (``max_correction_steps`` became dead configuration), so
    a one-shot contract miss was recorded as a permanent failure.
    """

    def parse(text: str) -> dict[str, Any]:
        if capture is not None:
            # Keep the last raw response per column call: on rejection it is the only way to
            # see what the model actually produced.
            capture.clear()
            capture.append(str(text))
        snippet = " ".join(str(text).split())[:280]
        try:
            obj = _extract_json(text)
        except ValueError as error:
            raise ParserException(f"{tag}: response did not contain a JSON object (got: {snippet!r})") from error
        if not isinstance(obj, dict):
            raise ParserException(f"{tag}: response must be a JSON object (got: {snippet!r})")
        try:
            validated = model.model_validate(obj)
        except ValidationError as error:
            details = "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors()[:6]
            )
            # "validation error" is the wording ModelFacade classifies as schema_validation.
            raise ParserException(
                f"{tag} schema validation error: {details} "
                f"(parsed top-level keys: {sorted(obj)[:14]}; response head: {snippet[:160]!r})"
            ) from error
        data = validated.model_dump(mode="json")
        if post is not None:
            outcome = post(data)
            errors = outcome.errors if isinstance(outcome, CheckResult) else list(outcome)
            if errors:
                raise ParserException(f"{tag} contract error: " + " | ".join(errors[:6]))
        return data

    return parse


def _is_retryable(error: BaseException) -> bool:
    """Transient provider errors must keep propagating so Data Designer retries them."""
    try:
        from data_designer.engine.models.errors import RETRYABLE_MODEL_ERRORS
    except Exception:  # pragma: no cover - defensive
        return False
    return isinstance(error, RETRYABLE_MODEL_ERRORS)


def _record_failure(
    row: dict,
    name: str,
    error: BaseException,
    *,
    raw_response: str = "",
) -> dict:
    """Record a permanent step failure as a column value instead of raising.

    Omega isolates a bad paper (``failed_stage`` + ``error``) and keeps the batch
    running. A raised exception in a Data Designer column instead counts toward
    ``shutdown_error_rate`` and can abort the whole run, so permanent failures are
    recorded and the row is gated out at the next stage boundary.

    The rejected draft is discarded (it must not travel downstream) but the raw response
    is kept, bounded, under ``<name>_failure``: without it a contract rejection cannot be
    diagnosed after the fact.
    """
    row[name] = ""
    _record_step_failure(row, name, error, raw_response=raw_response)
    return row


def _record_step_failure(row: dict, prefix: str, error: BaseException, *, raw_response: str = "") -> None:
    """Record a failure of one sub-step, which need not own a column of its own.

    A generator that folds two model calls (draft, then check-and-correct) reports each
    call's outcome separately, so a rejected draft is still attributable to the call that
    produced it.
    """
    message = f"{type(error).__name__}: {error}"
    row[f"{prefix}_ok"] = False
    row[f"{prefix}_errors"] = [message[:2000]]
    row[f"{prefix}_failure"] = json.dumps(
        {"error": message[:2000], "raw_response": " ".join(raw_response.split())[:4000]},
        ensure_ascii=False,
    )


def _record_skipped_step(row: dict, prefix: str, reason: str) -> None:
    """Mark a sub-step that never ran because the step feeding it failed.

    Data Designer requires every declared ``side_effect_columns`` entry to be written on
    every path, including paths where the generator bailed out early.
    """
    row[f"{prefix}_ok"] = False
    row[f"{prefix}_errors"] = [reason]
    row[f"{prefix}_failure"] = ""


def _loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


# --------------------------------------------------------------------------------------
# stage 1 -- structural screen (deterministic)
# --------------------------------------------------------------------------------------


@dd.custom_column_generator(
    required_columns=["kg_path"],
    # The largest-component graph is NOT written: every consumer re-derives it from
    # ``kg_path`` (``read_kg_screen``), and it was 53 KB/row -- the single largest column in
    # the pipeline, written only to be projected away at the next boundary.
    side_effect_columns=["kg_passed", "kg_failed_checks", "kg_metrics"],
)
def kg_screen(row: dict, generator_params: Settings) -> dict:
    result = graph_screen.assess_file(row["kg_path"], thresholds=generator_params.screen)
    assessment = result.assessment
    row["kg_screen"] = json.dumps(assessment, ensure_ascii=False, sort_keys=True)
    row["kg_passed"] = bool(assessment["passed"])
    row["kg_failed_checks"] = list(assessment["failed_checks"])
    row["kg_metrics"] = assessment["metrics"]
    return row


@functools.lru_cache(maxsize=32)
def _assess_cached(kg_path: str, size: int, mtime_ns: int, thresholds_json: str) -> ScreenResult:
    """Bounded cache: only the current row group's KGs are ever resident (32 x ~50 KB)."""
    from heta_framework.generation.graph._native.config import ScreenThresholds

    thresholds = ScreenThresholds.model_validate_json(thresholds_json)
    return graph_screen.assess_file(kg_path, thresholds=thresholds)


def read_kg_screen(row: dict, settings: Settings) -> ScreenResult:
    """Re-derive the structural assessment from the KG asset instead of carrying it.

    The assessment and the largest-component graph are a *pure function of the KG file and
    the thresholds*, so unlike an LLM product they can be recomputed on demand. Carrying
    ``kg_lcf_json`` as a column cost 50 KB per row through three stages (and 3.16 KB for
    ``kg_screen``); reading it back costs ~14 ms, against 30-90 s model calls.
    """
    path = Path(str(row.get("kg_path") or ""))
    if not path.is_file():
        raise FileNotFoundError(f"KG asset not found: {path}")
    stat = path.stat()
    return _assess_cached(str(path), stat.st_size, stat.st_mtime_ns, settings.screen.model_dump_json())


def read_paper_text(row: dict) -> str:
    """Read the paper body from the asset path.

    The text is deliberately NOT a column: every stage's seed is the previous stage's
    dataset, so a text column would be copied once per stage (measured: 6 copies of the
    whole body by the time implementation runs). Only the path travels; the body is read
    where it is used, and each prompt truncates to its own budget.
    """
    path = Path(str(row.get("paper_text_path") or ""))
    if not path.is_file():
        # Reading late is only safe if a missing asset is loud: an empty body would reach
        # the model as "insufficient content" and be rejected for the wrong reason.
        raise FileNotFoundError(f"paper text asset not found: {path}")
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# stage 2 -- semantic review (cheap model)
# --------------------------------------------------------------------------------------


def _semantic_review_payload(row: dict, settings: Settings) -> dict[str, Any]:
    screen = read_kg_screen(row, settings)
    assessment = screen.assessment
    text = read_paper_text(row)
    component = json.dumps(screen.largest_component_graph, ensure_ascii=False, sort_keys=True)
    content_limit = settings.limits.cheap_max_content_chars
    component_limit = settings.limits.cheap_max_component_chars
    # Omega builds this payload in ``_cheap_prompt`` (pipeline.py:2288-2339).
    return {
        "goal": (
            "Extract task-relevant scientific evidence and decide feasibility only. Do not invent, "
            "select, or specify a concrete terminal task; a strong model performs ideation later."
        ),
        "structural_assessment": assessment,
        "paper_text": text[:content_limit],
        "paper_text_truncated": len(text) > content_limit,
        "largest_component_graph": component[:component_limit],
        "component_graph_truncated": len(component) > component_limit,
        "required_output": {
            "paper_summary": "string",
            "component_summary": "string",
            "suitable_for_task_generation": "boolean",
            "suitability_score": "number in [0,1]",
            "evidence_inventory": {key: [f"{key} evidence"] for key in sorted(REQUIRED_EVIDENCE_INVENTORY)},
            "computational_affordances": [
                {
                    "name": "descriptive capability, not a task proposal",
                    "evidence": ["paper/KG evidence supporting the capability"],
                    "inputs": ["scientifically available input type"],
                    "outputs": ["measurable output type"],
                    "generator_feasibility": "how deterministic local instances could exist",
                    "verifier_feasibility": "how outputs could be checked objectively",
                }
            ],
            "constraint_checks": {key: "boolean" for key in sorted(REQUIRED_FEASIBILITY_CHECKS)},
            "blockers": ["string"],
        },
    }


@dd.custom_column_generator(
    required_columns=["kg_path", "paper_text_path"],
    side_effect_columns=[
        "semantic_review_ok",
        "semantic_review_errors",
        "semantic_review_failure",
        "semantic_review_trace",
        "semantic_review_system_prompt_sha256",
        "semantic_review_user_payload_sha256",
    ],
    model_aliases=[CHEAP_ALIAS],
)
async def semantic_review(row: dict, generator_params: Settings, models: dict) -> dict:
    capture: list[str] = []
    from heta_framework.generation.graph._native.io_utils import canonical_json_sha256, sha256_text

    settings = generator_params
    alias = CHEAP_ALIAS
    system = prompt_text(SEMANTIC_REVIEW_PROMPT)
    parser = validated_parser(SemanticReview, "semantic_review", capture=capture)
    try:
        payload = _semantic_review_payload(row, settings)
        draft, trace = await models[alias].agenerate(
            prompt=user_payload(payload, SemanticReview),
            system_prompt=system,
            parser=parser,
            max_correction_steps=settings.repair.correction_steps,
            purpose="paper_task_semantic_review",
        )
    except Exception as error:  # noqa: BLE001 - see _record_failure
        if _is_retryable(error):
            raise
        # The payload may not have been built at all (e.g. a missing asset), so its hash is
        # recorded as unknown rather than referencing an unbound local.
        row["semantic_review_system_prompt_sha256"] = sha256_text(system)
        row["semantic_review_user_payload_sha256"] = ""
        row["semantic_review_trace"] = ""
        return _record_failure(row, "semantic_review", error, raw_response=capture[-1] if capture else "")
    row["semantic_review_ok"] = True
    row["semantic_review"] = json.dumps(draft, ensure_ascii=False, sort_keys=True)
    row["semantic_review_errors"] = []
    row["semantic_review_failure"] = ""
    # The trace was 114 KB/row (61% of the review stage) and nothing consumes it. Keep only
    # the final assistant message, bounded, and drop the column at the review boundary.
    last_message = getattr(trace[-1], "content", "") if trace else ""
    row["semantic_review_trace"] = str(last_message)[:4000]
    row["semantic_review_system_prompt_sha256"] = sha256_text(system)
    row["semantic_review_user_payload_sha256"] = canonical_json_sha256(payload)
    return row


@dd.custom_column_generator(
    required_columns=["semantic_review"],
    side_effect_columns=["review_suitable", "review_gate_json"],
)
def review_gate(row: dict, generator_params: Settings) -> dict:
    """Deterministic regate: never trust the model's own suitability claim."""
    settings = generator_params
    review = _loads(row["semantic_review"], {})
    declared = bool(review.get("suitable_for_task_generation"))
    try:
        score = float(review.get("suitability_score"))
    except (TypeError, ValueError):
        score = -1.0
    constraint_checks = review.get("constraint_checks") or {}
    failed = sorted(key for key, value in constraint_checks.items() if value is not True)
    checks_passed = bool(constraint_checks) and not failed
    suitable = declared and score >= settings.review.minimum_suitability_score and checks_passed
    row["review_gate"] = "accepted" if suitable else "rejected_semantic"
    row["review_suitable"] = suitable
    row["review_gate_json"] = json.dumps(
        {
            "model_declared_suitable": declared,
            "suitability_score": score,
            "minimum_suitability_score": settings.review.minimum_suitability_score,
            "constraint_gate_passed": checks_passed,
            "failed_constraint_checks": failed,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return row


# --------------------------------------------------------------------------------------
# stage 3 -- candidate generation (strong, 1 call -> N candidates)
# --------------------------------------------------------------------------------------


def _requested_tiers_payload(settings: Settings) -> dict[str, Any]:
    """Omega sends every requested tier (pipeline.py:382-388, 457-462), not just one.

    The candidate prompt uses the bands to spread candidates across tiers; the tier that a
    given candidate lands in decides which fan-out row it can be selected for.
    """
    tiers = list(settings.difficulty.tiers)
    return {
        "requested_difficulty_tiers": tiers,
        "difficulty_tier_bands": {tier: sorted(DIFFICULTY_TIER_BANDS[tier]) for tier in tiers},
    }


def row_tier(row: dict, settings: Settings) -> str:
    """The tier this row generates a variant for.

    The fan-out callback writes ``tier``, so downstream stages read it per row. The fallback
    keeps a single-variant payload honest when the column is absent (for example a caller
    running one stage in isolation): it is the primary requested tier.
    """
    tier = str(row.get("tier") or "")
    return tier if tier in DIFFICULTY_TIER_BANDS else settings.difficulty.tier


def _candidate_payload(row: dict, settings: Settings) -> dict[str, Any]:
    assessment = read_kg_screen(row, settings).assessment
    review = _loads(row["semantic_review"], {})
    text = read_paper_text(row)
    component = json.dumps(read_kg_screen(row, settings).largest_component_graph, ensure_ascii=False, sort_keys=True)
    content_limit = settings.limits.strong_candidate_max_content_chars
    component_limit = settings.limits.strong_candidate_max_component_chars
    # Omega: pipeline.py:365-389.
    return {
        "paper_identity": assessment.get("paper"),
        "structural_metrics": assessment.get("metrics"),
        "paper_summary": review.get("paper_summary"),
        "component_summary": review.get("component_summary"),
        "evidence_inventory": review.get("evidence_inventory"),
        "computational_affordances": review.get("computational_affordances"),
        "paper_evidence_excerpt": text[:content_limit],
        "paper_evidence_truncated": len(text) > content_limit,
        "largest_component_graph": component[:component_limit],
        "component_graph_truncated": len(component) > component_limit,
        "candidate_count": settings.candidates.candidate_count,
        "generation_guidance": settings.generation_guidance,
        **_requested_tiers_payload(settings),
    }


def _candidate_post(settings: Settings) -> Callable[[dict[str, Any]], CheckResult]:
    def post(draft: dict[str, Any]) -> CheckResult:
        return checks.check_candidate_list(draft, settings)

    return post


@dd.custom_column_generator(
    required_columns=["kg_path", "paper_text_path", "semantic_review"],
    side_effect_columns=["candidates_ok", "candidates_errors", "candidates_failure"],
    model_aliases=[STRONG_CREATIVE_ALIAS],
)
async def candidates(row: dict, generator_params: Settings, models: dict) -> dict:
    capture: list[str] = []
    settings = generator_params
    alias = STRONG_CREATIVE_ALIAS
    parser = validated_parser(CandidateList, "candidate_generation", post=_candidate_post(settings), capture=capture)
    try:
        payload = _candidate_payload(row, settings)
        draft, _trace = await models[alias].agenerate(
            prompt=user_payload(payload, CandidateList),
            system_prompt=prompt_text(CANDIDATE_GENERATION_PROMPT),
            parser=parser,
            max_correction_steps=settings.repair.correction_steps,
            purpose="paper_task_candidate_generation",
        )
    except Exception as error:  # noqa: BLE001
        if _is_retryable(error):
            raise
        return _record_failure(row, "candidates", error, raw_response=capture[-1] if capture else "")
    row["candidates_ok"] = True
    row["candidates_errors"] = []
    row["candidates_failure"] = ""
    # One column, not three: `candidates_json` was a byte-identical alias and
    # `candidates_by_id_json` is a re-keying of the same data, derived where it is used.
    row["candidates"] = json.dumps(draft, ensure_ascii=False, sort_keys=True)
    return row


# --------------------------------------------------------------------------------------
# stage 4 -- candidate review + deterministic selection
# --------------------------------------------------------------------------------------


def _candidate_review_payload(row: dict, settings: Settings) -> dict[str, Any]:
    assessment = read_kg_screen(row, settings).assessment
    review = _loads(row["semantic_review"], {})
    draft = _loads(row["candidates"], {"candidates": []})
    # Omega: pipeline.py:446-461.
    return {
        "paper_identity": assessment.get("paper"),
        "paper_summary": review.get("paper_summary"),
        "component_summary": review.get("component_summary"),
        "evidence_inventory": review.get("evidence_inventory"),
        "computational_affordances": review.get("computational_affordances"),
        "candidates": draft.get("candidates"),
        "selection_thresholds": {
            "minimum_overall_score": settings.candidates.minimum_candidate_score,
            "minimum_dimension_score": settings.candidates.minimum_candidate_dimension_score,
        },
        **_requested_tiers_payload(settings),
    }


@dd.custom_column_generator(
    required_columns=["kg_path", "semantic_review", "candidates"],
    side_effect_columns=["candidate_review_ok", "candidate_review_errors", "candidate_review_failure"],
    model_aliases=[STRONG_DETERMINISTIC_ALIAS],
)
async def candidate_review(row: dict, generator_params: Settings, models: dict) -> dict:
    capture: list[str] = []
    settings = generator_params
    by_id = {item["candidate_id"]: item for item in _loads(row["candidates"], {"candidates": []}).get("candidates", [])}

    def post(draft: dict[str, Any]) -> CheckResult:
        return checks.check_candidate_review(draft, by_id, settings)

    parser = validated_parser(CandidateReview, "candidate_review", post=post, capture=capture)
    try:
        payload = _candidate_review_payload(row, settings)
        draft, _trace = await models[STRONG_DETERMINISTIC_ALIAS].agenerate(
            prompt=user_payload(payload, CandidateReview),
            system_prompt=prompt_text(CANDIDATE_REVIEW_PROMPT),
            parser=parser,
            max_correction_steps=settings.repair.correction_steps,
            purpose="paper_task_candidate_review",
        )
    except Exception as error:  # noqa: BLE001
        if _is_retryable(error):
            raise
        return _record_failure(row, "candidate_review", error, raw_response=capture[-1] if capture else "")
    row["candidate_review_ok"] = True
    row["candidate_review_errors"] = []
    row["candidate_review_failure"] = ""
    row["candidate_review"] = json.dumps(draft, ensure_ascii=False, sort_keys=True)
    return row


@dd.custom_column_generator(
    required_columns=["candidates", "candidate_review"],
    # No `candidate_pool_json`: it dumped all 5 candidates plus their reviews (32 KB/row) and
    # no reader ever existed -- not a prompt, not a gate, not a ledger. The per-tier decisions
    # live in `selection` (one line per selected tier) and `selection_blockers`.
    #
    # No `selected_candidate_json` either: with one requested tier it is a single value, but
    # with several the value is *per tier row*, so the fan-out callback writes it (it is the
    # only place that knows which tier a row belongs to). See `tier_selection_summary`.
    side_effect_columns=["selection_ok", "selection_blockers"],
)
def selection(row: dict, generator_params: Settings) -> dict:
    """Per-tier selection, reduced to one gate + one summary line.

    Omega picks one candidate *per requested tier* and generates a package for each
    (``_select_difficulty_variants``). This column decides which tiers are satisfiable and
    records why the others are not; the boundary callback then fans the row out.
    """
    settings = generator_params
    candidates_draft = _loads(row["candidates"], {"candidates": []})
    review_draft = _loads(row["candidate_review"], {})
    by_id = {item["candidate_id"]: item for item in candidates_draft.get("candidates", [])}
    selections: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    outcome = CheckResult(ok=True)
    if review_draft.get("rejected_all"):
        outcome = CheckResult(ok=False, errors=["candidate review rejected all candidates"])
    else:
        selections, missing, outcome = checks.select_variants(by_id, review_draft, settings.difficulty.tiers, settings)

    row["selection"] = tier_selection_summary(selections, missing)
    row["selection_ok"] = bool(selections)
    row["selection_blockers"] = list(outcome.errors) + [f"missing_candidate:{tier}" for tier in missing]
    return row


def tier_selection_summary(selections: dict[str, dict[str, Any]], missing: list[str]) -> str:
    """One compact line, e.g. ``simple=cand-02,complex=cand-05;missing:medium``.

    Small enough to carry to the final dataset, where it is the only record of which
    candidate each generated variant came from (the per-tier candidate JSON is not).
    """
    parts = [f"{tier}={candidate.get('candidate_id')}" for tier, candidate in selections.items()]
    if missing:
        parts.append("missing:" + ",".join(missing))
    return ",".join(parts)


# --------------------------------------------------------------------------------------
# stage 5 -- task spec (strong) with canonicalization inside the parser
# --------------------------------------------------------------------------------------


def _reviewer_pool_text() -> str:
    return (TBS_REFERENCE_DIR / "reviewer-pool.yml").read_text(encoding="utf-8")


def _source_evidence(row: dict, settings: Settings) -> dict[str, Any]:
    """The evidence bundle both spec and implementation need.

    Extracted so the implementation payload can reuse it WITHOUT building the spec-only
    fields (selected_candidate / candidate_review / selection_reason). Building them there
    forced those columns to be carried into the implementation stage, where nothing reads
    them: measured 11.95 KB per row (~28% of that stage's seed).
    """
    assessment = read_kg_screen(row, settings).assessment
    review = _loads(row["semantic_review"], {})
    text = read_paper_text(row)
    excerpt_limit = 40_000
    return {
        "paper_identity": assessment.get("paper"),
        "paper_artifact": {
            "content_file": row.get("paper_text_path"),
            "content_sha256": sha256_text(text),
            "content_chars": len(text),
        },
        "evidence_inventory": review.get("evidence_inventory"),
        "paper_summary": review.get("paper_summary"),
        "content_sha256": sha256_text(text),
        "excerpt": text[:excerpt_limit],
        "truncated": len(text) > excerpt_limit,
        "missing_metadata_policy": "Preserve unknown values; never invent a title, year or equation number.",
    }


def _spec_payload(row: dict, settings: Settings) -> dict[str, Any]:

    assessment = read_kg_screen(row, settings).assessment
    review = _loads(row["semantic_review"], {})
    selected = _loads(row.get("selected_candidate_json"), None)
    # The fan-out projected exactly these two values onto the row, so the full
    # `candidate_review` blob (~8 KB, identical for every tier of a paper) does not travel.
    selected_review = _loads(row.get("selected_review_json"), None)
    selection_reason = str(row.get("selection_reason") or "")
    tier = row_tier(row, settings)
    return {
        "generation_guidance": settings.generation_guidance,
        "source_evidence": _source_evidence(row, settings),
        "tbs_reviewer_pool": _reviewer_pool_text(),
        "paper_identity": assessment.get("paper"),
        "structural_metrics": assessment.get("metrics"),
        "paper_summary": review.get("paper_summary"),
        "component_summary": review.get("component_summary"),
        "evidence_inventory": review.get("evidence_inventory"),
        "computational_affordances": review.get("computational_affordances"),
        "selected_candidate": selected,
        "candidate_review": selected_review,
        "selection_reason": (f"Selected as the highest-scoring eligible {tier} candidate. " + selection_reason),
        "difficulty_tier": tier,
        "difficulty_tier_band": sorted(DIFFICULTY_TIER_BANDS[tier]),
        "generation_contract": {
            "step": "task_spec",
            "model_required_files": ["instruction.md", "task.toml"],
            "required_task_spec_fields": sorted(REQUIRED_TASK_SPEC_FIELDS),
            "runtime_profile": "python_scientific",
        },
    }


def _site_contract_errors(files: dict[str, str], settings: Settings) -> list[str]:
    """Keep task.toml consistent with where the package will be staged.

    The vendored ``check-task-path.sh`` / ``check-task-fields.sh`` require
    ``[task].name == terminal-bench-science/<dir>`` and ``[metadata].domain/field`` to
    match ``tasks/<domain>/<field>/<task>``. Catching this at spec time (and feeding it
    back through the repair loop) is much cheaper than discovering it after materialization.
    """
    from heta_framework.generation.graph._native._compat import tomllib

    errors: list[str] = []
    try:
        parsed = tomllib.loads(files.get("task.toml", ""))
    except Exception as error:  # noqa: BLE001 - reported, never raised
        return [f"task.toml does not parse: {error}"]
    package_name = str(parsed.get("task", {}).get("name", "")).split("/")[-1]
    expected_name = f"terminal-bench-science/{package_name}"
    if str(parsed.get("task", {}).get("name", "")) != expected_name:
        errors.append(f"task.toml [task].name must be {expected_name!r}")
    metadata = parsed.get("metadata") or {}
    if metadata.get("domain") != settings.task_domain:
        errors.append(f"task.toml [metadata].domain must be {settings.task_domain!r}")
    if metadata.get("field") != settings.task_field:
        errors.append(f"task.toml [metadata].field must be {settings.task_field!r}")
    return errors


def _set_top_level_key(text: str, key: str, value: str) -> str:
    """Set a TOML key that lives before the first table header."""
    line = re.compile(rf"(?m)^[ \t]*{re.escape(key)}[ \t]*=.*$")
    if line.search(text):
        return line.sub(f"{key} = {value}", text, count=1)
    return f"{key} = {value}\n" + text


def _set_toml_key(text: str, section: str, key: str, value: str) -> str:
    """Set ``key = value`` inside ``[section]``, replacing it or inserting it.

    Line-based, so the rest of the model-authored TOML (comments, ordering) survives.
    """
    header = re.compile(rf"(?m)^\[{re.escape(section)}\][ \t]*(?:#.*)?$")
    match = header.search(text)
    if not match:
        return text.rstrip() + f"\n\n[{section}]\n{key} = {value}\n"
    end = re.search(r"(?m)^\[", text[match.end() :])
    body_end = match.end() + (end.start() if end else len(text) - match.end())
    body = text[match.end() : body_end]
    line = re.compile(rf"(?m)^[ \t]*{re.escape(key)}[ \t]*=.*$")
    if line.search(body):
        body = line.sub(f"{key} = {value}", body, count=1)
    else:
        body = f"\n{key} = {value}" + body
    return text[: match.end()] + body + text[body_end:]


# Names that mark a file as verifier-owned truth rather than a hidden case input.
_TRUTH_ARTIFACT_MARKERS = (
    "reference",
    "answer",
    "expected",
    "solution",
    "optimal",
    "ground_truth",
    "ground-truth",
    "truth",
    "label",
)


def _evaluation_consistency_errors(spec: dict[str, Any]) -> list[str]:
    """The hidden-case glob must select case inputs, not the answers.

    A glob like ``/tests/data/*.json`` silently matches a reference/answer artifact
    placed in the same directory, so the verifier would score the answer key as a case.
    Omega only checks (at generation time) that the glob matches at least one file; this
    rule catches the dangerous direction at spec time, where the repair loop can fix it.
    """
    topology = spec.get("evaluation_topology") or {}
    contract = spec.get("data_contract") or {}
    glob = str(topology.get("hidden_case_glob") or "")
    if not glob:
        return []
    directory = str(PurePosixPath(glob).parent)
    offenders: list[str] = []
    for entry in contract.get("test_files") or []:
        path = str(entry.get("path") or "") if isinstance(entry, dict) else str(entry)
        if not path or not path.startswith(directory + "/"):
            continue
        name = PurePosixPath(path).name.casefold()
        if any(marker in name for marker in _TRUTH_ARTIFACT_MARKERS):
            offenders.append(path)
    if not offenders:
        return []
    return [
        "evaluation_topology.hidden_case_glob "
        f"{glob!r} also matches verifier-owned truth artifacts: {', '.join(offenders)}. "
        "Fix this WITHOUT moving the glob out of /tests/data/ (that is a hard requirement): "
        "either move the reference/answer files to another directory such as /tests/reference/ "
        "and keep them out of the glob, or narrow the glob to select only case inputs, "
        "for example /tests/data/case_*.json."
    ]


def _drop_section(text: str, section: str) -> str:
    """Remove ``[section]`` and everything under it up to the next table header."""
    header = re.compile(rf"(?m)^\[{re.escape(section)}\][ \t]*(?:#.*)?$")
    match = header.search(text)
    if not match:
        return text
    end = re.search(r"(?m)^\[", text[match.end() :])
    body_end = match.end() + (end.start() if end else len(text) - match.end())
    return (text[: match.start()] + text[body_end:]).rstrip() + "\n"


def _write_runtime_sections(task_toml: str, runtime: dict[str, Any]) -> str:
    """Author the TBS runtime sections from ``runtime_spec`` instead of trusting the model.

    Every value here is either a TBS constant (``environment_mode``, the two
    ``network_mode`` values) or a copy of a ``runtime_spec`` field. Omega only lets the
    model decide *estimates*; letting it restate constants by hand adds a failure mode
    with no scientific content — observed in practice as ``memory``/``storage`` instead
    of ``memory_mb``/``storage_mb`` and a missing ``network_mode``.
    """
    import json as _json

    artifacts = runtime.get("artifact_paths") or []
    text = _set_top_level_key(task_toml, "artifacts", _json.dumps(artifacts))
    for section in ("verifier", "verifier.environment", "agent", "environment"):
        text = _drop_section(text, section)
    agent_timeout = runtime.get("agent_timeout_sec")
    verifier_timeout = runtime.get("verifier_timeout_sec")
    text += (
        "\n[verifier]\n"
        'environment_mode = "separate"\n'
        f"timeout_sec = {int(verifier_timeout)}\n"
        "\n[verifier.environment]\n"
        'network_mode = "no-network"\n'
        "\n[agent]\n"
        f"timeout_sec = {int(agent_timeout)}\n"
        "\n[environment]\n"
        'network_mode = "public"\n'
        f"build_timeout_sec = {int(runtime['build_timeout_sec'])}\n"
        f"cpus = {int(runtime['cpus'])}\n"
        f"memory_mb = {int(runtime['memory_mb'])}\n"
        f"storage_mb = {int(runtime['storage_mb'])}\n"
        "gpus = 0\n"
    )
    # Dropping and rewriting the runtime sections can remove a canary the model placed
    # inside them; the canary is a constant, so Python guarantees it rather than
    # re-prompting for it.
    if TBS_CANARY not in text:
        text = text.rstrip() + f"\n# {TBS_CANARY}\n"
    return text


def _apply_site_contract(files: dict[str, str], runtime: dict[str, Any], settings: Settings) -> dict[str, str]:
    """Python owns the values that describe *where* the task lives.

    The site triple and the verifier timeout are pure functions of the chosen
    ``tasks/<domain>/<field>/<slug>`` location and the runtime contract, so the model
    is never trusted with them — this mirrors Omega taking ownership of the derived
    time budget instead of asking the model to copy it correctly.
    """
    from heta_framework.generation.graph._native._compat import tomllib

    task_toml = files.get("task.toml", "")
    try:
        package_slug = str(tomllib.loads(task_toml).get("task", {}).get("name", "")).split("/")[-1]
    except Exception:  # noqa: BLE001 - malformed TOML is reported by the contract check
        return files
    if not package_slug or not KEBAB.fullmatch(package_slug):
        return files
    task_toml = _set_top_level_key(task_toml, "schema_version", '"1.0"')
    task_toml = _set_toml_key(task_toml, "task", "name", f'"terminal-bench-science/{package_slug}"')
    task_toml = _set_toml_key(task_toml, "metadata", "domain", f'"{settings.task_domain}"')
    task_toml = _set_toml_key(task_toml, "metadata", "field", f'"{settings.task_field}"')
    task_toml = _write_runtime_sections(task_toml, runtime)
    files["task.toml"] = task_toml
    return files


def _enforce_submission_contract(spec: dict[str, Any]) -> None:
    """Derive the upload contract from the evaluation topology.

    TBS requires program-on-hidden-cases submissions to upload exactly one reusable
    program. The model kept declaring extra artifact paths, so Python derives the list
    from ``program_artifact`` instead of asking for a consistent copy.
    """
    topology = spec.get("evaluation_topology") or {}
    runtime = spec.get("runtime_spec") or {}
    if topology.get("mode") == "submitted_program_on_hidden_cases":
        artifact = topology.get("program_artifact")
        if isinstance(artifact, str) and artifact:
            runtime["artifact_paths"] = [artifact]
            spec["runtime_spec"] = runtime


def make_spec_post(settings: Settings, selected: dict | None) -> Callable[[dict[str, Any]], CheckResult]:
    """Canonicalize first (Omega owns the derived fields), then contract-check."""

    def post(draft: dict[str, Any]) -> CheckResult:
        # The selected candidate owns the concept set; the model only copies it, so
        # Python re-imposes the exact list rather than trusting a paraphrase.
        if selected and selected.get("concepts_used"):
            draft["difficulty_contract"]["concepts_used"] = list(selected["concepts_used"])
        hours = float(draft["difficulty_contract"]["expert_time_estimate_hours"])
        agent_timeout_sec = int(hours * 3600)
        runtime = dict(draft["runtime_spec"])
        runtime["agent_timeout_sec"] = agent_timeout_sec
        draft["runtime_spec"] = runtime

        files = canonicalize.canonicalize_spec_time_budget(
            dict(draft["files"]),
            expert_hours=hours,
            agent_timeout_sec=agent_timeout_sec,
        )
        files["task.toml"] = canonicalize.fill_environment_fields(files["task.toml"], runtime)
        files["instruction.md"] = canonicalize.absolute_instruction_paths(
            files["instruction.md"], runtime, draft.get("data_contract") or {}
        )
        _enforce_submission_contract(draft)
        runtime = draft["runtime_spec"]
        files = _apply_site_contract(files, runtime, settings)
        draft["files"] = files
        outcome = checks.check_spec_draft(draft, settings, selected_candidate=selected)
        site_errors = _site_contract_errors(files, settings) + _evaluation_consistency_errors(draft)
        if site_errors:
            outcome = CheckResult(ok=False, errors=site_errors + outcome.errors, warnings=outcome.warnings)
        return outcome

    return post


@dd.custom_column_generator(
    # `selection` is required so the DAG forces it to run before spec: the payload reads
    # `selected_candidate_json`, which is a side effect of the selection column.
    # `paper_text_path` is read by `_source_evidence` (the payload re-reads the body to hash
    # the whole text) and was missing here: the audit test in tests/test_column_lifecycle.py
    # now resolves the asset readers, which is how the omission surfaced.
    # `selected_candidate_json` and `tier` are the per-row values the fan-out callback writes:
    # the spec now depends on the *tier's* candidate, not on a paper-level selection.
    required_columns=[
        "kg_path",
        "paper_text_path",
        "semantic_review",
        "selected_candidate_json",
        "selected_review_json",
        "selection_reason",
        "tier",
        "tier_selected",
    ],
    side_effect_columns=["spec_ok", "spec_errors", "spec_failure"],
    model_aliases=[STRONG_ALIAS],
)
async def spec(row: dict, generator_params: Settings, models: dict) -> dict:
    capture: list[str] = []
    settings = generator_params
    selected = _loads(row.get("selected_candidate_json"), None)
    parser = validated_parser(SpecDraft, "task_spec", post=make_spec_post(settings, selected), capture=capture)
    try:
        payload = _spec_payload(row, settings)
        draft, _trace = await models[STRONG_ALIAS].agenerate(
            prompt=user_payload(payload, SpecDraft),
            system_prompt=prompt_text(SPEC_GENERATION_PROMPT),
            parser=parser,
            max_correction_steps=settings.repair.correction_steps,
            purpose="paper_task_spec_generation",
        )
    except Exception as error:  # noqa: BLE001
        if _is_retryable(error):
            raise
        return _record_failure(row, "spec", error, raw_response=capture[-1] if capture else "")
    row["spec"] = json.dumps(draft, ensure_ascii=False, sort_keys=True)
    row["spec_ok"] = True
    row["spec_errors"] = []
    row["spec_failure"] = ""
    return row


# --------------------------------------------------------------------------------------
# stage 6 -- implementation (strong) + model check + deterministic package assembly
# --------------------------------------------------------------------------------------


def _implementation_contract(spec_draft: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable checklist for the implementation step.

    The vendored prompt states these rules in prose (lines 28-30, 80-81) and the repair
    loop feeds back the validator's errors, but a model can still miss them. The spec step
    already receives a ``generation_contract``; giving the implementation step the same,
    with the paths derived from ``program_artifact``, removes the guesswork.
    """
    topology = spec_draft.get("evaluation_topology") or {}
    artifact = str(topology.get("program_artifact") or "")
    required: list[str] = []
    if topology.get("mode") == "submitted_program_on_hidden_cases" and artifact:
        relative = artifact[len("/app/") :] if artifact.startswith("/app/") else artifact.lstrip("/")
        required = [f"solution/{relative}", f"environment/starter/{relative}"]
    required_files = sorted(REQUIRED_IMPLEMENTATION_FILES | set(required))
    return {
        "required_files": required_files,
        "program_artifact": artifact or None,
        "reference_program_paths": required,
        "solve_sh": "solution/solve.sh must install the reference program into program_artifact",
        "verifier": (
            "tests/test_outputs.py MUST contain executable assertions (`assert ...`) that "
            "inspect the submitted program's computed output; a placeholder or "
            "trivially-passing test is rejected. The prompt's rule \"do not turn warnings "
            'into verifier assertions" applies ONLY to Omega-only preferences (comparator '
            "success, difficulty heuristics, fixed concept/step counts, byte-for-byte "
            "repeatability). Scientific correctness, interface and hidden-case checks MUST "
            "be asserted, not merely logged."
        ),
        "generator_cli": {
            "flags": ["--public-output-dir", "--test-output-dir", "--public-seed", "--hidden-seeds"],
            "hidden_seeds_format": (
                "ONE comma-separated string, e.g. --hidden-seeds 11,22,33. Do NOT use nargs or "
                "repeated flags: the materializer passes a single comma-joined argument."
            ),
            "invocation": (
                "python3 /generator/generation_data.py --public-output-dir /outputs/public "
                "--test-output-dir /outputs/test --public-seed <int> --hidden-seeds <a,b,c>"
            ),
        },
        "forbidden": [
            "instruction.md",
            "task.toml",
            "execution.md",
            "Dockerfiles (Omega renders them)",
            "tests/support/omega_runtime.py",
            "tests/omega_test_body.sh",
        ],
    }


def _implementation_payload(row: dict, settings: Settings) -> dict[str, Any]:
    spec_draft = _loads(row["spec"], {})
    return {
        "generation_guidance": settings.generation_guidance,
        "source_evidence": _source_evidence(row, settings),
        "package_name": spec_draft["package_name"],
        "difficulty_tier": row_tier(row, settings),
        "task_spec": spec_draft["task_spec"],
        "runtime_spec": spec_draft["runtime_spec"],
        "data_contract": spec_draft["data_contract"],
        "evaluation_topology": spec_draft["evaluation_topology"],
        "difficulty_contract": spec_draft["difficulty_contract"],
        "runtime_support_api": "from support.omega_runtime import private_case, private_program, run_submitted",
        "implementation_contract": _implementation_contract(spec_draft),
        "instruction_md": spec_draft["files"]["instruction.md"],
        "task_toml": spec_draft["files"]["task.toml"],
    }


def _generator_cli_errors(files: dict[str, str]) -> list[str]:
    """The generator must accept the exact invocation the materializer uses.

    ``check_implementation_draft`` only asserts the four flags are mentioned; a generator
    declared with ``nargs="+"`` still passes that and then dies with argparse rc=2 inside the
    container (observed). Static inspection is the cheap place to catch it — running the
    model's code outside the container is explicitly not allowed.
    """
    import ast

    source = files.get("generation_data.py", "")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # syntax errors are reported by the contract check
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        positional = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
        if "--hidden-seeds" not in positional:
            continue
        for keyword in node.keywords:
            if keyword.arg == "nargs":
                return [
                    "generation_data.py: --hidden-seeds must take ONE comma-separated string "
                    f"(found nargs={ast.unparse(keyword.value)}); the materializer passes '1,2,3'"
                ]
    return []


def make_implementation_post(settings: Settings, spec_draft: dict[str, Any]) -> Callable[[dict[str, Any]], CheckResult]:
    def post(draft: dict[str, Any]) -> CheckResult:
        files = {
            path: canonicalize.fix_reward_initialization(canonicalize.add_canary(content, path))
            for path, content in draft["files"].items()
        }
        draft["files"] = files
        outcome = checks.check_implementation_draft(draft, spec_draft, settings)
        cli_errors = _generator_cli_errors(files)
        if cli_errors:
            outcome = CheckResult(ok=False, errors=cli_errors + outcome.errors, warnings=outcome.warnings)
        return outcome

    return post


@dd.custom_column_generator(
    required_columns=["spec", "kg_path", "paper_text_path", "semantic_review", "tier"],
    side_effect_columns=[
        "implementation_ok",
        "implementation_errors",
        "implementation_failure",
        "implementation_check_ok",
        "implementation_check_errors",
        "implementation_check_failure",
    ],
    model_aliases=[STRONG_ALIAS, STRONG_NO_RETRY_ALIAS],
)
async def implementation(row: dict, generator_params: Settings, models: dict) -> dict:
    """Draft, then check-and-correct: two model calls, one column.

    Omega runs these as two steps, and the first version of this migration mirrored that as
    two columns. Nothing reads the draft once the check has run — the package render consumes
    the corrected set only — so the pair doubled the largest in-flight payload of the stage
    (~21 KB/row) and added a DAG node with no consumer. Data Designer cannot drop a real
    column mid-stage, so the calls are folded here instead; the check keeps its own
    observable outcome through ``implementation_check_*``.
    """
    capture: list[str] = []
    settings = generator_params
    spec_draft = _loads(row["spec"], {})
    parser = validated_parser(
        ImplementationDraft,
        "implementation",
        post=make_implementation_post(settings, spec_draft),
        capture=capture,
    )
    try:
        payload = _implementation_payload(row, settings)
        draft, _trace = await models[STRONG_ALIAS].agenerate(
            prompt=user_payload(payload, ImplementationDraft),
            system_prompt=prompt_text(IMPLEMENTATION_GENERATION_PROMPT),
            parser=parser,
            max_correction_steps=settings.repair.correction_steps,
            purpose="paper_task_implementation_generation",
        )
    except Exception as error:  # noqa: BLE001
        if _is_retryable(error):
            raise
        _record_failure(row, "implementation", error, raw_response=capture[-1] if capture else "")
        _record_skipped_step(row, "implementation_check", "implementation draft failed; check not attempted")
        return row

    check_parser = validated_parser(
        ImplementationDraft,
        "implementation_check",
        post=make_implementation_post(settings, spec_draft),
        capture=capture,
    )
    try:
        # Omega's check-and-correct step: exactly one call, no correction rounds.
        corrected, _trace = await models[STRONG_NO_RETRY_ALIAS].agenerate(
            prompt=user_payload(_implementation_check_payload(spec_draft, draft, settings), ImplementationDraft),
            system_prompt=prompt_text(IMPLEMENTATION_CHECK_PROMPT),
            parser=check_parser,
            max_correction_steps=settings.repair.implementation_check_correction_steps,
            purpose="paper_task_implementation_check",
        )
    except Exception as error:  # noqa: BLE001
        if _is_retryable(error):
            raise
        # The draft survives, but it is unverified: `package_check` gates the row out on
        # `implementation_check_ok is False`, so nothing can ship an unchecked draft.
        row["implementation"] = json.dumps(draft, ensure_ascii=False, sort_keys=True)
        row["implementation_ok"] = True
        row["implementation_errors"] = []
        row["implementation_failure"] = ""
        _record_step_failure(row, "implementation_check", error, raw_response=capture[-1] if capture else "")
        return row

    row["implementation"] = json.dumps(corrected, ensure_ascii=False, sort_keys=True)
    row["implementation_ok"] = True
    row["implementation_errors"] = []
    row["implementation_failure"] = ""
    row["implementation_check_ok"] = True
    row["implementation_check_errors"] = []
    row["implementation_check_failure"] = ""
    return row


def _implementation_check_payload(
    spec_draft: dict[str, Any], draft: dict[str, Any], settings: Settings
) -> dict[str, Any]:
    """The check sees the draft by value: it is a local, never a dataset column."""
    return {
        "generation_guidance": settings.generation_guidance,
        "binding_contract": {
            "implementation_contract": _implementation_contract(spec_draft),
            "package_name": spec_draft["package_name"],
            "task_spec": spec_draft["task_spec"],
            "runtime_spec": spec_draft["runtime_spec"],
            "data_contract": spec_draft["data_contract"],
            "evaluation_topology": spec_draft["evaluation_topology"],
            "difficulty_contract": spec_draft["difficulty_contract"],
        },
        "implementation": draft,
        "tbs_implementation_rubric": (TBS_REFERENCE_DIR / "task-implementation.toml").read_text(encoding="utf-8"),
        "omega_runtime_source": (ASSETS_DIR / "templates" / "runtime_support.py").read_text(encoding="utf-8"),
        "instruction": (
            "Check the implementation against the binding contract and the rubric. Correct every "
            "fixable defect and return the complete corrected file set. Do not report a fixable "
            "defect as a blocker."
        ),
    }


def render_package_files(spec_draft: dict[str, Any], implementation_draft: dict[str, Any]) -> dict[str, str]:
    """Render the ``task_package`` file map.

    A pure function of the spec, the corrected implementation and the vendored assets, so it
    is NOT a column: it was 43 KB/row (the largest column in the pipeline) whose only readers
    are the two deterministic steps that follow it in the same stage — ``package_check`` and
    the materialize callback, which call this function instead. The map's identity is recorded
    as ``files_sha256`` in the materialized ``generation_manifest.json``.
    """
    task_files = {
        "instruction.md": spec_draft["files"]["instruction.md"],
        "task.toml": spec_draft["files"]["task.toml"],
        **implementation_draft.get("files", {}),
    }
    rendered = scaffold.render_scaffold(spec_draft, spec_draft["runtime_spec"], task_files)
    rendered.update(canonicalize.verifier_entrypoint(rendered["tests/test.sh"]))
    return rendered


@dd.custom_column_generator(
    required_columns=["spec", "implementation"],
    side_effect_columns=["package_ok", "package_errors", "package_name"],
)
def package_check(row: dict, generator_params: Settings) -> dict:
    spec_draft = _loads(row["spec"], {})
    implementation = _loads(row.get("implementation"), {})
    errors: list[str] = []
    gated = row.get("implementation_check_ok") is False or row.get("spec_ok") is False
    try:
        files = {} if gated else render_package_files(spec_draft, implementation)
    except Exception as error:  # noqa: BLE001 - a render failure rejects the row, never the run
        files = {}
        errors.append(f"package rendering failed: {type(error).__name__}: {error}")
    if not files:
        errors.append("no package files were produced (upstream step failed)")
    for path in files:
        try:
            checks.safe_relative_path(path, stage="package")
        except ValueError as error:
            errors.append(str(error))
    missing = sorted(REQUIRED_PACKAGE_FILES - set(files))
    if missing:
        errors.append(f"missing required package files: {', '.join(missing)}")
    for name in ("solution/solve.sh", "tests/test.sh"):
        if name in files and not files[name].startswith("#!"):
            errors.append(f"{name} must start with a shebang")
    row["package_check"] = "ready" if not errors else "invalid"
    row["package_ok"] = not errors
    row["package_errors"] = errors
    row["package_name"] = spec_draft.get("package_name", "")
    return row


# --------------------------------------------------------------------------------------
# stage 7 -- static checks (deterministic column, runs after materialization)
# --------------------------------------------------------------------------------------


@dd.custom_column_generator(
    required_columns=["package_dir"],
    side_effect_columns=["static_passed", "static_summary", "static_log_dir"],
)
def static_check(row: dict, generator_params: Settings) -> dict:
    """Run the vendored TBS checks on a materialized package.

    Error isolation is deliberate: a failing or missing check becomes a column value,
    never an exception, because an exception would count toward Data Designer's
    ``shutdown_error_rate`` and could abort an otherwise healthy run.
    """
    settings = generator_params
    package_dir = Path(str(row["package_dir"]))
    staging_root = package_dir.parent / "_staging"
    try:
        outcomes = static_checks.run_static_checks(
            package_dir,
            staging_root=staging_root,
            checks_dir=CHECK_SCRIPTS_DIR,
            reviewer_pool=TBS_REFERENCE_DIR / "reviewer-pool.yml",
            slug=package_dir.name,
            domain=settings.task_domain,
            field=settings.task_field,
        )
        summary = static_checks.summarize_static(outcomes)
    except Exception as error:  # noqa: BLE001 - never fail the row for infrastructure reasons
        summary = {"passed": False, "attempted": 0, "failed": ["<runner>"], "warnings": {}, "error": str(error)}
    row["static_check"] = "passed" if summary.get("passed") else "failed"
    row["static_passed"] = bool(summary.get("passed"))
    row["static_summary"] = summary
    row["static_log_dir"] = str(staging_root)
    return row


# --------------------------------------------------------------------------------------
# keep-only row filter (used as a stage output processor)
# --------------------------------------------------------------------------------------


def keep_rows_dataframe(dataframe: Any, column: str) -> Any:
    """Filter helper shared by the workflow's output processors."""
    if column not in dataframe.columns:
        return dataframe.iloc[0:0]
    return dataframe[dataframe[column].fillna(False).astype(bool)].reset_index(drop=True)


def materialize_package_files(files: dict[str, str], target: Path) -> list[str]:
    """Write a package file map to disk (called from the workflow callback, never a column)."""
    written: list[str] = []
    for relative, content in sorted(files.items()):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        written.append(relative)
    for executable in ("solution/solve.sh", "tests/test.sh"):
        candidate = target / executable
        if candidate.is_file():
            os.chmod(candidate, 0o755)
    return written
