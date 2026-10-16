# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite-workflow assembly: one stage per Omega pipeline step.

Two things Data Designer cannot express as columns live in ``on_success`` callbacks:

* **row gating** — there is no column-level row-delete primitive; only a stage
  boundary can change the row count, so rejected rows are filtered between stages
  and written to a reject ledger instead of being deleted (better audit than Omega's
  directory removal).
* **materialization** — columns are values. Writing the ``task_package/`` tree is the
  one place that touches the filesystem, and its output parquet is the next stage's seed.

Every callback that changes data is versioned via ``on_success_version`` because the
composite workflow cannot hash arbitrary Python: without a version, a changed callback
would silently reuse stale output.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import data_designer.config as dd
import pandas as pd

from heta_framework.generation.graph._native.config import Settings
from heta_framework.generation.graph._native.dd import columns, validation
from heta_framework.generation.graph._native.domain import checks
from heta_framework.generation.graph._native.io_utils import as_gate, atomic_write_json

PARQUET_DIR = "parquet-files"
CALLBACK_DIR = "callback-outputs"

# Column life cycle. Each stage boundary writes a projection, and that projection IS the
# next stage's seed, so a column dropped here costs nothing downstream. Two machine-checked
# audits keep the lists honest (tests/test_column_lifecycle.py): a projection may not drop a
# column a later stage reads, and it may not carry a column no later stage reads.
#
# `paper_text_chars` is deliberately absent from every projection: the body is re-read from
# `paper_text_path` and measured there, and nothing downstream consumes the stored length.
SEED_KEEP: tuple[str, ...] = ("paper_id", "title", "paper_text_path", "kg_path")
# screen -> review: `kg_screen` / `kg_lcf_json` are a pure function of the KG asset and the
# thresholds, so the payloads re-derive them from `kg_path` (kept in SEED_KEEP) instead of the
# pipeline carrying 53 KB/row through three stages. Only the gate outcome is kept, for audit.
SCREEN_KEEP: tuple[str, ...] = SEED_KEEP + ("kg_passed",)
# review -> candidates: the trace (114 KB/row) and per-step error lists are debugging only.
# `review_suitable` is the gate's own boolean and stops here: `review_gate` already carries the
# same decision as prose, and nothing downstream reads the boolean.
REVIEW_KEEP: tuple[str, ...] = SCREEN_KEEP + ("semantic_review", "review_gate")
# candidates -> candidate_review: the component graph is no longer read by any prompt.
CANDIDATES_KEEP: tuple[str, ...] = SEED_KEEP + (
    "kg_passed",
    "semantic_review",
    "review_gate",
    "candidates",
    # Referenced by the NEXT stage's skip.when ({"{{ candidates_ok == false }}"}). A
    # projection must keep every column a later skip expression reads, not just the ones a
    # payload reads; tests/test_column_lifecycle.py now checks that channel too.
    "candidates_ok",
)
# candidate_review -> spec. This is the fan-out boundary (one row per requested tier), so the
# list is produced BY the callback for the per-tier values (`tier`, `selected_candidate_json`,
# `selection`) and covers only what the spec/implementation payloads read afterwards.
# `candidates` (29 KB/row), `selection_ok` and `selection_blockers` are consumed by the
# callback itself and have no reader past it: the per-tier candidate replaces the pool, and the
# coverage gaps go to `rejected_difficulty.json`.
CANDIDATE_REVIEW_KEEP: tuple[str, ...] = SEED_KEEP + (
    "kg_passed",
    "semantic_review",
    "review_gate",
    "tier",
    "tier_selected",
    "selected_candidate_json",
    # Only the two review values the spec payload reads; `candidate_review` itself (~8 KB) stops
    # at the fan-out instead of being duplicated into every tier row.
    "selected_review_json",
    "selection_reason",
    "selection",
)
# spec -> implementation: the implementation payload only needs `spec` plus the shared
# source_evidence (semantic_review / paper_text_path / the re-derived screen). `candidate_review`
# and `selected_candidate_json` are spec-stage inputs, and building them for the implementation
# payload was pure carry-over (11.95 KB/row, 28% of that seed).
SPEC_KEEP: tuple[str, ...] = SEED_KEEP + (
    "kg_passed",
    "semantic_review",
    "review_gate",
    "selection",
    "tier",  # the per-row variant identity: the implementation payload needs it
    # Read by the materialize ledger, which has to tell "this tier had no candidate" apart from
    # "this tier's authoring failed" (found live: without it the ledger said missing=False).
    "tier_selected",
    "spec",
    "spec_ok",
    "spec_errors",
)

STAGE_KEEP_COLUMNS: dict[str, tuple[str, ...]] = {
    "screen": SCREEN_KEEP,
    "review": REVIEW_KEEP,
    "candidates": CANDIDATES_KEEP,
    "candidate_review": CANDIDATE_REVIEW_KEEP,
    "spec": SPEC_KEEP,
}


# --------------------------------------------------------------------------------------
# stage output helpers
# --------------------------------------------------------------------------------------


def read_stage_frame(stage_dir: Path) -> pd.DataFrame:
    """Concatenate a stage's row-group parquets into one frame.

    Every value is read back by name and coerced explicitly (``omegadesigner.io_utils``), so the
    pandas FutureWarning about dtype inference for all-NA columns is noise here: a column whose
    producer was skipped is legitimately all-NA, and narrowing it away would change the schema.
    """
    import warnings

    parquet_dir = Path(stage_dir) / PARQUET_DIR
    frames = [pd.read_parquet(path) for path in sorted(parquet_dir.glob("*.parquet"))]
    if not frames:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*empty or all-NA.*")
        return pd.concat(frames, ignore_index=True)


def _write_callback_frame(stage_dir: Path, version: str, frame: pd.DataFrame) -> Path:
    out = Path(stage_dir) / CALLBACK_DIR / version
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "data.parquet", index=False)
    return out


def _project(frame: pd.DataFrame, keep_columns: tuple[str, ...] | None) -> pd.DataFrame:
    if keep_columns is None:
        return frame
    keep = set(keep_columns)
    return frame[[column for column in frame.columns if column in keep]]


def make_projection_callback(
    keep_columns: tuple[str, ...],
    *,
    version: str,
) -> Callable[[Path], Path]:
    """A boundary that only projects columns (no row filtering)."""

    def callback(stage_dir: Path) -> Path:
        stage_dir = Path(stage_dir)
        frame = read_stage_frame(stage_dir)
        return _write_callback_frame(stage_dir, version, _project(frame, keep_columns))

    return callback


def make_runtime_seed_callback(*, version: str) -> Callable[[Path], Path]:
    """Keep all static verdicts with a stable schema for the runtime seed.

    Row groups infer empty error lists as list<null> and populated lists as
    list<string>. Runtime readers already accept JSON via as_json_object.
    """

    def callback(stage_dir: Path) -> Path:
        stage_dir = Path(stage_dir)
        frame = read_stage_frame(stage_dir)
        if "static_summary" in frame:
            # pandas handles nested numpy arrays without flattening singleton
            # lists or stringifying multi-element arrays.
            records = json.loads(frame[["static_summary"]].to_json(orient="records"))
            summaries = []
            for record in records:
                value = record["static_summary"]
                if isinstance(value, str):
                    value = json.loads(value)
                summaries.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
            frame["static_summary"] = pd.Series(summaries, index=frame.index, dtype="string")
        return _write_callback_frame(stage_dir, version, frame)

    return callback


def make_filter_callback(
    column: str,
    *,
    version: str,
    reject_path_name: str,
    reason_columns: tuple[str, ...] = (),
    keep_columns: tuple[str, ...] | None = None,
) -> Callable[[Path], Path]:
    """Keep only rows whose boolean ``column`` is true; ledger the rest.

    Omega deletes rejected paper directories outright. Keeping the reject record in a
    ledger preserves the audit trail the batch layer used to keep in
    ``review_rejections.json``.
    """

    def callback(stage_dir: Path) -> Path:
        stage_dir = Path(stage_dir)
        frame = read_stage_frame(stage_dir)
        if frame.empty:
            return _write_callback_frame(stage_dir, version, frame)
        keep = frame[column].fillna(False).astype(bool)
        kept = frame[keep].reset_index(drop=True)
        rejected = frame[~keep]
        ledger = []
        for record in rejected.to_dict(orient="records"):
            entry = {"paper_id": record.get("paper_id"), "title": record.get("title"), "gate": column}
            for name in reason_columns:
                if name in record:
                    entry[name] = _jsonable(record[name])
            ledger.append(entry)
        atomic_write_json(stage_dir / reject_path_name, {"count": len(ledger), "rejected": ledger})
        return _write_callback_frame(stage_dir, version, _project(kept, keep_columns))

    return callback


def make_tier_fanout_callback(
    settings: Settings,
    *,
    version: str,
    reject_path_name: str,
    keep_columns: tuple[str, ...],
) -> Callable[[Path], Path]:
    """One row per **requested** difficulty tier — Omega's ``variants/<tier>/`` fan-out.

    Omega reviews the candidate pool once and then, *per requested tier*, selects the
    highest-scoring eligible candidate and generates a complete package
    (``pipeline.py:1209-1237`` + the ``generate_tier`` loop). A stage boundary is the only
    place Data Designer lets a row count change, so the fan-out happens here: screen, review,
    candidates and candidate_review have all run **once per paper**, and every tier row inherits
    their outputs — no model call is repeated for a tier.

    Every requested tier gets a row, including the ones with no eligible candidate
    (``tier_selected=false``). That makes the row count exactly ``papers x tiers`` instead of
    "however many tiers happened to qualify", keeps each tier's outcome visible as data, and
    lets the next stages skip the empty tiers (``spec`` skips on ``tier_selected == false``) so
    a missing tier costs no model call and no Docker work. The ledger still lists which tiers
    were missing and why.

    The per-tier choice is *recomputed* from ``candidates`` + ``candidate_review`` rather than
    carried as a map: it is a deterministic function of columns this stage already has, and the
    map would have been the largest column in the pipeline. Only the two review values the spec
    payload actually reads are projected onto the row (``selected_review_json`` and
    ``selection_reason``), which is why the full ``candidate_review`` blob stops here instead of
    being duplicated into every tier row (~8 KB -> ~2 KB per row).
    """

    def callback(stage_dir: Path) -> Path:
        stage_dir = Path(stage_dir)
        frame = read_stage_frame(stage_dir)
        if frame.empty:
            return _write_callback_frame(stage_dir, version, frame)

        rows: list[dict[str, Any]] = []
        ledger: list[dict[str, Any]] = []
        for record in frame.to_dict(orient="records"):
            entry = {"paper_id": record.get("paper_id"), "title": record.get("title")}
            tiers = list(settings.difficulty.tiers)
            if record.get("selection_ok") is not True:
                # No tier can be built: the paper leaves the pipeline here and every requested
                # tier is recorded against it.
                ledger.append(
                    {
                        **entry,
                        "gate": "selection_ok",
                        "missing_tiers": tiers,
                        "selection": _jsonable(record.get("selection")),
                        "selection_blockers": _jsonable(record.get("selection_blockers")),
                    }
                )
                continue
            review_draft = _json_loads(record.get("candidate_review"), {})
            candidates = _json_loads(record.get("candidates"), {})
            by_id = {item["candidate_id"]: item for item in candidates.get("candidates", [])}
            selections, missing, _outcome = checks.select_variants(by_id, review_draft, tiers, settings)
            if missing:
                # A partial paper still ships its satisfiable tiers; the gap stays auditable.
                ledger.append({**entry, "gate": "missing_tier", "missing_tiers": list(missing)})
            reviews_by_id = {
                item["candidate_id"]: item
                for item in review_draft.get("reviews", [])
                if isinstance(item, dict) and isinstance(item.get("candidate_id"), str)
            }
            for tier in tiers:  # every requested tier, in the requested order
                candidate = selections.get(tier)
                rows.append(
                    {
                        **record,
                        "tier": tier,
                        "tier_selected": candidate is not None,
                        "selected_candidate_json": (
                            json.dumps(candidate, ensure_ascii=False, sort_keys=True) if candidate else ""
                        ),
                        "selected_review_json": (
                            json.dumps(
                                reviews_by_id.get(candidate["candidate_id"], {}), ensure_ascii=False, sort_keys=True
                            )
                            if candidate
                            else ""
                        ),
                        "selection_reason": str(review_draft.get("selection_reason") or ""),
                        "selection": columns.tier_selection_summary(
                            {tier: candidate} if candidate else {}, [] if candidate else [tier]
                        ),
                    }
                )
        atomic_write_json(stage_dir / reject_path_name, {"count": len(ledger), "rejected": ledger})
        if not rows:
            return _write_callback_frame(stage_dir, version, frame.iloc[0:0])
        return _write_callback_frame(stage_dir, version, _project(pd.DataFrame(rows), keep_columns))

    return callback


def _json_loads(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _jsonable(value: Any) -> Any:
    """JSON-friendly copy of a dataset value.

    Ledgers record values that came out of a frame, so they can be numpy scalars or arrays;
    iterating generically keeps ``[np.int64(1)]`` from being flattened into the string "[1]".
    """
    if isinstance(value, (str, bool)) or value is None:
        return value
    if hasattr(value, "item"):  # numpy scalar
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "__iter__"):
        try:
            return [_jsonable(item) for item in value]
        except TypeError:
            return str(value)
    return str(value)


# --------------------------------------------------------------------------------------
# materialization (the only filesystem writer)
# --------------------------------------------------------------------------------------


# The produced dataset's schema: the package pointer plus the provenance of every gate the row
# passed. Named at module level so `tests/test_column_lifecycle.py` can audit it as the final
# consumer of the columns the earlier stages carry.
MATERIALIZE_KEEP_COLUMNS: tuple[str, ...] = (
    "paper_id",
    "title",
    "package_name",
    "package_ok",
    "tier",
    "selection",
    "review_gate",
    "kg_passed",
)


def make_materialize_callback(
    *,
    version: str,
    packages_root: Path,
    keep_columns: tuple[str, ...] = MATERIALIZE_KEEP_COLUMNS,
    tier_namespaced: bool = False,
    paper_namespaced: bool = False,
) -> Callable[[Path], Path]:
    """Write every accepted row's package tree.

    With several requested tiers the packages go to ``<packages_root>/<tier>/<slug>``: the
    variants are independent generations and may legitimately pick the same slug, and Omega
    keeps the same ``task_packages/<tier>/`` layout (``pipeline.py:584-585``).
    """

    def callback(stage_dir: Path) -> Path:
        stage_dir = Path(stage_dir)
        frame = read_stage_frame(stage_dir)
        if frame.empty:
            return _write_callback_frame(stage_dir, version, frame)

        packages_root.mkdir(parents=True, exist_ok=True)
        package_dirs: list[str] = []
        written_counts: list[int] = []
        ledger: list[dict[str, Any]] = []
        for record in frame.to_dict(orient="records"):
            if record.get("package_ok") is not True:
                ledger.append(_package_rejection_entry(record))
                package_dirs.append("")
                written_counts.append(0)
                continue
            # The staging directory name must equal the slug in task.toml
            # [task].name; derive it from task.toml rather than trusting a
            # separately-generated package_name.
            name = _package_slug(record) or f"task-{record.get('paper_id', 'unknown')}"
            target = packages_root / str(record.get("tier")) / name if tier_namespaced else packages_root / name
            if paper_namespaced:
                # Heta batches independent papers; equal model-generated slugs must not overwrite.
                from hashlib import sha256
                paper_namespace = sha256(str(record.get("paper_id", "")).encode()).hexdigest()[:24]
                target = packages_root / paper_namespace / target.relative_to(packages_root)
            # Start clean: a leftover tree from an earlier run would otherwise survive
            # (and make the materialization gate report "differs from existing").
            shutil.rmtree(target, ignore_errors=True)
            try:
                files = columns.render_package_files(
                    json.loads(record.get("spec") or "{}"), json.loads(record.get("implementation") or "{}")
                )
                written = columns.materialize_package_files(files, target)
                manifest = _build_manifest(record, files)
            except Exception as error:  # noqa: BLE001 - a bad row must not abort the stage
                ledger.append(
                    {
                        "paper_id": record.get("paper_id"),
                        "title": record.get("title"),
                        "package_name": record.get("package_name"),
                        "errors": [f"materialization failed: {type(error).__name__}: {error}"],
                    }
                )
                shutil.rmtree(target, ignore_errors=True)
                package_dirs.append("")
                written_counts.append(0)
                continue
            atomic_write_json(target / "generation_manifest.json", manifest)
            package_dirs.append(str(target))
            written_counts.append(len(written) + 1)

        produced = frame.copy()
        produced["package_dir"] = package_dirs
        produced["package_written_files"] = written_counts
        keep = set(keep_columns) | {"package_dir", "package_written_files"}
        produced = produced[[c for c in produced.columns if c in keep]]
        # Rows that never produced a valid package leave the pipeline here; the ledger
        # keeps Omega's rejection audit without deleting the record.
        produced = produced[produced["package_dir"].astype(bool)].reset_index(drop=True)
        atomic_write_json(stage_dir / "rejected_package.json", {"count": len(ledger), "rejected": ledger})
        return _write_callback_frame(stage_dir, version, produced)

    return callback


def _package_rejection_entry(record: dict[str, Any]) -> dict[str, Any]:
    """Why a row left the pipeline at the materialization boundary.

    ``tier_selected`` distinguishes a tier that never had an eligible candidate (its authoring
    columns were skipped) from a tier whose authoring genuinely failed; both arrive here with
    ``package_ok`` unset, and without the flag the ledger reads like a build failure. The value
    comes back from a stage boundary, so it is coerced rather than compared with ``is False``.
    """
    tier_selected = record.get("tier_selected")
    return {
        "paper_id": record.get("paper_id"),
        "title": record.get("title"),
        "tier": record.get("tier"),
        "missing_candidate": bool(tier_selected is not None and not as_gate(tier_selected)),
        "package_name": record.get("package_name"),
        "errors": _jsonable(record.get("package_errors")),
        "spec_errors": _jsonable(record.get("spec_errors")),
        "implementation_errors": _jsonable(record.get("implementation_errors")),
        "implementation_check_errors": _jsonable(record.get("implementation_check_errors")),
    }


def _package_slug(record: dict[str, Any]) -> str:
    from heta_framework.generation.graph._native._compat import tomllib

    try:
        spec = json.loads(record.get("spec") or "{}")
        task_toml = (spec.get("files") or {}).get("task.toml", "")
        name = str(tomllib.loads(task_toml).get("task", {}).get("name", ""))
        slug = name.split("/")[-1].strip()
    except Exception:  # noqa: BLE001 - fall back to the model-declared name
        return str(record.get("package_name") or "")
    return slug


def _build_manifest(record: dict[str, Any], files: dict[str, str]) -> dict[str, Any]:
    from heta_framework.generation.graph._native.io_utils import canonical_json_sha256

    spec = json.loads(record["spec"])
    manifest = {
        "schema_version": "omegadesigner.task_package_manifest.v1",
        "status": "draft_requires_runtime_validation",
        "package_name": spec.get("package_name"),
        "paper_id": record.get("paper_id"),
        # Which difficulty variant this package is (Omega records the same key and refuses to
        # reuse a package whose tier differs).
        "difficulty_tier": record.get("tier"),
        "task_spec": spec.get("task_spec"),
        "data_contract": spec.get("data_contract"),
        "evaluation_topology": spec.get("evaluation_topology"),
        "difficulty_contract": spec.get("difficulty_contract"),
        "runtime_spec": spec.get("runtime_spec"),
        "scaffold_renderer": "omegadesigner.scaffold.v1",
        "validation": {
            "generation_contract_passed": True,
            "static_contract_passed": False,
            "data_materialized": False,
            "runtime_tests_run": False,
            "terminal_bench_ready": False,
            "next_step": "Run the vendored TBS static checks, then Harbor oracle/nop.",
        },
    }
    contract = {
        "task_spec": spec.get("task_spec"),
        "data_contract": spec.get("data_contract"),
        "evaluation_topology": spec.get("evaluation_topology"),
        "difficulty_contract": spec.get("difficulty_contract"),
        "runtime_spec": spec.get("runtime_spec"),
        "instruction_md": spec.get("files", {}).get("instruction.md"),
    }
    manifest["scientific_contract_sha256"] = canonical_json_sha256(contract)
    # The file map is re-derived rather than carried as a column, so its identity is pinned
    # here: the hash ties the manifest to the exact tree that was written to disk.
    manifest["files_sha256"] = canonical_json_sha256(files)
    manifest["file_count"] = len(files)
    return manifest


# --------------------------------------------------------------------------------------
# stage configs
# --------------------------------------------------------------------------------------


def _base_builder(settings: Settings, model_configs: list[dd.ModelConfig]) -> dd.DataDesignerConfigBuilder:
    """Every stage's builder. Concurrency stays with Data Designer's own controls
    (request admission / per-model parallelism); this project only shapes the columns."""
    builder = dd.DataDesignerConfigBuilder()
    for config in model_configs:
        builder.add_model_config(config)
    return builder


def stage_screen(
    settings: Settings, model_configs: list[dd.ModelConfig], seed_path: Path
) -> dd.DataDesignerConfigBuilder:
    builder = _base_builder(settings, model_configs)
    builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed_path)))
    builder.add_column(
        dd.CustomColumnConfig(name="kg_screen", generator_function=columns.kg_screen, generator_params=settings)
    )
    return builder


def stage_review(settings: Settings, model_configs: list[dd.ModelConfig]) -> dd.DataDesignerConfigBuilder:
    builder = _base_builder(settings, model_configs)
    builder.add_column(
        dd.CustomColumnConfig(
            name="semantic_review", generator_function=columns.semantic_review, generator_params=settings
        )
    )
    builder.add_column(
        dd.CustomColumnConfig(name="review_gate", generator_function=columns.review_gate, generator_params=settings)
    )
    return builder


def stage_candidates(settings: Settings, model_configs: list[dd.ModelConfig]) -> dd.DataDesignerConfigBuilder:
    builder = _base_builder(settings, model_configs)
    builder.add_column(
        dd.CustomColumnConfig(name="candidates", generator_function=columns.candidates, generator_params=settings)
    )
    return builder


def stage_candidate_review(settings: Settings, model_configs: list[dd.ModelConfig]) -> dd.DataDesignerConfigBuilder:
    builder = _base_builder(settings, model_configs)
    builder.add_column(
        dd.CustomColumnConfig(
            name="candidate_review",
            generator_function=columns.candidate_review,
            generator_params=settings,
            # Gating keeps a permanently failed step from re-running downstream prompts.
            skip=dd.SkipConfig(when="{{ candidates_ok == false }}", value=None),
        )
    )
    builder.add_column(
        dd.CustomColumnConfig(name="selection", generator_function=columns.selection, generator_params=settings)
    )
    return builder


def stage_spec(settings: Settings, model_configs: list[dd.ModelConfig]) -> dd.DataDesignerConfigBuilder:
    builder = _base_builder(settings, model_configs)
    builder.add_column(
        dd.CustomColumnConfig(
            name="spec",
            generator_function=columns.spec,
            generator_params=settings,
            # A tier with no eligible candidate has nothing to specify: skip it so it costs no
            # model call, and let skip propagation carry it to the materialization ledger.
            skip=dd.SkipConfig(when="{{ tier_selected == false }}", value=None),
        )
    )
    return builder


def stage_implementation(settings: Settings, model_configs: list[dd.ModelConfig]) -> dd.DataDesignerConfigBuilder:
    builder = _base_builder(settings, model_configs)
    # Draft + check-and-correct are ONE column: the draft has no consumer of its own (see
    # ``columns.implementation``) and Data Designer cannot drop a real column mid-stage.
    builder.add_column(
        dd.CustomColumnConfig(
            name="implementation",
            generator_function=columns.implementation,
            generator_params=settings,
            skip=dd.SkipConfig(when="{{ spec_ok == false }}", value=None),
        )
    )
    # No `package_files` column: the file map is a deterministic render of the two drafts and
    # was 43 KB/row carried only between `package_check` and the materialize callback, both of
    # which re-derive it (see ``columns.render_package_files``).
    builder.add_column(
        dd.CustomColumnConfig(name="package_check", generator_function=columns.package_check, generator_params=settings)
    )
    return builder


def stage_static_checks(settings: Settings, model_configs: list[dd.ModelConfig]) -> dd.DataDesignerConfigBuilder:
    builder = _base_builder(settings, model_configs)
    builder.add_column(
        dd.CustomColumnConfig(name="static_check", generator_function=columns.static_check, generator_params=settings)
    )
    return builder


def stage_validate_runtime(settings: Settings, model_configs: list[dd.ModelConfig]) -> dd.DataDesignerConfigBuilder:
    """Runtime validation as columns: Docker materialization, Harbor oracle/nop, verdict.

    Ordering is expressed with ``required_columns`` (and a ``skip.when`` on the Harbor column),
    so a package that fails to materialize never pays for a Harbor trial. All three columns are
    cell_by_cell: one task per row means a slow trial only holds its own row, and the scheduler's
    row-group concurrency is the only knob that bounds parallel Docker/Harbor work.
    """
    builder = _base_builder(settings, model_configs)
    builder.add_column(
        dd.CustomColumnConfig(
            name="runtime_materialize",
            generator_function=validation.runtime_materialize,
            generator_params=settings,
        )
    )
    builder.add_column(
        dd.CustomColumnConfig(
            name="harbor_validation",
            generator_function=validation.harbor_validation,
            generator_params=settings,
            skip=dd.SkipConfig(when="{{ materialize_ok == false }}", value=None),
        )
    )
    builder.add_column(
        dd.CustomColumnConfig(
            name="terminal_bench_ready",
            generator_function=validation.terminal_bench_ready,
            generator_params=settings,
            # A summary column that reads "may be skipped" inputs must not skip-propagate:
            # otherwise the rows that failed materialization get a skip marker instead of the
            # `terminal_bench_ready=false` verdict (and their report) they exist to record.
            propagate_skip=False,
        )
    )
    return builder


STAGE_NAMES = (
    "screen",
    "review",
    "candidates",
    "candidate_review",
    "spec",
    "implementation",
    "validate_static",
    "validate_runtime",
)


def build_workflow(
    data_designer: Any,
    settings: Settings,
    *,
    seed_path: Path,
    model_configs: list[dd.ModelConfig],
    workflow_name: str,
    packages_root: Path,
    num_records: int | None = None,
    targets: str | None = None,
    paper_namespaced: bool = False,
) -> Any:
    """Assemble the composite workflow. ``targets`` limits execution (e.g. ``"screen"``)."""
    workflow = data_designer.compose_workflow(name=workflow_name)

    workflow.add_stage(
        "screen",
        stage_screen(settings, model_configs, seed_path),
        # One output row per seed row: without this Data Designer's default of 10
        # would cycle the seed and generate duplicates.
        num_records=num_records,
        allow_empty=True,
        on_success=make_filter_callback(
            "kg_passed",
            version="screen-filter-v3",
            reject_path_name="rejected_structure.json",
            reason_columns=("kg_failed_checks", "kg_metrics"),
            keep_columns=SCREEN_KEEP,
        ),
        on_success_version="screen-filter-v3",
    )
    workflow.add_stage(
        "review",
        stage_review(settings, model_configs),
        allow_empty=True,
        on_success=make_filter_callback(
            "review_suitable",
            version="review-filter-v3",
            reject_path_name="rejected_semantic.json",
            reason_columns=("review_gate_json",),
            keep_columns=REVIEW_KEEP,
        ),
        on_success_version="review-filter-v3",
    )
    workflow.add_stage(
        "candidates",
        stage_candidates(settings, model_configs),
        allow_empty=True,
        on_success=make_projection_callback(CANDIDATES_KEEP, version="candidates-project-v2"),
        on_success_version="candidates-project-v2",
    )
    workflow.add_stage(
        "candidate_review",
        stage_candidate_review(settings, model_configs),
        allow_empty=True,
        on_success=make_tier_fanout_callback(
            settings,
            version="tier-fanout-v1",
            reject_path_name="rejected_difficulty.json",
            keep_columns=CANDIDATE_REVIEW_KEEP,
        ),
        on_success_version="tier-fanout-v1",
    )
    workflow.add_stage(
        "spec",
        stage_spec(settings, model_configs),
        allow_empty=True,
        on_success=make_projection_callback(SPEC_KEEP, version="spec-project-v2"),
        on_success_version="spec-project-v2",
    )
    workflow.add_stage(
        "implementation",
        stage_implementation(settings, model_configs),
        allow_empty=True,
        on_success=make_materialize_callback(
            version="heta-materialize-v1" if paper_namespaced else "materialize-v3",
            packages_root=packages_root,
            tier_namespaced=len(settings.difficulty.tiers) > 1,
            paper_namespaced=paper_namespaced,
        ),
        on_success_version="heta-materialize-v1" if paper_namespaced else "materialize-v3",
    )
    workflow.add_stage(
        "validate_static",
        stage_static_checks(settings, model_configs),
        allow_empty=True,
        on_success=make_runtime_seed_callback(version="static-runtime-seed-v1"),
        on_success_version="static-runtime-seed-v1",
    )
    workflow.add_stage(
        "validate_runtime",
        stage_validate_runtime(settings, model_configs),
        allow_empty=True,
        on_success_version="validate-runtime-v1",
    )
    return workflow
