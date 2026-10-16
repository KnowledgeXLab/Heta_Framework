# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""POC settings.

Everything that influences a gate or a prompt lives here so it can be (a) passed
into Data Designer as ``generator_params`` and therefore land in the stage
fingerprint, and (b) diffed against Omega's ``config/science.default.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DifficultyTier = Literal["simple", "medium", "complex"]

# Omega ``_DIFFICULTY_TIERS`` (pipeline.py:36-40): a candidate carries one integer
# difficulty and only one band can claim it, so the same candidate never spans tiers.
DIFFICULTY_TIER_BANDS: dict[str, set[int]] = {
    "simple": {1, 2},
    "medium": {3},
    "complex": {4, 5},
}


class ScreenThresholds(BaseModel):
    """Structural gate thresholds (Omega ``settings["structure"]``)."""

    model_config = ConfigDict(extra="forbid")

    min_component_nodes: int = Field(default=20, ge=1)
    min_component_edges: int = Field(default=20, ge=0)
    min_component_node_fraction: float = Field(default=0.35, ge=0.0, le=1.0)
    min_relation_types: int = Field(default=3, ge=0)
    min_b_anchor_nodes: int = Field(default=1, ge=0)


class CandidateThresholds(BaseModel):
    """Candidate review gates and fan-out size."""

    model_config = ConfigDict(extra="forbid")

    candidate_count: int = Field(default=5, ge=3, le=8)
    minimum_candidate_score: float = Field(default=0.75, ge=0.0, le=1.0)
    minimum_candidate_dimension_score: int = Field(default=3, ge=1, le=5)
    preferred_candidate_ids: dict[str, str] = Field(default_factory=dict)


class ReviewThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_suitability_score: float = Field(default=0.7, ge=0.0, le=1.0)


class TruncationLimits(BaseModel):
    """Character budgets for prompt payloads (Omega ``*_max_*_chars``)."""

    model_config = ConfigDict(extra="forbid")

    cheap_max_content_chars: int = 60_000
    cheap_max_component_chars: int = 40_000
    strong_candidate_max_content_chars: int = 80_000
    strong_candidate_max_component_chars: int = 40_000
    max_generated_files: int = 40
    max_generated_file_chars: int = 200_000


class RepairPolicy(BaseModel):
    """Omega validation-attempt semantics, mapped onto DD correction steps.

    ``max_correction_steps`` is *corrections after the first attempt*, so Omega's
    ``*_validation_attempts = 3`` becomes ``2`` here.
    """

    model_config = ConfigDict(extra="forbid")

    validation_attempts: int = Field(default=3, ge=1, le=5)
    implementation_check_attempts: int = Field(default=1, ge=1, le=5)

    @property
    def correction_steps(self) -> int:
        return self.validation_attempts - 1

    @property
    def implementation_check_correction_steps(self) -> int:
        return self.implementation_check_attempts - 1


class ValidationSettings(BaseModel):
    """Runtime validation: Docker data materialization + Harbor oracle/nop.

    Lives in ``Settings`` because it is passed to the validation columns as
    ``generator_params``, which puts it in the stage fingerprint: changing the number of
    oracle repeats invalidates the stage instead of silently reusing verdicts from another
    configuration.
    """

    model_config = ConfigDict(extra="forbid")

    oracle_repeats: int = Field(default=2, ge=1, le=5)
    harbor_env: str = "docker"


class ModelAliases(BaseModel):
    """DD model aliases used by the columns.

    Kept as documentation of the contract: the alias strings themselves are module
    constants in :mod:`omegadesigner.dd.columns` because ``@custom_column_generator``
    metadata is evaluated at import time and therefore cannot depend on settings.
    Generation/ingest deliberately omit temperature, preserving the Omega contract.
    Search can opt in independently via OD_SEARCH_LLM_TEMPERATURE.
    """

    model_config = ConfigDict(extra="forbid")

    cheap: str = "cheap"
    strong: str = "strong"
    strong_creative: str = "strong_creative"
    strong_deterministic: str = "strong_deterministic"
    strong_no_retry: str = "strong_no_retry"


class DifficultySettings(BaseModel):
    """Which difficulty variants to generate from one paper.

    Omega generates one complete task package per requested tier (``variants/<tier>/``,
    ``task_packages/<tier>/``) and records tiers with no eligible candidate as
    ``missing_tiers`` rather than failing the paper. ``tiers`` keeps that order stable,
    which matters because every packaged artifact is addressed by it.
    """

    model_config = ConfigDict(extra="forbid")

    tiers: tuple[DifficultyTier, ...] = ("complex",)

    @field_validator("tiers")
    @classmethod
    def _tiers_are_known_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("difficulty tiers must not be empty; Omega rejects an empty list too")
        unknown = sorted(set(value) - set(DIFFICULTY_TIER_BANDS))
        if unknown:
            raise ValueError(f"unknown difficulty tiers: {', '.join(unknown)}")
        if len(set(value)) != len(value):
            raise ValueError("difficulty tiers must not contain duplicates")
        return value

    @property
    def tier(self) -> str:
        """The primary tier: what a single-variant payload shows as ``difficulty_tier``."""
        return self.tiers[0]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation_guidance: str = ""

    screen: ScreenThresholds = Field(default_factory=ScreenThresholds)
    candidates: CandidateThresholds = Field(default_factory=CandidateThresholds)
    review: ReviewThresholds = Field(default_factory=ReviewThresholds)
    limits: TruncationLimits = Field(default_factory=TruncationLimits)
    repair: RepairPolicy = Field(default_factory=RepairPolicy)
    difficulty: DifficultySettings = Field(default_factory=DifficultySettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    task_slug_root: str = "tasks"
    # Must be a canonical TBS domain and a field present in reviewer-pool.yml;
    # the vendored check-task-path.sh enforces both against the package path.
    task_domain: str = "physical-sciences"
    task_field: str = "physics"

    @classmethod
    def load(cls, path: str | Path | None) -> Settings:
        if path is None:
            return cls()
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
