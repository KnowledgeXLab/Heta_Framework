"""Internal model outputs, independent of the public producer protocol."""

from typing import Literal

from pydantic import BaseModel, Field


class SourceEvidence(BaseModel):
    file_path: str = Field(min_length=1)
    finding: str = Field(min_length=1)


class TaskPattern(BaseModel):
    workflow: list[str] = Field(min_length=1)
    capabilities: list[str] = Field(min_length=1)
    scientific_outputs: list[str] = Field(min_length=1)
    hardness_knobs: list[str]
    derivation_rules: list[str] = Field(min_length=1)
    source_evidence: list[SourceEvidence] = Field(min_length=1)


class TaskFile(BaseModel):
    path: str
    content: str


class Candidate(BaseModel):
    title: str
    domain: Literal[
        "life-sciences",
        "physical-sciences",
        "earth-sciences",
        "mathematical-sciences",
        "engineering-sciences",
    ]
    field: str
    description: str
    instruction_md: str
    artifacts: list[str] = Field(min_length=1)
    public_files: list[TaskFile]
    files: list[TaskFile] = Field(min_length=1)
    incorrect_submission: str = Field(min_length=1)
    incorrect_submission_rationale: str = Field(min_length=1)


class NoveltyReview(BaseModel):
    preserves_capabilities: bool
    independent_task: bool
    findings: list[str]
