"""Configuration for evidence-grounded task generation from a KB or prepared objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from heta_framework.generation.derivation.config import HetaGenTaskProducerDerivationConfig


@dataclass(frozen=True)
class HetaGenTaskProducerGraphConfig:
    input_mode: Literal["knowledge_base", "objects"] = "knowledge_base"
    query_mode: str = "heta_graph_search"
    top_k: int = 20
    max_documents: int = 3
    generation_guidance: str = ""
    model: str = "bailian/glm-5.3"
    review_model: str | None = None
    api_base_url: str | None = None
    reasoning_effort: str | None = None
    max_tokens: int = 32000
    request_timeout_sec: int = 600
    max_concurrency: int = 1
    settings_file: Path | None = None
    env_file: Path = Path(".env")
    workspace: Path | None = None
    parsed_documents_artifact: str = "parsed_document_keys"
    entities_artifact: str = "entity_keys"
    relations_artifact: str = "relation_keys"
    chunks_artifact: str = "chunk_keys"
    entity_type_mapping: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_mode not in {"knowledge_base", "objects"}:
            raise ValueError("input_mode must be knowledge_base or objects")
        for name in (
            "top_k",
            "max_documents",
            "max_tokens",
            "request_timeout_sec",
            "max_concurrency",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "query_mode",
            "model",
            "parsed_documents_artifact",
            "entities_artifact",
            "relations_artifact",
            "chunks_artifact",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty text")
        if not isinstance(self.generation_guidance, str):
            raise TypeError("generation_guidance must be text")
        native_groups = {
            "Tasks",
            "Methods",
            "Datasets",
            "Metrics",
            "Baselines",
            "Implementation",
            "Theorems",
            "Figures_Tables",
            "Domain_Terms",
        }
        if any(
            not isinstance(key, str) or not key.strip() or value not in native_groups
            for key, value in self.entity_type_mapping.items()
        ):
            raise ValueError(
                "entity_type_mapping maps nonempty entity types to native Omega groups"
            )
        mapping = {key.upper(): value for key, value in self.entity_type_mapping.items()}
        if len(mapping) != len(self.entity_type_mapping):
            raise ValueError("entity_type_mapping contains case-insensitive duplicate keys")
        object.__setattr__(self, "entity_type_mapping", MappingProxyType(mapping))
        if self.review_model is not None and (
            not isinstance(self.review_model, str) or not self.review_model.strip()
        ):
            raise ValueError("review_model must be nonempty text or None")
        # Share the existing endpoint and inference-parameter validation contract.
        HetaGenTaskProducerDerivationConfig(
            api_base_url=self.api_base_url,
            reasoning_effort=self.reasoning_effort,
        )
        for name in ("env_file", "settings_file", "workspace"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
