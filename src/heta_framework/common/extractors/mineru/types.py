"""MinerU extractor configuration and artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


MinerULocalApiMode = Literal["tasks", "file_parse"]
MinerUProvider = Literal["cloud", "local"]


@dataclass(frozen=True)
class MinerUClientConfig:
    """Configuration for MinerU Cloud or a local MinerU service."""

    provider: MinerUProvider
    api_key: str | None = None
    endpoint_url: str | None = None
    cloud_base_url: str = "https://mineru.net/api/v4"
    local_api_mode: MinerULocalApiMode = "tasks"
    request_timeout: float = 120.0
    parse_timeout: float = 300.0
    poll_interval: float = 2.0

    def __post_init__(self) -> None:
        if self.provider not in {"cloud", "local"}:
            raise ValueError("provider must be 'cloud' or 'local'")
        if self.provider == "cloud" and not (self.api_key and self.api_key.strip()):
            raise ValueError("api_key is required for MinerU cloud")
        if self.provider == "local" and not (self.endpoint_url and self.endpoint_url.strip()):
            raise ValueError("endpoint_url is required for local MinerU")
        if self.local_api_mode not in {"tasks", "file_parse"}:
            raise ValueError("local_api_mode must be 'tasks' or 'file_parse'")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero")
        if self.parse_timeout <= 0:
            raise ValueError("parse_timeout must be greater than zero")
        if self.poll_interval <= 0:
            raise ValueError("poll_interval must be greater than zero")


@dataclass(frozen=True)
class MinerUParseOptions:
    """MinerU-specific parse options."""

    language: str = "ch"
    backend: str = "hybrid-auto-engine"
    parse_method: str = "auto"
    model_version: str = "vlm"
    effort: str = "medium"
    enable_table: bool = True
    enable_formula: bool = True
    image_analysis: bool = True
    include_images: bool = True
    start_page_id: int = 0
    end_page_id: int | None = None

    def __post_init__(self) -> None:
        if self.language.strip() == "":
            raise ValueError("language must not be empty")
        if self.backend.strip() == "":
            raise ValueError("backend must not be empty")
        if self.parse_method.strip() == "":
            raise ValueError("parse_method must not be empty")
        if self.model_version.strip() == "":
            raise ValueError("model_version must not be empty")
        if self.effort.strip() == "":
            raise ValueError("effort must not be empty")
        if self.start_page_id < 0:
            raise ValueError("start_page_id must not be negative")
        if self.end_page_id is not None and self.end_page_id < self.start_page_id:
            raise ValueError("end_page_id must be greater than or equal to start_page_id")


@dataclass(frozen=True)
class MinerUArtifact:
    """Raw structured output extracted from a MinerU zip response."""

    markdown: str
    content_list: tuple[dict[str, Any], ...] = ()
    content_list_v2: tuple[tuple[dict[str, Any], ...], ...] = ()
    images: dict[str, bytes] | None = None

    def __post_init__(self) -> None:
        if self.markdown.strip() == "":
            raise ValueError("markdown must not be empty")
