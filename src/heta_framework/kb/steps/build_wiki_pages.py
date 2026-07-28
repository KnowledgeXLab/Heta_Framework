"""Build readable Wiki pages from parsed documents."""

from __future__ import annotations

import json
import posixpath
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

from heta_framework.common.models import ModelOptions, ModelRequest
from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import (
    join_object_key,
    validate_object_key,
    validate_object_prefix,
)
from heta_framework.kb._wiki import WikiPage, parse_wiki_page, render_wiki_page
from heta_framework.kb.chunking.splitters import get_text_encoding
from heta_framework.kb.cleanup import StepCleanupPlan, object_key_targets
from heta_framework.kb.parsing import ParsedDocument, ParsedSource
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import (
    IssueResolution,
    IssueSubject,
    StepCapabilities,
    StepIssue,
    StepRequirements,
    model_ref,
    store_ref,
)

WikiSummaryMode = Literal["model", "extractive"]
OversizedDocumentPolicy = Literal["fail", "skip"]


class WikiDocumentLimitError(ValueError):
    """Raised when one parsed document exceeds the configured Wiki build budget."""


@dataclass(frozen=True)
class BuildWikiPagesConfig:
    """Configuration for BuildWikiPages."""

    pages_prefix: str = "wiki/pages"
    index_key: str = "wiki/index.md"
    log_key: str = "wiki/page_log.json"
    summary_mode: WikiSummaryMode = "model"
    summary_input_chars: int = 12_000
    summary_max_output_tokens: int = 512
    summary_max_attempts: int = 3
    max_document_pages: int = 80
    max_document_tokens: int = 262_144
    encoding_name: str = "cl100k_base"
    oversized_document_policy: OversizedDocumentPolicy = "fail"
    object_store: str | None = None
    language_model: str | None = None
    parsed_document_keys_artifact: str = "parsed_document_keys"

    def __post_init__(self) -> None:
        validate_object_prefix(self.pages_prefix)
        validate_object_key(self.index_key)
        validate_object_key(self.log_key)
        if self.summary_mode not in {"model", "extractive"}:
            raise ValueError("summary_mode must be one of: model, extractive")
        if self.summary_input_chars <= 0:
            raise ValueError("summary_input_chars must be greater than zero")
        if self.summary_max_output_tokens <= 0:
            raise ValueError("summary_max_output_tokens must be greater than zero")
        if self.summary_max_attempts <= 0:
            raise ValueError("summary_max_attempts must be greater than zero")
        if self.max_document_pages <= 0:
            raise ValueError("max_document_pages must be greater than zero")
        if self.max_document_tokens <= 0:
            raise ValueError("max_document_tokens must be greater than zero")
        if self.encoding_name.strip() == "":
            raise ValueError("encoding_name must not be empty")
        if self.oversized_document_policy not in {"fail", "skip"}:
            raise ValueError("oversized_document_policy must be one of: fail, skip")
        if self.parsed_document_keys_artifact.strip() == "":
            raise ValueError("parsed_document_keys_artifact must not be empty")


@dataclass(frozen=True)
class BuildWikiPagesResult:
    """Artifacts produced by BuildWikiPages."""

    page_keys: tuple[str, ...]
    index_key: str
    log_key: str
    document_count: int
    page_count: int
    summary_mode: WikiSummaryMode
    summary_model: str | None = None
    skipped_document_keys: tuple[str, ...] = ()
    issues: tuple[StepIssue, ...] = ()


class BuildWikiPages:
    """Convert parsed documents into readable and resumable Wiki pages."""

    name = "build_wiki_pages"

    def __init__(self, config: BuildWikiPagesConfig | None = None) -> None:
        self.config = config or BuildWikiPagesConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components and artifacts required by this step."""
        components = {store_ref("objects", self.config.object_store)}
        if self.config.summary_mode == "model":
            components.add(model_ref("language", self.config.language_model))
        return StepRequirements(
            components=frozenset(components),
            artifacts=frozenset({self.config.parsed_document_keys_artifact}),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts produced by this step."""
        return StepCapabilities(
            artifacts=frozenset(
                {
                    "build_wiki_pages_result",
                    "wiki_page_keys",
                    "wiki_index_key",
                    "wiki_log_key",
                }
            )
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        """Return Wiki page, index, and log objects produced by this step."""
        object_store_ref = store_ref("objects", self.config.object_store).key
        return StepCleanupPlan(
            (
                *object_key_targets(
                    artifacts,
                    "wiki_page_keys",
                    component=object_store_ref,
                ),
                *object_key_targets(
                    artifacts,
                    "wiki_index_key",
                    component=object_store_ref,
                ),
                *object_key_targets(
                    artifacts,
                    "wiki_log_key",
                    component=object_store_ref,
                ),
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Build Wiki pages and their navigation artifacts."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        language_model = None
        if self.config.summary_mode == "model":
            language_model = _require_language_model(
                context.get_component(model_ref("language", self.config.language_model).key)
            )

        parsed_document_keys = tuple(
            context.get_artifact(self.config.parsed_document_keys_artifact)
        )
        pages: list[WikiPage] = []
        skipped_document_keys: list[str] = []
        issues: list[StepIssue] = []

        for page_number, document_key in enumerate(parsed_document_keys, start=1):
            document = ParsedDocument.from_json(await object_store.get(document_key))
            source_content = _document_content(document)
            violation = _document_limit_violation(
                document,
                content=source_content,
                config=self.config,
            )
            if violation is not None:
                if self.config.oversized_document_policy == "fail":
                    raise WikiDocumentLimitError(violation)
                skipped_document_keys.append(document_key)
                issues.append(_document_limit_issue(document, violation=violation))
                continue

            title, content_without_title = _title_and_content(document, source_content)
            content = _normalize_content(content_without_title)
            page_id, page_key = _page_identity(
                document,
                title=title,
                page_number=page_number,
                config=self.config,
            )
            page = await _load_existing_page(
                document,
                page_id=page_id,
                page_key=page_key,
                object_store=object_store,
            )
            if page is None:
                summary = await _summarize(
                    title=title,
                    content=source_content,
                    source=document.source,
                    language_model=language_model,
                    config=self.config,
                )
                page = WikiPage(
                    page_id=page_id,
                    title=title,
                    summary=summary,
                    content=content,
                    source=document.source,
                    key=page_key,
                )
                await object_store.put(page.key, render_wiki_page(page).encode("utf-8"))
            pages.append(page)

        await object_store.put(
            self.config.index_key,
            _render_index(pages, index_key=self.config.index_key).encode("utf-8"),
        )
        await object_store.put(
            self.config.log_key,
            json.dumps(_log_payload(pages), ensure_ascii=False, indent=2).encode("utf-8"),
        )

        result = BuildWikiPagesResult(
            page_keys=tuple(page.key for page in pages),
            index_key=self.config.index_key,
            log_key=self.config.log_key,
            document_count=len(parsed_document_keys),
            page_count=len(pages),
            summary_mode=self.config.summary_mode,
            summary_model=language_model.model_name if language_model is not None else None,
            skipped_document_keys=tuple(skipped_document_keys),
            issues=tuple(issues),
        )
        context.set_artifact("build_wiki_pages_result", result)
        context.set_artifact("wiki_page_keys", result.page_keys)
        context.set_artifact("wiki_index_key", result.index_key)
        context.set_artifact("wiki_log_key", result.log_key)


async def _load_existing_page(
    document: ParsedDocument,
    *,
    page_id: str,
    page_key: str,
    object_store: ObjectStoreProtocol,
) -> WikiPage | None:
    if not await object_store.exists(page_key):
        return None
    try:
        page = parse_wiki_page(await object_store.get(page_key), key=page_key)
    except (TypeError, ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if page.page_id != page_id or page.source != document.source:
        return None
    return page


async def _summarize(
    *,
    title: str,
    content: str,
    source: ParsedSource,
    language_model: LanguageModelProtocol | None,
    config: BuildWikiPagesConfig,
) -> str:
    if config.summary_mode == "extractive":
        return _extractive_summary(content)
    if language_model is None:
        raise TypeError("summary_mode='model' requires a language model")

    last_error: Exception | None = None
    for attempt in range(1, config.summary_max_attempts + 1):
        try:
            result = await language_model.invoke(
                ModelRequest(
                    system_prompt=(
                        "You write concise factual wiki summaries. "
                        "Return one short paragraph only."
                    ),
                    prompt=(
                        f"Title: {title}\n"
                        f"Source: {source.name}\n\n"
                        "Summarize the document for a retrieval wiki page.\n\n"
                        f"{content[: config.summary_input_chars]}"
                    ),
                    options=ModelOptions(
                        temperature=0,
                        max_output_tokens=config.summary_max_output_tokens,
                    ),
                    trace_context={
                        "step": "build_wiki_pages",
                        "stage": "summary",
                        "attempt": attempt,
                    },
                )
            )
            summary = _normalize_summary(result.text)
            if summary:
                return summary
            last_error = ValueError("language model returned an empty Wiki summary")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    assert last_error is not None
    raise last_error


def _document_limit_violation(
    document: ParsedDocument,
    *,
    content: str,
    config: BuildWikiPagesConfig,
) -> str | None:
    page_count = len(document.pages)
    if page_count > config.max_document_pages:
        return (
            f"Wiki document exceeds max_document_pages: source={document.source.key}, "
            f"pages={page_count}, limit={config.max_document_pages}"
        )
    token_count = len(get_text_encoding(config.encoding_name).encode(content))
    if token_count > config.max_document_tokens:
        return (
            f"Wiki document exceeds max_document_tokens: source={document.source.key}, "
            f"tokens={token_count}, limit={config.max_document_tokens}"
        )
    return None


def _document_limit_issue(document: ParsedDocument, *, violation: str) -> StepIssue:
    return StepIssue(
        step="build_wiki_pages",
        subject=IssueSubject(type="parsed_document", id=document.document_id),
        code="wiki_document_limit_exceeded",
        severity="error",
        message=violation,
        resolution=IssueResolution(
            action="skip_document",
            outcome="No Wiki page was generated; partition the source before rebuilding.",
        ),
        details={"source_key": document.source.key},
    )


def _title_and_content(document: ParsedDocument, content: str) -> tuple[str, str]:
    lines = content.splitlines()
    in_code = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if _is_fence(stripped):
            in_code = not in_code
            continue
        match = None if in_code else re.match(r"^#\s+(.+?)\s*#*\s*$", line)
        if match is None:
            continue
        title = match.group(1).strip()
        remaining = "\n".join([*lines[:index], *lines[index + 1 :]]).strip()
        return title, remaining or "(empty document)"
    return _title_from_filename(document.source.name), content.strip() or "(empty document)"


def _normalize_content(content: str) -> str:
    lines: list[str] = []
    in_code = False
    for line in content.splitlines():
        stripped = line.strip()
        if _is_fence(stripped):
            in_code = not in_code
            lines.append(line)
            continue
        match = None if in_code else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match is not None:
            level = min(max(3, len(match.group(1)) + 2), 6)
            line = f"{'#' * level} {match.group(2).strip()}"
        lines.append(line)
    return "\n".join(lines).strip() or "(empty document)"


def _document_content(document: ParsedDocument) -> str:
    original_content = document.original_content
    if (
        original_content is not None
        and original_content.media_type == "text/markdown"
        and original_content.text.strip()
    ):
        return original_content.text.strip()

    pages = sorted(document.pages, key=lambda page: page.page_index)
    parts: list[str] = []
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        if len(pages) == 1:
            parts.append(text)
        else:
            parts.append(f"## Page {page.page_index + 1}\n\n{text}")
    return "\n\n".join(parts).strip() or "(empty document)"


def _page_identity(
    document: ParsedDocument,
    *,
    title: str,
    page_number: int,
    config: BuildWikiPagesConfig,
) -> tuple[str, str]:
    slug = _slugify(title, fallback=document.document_id)
    return (
        f"wiki_page_{page_number}",
        join_object_key(config.pages_prefix, f"{page_number}-{slug}.md"),
    )


def _render_index(pages: list[WikiPage], *, index_key: str) -> str:
    lines = ["# Wiki Index", ""]
    index_directory = PurePosixPath(index_key).parent.as_posix()
    for page in pages:
        link = posixpath.relpath(page.key, start=index_directory)
        lines.append(f"- [{page.title}]({link}) — {page.summary}")
    lines.append("")
    return "\n".join(lines)


def _log_payload(pages: list[WikiPage]) -> dict[str, Any]:
    return {
        "pages": [
            {
                "page_id": page.page_id,
                "title": page.title,
                "summary": page.summary,
                "key": page.key,
                "source": {
                    "key": page.source.key,
                    "name": page.source.name,
                    "file_type": page.source.file_type,
                    "content_sha256": page.source.content_sha256,
                },
            }
            for page in pages
        ]
    }


def _extractive_summary(content: str) -> str:
    normalized = " ".join(content.split())
    if len(normalized) <= 280:
        return normalized
    return normalized[:277].rstrip() + "..."


def _normalize_summary(text: str) -> str:
    lines = [line.strip(" -\t") for line in text.splitlines()]
    return " ".join(line for line in lines if line).strip()


def _title_from_filename(name: str) -> str:
    stem = PurePosixPath(name).stem
    cleaned = re.sub(r"[_-]+", " ", stem).strip()
    return cleaned or name


def _slugify(text: str, *, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    slug = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", normalized).strip("-")
    if not slug:
        slug = re.sub(r"[^a-z0-9]+", "-", fallback.lower()).strip("-")
    return (slug or "page")[:80].strip("-") or "page"


def _is_fence(line: str) -> bool:
    return line.startswith("```") or line.startswith("~~~")


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component


def _require_language_model(component: object) -> LanguageModelProtocol:
    if not isinstance(component, LanguageModelProtocol):
        raise TypeError("models.language must satisfy LanguageModelProtocol")
    return component
