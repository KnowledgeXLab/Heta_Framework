"""Split generated Wiki pages into heading-aware retrieval chunks."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.common.stores.object.types import join_object_key, validate_object_prefix
from heta_framework.kb._wiki import WikiPage, parse_wiki_page, wiki_page_source
from heta_framework.kb.chunking import ParsedChunk, make_chunk_id, split_text
from heta_framework.kb.chunking.splitters import get_text_encoding
from heta_framework.kb.cleanup import StepCleanupPlan, object_key_targets
from heta_framework.kb.steps.protocols import StepContextProtocol
from heta_framework.kb.steps.types import (
    IssueResolution,
    IssueSubject,
    StepCapabilities,
    StepIssue,
    StepRequirements,
    store_ref,
)

OversizedWikiPagePolicy = Literal["fail", "skip"]
_HEADING_RE = re.compile(r"^(#{3,6})\s+(.+?)\s*#*\s*$")


class WikiPageChunkLimitError(ValueError):
    """Raised when one Wiki page exceeds its configured chunk budget."""


@dataclass(frozen=True)
class SplitWikiPagesConfig:
    """Configuration for SplitWikiPages."""

    chunks_prefix: str = "wiki_chunks"
    chunk_size: int = 1024
    overlap: int = 50
    encoding_name: str = "cl100k_base"
    split_punctuation: tuple[str, ...] = (
        "\n\n",
        "\n",
        "。",
        ".",
        ",",
        "，",
        "!",
        "?",
        "！",
        "？",
    )
    max_chunks_per_page: int = 256
    oversized_page_policy: OversizedWikiPagePolicy = "fail"
    object_store: str | None = None
    wiki_page_keys_artifact: str = "wiki_page_keys"

    def __post_init__(self) -> None:
        validate_object_prefix(self.chunks_prefix)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.overlap < 0:
            raise ValueError("overlap must not be negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if self.encoding_name.strip() == "":
            raise ValueError("encoding_name must not be empty")
        if not self.split_punctuation:
            raise ValueError("split_punctuation must not be empty")
        if any(mark == "" for mark in self.split_punctuation):
            raise ValueError("split_punctuation must not contain empty values")
        if self.max_chunks_per_page <= 0:
            raise ValueError("max_chunks_per_page must be greater than zero")
        if self.oversized_page_policy not in {"fail", "skip"}:
            raise ValueError("oversized_page_policy must be one of: fail, skip")
        if self.wiki_page_keys_artifact.strip() == "":
            raise ValueError("wiki_page_keys_artifact must not be empty")


@dataclass(frozen=True)
class SplitWikiPagesResult:
    """Artifacts produced by SplitWikiPages."""

    chunk_keys: tuple[str, ...]
    page_count: int
    chunk_count: int
    skipped_page_keys: tuple[str, ...] = ()
    issues: tuple[StepIssue, ...] = ()


@dataclass(frozen=True)
class _WikiSection:
    heading_path: tuple[str, ...]
    body: str


class SplitWikiPages:
    """Create heading-aware ParsedChunk objects from generated Wiki pages."""

    name = "split_wiki_pages"

    def __init__(self, config: SplitWikiPagesConfig | None = None) -> None:
        self.config = config or SplitWikiPagesConfig()

    @property
    def requirements(self) -> StepRequirements:
        """Return components and artifacts required by this step."""
        return StepRequirements(
            components=frozenset({store_ref("objects", self.config.object_store)}),
            artifacts=frozenset({self.config.wiki_page_keys_artifact}),
        )

    @property
    def capabilities(self) -> StepCapabilities:
        """Return artifacts produced by this step."""
        return StepCapabilities(
            artifacts=frozenset({"split_wiki_pages_result", "wiki_chunk_keys"})
        )

    def cleanup_plan(self, artifacts: Mapping[str, Any]) -> StepCleanupPlan:
        """Return Wiki chunk objects produced by this step."""
        return StepCleanupPlan(
            object_key_targets(
                artifacts,
                "wiki_chunk_keys",
                component=store_ref("objects", self.config.object_store).key,
            )
        )

    async def run(self, context: StepContextProtocol) -> None:
        """Split Wiki pages and store their retrieval chunks."""
        object_store = _require_object_store(
            context.get_component(store_ref("objects", self.config.object_store).key)
        )
        page_keys = tuple(context.get_artifact(self.config.wiki_page_keys_artifact))
        chunk_keys: list[str] = []
        skipped_page_keys: list[str] = []
        issues: list[StepIssue] = []

        for page_key in page_keys:
            page = parse_wiki_page(await object_store.get(page_key), key=page_key)
            try:
                chunks = _split_wiki_page(page, config=self.config)
            except WikiPageChunkLimitError as exc:
                if self.config.oversized_page_policy == "fail":
                    raise
                skipped_page_keys.append(page_key)
                issues.append(_page_limit_issue(page, message=str(exc)))
                continue

            planned_keys = [
                join_object_key(self.config.chunks_prefix, f"{chunk.chunk_id}.json")
                for chunk in chunks
            ]
            for chunk, chunk_key in zip(chunks, planned_keys, strict=True):
                if not await _stored_chunk_matches(object_store, chunk_key, chunk):
                    await object_store.put(chunk_key, chunk.to_json_bytes())
            chunk_keys.extend(planned_keys)

        result = SplitWikiPagesResult(
            chunk_keys=tuple(chunk_keys),
            page_count=len(page_keys),
            chunk_count=len(chunk_keys),
            skipped_page_keys=tuple(skipped_page_keys),
            issues=tuple(issues),
        )
        context.set_artifact("split_wiki_pages_result", result)
        context.set_artifact("wiki_chunk_keys", result.chunk_keys)


def _split_wiki_page(
    page: WikiPage,
    *,
    config: SplitWikiPagesConfig,
) -> list[ParsedChunk]:
    encoding = get_text_encoding(config.encoding_name)
    source = wiki_page_source(page)
    chunks: list[ParsedChunk] = []
    seen_content_hashes: set[str] = set()

    for section in _content_sections(page.content):
        prefix = _chunk_prefix(page, section.heading_path)
        prefix_tokens = len(encoding.encode(prefix))
        body_budget = config.chunk_size - prefix_tokens
        if body_budget <= 0:
            raise WikiPageChunkLimitError(
                f"Wiki chunk context exceeds chunk_size: page={page.key}, "
                f"context_tokens={prefix_tokens}, limit={config.chunk_size}"
            )
        body_overlap = min(config.overlap, max(0, body_budget - 1))
        pieces = split_text(
            section.body,
            chunk_size=body_budget,
            overlap=body_overlap,
            encoding_name=config.encoding_name,
            split_punctuation=config.split_punctuation,
        )
        for piece in pieces:
            chunk_text = f"{prefix}{piece.text}".strip()
            content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            if content_hash in seen_content_hashes:
                continue
            seen_content_hashes.add(content_hash)
            chunk_index = len(chunks)
            chunks.append(
                ParsedChunk(
                    chunk_id=make_chunk_id(
                        document_id=page.page_id,
                        page_index=0,
                        chunk_index=chunk_index,
                        text=chunk_text,
                    ),
                    document_id=page.page_id,
                    source=source,
                    page_index=0,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    token_start=piece.token_start,
                    token_end=piece.token_end,
                    heading_path=section.heading_path,
                    origin_source=page.source,
                )
            )

    if len(chunks) > config.max_chunks_per_page:
        raise WikiPageChunkLimitError(
            f"Wiki page exceeds max_chunks_per_page: page={page.key}, "
            f"chunks={len(chunks)}, limit={config.max_chunks_per_page}"
        )
    return chunks


def _content_sections(content: str) -> list[_WikiSection]:
    sections: list[_WikiSection] = []
    heading_stack: list[tuple[int, str]] = []
    current_heading_path: tuple[str, ...] = ()
    current_lines: list[str] = []
    in_code = False

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(_WikiSection(heading_path=current_heading_path, body=body))

    for line in content.splitlines():
        stripped = line.strip()
        if _is_fence(stripped):
            in_code = not in_code
            current_lines.append(line)
            continue
        match = None if in_code else _HEADING_RE.match(line)
        if match is None:
            current_lines.append(line)
            continue

        flush()
        current_lines = []
        level = len(match.group(1))
        title = match.group(2).strip()
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))
        current_heading_path = tuple(heading for _, heading in heading_stack)

    flush()
    if sections:
        return sections
    fallback = content.strip()
    return [_WikiSection(heading_path=(), body=fallback)] if fallback else []


def _chunk_prefix(page: WikiPage, heading_path: tuple[str, ...]) -> str:
    section = " > ".join(heading_path) or "Content"
    return f"Page: {page.title}\nSummary: {page.summary}\nSection: {section}\n\n"


async def _stored_chunk_matches(
    object_store: ObjectStoreProtocol,
    key: str,
    chunk: ParsedChunk,
) -> bool:
    if not await object_store.exists(key):
        return False
    try:
        existing = ParsedChunk.from_json(await object_store.get(key))
    except (TypeError, ValueError, KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return existing == chunk


def _page_limit_issue(page: WikiPage, *, message: str) -> StepIssue:
    return StepIssue(
        step="split_wiki_pages",
        subject=IssueSubject(type="wiki_page", id=page.page_id),
        code="wiki_page_chunk_limit_exceeded",
        severity="error",
        message=message,
        resolution=IssueResolution(
            action="skip_page",
            outcome="No retrieval chunks were generated for this Wiki page.",
        ),
        details={"page_key": page.key},
    )


def _is_fence(line: str) -> bool:
    return line.startswith("```") or line.startswith("~~~")


def _require_object_store(component: object) -> ObjectStoreProtocol:
    if not isinstance(component, ObjectStoreProtocol):
        raise TypeError("stores.objects must satisfy ObjectStoreProtocol")
    return component
