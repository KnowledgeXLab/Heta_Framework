"""Internal Wiki page format shared by build and split steps."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from heta_framework.kb.parsing import ParsedSource, compute_content_sha256


@dataclass(frozen=True)
class WikiPage:
    """One generated Wiki page and its original source."""

    page_id: str
    title: str
    summary: str
    content: str
    source: ParsedSource
    key: str


def render_wiki_page(page: WikiPage) -> str:
    """Render a Wiki page in the framework-owned Markdown format."""
    return "\n".join(
        (
            "---",
            f"title: {_yaml_string(page.title)}",
            "sources:",
            f"  - key: {_yaml_string(page.source.key)}",
            f"    name: {_yaml_string(page.source.name)}",
            f"    file_type: {_yaml_string(page.source.file_type)}",
            f"    content_sha256: {_yaml_string(page.source.content_sha256)}",
            "---",
            "",
            f"# {page.title}",
            "",
            "## Summary",
            "",
            page.summary,
            "",
            "## Content",
            "",
            page.content,
            "",
            "## Related Pages",
            "",
            "",
            "## Source",
            "",
            f"- `{page.source.key}`",
            "",
        )
    )


def parse_wiki_page(data: bytes, *, key: str) -> WikiPage:
    """Parse and validate a framework-generated Wiki page."""
    text = data.decode("utf-8")
    frontmatter = _frontmatter(text)
    page = WikiPage(
        page_id=wiki_page_id_from_key(key),
        title=_frontmatter_value(frontmatter, "title"),
        summary=_section_text(text, start="Summary", end="Content"),
        content=_section_text(text, start="Content", end="Related Pages"),
        source=ParsedSource(
            key=_frontmatter_value(frontmatter, "key"),
            name=_frontmatter_value(frontmatter, "name"),
            file_type=_frontmatter_value(frontmatter, "file_type"),
            content_sha256=_frontmatter_value(frontmatter, "content_sha256"),
        ),
        key=key,
    )
    if render_wiki_page(page).encode("utf-8") != data:
        raise ValueError("Wiki page is not a canonical framework-generated artifact")
    return page


def wiki_page_source(page: WikiPage) -> ParsedSource:
    """Return source metadata for chunks derived from a Wiki page."""
    return ParsedSource(
        key=page.key,
        name=PurePosixPath(page.key).name,
        file_type="wiki",
        content_sha256=compute_content_sha256(render_wiki_page(page).encode("utf-8")),
    )


def wiki_page_id_from_key(key: str) -> str:
    """Derive the stable page id from a generated Wiki page key."""
    match = re.match(r"^(?P<number>[1-9][0-9]*)-.*\.md$", PurePosixPath(key).name)
    if match is None:
        raise ValueError(f"invalid generated Wiki page key: {key}")
    return f"wiki_page_{match.group('number')}"


def _frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        raise ValueError("Wiki page is missing frontmatter")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError("Wiki page frontmatter is not closed")
    return text[4:end]


def _frontmatter_value(frontmatter: str, key: str) -> str:
    match = re.search(
        rf"^\s*(?:-\s*)?{re.escape(key)}:\s*(.+?)\s*$",
        frontmatter,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Wiki page frontmatter is missing {key}")
    value = json.loads(match.group(1))
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"Wiki page frontmatter field must be a non-empty string: {key}")
    return value


def _section_text(text: str, *, start: str, end: str) -> str:
    match = re.search(
        rf"^## {re.escape(start)}\s*$\n(?P<body>.*?)(?=^## {re.escape(end)}\s*$)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Wiki page is missing section: {start}")
    return match.group("body").strip()


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)
