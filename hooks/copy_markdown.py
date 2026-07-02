"""Expose page Markdown sources for the docs copy button."""

from __future__ import annotations

import html
import posixpath
from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.pages import Page


def on_post_page(output: str, *, page: Page, config: MkDocsConfig) -> str:
    """Write the current page source Markdown and expose its URL in page HTML."""
    source_path = Path(page.file.abs_src_path)
    if not source_path.exists():
        return output

    target_rel = _markdown_target_for_page(page.url)
    target_path = Path(config.site_dir) / Path(*target_rel.split("/"))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")

    markdown_url = _relative_url(page.url, target_rel)
    meta = f'<meta name="heta-markdown-source" content="{html.escape(markdown_url)}">'
    if "</head>" in output:
        return output.replace("</head>", f"{meta}\n</head>", 1)
    return output


def _markdown_target_for_page(page_url: str) -> str:
    normalized = page_url.strip("/")
    if normalized == "":
        return "_markdown/index.md"
    return f"_markdown/{normalized}/index.md"


def _relative_url(page_url: str, target_rel: str) -> str:
    page_dir = page_url.strip("/")
    if page_dir == "":
        start = "."
    else:
        start = page_dir
    return posixpath.relpath(target_rel, start=start)
