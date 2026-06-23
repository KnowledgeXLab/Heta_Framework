"""HTML parser."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin

from heta_framework.common.extractors import (
    DocumentExtractorProtocol,
    DocumentInput,
    ExtractedBlock,
    ExtractedDocument,
)
from heta_framework.common.models import ImagePart, ModelRequest, TextPart
from heta_framework.common.models.protocols import LanguageModelProtocol
from heta_framework.kb.parsing.prompts import DEFAULT_HTML_IMAGE_DESCRIPTION_PROMPT
from heta_framework.kb.parsing.text import _decode_text
from heta_framework.kb.parsing.types import (
    ParsedDocument,
    ParsedPage,
    ParsedSource,
    make_document_id,
)


@dataclass(frozen=True)
class HtmlParserConfig:
    """Configuration for HTML parsers."""

    encodings: tuple[str, ...] = ("utf-8", "utf-8-sig", "gb18030", "latin-1")
    source_url: str = ""
    describe_images: bool = False
    max_described_images: int = 20
    image_prompt: str = DEFAULT_HTML_IMAGE_DESCRIPTION_PROMPT

    def __post_init__(self) -> None:
        if not self.encodings:
            raise ValueError("encodings must not be empty")
        if any(encoding.strip() == "" for encoding in self.encodings):
            raise ValueError("encodings must not contain empty values")
        if self.max_described_images < 0:
            raise ValueError("max_described_images must not be negative")
        if self.image_prompt.strip() == "":
            raise ValueError("image_prompt must not be empty")


class HtmlParser:
    """Parse HTML into one text page."""

    supported_file_types = {"html", "htm"}

    def __init__(
        self,
        config: HtmlParserConfig | None = None,
        *,
        vision_model: LanguageModelProtocol | None = None,
        extractor: DocumentExtractorProtocol | None = None,
    ) -> None:
        self.config = config or HtmlParserConfig()
        self._vision_model = vision_model
        self._extractor = extractor or BasicHtmlExtractor(
            config=self.config,
            vision_model=self._vision_model,
        )
        if self.config.describe_images and self._vision_model is None:
            raise ValueError("vision_model is required when describe_images is enabled")

    async def parse(self, source: ParsedSource, data: bytes) -> ParsedDocument:
        """Parse raw HTML bytes into a ParsedDocument."""
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        file_type = source.file_type.lower().lstrip(".")
        if file_type not in self.supported_file_types:
            raise ValueError(f"unsupported file type for HtmlParser: {source.file_type}")

        extracted = await self._extractor.extract(
            DocumentInput(
                data=data,
                filename=source.name,
                media_type="text/html",
            )
        )
        return ParsedDocument(
            document_id=make_document_id(source.content_sha256),
            source=source,
            pages=[ParsedPage(page_index=0, text=extracted.to_text())],
        )


class BasicHtmlExtractor:
    """Small built-in extractor for common article, documentation, and tutorial pages."""

    def __init__(
        self,
        config: HtmlParserConfig | None = None,
        *,
        vision_model: LanguageModelProtocol | None = None,
    ) -> None:
        self.config = config or HtmlParserConfig()
        self._vision_model = vision_model
        if self.config.describe_images and self._vision_model is None:
            raise ValueError("vision_model is required when describe_images is enabled")

    async def extract(
        self,
        document: DocumentInput,
        options: object | None = None,
    ) -> ExtractedDocument:
        """Extract text, tables, and selected image context from HTML bytes."""
        del options
        html = _decode_text(document.data, self.config.encodings)
        text, title, description = await _extract_html_text(
            html,
            source_url=self.config.source_url,
            config=self.config,
            vision_model=self._vision_model,
        )
        return ExtractedDocument(
            text=text,
            blocks=(ExtractedBlock(kind="text", text=text),),
            metadata={
                "title": title,
                "description": description,
                "provider": "basic_html",
            },
        )


@dataclass(frozen=True)
class _ImageCandidate:
    url: str
    caption: str = ""


@dataclass
class _ImageDescriptionSlot:
    candidate: _ImageCandidate
    text: str = ""


async def _extract_html_text(
    html: str,
    *,
    source_url: str,
    config: HtmlParserConfig,
    vision_model: LanguageModelProtocol | None,
) -> tuple[str, str, str]:
    try:
        from bs4 import BeautifulSoup, Comment
    except ImportError as exc:  # pragma: no cover - dependency is installed in normal runtime.
        raise ImportError("beautifulsoup4 is not installed") from exc

    soup = BeautifulSoup(html, "html.parser")
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    description = _extract_description(soup)
    _remove_noise_tags(soup)

    root = _select_content_root(soup)
    parts: list[object] = []
    if title:
        parts.append(f"Title: {title}")
    if description:
        parts.append(f"Description: {description}")

    seen_images: set[str] = set()
    described_image_count = [0]
    _walk_html(
        root,
        source_url=source_url,
        parts=parts,
        seen_images=seen_images,
        describe_images=config.describe_images,
        described_image_count=described_image_count,
        max_described_images=config.max_described_images,
    )
    if config.describe_images and vision_model is not None:
        await _fill_image_descriptions(parts, vision_model=vision_model, prompt=config.image_prompt)
    return _normalize_text("\n".join(_render_parts(parts))), title, description


def _extract_description(soup: object) -> str:
    desc_tag = soup.find("meta", attrs={"name": "description"})
    og_desc_tag = soup.find("meta", attrs={"property": "og:description"})
    tag = desc_tag or og_desc_tag
    if tag is None:
        return ""
    return str(tag.get("content", "")).strip()


def _remove_noise_tags(soup: object) -> None:
    for tag in soup.find_all(
        [
            "script",
            "style",
            "footer",
            "nav",
            "aside",
            "iframe",
            "button",
            "noscript",
            "form",
            "select",
            "option",
            "header",
        ]
    ):
        tag.decompose()


def _select_content_root(soup: object) -> object:
    selectors = [
        "main",
        "article",
        "[role='main']",
        "#main",
        "#content",
        "#main-content",
        "#article",
        ".main",
        ".content",
        ".main-content",
        ".article",
        ".post-content",
        ".entry-content",
        ".markdown-body",
        ".document",
    ]
    candidates = []
    for selector in selectors:
        for node in soup.select(selector):
            score = _content_score(node)
            if score > 0:
                candidates.append((score, node))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
    return soup.find("body") or soup


def _content_score(node: object) -> int:
    text = node.get_text(" ", strip=True)
    if len(text) < 200:
        return 0
    marker_text = _node_marker_text(node)
    if _looks_like_noise_container(marker_text):
        return 0
    return len(text)


def _walk_html(
    node: object,
    *,
    source_url: str,
    parts: list[object],
    seen_images: set[str],
    describe_images: bool,
    described_image_count: list[int],
    max_described_images: int,
) -> None:
    for child in getattr(node, "children", []):
        name = getattr(child, "name", None)
        if name is None:
            text = str(child).strip()
            if text:
                parts.append(text)
            continue
        if _looks_like_noise_container(_node_marker_text(child)):
            continue

        if name == "img":
            candidate = _image_candidate(child, source_url=source_url)
            if candidate is None:
                continue
            if not _add_image_candidate(candidate, seen_images):
                continue
            image_text = _image_text(candidate)
            if image_text:
                parts.append(image_text)
            _append_description_slot(
                candidate,
                parts=parts,
                describe_images=describe_images,
                described_image_count=described_image_count,
                max_described_images=max_described_images,
            )
            continue

        if name == "source" and child.get("srcset"):
            candidates = [
                candidate
                for candidate in _source_candidates(child, source_url=source_url)
                if _add_image_candidate(candidate, seen_images)
            ]
            source_text = _source_text(candidates)
            if source_text:
                parts.append(source_text)
            for candidate in candidates:
                _append_description_slot(
                    candidate,
                    parts=parts,
                    describe_images=describe_images,
                    described_image_count=described_image_count,
                    max_described_images=max_described_images,
                )
            continue

        if name == "video" and child.get("poster"):
            poster = urljoin(source_url, str(child.get("poster")))
            candidate = _ImageCandidate(url=poster, caption="video poster")
            if _should_keep_image(candidate) and _add_image_candidate(candidate, seen_images):
                parts.append(f"Video poster: {poster}")
                _append_description_slot(
                    candidate,
                    parts=parts,
                    describe_images=describe_images,
                    described_image_count=described_image_count,
                    max_described_images=max_described_images,
                )

        if name == "table":
            table_text = _table_text(child)
            if table_text:
                parts.append(table_text)
            continue

        _append_background_images(child, source_url=source_url, parts=parts)
        _walk_html(
            child,
            source_url=source_url,
            parts=parts,
            seen_images=seen_images,
            describe_images=describe_images,
            described_image_count=described_image_count,
            max_described_images=max_described_images,
        )


def _image_candidate(node: object, *, source_url: str) -> _ImageCandidate | None:
    if _is_in_noise_container(node):
        return None
    src = _get_image_url(node)
    if not src:
        return None
    url = urljoin(source_url, src)
    caption = str(node.get("alt") or node.get("title") or "").strip()
    candidate = _ImageCandidate(url=url, caption=caption)
    if not _should_keep_image(candidate, node=node):
        return None
    return candidate


def _image_text(candidate: _ImageCandidate) -> str:
    if candidate.caption:
        return f"Image: {candidate.caption} ({candidate.url})"
    return f"Image: {candidate.url}"


def _source_candidates(node: object, *, source_url: str) -> list[_ImageCandidate]:
    candidates = []
    for part in str(node.get("srcset", "")).split(","):
        url_part = part.strip().split()[0] if part.strip() else ""
        if url_part:
            candidate = _ImageCandidate(url=urljoin(source_url, url_part), caption="source image")
            if _should_keep_image(candidate):
                candidates.append(candidate)
    return candidates


def _source_text(candidates: list[_ImageCandidate]) -> str:
    if not candidates:
        return ""
    return "Image source: " + " ".join(candidate.url for candidate in candidates)


def _table_text(node: object) -> str:
    rows = []
    caption = ""
    caption_tag = node.find("caption")
    if caption_tag is not None:
        caption = caption_tag.get_text(" ", strip=True)
    for row in node.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(" | ".join(cells))
    if not rows:
        return ""
    body = "\n".join(rows)
    if caption:
        return f"Table: {caption}\n{body}"
    return f"Table:\n{body}"


def _append_background_images(node: object, *, source_url: str, parts: list[object]) -> None:
    style = str(node.get("style") or "")
    if "url(" not in style:
        return
    for raw_url in re.findall(r'url\([\'"]?(.*?)[\'"]?\)', style):
        url = urljoin(source_url, raw_url.strip())
        if url and not _is_svg(url):
            parts.append(f"Background image: {url}")


def _get_image_url(node: object) -> str:
    candidates = [
        node.get("data-src"),
        node.get("data-original"),
        node.get("data-lazy-src"),
        node.get("src"),
    ]
    srcset = node.get("srcset")
    if srcset:
        candidates.append(str(srcset).split(",")[-1].strip().split()[0])
    for candidate in candidates:
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return ""


def _is_svg(url: str) -> bool:
    return ".svg" in url.lower()


def _should_keep_image(candidate: _ImageCandidate, node: object | None = None) -> bool:
    url = candidate.url.strip()
    if not url or _is_svg(url):
        return False

    lowered = url.lower()
    noisy_markers = (
        "favicon",
        "icon",
        "logo",
        "sprite",
        "spacer",
        "tracking",
        "tracker",
        "pixel",
        "avatar",
        "badge",
        "button",
        "fullaccess",
        "promo",
        "promotion",
        "subscribe",
        "upgrade",
        "ads/",
        "/ad/",
        "doubleclick",
    )
    caption_lowered = candidate.caption.lower()
    if any(marker in lowered or marker in caption_lowered for marker in noisy_markers):
        return False

    if node is not None:
        width = _int_attr(node, "width")
        height = _int_attr(node, "height")
        if width is not None and height is not None and width <= 64 and height <= 64:
            return False
    return True


def _is_in_noise_container(node: object) -> bool:
    current = getattr(node, "parent", None)
    while current is not None:
        if _looks_like_noise_container(_node_marker_text(current)):
            return True
        current = getattr(current, "parent", None)
    return False


def _node_marker_text(node: object) -> str:
    if getattr(node, "attrs", None) is None:
        return ""
    marker_values = []
    node_id = node.get("id") if hasattr(node, "get") else None
    node_class = node.get("class") if hasattr(node, "get") else None
    node_role = node.get("role") if hasattr(node, "get") else None
    if node_id:
        marker_values.append(str(node_id).lower())
    if node_class:
        marker_values.append(" ".join(str(value).lower() for value in node_class))
    if node_role:
        marker_values.append(str(node_role).lower())
    return " ".join(marker_values)


def _looks_like_noise_container(marker_text: str) -> bool:
    if not marker_text:
        return False
    noisy_markers = (
        "nav",
        "menu",
        "sidebar",
        "header",
        "footer",
        "topnav",
        "sidenav",
        "advert",
        "advertisement",
        "cookie",
        "banner",
        "social",
        "share",
        "breadcrumb",
        "pagination",
        "newsletter",
        "subscribe",
        "related",
        "recommended",
        "comment",
        "modal",
        "popup",
    )
    tokens = {token for token in re.split(r"[^a-z0-9]+", marker_text.lower()) if token}
    return any(marker in tokens for marker in noisy_markers)


def _int_attr(node: object, name: str) -> int | None:
    value = node.get(name)
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def _add_image_candidate(candidate: _ImageCandidate, seen_images: set[str]) -> bool:
    if candidate.url in seen_images:
        return False
    seen_images.add(candidate.url)
    return True


def _append_description_slot(
    candidate: _ImageCandidate,
    *,
    parts: list[object],
    describe_images: bool,
    described_image_count: list[int],
    max_described_images: int,
) -> None:
    if not describe_images:
        return
    if described_image_count[0] >= max_described_images:
        return
    parts.append(_ImageDescriptionSlot(candidate=candidate))
    described_image_count[0] += 1


async def _fill_image_descriptions(
    parts: list[object],
    *,
    vision_model: LanguageModelProtocol,
    prompt: str,
) -> None:
    for part in parts:
        if not isinstance(part, _ImageDescriptionSlot):
            continue
        result = await vision_model.invoke(
            ModelRequest(
                content=[
                    TextPart(_build_html_image_prompt(prompt, candidate=part.candidate)),
                    ImagePart.from_uri(part.candidate.url),
                ]
            )
        )
        text = result.text.strip()
        if text:
            part.text = f"Image description: {text}"


def _build_html_image_prompt(prompt: str, *, candidate: _ImageCandidate) -> str:
    parts = [
        prompt.strip(),
        "",
        "Image metadata:",
        f"- image_url: {candidate.url}",
    ]
    if candidate.caption:
        parts.append(f"- existing_caption_or_alt: {candidate.caption}")
    return "\n".join(parts)


def _render_parts(parts: list[object]) -> list[str]:
    rendered = []
    for part in parts:
        if isinstance(part, _ImageDescriptionSlot):
            if part.text.strip():
                rendered.append(part.text)
            continue
        text = str(part).strip()
        if text:
            rendered.append(text)
    return rendered


def _normalize_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line and line not in {"-->", "<!--"})
