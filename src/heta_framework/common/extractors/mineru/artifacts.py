"""MinerU artifact parsing and mapping."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from heta_framework.common.extractors.types import (
    BoundingBox,
    ExtractedAsset,
    ExtractedBlock,
    ExtractedDocument,
)
from heta_framework.common.extractors.mineru.types import MinerUArtifact


_MINERU_IMAGE_RE = re.compile(r"^!\[[^\]]*\]\(images/([^)]+)\)\s*$")
_TABLE_OPEN_RE = re.compile(r"<table[\s>]", re.IGNORECASE)
_TABLE_CLOSE_RE = re.compile(r"</table>", re.IGNORECASE)


@dataclass(frozen=True)
class _ArtifactMapping:
    source: str
    blocks: tuple[ExtractedBlock, ...]
    assets: tuple[ExtractedAsset, ...]
    metadata: dict[str, object]


def parse_mineru_zip(zip_content: bytes) -> MinerUArtifact:
    """Parse a MinerU zip response into raw artifacts."""
    if not isinstance(zip_content, bytes):
        raise TypeError("zip_content must be bytes")

    markdown = ""
    content_list: list[dict[str, Any]] = []
    content_list_v2: list[tuple[dict[str, Any], ...]] = []
    images: dict[str, bytes] = {}

    with zipfile.ZipFile(BytesIO(zip_content)) as archive:
        names = archive.namelist()
        markdown_name = next((name for name in names if name.endswith("full.md")), None)
        if markdown_name is None:
            markdown_name = next((name for name in names if name.endswith(".md")), None)
        if markdown_name is None:
            raise ValueError("MinerU zip does not include markdown output")
        markdown = archive.read(markdown_name).decode("utf-8")

        content_list_name = next(
            (
                name
                for name in names
                if name.endswith("_content_list.json") and not name.endswith("_v2.json")
            ),
            None,
        )
        if content_list_name is not None:
            content_list = _read_content_list(archive.read(content_list_name))

        content_list_v2_name = next(
            (name for name in names if name.endswith("_content_list_v2.json")),
            None,
        )
        if content_list_v2_name is not None:
            content_list_v2 = _read_content_list_v2(archive.read(content_list_v2_name))

        for name in names:
            image_index = name.find("images/")
            if image_index == -1:
                continue
            relative_path = name[image_index:]
            if relative_path == "images/" or relative_path.endswith("/"):
                continue
            images[relative_path] = archive.read(name)

    return MinerUArtifact(
        markdown=markdown,
        content_list=tuple(content_list),
        content_list_v2=tuple(content_list_v2),
        images=images,
    )


def mineru_artifact_to_extracted_document(
    artifact: MinerUArtifact,
    *,
    asset_key_prefix: str = "artifacts/",
) -> ExtractedDocument:
    """Map a MinerU artifact to the shared extracted document protocol."""
    mapping = _select_artifact_mapping(
        (
            _map_content_list_v2(artifact, asset_key_prefix=asset_key_prefix),
            _map_content_list_v1(artifact, asset_key_prefix=asset_key_prefix),
            _map_markdown(artifact, asset_key_prefix=asset_key_prefix),
        ),
        markdown=artifact.markdown,
    )
    return ExtractedDocument(
        markdown=artifact.markdown,
        blocks=mapping.blocks,
        assets=mapping.assets,
        metadata={
            "provider": "mineru",
            "content_list_count": len(artifact.content_list),
            "content_list_v2_page_count": len(artifact.content_list_v2),
            "mapping_source": mapping.source,
            **mapping.metadata,
        },
    )


def _map_content_list_v2(
    artifact: MinerUArtifact,
    *,
    asset_key_prefix: str,
) -> _ArtifactMapping | None:
    if not artifact.content_list_v2:
        return None
    blocks, assets = _blocks_from_content_list_v2(
        artifact,
        asset_key_prefix=asset_key_prefix,
    )
    return _ArtifactMapping(
        source="content_list_v2",
        blocks=tuple(blocks),
        assets=tuple(assets),
        metadata={"used_content_list": "v2"},
    )


def _map_content_list_v1(
    artifact: MinerUArtifact,
    *,
    asset_key_prefix: str,
) -> _ArtifactMapping | None:
    if not artifact.content_list:
        return None
    blocks, assets = _blocks_from_content_list(
        artifact,
        asset_key_prefix=asset_key_prefix,
    )
    return _ArtifactMapping(
        source="content_list_v1",
        blocks=tuple(blocks),
        assets=tuple(assets),
        metadata={"used_content_list": "v1"},
    )


def _map_markdown(
    artifact: MinerUArtifact,
    *,
    asset_key_prefix: str,
) -> _ArtifactMapping:
    images = artifact.images or {}
    image_items = [item for item in artifact.content_list if item.get("type") == "image"]
    table_items = [item for item in artifact.content_list if item.get("type") == "table"]
    used_image_items: set[int] = set()
    used_table_items: set[int] = set()
    blocks: list[ExtractedBlock] = []
    assets: list[ExtractedAsset] = []
    text_buffer: list[str] = []

    lines = artifact.markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        image_match = _MINERU_IMAGE_RE.match(line.strip())
        if image_match is not None:
            _flush_text(text_buffer, blocks)
            image_path = "images/" + image_match.group(1)
            item, item_index = _match_image_item(image_path, image_items, used_image_items)
            if item_index is not None:
                used_image_items.add(item_index)
            asset = _image_asset(
                image_path=item.get("img_path") or image_path,
                images=images,
                asset_key_prefix=asset_key_prefix,
            )
            assets.append(asset)
            blocks.append(
                ExtractedBlock(
                    kind="image",
                    text=_caption_text(item),
                    page_index=_page_index(item),
                    bbox=_bbox(item),
                    asset=asset,
                    metadata={"provider": "mineru", "source_path": image_path},
                )
            )
            index += 1
            continue

        if _TABLE_OPEN_RE.search(line):
            _flush_text(text_buffer, blocks)
            table_lines = [line]
            while not _TABLE_CLOSE_RE.search(table_lines[-1]) and index + 1 < len(lines):
                index += 1
                table_lines.append(lines[index])
            table_html = "\n".join(table_lines)
            item, item_index = _match_table_item(table_html, table_items, used_table_items)
            if item_index is not None:
                used_table_items.add(item_index)
            blocks.append(
                ExtractedBlock(
                    kind="table",
                    text=_table_text(item, table_html),
                    page_index=_page_index(item),
                    bbox=_bbox(item),
                    metadata={"provider": "mineru"},
                )
            )
            index += 1
            continue

        text_buffer.append(line)
        index += 1

    _flush_text(text_buffer, blocks)
    return _ArtifactMapping(
        source="markdown",
        blocks=tuple(blocks),
        assets=tuple(assets),
        metadata={
            "used_content_list": False,
            "unmatched_table_count": len(table_items) - len(used_table_items),
            "unmatched_image_count": len(image_items) - len(used_image_items),
        },
    )


def _select_artifact_mapping(
    mappings: tuple[_ArtifactMapping | None, ...],
    *,
    markdown: str,
) -> _ArtifactMapping:
    for mapping in mappings:
        if mapping is None:
            continue
        if mapping.source == "markdown":
            return mapping
        if _structured_mapping_is_complete_enough(mapping, markdown):
            return mapping
    raise ValueError("MinerU artifact does not contain mappable content")


def _structured_mapping_is_complete_enough(mapping: _ArtifactMapping, markdown: str) -> bool:
    if not mapping.blocks:
        return False
    has_body_text = any(block.kind == "text" and block.text.strip() for block in mapping.blocks)
    if not has_body_text:
        return False
    has_page_indexes = any(block.page_index is not None for block in mapping.blocks)
    if not has_page_indexes:
        return False
    mapped_chars = sum(len(block.text.strip()) for block in mapping.blocks if block.text.strip())
    markdown_chars = len(re.sub(r"\s+", "", markdown))
    return markdown_chars == 0 or mapped_chars >= markdown_chars * 0.75


def _blocks_from_content_list(
    artifact: MinerUArtifact,
    *,
    asset_key_prefix: str,
) -> tuple[list[ExtractedBlock], list[ExtractedAsset]]:
    images = artifact.images or {}
    blocks: list[ExtractedBlock] = []
    assets: list[ExtractedAsset] = []

    for item in artifact.content_list:
        kind = str(item.get("type") or "").strip()
        if kind in {"header", "footer", "page_number"}:
            continue
        if kind == "image":
            image_path = item.get("img_path")
            if not isinstance(image_path, str) or not image_path.strip():
                continue
            asset = _image_asset(
                image_path=image_path,
                images=images,
                asset_key_prefix=asset_key_prefix,
            )
            assets.append(asset)
            blocks.append(
                ExtractedBlock(
                    kind="image",
                    text=_caption_text(item),
                    page_index=_page_index(item),
                    bbox=_bbox(item),
                    asset=asset,
                    metadata={"provider": "mineru", "source_path": image_path},
                )
            )
            continue
        if kind == "table":
            text = _table_text(item, "")
        elif kind == "list":
            text = _list_text(item)
        else:
            text = _content_text(item)
        if not text.strip():
            continue
        blocks.append(
            ExtractedBlock(
                kind="text" if kind not in {"table"} else "table",
                text=text,
                page_index=_page_index(item),
                bbox=_bbox(item),
                metadata={"provider": "mineru", "source_type": kind},
            )
        )

    return blocks, assets


def _blocks_from_content_list_v2(
    artifact: MinerUArtifact,
    *,
    asset_key_prefix: str,
) -> tuple[list[ExtractedBlock], list[ExtractedAsset]]:
    images = artifact.images or {}
    blocks: list[ExtractedBlock] = []
    assets: list[ExtractedAsset] = []

    for page_index, page_items in enumerate(artifact.content_list_v2):
        for item in page_items:
            kind = str(item.get("type") or "").strip()
            if kind in {"page_header", "page_footer", "page_number"}:
                continue
            if kind == "table":
                text = _v2_table_text(item)
                if text.strip():
                    blocks.append(
                        ExtractedBlock(
                            kind="table",
                            text=text,
                            page_index=page_index,
                            bbox=_bbox(item),
                            metadata={"provider": "mineru", "source_type": kind},
                        )
                    )
                continue
            if kind == "image":
                asset = _v2_image_asset(
                    item,
                    images=images,
                    asset_key_prefix=asset_key_prefix,
                )
                if asset is None:
                    continue
                assets.append(asset)
                blocks.append(
                    ExtractedBlock(
                        kind="image",
                        text=_v2_item_text(item),
                        page_index=page_index,
                        bbox=_bbox(item),
                        asset=asset,
                        metadata={"provider": "mineru", "source_type": kind},
                    )
                )
                continue
            text = _v2_item_text(item)
            if not text.strip():
                continue
            blocks.append(
                ExtractedBlock(
                    kind="text",
                    text=text,
                    page_index=page_index,
                    bbox=_bbox(item),
                    metadata={"provider": "mineru", "source_type": kind},
                )
            )

    return blocks, assets


def _read_content_list(data: bytes) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _read_content_list_v2(data: bytes) -> list[tuple[dict[str, Any], ...]]:
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    pages: list[tuple[dict[str, Any], ...]] = []
    for page in parsed:
        if not isinstance(page, list):
            continue
        pages.append(tuple(item for item in page if isinstance(item, dict)))
    return pages


def _flush_text(buffer: list[str], blocks: list[ExtractedBlock]) -> None:
    text = "\n".join(buffer).strip()
    if text:
        blocks.append(ExtractedBlock(kind="text", text=text, metadata={"provider": "mineru"}))
    buffer.clear()


def _match_image_item(
    image_path: str,
    image_items: list[dict[str, Any]],
    used_indexes: set[int],
) -> tuple[dict[str, Any], int | None]:
    for index, item in enumerate(image_items):
        if index in used_indexes:
            continue
        if item.get("img_path") == image_path:
            return item, index
    for index, item in enumerate(image_items):
        if index not in used_indexes:
            return item, index
    return {}, None


def _match_table_item(
    table_html: str,
    table_items: list[dict[str, Any]],
    used_indexes: set[int],
) -> tuple[dict[str, Any], int | None]:
    normalized_html = _normalize_markup(table_html)
    for index, item in enumerate(table_items):
        if index in used_indexes:
            continue
        table_body = item.get("table_body")
        if isinstance(table_body, str) and _normalize_markup(table_body) == normalized_html:
            return item, index
    return {}, None


def _normalize_markup(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _image_asset(
    *,
    image_path: str,
    images: dict[str, bytes],
    asset_key_prefix: str,
) -> ExtractedAsset:
    data = images.get(image_path)
    media_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
    digest = hashlib.sha256(data).hexdigest() if data is not None else None
    return ExtractedAsset(
        name=Path(image_path).name,
        media_type=media_type,
        data=data,
        key=asset_key_prefix.rstrip("/") + "/" + image_path,
        content_sha256=digest,
        size_bytes=len(data) if data is not None else None,
    )


def _caption_text(item: dict[str, Any]) -> str:
    captions = item.get("image_caption") or item.get("img_caption") or item.get("caption")
    if isinstance(captions, list):
        return "\n".join(str(caption) for caption in captions if str(caption).strip())
    if isinstance(captions, str):
        return captions.strip()
    return ""


def _table_text(item: dict[str, Any], fallback_html: str) -> str:
    caption = item.get("table_caption")
    caption_text = ""
    if isinstance(caption, list):
        caption_text = " ".join(str(value) for value in caption if str(value).strip())
    elif isinstance(caption, str):
        caption_text = caption.strip()
    body = item.get("table_body")
    table_body = body if isinstance(body, str) and body.strip() else fallback_html
    return f"{caption_text}\n{table_body}".strip() if caption_text else table_body.strip()


def _content_text(item: dict[str, Any]) -> str:
    text = item.get("text")
    if isinstance(text, str):
        return text.strip()
    return ""


def _list_text(item: dict[str, Any]) -> str:
    list_items = item.get("list_items")
    if not isinstance(list_items, list):
        return ""
    lines: list[str] = []
    for value in list_items:
        if isinstance(value, str) and value.strip():
            lines.append(value.strip())
        elif isinstance(value, dict):
            text = value.get("text") or value.get("content")
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())
    return "\n".join(lines)


def _v2_item_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, dict):
        return ""
    kind = str(item.get("type") or "")
    if kind == "title":
        return _v2_fragments_text(content.get("title_content"))
    if kind == "paragraph":
        return _v2_fragments_text(content.get("paragraph_content"))
    if kind == "list":
        list_items = content.get("list_items")
        if not isinstance(list_items, list):
            return ""
        lines: list[str] = []
        for list_item in list_items:
            if not isinstance(list_item, dict):
                continue
            text = _v2_fragments_text(list_item.get("item_content"))
            if text:
                lines.append(text)
        return "\n".join(lines)
    return _v2_fragments_text(content)


def _v2_fragments_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_v2_fragments_text(item) for item in value]
        return "".join(part for part in parts if part)
    if isinstance(value, dict):
        text = value.get("content") or value.get("text")
        if isinstance(text, str):
            return text.strip()
        parts = [_v2_fragments_text(item) for item in value.values()]
        return "".join(part for part in parts if part)
    return ""


def _v2_table_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, dict):
        return ""
    caption = content.get("table_caption")
    caption_text = _v2_fragments_text(caption)
    table_html = content.get("html")
    body = table_html.strip() if isinstance(table_html, str) else ""
    return f"{caption_text}\n{body}".strip() if caption_text else body


def _v2_image_asset(
    item: dict[str, Any],
    *,
    images: dict[str, bytes],
    asset_key_prefix: str,
) -> ExtractedAsset | None:
    content = item.get("content")
    if not isinstance(content, dict):
        return None
    image_source = content.get("image_source")
    if not isinstance(image_source, dict):
        return None
    image_path = image_source.get("path")
    if not isinstance(image_path, str) or not image_path.strip() or image_path.endswith("/"):
        return None
    return _image_asset(
        image_path=image_path,
        images=images,
        asset_key_prefix=asset_key_prefix,
    )


def _page_index(item: dict[str, Any]) -> int | None:
    page_index = item.get("page_idx")
    return page_index if isinstance(page_index, int) and page_index >= 0 else None


def _bbox(item: dict[str, Any]) -> BoundingBox | None:
    bbox = item.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    if not all(isinstance(value, (int, float)) for value in bbox):
        return None
    return BoundingBox(
        left=float(bbox[0]),
        top=float(bbox[1]),
        right=float(bbox[2]),
        bottom=float(bbox[3]),
    )
