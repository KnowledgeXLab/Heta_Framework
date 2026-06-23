"""Text splitting helpers for knowledge base chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class TextEncoding(Protocol):
    """Minimal tokenizer interface used by the splitter."""

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token ids back into text."""
        ...


@dataclass(frozen=True)
class TextSplit:
    """One token-positioned text split."""

    text: str
    token_start: int
    token_end: int


def split_text(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    encoding_name: str,
    split_punctuation: tuple[str, ...],
) -> list[TextSplit]:
    """Split text into token windows with overlap and punctuation-aware boundaries."""
    if text.strip() == "":
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    encoding = get_text_encoding(encoding_name)
    tokens = encoding.encode(text)
    total = len(tokens)
    results: list[TextSplit] = []
    start = 0

    while start < total:
        end = min(start + chunk_size, total)
        window_text = encoding.decode(tokens[start:end])
        if end < total:
            final_text, end = _trim_to_boundary(
                window_text,
                encoding=encoding,
                start=start,
                fallback_end=end,
                split_punctuation=split_punctuation,
            )
        else:
            final_text = window_text
        final_text = final_text.strip()
        if final_text:
            results.append(TextSplit(text=final_text, token_start=start, token_end=end))

        if end >= total:
            break

        next_start = end - overlap
        if next_start <= start:
            break
        start = next_start

    return results


def _trim_to_boundary(
    window_text: str,
    *,
    encoding: TextEncoding,
    start: int,
    fallback_end: int,
    split_punctuation: tuple[str, ...],
) -> tuple[str, int]:
    boundary = max((window_text.rfind(mark) for mark in split_punctuation), default=-1)
    if boundary <= 0:
        return window_text, fallback_end
    final_text = window_text[: boundary + 1]
    final_tokens = encoding.encode(final_text)
    return final_text, start + len(final_tokens)


def get_text_encoding(encoding_name: str) -> TextEncoding:
    """Return the tokenizer used by chunk splitting."""
    if encoding_name == "unicode":
        return _UnicodeEncoding()
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - dependency is installed in normal runtime.
        raise ImportError("tiktoken is required to split documents into token chunks") from exc
    return tiktoken.get_encoding(encoding_name)


class _UnicodeEncoding:
    """Offline fallback tokenizer that treats each Unicode code point as one token."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)
