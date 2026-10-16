# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asset ingestion: the KG and the paper text are *inputs*, nothing is fetched.

There is no SciVerse call and no PDF parsing anywhere in this project. A paper body is
expected to arrive as plain text; this module only locates it and pairs it with its KG.

Three ingestion paths produce the *same* seed table:

* ``layout="dirs"`` (default, what ``~/data/sample`` uses) --
  ``<txt_root>/<paper_dir>/content.txt``, where the directory name carries a
  ``paper_<index>_<slug>`` prefix that is stripped before title matching.
* ``layout="flat"`` -- ``<txt_root>/<title>.txt``, for a directory of plain text files
  named after the paper.
* a **search query** (:func:`build_search_seed`) -- the papers behind the key nodes a LeanRAG
  query returned, in relevance order. The "assets" are already on disk by then (the search side
  flattened the KG and parsed the PDFs), so this path only maps search payloads onto the very
  same columns.

Because the three paths agree on the schema, nothing downstream knows which one was used.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from heta_framework.generation.graph._native.io_utils import atomic_write_json, normalize_title, sha256_file, title_tokens

TextLayout = Literal["dirs", "flat"]

#: The seed table's columns, in order. This is the contract every stage reads: the prompts
#: re-read ``kg_path`` and ``paper_text_path``, the gates key on ``paper_id``/``title``, and
#: ``omegadesigner.dd.workflow.SEED_KEEP`` projects them forward. Every ingestion path builds its
#: rows through :func:`_seed_frame`, so a column added here cannot appear in one path only.
SEED_COLUMNS: tuple[str, ...] = (
    "paper_id",
    "row_index",
    "title",
    "kg_path",
    "kg_sha256",
    "paper_dir",
    "paper_text_path",
    "paper_text_chars",
)

PAPER_DIR_PREFIX = re.compile(r"^paper_\d+_")

# Fuzzy fallback thresholds. KG files are named after the paper title while text assets
# may be slugs, so exact normalized equality is not enough on its own (measured on
# ~/data/sample: exact gets 232 pairs, adding this fallback gets 535).
MIN_TOKEN_INTERSECTION = 5
MIN_TOKEN_JACCARD = 0.6


@dataclass(frozen=True)
class AssetPair:
    """One KG document and the paper text that goes with it."""

    kg_path: Path
    body_path: Path
    title: str
    kg_sha256: str
    strategy: str = "exact"
    # Set only by the search path: the graph key the paper is addressable by (a node's
    # ``source_id``), and which extractor produced the body. Both are provenance for
    # ``asset_report.json``; the seed table itself carries paths, not keys.
    paper_key: str = ""
    text_provider: str | None = None


@dataclass
class AssetReport:
    matched: int = 0
    unmatched_kg: list[str] = field(default_factory=list)
    unmatched_body: list[str] = field(default_factory=list)
    pairs: list[AssetPair] = field(default_factory=list)
    exact: int = 0
    fuzzy: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "exact": self.exact,
            "fuzzy": self.fuzzy,
            "unmatched_kg_count": len(self.unmatched_kg),
            "unmatched_body_count": len(self.unmatched_body),
            "unmatched_kg_sample": sorted(self.unmatched_kg)[:50],
            "unmatched_body_sample": sorted(self.unmatched_body)[:50],
            "sample_pairs": [
                {"kg": pair.kg_path.name, "body": str(pair.body_path), "title": pair.title} for pair in self.pairs[:10]
            ],
        }


# normalize_title / title_tokens moved to io_utils so the search package can pair KG files with
# PDFs by title without importing pandas; they are re-exported here for existing callers.
def _best_fuzzy_body(kg_tokens: set[str], bodies: dict[str, Path]) -> tuple[Path, str] | None:
    """Closest body by token overlap, or ``None`` when nothing clears the thresholds."""
    best: tuple[float, int, Path, str] | None = None
    for key, path in bodies.items():
        body_tokens = set(key.split())
        if not kg_tokens or not body_tokens:
            continue
        intersection = len(kg_tokens & body_tokens)
        union = len(kg_tokens | body_tokens)
        jaccard = intersection / union if union else 0.0
        if intersection < MIN_TOKEN_INTERSECTION or jaccard < MIN_TOKEN_JACCARD:
            continue
        if best is None or (jaccard, intersection) > (best[0], best[1]):
            best = (jaccard, intersection, path, key)
    return (best[2], best[3]) if best else None


def _collect_bodies(txt_root: Path, layout: TextLayout) -> dict[str, Path]:
    if layout == "flat":
        return {
            normalize_title(path.stem): path
            for path in sorted(txt_root.iterdir())
            if path.is_file() and path.suffix == ".txt"
        }
    bodies: dict[str, Path] = {}
    for directory in sorted(path for path in txt_root.iterdir() if path.is_dir()):
        content = directory / "content.txt"
        if content.is_file():
            slug = PAPER_DIR_PREFIX.sub("", directory.name)
            bodies.setdefault(normalize_title(slug), content)
    return bodies


def discover_asset_pairs(*, kg_root: str | Path, txt_root: str | Path, layout: TextLayout = "dirs") -> AssetReport:
    """Pair every KG JSON with the text file carrying the same normalized title."""
    kg_root = Path(kg_root)
    bodies = _collect_bodies(Path(txt_root), layout)
    report = AssetReport()

    remaining = dict(bodies)
    for kg_path in sorted(kg_root.glob("*.json")):
        title = kg_path.stem
        key = normalize_title(title)
        body = remaining.get(key)
        strategy = "exact"
        if body is None:
            fuzzy = _best_fuzzy_body(title_tokens(title), remaining)
            if fuzzy is None:
                report.unmatched_kg.append(kg_path.name)
                continue
            body, key = fuzzy
            strategy = "fuzzy"
        remaining.pop(key, None)
        report.matched += 1
        report.exact += strategy == "exact"
        report.fuzzy += strategy == "fuzzy"
        report.pairs.append(
            AssetPair(
                kg_path=kg_path,
                body_path=body,
                title=title,
                kg_sha256=sha256_file(kg_path),
                strategy=strategy,
            )
        )

    report.unmatched_body = sorted(str(path) for path in remaining.values())
    return report


def _seed_frame(pairs: list[AssetPair]) -> pd.DataFrame:
    """One seed row per pair, in :data:`SEED_COLUMNS` order.

    The seed carries the *path* to the paper text plus its length, never the text itself:
    the ``paper_text`` column reads the file, and each prompt truncates to its own budget
    (review 60k, candidates 80k, spec evidence 40k). Carrying a truncated copy here would
    duplicate the body and invite the reader to think that copy is what the model sees.
    """
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, 1):
        text = pair.body_path.read_text(encoding="utf-8")
        rows.append(
            {
                "paper_id": pair.kg_sha256[:24],
                "row_index": index,
                "title": pair.title,
                "kg_path": str(pair.kg_path.resolve()),
                "kg_sha256": pair.kg_sha256,
                "paper_dir": str(pair.body_path.parent.resolve()),
                "paper_text_path": str(pair.body_path.resolve()),
                "paper_text_chars": len(text),
            }
        )
    return pd.DataFrame(rows, columns=list(SEED_COLUMNS))


def build_asset_seed(pairs: list[AssetPair]) -> pd.DataFrame:
    """The seed table for a list of asset pairs."""
    return _seed_frame(pairs)


def write_seed_table(frame: pd.DataFrame, *, output_dir: str | Path, report: Mapping[str, Any]) -> Path:
    """Write ``papers.parquet`` + ``asset_report.json`` and return the parquet path.

    Every ingestion path goes through this writer. The seed is a run's input record, so "what was
    seeded, and why those papers" must be answerable from the same two files whichever way the
    papers were found -- a second writer would eventually disagree about that.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / "papers.parquet"
    frame.to_parquet(seed_path, index=False)
    atomic_write_json(output_dir / "asset_report.json", dict(report))
    return seed_path


def prepare_asset_seed(
    *,
    kg_root: str | Path,
    txt_root: str | Path,
    output_dir: str | Path,
    layout: TextLayout = "dirs",
    limit: int | None = None,
) -> tuple[Path, AssetReport]:
    report = discover_asset_pairs(kg_root=kg_root, txt_root=txt_root, layout=layout)
    report.pairs = report.pairs[:limit] if limit else report.pairs
    report.matched = len(report.pairs)
    if not report.pairs:
        raise RuntimeError(f"no KG/text asset pairs matched under {kg_root} and {txt_root} (layout={layout})")

    frame = build_asset_seed(report.pairs)
    payload = report.as_dict()
    payload["layout"] = layout
    payload["seed_rows"] = len(frame)
    seed_path = write_seed_table(frame, output_dir=output_dir, report=payload)
    return seed_path, report


@dataclass
class SearchSeedReport:
    """What a search-backed seed run did; serialized to the run's ``asset_report.json``.

    ``retrieved`` counts the distinct papers the query returned, ``matched`` the rows actually
    written, and ``rows`` the provenance the seed table deliberately does not carry (which key
    nodes pointed at each paper, and at which rank/score).
    """

    query: str = ""
    search_work_dir: str = ""
    topk: int = 10
    level_mode: int = 2
    retrieved: int = 0
    matched: int = 0
    skipped: list[dict[str, Any]] = field(default_factory=list)
    screen_rejected: list[dict[str, Any]] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            # `layout` stays the key that says how the seed was paired (test_cli asserts it);
            # "search" is the third value it can take, next to "dirs" and "flat".
            "source": "search",
            "layout": "search",
            "query": self.query,
            "search_work_dir": self.search_work_dir,
            "topk": self.topk,
            "level_mode": self.level_mode,
            "retrieved": self.retrieved,
            "matched": self.matched,
            "seed_rows": self.matched,
            "skipped_count": len(self.skipped),
            "skipped": self.skipped,
            "screen_rejected_count": len(self.screen_rejected),
            "screen_rejected": self.screen_rejected,
            "rows": self.rows,
        }


def search_paper_pairs(
    papers: Iterable[Mapping[str, Any]],
) -> tuple[list[AssetPair], list[dict[str, Any]]]:
    """Map ``SearchIndex.papers()`` payloads onto the pairs the seed is built from.

    Returns ``(pairs, skipped)``. A paper whose KG or parsed text is missing is *skipped with a
    reason* rather than seeded with an empty path: every prompt re-reads both files, so such a row
    would fail three stages later with an error that no longer names the paper. Both paths are
    already on disk at this point -- the search side flattened the KG and parsed the PDFs -- so the
    checks here are about the query's papers being complete, not about fetching anything.
    """
    pairs: list[AssetPair] = []
    skipped: list[dict[str, Any]] = []
    for paper in papers:
        problems: list[str] = []
        for name in ("kg_path", "text_path"):
            raw = str(paper.get(name) or "").strip()
            if not raw:
                # A paper the PDF was never parsed for: `text_path` is null until `search parse`
                # has run, and the registry does not pretend otherwise.
                problems.append(f"{name} is unset; run `omegadesigner search parse` first")
            elif not Path(raw).expanduser().is_file():
                problems.append(f"{name}={raw} is not a file")
        if problems:
            skipped.append(
                {
                    "paper_key": str(paper.get("paper_key") or ""),
                    "title": str(paper.get("title") or ""),
                    "reason": "; ".join(problems),
                    "pdf_path": str(paper.get("pdf_path") or ""),
                }
            )
            continue

        kg_path = Path(str(paper["kg_path"])).expanduser()
        text_path = Path(str(paper["text_path"])).expanduser()
        pairs.append(
            AssetPair(
                kg_path=kg_path,
                body_path=text_path,
                title=str(paper.get("title") or "").strip() or kg_path.stem,
                kg_sha256=sha256_file(kg_path),
                strategy="search",
                paper_key=str(paper.get("paper_key") or ""),
                text_provider=str(paper.get("text_provider") or "") or None,
            )
        )
    return pairs, skipped


def build_search_seed(
    papers: Iterable[Mapping[str, Any]],
    *,
    report: SearchSeedReport,
    notes: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[pd.DataFrame, SearchSeedReport]:
    """The seed table (and its report) for the papers a search query returned, in query order.

    The frame is column-for-column the one :func:`prepare_asset_seed` writes, so nothing downstream
    can tell whether a paper was paired on disk or retrieved by a query. Nothing here calls a
    model: the caller already ran the query and passed its papers.
    """
    papers = list(papers)
    pairs, skipped = search_paper_pairs(papers)
    if not pairs:
        reasons = "; ".join(sorted({str(item["reason"]) for item in skipped})) or "no papers were retrieved"
        raise RuntimeError(f"none of the query's papers has a usable KG + text pair ({reasons})")

    notes = notes or {}
    frame = _seed_frame(pairs)
    report.retrieved = len(pairs) + len(skipped)
    report.matched = len(frame)
    report.skipped = skipped
    report.rows = [
        {
            "paper_key": pair.paper_key,
            "title": pair.title,
            "paper_id": pair.kg_sha256[:24],
            "kg_path": str(pair.kg_path.resolve()),
            "paper_text_path": str(pair.body_path.resolve()),
            "text_provider": pair.text_provider or "",
            **dict(notes.get(pair.paper_key, {})),
        }
        for pair in pairs
    ]
    return frame, report
