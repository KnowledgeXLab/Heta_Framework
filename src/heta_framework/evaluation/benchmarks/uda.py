"""UDA-Benchmark adapter."""

from __future__ import annotations

import csv
import json
import os
import zipfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from heta_framework.evaluation.evaluators.answer import AnswerContains
from heta_framework.evaluation.protocols import BenchmarkEvaluatorProtocol
from heta_framework.evaluation.types import (
    BenchmarkCase,
    BenchmarkDocument,
    BenchmarkEvidence,
    BenchmarkExpected,
    BenchmarkManifest,
    BenchmarkResource,
    BenchmarkRunUnit,
    BenchmarkWorkspace,
    PreparedBenchmark,
)

UdaSubset = Literal["fin", "tat", "paper_tab", "paper_text", "feta", "nq"]

UDA_GITHUB_URL = "https://github.com/qinchuanhui/UDA-Benchmark"
UDA_QA_BASE_URL = (
    "https://raw.githubusercontent.com/qinchuanhui/UDA-Benchmark/main/dataset/qa"
)
UDA_EXTENDED_BENCH_BASE_URL = (
    "https://raw.githubusercontent.com/qinchuanhui/UDA-Benchmark/main/"
    "dataset/extended_qa_info_bench"
)
UDA_SOURCE_DOCS_URL = "https://huggingface.co/datasets/qinchuanhui/UDA-QA"
UDA_SOURCE_DOCS_REPO_ID = "qinchuanhui/UDA-QA"

_SUBSET_QA_FILES: Mapping[UdaSubset, str] = {
    "fin": "fin_qa.csv",
    "tat": "tat_qa.csv",
    "paper_tab": "paper_tab_qa.csv",
    "paper_text": "paper_text_qa.csv",
    "feta": "feta_qa.csv",
    "nq": "nq_qa.csv",
}

_SUBSET_EXTENDED_FILES: Mapping[UdaSubset, str] = {
    "fin": "bench_fin_qa.json",
    "tat": "bench_tat_qa.json",
    "paper_tab": "bench_paper_tab_qa.json",
    "paper_text": "bench_paper_text_qa.json",
    "feta": "bench_feta_qa.json",
    "nq": "bench_nq_qa.json",
}

_SUBSET_SOURCE_DIRS: Mapping[UdaSubset, tuple[str, ...]] = {
    "fin": ("fin_docs",),
    "tat": ("tat_docs",),
    "paper_tab": ("paper_docs",),
    "paper_text": ("paper_docs",),
    "feta": ("wiki_feta_docs",),
    "nq": ("wiki_nq_docs",),
}


@dataclass(frozen=True)
class UdaBenchmark:
    """Adapter for one UDA-Benchmark subset."""

    subset: UdaSubset
    source_root: Path | None = None
    qa_path: Path | None = None
    extended_info_path: Path | None = None
    download_metadata: bool = True
    download_source_documents: bool = True
    source_documents_url: str = UDA_SOURCE_DOCS_URL
    evaluator_list: tuple[BenchmarkEvaluatorProtocol, ...] = field(
        default_factory=lambda: (AnswerContains(),)
    )

    def __post_init__(self) -> None:
        if self.subset not in _SUBSET_QA_FILES:
            raise ValueError(
                "subset must be one of: fin, tat, paper_tab, paper_text, feta, nq"
            )
        object.__setattr__(
            self,
            "source_root",
            Path(self.source_root) if self.source_root is not None else None,
        )
        object.__setattr__(
            self,
            "qa_path",
            Path(self.qa_path) if self.qa_path is not None else None,
        )
        object.__setattr__(
            self,
            "extended_info_path",
            Path(self.extended_info_path) if self.extended_info_path is not None else None,
        )
        object.__setattr__(self, "evaluator_list", tuple(self.evaluator_list))

    @property
    def manifest(self) -> BenchmarkManifest:
        """Return stable benchmark identity metadata."""
        return BenchmarkManifest(
            name=f"uda_{self.subset}",
            version="official",
            split="all",
            task_type="document_qa",
            build_scope="case",
            homepage=UDA_GITHUB_URL,
            license="CC-BY-SA-4.0",
            citation=(
                "UDA: A Benchmark Suite for Retrieval Augmented Generation in "
                "Real-world Document Analysis"
            ),
            metadata={
                "subset": self.subset,
                "qa_url": _qa_url(self.subset),
                "extended_info_url": _extended_info_url(self.subset),
                "source_docs_url": self.source_documents_url,
            },
        )

    def resources(self) -> tuple[BenchmarkResource, ...]:
        """Return UDA metadata and source-document resources."""
        return (
            BenchmarkResource(
                name=f"{self.subset}_qa",
                uri=_qa_url(self.subset),
                kind="file",
                metadata={"filename": _SUBSET_QA_FILES[self.subset]},
            ),
            BenchmarkResource(
                name=f"{self.subset}_extended_info_bench",
                uri=_extended_info_url(self.subset),
                kind="file",
                required=False,
                metadata={"filename": _SUBSET_EXTENDED_FILES[self.subset]},
            ),
            BenchmarkResource(
                name="source_documents",
                uri=self.source_documents_url,
                kind="dataset",
                metadata={
                    "default_cache_root": f"uda_{self.subset}/source_documents",
                    "expected_root": "dataset/src_doc_files or extracted UDA-QA source zip",
                    "source_archives": [
                        _source_documents_archive_url(
                            source_documents_url=self.source_documents_url,
                            source_dir=source_dir,
                        )
                        for source_dir in _SUBSET_SOURCE_DIRS[self.subset]
                    ],
                    "subset_dirs": list(_SUBSET_SOURCE_DIRS[self.subset]),
                },
            ),
        )

    async def prepare(self, workspace: BenchmarkWorkspace) -> PreparedBenchmark:
        """Locate source documents and QA metadata."""
        root_dir = workspace.cache_dir / self.manifest.name
        root_dir.mkdir(parents=True, exist_ok=True)
        qa_path = self.qa_path or root_dir / _SUBSET_QA_FILES[self.subset]
        extended_path = self.extended_info_path or root_dir / _SUBSET_EXTENDED_FILES[self.subset]
        if self.download_metadata:
            _download_if_missing(_qa_url(self.subset), qa_path)
            _download_if_missing(_extended_info_url(self.subset), extended_path)
        source_root = self.source_root or root_dir / "source_documents"
        if self.source_root is None and self.download_source_documents:
            _download_source_documents(
                source_documents_url=self.source_documents_url,
                subset=self.subset,
                target_root=source_root,
            )
        if not qa_path.exists():
            raise FileNotFoundError(
                f"UDA QA file not found for subset {self.subset!r}. "
                "Provide qa_path or set download_metadata=True."
            )
        if not source_root.exists():
            raise FileNotFoundError(
                f"UDA source_root does not exist: {source_root}. "
                "Pass source_root or keep download_source_documents=True."
            )
        return PreparedBenchmark(
            manifest=self.manifest,
            root_dir=root_dir,
            resources=self.resources(),
            metadata={
                "subset": self.subset,
                "qa_path": str(qa_path),
                "extended_info_path": str(extended_path) if extended_path.exists() else None,
                "source_root": str(source_root),
            },
        )

    async def documents(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkDocument]:
        """Yield unique source documents referenced by the QA rows."""
        qa_rows = _read_qa_rows(Path(str(prepared.metadata["qa_path"])))
        source_root = Path(str(prepared.metadata["source_root"]))
        seen: set[str] = set()
        for row in qa_rows:
            doc_name = row["doc_name"]
            if doc_name in seen:
                continue
            seen.add(doc_name)
            path = _find_source_document(source_root, self.subset, doc_name)
            yield BenchmarkDocument(
                document_id=_document_id(doc_name),
                name=path.name,
                media_type=_media_type(path),
                path=path,
                metadata={
                    "doc_name": doc_name,
                    "subset": self.subset,
                    "source_path": str(path),
                },
            )

    async def cases(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkCase]:
        """Yield QA cases for the selected UDA subset."""
        qa_rows = _read_qa_rows(Path(str(prepared.metadata["qa_path"])))
        extended = _load_extended_info(prepared)
        for row in qa_rows:
            doc_name = row["doc_name"]
            q_uid = row["q_uid"]
            extended_item = extended.get((doc_name, q_uid))
            yield BenchmarkCase(
                case_id=q_uid,
                query=row["question"],
                expected=BenchmarkExpected(
                    answers=_answers(row, extended_item),
                    evidence=_evidence(doc_name, extended_item, self.manifest),
                    value=_expected_value(extended_item),
                    metadata={
                        "subset": self.subset,
                        "doc_name": doc_name,
                        "q_uid": q_uid,
                    },
                ),
                labels={"subset": self.subset, "doc_name": doc_name},
                metadata={
                    "doc_name": doc_name,
                    "q_uid": q_uid,
                    "has_extended_info": extended_item is not None,
                },
            )

    async def run_units(
        self,
        prepared: PreparedBenchmark,
    ) -> AsyncIterator[BenchmarkRunUnit]:
        """Yield one run unit per source document.

        UDA questions are scoped by ``doc_name``. Building one KB per document
        preserves that benchmark meaning and avoids cross-document financial
        table contamination.
        """
        qa_rows = _read_qa_rows(Path(str(prepared.metadata["qa_path"])))
        cases_by_document: dict[str, list[str]] = {}
        for row in qa_rows:
            cases_by_document.setdefault(row["doc_name"], []).append(row["q_uid"])
        for doc_name, case_ids in cases_by_document.items():
            yield BenchmarkRunUnit(
                unit_id=_document_id(doc_name),
                document_ids=(_document_id(doc_name),),
                case_ids=tuple(case_ids),
                metadata={"doc_name": doc_name, "subset": self.subset},
            )

    def evaluators(self) -> tuple[BenchmarkEvaluatorProtocol, ...]:
        """Return default evaluators for UDA."""
        return self.evaluator_list


def _read_qa_rows(path: Path) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="|")
        required = {"doc_name", "q_uid", "question"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"UDA QA file missing required columns: {sorted(missing)}")
        for row in reader:
            rows.append({key: (value or "").strip() for key, value in row.items()})
    return tuple(rows)


def _load_extended_info(
    prepared: PreparedBenchmark,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    path_value = prepared.metadata.get("extended_info_path")
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("UDA extended info must be a JSON object keyed by doc_name")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for doc_name, items in value.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, Mapping) and isinstance(item.get("q_uid"), str):
                result[(str(doc_name), str(item["q_uid"]))] = item
    return result


def _answers(row: Mapping[str, str], extended_item: Mapping[str, Any] | None) -> tuple[str, ...]:
    answers: list[str] = []
    if extended_item is not None:
        raw_answers = extended_item.get("answers")
        if isinstance(raw_answers, Mapping):
            for value in raw_answers.values():
                if value is not None and str(value).strip():
                    answers.append(str(value).strip())
    for key, value in row.items():
        if key.startswith("answer_") and value.strip():
            answers.append(value.strip())
    return tuple(dict.fromkeys(answers))


def _expected_value(extended_item: Mapping[str, Any] | None) -> Any | None:
    if extended_item is None:
        return None
    answers = extended_item.get("answers")
    if isinstance(answers, Mapping) and "exe_answer" in answers:
        return answers["exe_answer"]
    return None


def _evidence(
    doc_name: str,
    extended_item: Mapping[str, Any] | None,
    manifest: BenchmarkManifest,
) -> tuple[BenchmarkEvidence, ...]:
    if extended_item is None:
        return ()
    source_key_prefix = f"raw/benchmarks/{manifest.name}/{manifest.split}/{_document_id(doc_name)}/"
    evidence = extended_item.get("evidence")
    if not isinstance(evidence, Mapping):
        return ()
    items: list[BenchmarkEvidence] = []
    for name, text in evidence.items():
        if text is None or str(text).strip() == "":
            continue
        items.append(
            BenchmarkEvidence(
                reference_id=str(name),
                locator={"source_key_prefix": source_key_prefix},
                text=str(text).strip(),
                metadata={"doc_name": doc_name, "evidence_name": str(name)},
            )
        )
    return tuple(items)


def _find_source_document(source_root: Path, subset: UdaSubset, doc_name: str) -> Path:
    search_roots = [
        source_root / item for item in _SUBSET_SOURCE_DIRS[subset]
    ] + [source_root]
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        candidates.extend(path for path in root.rglob(f"{doc_name}.*") if path.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"source document for doc_name={doc_name!r} not found under {source_root}"
        )
    candidates.sort(key=lambda item: (item.suffix != ".pdf", len(str(item)), str(item)))
    return candidates[0]


def _document_id(doc_name: str) -> str:
    normalized = "".join(char if char.isalnum() else "_" for char in doc_name.strip())
    normalized = "_".join(item for item in normalized.split("_") if item)
    if not normalized:
        raise ValueError("doc_name must contain at least one alphanumeric character")
    return normalized


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".html", ".htm"}:
        return "text/html"
    if suffix in {".txt", ".md"}:
        return "text/plain"
    return "application/octet-stream"


def _qa_url(subset: UdaSubset) -> str:
    return f"{UDA_QA_BASE_URL}/{_SUBSET_QA_FILES[subset]}"


def _extended_info_url(subset: UdaSubset) -> str:
    return f"{UDA_EXTENDED_BENCH_BASE_URL}/{_SUBSET_EXTENDED_FILES[subset]}"


def _download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_read_url(url))


def _download_source_documents(
    *,
    source_documents_url: str,
    subset: UdaSubset,
    target_root: Path,
) -> None:
    if _source_documents_present(target_root, subset):
        return
    repo_id = _huggingface_dataset_repo_id(source_documents_url)
    try:
        for source_dir in _SUBSET_SOURCE_DIRS[subset]:
            archive_path = target_root / "_downloads" / f"{source_dir}.zip"
            _download_if_missing(
                _huggingface_dataset_file_url(
                    repo_id=repo_id,
                    file_path=f"src_doc_files/{source_dir}.zip",
                ),
                archive_path,
            )
            _extract_zip(archive_path, target_root)
    except Exception as exc:
        raise RuntimeError(
            "failed to download UDA source documents. "
            "Pass source_root to use local documents, or set HF_TOKEN / "
            "HUGGING_FACE_HUB_TOKEN if the Hugging Face dataset requires access."
        ) from exc


def _source_documents_present(target_root: Path, subset: UdaSubset) -> bool:
    return all(
        (target_root / source_dir).exists()
        and any(path.is_file() for path in (target_root / source_dir).rglob("*"))
        for source_dir in _SUBSET_SOURCE_DIRS[subset]
    )


def _huggingface_dataset_repo_id(source_documents_url: str) -> str:
    parsed = urlparse(source_documents_url)
    if parsed.netloc != "huggingface.co":
        raise ValueError(
            "UDA source document download currently supports Hugging Face dataset URLs. "
            "Pass source_root to use local or externally downloaded source documents."
        )
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3 or parts[0] != "datasets":
        raise ValueError(f"invalid Hugging Face dataset URL: {source_documents_url}")
    return f"{parts[1]}/{parts[2]}"


def _huggingface_dataset_file_url(*, repo_id: str, file_path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{quote(repo_id, safe='/')}/resolve/main/"
        f"{quote(file_path, safe='/')}?download=true"
    )


def _source_documents_archive_url(*, source_documents_url: str, source_dir: str) -> str:
    repo_id = _huggingface_dataset_repo_id(source_documents_url)
    return _huggingface_dataset_file_url(
        repo_id=repo_id,
        file_path=f"src_doc_files/{source_dir}.zip",
    )


def _extract_zip(archive_path: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            destination = _safe_zip_destination(target_root, member.filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as output:
                output.write(source.read())


def _safe_zip_destination(target_root: Path, member_name: str) -> Path:
    destination = (target_root / member_name).resolve()
    root = target_root.resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"unsafe zip member path: {member_name}")
    return destination


def _read_url(url: str) -> bytes:
    request = Request(url, headers=_download_headers(url))
    with urlopen(request, timeout=120) as response:
        return response.read()


def _download_headers(url: str) -> dict[str, str]:
    headers = {"User-Agent": "heta-framework-evaluation/0.1"}
    if urlparse(url).netloc == "huggingface.co":
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


__all__ = [
    "UDA_EXTENDED_BENCH_BASE_URL",
    "UDA_GITHUB_URL",
    "UDA_QA_BASE_URL",
    "UDA_SOURCE_DOCS_URL",
    "UDA_SOURCE_DOCS_REPO_ID",
    "UdaBenchmark",
    "UdaSubset",
]
