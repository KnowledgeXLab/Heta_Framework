"""Filesystem contracts shared by the orchestrator and backends."""

import hashlib
import json
import shutil
from pathlib import Path


class NeedsMaterialsError(Exception):
    """A valid author response reporting insufficient source material."""


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def safe_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Artifact path must be relative and remain inside its output directory")
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes output directory")
    return target


def fingerprint(root: Path) -> str:
    """Hash normalized contents; permit internal links, reject escape/cycles/special files."""
    digest = hashlib.sha256()
    boundary = root.resolve(strict=True)

    def files(directory: Path, ancestors: frozenset[Path]):
        for path in sorted(directory.iterdir()):
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(boundary):
                raise ValueError(f"Package link escapes root: {path.relative_to(root)}")
            if path.is_dir():
                if resolved in ancestors:
                    raise ValueError(
                        f"Package contains cyclic directory link: {path.relative_to(root)}"
                    )
                yield from files(path, ancestors | {resolved})
            elif path.is_file():
                yield path
            else:
                raise ValueError(f"Package contains special file: {path.relative_to(root)}")

    for path in files(root, frozenset({boundary})):
        name = path.relative_to(root).as_posix().encode()
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def copy_tree(source: Path, destination: Path) -> None:
    fingerprint(source)  # Validate link boundaries/cycles before materializing links.
    shutil.copytree(source, destination)


def candidate_from_output(output: Path, task_id: str) -> Path:
    index = read_json(output / "author-result.json")
    if index.get("status") == "needs_materials":
        raise NeedsMaterialsError(
            "Author needs additional source material; inspect author-result.json"
        )
    if index.get("status") not in {"ready", "partial"}:
        raise ValueError("Author did not deliver a candidate; inspect author-result.json")
    tasks = index.get("tasks", [])
    if len(tasks) != 1 or tasks[0].get("task_id") != task_id:
        raise ValueError("Each task request must return exactly its assigned task_id")
    package = safe_path(output, tasks[0]["package_path"])
    required = (
        "task/instruction.md",
        "task/task.toml",
        "task/environment/Dockerfile",
        "task/solution/solve.sh",
        "task/tests/test.sh",
        "authoring/DESIGN.md",
        "authoring/sources.json",
    )
    missing = [relative for relative in required if not (package / relative).is_file()]
    if missing:
        raise ValueError("Missing candidate files: " + ", ".join(missing))
    fingerprint(package)
    return package
