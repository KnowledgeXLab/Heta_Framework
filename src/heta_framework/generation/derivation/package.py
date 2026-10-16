"""Compile task content into two environments with identical trusted inputs."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from ._tb.task_factory import CANARY
from .config import HetaGenTaskProducerDerivationConfig
from .format import task_toml


def _file_map(items: list[dict]) -> dict[str, str]:
    files = {}
    for item in items:
        name = item["path"]
        path = PurePosixPath(name)
        if (
            not name
            or not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or path.as_posix() != name
            or any(ord(char) < 32 for char in name)
        ):
            raise ValueError(f"file path must be canonical and relative: {name!r}")
        if name in files:
            raise ValueError(f"candidate contains duplicate file paths: {name}")
        files[name] = item["content"]
    for name in files:
        if any(parent.as_posix() in files for parent in PurePosixPath(name).parents):
            raise ValueError(f"file path conflicts with a directory: {name}")
    return files


def compile_candidate(
    candidate: dict, task_id: str, config: HetaGenTaskProducerDerivationConfig
) -> dict:
    """Keep model code/data intact; own Docker wiring, metadata and input copies."""
    result = dict(candidate)
    files = _file_map(candidate["files"])
    public = _file_map(candidate["public_files"])
    if any(PurePosixPath(name).parts[0] == ".heta-inputs" for name in public):
        raise ValueError(".heta-inputs is reserved for the input directory marker")
    for name in files:
        path = PurePosixPath(name)
        if (
            len(path.parts) < 2
            or path.parts[0] not in {"solution", "tests"}
            or name in {"tests/Dockerfile", "tests/.dockerignore"}
            or path.parts[:2] == ("tests", "inputs")
        ):
            raise ValueError(f"files must contain solution/ or tests/ content only: {name}")
    required = {"solution/solve.sh", "tests/test.sh"}
    if not required <= files.keys():
        raise ValueError("files must include solution/solve.sh and tests/test.sh")

    # Raw CSV/JSON bytes remain valid; canaries belong in code and Dockerfiles.
    # Always create the input directory so COPY works for tasks with no fixtures.
    public.setdefault(".heta-inputs", "# " + CANARY + "\n")
    for name, content in public.items():
        files[f"environment/inputs/{name}"] = content
        files[f"tests/inputs/{name}"] = content
    common = (
        f"# {CANARY}\nFROM {config.environment_image}\n"
        "USER root\nWORKDIR /workspace\n"
        "RUN mkdir -p /workspace/inputs /workspace/output\n"
        "COPY inputs/ /workspace/inputs/\n"
        "ENTRYPOINT []\n"
    )
    files["environment/Dockerfile"] = common + 'CMD ["sleep", "infinity"]\n'
    files["tests/Dockerfile"] = common + 'COPY . /tests/\nCMD ["sleep", "infinity"]\n'
    suffix = (
        f"You have {config.validation_timeout_sec} seconds to complete this task. "
        "Do not cheat by using online solutions or hints specific to this task."
    )
    instruction = re.sub(
        r"You have [0-9]+ seconds to complete this task\. Do not cheat by using online "
        r"solutions or hints specific to this task\.",
        "",
        candidate["instruction_md"],
    ).rstrip()
    result.update(
        files=files,
        slug=task_id,
        instruction_md=instruction + "\n\n" + suffix + "\n",
        task_toml=task_toml(candidate, task_id, config),
    )
    return result
