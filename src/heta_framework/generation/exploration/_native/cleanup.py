"""Remove only Docker resources owned by trials recorded in one Harbor job."""

import re
import subprocess
from pathlib import Path

from .artifacts import read_json


def cleanup_job(job_dir: Path, execution_id: str | None = None) -> bool:
    prefixes = []
    for path in job_dir.glob("*/config.json"):
        try:
            name = read_json(path).get("trial_name", "")
        except (OSError, ValueError):
            return False
        if not name:
            continue
        # Harbor 0.23.0 derives all per-trial environment projects from this name.
        value = name.lower()
        if not re.match(r"^[a-z0-9]", value):
            value = "0" + value
        prefixes.append(re.sub(r"[^a-z0-9_-]", "-", value) + "__")
    if not prefixes and not execution_id:
        return True
    try:
        for _kind, list_args, remove_args in (
            ("container", ["ps", "-a"], ["rm", "-f", "-v"]),
            ("network", ["network", "ls"], ["network", "rm"]),
        ):
            listing = subprocess.run(
                [
                    "docker",
                    *list_args,
                    "--filter",
                    "label=com.docker.compose.project",
                    "--format",
                    '{{.ID}} {{.Label "com.docker.compose.project"}} {{.Label "sci.pipeline.execution"}}',
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if listing.returncode:
                return False
            owned = []
            for line in listing.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and (
                    any(parts[1].startswith(prefix) for prefix in prefixes)
                    or (execution_id and parts[-1] == execution_id)
                ):
                    owned.append(parts[0])
            if owned:
                removed = subprocess.run(
                    ["docker", *remove_args, *owned], capture_output=True, timeout=30
                )
                if removed.returncode:
                    return False
                check = subprocess.run(
                    ["docker", *list_args, "--format", "{{.ID}}"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if check.returncode or set(owned) & set(check.stdout.split()):
                    return False
    except (OSError, subprocess.TimeoutExpired):
        return False
    return True
