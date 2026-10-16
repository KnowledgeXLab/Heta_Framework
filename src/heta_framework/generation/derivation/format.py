"""Deterministic Harbor metadata; models generate task content, not TOML plumbing."""

import json
from pathlib import PurePosixPath

from .config import HetaGenTaskProducerDerivationConfig


def task_toml(candidate: dict, task_id: str, config: HetaGenTaskProducerDerivationConfig) -> str:
    artifacts = candidate["artifacts"]
    if len(artifacts) != len(set(artifacts)):
        raise ValueError("duplicate output artifacts")
    for value in artifacts:
        path = PurePosixPath(value)
        if (
            not path.is_absolute()
            or ".." in path.parts
            or len(path.parts) < 3
            or (path.parts[1] not in {"workspace", "app"})
            or path == PurePosixPath("/workspace/inputs")
            or path.is_relative_to("/workspace/inputs")
        ):
            raise ValueError("output artifacts must be absolute paths under /workspace or /app")
    quote = lambda value: json.dumps(value, ensure_ascii=False)  # noqa: E731
    domain, field = candidate["domain"], candidate["field"]
    return (
        'schema_version = "1.4"\n'
        f"artifacts = {quote(artifacts)}\n\n"
        f"[task]\nname = {quote('terminal-bench-science/' + task_id)}\n"
        'authors = [{name = "HetaGen"}]\n'
        f"keywords = {quote([domain, field, 'derived'])}\n\n"
        '[metadata]\nauthor_name = "HetaGen"\nauthor_email = "generated@heta.invalid"\n'
        'author_organization = "Heta"\n'
        f"domain = {quote(domain)}\nfield = {quote(field)}\n"
        f"description = {quote(candidate['description'])}\ntags = {quote([domain, field])}\n\n"
        f"[agent]\ntimeout_sec = {config.validation_timeout_sec}\n\n"
        f"[environment]\ncpus = {config.environment_cpus}\n"
        f'memory_mb = {config.environment_memory_mb}\nnetwork_mode = "public"\n'
        f"build_timeout_sec = {config.job_overhead_timeout_sec}\n\n"
        '[verifier]\nenvironment_mode = "separate"\n'
        f"timeout_sec = {config.validation_timeout_sec}\n\n"
        f"[verifier.environment]\ncpus = {config.environment_cpus}\n"
        f'memory_mb = {config.environment_memory_mb}\nnetwork_mode = "no-network"\n'
        f"build_timeout_sec = {config.job_overhead_timeout_sec}\n"
    )
