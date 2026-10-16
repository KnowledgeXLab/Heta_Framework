"""Stage Harbor seeds and expose only their explicitly public surface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from heta_framework.common.stores.object import ObjectStoreProtocol
from heta_framework.generation.types import ObjectInput


async def stage_seeds(
    selection: ObjectInput,
    store: ObjectStoreProtocol,
    root: Path,
    output_prefix: str,
    max_public_bytes: int,
) -> tuple[list[dict], list[dict]]:
    keys = (
        selection.keys
        if selection.prefix is None
        else tuple(
            item.key
            for item in await store.list(selection.prefix)
            if not selection.prefix or item.key.startswith(selection.prefix.rstrip("/") + "/")
        )
    )
    if not keys:
        raise ValueError("seed_tasks selection contains no objects")
    bases = sorted(
        {PurePosixPath(key).parent for key in keys if PurePosixPath(key).name == "task.toml"},
        key=str,
    )
    if not bases:
        raise ValueError("seed_tasks must contain Harbor task roots with task.toml")
    if any(a != b and b.is_relative_to(a) for a in bases for b in bases):
        raise ValueError("nested Harbor task roots are not supported")
    inputs, rows = [], []
    for base in bases:
        if str(base) != "." and PurePosixPath(output_prefix).is_relative_to(base):
            raise ValueError("output prefix overlaps a selected seed root")
        selected = [key for key in keys if PurePosixPath(key).is_relative_to(base)]
        relative_names = {PurePosixPath(key).relative_to(base).as_posix() for key in selected}
        if not {"instruction.md", "task.toml"} <= relative_names:
            raise ValueError(f"seed task is missing instruction.md or task.toml: {base}")
        seed_id = "seed-" + hashlib.sha256(str(base).encode()).hexdigest()[:16]
        directory = root / seed_id
        public = {}
        public_size = 0
        content_hashes = []
        for key in sorted(selected):
            if key == output_prefix or key.startswith(output_prefix + "/"):
                raise ValueError("output prefix overlaps selected seed tasks")
            relative = PurePosixPath(key).relative_to(base).as_posix()
            data = await store.get(key)
            digest = hashlib.sha256(data).hexdigest()
            inputs.append({"key": key, "sha256": digest})
            content_hashes.append((relative, digest))
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            # task.toml may contain private verifier commands; expose public files only.
            if relative == "instruction.md" or relative.startswith("environment/"):
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    public[relative] = {"bytes": len(data), "sha256": digest, "binary": True}
                else:
                    public_size += len(data)
                    if public_size > max_public_bytes:
                        raise ValueError(f"seed public text exceeds max_public_bytes: {base}")
                    public[relative] = text
        rows.append(
            {
                "seed_task_id": seed_id,
                "seed_path": str(directory),
                "source_prefix": str(base),
                "seed_sha256": hashlib.sha256(json.dumps(content_hashes).encode()).hexdigest(),
                "public_task": json.dumps(public, ensure_ascii=False),
            }
        )
    if any(not any(PurePosixPath(key).is_relative_to(base) for base in bases) for key in keys):
        raise ValueError("all selected objects must belong to a Harbor task root")
    return rows, inputs
