"""Run one native Harbor job in a dedicated process with scoped credentials."""

import asyncio
import fcntl
import os
import signal
import sys
from pathlib import Path

from harbor.job import Job
from harbor.models.job.config import JobConfig

from .artifacts import read_json, write_json
from .lifecycle import identity_alive, process_identity, run_root


async def run(config_path: Path, timeout: float) -> None:
    loop = asyncio.get_running_loop()
    current = asyncio.current_task()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, current.cancel)
    config = JobConfig.model_validate_json(config_path.read_text())
    async with asyncio.timeout(timeout):
        job = await Job.create(config)
        await job.run()


if __name__ == "__main__":
    config_path = Path(sys.argv[1]).resolve()
    directory = config_path.parent
    with (directory / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = read_json(directory / "execution.json")
        record = read_json(run_root(directory) / "executions" / (record["execution_id"] + ".json"))
        if record["state"] not in {"starting", "running"} or not identity_alive(record["owner"]):
            sys.exit(125)
        write_json(
            directory / "worker-identity.json",
            {
                "execution_id": record["execution_id"],
                "identity": process_identity(os.getpid()),
            },
        )
        asyncio.run(run(config_path, float(sys.argv[2])))
