"""Local Linux execution ownership, cooperative cancellation and crash recovery."""

import fcntl
import os
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .artifacts import read_json, write_json
from .cleanup import cleanup_job


class LifecycleError(RuntimeError):
    """Execution cannot safely start or finish."""


def process_identity(pid: int) -> dict | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rindex(")") + 2 :].split()
        if fields[0] == "Z":
            return None
        return {
            "pid": pid,
            "start_ticks": fields[19],
            "pgid": int(fields[2]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def identity_alive(identity: dict | None) -> bool:
    return bool(identity and process_identity(identity["pid"]) == identity)


def run_root(directory: Path) -> Path:
    for parent in (directory, *directory.parents):
        if (parent / "run-config.json").is_file():
            return parent
    return directory


def request_cancel(root: Path) -> None:
    if not root.is_dir():
        raise ValueError("Cancellation target must be an existing run directory")
    (root / "cancel.request").touch()


def cancelled(root: Path) -> bool:
    return (root / "cancel.request").exists()


@contextmanager
def cancellation_signals(root: Path):
    """Main-thread handlers request cancellation; worker threads own their cleanup."""
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, lambda signum, frame: request_cancel(root))
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def group_alive(identity: dict) -> bool:
    if identity["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        return False
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            member = process_identity(int(path.name))
            if member and member["pgid"] == identity["pgid"]:
                return True
    return False


def stop_identity(identity: dict, grace: float) -> bool:
    """Stop the owned group, including descendants left after its leader exits."""
    if not group_alive(identity):
        return True
    current = process_identity(identity["pid"])
    if (current is not None and current != identity) or identity["pgid"] != identity["pid"]:
        return False
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
        if not group_alive(identity):
            return True
        current = process_identity(identity["pid"])
        if current is not None and current != identity:
            return False
        try:
            os.killpg(identity["pgid"], sig)
        except ProcessLookupError:
            return True
        deadline = time.monotonic() + wait
        while group_alive(identity) and time.monotonic() < deadline:
            time.sleep(0.1)
    return not group_alive(identity)


class ExecutionSession:
    """One Harbor worker and its owned Docker resources, with durable status."""

    def __init__(self, directory: Path, job_dir: Path, grace: float = 30):
        self.directory = directory.resolve()
        self.root = run_root(self.directory)
        self.path = self.directory / "execution.json"
        self.job_dir = job_dir.resolve()
        self.grace = grace
        self.process = None
        self.record = {
            "execution_id": uuid4().hex,
            "host": socket.gethostname(),
            "owner": process_identity(os.getpid()),
            "state": "starting",
            "job_dir": str(self.job_dir),
            "directory": str(self.directory),
            "started_at": time.time(),
            "outcome": None,
            "worker": None,
            "stop_grace_sec": grace,
            "docker_context": {
                key: os.environ.get(key)
                for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG")
            },
        }

    def save(self, state: str, **details) -> None:
        self.record.update(state=state, **details)
        write_json(self.root / "executions" / (self.record["execution_id"] + ".json"), self.record)
        write_json(self.path, self.record)

    def __enter__(self):
        if cancelled(self.root):
            raise LifecycleError("Run cancellation requested; no new worker started")
        self.lock = (self.directory / ".execution.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.save("starting")
            return self
        except BaseException:
            self.lock.close()
            raise

    def attach(self, process) -> None:
        self.process = process
        identity = process_identity(process.pid)
        self.save("running", worker=identity)

    def wait(self, deadline: float) -> int:
        until = time.monotonic() + deadline
        while True:
            if cancelled(self.root):
                self.record["outcome"] = "cancelled"
                raise LifecycleError("Run cancellation requested")
            remaining = until - time.monotonic()
            if remaining <= 0:
                self.record["outcome"] = "timeout"
                raise LifecycleError("Harbor worker exceeded deadline")
            try:
                code = self.process.wait(timeout=min(0.5, remaining))
                self.record["outcome"] = "completed" if code == 0 else "worker_failed"
                return code
            except subprocess.TimeoutExpired:
                pass

    def stop(self) -> bool:
        if self.record["state"] == "finished":
            return True
        self.save("cleaning")
        worker_file = self.directory / "worker-identity.json"
        identity = self.record.get("worker")
        if worker_file.exists():
            worker = read_json(worker_file)
            if worker.get("execution_id") != self.record["execution_id"]:
                return self.pending("Worker identity mismatch")
            identity = worker["identity"]
        if identity and not stop_identity(identity, self.grace):
            return self.pending("Worker did not stop")
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return self.pending("Worker exit not confirmed")
        # Worker startup holds this lock before registering or touching Harbor.
        with (self.directory / ".worker.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return self.pending("Worker startup or shutdown is still active")
            if not cleanup_job(self.job_dir, execution_id=self.job_dir.name):
                return self.pending("Owned Docker resources could not be confirmed absent")
            self.save("finished", finished_at=time.time(), cleanup_error=None)
            (self.directory / "cleanup-pending.json").unlink(missing_ok=True)
        return True

    def pending(self, reason: str) -> bool:
        self.save("cleanup_pending", cleanup_error=reason)
        write_json(
            self.directory / "cleanup-pending.json",
            {"execution_id": self.record["execution_id"], "reason": reason},
        )
        request_cancel(self.root)
        return False

    def __exit__(self, exc_type, exc, tb):
        if self.record["outcome"] is None:
            self.record["outcome"] = "interrupted" if exc_type else "completed"
        try:
            if not self.stop():
                raise LifecycleError(f"Cleanup incomplete; inspect {self.path}") from None
        finally:
            self.lock.close()


def recover_executions(root: Path) -> list[dict]:
    """Caller holds the run lock. Live owners are never displaced by recovery."""
    recovered = []
    root = root.resolve()
    for registry_path in sorted((root / "executions").glob("*.json")):
        record = read_json(registry_path)
        path = Path(record["directory"]) / "execution.json"
        if not path.resolve().is_relative_to(root):
            raise LifecycleError(f"Execution path escapes run directory: {registry_path}")
        if record.get("state") == "finished":
            continue
        if record.get("host") != socket.gethostname():
            raise LifecycleError(f"Execution belongs to another host: {path}")
        if identity_alive(record.get("owner")):
            raise LifecycleError(f"Execution owner is still alive: {path}")
        directory = path.parent
        job_dir = Path(record["job_dir"])
        if not job_dir.resolve().is_relative_to(directory.resolve()):
            raise LifecycleError(f"Job path escapes execution directory: {path}")
        with (directory / ".execution.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise LifecycleError(f"Execution is still locked: {path}") from None
            session = ExecutionSession(directory, job_dir)
            session.record = record
            session.grace = record.get("stop_grace_sec", 30)
            expected_context = {
                key: os.environ.get(key)
                for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG")
            }
            if record.get("docker_context", expected_context) != expected_context:
                raise LifecycleError(f"Docker connection settings changed: {path}")
            session.record["outcome"] = record.get("outcome") or "orphaned"
            if not session.stop():
                raise LifecycleError(f"Recovery cleanup incomplete: {path}")
            recovered.append({"execution_id": record["execution_id"], "state": "finished"})
    return recovered
