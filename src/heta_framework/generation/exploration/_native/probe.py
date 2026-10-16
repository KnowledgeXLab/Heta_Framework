"""A deterministic Harbor agent for a builder-authored negative submission."""

from pathlib import Path

from harbor.agents.base import BaseAgent


class IncorrectSubmissionAgent(BaseAgent):
    def __init__(self, *args, probe_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_path = Path(probe_path)

    @staticmethod
    def name() -> str:
        return "incorrect-submission"

    def version(self) -> str:
        return "1.0"

    async def setup(self, environment) -> None:
        await environment.upload_file(self.probe_path, "/tmp/incorrect-submission.sh")

    async def run(self, instruction, environment, context) -> None:
        result = await environment.exec("sh /tmp/incorrect-submission.sh", timeout_sec=120)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "probe.txt").write_text(
            f"exit_code={result.return_code}\n{result.stdout}\n{result.stderr}\n"
        )
        if result.return_code != 0:
            raise RuntimeError("Negative submission driver failed; see probe.txt")
