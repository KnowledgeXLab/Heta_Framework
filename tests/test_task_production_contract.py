"""Public task-production boundaries, without a native production runtime."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heta_framework.generation import (  # noqa: E402
    InputRequirement,
    ObjectInput,
    TaskRequest,
    TaskRequirements,
    TaskResult,
)


@pytest.mark.parametrize(
    "bindings, error",
    [
        ({}, ValueError),
        ({"materials": "raw/papers"}, TypeError),
        ({"materials": ObjectInput(prefix="raw/"), "extra": "x"}, ValueError),
        ({"materials": ObjectInput(prefix="raw/"), "brief": " "}, ValueError),
    ],
)
def test_invalid_bindings_are_rejected_before_execution(bindings, error):
    requirements = TaskRequirements(
        inputs=(
            InputRequirement("materials"),
            InputRequirement("brief", required=False, value_type="text"),
        )
    )
    with pytest.raises(error):
        requirements.validate(TaskRequest(output_prefix="tasks/run", inputs=bindings))


def test_raw_files_need_no_paper_metadata():
    requirements = TaskRequirements(inputs=(InputRequirement("materials"),))
    request = TaskRequest(
        output_prefix="tasks/run/",
        inputs={"materials": ObjectInput(keys=("raw/paper.pdf", "raw/code.py"))},
    )
    requirements.validate(request)
    assert request.output_prefix == "tasks/run"


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"keys": ("a",), "prefix": "raw"}, {"keys": ("a", "a")}, {"keys": ("/a",)}],
)
def test_ambiguous_or_invalid_object_selections_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ObjectInput(**kwargs)


def test_results_reference_store_objects_even_for_failed_runs():
    result = TaskResult("failed", "tasks/run/tasks.json", "tasks/run/report.json")
    assert result.tasks_manifest_key == "tasks/run/tasks.json"
    with pytest.raises(ValueError):
        TaskResult("failed", "", "tasks/run/report.json")
