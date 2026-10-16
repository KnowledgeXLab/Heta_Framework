"""Input declarations and object-store handoffs for task production."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from heta_framework.common.stores.object.types import validate_object_key, validate_object_prefix


@dataclass(frozen=True)
class ObjectInput:
    """Select explicit object keys or a prefix from the supplied object store.

    Producers resolve selections and record the actual files used. A prefix
    identifies a collection, not an atomic snapshot or a scientific category.
    """

    keys: tuple[str, ...] = ()
    prefix: str | None = None

    def __post_init__(self) -> None:
        keys = tuple(validate_object_key(key) for key in self.keys)
        if bool(keys) == (self.prefix is not None):
            raise ValueError("provide nonempty keys or a prefix, exclusively")
        if len(set(keys)) != len(keys):
            raise ValueError("object keys must be unique")
        object.__setattr__(self, "keys", keys)
        if self.prefix is not None:
            object.__setattr__(self, "prefix", validate_object_prefix(self.prefix))


@dataclass(frozen=True)
class InputRequirement:
    """One named input slot; value_type describes transport, not paper content."""

    name: str
    description: str = ""
    required: bool = True
    value_type: Literal["objects", "text"] = "objects"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("input name must not be empty")
        object.__setattr__(self, "name", self.name.strip())
        if self.value_type not in {"objects", "text"}:
            raise ValueError("value_type must be objects or text")


@dataclass(frozen=True)
class TaskRequest:
    """Actual input bindings and the destination for one production run."""

    output_prefix: str
    inputs: Mapping[str, ObjectInput | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        prefix = validate_object_prefix(self.output_prefix)
        if not prefix:
            raise ValueError("output_prefix must not be empty")
        object.__setattr__(self, "output_prefix", prefix)
        bindings = dict(self.inputs)
        for name, value in bindings.items():
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ValueError("input names must be nonempty without surrounding whitespace")
            if not isinstance(value, (ObjectInput, str)):
                raise TypeError("input values must be ObjectInput or text")
        object.__setattr__(self, "inputs", bindings)


@dataclass(frozen=True)
class TaskRequirements:
    """Required input slots for the producer's selected configuration mode."""

    inputs: tuple[InputRequirement, ...] = ()

    def __post_init__(self) -> None:
        inputs = tuple(self.inputs)
        if len({item.name for item in inputs}) != len(inputs):
            raise ValueError("input requirement names must be unique")
        object.__setattr__(self, "inputs", inputs)

    def validate(self, request: TaskRequest) -> None:
        """Check bindings before production; native formats are adapter-owned.

        This performs no I/O. Object existence and empty prefix selections are
        checked by the producer when preparing the actual inputs.
        """
        slots = {item.name: item for item in self.inputs}
        unknown = request.inputs.keys() - slots.keys()
        if unknown:
            raise ValueError(f"unknown inputs: {', '.join(sorted(unknown))}")
        for item in self.inputs:
            if item.name not in request.inputs:
                if item.required:
                    raise ValueError(f"missing required input: {item.name}")
                continue
            value = request.inputs[item.name]
            if item.value_type == "objects" and not isinstance(value, ObjectInput):
                raise TypeError(f"input {item.name} must be ObjectInput")
            if item.value_type == "text":
                if not isinstance(value, str):
                    raise TypeError(f"input {item.name} must be text")
                if not value.strip():
                    raise ValueError(f"input {item.name} must not be empty text")


@dataclass(frozen=True)
class TaskResult:
    """Published task collection and report, separate from native exit status.

    succeeded can contain zero tasks after normal screening. partial/failed
    may retain delivered tasks; consult the manifest rather than status alone.
    Validation evidence belongs in the report, not a global validated boolean.
    """

    status: Literal["succeeded", "partial", "failed"]
    tasks_manifest_key: str
    report_key: str

    def __post_init__(self) -> None:
        if self.status not in {"succeeded", "partial", "failed"}:
            raise ValueError("status must be succeeded, partial, or failed")
        object.__setattr__(self, "tasks_manifest_key", validate_object_key(self.tasks_manifest_key))
        object.__setattr__(self, "report_key", validate_object_key(self.report_key))
