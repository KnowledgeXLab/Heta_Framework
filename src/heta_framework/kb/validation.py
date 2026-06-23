"""Static validation types for knowledge recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

ValidationSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class RecipeValidationIssue:
    """One static recipe validation issue."""

    severity: ValidationSeverity
    code: str
    message: str
    step_index: int | None = None
    step_name: str | None = None
    details: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code.strip() == "":
            raise ValueError("code must not be empty")
        if self.message.strip() == "":
            raise ValueError("message must not be empty")
        if self.step_name is not None and self.step_name.strip() == "":
            raise ValueError("step_name must not be empty")
        object.__setattr__(
            self,
            "details",
            {
                str(key).strip(): str(value).strip()
                for key, value in self.details.items()
                if str(key).strip() and str(value).strip()
            },
        )


@dataclass(frozen=True)
class RecipeValidationResult:
    """Static validation result for a knowledge recipe."""

    issues: tuple[RecipeValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether the validation result contains no errors."""
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> tuple[RecipeValidationIssue, ...]:
        """Return validation errors."""
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[RecipeValidationIssue, ...]:
        """Return validation warnings."""
        return tuple(issue for issue in self.issues if issue.severity == "warning")


class RecipeValidationError(ValueError):
    """Raised when a recipe fails static validation."""

    def __init__(self, result: RecipeValidationResult) -> None:
        self.result = result
        messages = "; ".join(issue.message for issue in result.errors)
        super().__init__(messages or "recipe validation failed")
