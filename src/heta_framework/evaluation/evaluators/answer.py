"""Reusable answer evaluators for benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass

from heta_framework.evaluation.types import BenchmarkCase, EvaluationScore
from heta_framework.kb.search import QueryResponse


@dataclass(frozen=True)
class AnswerContains:
    """Whether the generated answer contains any expected answer string."""

    @property
    def name(self) -> str:
        """Return the stable evaluator name."""
        return "answer_contains"

    async def evaluate(
        self,
        *,
        case: BenchmarkCase,
        response: QueryResponse,
    ) -> EvaluationScore:
        """Score answer containment for one response."""
        expected = case.expected.answers
        if not expected:
            return EvaluationScore(
                name=self.name,
                value=False,
                passed=None,
                metadata={"skipped": True, "reason": "case has no expected answers"},
            )
        actual = _normalize(response.answer or "")
        matched = tuple(answer for answer in expected if _normalize(answer) in actual)
        return EvaluationScore(
            name=self.name,
            value=bool(matched),
            passed=bool(matched),
            metadata={"matched_answers": list(matched)},
        )


@dataclass(frozen=True)
class AnswerExactMatch:
    """Whether the generated answer exactly matches an expected answer."""

    @property
    def name(self) -> str:
        """Return the stable evaluator name."""
        return "answer_exact_match"

    async def evaluate(
        self,
        *,
        case: BenchmarkCase,
        response: QueryResponse,
    ) -> EvaluationScore:
        """Score exact answer match for one response."""
        expected = case.expected.answers
        if not expected:
            return EvaluationScore(
                name=self.name,
                value=False,
                passed=None,
                metadata={"skipped": True, "reason": "case has no expected answers"},
            )
        actual = _normalize(response.answer or "")
        matched = tuple(answer for answer in expected if _normalize(answer) == actual)
        return EvaluationScore(
            name=self.name,
            value=bool(matched),
            passed=bool(matched),
            metadata={"matched_answers": list(matched)},
        )


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


__all__ = [
    "AnswerContains",
    "AnswerExactMatch",
]
