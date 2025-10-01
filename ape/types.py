from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Example:
    """A single instruction tuning example consisting of input and target text."""

    input_text: str
    target_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptCandidate:
    """Represents a prompt candidate discovered during the search process."""

    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Stores detailed evaluation statistics for a single prompt candidate."""

    candidate: PromptCandidate
    predictions: List[str]
    references: List[str]
    scores: List[float]

    @property
    def average_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)
