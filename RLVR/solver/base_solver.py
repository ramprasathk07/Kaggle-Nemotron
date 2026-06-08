from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SolverResult:
    answer: Optional[str]
    category: str
    confidence: float          # 0.0-1.0
    reasoning: str             # human-readable explanation of solver steps
    verified: bool = False     # True if answer cross-checked against examples
    extra: dict = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.answer is not None and self.confidence > 0.0


class BaseSolver(ABC):
    """Abstract interface all category solvers must implement."""

    @abstractmethod
    def solve(self, prompt: str) -> SolverResult:
        """Parse the prompt, apply the deterministic algorithm, return a result."""
        ...

    def _parse_examples(self, prompt: str) -> list[tuple[str, str]]:
        """
        Extract (input, output) pairs from lines like:
            input -> output
        Returns a list of (input_str, output_str) tuples.
        """
        examples = []
        for line in prompt.split("\n"):
            line = line.strip()
            if " -> " in line:
                parts = line.split(" -> ", 1)
                if len(parts) == 2:
                    examples.append((parts[0].strip(), parts[1].strip()))
        return examples
