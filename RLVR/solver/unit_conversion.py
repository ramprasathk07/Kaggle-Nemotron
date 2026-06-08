"""
Solver for unit conversion problems.

The problem gives examples:  X m becomes Y
Strategy:
  1. Parse (input, output) float pairs.
  2. Compute ratio_i = output_i / input_i for each example.
  3. Average the ratios.
  4. Parse the query value.
  5. answer = query * ratio, rounded to 2 decimal places.
"""

import re
import statistics
from .base_solver import BaseSolver, SolverResult


class UnitConversionSolver(BaseSolver):

    _EXAMPLE_RE = re.compile(
        r"([\d.]+)\s*m\s+becomes\s+([\d.]+)", re.IGNORECASE
    )
    _QUERY_RE = re.compile(
        r"convert the following measurement:\s*([\d.]+)\s*m", re.IGNORECASE
    )

    def solve(self, prompt: str) -> SolverResult:
        examples = self._EXAMPLE_RE.findall(prompt)
        query_match = self._QUERY_RE.search(prompt)

        if not examples:
            return SolverResult(
                answer=None, category="unit_conversion", confidence=0.0,
                reasoning="Could not parse conversion examples."
            )
        if not query_match:
            return SolverResult(
                answer=None, category="unit_conversion", confidence=0.0,
                reasoning="Could not find query value."
            )

        query = float(query_match.group(1))
        ratios = []
        steps = []
        for inp_str, out_str in examples:
            inp = float(inp_str)
            out = float(out_str)
            ratio = out / inp
            ratios.append(ratio)
            steps.append(f"  {inp} m → {out}: ratio = {ratio:.6f}")

        ratio = statistics.mean(ratios)
        answer_val = query * ratio
        answer = f"{answer_val:.2f}"

        reasoning = (
            f"Ratios from {len(ratios)} examples:\n" +
            "\n".join(steps) +
            f"\nMean ratio = {ratio:.6f}\n"
            f"Answer = {query} * {ratio:.6f} = {answer_val:.6f} ≈ {answer}"
        )

        return SolverResult(
            answer=answer,
            category="unit_conversion",
            confidence=1.0,
            reasoning=reasoning,
            verified=True,
            extra={"ratio": ratio, "query": query},
        )
