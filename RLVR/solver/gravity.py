"""
Solver for gravitational constant problems.

The problem gives observations:  For t = X s, distance = Y m
The model:  d = 0.5 * g * t^2
Strategy:
  1. Parse (t, d) pairs from examples.
  2. Compute g_i = 2*d_i / t_i^2 for each example.
  3. Average g estimates (robust to floating-point noise).
  4. Parse the query t from the last line.
  5. Compute answer = 0.5 * g * t_query^2, round to 2 decimal places.
"""

import re
import statistics
from .base_solver import BaseSolver, SolverResult


class GravitySolver(BaseSolver):

    _EXAMPLE_RE = re.compile(
        r"For\s+t\s*=\s*([\d.]+)\s*s,\s*distance\s*=\s*([\d.]+)\s*m", re.IGNORECASE
    )
    _QUERY_RE = re.compile(
        r"determine the falling distance for\s+t\s*=\s*([\d.]+)\s*s", re.IGNORECASE
    )

    def solve(self, prompt: str) -> SolverResult:
        examples = self._EXAMPLE_RE.findall(prompt)
        query_match = self._QUERY_RE.search(prompt)

        if not examples:
            return SolverResult(
                answer=None, category="gravity", confidence=0.0,
                reasoning="Could not parse any (t, d) examples from prompt."
            )
        if not query_match:
            return SolverResult(
                answer=None, category="gravity", confidence=0.0,
                reasoning="Could not find query time in prompt."
            )

        t_query = float(query_match.group(1))

        g_estimates = []
        steps = []
        for t_str, d_str in examples:
            t = float(t_str)
            d = float(d_str)
            g = 2.0 * d / (t ** 2)
            g_estimates.append(g)
            steps.append(f"  t={t}, d={d} → g = 2*{d}/{t}^2 = {g:.6f}")

        g = statistics.mean(g_estimates)
        answer_val = 0.5 * g * t_query ** 2
        answer = f"{answer_val:.2f}"

        reasoning = (
            f"Computed g from {len(g_estimates)} examples:\n" +
            "\n".join(steps) +
            f"\nMean g = {g:.6f}\n"
            f"Answer = 0.5 * {g:.6f} * {t_query}^2 = {answer_val:.6f} ≈ {answer}"
        )

        return SolverResult(
            answer=answer,
            category="gravity",
            confidence=1.0,
            reasoning=reasoning,
            verified=True,
            extra={"g": g, "t_query": t_query},
        )
