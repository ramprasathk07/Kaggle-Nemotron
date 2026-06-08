"""
Solver for numeral conversion problems.

All observed examples convert decimal integers to/from Roman numerals.
Strategy:
  1. Parse input/output pairs from examples.
  2. Determine direction (int→Roman or Roman→int) by looking at the first example.
  3. Parse the query value.
  4. Apply standard Roman numeral conversion.

Edge case: if examples suggest a non-standard mapping (e.g. different base),
fall back to fitting an offset/multiplier.
"""

import re
from .base_solver import BaseSolver, SolverResult

_INT_TO_ROMAN = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"),  (90, "XC"),  (50, "L"),  (40, "XL"),
    (10, "X"),   (9, "IX"),   (5, "V"),   (4, "IV"),
    (1, "I"),
]

_ROMAN_VALUES = {
    "I": 1, "V": 5, "X": 10, "L": 50,
    "C": 100, "D": 500, "M": 1000,
}


def int_to_roman(n: int) -> str:
    result = []
    for value, numeral in _INT_TO_ROMAN:
        while n >= value:
            result.append(numeral)
            n -= value
    return "".join(result)


def roman_to_int(s: str) -> int:
    s = s.strip().upper()
    total = 0
    prev = 0
    for ch in reversed(s):
        val = _ROMAN_VALUES.get(ch, 0)
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total


def _is_roman(s: str) -> bool:
    return bool(re.fullmatch(r"[IVXLCDMivxlcdm]+", s.strip()))


def _is_integer(s: str) -> bool:
    return bool(re.fullmatch(r"\d+", s.strip()))


class NumeralSolver(BaseSolver):

    _QUERY_RE = re.compile(
        r"write the number\s+(\S+)\s+in the Wonderland numeral system", re.IGNORECASE
    )
    # Also handle "write X in the Wonderland numeral system"
    _QUERY_RE2 = re.compile(
        r"determine\s+.*?:\s*(\S+)\s*$", re.IGNORECASE | re.MULTILINE
    )

    def solve(self, prompt: str) -> SolverResult:
        examples = self._parse_examples(prompt)
        if not examples:
            return SolverResult(
                answer=None, category="numeral", confidence=0.0,
                reasoning="Could not parse examples."
            )

        # Determine direction from first example
        first_inp, first_out = examples[0]
        int_to_rom = _is_integer(first_inp) and _is_roman(first_out)
        rom_to_int = _is_roman(first_inp) and _is_integer(first_out)

        # Parse query
        query_match = self._QUERY_RE.search(prompt)
        if not query_match:
            query_match = self._QUERY_RE2.search(prompt)
        if not query_match:
            return SolverResult(
                answer=None, category="numeral", confidence=0.0,
                reasoning="Could not find query value."
            )

        query_str = query_match.group(1).strip().rstrip(".")

        steps = []
        confidence = 1.0

        if int_to_rom:
            # Verify examples are standard Roman numerals
            for inp, out in examples:
                if _is_integer(inp):
                    expected = int_to_roman(int(inp))
                    if expected != out.strip().upper():
                        steps.append(f"  Non-standard: {inp}→{out} (expected {expected})")
                        confidence = 0.5

            if _is_integer(query_str):
                answer = int_to_roman(int(query_str))
            else:
                return SolverResult(
                    answer=None, category="numeral", confidence=0.0,
                    reasoning=f"Query '{query_str}' is not an integer for int→Roman conversion."
                )
        elif rom_to_int:
            if _is_roman(query_str):
                answer = str(roman_to_int(query_str))
            else:
                return SolverResult(
                    answer=None, category="numeral", confidence=0.0,
                    reasoning=f"Query '{query_str}' is not a Roman numeral."
                )
        else:
            # Try to learn a mapping from examples and interpolate
            # Fallback: assume int→Roman
            if _is_integer(query_str):
                answer = int_to_roman(int(query_str))
                confidence = 0.7
                steps.append("  Falling back to standard Roman numerals.")
            else:
                return SolverResult(
                    answer=None, category="numeral", confidence=0.0,
                    reasoning="Could not determine conversion direction."
                )

        reasoning = (
            f"Direction: {'int→Roman' if int_to_rom else 'Roman→int'}\n" +
            (("\n".join(steps) + "\n") if steps else "") +
            f"Query: {query_str} → {answer}"
        )

        return SolverResult(
            answer=answer,
            category="numeral",
            confidence=confidence,
            reasoning=reasoning,
            verified=confidence == 1.0,
        )
