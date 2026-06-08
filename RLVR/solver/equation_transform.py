"""
Solver for equation transformation problems.

Sub-types observed:
  1. Char-level substitution (same-length LHS→RHS): build cipher-like map.
  2. Operator deletion: LHS = A_op_B, RHS = AB (operator char removed from fixed position).
  3. Operator-dependent rules:
       - concat(A,B), concat(B,A), concat(rev(A),B), etc.
       - Per-char binary ops (XOR/OR/AND on ASCII values) with operator type determining rule
  4. Arithmetic (numeric operands with operator): various numeric functions.

Strategy:
  Detect operator position (fixed column in all LHS of same length).
  Test rule hypotheses against all examples, keep first consistent rule.
  Fall back to partial char-map for unmapped queries.
"""

import re
import itertools
from typing import Callable
from .base_solver import BaseSolver, SolverResult


# ── Char-map helpers (same-length substitution) ─────────────────────────────

def _build_char_map(examples: list[tuple[str, str]]) -> dict[str, str]:
    char_map: dict[str, str] = {}
    for inp, out in examples:
        if len(inp) != len(out):
            continue
        for ci, co in zip(inp, out):
            if ci not in char_map:
                char_map[ci] = co
    return char_map


def _verify_char_map(char_map: dict[str, str], examples: list[tuple[str, str]]) -> bool:
    for inp, expected in examples:
        if len(inp) != len(expected):
            return False
        predicted = "".join(char_map.get(c, "\x00") for c in inp)
        if predicted != expected:
            return False
    return True


def _apply_char_map(char_map: dict[str, str], text: str) -> tuple[str, bool]:
    result, ok = [], True
    for c in text:
        if c in char_map:
            result.append(char_map[c])
        else:
            result.append("?")
            ok = False
    return "".join(result), ok


# ── Operator-position detection ──────────────────────────────────────────────

def _find_operator_position(lhs_strings: list[str]) -> int | None:
    """
    Find a position where chars VARY across examples (i.e. it's an operand slot or operator slot).
    More precisely: find the position where the character is a known operator symbol
    (+, -, *, /, |, \, ^, <, >, ?, !, @, etc.) in at least one LHS.
    We look for the position that acts as the 'operator' — typically the one where different
    examples have different chars from a set of operator-like symbols.
    """
    if not lhs_strings or len(set(len(s) for s in lhs_strings)) > 1:
        return None
    n = len(lhs_strings[0])
    # Candidate operator chars (typically not alpha/digit but single-position symbols)
    op_chars = set("+-*/|^<>?!@#$%&~':;,=")
    for pos in range(1, n - 1):  # operators won't be at the very start or end usually
        chars_at_pos = set(s[pos] for s in lhs_strings)
        # If all chars at this position are operator-like symbols, it's the operator position
        if chars_at_pos.issubset(op_chars):
            return pos
    # Fallback: position where chars vary the most
    variance_pos = max(range(n), key=lambda p: len(set(s[p] for s in lhs_strings)))
    if len(set(s[variance_pos] for s in lhs_strings)) > 1:
        return variance_pos
    return None


# ── Operator rule candidates ─────────────────────────────────────────────────

def _rule_delete_op(left: str, op: str, right: str) -> str:
    return left + right

def _rule_concat_ba(left: str, op: str, right: str) -> str:
    return right + left

def _rule_keep_left(left: str, op: str, right: str) -> str:
    return left

def _rule_keep_right(left: str, op: str, right: str) -> str:
    return right

def _rule_concat_rev_left(left: str, op: str, right: str) -> str:
    return left[::-1] + right

def _rule_concat_left_rev_right(left: str, op: str, right: str) -> str:
    return left + right[::-1]

def _rule_concat_rev_both(left: str, op: str, right: str) -> str:
    return left[::-1] + right[::-1]

def _rule_op_then_left_right(left: str, op: str, right: str) -> str:
    return op + left + right

def _rule_op_then_right_left(left: str, op: str, right: str) -> str:
    return op + right + left

def _rule_keep_diff_with_op(left: str, op: str, right: str) -> str:
    """Keep chars that differ between left and right, prepend op."""
    diff = "".join(
        l for l, r in zip(left, right) if l != r
    ) + "".join(
        r for l, r in zip(left, right) if l != r
    )
    return op + "".join(l for l, r in zip(left, right) if l != r)

def _rule_sym_diff_op(left: str, op: str, right: str) -> str:
    """Chars from left that differ from right, preceded by op."""
    differ = [l for l, r in zip(left, right) if l != r]
    return op + "".join(differ)

def _rule_zip_map(fn: Callable[[str, str], str]):
    """Per-char binary operation."""
    def rule(left: str, op: str, right: str) -> str:
        if len(left) != len(right):
            return ""
        return "".join(fn(l, r) for l, r in zip(left, right))
    return rule

def _chr_xor(a: str, b: str) -> str:
    return chr(ord(a) ^ ord(b))

def _chr_and(a: str, b: str) -> str:
    return chr(ord(a) & ord(b))

def _chr_or(a: str, b: str) -> str:
    return chr(ord(a) | ord(b))

def _chr_add_mod(a: str, b: str) -> str:
    return chr((ord(a) + ord(b)) % 128)

def _chr_sub_mod(a: str, b: str) -> str:
    return chr((ord(a) - ord(b)) % 128)


_OP_RULES: list[tuple[str, Callable]] = [
    ("delete_op", _rule_delete_op),
    ("concat_ba", _rule_concat_ba),
    ("keep_left", _rule_keep_left),
    ("keep_right", _rule_keep_right),
    ("concat_rev_left_right", _rule_concat_rev_left),
    ("concat_left_rev_right", _rule_concat_left_rev_right),
    ("concat_rev_both", _rule_concat_rev_both),
    ("op_left_right", _rule_op_then_left_right),
    ("op_right_left", _rule_op_then_right_left),
    ("sym_diff_op", _rule_sym_diff_op),
    ("keep_diff_with_op", _rule_keep_diff_with_op),
    ("zip_xor", _rule_zip_map(_chr_xor)),
    ("zip_and", _rule_zip_map(_chr_and)),
    ("zip_or", _rule_zip_map(_chr_or)),
    ("zip_add_mod", _rule_zip_map(_chr_add_mod)),
    ("zip_sub_mod", _rule_zip_map(_chr_sub_mod)),
]


# ── Numeric strategies ────────────────────────────────────────────────────────

_NUMERIC_EXPR_RE = re.compile(
    r"^(\d+)\s*([^0-9\s]+)\s*(\d+)$"
)

_NUMERIC_OPS: list[tuple[str, Callable]] = [
    ("sub",       lambda a, b: a - b),
    ("sub_abs",   lambda a, b: abs(a - b)),
    ("add",       lambda a, b: a + b),
    ("mul",       lambda a, b: a * b),
    ("rev_a_mul_rev_b_then_rev",
        lambda a, b: int(str(int(str(a)[::-1]) * int(str(b)[::-1]))[::-1])),
    ("rev_a_sub_rev_b", lambda a, b: int(str(a)[::-1]) - int(str(b)[::-1])),
    ("rev_a_add_rev_b", lambda a, b: int(str(a)[::-1]) + int(str(b)[::-1])),
    ("digit_sum_sub", lambda a, b: sum(int(d) for d in str(a)) - sum(int(d) for d in str(b))),
    ("digit_sum_add", lambda a, b: sum(int(d) for d in str(a)) + sum(int(d) for d in str(b))),
    ("digit_sum_mul", lambda a, b: sum(int(d) for d in str(a)) * sum(int(d) for d in str(b))),
    ("concat_ab",  lambda a, b: int(str(a) + str(b))),
    ("concat_ba",  lambda a, b: int(str(b) + str(a))),
    ("mod",        lambda a, b: a % b if b != 0 else None),
    ("floor_div",  lambda a, b: a // b if b != 0 else None),
]


def _try_per_operator_rules(
    examples: list[tuple[str, str]],
    op_pos: int,
) -> tuple[dict[str, Callable] | None, str]:
    """
    For each distinct operator in examples, find a consistent rule.
    Returns ({op_char: rule_fn}, description_str) or (None, "").
    """
    # Group examples by operator char
    by_op: dict[str, list[tuple[str, str, str, str]]] = {}
    for lhs, rhs in examples:
        if len(lhs) <= op_pos:
            continue
        op_char = lhs[op_pos]
        left = lhs[:op_pos]
        right = lhs[op_pos + 1:]
        by_op.setdefault(op_char, []).append((left, op_char, right, rhs))

    op_rules: dict[str, Callable] = {}
    descs: list[str] = []

    for op_char, op_examples in by_op.items():
        # Try numeric rule first
        numeric_examples = []
        all_numeric = True
        for left, _, right, rhs in op_examples:
            if left.isdigit() and right.isdigit():
                numeric_examples.append((int(left), int(right), rhs))
            else:
                all_numeric = False
                break

        if all_numeric and numeric_examples:
            for name, nfn in _NUMERIC_OPS:
                try:
                    ok = all(
                        str(nfn(a, b)) == rhs
                        for a, b, rhs in numeric_examples
                        if nfn(a, b) is not None
                    ) and all(nfn(a, b) is not None for a, b, _ in numeric_examples)
                    if ok:
                        a_fn = nfn
                        op_rules[op_char] = lambda l, o, r, fn=a_fn: str(fn(int(l), int(r)))
                        descs.append(f"  op='{op_char}': numeric {name}")
                        break
                except Exception:
                    continue
            if op_char in op_rules:
                continue

        # Try symbolic rules
        for rule_name, rule_fn in _OP_RULES:
            try:
                ok = all(
                    rule_fn(left, op_char, right) == rhs
                    for left, _, right, rhs in op_examples
                )
                if ok:
                    op_rules[op_char] = rule_fn
                    descs.append(f"  op='{op_char}': {rule_name}")
                    break
            except Exception:
                continue

    if not op_rules:
        return None, ""
    return op_rules, "\n".join(descs)


# ── Main solver ───────────────────────────────────────────────────────────────

class EquationTransformSolver(BaseSolver):

    _QUERY_RE = re.compile(
        r"determine the result for:\s*(.+)$", re.IGNORECASE | re.MULTILINE
    )

    def solve(self, prompt: str) -> SolverResult:
        # Parse examples from lines between header and query
        lines = prompt.split("\n")
        examples: list[tuple[str, str]] = []
        in_section = False

        for line in lines:
            stripped = line.strip()
            if "below are a few examples" in stripped.lower():
                in_section = True
                continue
            if "now, determine" in stripped.lower():
                break
            if not in_section or not stripped:
                continue
            if " = " in stripped:
                parts = stripped.split(" = ", 1)
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    examples.append((parts[0].strip(), parts[1].strip()))
            elif "=" in stripped:
                parts = stripped.split("=", 1)
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    examples.append((parts[0].strip(), parts[1].strip()))

        query_match = self._QUERY_RE.search(prompt)
        if not query_match:
            return SolverResult(
                answer=None, category="equation", confidence=0.0,
                reasoning="No query found."
            )
        query = query_match.group(1).strip()

        if not examples:
            return SolverResult(
                answer=None, category="equation", confidence=0.0,
                reasoning="No examples parsed."
            )

        # Strategy 1: Same-length char-level substitution
        same_len_examples = [(i, o) for i, o in examples if len(i) == len(o)]
        if same_len_examples:
            char_map = _build_char_map(same_len_examples)
            if char_map and _verify_char_map(char_map, same_len_examples):
                # Check all examples are same-length
                if all(len(i) == len(o) for i, o in examples):
                    answer_str, fully_mapped = _apply_char_map(char_map, query)
                    if fully_mapped:
                        return SolverResult(
                            answer=answer_str, category="equation", confidence=1.0,
                            reasoning=f"Char substitution ({len(char_map)} mappings). "
                                      f"Query '{query}' → '{answer_str}'",
                            verified=True, extra={"char_map": char_map},
                        )

        # Strategy 2: Operator-at-fixed-position with per-op rules
        lhs_strings = [i for i, _ in examples]
        all_same_len = len(set(len(s) for s in lhs_strings)) == 1

        if all_same_len and len(lhs_strings[0]) == len(query):
            op_pos = _find_operator_position(lhs_strings)
            if op_pos is not None and 0 < op_pos < len(lhs_strings[0]) - 1:
                op_rules, op_desc = _try_per_operator_rules(examples, op_pos)
                if op_rules:
                    query_op = query[op_pos]
                    query_left = query[:op_pos]
                    query_right = query[op_pos + 1:]
                    rule_fn = op_rules.get(query_op)
                    if rule_fn is not None:
                        try:
                            answer_str = rule_fn(query_left, query_op, query_right)
                            if answer_str:
                                return SolverResult(
                                    answer=answer_str, category="equation", confidence=0.9,
                                    reasoning=f"Operator rules at pos {op_pos}:\n{op_desc}\n"
                                              f"Query '{query}' → '{answer_str}'",
                                    verified=True,
                                )
                        except Exception:
                            pass

        # Strategy 3: Partial char map (best-effort, low confidence)
        if same_len_examples:
            char_map = _build_char_map(same_len_examples)
            if char_map:
                answer_str, fully_mapped = _apply_char_map(char_map, query)
                answer_str = answer_str.replace("?", "")
                if answer_str and len(answer_str) > 0:
                    return SolverResult(
                        answer=answer_str, category="equation", confidence=0.3,
                        reasoning=f"Partial char map (low confidence). "
                                  f"Query '{query}' → '{answer_str}'",
                        verified=False,
                    )

        return SolverResult(
            answer=None, category="equation", confidence=0.0,
            reasoning="Could not solve with any strategy.",
        )
