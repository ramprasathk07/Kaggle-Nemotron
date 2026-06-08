"""
Solver for 8-bit boolean function problems.

Each output bit position is an independent boolean function of the 8 input bits.
Strategy (per Donald Galliano's analysis):
  For each output bit position j (0..7):
    Given N examples, find the simplest boolean function f such that
    f(input_bits) == output_bit_j for all examples.
  Search order (first match wins):
    1. Constants: 0, 1
    2. Identity: b[i] for each i in 0..7
    3. NOT:       ~b[i]
    4. 2-input:   all pairs i<k, all 14 two-input gates
    5. 3-input:   MAJ, CHO, PAR3, and composite (AO, OA, AX, OA, XA, XO)
    6. 4-input:   AOA, OAO, PAR4, XXXX, AXA
    7. Fallback:  brute-force BDD-style search or "unknown"
"""

from itertools import combinations, product
from .base_solver import BaseSolver, SolverResult


# ── Boolean gate helpers ─────────────────────────────────────────────────────

def _maj(a: int, b: int, c: int) -> int:
    return 1 if (a + b + c) >= 2 else 0

def _cho(a: int, b: int, c: int) -> int:
    return b if a else c

def _par3(a: int, b: int, c: int) -> int:
    return (a ^ b ^ c)

def _par4(a: int, b: int, c: int, d: int) -> int:
    return (a ^ b ^ c ^ d)

# All 14 non-trivial two-input boolean functions (excluding constant 0/1 and
# single-variable which are handled separately).
_TWO_INPUT_GATES: list[tuple[str, callable]] = [
    ("AND",  lambda a, b: a & b),
    ("OR",   lambda a, b: a | b),
    ("XOR",  lambda a, b: a ^ b),
    ("NAND", lambda a, b: 1 - (a & b)),
    ("NOR",  lambda a, b: 1 - (a | b)),
    ("XNOR", lambda a, b: 1 - (a ^ b)),
    ("A_AND_NOT_B",    lambda a, b: a & (1 - b)),
    ("NOT_A_AND_B",    lambda a, b: (1 - a) & b),
    ("A_OR_NOT_B",     lambda a, b: a | (1 - b)),
    ("NOT_A_OR_B",     lambda a, b: (1 - a) | b),
    ("A_XNOR_NOT_B",   lambda a, b: 1 - (a ^ (1 - b))),  # = XNOR(a, not b) = a XOR b
    ("NOT_A_XNOR_NOT_B", lambda a, b: 1 - ((1-a) ^ (1-b))),
    ("NOT_A_XNOR_B",   lambda a, b: 1 - ((1 - a) ^ b)),
    ("A_XNOR_B_AND_C", lambda a, b: (a ^ b)),  # same as XOR, kept for completeness
]

# Deduplicate by function identity (keep unique by name)
_TWO_INPUT_GATES = [
    ("AND",  lambda a, b: a & b),
    ("OR",   lambda a, b: a | b),
    ("XOR",  lambda a, b: a ^ b),
    ("NAND", lambda a, b: 1 - (a & b)),
    ("NOR",  lambda a, b: 1 - (a | b)),
    ("XNOR", lambda a, b: 1 - (a ^ b)),
    ("A_AND_NOT_B",  lambda a, b: a & (1 - b)),
    ("NOT_A_AND_B",  lambda a, b: (1 - a) & b),
    ("A_OR_NOT_B",   lambda a, b: a | (1 - b)),
    ("NOT_A_OR_B",   lambda a, b: (1 - a) | b),
]

# Three-input function groups
_THREE_INPUT_GATES: list[tuple[str, callable]] = [
    ("MAJ",  _maj),
    ("CHO",  _cho),
    ("PAR3", _par3),
    # Composite AO: (a AND b) OR c
    ("AO",   lambda a, b, c: (a & b) | c),
    # Composite OA: (a OR b) AND c
    ("OA",   lambda a, b, c: (a | b) & c),
    # Composite AX: (a AND b) XOR c
    ("AX",   lambda a, b, c: (a & b) ^ c),
    # Composite OX: (a OR b) XOR c
    ("OX",   lambda a, b, c: (a | b) ^ c),
    # Composite XA: (a XOR b) AND c
    ("XA",   lambda a, b, c: (a ^ b) & c),
    # Composite XO: (a XOR b) OR c
    ("XO",   lambda a, b, c: (a ^ b) | c),
    # With negations
    ("NOT_MAJ", lambda a, b, c: 1 - _maj(a, b, c)),
    ("NOT_PAR3", lambda a, b, c: 1 - _par3(a, b, c)),
]

_FOUR_INPUT_GATES: list[tuple[str, callable]] = [
    ("PAR4",  _par4),
    ("NOT_PAR4", lambda a, b, c, d: 1 - _par4(a, b, c, d)),
    ("AOA",   lambda a, b, c, d: ((a & b) | c) & d),
    ("OAO",   lambda a, b, c, d: ((a | b) & c) | d),
    ("AXA",   lambda a, b, c, d: ((a & b) ^ c) & d),
    ("XXXX",  lambda a, b, c, d: a ^ b ^ c ^ d),
]


# ── Per-bit solver ────────────────────────────────────────────────────────────

def _find_bit_function(
    examples: list[tuple[list[int], int]]  # [(input_bits, target_output_bit)]
) -> "tuple[str, object] | None":
    """
    Find the simplest boolean function consistent with all examples.
    Returns (description, lambda(bits: list[int]) -> int) or None.
    """
    if not examples:
        return None

    def consistent(fn):
        return all(fn(inp) == out for inp, out in examples)

    # 1. Constants
    if consistent(lambda _: 0):
        return ("CONST_0", lambda bits: 0)
    if consistent(lambda _: 1):
        return ("CONST_1", lambda bits: 1)

    # 2. Identity / NOT for each input bit
    for i in range(8):
        if consistent(lambda bits, i=i: bits[i]):
            return (f"b[{i}]", lambda bits, i=i: bits[i])
        if consistent(lambda bits, i=i: 1 - bits[i]):
            return (f"NOT_b[{i}]", lambda bits, i=i: 1 - bits[i])

    # 3. Two-input gates for all pairs
    for i, k in combinations(range(8), 2):
        for name, fn2 in _TWO_INPUT_GATES:
            gate_fn = (lambda bits, i=i, k=k, fn2=fn2: fn2(bits[i], bits[k]))
            if consistent(gate_fn):
                return (f"{name}(b[{i}],b[{k}])", gate_fn)
            # Also try negated output
            neg_gate_fn = (lambda bits, i=i, k=k, fn2=fn2: 1 - fn2(bits[i], bits[k]))
            if consistent(neg_gate_fn):
                return (f"NOT_{name}(b[{i}],b[{k}])", neg_gate_fn)

    # 4. Three-input gates for all triples
    for i, j, k in combinations(range(8), 3):
        for name, fn3 in _THREE_INPUT_GATES:
            gate_fn = (lambda bits, i=i, j=j, k=k, fn3=fn3: fn3(bits[i], bits[j], bits[k]))
            if consistent(gate_fn):
                return (f"{name}(b[{i}],b[{j}],b[{k}])", gate_fn)

    # 5. Four-input gates for all quadruples
    for combo in combinations(range(8), 4):
        i, j, k, l = combo
        for name, fn4 in _FOUR_INPUT_GATES:
            gate_fn = (
                lambda bits, i=i, j=j, k=k, l=l, fn4=fn4: fn4(bits[i], bits[j], bits[k], bits[l])
            )
            if consistent(gate_fn):
                return (f"{name}(b[{i}],b[{j}],b[{k}],b[{l}])", gate_fn)

    return None


# ── Whole-vector transformation strategies ───────────────────────────────────

def _rol8(bits: list[int], k: int) -> list[int]:
    k = k % 8
    return bits[k:] + bits[:k]

def _ror8(bits: list[int], k: int) -> list[int]:
    k = k % 8
    return bits[-k:] + bits[:-k] if k > 0 else bits[:]

def _xor_bits(a: list[int], b: list[int]) -> list[int]:
    return [x ^ y for x, y in zip(a, b)]

def _not_bits(a: list[int]) -> list[int]:
    return [1 - x for x in a]

def _and_bits(a: list[int], b: list[int]) -> list[int]:
    return [x & y for x, y in zip(a, b)]

def _or_bits(a: list[int], b: list[int]) -> list[int]:
    return [x | y for x, y in zip(a, b)]


def _try_whole_vector_functions(
    parsed: list[tuple[list[int], list[int]]]
) -> "tuple[object, str]":
    """
    Try whole-vector transformations (rotations, XOR combos, etc.) that may
    explain the full 8-bit transformation rather than per-bit functions.
    Returns (fn_or_None, description).
    """
    if not parsed:
        return None, ""

    def consistent(fn):
        return all(fn(inp) == out for inp, out in parsed)

    # Single rotations (with optional NOT)
    for k in range(1, 8):
        if consistent(lambda b, k=k: _rol8(b, k)):
            return (lambda b, k=k: _rol8(b, k)), f"ROL({k})"
        if consistent(lambda b, k=k: _ror8(b, k)):
            return (lambda b, k=k: _ror8(b, k)), f"ROR({k})"
        if consistent(lambda b, k=k: _not_bits(_rol8(b, k))):
            return (lambda b, k=k: _not_bits(_rol8(b, k))), f"NOT(ROL({k}))"
        if consistent(lambda b, k=k: _not_bits(_ror8(b, k))):
            return (lambda b, k=k: _not_bits(_ror8(b, k))), f"NOT(ROR({k}))"

    # XOR of two rotations (SHA-256 sigma-like)
    for k1 in range(8):
        for k2 in range(k1 + 1, 8):
            if consistent(lambda b, k1=k1, k2=k2: _xor_bits(_rol8(b, k1), _rol8(b, k2))):
                return (lambda b, k1=k1, k2=k2: _xor_bits(_rol8(b, k1), _rol8(b, k2))), f"ROL({k1}) XOR ROL({k2})"
            if consistent(lambda b, k1=k1, k2=k2: _xor_bits(_ror8(b, k1), _ror8(b, k2))):
                return (lambda b, k1=k1, k2=k2: _xor_bits(_ror8(b, k1), _ror8(b, k2))), f"ROR({k1}) XOR ROR({k2})"
            if consistent(lambda b, k1=k1, k2=k2: _xor_bits(_rol8(b, k1), _ror8(b, k2))):
                return (lambda b, k1=k1, k2=k2: _xor_bits(_rol8(b, k1), _ror8(b, k2))), f"ROL({k1}) XOR ROR({k2})"

    # XOR of three rotations
    for k1 in range(8):
        for k2 in range(k1 + 1, 8):
            for k3 in range(k2 + 1, 8):
                if consistent(lambda b, k1=k1, k2=k2, k3=k3:
                               _xor_bits(_xor_bits(_rol8(b, k1), _rol8(b, k2)), _rol8(b, k3))):
                    return (
                        lambda b, k1=k1, k2=k2, k3=k3: _xor_bits(_xor_bits(_rol8(b, k1), _rol8(b, k2)), _rol8(b, k3)),
                        f"ROL({k1}) XOR ROL({k2}) XOR ROL({k3})"
                    )

    # XOR with constant mask
    for mask in range(256):
        mask_bits = [(mask >> (7 - i)) & 1 for i in range(8)]
        if consistent(lambda b, mb=mask_bits: _xor_bits(b, mb)):
            return (lambda b, mb=mask_bits: _xor_bits(b, mb)), f"XOR with mask {mask:08b}"

    # ROL then XOR with mask
    for k in range(1, 8):
        for mask in range(256):
            mask_bits = [(mask >> (7 - i)) & 1 for i in range(8)]
            if consistent(lambda b, k=k, mb=mask_bits: _xor_bits(_rol8(b, k), mb)):
                return (
                    lambda b, k=k, mb=mask_bits: _xor_bits(_rol8(b, k), mb),
                    f"ROL({k}) XOR mask {mask:08b}"
                )

    return None, ""


# ── Main solver ───────────────────────────────────────────────────────────────

class BitManipulationSolver(BaseSolver):

    def solve(self, prompt: str) -> SolverResult:
        raw_examples = self._parse_examples(prompt)
        # Also parse query from the last line
        query_str = None
        for line in reversed(prompt.strip().split("\n")):
            line = line.strip()
            if line.startswith("Now, determine") and ":" in line:
                query_str = line.split(":", 1)[1].strip()
                break

        if not raw_examples:
            return SolverResult(
                answer=None, category="bit_manipulation", confidence=0.0,
                reasoning="No examples parsed."
            )
        if query_str is None:
            return SolverResult(
                answer=None, category="bit_manipulation", confidence=0.0,
                reasoning="No query found."
            )

        # Parse examples into bit arrays
        parsed: list[tuple[list[int], list[int]]] = []
        for inp, out in raw_examples:
            inp = inp.strip()
            out = out.strip()
            if len(inp) != 8 or len(out) != 8:
                continue
            if not all(c in "01" for c in inp + out):
                continue
            parsed.append(([int(c) for c in inp], [int(c) for c in out]))

        if not parsed:
            return SolverResult(
                answer=None, category="bit_manipulation", confidence=0.0,
                reasoning="Could not parse any 8-bit examples."
            )

        query_str = query_str.strip()
        if len(query_str) != 8 or not all(c in "01" for c in query_str):
            return SolverResult(
                answer=None, category="bit_manipulation", confidence=0.0,
                reasoning=f"Query '{query_str}' is not a valid 8-bit string."
            )

        query_bits = [int(c) for c in query_str]

        # Strategy 1: Whole-vector transformation (faster, covers rotation patterns)
        whole_fn, whole_desc = _try_whole_vector_functions(parsed)
        if whole_fn is not None:
            result_bits = whole_fn(query_bits)
            answer = "".join(str(b) for b in result_bits)
            reasoning = (
                f"Whole-vector function: {whole_desc}\n"
                f"Query: {query_str} → {answer}"
            )
            return SolverResult(
                answer=answer,
                category="bit_manipulation",
                confidence=1.0,
                reasoning=reasoning,
                verified=True,
            )

        # Strategy 2: Per-bit independent function search
        output_bits: list[str] = []
        descriptions: list[str] = []
        confidence = 1.0

        for bit_pos in range(8):
            bit_examples = [
                (inp_bits, out_bits[bit_pos])
                for inp_bits, out_bits in parsed
            ]
            result = _find_bit_function(bit_examples)
            if result is None:
                output_bits.append("?")
                descriptions.append(f"  bit[{bit_pos}]: UNKNOWN (no function found)")
                confidence = 0.5
            else:
                desc, fn = result
                bit_val = fn(query_bits)
                output_bits.append(str(bit_val))
                descriptions.append(f"  bit[{bit_pos}]: {desc} → {bit_val}")

        if "?" in output_bits:
            answer = None
            confidence = 0.0
        else:
            answer = "".join(output_bits)

        reasoning = (
            f"Solved each of 8 output bits independently from {len(parsed)} examples:\n" +
            "\n".join(descriptions) +
            f"\nQuery: {query_str} → {answer}"
        )

        return SolverResult(
            answer=answer,
            category="bit_manipulation",
            confidence=confidence,
            reasoning=reasoning,
            verified=confidence == 1.0,
        )
