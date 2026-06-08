"""
Synthetic problem generators for all 6 categories.

These generators create new problems with known ground-truth answers,
enabling unlimited training data generation.

Each generator produces problems in the SAME FORMAT as the competition data,
so the trained model generalizes to the actual test set.
"""

import random
import string
import math
import re
from itertools import combinations
from typing import Callable

# Import solvers to verify synthetic answers
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Vocabulary for cipher problems (same as competition) ─────────────────────

CIPHER_SUBJECTS = [
    'alice', 'cat', 'rabbit', 'mouse', 'hatter', 'queen', 'king', 'knight',
    'princess', 'wizard', 'dragon', 'bird', 'turtle', 'student', 'teacher',
]
CIPHER_VERBS = [
    'follows', 'creates', 'draws', 'dreams', 'chases', 'reads', 'sees',
    'watches', 'explores', 'discovers', 'imagines', 'writes', 'studies',
    'finds', 'found',
]
CIPHER_OBJECTS = [
    'castle', 'garden', 'book', 'mirror', 'key', 'map', 'door', 'forest',
    'puzzle', 'treasure', 'secret', 'tower', 'crystal', 'potion', 'message',
]
CIPHER_PREPOSITIONS = [
    'in', 'near', 'inside', 'through', 'under', 'around', 'beyond', 'above',
]
CIPHER_PLACES = [
    'wonderland', 'castle', 'forest', 'garden', 'valley', 'village',
    'palace', 'mountain', 'library', 'island', 'ocean', 'cave', 'school',
]
CIPHER_ADJECTIVES = [
    'the', 'magical', 'mysterious', 'golden', 'silver', 'bright', 'dark',
    'clever', 'colorful', 'hidden', 'ancient', 'strange', 'curious', 'wise',
]

ALL_CIPHER_WORDS = list(set(
    CIPHER_SUBJECTS + CIPHER_VERBS + CIPHER_OBJECTS +
    CIPHER_PREPOSITIONS + CIPHER_PLACES + CIPHER_ADJECTIVES
))


def _random_phrase(rng: random.Random, n_words: int = None) -> str:
    if n_words is None:
        n_words = rng.randint(2, 5)
    words = []
    for _ in range(n_words):
        pool = CIPHER_SUBJECTS + CIPHER_VERBS + CIPHER_OBJECTS + CIPHER_PLACES + CIPHER_ADJECTIVES
        words.append(rng.choice(pool))
    return " ".join(words)


def _make_substitution_cipher(rng: random.Random) -> dict[str, str]:
    """Create a random bijective substitution cipher over lowercase letters."""
    letters = list(string.ascii_lowercase)
    shuffled = letters[:]
    rng.shuffle(shuffled)
    return dict(zip(letters, shuffled))


def _apply_cipher(text: str, cipher: dict[str, str]) -> str:
    return "".join(cipher.get(c, c) for c in text)


# ── Gravity problem generator ─────────────────────────────────────────────────

def generate_gravity_problem(rng: random.Random, n_examples: int = 5) -> tuple[str, str]:
    """Returns (prompt, answer)."""
    g = round(rng.uniform(3.0, 30.0), 2)
    t_values = [round(rng.uniform(0.5, 6.0), 2) for _ in range(n_examples + 1)]
    examples = t_values[:n_examples]
    query_t = t_values[-1]

    lines = ["In Alice's Wonderland, the gravitational constant has been secretly changed. "
             "Here are some example observations:"]
    for t in examples:
        d = round(0.5 * g * t ** 2, 2)
        lines.append(f"For t = {t}s, distance = {d} m")
    lines.append(f"Now, determine the falling distance for t = {query_t}s given d = 0.5*g*t^2.")

    answer = str(round(0.5 * g * query_t ** 2, 2))
    return "\n".join(lines), answer


# ── Unit conversion problem generator ────────────────────────────────────────

def generate_unit_conversion_problem(rng: random.Random, n_examples: int = 5) -> tuple[str, str]:
    ratio = round(rng.uniform(0.3, 5.0), 4)
    inputs = [round(rng.uniform(1.0, 50.0), 2) for _ in range(n_examples + 1)]
    examples = inputs[:n_examples]
    query_inp = inputs[-1]

    lines = ["In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:"]
    for inp in examples:
        out = round(inp * ratio, 2)
        lines.append(f"{inp} m becomes {out:.2f}")
    lines.append(f"Now, convert the following measurement: {query_inp} m")

    answer = str(round(query_inp * ratio, 2))
    return "\n".join(lines), answer


# ── Numeral conversion problem generator ─────────────────────────────────────

_INT_TO_ROMAN_TABLE = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]

def _int_to_roman(n: int) -> str:
    result = []
    for val, sym in _INT_TO_ROMAN_TABLE:
        while n >= val:
            result.append(sym)
            n -= val
    return "".join(result)


def generate_numeral_problem(rng: random.Random, n_examples: int = 4) -> tuple[str, str]:
    used = set()
    examples_nums = []
    while len(examples_nums) < n_examples:
        n = rng.randint(1, 3999)
        if n not in used:
            used.add(n)
            examples_nums.append(n)
    query_n = rng.randint(1, 3999)
    while query_n in used:
        query_n = rng.randint(1, 3999)

    lines = ["In Alice's Wonderland, numbers are secretly converted into a different numeral system. "
             "Some examples are given below:"]
    for n in examples_nums:
        lines.append(f"{n} -> {_int_to_roman(n)}")
    lines.append(f"Now, write the number {query_n} in the Wonderland numeral system.")

    return "\n".join(lines), _int_to_roman(query_n)


# ── Cipher problem generator ──────────────────────────────────────────────────

def generate_cipher_problem(rng: random.Random, n_examples: int = 5) -> tuple[str, str]:
    cipher = _make_substitution_cipher(rng)

    # Generate example phrases and query
    all_phrases = [_random_phrase(rng, rng.randint(2, 5)) for _ in range(n_examples + 1)]
    example_phrases = all_phrases[:n_examples]
    query_phrase = all_phrases[-1]

    lines = ["In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:"]
    for phrase in example_phrases:
        encrypted = _apply_cipher(phrase, cipher)
        lines.append(f"{encrypted} -> {phrase}")
    lines.append(f"Now, decrypt the following text: {_apply_cipher(query_phrase, cipher)}")

    return "\n".join(lines), query_phrase


# ── Bit manipulation problem generator ───────────────────────────────────────

def _random_8bit_function(rng: random.Random) -> Callable[[list[int]], list[int]]:
    """Generate a random (but tractable) 8-bit boolean transformation."""
    fn_type = rng.choice(["xor_mask", "rol", "ror", "not_rol", "xor_two_rols", "per_bit_gate"])

    if fn_type == "xor_mask":
        mask = [rng.randint(0, 1) for _ in range(8)]
        return lambda bits, m=mask: [b ^ m_i for b, m_i in zip(bits, m)]
    elif fn_type == "rol":
        k = rng.randint(1, 7)
        return lambda bits, k=k: bits[k:] + bits[:k]
    elif fn_type == "ror":
        k = rng.randint(1, 7)
        return lambda bits, k=k: bits[-k:] + bits[:-k]
    elif fn_type == "not_rol":
        k = rng.randint(1, 7)
        return lambda bits, k=k: [1 - b for b in bits[k:] + bits[:k]]
    elif fn_type == "xor_two_rols":
        k1 = rng.randint(0, 7)
        k2 = rng.randint(0, 7)
        while k2 == k1:
            k2 = rng.randint(0, 7)
        def fn(bits, k1=k1, k2=k2):
            a = bits[k1:] + bits[:k1]
            b = bits[k2:] + bits[:k2]
            return [x ^ y for x, y in zip(a, b)]
        return fn
    else:  # per_bit_gate
        # Choose a random gate for each output bit
        bit_maps = []
        for _ in range(8):
            in_idx = [rng.randint(0, 7) for _ in range(rng.randint(1, 3))]
            gate = rng.choice(["and", "or", "xor", "nand", "nor"])
            bit_maps.append((in_idx, gate))

        def per_bit(bits, bm=bit_maps):
            result = []
            for in_idx, gate in bm:
                vals = [bits[i] for i in in_idx]
                if gate == "and":
                    v = 1
                    for x in vals: v &= x
                elif gate == "or":
                    v = 0
                    for x in vals: v |= x
                elif gate == "xor":
                    v = 0
                    for x in vals: v ^= x
                elif gate == "nand":
                    v = 1
                    for x in vals: v &= x
                    v = 1 - v
                else:  # nor
                    v = 0
                    for x in vals: v |= x
                    v = 1 - v
                result.append(v)
            return result
        return per_bit


def generate_bit_manipulation_problem(
    rng: random.Random, n_examples: int = 8
) -> tuple[str, str]:
    fn = _random_8bit_function(rng)

    def rand_bits():
        return [rng.randint(0, 1) for _ in range(8)]

    inputs = [rand_bits() for _ in range(n_examples + 1)]
    example_inputs = inputs[:n_examples]
    query_input = inputs[-1]

    lines = ["In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
             "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
             "and possibly majority or choice functions.",
             "",
             "Here are some examples of input -> output:"]
    for bits_in in example_inputs:
        bits_out = fn(bits_in)
        inp_str = "".join(map(str, bits_in))
        out_str = "".join(map(str, bits_out))
        lines.append(f"{inp_str} -> {out_str}")
    lines.append("")
    query_str = "".join(map(str, query_input))
    lines.append(f"Now, determine the output for: {query_str}")

    answer = "".join(map(str, fn(query_input)))
    return "\n".join(lines), answer


# ── Equation transformation problem generator ─────────────────────────────────

_SYM_CHARS = list("!@#$%^&*()[]{}|<>?/\\'\"`~;:,.+-=")


def _random_symbol_string(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(_SYM_CHARS) for _ in range(length))


def generate_equation_problem(
    rng: random.Random, n_examples: int = 4
) -> tuple[str, str]:
    """Generate a char-substitution equation transformation problem."""
    # Choose a random bijective substitution over the symbol set
    all_chars = list(set(_SYM_CHARS))
    rng.shuffle(all_chars)
    cipher = {}
    for i, c in enumerate(all_chars[:len(all_chars) // 2]):
        cipher[c] = all_chars[i + len(all_chars) // 2]
    # Make sure we have at least 10 mappings
    while len(cipher) < 10:
        c = rng.choice(_SYM_CHARS)
        if c not in cipher:
            tgt = rng.choice(_SYM_CHARS)
            if tgt not in cipher.values():
                cipher[c] = tgt

    def apply(s):
        return "".join(cipher.get(c, c) for c in s)

    expr_len = rng.randint(3, 6)
    inputs = [_random_symbol_string(rng, expr_len) for _ in range(n_examples + 1)]
    query = inputs[-1]

    lines = ["In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
             "Below are a few examples:"]
    for inp in inputs[:n_examples]:
        out = apply(inp)
        lines.append(f"{inp} = {out}")
    lines.append(f"Now, determine the result for: {query}")

    return "\n".join(lines), apply(query)


# ── Main generator ────────────────────────────────────────────────────────────

_GENERATORS = {
    "bit_manipulation": generate_bit_manipulation_problem,
    "cipher": generate_cipher_problem,
    "numeral": generate_numeral_problem,
    "unit_conversion": generate_unit_conversion_problem,
    "gravity": generate_gravity_problem,
    "equation": generate_equation_problem,
}


def generate_synthetic_problems(
    categories: list[str] | None = None,
    n_per_category: int = 2000,
    seed: int = 1337,
    verify_with_solver: bool = True,
) -> list[dict]:
    """
    Generate synthetic problems for all (or specified) categories.
    Returns list of dicts with: prompt, answer, category, source="synthetic".
    """
    if categories is None:
        categories = list(_GENERATORS.keys())

    rng = random.Random(seed)
    all_problems = []

    for cat in categories:
        gen_fn = _GENERATORS.get(cat)
        if gen_fn is None:
            print(f"  No generator for category '{cat}', skipping.")
            continue

        print(f"  Generating {n_per_category} synthetic '{cat}' problems...", end=" ", flush=True)
        problems = []
        attempts = 0

        while len(problems) < n_per_category and attempts < n_per_category * 5:
            attempts += 1
            try:
                prompt, answer = gen_fn(rng)
            except Exception as e:
                continue

            if verify_with_solver:
                # Quick sanity check: solver should agree
                try:
                    from solvers import solve, classify_problem
                    r = solve(prompt)
                    if r.answer is not None:
                        # Accept solver-verified OR answer directly
                        if r.answer.strip() == answer.strip():
                            problems.append({
                                "prompt": prompt,
                                "answer": answer,
                                "category": cat,
                                "source": "synthetic",
                            })
                        # For categories where solver works perfectly, require agreement
                        # For equation/bit_manip, accept the generated answer directly
                        elif cat in ("equation", "bit_manipulation"):
                            problems.append({
                                "prompt": prompt,
                                "answer": answer,
                                "category": cat,
                                "source": "synthetic",
                            })
                    else:
                        # Solver can't verify — still accept for hard categories
                        if cat in ("equation", "bit_manipulation"):
                            problems.append({
                                "prompt": prompt,
                                "answer": answer,
                                "category": cat,
                                "source": "synthetic",
                            })
                except Exception:
                    problems.append({
                        "prompt": prompt,
                        "answer": answer,
                        "category": cat,
                        "source": "synthetic",
                    })
            else:
                problems.append({
                    "prompt": prompt,
                    "answer": answer,
                    "category": cat,
                    "source": "synthetic",
                })

        print(f"{len(problems)} generated ({attempts} attempts)")
        all_problems.extend(problems)

    return all_problems


if __name__ == "__main__":
    import json, os
    print("Generating synthetic problems...")
    problems = generate_synthetic_problems(n_per_category=500, verify_with_solver=True)
    os.makedirs("data/synthetic", exist_ok=True)
    out_path = "data/synthetic/synthetic_problems.jsonl"
    with open(out_path, "w") as f:
        for p in problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(problems)} synthetic problems to {out_path}")
