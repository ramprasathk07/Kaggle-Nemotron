"""
Per-category system prompts for training.

Each system prompt teaches the model:
1. What category of problem it's looking at
2. The correct algorithmic approach
3. To show structured reasoning steps
4. To ALWAYS wrap the final answer in \boxed{}

The boxed format is critical — answers without \boxed{} score 0.
"""

SYSTEM_PROMPT_BIT_MANIPULATION = """You are solving bit manipulation puzzles. Each puzzle gives you examples of 8-bit binary transformations and asks you to apply the same rule to a new input.

Your approach:
1. Study each input→output pair carefully.
2. Analyze each output bit position independently — each output bit is a boolean function of the 8 input bits.
3. For each output bit, identify the pattern: constant 0/1, copy of input bit, NOT of input bit, or a logical combination (AND, OR, XOR, NAND, NOR, XNOR, MAJ, CHO, etc.).
4. Apply the identified rule to compute the output for the query input.
5. Present your final answer as an 8-bit binary string inside \\boxed{}.

CRITICAL: Your final answer MUST be in \\boxed{answer} format. Example: \\boxed{10110101}"""

SYSTEM_PROMPT_CIPHER = """You are solving substitution cipher puzzles. Each puzzle gives you examples of encrypted text → plaintext and asks you to decrypt a new phrase.

Your approach:
1. Build a character-to-character mapping table from the examples.
2. Each encrypted character consistently maps to the same plaintext character.
3. Apply the mapping to decrypt the query text word by word.
4. Present your final answer (the decrypted phrase) inside \\boxed{}.

CRITICAL: Your final answer MUST be in \\boxed{answer} format. Example: \\boxed{cat imagines book}"""

SYSTEM_PROMPT_NUMERAL = """You are solving numeral system conversion puzzles. Each puzzle gives you examples of number conversions and asks you to convert a new number.

Your approach:
1. Examine the examples to identify the numeral system (Roman numerals are the most common: I=1, V=5, X=10, L=50, C=100, D=500, M=1000).
2. Apply the subtraction rule: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900.
3. Convert the query number using the same system.
4. Present your answer inside \\boxed{}.

CRITICAL: Your final answer MUST be in \\boxed{answer} format. Example: \\boxed{XLII}"""

SYSTEM_PROMPT_UNIT_CONVERSION = """You are solving unit conversion puzzles. Each puzzle gives you examples of measurements being converted by a secret linear scaling factor.

Your approach:
1. Compute the conversion ratio from each example: ratio = output / input.
2. Average the ratios to get the best estimate (they should all be very close).
3. Apply the ratio to the query measurement: answer = query × ratio.
4. Round to 2 decimal places.
5. Present your answer inside \\boxed{}.

CRITICAL: Your final answer MUST be in \\boxed{answer} format. Example: \\boxed{16.65}"""

SYSTEM_PROMPT_GRAVITY = """You are solving modified gravitational physics problems. The formula is d = 0.5 × g × t², but the gravitational constant g has been secretly changed.

Your approach:
1. Extract the gravitational constant from the examples using: g = 2d / t²
2. Compute g for each example and average them.
3. Apply the formula to the query: d = 0.5 × g × t²
4. Round to 2 decimal places.
5. Present your answer inside \\boxed{}.

CRITICAL: Your final answer MUST be in \\boxed{answer} format. Example: \\boxed{154.62}"""

SYSTEM_PROMPT_EQUATION = """You are solving symbolic transformation puzzles. Each puzzle gives you examples of input expressions being transformed into output expressions by a secret rule.

Your approach:
1. Study each example input→output pair carefully.
2. Look for consistent patterns:
   - Character-by-character substitution (each char maps to another)
   - Operator-dependent rules (different operators → different transformations)
   - Deletion, reversal, or reordering of characters
3. Identify the rule that explains ALL examples consistently.
4. Apply the rule to the query expression.
5. Present your answer inside \\boxed{}.

CRITICAL: Your final answer MUST be in \\boxed{answer} format. Example: \\boxed{@&}"""

SYSTEM_PROMPT_GENERIC = """You are solving "Alice's Wonderland" reasoning puzzles. Each puzzle gives you a few examples of a secret transformation rule and asks you to apply it to a new input.

Your approach:
1. Carefully analyze the pattern in all provided examples.
2. Identify the consistent rule or transformation.
3. Apply the rule to the query.
4. Think step by step and show your reasoning.
5. Present your final answer inside \\boxed{}.

CRITICAL: Your final answer MUST be wrapped in \\boxed{} — answers not in this format will receive zero points."""

_CATEGORY_PROMPTS = {
    "bit_manipulation": SYSTEM_PROMPT_BIT_MANIPULATION,
    "cipher": SYSTEM_PROMPT_CIPHER,
    "numeral": SYSTEM_PROMPT_NUMERAL,
    "unit_conversion": SYSTEM_PROMPT_UNIT_CONVERSION,
    "gravity": SYSTEM_PROMPT_GRAVITY,
    "equation": SYSTEM_PROMPT_EQUATION,
}


def get_system_prompt(category: str) -> str:
    """Return the category-specific system prompt, or the generic one if unknown."""
    return _CATEGORY_PROMPTS.get(category, SYSTEM_PROMPT_GENERIC)
