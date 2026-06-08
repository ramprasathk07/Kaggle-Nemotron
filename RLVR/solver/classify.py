"""Deterministic category classifier based on keyword matching."""

CATEGORIES = [
    "bit_manipulation",
    "cipher",
    "numeral",
    "unit_conversion",
    "gravity",
    "equation",
]

_KEYWORDS = {
    "bit_manipulation": [
        "bit manipulation",
        "8-bit binary",
        "bit shifts",
        "XOR",
    ],
    "gravity": [
        "gravitational constant",
        "falling distance",
        "d = 0.5*g*t",
    ],
    "unit_conversion": [
        "unit conversion",
        "secret unit conversion",
    ],
    "cipher": [
        "secret encryption",
        "encryption rules",
        "decrypt the following",
    ],
    "numeral": [
        "numeral system",
        "different numeral system",
        "wonderland numeral system",
    ],
    "equation": [
        "transformation rules",
        "secret set of transformation",
        "determine the result for",
    ],
}


def classify_problem(prompt: str) -> str:
    """Return the category of the problem based on keyword matching."""
    p = prompt.lower()
    for category, keywords in _KEYWORDS.items():
        if any(kw.lower() in p for kw in keywords):
            return category
    return "unknown"
