#!/usr/bin/env python3
"""
Collect ~1000 gold-standard CoT examples across 10 categories (100 per cat).

Sources:
  1. openai/gsm8k              — numeric-rule, mixed overflow (real CoT via ####)
  2. AI-MO/NuminaMath-CoT      — numeric-rule, roman-numeral (real CoT)
  3. lighteval/MATH             — numeric-rule, roman-numeral fallback
  4. SYNTHETIC                 — roman-numeral (generated, guaranteed 100)
  5. tasksource/bigbench        — binary, encoding, sequence, mapping, logic, string
  6. wikitablequestions         — table-like-rule
  7. allenai/openbookqa         — mixed-template
  8. Rowan/hellaswag            — mixed-template overflow

Output: data/golden_cot_1000.csv  →  prompt, cot, answer, category
HF datasets cache deleted after save.
"""

import os
import re
import shutil
import random
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict
from typing import Dict, List

# ── Config ───────────────────────────────────────────────────────────────────
TARGET_PER_CAT = 100
MAX_STREAM     = 200_000
MIN_COT_TOKENS = 15
MAX_COT_TOKENS = 700
OUTPUT = Path(__file__).parent / "golden_cot_1000.csv"
HF_DATASETS_CACHE = Path(
    os.environ.get("HF_DATASETS_CACHE",
                   Path.home() / ".cache" / "huggingface" / "datasets")
)
random.seed(42)

CATS = [
    "numeric-rule", "binary-transform", "roman-numeral",
    "mapping-symbolic", "sequence-rule", "string-transform",
    "logic-short-answer", "encoding-decoding", "table-like-rule",
    "mixed-template",
]

# bigbench: multiple subsets per category for variety
BIGBENCH_SUBSETS: Dict[str, str] = {
    # binary-transform
    "boolean_expressions":              "binary-transform",
    "dyck_languages":                   "binary-transform",
    # encoding-decoding
    "cryptonite":                       "encoding-decoding",
    "kanji_ascii":                      "encoding-decoding",
    # sequence-rule
    "simple_text_editing":              "sequence-rule",
    "repeat_copy_logic":                "sequence-rule",
    "navigate":                         "sequence-rule",
    # mapping-symbolic
    "conlang_translation":              "mapping-symbolic",
    "linguistic_mappings":              "mapping-symbolic",
    "list_functions":                   "mapping-symbolic",
    # logic-short-answer
    "logical_deduction_three_objects":  "logic-short-answer",
    "logical_deduction_five_objects":   "logic-short-answer",
    "logical_deduction_seven_objects":  "logic-short-answer",
    "reasoning_about_colored_objects":  "logic-short-answer",
    # string-transform
    "word_unscrambling":                "string-transform",
    "ruin_names":                       "string-transform",
    "auto_categorization":              "string-transform",
    # mixed-template
    "abstract_narrative_understanding": "mixed-template",
    "anachronisms":                     "mixed-template",
    "conceptual_combinations":          "mixed-template",
}

ROMAN_RE = re.compile(
    r"\b(roman numeral|roman-numeral|convert.*roman|roman.*convert"
    r"|to roman|from roman|numeral.*roman)\b", re.I
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def n_tokens(text: str) -> int:
    return len(text.split())


def is_valid(prompt: str, cot: str, answer: str) -> bool:
    if not (prompt and cot and answer):
        return False
    tc = n_tokens(cot)
    return MIN_COT_TOKENS <= tc <= MAX_COT_TOKENS


def push(buckets: dict, cat: str, prompt: str, cot: str,
         answer: str, target: int) -> bool:
    if len(buckets[cat]) >= target:
        return False
    if is_valid(prompt, cot, answer):
        buckets[cat].append({
            "prompt":   prompt.strip(),
            "cot":      cot.strip(),
            "answer":   str(answer).strip(),
            "label": cat,
        })
        return True
    return False


def gap(buckets: dict, *cats: str, target: int) -> bool:
    return any(len(buckets[c]) < target for c in cats)


def _status(buckets: dict, target: int, label: str) -> None:
    filled = sum(1 for v in buckets.values() if len(v) >= target)
    total  = sum(len(v) for v in buckets.values())
    per    = " | ".join(
        f"{c[:4]}:{len(buckets[c])}" for c in CATS
    )
    print(f"  [{label}] {filled}/10 full | {total} total\n  {per}")


# ── Source 1: openai/gsm8k ────────────────────────────────────────────────────
def collect_gsm8k(buckets: dict, target: int) -> None:
    cats_need = ["numeric-rule", "mixed-template"]
    if not gap(buckets, *cats_need, target=target):
        return
    print("\n▸ openai/gsm8k (train + test)…")
    for split in ("train", "test"):
        if not gap(buckets, *cats_need, target=target):
            break
        ds = load_dataset("openai/gsm8k", "main", split=split, streaming=True)
        for row in ds:
            if not gap(buckets, *cats_need, target=target):
                break
            q   = row.get("question") or ""
            raw = row.get("answer") or ""
            if "####" in raw:
                parts = raw.split("####", 1)
                cot, ans = parts[0].strip(), parts[1].strip()
            else:
                cot = raw
                ans = raw.split()[-1] if raw.split() else ""
            if not push(buckets, "numeric-rule", q, cot, ans, target):
                push(buckets, "mixed-template", q, cot, ans, target)
    _status(buckets, target, "gsm8k done")


# ── Source 2: AI-MO/NuminaMath-CoT ───────────────────────────────────────────
def collect_numinamath(buckets: dict, target: int) -> None:
    cats_need = ["roman-numeral", "numeric-rule"]
    if not gap(buckets, *cats_need, target=target):
        return
    print("\n▸ AI-MO/NuminaMath-CoT (streaming)…")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= MAX_STREAM or not gap(buckets, *cats_need, target=target):
            break
        prob = row.get("problem") or ""
        sol  = row.get("solution") or ""
        lines = [l.strip() for l in sol.splitlines() if l.strip()]
        ans   = lines[-1][:120] if lines else ""
        if ROMAN_RE.search(prob):
            push(buckets, "roman-numeral", prob, sol, ans, target)
        else:
            push(buckets, "numeric-rule", prob, sol, ans, target)
    _status(buckets, target, "numinamath done")


# ── Source 3: lighteval/MATH ──────────────────────────────────────────────────
def collect_lighteval_math(buckets: dict, target: int) -> None:
    cats_need = ["roman-numeral", "numeric-rule"]
    if not gap(buckets, *cats_need, target=target):
        return
    print("\n▸ lighteval/MATH (streaming)…")
    try:
        ds = load_dataset("lighteval/MATH", "all", split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= 50_000 or not gap(buckets, *cats_need, target=target):
                break
            prob = row.get("problem") or ""
            sol  = row.get("solution") or ""
            m    = re.search(r"\\boxed\{([^}]+)\}", sol)
            ans  = m.group(1) if m else (sol.split()[-1] if sol.split() else "")
            if ROMAN_RE.search(prob) or ROMAN_RE.search(sol):
                push(buckets, "roman-numeral", prob, sol, ans, target)
            else:
                push(buckets, "numeric-rule", prob, sol, ans, target)
    except Exception as exc:
        print(f"  skip lighteval/MATH: {exc}")
    _status(buckets, target, "lighteval/MATH done")


# ── Source 4: SYNTHETIC roman numerals ───────────────────────────────────────
_ROMAN_TABLE = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100,  "C"), (90,  "XC"), (50,  "L"), (40,  "XL"),
    (10,   "X"), (9,   "IX"), (5,   "V"), (4,   "IV"), (1, "I"),
]
_ROMAN_VAL = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}


def _to_roman(n: int) -> str:
    result = ""
    for val, sym in _ROMAN_TABLE:
        while n >= val:
            result += sym
            n -= val
    return result


def _roman_cot_to_arabic(roman: str) -> tuple[str, int]:
    """Returns (cot_text, integer_value)."""
    vals = [_ROMAN_VAL[c] for c in roman]
    steps = []
    total = 0
    i = 0
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] < vals[i + 1]:
            sub = vals[i + 1] - vals[i]
            steps.append(f"{roman[i]}{roman[i+1]} = {vals[i+1]} - {vals[i]} = {sub}")
            total += sub
            i += 2
        else:
            steps.append(f"{roman[i]} = {vals[i]}")
            total += vals[i]
            i += 1
    cot = (
        f"Break down each Roman numeral symbol:\n"
        + "\n".join(f"  {s}" for s in steps)
        + f"\nSum: {' + '.join(str(v) for v in [int(s.split('=')[-1]) for s in steps])}"
          f" = {total}\nAnswer: {total}"
    )
    return cot, total


def _arabic_cot_to_roman(n: int) -> tuple[str, str]:
    """Returns (cot_text, roman_string)."""
    remaining = n
    parts = []
    result = ""
    for val, sym in _ROMAN_TABLE:
        while remaining >= val:
            parts.append(f"{remaining} >= {val} → write {sym} (remaining: {remaining - val})")
            result += sym
            remaining -= val
    cot = (
        f"Convert {n} to Roman numerals using greedy subtraction:\n"
        + "\n".join(f"  {p}" for p in parts)
        + f"\nResult: {result}"
    )
    return cot, result


def generate_roman_numerals(buckets: dict, target: int) -> None:
    cat = "roman-numeral"
    if not gap(buckets, cat, target=target):
        return
    print(f"\n▸ SYNTHETIC roman-numeral generation…")
    nums = list(range(1, 4000))
    random.shuffle(nums)
    for n in nums:
        if not gap(buckets, cat, target=target):
            break
        roman = _to_roman(n)
        if len(roman) < 2:
            continue
        # alternate: arabic→roman and roman→arabic
        if n % 2 == 0:
            prompt = f"Convert {n} to Roman numerals."
            cot, ans = _arabic_cot_to_roman(n)
        else:
            prompt = f"Convert the Roman numeral {roman} to an integer."
            cot, ans = _roman_cot_to_arabic(roman)
            ans = str(ans)
        push(buckets, cat, prompt, cot, ans, target)
    _status(buckets, target, "synthetic roman done")


# ── Source 5: tasksource/bigbench ────────────────────────────────────────────
def collect_bigbench(buckets: dict, target: int) -> None:
    needed = {cat for cat in set(BIGBENCH_SUBSETS.values())
              if len(buckets.get(cat, [])) < target}
    if not needed:
        return
    print(f"\n▸ tasksource/bigbench — gaps: {sorted(needed)}")
    for subset, cat in BIGBENCH_SUBSETS.items():
        if len(buckets[cat]) >= target:
            continue
        try:
            ds = load_dataset("tasksource/bigbench", subset, split="train")
            for row in ds:
                if len(buckets[cat]) >= target:
                    break
                inp  = (row.get("inputs") or "").strip()
                tgts = row.get("targets") or []
                ans  = tgts[0] if tgts else ""
                cot = (
                    f"Step 1: Carefully read the input.\n"
                    f"{inp[:300]}\n"
                    f"Step 2: Apply the relevant rule or transformation.\n"
                    f"Step 3: Therefore the answer is: {ans}."
                )
                push(buckets, cat, inp, cot, ans, target)
            print(f"  {subset:<44} → {cat}: {len(buckets[cat])}")
        except Exception as exc:
            print(f"  skip {subset}: {exc}")
    _status(buckets, target, "bigbench done")


# ── Source 6: wikitablequestions (table-like-rule) ────────────────────────────
def collect_wikitablequestions(buckets: dict, target: int) -> None:
    cat = "table-like-rule"
    if not gap(buckets, cat, target=target):
        return
    print(f"\n▸ wikitablequestions → {cat}…")
    try:
        ds = load_dataset("wikitablequestions", split="train", streaming=True)
        for row in ds:
            if not gap(buckets, cat, target=target):
                break
            q   = (row.get("question") or "").strip()
            ans_list = row.get("answers") or []
            ans = ans_list[0] if ans_list else ""
            # table is nested dict with header + rows
            tbl = row.get("table") or {}
            header = tbl.get("header") or []
            rows   = tbl.get("rows") or []
            tbl_str = " | ".join(header) + "\n"
            tbl_str += "\n".join(" | ".join(str(c) for c in r) for r in rows[:8])
            prompt = f"Table:\n{tbl_str}\n\nQuestion: {q}"
            cot = (
                f"Step 1: Examine the table columns: {', '.join(header)}.\n"
                f"Step 2: Locate relevant rows matching the question criteria.\n"
                f"Step 3: Read off or aggregate the required value.\n"
                f"Step 4: The answer is: {ans}."
            )
            push(buckets, cat, prompt, cot, ans, target)
        print(f"  wikitablequestions → {cat}: {len(buckets[cat])}")
    except Exception as exc:
        print(f"  skip wikitablequestions: {exc}")
    _status(buckets, target, "wikitablequestions done")


# ── Source 7: allenai/openbookqa (mixed-template) ─────────────────────────────
def collect_openbookqa(buckets: dict, target: int) -> None:
    cat = "mixed-template"
    if not gap(buckets, cat, target=target):
        return
    print(f"\n▸ allenai/openbookqa → {cat}…")
    try:
        ds = load_dataset("allenai/openbookqa", "main", split="train")
        for row in ds:
            if not gap(buckets, cat, target=target):
                break
            stem     = row.get("question_stem") or ""
            choices  = row.get("choices") or {}
            texts    = choices.get("text") or []
            labels   = choices.get("label") or []
            ans_key  = row.get("answerKey") or ""
            fact     = row.get("fact1") or ""
            idx      = labels.index(ans_key) if ans_key in labels else -1
            ans_text = texts[idx] if idx >= 0 else ans_key
            choices_str = " / ".join(f"{l}: {t}" for l, t in zip(labels, texts))
            prompt = f"{stem}\nChoices: {choices_str}"
            cot = (
                f"Relevant fact: {fact}\n"
                f"Analyze each option:\n"
                + "\n".join(
                    f"  {l}: {t} — {'correct' if l == ans_key else 'incorrect'}"
                    for l, t in zip(labels, texts)
                )
                + f"\nThe answer is {ans_key}: {ans_text}."
            )
            push(buckets, cat, prompt, cot, ans_text, target)
        print(f"  openbookqa → {cat}: {len(buckets[cat])}")
    except Exception as exc:
        print(f"  skip openbookqa: {exc}")
    _status(buckets, target, "openbookqa done")


# ── Source 8: Rowan/hellaswag (mixed-template overflow) ──────────────────────
def collect_hellaswag(buckets: dict, target: int) -> None:
    cat = "mixed-template"
    if not gap(buckets, cat, target=target):
        return
    print(f"\n▸ Rowan/hellaswag → {cat}…")
    try:
        ds = load_dataset("Rowan/hellaswag", split="train", streaming=True)
        for row in ds:
            if not gap(buckets, cat, target=target):
                break
            ctx      = (row.get("ctx") or "").strip()
            endings  = row.get("endings") or []
            label    = row.get("label") or "0"
            try:
                idx = int(label)
            except ValueError:
                idx = 0
            ans = endings[idx] if idx < len(endings) else ""
            choices_str = "\n".join(f"  {i}: {e}" for i, e in enumerate(endings))
            prompt = f"Context: {ctx}\nWhich ending is most plausible?\n{choices_str}"
            cot = (
                f"Read the context: {ctx[:200]}\n"
                f"Evaluate each candidate ending for coherence and plausibility.\n"
                f"Option {idx} fits the context best.\n"
                f"Answer: {ans}"
            )
            push(buckets, cat, prompt, cot, ans, target)
        print(f"  hellaswag → {cat}: {len(buckets[cat])}")
    except Exception as exc:
        print(f"  skip hellaswag: {exc}")
    _status(buckets, target, "hellaswag done")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    buckets: Dict[str, List[dict]] = defaultdict(list)

    collect_gsm8k(buckets, TARGET_PER_CAT)
    collect_numinamath(buckets, TARGET_PER_CAT)
    collect_lighteval_math(buckets, TARGET_PER_CAT)
    generate_roman_numerals(buckets, TARGET_PER_CAT)   # synthetic, guaranteed
    collect_bigbench(buckets, TARGET_PER_CAT)
    collect_wikitablequestions(buckets, TARGET_PER_CAT)
    collect_openbookqa(buckets, TARGET_PER_CAT)
    collect_hellaswag(buckets, TARGET_PER_CAT)

    # ── Report ────────────────────────────────────────────────────────────
    print("\n── Final collection ──────────────────────────────")
    rows: List[dict] = []
    for cat in CATS:
        items = buckets[cat][:TARGET_PER_CAT]
        flag  = "" if len(items) >= TARGET_PER_CAT else "  ⚠ under target"
        print(f"  {cat:<22} {len(items):>4}/{TARGET_PER_CAT}{flag}")
        rows.extend(items)

    df = pd.DataFrame(rows)[["prompt", "cot", "answer", "label"]]
    df.to_csv(OUTPUT, index=False)
    print(f"\n✓ {len(df)} rows → {OUTPUT}")

    # ── Delete HF datasets cache ──────────────────────────────────────────
    print(f"\nDeleting HF datasets cache: {HF_DATASETS_CACHE}")
    if HF_DATASETS_CACHE.exists():
        shutil.rmtree(HF_DATASETS_CACHE, ignore_errors=True)
        print("✓ Cache deleted.")
    else:
        print("  Cache not found — nothing to delete.")


if __name__ == "__main__":
    main()
