"""
build_golden_cot_dataset.py
===========================
Gather a 250-per-category golden chain-of-thought (CoT) SFT corpus for the
Nemotron-3-Nano LoRA, aligned to the 10 reasoning families in EDA.ipynb.

Two sources, blended per category:

  1. SYNTHETIC golden CoT (primary)  -- deterministic solver writes the reasoning,
     so labels are perfect and the CoT is verifiably correct. This matches the
     competition (Alice-Wonderland *rule* puzzles: bit-manipulation, base
     conversion, ciphers, unit/seq rules) far better than generic web CoT.

  2. HuggingFace human CoT (optional augmentation) -- GSM8K / ProofWriter /
     ProntoQA / AQuA-RAT for the families where human rationales add value.
     Guarded by try/except: if `datasets` is missing or there's no internet,
     the script silently falls back to 100% synthetic. Toggle with USE_HF.

Output schema (matches data_manipulation/dataset_generated.csv so the existing
SFT pipeline -- build_assistant_text -> <think>{cot}</think>\\boxed{answer} --
consumes it unchanged):

    id, prompt, answer, type, generated_cot

`generated_cot` holds the reasoning ONLY (no \\boxed{}); the SFT notebook wraps
it and appends the boxed answer from the `answer` column.
"""

from __future__ import annotations
import os, csv, json, random, hashlib, argparse
from collections import defaultdict

# ───────────────────────────── config ─────────────────────────────
FAMILY_NAME_MAP = {
    0: "numeric-rule",
    1: "binary-transform",
    2: "roman-numeral",
    3: "mapping-symbolic",
    4: "sequence-rule",
    5: "string-transform",
    6: "logic-short-answer",
    7: "encoding-decoding",
    8: "table-like-rule",
    9: "mixed-template",
}
PER_CATEGORY = 250
SEED = 42
OUT_CSV = os.path.join(os.path.dirname(__file__), "golden_cot_2500.csv")
USE_HF = False          # set True to blend in HuggingFace human-CoT (needs internet + datasets)
HF_FRACTION = 0.4       # up to this fraction of a category may come from HF when available

rng = random.Random(SEED)

# ───────────────────────────── helpers ─────────────────────────────
WORDS = ["alice", "queen", "rabbit", "dragon", "wizard", "mouse", "castle",
         "garden", "mirror", "potion", "silver", "golden", "ancient", "hidden",
         "river", "forest", "tower", "secret", "puzzle", "island"]

def _rid(prompt: str, answer: str) -> str:
    return hashlib.sha1((prompt + "||" + str(answer)).encode("utf-8")).hexdigest()[:12]

def int_to_roman(n: int) -> str:
    vals = [(1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),
            (50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")]
    out, steps = [], []
    for v, sym in vals:
        while n >= v:
            out.append(sym); n -= v
            steps.append(f"subtract {v} -> append '{sym}' (remainder {n})")
    return "".join(out), steps

def roman_to_int(s: str):
    m = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
    total, prev, steps = 0, 0, []
    for ch in reversed(s):
        v = m[ch]
        if v < prev:
            total -= v; steps.append(f"'{ch}'={v} < {prev} -> subtract")
        else:
            total += v; steps.append(f"'{ch}'={v} >= {prev} -> add"); prev = v
    return total, list(reversed(steps))

# ──────────────────────── synthetic generators ────────────────────────
# Each returns (prompt, answer, cot) with a deterministic golden CoT.

def gen_numeric_rule():
    kind = rng.choice(["series_missing", "multistep"])
    if kind == "series_missing":
        a0 = rng.randint(1, 9); d = rng.randint(2, 9)
        seq = [a0 + d*i for i in range(6)]
        hole = rng.randint(1, 4)
        shown = [("?" if i == hole else seq[i]) for i in range(6)]
        ans = seq[hole]
        prompt = f"Find the missing number in the arithmetic series: {', '.join(map(str, shown))}"
        cot = (f"The series is arithmetic. Common difference d = {seq[1]}-{seq[0]} = {d}.\n"
               f"Position {hole} (0-indexed) = first term + d*{hole} = {a0} + {d}*{hole} = {ans}.")
        return prompt, str(ans), cot
    else:
        p = rng.randint(2, 12); q = rng.randint(2, 12); r = rng.randint(1, 9)
        ans = p*q + r
        prompt = (f"A box holds {p} bags with {q} marbles each, plus {r} loose marbles. "
                  f"How many marbles in total?")
        cot = (f"Marbles in bags = {p} * {q} = {p*q}.\n"
               f"Add loose marbles: {p*q} + {r} = {ans}.")
        return prompt, str(ans), cot

def gen_binary_transform():
    x = rng.randint(0, 255)
    op = rng.choice(["NOT", "XOR", "ROL", "ROR", "SHL", "SHR"])
    bx = format(x, "08b")
    if op == "NOT":
        y = (~x) & 0xFF
        by = format(y, "08b")
        cot = (f"Input  = {bx}\nNOT flips every bit.\nOutput = {by}")
        rule = "apply bitwise NOT (flip every bit)"
    elif op == "XOR":
        k = rng.randint(1, 255); y = x ^ k
        by, bk = format(y, "08b"), format(k, "08b")
        cot = (f"Input = {bx}\nXOR with key {bk}.\nBit-by-bit XOR -> {by}")
        rule = f"XOR with the 8-bit key {bk}"
    elif op == "ROL":
        n = rng.randint(1, 7); y = ((x << n) | (x >> (8-n))) & 0xFF
        by = format(y, "08b")
        cot = (f"Input = {bx}\nRotate left by {n}: move the top {n} bits to the bottom.\nOutput = {by}")
        rule = f"rotate left by {n}"
    elif op == "ROR":
        n = rng.randint(1, 7); y = ((x >> n) | (x << (8-n))) & 0xFF
        by = format(y, "08b")
        cot = (f"Input = {bx}\nRotate right by {n}: move the bottom {n} bits to the top.\nOutput = {by}")
        rule = f"rotate right by {n}"
    elif op == "SHL":
        n = rng.randint(1, 4); y = (x << n) & 0xFF
        by = format(y, "08b")
        cot = (f"Input = {bx}\nShift left by {n}, drop overflow, fill with zeros.\nOutput = {by}")
        rule = f"shift left by {n} (zeros in, overflow dropped)"
    else:  # SHR
        n = rng.randint(1, 4); y = (x >> n) & 0xFF
        by = format(y, "08b")
        cot = (f"Input = {bx}\nShift right by {n}, fill with zeros.\nOutput = {by}")
        rule = f"shift right by {n} (zeros in)"
    prompt = (f"In Alice's Wonderland, an 8-bit binary number is transformed by a secret rule: "
              f"{rule}. Determine the output for: {bx}")
    return prompt, format(y, "08b"), cot

def gen_roman_numeral():
    if rng.random() < 0.5:
        n = rng.randint(1, 3999)
        ans, steps = int_to_roman(n)
        prompt = f"Convert the integer {n} to a Roman numeral."
        cot = "Greedy largest-value-first:\n" + "\n".join(steps[:12])
        return prompt, ans, cot
    else:
        n = rng.randint(1, 3999)
        r, _ = int_to_roman(n)
        val, steps = roman_to_int(r)
        prompt = f"Convert the Roman numeral {r} to an integer."
        cot = "Scan right-to-left; subtract a smaller value before a larger one:\n" + "\n".join(steps[:12]) + f"\nTotal = {val}."
        return prompt, str(val), cot

def gen_mapping_symbolic():
    # random substitution cipher; decode one word
    letters = list("abcdefghijklmnopqrstuvwxyz")
    shuf = letters[:]; rng.shuffle(shuf)
    enc = dict(zip(letters, shuf))
    dec = {v: k for k, v in enc.items()}
    plain = rng.choice(WORDS)
    cipher = "".join(enc[c] for c in plain)
    examples = rng.sample(WORDS, 3)
    ex_lines = "\n".join(f"{''.join(enc[c] for c in w)} -> {w}" for w in examples)
    cot_lines = []
    for c in cipher:
        cot_lines.append(f"'{c}' maps to '{dec[c]}'")
    cot = ("Build the letter map from the examples, then decode each character:\n"
           + "\n".join(cot_lines) + f"\nDecoded word: {plain}")
    prompt = (f"A substitution cipher maps letters consistently. Examples:\n{ex_lines}\n"
              f"Decode: {cipher}")
    return prompt, plain, cot

def gen_sequence_rule():
    kind = rng.choice(["arith", "geom", "fib", "square"])
    if kind == "arith":
        a = rng.randint(1, 40); d = rng.randint(2, 19)
        seq = [a + d*i for i in range(5)]; nxt = a + d*5
        cot = f"Differences are constant: {d}. Next = {seq[-1]} + {d} = {nxt}."
    elif kind == "geom":
        a = rng.randint(1, 8); r = rng.randint(2, 5)
        seq = [a*(r**i) for i in range(5)]; nxt = a*(r**5)
        cot = f"Each term multiplies by {r}. Next = {seq[-1]} * {r} = {nxt}."
    elif kind == "fib":
        a, b = rng.randint(1, 12), rng.randint(1, 15)
        seq = [a, b]
        for _ in range(3): seq.append(seq[-1] + seq[-2])
        nxt = seq[-1] + seq[-2]
        cot = f"Each term is the sum of the previous two. Next = {seq[-1]} + {seq[-2]} = {nxt}."
    else:
        s = rng.randint(1, 25)
        seq = [(s+i)**2 for i in range(5)]; nxt = (s+5)**2
        cot = f"Terms are perfect squares of {s},{s+1},.... Next = {s+5}^2 = {nxt}."
    prompt = f"Find the next number in the sequence: {', '.join(map(str, seq))}, ?"
    return prompt, str(nxt), cot

def gen_string_transform():
    word = rng.choice(WORDS)
    op = rng.choice(["reverse", "caesar", "swapcase", "remove_vowels"])
    if op == "reverse":
        out = word[::-1]; rule = "reverse the string"
        cot = f"Write the characters back to front: {' '.join(word)} -> {' '.join(out)}."
    elif op == "caesar":
        k = rng.randint(1, 25)
        out = "".join(chr((ord(c)-97+k) % 26 + 97) for c in word)
        rule = f"shift every letter forward by {k} (Caesar)"
        cot = (f"Shift each letter by {k}:\n" +
               "\n".join(f"'{c}' -> '{chr((ord(c)-97+k)%26+97)}'" for c in word) +
               f"\nResult: {out}")
    elif op == "swapcase":
        src = word.capitalize()
        out = src.swapcase(); rule = "swap the case of every letter"
        cot = f"Flip case of each char of '{src}': {out}."
        word = src
    else:
        out = "".join(c for c in word if c not in "aeiou")
        rule = "remove all vowels"
        cot = f"Drop a,e,i,o,u from '{word}': keep consonants -> {out}."
    prompt = f"Apply this transformation to '{word}': {rule}. What is the result?"
    return prompt, out, cot

def gen_logic_short_answer():
    kind = rng.choice(["syllogism", "family", "boolean"])
    if kind == "syllogism":
        A, B, C = rng.sample(["dragons","wizards","knights","mages","trolls","elves"], 3)
        prompt = f"All {A} are {B}. All {B} are {C}. Are all {A} necessarily {C}? Answer yes or no."
        cot = (f"{A} ⊆ {B} and {B} ⊆ {C}. Subset relation is transitive, so {A} ⊆ {C}.")
        return prompt, "yes", cot
    elif kind == "family":
        x, y, z = rng.sample(WORDS, 3)
        prompt = f"{x} is the father of {y}. {y} is the father of {z}. What is {x} to {z}?"
        cot = f"{x} -> father of {y} -> father of {z}. Two generations up = grandfather."
        return prompt, "grandfather", cot
    else:
        a, b, c = (rng.choice([True, False]) for _ in range(3))
        val = a and (b or c)
        prompt = f"Evaluate the boolean: {a} AND ({b} OR {c})."
        cot = (f"Inner: {b} OR {c} = {b or c}. Then {a} AND {b or c} = {val}.")
        return prompt, str(val), cot

def gen_encoding_decoding():
    kind = rng.choice(["caesar_dec", "base", "ascii"])
    if kind == "caesar_dec":
        word = rng.choice(WORDS); k = rng.randint(1, 25)
        enc = "".join(chr((ord(c)-97+k) % 26 + 97) for c in word)
        prompt = f"Decode this Caesar cipher (shift {k}): {enc}"
        cot = ("Shift each letter back by " + str(k) + ":\n" +
               "\n".join(f"'{c}' -> '{chr((ord(c)-97-k)%26+97)}'" for c in enc) +
               f"\nDecoded: {word}")
        return prompt, word, cot
    elif kind == "base":
        n = rng.randint(8, 4000); base = rng.choice([2, 8, 16])
        digits = {2:"binary",8:"octal",16:"hexadecimal"}[base]
        if base == 2: enc = format(n, "b")
        elif base == 8: enc = format(n, "o")
        else: enc = format(n, "x").upper()
        prompt = f"Convert the {digits} number {enc} to decimal."
        cot = f"Interpret {enc} in base {base}: positional sum = {n}."
        return prompt, str(n), cot
    else:
        word = rng.choice(WORDS)[:4]
        codes = " ".join(str(ord(c)) for c in word)
        prompt = f"Decode this ASCII code sequence to text: {codes}"
        cot = ("Map each code to its character:\n" +
               "\n".join(f"{ord(c)} -> '{c}'" for c in word) + f"\nText: {word}")
        return prompt, word, cot

def gen_table_like_rule():
    items = rng.sample(["sword","shield","potion","map","key","torch","rope","gem"], 4)
    prices = {it: rng.randint(2, 20) for it in items}
    table = "\n".join(f"| {it:8} | {prices[it]:2} |" for it in items)
    buy = rng.sample(items, 2)
    total = sum(prices[b] for b in buy)
    prompt = (f"Price table:\n| item     | gold |\n{table}\n"
              f"How much gold to buy one {buy[0]} and one {buy[1]}?")
    cot = (f"Look up {buy[0]} = {prices[buy[0]]} gold and {buy[1]} = {prices[buy[1]]} gold.\n"
           f"Total = {prices[buy[0]]} + {prices[buy[1]]} = {total}.")
    return prompt, str(total), cot

SYNTH = {
    "numeric-rule":      gen_numeric_rule,
    "binary-transform":  gen_binary_transform,
    "roman-numeral":     gen_roman_numeral,
    "mapping-symbolic":  gen_mapping_symbolic,
    "sequence-rule":     gen_sequence_rule,
    "string-transform":  gen_string_transform,
    "logic-short-answer":gen_logic_short_answer,
    "encoding-decoding": gen_encoding_decoding,
    "table-like-rule":   gen_table_like_rule,
}

def gen_mixed_template():
    fam = rng.choice(list(SYNTH.keys()))
    return SYNTH[fam]()

SYNTH["mixed-template"] = gen_mixed_template

# ──────────────────────── optional HF augmentation ────────────────────────
def hf_rows(family: str, want: int):
    """Return up to `want` (prompt, answer, cot) tuples from HuggingFace for the
    few families with genuine human CoT. Silent [] on any failure (offline etc.)."""
    if not USE_HF or want <= 0:
        return []
    try:
        from datasets import load_dataset
    except Exception:
        return []
    out = []
    try:
        if family == "numeric-rule":
            ds = load_dataset("openai/gsm8k", "main", split="train")
            for r in ds.shuffle(seed=SEED).select(range(min(want*3, len(ds)))):
                ans = r["answer"].split("####")[-1].strip()
                cot = r["answer"].split("####")[0].strip()
                if ans and cot:
                    out.append((r["question"].strip(), ans, cot))
                if len(out) >= want: break
        elif family == "logic-short-answer":
            ds = load_dataset("tasksource/proofwriter", split="train")
            for r in ds.shuffle(seed=SEED).select(range(min(want*3, len(ds)))):
                q = r.get("question") or r.get("theory") or ""
                a = str(r.get("answer", "")).strip()
                c = str(r.get("proof") or r.get("reasoning") or "").strip()
                if q and a and c:
                    out.append((q.strip(), a, c))
                if len(out) >= want: break
        elif family == "mapping-symbolic":
            ds = load_dataset("aqua_rat", split="train")
            for r in ds.shuffle(seed=SEED).select(range(min(want*3, len(ds)))):
                q = r["question"].strip(); a = str(r["correct"]).strip()
                c = " ".join(r["rationale"].split())
                if q and a and c:
                    out.append((q, a, c))
                if len(out) >= want: break
    except Exception as e:
        print(f"  [hf] {family}: load failed ({type(e).__name__}); using synthetic only")
        return []
    print(f"  [hf] {family}: pulled {len(out)} human-CoT rows")
    return out[:want]

# ───────────────────────────── build ─────────────────────────────
def build():
    rows = []
    for fam in FAMILY_NAME_MAP.values():
        seen, picked = set(), []

        # 1) optional HF human-CoT
        for prompt, answer, cot in hf_rows(fam, int(PER_CATEGORY * HF_FRACTION)):
            key = prompt.strip()
            if key in seen: continue
            seen.add(key)
            picked.append((prompt, answer, cot))

        # 2) fill the rest with synthetic golden CoT (dedup, bounded attempts)
        gen = SYNTH[fam]
        attempts = 0
        while len(picked) < PER_CATEGORY and attempts < PER_CATEGORY * 60:
            attempts += 1
            prompt, answer, cot = gen()
            key = prompt.strip()
            if key in seen: continue
            seen.add(key)
            picked.append((prompt, answer, cot))

        for prompt, answer, cot in picked[:PER_CATEGORY]:
            rows.append({
                "id": _rid(prompt, answer),
                "prompt": prompt,
                "answer": str(answer),
                "type": fam,
                "generated_cot": cot,
            })
        print(f"{fam:<20} -> {min(len(picked), PER_CATEGORY)} rows "
              f"({'short!' if len(picked) < PER_CATEGORY else 'ok'})")

    # dedup global id collisions
    by_id = {}
    for r in rows: by_id[r["id"]] = r
    rows = list(by_id.values())

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "type", "generated_cot"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-hf", action="store_true", help="blend in HuggingFace human CoT")
    ap.add_argument("--per-category", type=int, default=PER_CATEGORY)
    ap.add_argument("--out", default=OUT_CSV)
    a = ap.parse_args()
    USE_HF = a.use_hf or USE_HF
    PER_CATEGORY = a.per_category
    OUT_CSV = a.out
    build()
