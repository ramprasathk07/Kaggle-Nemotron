"""
generate_problems.py — synthetic, SOLVER-VERIFIED problems for the hard equation
categories, in Alice's-Wonderland prompt format.

Categories (match the syn_datagen solver taxonomy):
  equation_numeric_deduce  : visible 2-digit operands, symbol operator, COMMON op
  equation_numeric_guess   : visible 2-digit operands, symbol operator, RARE op
  cryptarithm_deduce       : symbol-encoded operands, forward concatenation
  cryptarithm_guess        : symbol-encoded operands, reverse concatenation

Correctness: every problem is generated from ONE chosen rule, then the matching
syn_datagen reasoner is run; we KEEP ONLY rows whose solver answer equals the
constructed answer (solver-verified -> guaranteed correct and well-posed). The
answer column is the ground truth GPT-5 will be checked against.

Out: synthetic_hard.csv  (id, category, rule, prompt, answer, n_examples, verified)
"""
from __future__ import annotations
import os, sys, glob, shutil, random, re, argparse, csv

HERE = os.path.dirname(os.path.abspath(__file__))

def _setup_reasoners():
    """Make syn_datagen importable as the `reasoners` package."""
    hits = glob.glob("/kaggle/input/**/store_types.py", recursive=True)
    root = os.path.dirname(sorted(hits, key=len)[0]) if hits else None
    if root is None:
        for p in [os.path.join(HERE, "..", "syn_datagen"),
                  r"F:/Hackathons/Kaggle-Nemotron/syn_datagen"]:
            if os.path.exists(os.path.join(p, "store_types.py")):
                root = p; break
    if root is None:
        raise FileNotFoundError("syn_datagen/store_types.py not found")
    parent = HERE
    pkg = os.path.join(parent, "reasoners")
    if os.path.abspath(root) != os.path.abspath(pkg):
        shutil.copytree(root, pkg, dirs_exist_ok=True)
    if parent not in sys.path:
        sys.path.insert(0, parent)

_setup_reasoners()
from reasoners.store_types import Problem, Example
from reasoners.equation_numeric import reasoning_equation_numeric, _common_candidates, _rare_candidates
from reasoners.cryptarithm import reasoning_cryptarithm
from reasoners.reasoning import compare_answer

_BX = re.compile(r"\\boxed\{([^{}]*)\}")
def _boxed(t):
    if not t:
        return None
    m = _BX.findall(t)
    return m[-1].strip() if m else None

# symbol pool for operators (non-digit, non-space, used by the comp)
_OP_SYMS = list("/|\\{}*`>-#@&^?![]:%+<")
# distinct symbol alphabet for cryptarithm digit-encoding
_ENC_SYMS = list("!@#$%^&*()_+=[]{}|;:,.<>?/`~")

EQ_COMMON = ["addition", "absolute difference", "subtraction (a-b)",
             "reverse subtraction (b-a)", "multiplication",
             "concatenation", "reverse concatenation"]
EQ_RARE = ["digit add mod10", "digit sub mod10", "digit multiply",
           "cross multiply", "determinant", "abs determinant",
           "modulo (a mod b)", "integer division (a/b)"]

PROMPT_HEAD = ("In Alice's Wonderland, a secret set of transformation rules is "
               "applied to equations. Below are a few examples:")
PROMPT_TAIL = "Now, determine the result for: {q}"

def _candidate_result(op_name, a, b):
    sa, sb = str(a), str(b)
    for name, res in _common_candidates(a, b, sa, sb) + _rare_candidates(a, b, sa, sb):
        if name == op_name:
            return res
    return None

def gen_equation(op_pool):
    op = random.choice(op_pool)
    sym = random.choice(_OP_SYMS)
    def pair():
        return random.randint(10, 99), random.randint(10, 99)
    exs, seen = [], set()
    while len(exs) < 4:
        a, b = pair()
        if (a, b) in seen:
            continue
        res = _candidate_result(op, a, b)
        if res is None:
            return None
        seen.add((a, b))
        exs.append(Example(f"{a}{sym}{b}", str(res)))
    qa, qb = pair()
    q_res = _candidate_result(op, qa, qb)
    if q_res is None:
        return None
    q = f"{qa}{sym}{qb}"
    prob = Problem("syn", "equation_numeric_deduce", exs, q, "")
    solved = _boxed(reasoning_equation_numeric(prob))
    if solved is None or not compare_answer(str(q_res), solved):
        return None
    ex_lines = "\n".join(f"{e.input_value} = {e.output_value}" for e in exs)
    prompt = PROMPT_HEAD + "\n" + ex_lines + "\n" + PROMPT_TAIL.format(q=q)
    return {"rule": f"{op} [op '{sym}']", "prompt": prompt, "answer": str(q_res),
            "n_examples": len(exs), "verified": True}

def gen_cryptarithm(reverse):
    # symbol-encode: each of 10 digits -> a distinct symbol; operands 2 symbols each;
    # rule = concat (fwd) or reverse concat (rev) of the two operands.
    syms = random.sample(_ENC_SYMS, 10)
    dmap = {str(d): syms[d] for d in range(10)}
    op = random.choice(_OP_SYMS)
    def enc(n2):
        return dmap[n2[0]] + dmap[n2[1]]
    def mk():
        a = f"{random.randint(0,9)}{random.randint(0,9)}"
        b = f"{random.randint(0,9)}{random.randint(0,9)}"
        inp = enc(a) + op + enc(b)                      # 5 chars: A1 A2 op B1 B2
        out = (enc(b) + enc(a)) if reverse else (enc(a) + enc(b))
        return inp, out
    exs, seen = [], set()
    while len(exs) < 4:
        i, o = mk()
        if i in seen:
            continue
        seen.add(i); exs.append(Example(i, o))
    qi, q_ans = mk()
    prob = Problem("syn", "cryptarithm_guess", exs, qi, "")
    solved = _boxed(reasoning_cryptarithm(prob))
    if solved is None or solved != q_ans:
        return None
    ex_lines = "\n".join(f"{e.input_value} = {e.output_value}" for e in exs)
    prompt = PROMPT_HEAD + "\n" + ex_lines + "\n" + PROMPT_TAIL.format(q=qi)
    return {"rule": f"{'reverse ' if reverse else ''}concat [enc]",
            "prompt": prompt, "answer": q_ans, "n_examples": len(exs), "verified": True}

SPECS = {
    "equation_numeric_deduce": lambda: gen_equation(EQ_COMMON),
    "equation_numeric_guess":  lambda: gen_equation(EQ_RARE),
    "cryptarithm_deduce":      lambda: gen_cryptarithm(reverse=False),
    "cryptarithm_guess":       lambda: gen_cryptarithm(reverse=True),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "synthetic_hard.csv"))
    a = ap.parse_args()
    random.seed(a.seed)

    rows, idx = [], 0
    for cat, gen in SPECS.items():
        got = tries = 0
        seen_prompts = set()
        while got < a.per_cat and tries < a.per_cat * 30:
            tries += 1
            r = gen()
            if not r or r["prompt"] in seen_prompts:
                continue
            seen_prompts.add(r["prompt"])
            rows.append({"id": f"syn_{cat}_{idx:05d}", "category": cat, **r})
            idx += 1; got += 1
        print(f"{cat:26s} {got}/{a.per_cat}  (tries {tries})")

    cols = ["id", "category", "rule", "prompt", "answer", "n_examples", "verified"]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {a.out}")
    from collections import Counter
    print("per-category:", dict(Counter(r["category"] for r in rows)))

if __name__ == "__main__":
    main()
