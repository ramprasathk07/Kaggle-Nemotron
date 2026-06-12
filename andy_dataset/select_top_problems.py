"""
select_top_problems.py — pick the highest-confidence andy279 traces per category,
ALIGNED to the original train.csv, and save a clean SFT CSV (one trace per problem).

Fixes vs v1:
  * andy user content = train_prompt + PROMPT_SUFFIX -> it did NOT equal train.csv.
    We strip the suffix and MATCH train.csv (100% match), then store the CANONICAL
    train.csv prompt + answer + id (not andy's suffixed version).
  * correctness is checked against train.csv's ground-truth answer (not majority vote).
  * generated_cot = the reasoning inside <think>...</think> (verified non-empty).
  * output path defaults to an ABSOLUTE path so the file actually persists.

"High correct count" = how many of a problem's traces hit the train.csv answer.
We keep the problems with the most correct traces (most reliably solved -> cleanest),
emitting ONE trace each: the shortest correct, well-formed, within-budget one.

Output CSV (drop-in for the v21/v22 SFT loader):
  id, type, prompt, answer, correct_count, n_traces, est_tokens, generated_cot

Usage:
  python3 select_top_problems.py --n 500 --per-cat \
      --train-csv ../data_generation/src/train.csv \
      --out top_per_category.csv
"""
import json, os, re, argparse
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
_BX = "\\boxed{"

def est_tokens(s): return int(len(s) / 3.5)

def classify(p):
    p = p.lower()
    if "bit manipulation" in p or "8-bit" in p: return "bit_manipulation"
    if "gravit" in p or "falling" in p:         return "gravity"
    if "numeral system" in p:                   return "numeral"
    if "unit conversion" in p or "m becomes" in p: return "unit_conversion"
    if "encrypt" in p or "decrypt" in p or "cipher" in p: return "cipher"
    if "transformation rule" in p or "equation" in p:     return "equation"
    return "other"

def strip_suffix(u):
    """Remove the appended boxed-instruction so the prompt matches train.csv."""
    i = u.find("\nPlease put your final answer")
    return (u[:i] if i != -1 else u).rstrip()

def extract_boxed(t):
    i = t.rfind(_BX)                       # LAST box (official-extractor behaviour)
    if i == -1: return None
    d, j = 1, i + len(_BX)
    while j < len(t) and d > 0:
        if t[j] == "{": d += 1
        elif t[j] == "}": d -= 1
        j += 1
    return t[i + len(_BX):j - 1].strip() if d == 0 else None

def well_formed(a):
    # box may sit inside OR after </think> (categories differ); the SFT loader
    # re-wraps the target anyway, so accept any trace with a think block + a box.
    return "</think>" in a and _BX in a

def reasoning_body(a):
    """The reasoning text, boxes removed (the v21 'generated_cot').

    Trace styles differ: some put reasoning INSIDE <think>..</think> (box after),
    others use an EMPTY <think></think> with the reasoning AFTER it (box at end).
    So just drop the think tags + all \\boxed{...} and keep whatever reasoning remains.
    """
    body = a.replace("<think>", "").replace("</think>", "")
    out, i = [], 0
    while i < len(body):
        k = body.find(_BX, i)
        if k == -1: out.append(body[i:]); break
        out.append(body[i:k]); j = k + len(_BX); d = 1
        while j < len(body) and d > 0:
            if body[j] == "{": d += 1
            elif body[j] == "}": d -= 1
            j += 1
        i = j
    return re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()

def clean_cot(s):
    """Strip teacher LaTeX so the reasoning is plain text (shorter, no LaTeX echo at eval)."""
    s = s.replace("\\(", "").replace("\\)", "").replace("\\[", "\n").replace("\\]", "\n")
    s = s.replace("$$", "").replace("$", "")
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = (s.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
           .replace("\\approx", "~=").replace("\\leq", "<=").replace("\\geq", ">=")
           .replace("\\neq", "!=").replace("\\pm", "+/-").replace("\\to", "->")
           .replace("\\rightarrow", "->").replace("\\Rightarrow", "=>"))
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\s?", "", s)   # drop any leftover \command
    s = s.replace("\\", "")                # stray backslashes
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    # drop the dangling box-leftover tail (orphan '}', 'answer:' from \boxed{x}})
    s = re.sub(r"(?im)\n[^\n]{0,40}?(?:answer|result|output|is)\s*[:=]?\s*$", "", s)
    return re.sub(r"[\s}:=.\-]+$", "", s).strip()

def ans_match(pred, exp):
    if pred is None: return False
    if pred.strip().lower() == str(exp).strip().lower(): return True
    try: return abs(float(pred) - float(exp)) <= 1e-2 * max(1.0, abs(float(exp)))
    except Exception: return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(HERE, "sft_train.jsonl"))
    ap.add_argument("--train-csv", default=os.path.join(HERE, "..", "data_generation", "src", "train.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "top_per_category.csv"))
    ap.add_argument("--n", type=int, default=500, help="cap (per-category if --per-cat, else global)")
    ap.add_argument("--per-cat", action="store_true", help="--n is the cap PER category")
    ap.add_argument("--max-tokens", type=int, default=8192, help="per-trace budget cap")
    ap.add_argument("--min-correct", type=int, default=2, help="require >= this many correct traces")
    ap.add_argument("--raw-cot", action="store_true", help="keep teacher LaTeX (default: clean to plain text)")
    a = ap.parse_args()
    if not os.path.isabs(a.out): a.out = os.path.join(HERE, a.out)   # always persist

    import pandas as pd
    tr = pd.read_csv(a.train_csv)
    norm = lambda s: re.sub(r"\s+", "", str(s)).lower()
    trmap = {norm(p): (str(i), str(p), str(ans))
             for i, p, ans in zip(tr["id"], tr["prompt"], tr["answer"])}
    print(f"train.csv: {len(tr)} rows ({len(trmap)} unique prompts)")

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    print(f"andy traces: {len(rows)}")

    # group andy traces by the matched train.csv problem
    groups = defaultdict(lambda: {"meta": None, "cat": None, "traces": []})
    unmatched = 0
    for r in rows:
        u = r["messages"][0]["content"]; asst = r["messages"][-1]["content"]
        key = norm(strip_suffix(u))
        meta = trmap.get(key)
        if meta is None: unmatched += 1; continue        # not a train.csv problem -> drop
        g = groups[key]
        g["meta"] = meta                                  # (id, canonical_prompt, answer)
        g["cat"] = g["cat"] or classify(meta[1])
        g["traces"].append(asst)
    print(f"matched to train.csv: {len(rows)-unmatched}/{len(rows)} traces, "
          f"{len(groups)} unique problems (unmatched dropped: {unmatched})")

    picked = []
    for key, g in groups.items():
        tid, prompt, answer = g["meta"]
        cands = [t for t in g["traces"]
                 if ans_match(extract_boxed(t), answer)
                 and well_formed(t) and est_tokens(t) <= a.max_tokens
                 and reasoning_body(t)]                    # non-empty CoT
        if len(cands) < a.min_correct: continue
        best = min(cands, key=len)
        cot = reasoning_body(best)
        if not a.raw_cot:
            cot = clean_cot(cot)
        if not cot.strip():
            continue
        picked.append({
            "id": tid, "type": g["cat"], "prompt": prompt, "answer": answer,
            "correct_count": len(cands), "n_traces": len(g["traces"]),
            "est_tokens": est_tokens(best), "generated_cot": cot,
        })

    picked.sort(key=lambda r: (-r["correct_count"], r["est_tokens"]))
    if a.per_cat:
        bycat, sel = defaultdict(int), []
        for r in picked:
            if bycat[r["type"]] < a.n:
                sel.append(r); bycat[r["type"]] += 1
    else:
        sel = picked[:a.n]

    df = pd.DataFrame(sel, columns=["id", "type", "prompt", "answer",
                                    "correct_count", "n_traces", "est_tokens", "generated_cot"])
    df.to_csv(a.out, index=False, encoding="utf-8")

    print(f"\neligible problems (>= {a.min_correct} correct): {len(picked)}")
    print(f"selected: {len(df)}  (mode={'per-category cap' if a.per_cat else 'global top-N'})")
    print("per-category:", dict(Counter(df["type"])))
    if len(df):
        bad = int((df["generated_cot"].astype(str).str.strip() == "").sum())
        print(f"generated_cot empty: {bad} | char len min/med/max="
              f"{df['generated_cot'].str.len().min()}/{int(df['generated_cot'].str.len().median())}/{df['generated_cot'].str.len().max()}")
        print(f"correct_count min/med/max={df['correct_count'].min()}/{int(df['correct_count'].median())}/{df['correct_count'].max()}")
        print("prompt[0] == train.csv:", df['prompt'].iloc[0] == trmap[norm(df['prompt'].iloc[0])][1])
    print(f"wrote -> {a.out}")

if __name__ == "__main__":
    main()
