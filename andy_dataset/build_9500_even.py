"""
build_9500_even.py — 9,498-row SFT corpus, perfectly even (1,583 x 6), 100% REAL CoT.

No synthetic / no fabrication: the multi-CoT pool (andy + Sonnet + Tong, each problem
can carry several distinct teacher CoTs = diverse reasoning paths) holds >1,583 distinct
(problem, CoT) rows in EVERY category, so an even 1,583/cat is reachable from real data.

Selection (per category) maximizes PROBLEM coverage, then adds extra CoTs:
  * round 1: best CoT for every distinct problem (source priority andy > merged > tong,
             then shortest within source)
  * round 2..k: the next-best *distinct* CoT for problems that have more, round-robin,
             until the category hits TARGET (1,583).
Answer + prompt are always train.csv canonical. type via classify().

Out: sft_9500_even.csv  (columns: id, type, prompt, answer, source, n_cot_for_problem, generated_cot)
"""
import pandas as pd, re, os
from collections import defaultdict, Counter

H = "F:/Hackathons/Kaggle-Nemotron/"
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = 1583
COT_TOK_CAP = 6000           # est tokens; leaves room under max_model_len=8192 with prompt+wrapper

norm = lambda s: re.sub(r"\s+", "", str(s)).lower()
est_tokens = lambda s: int(len(s) / 3.5)

def classify(p):
    p = p.lower()
    if "bit manipulation" in p or "8-bit" in p: return "bit_manipulation"
    if "gravit" in p or "falling" in p: return "gravity"
    if "numeral system" in p: return "numeral"
    if "unit conversion" in p or "m becomes" in p: return "unit_conversion"
    if "encrypt" in p or "decrypt" in p or "cipher" in p: return "cipher"
    if "transformation rule" in p or "equation" in p: return "equation"
    return "other"

tr = pd.read_csv(H + "data_generation/src/train.csv")
trmap = {norm(p): (str(i), str(p), str(a)) for i, p, a in zip(tr["id"], tr["prompt"], tr["answer"])}

# source priority: lower rank = higher quality
SRC = [("andy",   0, H + "andy_dataset/andy_best_per_cat.csv"),
       ("merged", 1, H + "data_manipulation/merged_sft_tong_sonnet.csv"),
       ("tong",   2, H + "data_generation/generated_cot/problem_ids_matched.csv")]

# gather: per train-key -> list of (src_rank, src_name, len, cot), de-duped on cot text
bykey = defaultdict(list)
seen = set()
for name, rank, f in SRC:
    if not os.path.exists(f):
        print("skip missing:", f); continue
    d = pd.read_csv(f)
    col = "generated_cot" if "generated_cot" in d.columns else ("cot" if "cot" in d.columns else None)
    if col is None:
        print("no cot col:", name); continue
    kept = 0
    for pr, c in zip(d["prompt"], d[col]):
        k = norm(pr)
        if k not in trmap or not isinstance(c, str) or not c.strip(): continue
        c = c.strip()
        if est_tokens(c) > COT_TOK_CAP: continue
        ck = (k, norm(c)[:120])
        if ck in seen: continue
        seen.add(ck)
        bykey[k].append((rank, name, len(c), c)); kept += 1
    print(f"{name:8s}: +{kept} distinct (problem,cot) rows")

# order each problem's CoTs best-first (source rank, then shortest)
for k in bykey:
    bykey[k].sort(key=lambda t: (t[0], t[2]))

# bucket problems by category
cat_keys = defaultdict(list)
for k in bykey:
    cat_keys[classify(trmap[k][1])].append(k)

CATS = ["bit_manipulation", "cipher", "unit_conversion", "gravity", "numeral", "equation"]
recs = []
for cat in CATS:
    keys = cat_keys[cat]
    # order problems by how many CoTs they have ASC -> spend scarce single-CoT problems
    # first so we cover the widest set before reusing problems for 2nd/3rd CoTs.
    keys.sort(key=lambda k: len(bykey[k]))
    taken = 0; depth = 0; out = []
    while taken < TARGET:
        progressed = False
        for k in keys:
            if depth < len(bykey[k]):
                progressed = True
                rank, name, _, cot = bykey[k][depth]
                tid, prompt, answer = trmap[k]
                out.append({"id": tid, "type": cat, "prompt": prompt, "answer": answer,
                            "source": name, "n_cot_for_problem": len(bykey[k]),
                            "generated_cot": cot})
                taken += 1
                if taken >= TARGET: break
        depth += 1
        if not progressed: break          # pool exhausted (won't happen: all >1583)
    recs.extend(out)
    uniq = len({r["id"] for r in out})
    print(f"{cat:18s}: rows={len(out)}  unique_problems={uniq}  max_depth_used={depth}")

df = pd.DataFrame(recs, columns=["id","type","prompt","answer","source","n_cot_for_problem","generated_cot"])
out_path = os.path.join(HERE, "sft_9500_even.csv")
df.to_csv(out_path, index=False, encoding="utf-8")

print("\n" + "="*60)
print("TOTAL rows:", len(df))
print("per-category:", dict(df["type"].value_counts()))
print("by source:", dict(df["source"].value_counts()))
print("distinct problems covered:", df["id"].nunique())
print("dup (prompt,cot) exact:", int(df.duplicated(subset=["prompt","generated_cot"]).sum()))
print("cot char len min/med/max: %d/%d/%d" % (
    df["generated_cot"].str.len().min(),
    int(df["generated_cot"].str.len().median()),
    df["generated_cot"].str.len().max()))
print("wrote ->", out_path)
