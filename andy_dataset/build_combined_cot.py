"""
build_combined_cot.py — one CoT per train.csv problem, best source first, balanced.

Sources (priority high->low): andy_best_per_cat.csv (cleaned, multi-correct verified)
-> merged_sft_tong_sonnet.csv (Sonnet+Tong) -> problem_ids_matched.csv (Tong).
Answer is always train.csv's canonical answer. type via classify().

Outputs:
  sft_combined_all.csv   — every covered train problem (max real data, uneven)
  sft_combined_even.csv  — capped to the smallest category (perfectly even, all real)
"""
import pandas as pd, re, os
from collections import Counter, defaultdict

H = "F:/Hackathons/Kaggle-Nemotron/"
HERE = os.path.dirname(os.path.abspath(__file__))
norm = lambda s: re.sub(r"\s+", "", str(s)).lower()

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

# CoT sources, highest quality first
SRC = [
    ("andy",   H + "andy_dataset/andy_best_per_cat.csv"),
    ("merged", H + "data_manipulation/merged_sft_tong_sonnet.csv"),
    ("tong",   H + "data_generation/generated_cot/problem_ids_matched.csv"),
]
cot_map = {}            # train-key -> (cot, source)
for name, f in SRC:
    if not os.path.exists(f):
        print("skip missing:", f); continue
    d = pd.read_csv(f)
    col = "generated_cot" if "generated_cot" in d.columns else ("cot" if "cot" in d.columns else None)
    if col is None:
        print("no cot column in", name); continue
    added = 0
    for p, c in zip(d["prompt"], d[col]):
        k = norm(p)
        if k in trmap and k not in cot_map and isinstance(c, str) and c.strip():
            cot_map[k] = (c.strip(), name)
            added += 1
    print(f"{name:8s}: +{added} (cumulative {len(cot_map)})")

# assemble per train problem that has a CoT
recs = []
for k, (cot, src) in cot_map.items():
    tid, prompt, answer = trmap[k]
    recs.append({"id": tid, "type": classify(prompt), "prompt": prompt,
                 "answer": answer, "source": src, "generated_cot": cot})
df = pd.DataFrame(recs)
print("\ncombined covered:", len(df), "of 9500")
print("per-category:", dict(df["type"].value_counts()))
print("by source:", dict(df["source"].value_counts()))

cols = ["id", "type", "prompt", "answer", "source", "generated_cot"]
df[cols].to_csv(os.path.join(HERE, "sft_combined_all.csv"), index=False, encoding="utf-8")

# even = cap every category to the smallest category's count
floor = df["type"].value_counts().min()
even = (df.groupby("type", group_keys=False)
          .apply(lambda g: g.head(floor)).reset_index(drop=True))
even[cols].to_csv(os.path.join(HERE, "sft_combined_even.csv"), index=False, encoding="utf-8")
print(f"\neven floor = {floor}/cat -> {len(even)} rows  (sft_combined_even.csv)")
print("even per-category:", dict(even["type"].value_counts()))
print("wrote sft_combined_all.csv + sft_combined_even.csv")
