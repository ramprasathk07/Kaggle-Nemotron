"""
merge_cot_datasets_v2.py
========================
Merge two CoT corpora of the SAME Alice-Wonderland problems but DIFFERENT-model
reasoning, into one SFT corpus that keeps both correct CoTs per shared problem
(diverse-reasoning augmentation, not harmful duplication).

  d1 = problem_ids_matched.csv        (Tong CoT; proven 0.85 source)
       cols: id, prompt, answer, type, generated_cot
  d2 = final_output_10052025_sonnet.csv  (Sonnet CoT; clean <think>..</think>\\boxed{})
       cols: prompt, answer, cot, label

Output: merged_sft_tong_sonnet.csv  (id, prompt, answer, type, generated_cot)
  - d1 kept as-is (trusted).
  - d2 kept only where brace-balanced boxed(cot) == answer (drops label-noise).
  - exact (normalized_prompt, generated_cot) duplicates removed.
  - both CoTs survive for prompts shared by d1 and d2.
Schema matches the v21 SFT loader's column auto-detection.
"""
import os, re, hashlib
import pandas as pd

HERE = os.path.dirname(__file__)
SRC  = os.path.join(HERE, "..", "data_generation", "generated_cot")
D1   = os.path.join(SRC, "problem_ids_matched.csv")
D2   = os.path.join(SRC, "final_output_10052025_sonnet.csv")
OUT  = os.path.join(HERE, "merged_sft_tong_sonnet.csv")

_BX = "\\boxed{"

def extract_boxed(text):
    """Brace-balanced \\boxed{...} (handles nested/odd braces in answers)."""
    t = str(text)
    i = t.find(_BX)
    if i == -1:
        return None
    depth, j = 1, i + len(_BX)
    while j < len(t) and depth > 0:
        if t[j] == "{": depth += 1
        elif t[j] == "}": depth -= 1
        j += 1
    return t[i + len(_BX):j - 1].strip() if depth == 0 else None

def _normans(x):
    return re.sub(r"[\s,]", "", str(x)).lower()

def _normprompt(x):
    return re.sub(r"\s+", "", str(x)).lower()

def _rid(prompt, answer):
    return hashlib.sha1((str(prompt) + "||" + str(answer)).encode("utf-8")).hexdigest()[:12]

# ── load ──────────────────────────────────────────────────────────────────
d1 = pd.read_csv(D1)
d2 = pd.read_csv(D2)
print(f"d1 (Tong)   : {len(d1)} rows  cols={list(d1.columns)}")
print(f"d2 (Sonnet) : {len(d2)} rows  cols={list(d2.columns)}")

# ── d1: keep as-is, ensure schema ─────────────────────────────────────────
d1_out = pd.DataFrame({
    "id":            d1["id"].astype(str),
    "prompt":        d1["prompt"].astype(str),
    "answer":        d1["answer"].astype(str),
    "type":          d1["type"].astype(str),
    "generated_cot": d1["generated_cot"].astype(str),
    "source":        "tong",
})

# ── d2: keep only correct rows, remap schema ──────────────────────────────
d2 = d2.dropna(subset=["prompt", "answer", "cot"]).copy()
d2["_ok"] = [_normans(extract_boxed(c)) == _normans(a) for c, a in zip(d2["cot"], d2["answer"])]
n_bad = int((~d2["_ok"]).sum())
d2c = d2[d2["_ok"]].copy()
print(f"d2: dropped {n_bad} rows where boxed(cot) != answer; kept {len(d2c)}")

d2_out = pd.DataFrame({
    "id":            [_rid(p, a) for p, a in zip(d2c["prompt"], d2c["answer"])],
    "prompt":        d2c["prompt"].astype(str),
    "answer":        d2c["answer"].astype(str),
    "type":          d2c["label"].astype(str),     # d2 taxonomy; only used for stratified batching
    "generated_cot": d2c["cot"].astype(str),
    "source":        "sonnet",
})

# ── concat + dedup exact (normalized prompt, cot) ─────────────────────────
merged = pd.concat([d1_out, d2_out], ignore_index=True)
before = len(merged)
merged["_k"] = merged["prompt"].map(_normprompt) + "||" + merged["generated_cot"].map(_normprompt)
merged = merged.drop_duplicates(subset="_k").drop(columns="_k").reset_index(drop=True)
print(f"concat {before} -> dedup exact (prompt,cot) -> {len(merged)} rows")

# ── stats ─────────────────────────────────────────────────────────────────
uniq_prompts = merged["prompt"].map(_normprompt).nunique()
two_cot = (merged.groupby(merged["prompt"].map(_normprompt)).size() >= 2).sum()
print(f"unique prompts: {uniq_prompts}  | prompts with >=2 CoTs: {two_cot}")
print("source mix:", merged["source"].value_counts().to_dict())
print("type dist :", dict(merged["type"].value_counts().head(15)))

merged.drop(columns="source").to_csv(OUT, index=False, encoding="utf-8")
print(f"\nWrote {len(merged)} rows -> {os.path.abspath(OUT)}")
print("cols:", ["id", "prompt", "answer", "type", "generated_cot"])
