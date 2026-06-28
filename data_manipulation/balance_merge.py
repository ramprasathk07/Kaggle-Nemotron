"""
balance_merge.py — one per-category-BALANCED SFT corpus from all verified sources:
  * Tong CoT       (data_generation/generated_cot/problem_ids_matched.csv)  -> all categories
  * andy good-hard (andy_dataset/good_hard_categories.csv)                  -> verified hard cats
  * GPT-5.5 traces (gpt5_trace_gen/traces.csv, correct only)                -> hard cats top-up

Standardize -> (id, type, prompt, answer, generated_cot). CRLF->LF. Dedup exact (prompt,cot).
Per type: cap the rich categories at CAP_PER_TYPE (prefer GPT-5.5 > andy > Tong for the hard cats),
keep the thin categories whole. -> balanced_sft.csv (drop-in for v31 SFT loader).

Usage: python balance_merge.py [--cap 1200]
"""
import os, re, argparse
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
norm = lambda s: re.sub(r"\s+", "", str(s)).lower()
clean = lambda s: str(s).replace("\r\n", "\n").replace("\r", "\n").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=1200, help="max rows per category (rich cats capped)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data_manipulation", "balanced_sft.csv"))
    a = ap.parse_args()

    tong = pd.read_csv(os.path.join(ROOT, "data_generation", "generated_cot", "problem_ids_matched.csv"))
    p2type = {norm(p): str(t) for p, t in zip(tong["prompt"], tong["type"])}

    def classify(p):
        t = p2type.get(norm(p))
        if t: return t
        pl = p.lower()
        if "encrypt" in pl or "decrypt" in pl or "cipher" in pl: return "cipher"
        if "bit manipulation" in pl or "8-bit" in pl: return "bit_manipulation"
        if "numeral system" in pl: return "numeral"
        if "unit conversion" in pl: return "unit_conversion"
        if "gravit" in pl: return "gravity"
        if "transformation rule" in pl or "equation" in pl:
            body = pl.split("examples:")[-1][:140]
            return "cryptarithm_deduce" if not any(c.isdigit() for c in body) else "equation_numeric_deduce"
        return "other"

    rows = []   # (src_rank, src, type, id, prompt, answer, cot)
    def add(df, src, rank, type_from):
        for r in df.itertuples():
            cot = clean(getattr(r, "generated_cot", "") or "")
            ans = getattr(r, "answer", None)
            if not cot or len(cot) < 5 or ans is None or not str(ans).strip():
                continue
            ty = (str(getattr(r, type_from)) if type_from and hasattr(r, type_from)
                  else classify(str(r.prompt)))
            rows.append((rank, src, ty, str(getattr(r, "id", "")), str(r.prompt), str(ans), cot))

    add(tong, "tong", 2, "type")
    andy_p = os.path.join(ROOT, "andy_dataset", "good_hard_categories.csv")
    if os.path.exists(andy_p):
        add(pd.read_csv(andy_p), "andy", 1, None)            # classify via Tong type lookup
    gpt5_p = os.path.join(ROOT, "gpt5_trace_gen", "traces.csv")
    if os.path.exists(gpt5_p):
        g = pd.read_csv(gpt5_p)
        if "correct" in g.columns:
            g = g[g["correct"] == True]
        g = g.rename(columns={"category": "type"})
        add(g, "gpt5", 0, "type")                            # 0 = highest priority for hard cats

    df = pd.DataFrame(rows, columns=["rank", "src", "type", "id", "prompt", "answer", "generated_cot"])
    df = df.drop_duplicates(subset=["prompt", "generated_cot"])           # exact dupes only
    # per type: prefer low rank (gpt5>andy>tong), then cap
    df = df.sort_values("rank")
    capped = df.groupby("type", group_keys=False).head(a.cap).reset_index(drop=True)

    out_cols = ["id", "type", "prompt", "answer", "generated_cot"]
    capped[out_cols].to_csv(a.out, index=False, encoding="utf-8")

    print(f"pooled {len(df)} unique -> capped {len(capped)} @ {a.cap}/type")
    print("per-type:", dict(capped["type"].value_counts()))
    print("by-source:", dict(capped["src"].value_counts()))
    print("wrote ->", a.out)

if __name__ == "__main__":
    main()
