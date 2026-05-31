#!/usr/bin/env python3
"""Merge 3 CoT CSVs → data/merged_cot_final.csv (prompt, cot, answer, label)."""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent

FILES = [
    ROOT / "data" / "golden_cot_1000.csv",
    ROOT / "final_output_10052025_sonnet.csv",
    # ROOT / "data" / "generated_cot" / "final_Nemotron_training_data.csv",
]

OUTPUT = ROOT / "data" / "merged_cot_subset.csv"
KEEP   = ["prompt", "cot", "answer", "label"]

# column aliases → canonical name
COL_MAP = {
    "generated_cot": "cot",
    "category":      "label",
    "question":      "prompt",
    "response":      "cot",
    "output":        "answer",
}


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COL_MAP)
    # drop any cols not in KEEP
    df = df[[c for c in KEEP if c in df.columns]]
    # drop rows missing any required field
    df = df.dropna(subset=["prompt", "cot", "answer"])
    df = df[df["prompt"].str.strip().astype(bool)]
    df = df[df["cot"].str.strip().astype(bool)]
    df = df[df["answer"].str.strip().astype(bool)]
    return df.reset_index(drop=True)


dfs = []
for path in FILES:
    raw = pd.read_csv(path)
    norm = normalise(raw)
    print(f"  {path.name:<48} {len(raw):>6} -> {len(norm)} rows")
    dfs.append(norm)

merged = pd.concat(dfs, ignore_index=True)

# ensure column order
for col in KEEP:
    if col not in merged.columns:
        merged[col] = ""
merged = merged[KEEP]

merged.to_csv(OUTPUT, index=False)
print(f"\nSaved {len(merged)} rows -> {OUTPUT}")
print(f"  label distribution:\n{merged['label'].value_counts().to_string()}")
