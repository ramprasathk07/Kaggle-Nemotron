"""
Build the golden SFT corpus by filtering + de-duping the generated_cot/ files
against src/train.csv ground truth.

Output structure:
    data/golden/
        sft_main.csv            # In-src prompts, max 2 diverse CoTs each
        sft_ood_sonnet.csv      # Out-of-distribution Sonnet prompts
        grpo_prompts.csv        # Prompts only — for GRPO stage
        stats.json              # Per-stage counts + per-source contribution

Run:  python3 build_golden_corpus.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pandas as pd

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH  = os.path.join(DATA_ROOT, "src", "train.csv")
GC_DIR    = os.path.join(DATA_ROOT, "generated_cot")
OUT_DIR   = os.path.join(DATA_ROOT, "golden")
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Boxed extraction + correctness  (mirrors backend scoring)
# -----------------------------------------------------------------------------
_BOXED_OPEN = r"\boxed{"


def extract_boxed(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    pos = text.rfind(_BOXED_OPEN)
    if pos == -1:
        return None
    start = pos + len(_BOXED_OPEN)
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if   text[i] == "{": depth += 1
        elif text[i] == "}": depth -= 1
        i += 1
    return text[start : i - 1].strip() if depth == 0 else None


def normalize(s: str | None) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    for tok in (r"\\,", r"\,", r"\\!", r"\!", r"\\ ", r"\ ", r"\text{", "}"):
        s = s.replace(tok, "")
    try:
        v = float(s.replace(",", ""))
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.8g}"
    except (ValueError, OverflowError):
        return s.lower().replace(" ", "")


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    p, g = normalize(pred), normalize(gold)
    if p == g:
        return True
    try:
        return abs(float(p.replace(",", "")) - float(g.replace(",", ""))) <= 1e-2
    except (ValueError, OverflowError):
        return False


# -----------------------------------------------------------------------------
# Quality scoring
# -----------------------------------------------------------------------------
SOURCE_BONUS = {
    "final_output_10052025_sonnet.csv": 2.0,
    "traincsv_sonnet.csv":              2.0,
    "problem_ids_matched.csv":          1.5,
    "merged_cot_subset.csv":            1.2,
    "train_split_with_cot.csv":         1.0,
    "train_v15_fixed.csv":              0.3,
    "final_Nemotron_training_data.csv": 0.5,
    "gemini-3.1-flash-lite-preview_enhanced_sft_dataset.csv": 0.7,
    "merged_cot_final.csv":             0.4,
}

LEAKY_PATTERNS = (
    "I am a reasoning model",
    "Kaggle competition",
    "I have 100% accuracy",
    "100% accuracy by following these rules",
)


def quality_score(cot: str, source_file: str, rewrapped: bool) -> float:
    s = SOURCE_BONUS.get(source_file, 0.0)
    n = len(cot)
    if 400 <= n <= 4000:
        s += 1.0
    elif 200 <= n < 400 or 4000 < n <= 8000:
        s += 0.5

    steps = len(re.findall(r"(?:Step\s*\d+|^\d+\.|^S\d+:|^-\s)", cot, re.MULTILINE))
    if steps >= 3:
        s += 1.0
    eqs = len(re.findall(r"[=≈<>≤≥]", cot))
    if eqs >= 5:
        s += 1.0
    if "<think>" in cot and "</think>" in cot:
        s += 0.5
    if any(p in cot for p in LEAKY_PATTERNS):
        s -= 1.0
    if rewrapped:
        s += 0.5
    return s


def fingerprint(cot: str) -> str:
    """Cheap structural fingerprint for clustering similar CoTs."""
    text = re.sub(r"\d+", "#", cot)
    text = re.sub(r"\s+", " ", text).strip()[:200]
    return hashlib.md5(text.encode()).hexdigest()[:8]


def length_bucket(n: int) -> str:
    for b in (500, 1000, 2000, 4000, 8000):
        if n < b:
            return f"L<{b}"
    return "L>=8000"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
SOURCES: list[tuple[str, str]] = [
    # (filename, cot column name)
    ("problem_ids_matched.csv",                                   "generated_cot"),
    ("train_split_with_cot.csv",                                  "generated_cot"),
    ("train_v15_fixed.csv",                                       "reasoning"),
    ("final_output_10052025_sonnet.csv",                          "cot"),
    ("traincsv_sonnet.csv",                                       "cot"),
    ("merged_cot_subset.csv",                                     "cot"),
    ("final_Nemotron_training_data.csv",                          "generated_cot"),
    ("gemini-3.1-flash-lite-preview_enhanced_sft_dataset.csv",    "generated_cot"),
]


@dataclass
class CoTRecord:
    prompt: str
    cot: str
    answer: str
    source: str
    score: float
    rewrapped: bool
    cot_type: str  # category if known
    fingerprint: str = field(default="")


def load_src_ground_truth() -> dict[str, str]:
    src = pd.read_csv(SRC_PATH).dropna(subset=["prompt", "answer"])
    src["prompt_h"] = src["prompt"].astype(str).map(
        lambda s: hashlib.md5(s.encode()).hexdigest()
    )
    return dict(zip(src["prompt_h"], src["answer"].astype(str)))


def get_type_map() -> dict[str, str]:
    """Use problem_ids_matched.csv to tag known categories."""
    pim = pd.read_csv(os.path.join(GC_DIR, "problem_ids_matched.csv"))
    pim["prompt_h"] = pim["prompt"].astype(str).map(
        lambda s: hashlib.md5(s.encode()).hexdigest()
    )
    base = dict(zip(pim["prompt_h"], pim["type"].astype(str)))
    # Augment with Sonnet labels for the new puzzle types
    son = pd.read_csv(os.path.join(GC_DIR, "final_output_10052025_sonnet.csv"))
    son["prompt_h"] = son["prompt"].astype(str).map(
        lambda s: hashlib.md5(s.encode()).hexdigest()
    )
    if "label" in son.columns:
        for h, lbl in zip(son["prompt_h"], son["label"].astype(str)):
            base.setdefault(h, lbl)
    return base


# -----------------------------------------------------------------------------
# Main filter passes
# -----------------------------------------------------------------------------
def build():
    src_gold     = load_src_ground_truth()
    type_map     = get_type_map()
    in_src_pool: dict[str, list[CoTRecord]] = defaultdict(list)
    ood_pool: list[CoTRecord]               = []
    stats = {"sources": {}, "totals": {}}

    for fname, cot_col in SOURCES:
        path = os.path.join(GC_DIR, fname)
        if not os.path.exists(path):
            print(f"SKIP (missing): {fname}")
            continue
        df = pd.read_csv(path)
        if "prompt" not in df.columns or cot_col not in df.columns:
            print(f"SKIP (schema): {fname}")
            continue
        df["prompt_h"] = df["prompt"].astype(str).map(
            lambda s: hashlib.md5(s.encode()).hexdigest()
        )

        n_total = len(df)
        n_kept_in_src = n_kept_ood = n_rewrap = 0
        for _, row in df.iterrows():
            prompt = str(row["prompt"])
            cot    = row.get(cot_col)
            if not isinstance(cot, str) or len(cot.strip()) < 20:
                continue
            row_answer = str(row.get("answer", "")).strip()
            h = row["prompt_h"]

            gold = src_gold.get(h)
            in_src = gold is not None
            boxed = extract_boxed(cot)

            if in_src:
                if boxed is not None and is_correct(boxed, gold):
                    rec_cot = cot
                    rewrapped = False
                elif row_answer and is_correct(row_answer, gold) and len(cot.strip()) > 200:
                    body = cot
                    if "<think>" not in body:
                        body = f"<think>\n{body.strip()}\n</think>"
                    body = re.sub(r"\\boxed\{[^}]*\}", "", body).rstrip()
                    rec_cot = f"{body}\n\\boxed{{{gold}}}"
                    rewrapped = True
                    n_rewrap += 1
                else:
                    continue
                score = quality_score(rec_cot, fname, rewrapped)
                cot_type = type_map.get(h, "unknown")
                in_src_pool[h].append(CoTRecord(
                    prompt=prompt, cot=rec_cot, answer=gold,
                    source=fname, score=score, rewrapped=rewrapped,
                    cot_type=cot_type, fingerprint=fingerprint(rec_cot),
                ))
                n_kept_in_src += 1
            else:
                # OOD: keep if Sonnet (high quality) and boxed verified vs its own answer
                if "sonnet" not in fname:
                    continue
                if boxed is None or not row_answer:
                    continue
                if not is_correct(boxed, row_answer):
                    continue
                score = quality_score(cot, fname, False)
                ood_pool.append(CoTRecord(
                    prompt=prompt, cot=cot, answer=row_answer,
                    source=fname, score=score, rewrapped=False,
                    cot_type=str(row.get("label", "unknown")),
                    fingerprint=fingerprint(cot),
                ))
                n_kept_ood += 1

        stats["sources"][fname] = {
            "rows":        n_total,
            "kept_in_src": n_kept_in_src,
            "kept_ood":    n_kept_ood,
            "rewrapped":   n_rewrap,
        }
        print(f"  {fname:60s} in_src={n_kept_in_src:5d}  ood={n_kept_ood:5d}  rewrap={n_rewrap}")

    # -----------------------------------------------------------------------
    # Diversity selection per src prompt: keep top-2 from different clusters
    # -----------------------------------------------------------------------
    selected: list[CoTRecord] = []
    diversity_counts = Counter()
    for h, recs in in_src_pool.items():
        recs.sort(key=lambda r: r.score, reverse=True)
        chosen, seen_clusters = [], set()
        for r in recs:
            key = (r.source, length_bucket(len(r.cot)), r.fingerprint[:4])
            if key in seen_clusters:
                continue
            chosen.append(r)
            seen_clusters.add(key)
            if len(chosen) == 2:
                break
        selected.extend(chosen)
        diversity_counts[len(chosen)] += 1

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    sft_main_path = os.path.join(OUT_DIR, "sft_main.csv")
    pd.DataFrame([{
        "prompt": r.prompt,
        "answer": r.answer,
        "cot":    r.cot,
        "source": r.source,
        "type":   r.cot_type,
        "score":  r.score,
    } for r in selected]).to_csv(sft_main_path, index=False)
    print(f"\nWrote {len(selected)} rows → {sft_main_path}")

    # OOD: dedup + keep top-N by score
    seen = set()
    ood_pool.sort(key=lambda r: r.score, reverse=True)
    ood_unique = []
    for r in ood_pool:
        ph = hashlib.md5(r.prompt.encode()).hexdigest()
        if ph in seen:
            continue
        seen.add(ph)
        ood_unique.append(r)
    ood_path = os.path.join(OUT_DIR, "sft_ood_sonnet.csv")
    pd.DataFrame([{
        "prompt": r.prompt,
        "answer": r.answer,
        "cot":    r.cot,
        "source": r.source,
        "type":   r.cot_type,
        "score":  r.score,
    } for r in ood_unique]).to_csv(ood_path, index=False)
    print(f"Wrote {len(ood_unique)} rows → {ood_path}")

    # GRPO prompt list: every src prompt that has ≥1 selected CoT
    # (we re-generate during RL; answer is the supervision signal)
    grpo_rows = []
    for h, recs in in_src_pool.items():
        if recs:
            grpo_rows.append({"prompt": recs[0].prompt, "answer": recs[0].answer,
                              "type":   recs[0].cot_type})
    grpo_df = pd.DataFrame(grpo_rows)
    grpo_path = os.path.join(OUT_DIR, "grpo_prompts.csv")
    grpo_df.to_csv(grpo_path, index=False)
    print(f"Wrote {len(grpo_df)} prompts → {grpo_path}")

    # -----------------------------------------------------------------------
    # Stats summary
    # -----------------------------------------------------------------------
    type_dist = Counter(r.cot_type for r in selected)
    src_dist  = Counter(r.source   for r in selected)
    stats["totals"] = {
        "src_prompts":           len(src_gold),
        "src_prompts_covered":   len(in_src_pool),
        "selected_sft_rows":     len(selected),
        "ood_sft_rows":          len(ood_unique),
        "grpo_prompts":          len(grpo_df),
        "selected_per_prompt":   dict(diversity_counts),
        "selected_type_dist":    dict(type_dist),
        "selected_source_dist":  dict(src_dist),
    }
    stats_path = os.path.join(OUT_DIR, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats written → {stats_path}")
    print(json.dumps(stats["totals"], indent=2))


if __name__ == "__main__":
    build()
