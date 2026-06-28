"""
extract_good_hard.py — pull VERIFIED-correct (problem, solution) rows for the hard
categories out of andy_dataset/sft_train.jsonl.

Category labels + ground-truth answers come from problem_ids_matched.csv (its `type`
column carries the exact syn_datagen taxonomy). For each andy trace we:
  1. strip the appended boxed-instruction -> match the canonical train prompt,
  2. look up (type, answer); keep only the 5 target categories,
  3. extract the trace's final \boxed{...} (the "reasoning answer"),
  4. KEEP the row only if reasoning answer == ground-truth answer (official metric).
One row per problem (shortest correct trace) -> id, prompt, answer, generated_cot.

Usage:  python extract_good_hard.py            (-> good_hard_categories.csv)
"""
import json, os, re, csv, math, argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = {"cipher", "cryptarithm_deduce", "cryptarithm_guess",
          "equation_numeric_deduce", "equation_numeric_guess"}

norm = lambda s: re.sub(r"\s+", "", str(s)).lower()

def strip_suffix(u):
    i = u.find("\nPlease put your final answer")
    return (u[:i] if i != -1 else u).rstrip()

def extract_boxed(t):
    """Last \\boxed{...}, brace-balanced (matches the official extractor)."""
    tok = "\\boxed{"
    i = t.rfind(tok)
    if i == -1:
        return None
    d, j = 1, i + len(tok)
    while j < len(t) and d > 0:
        if t[j] == "{": d += 1
        elif t[j] == "}": d -= 1
        j += 1
    return t[i + len(tok):j - 1].strip() if d == 0 else None

def compare_answer(stored, predicted):
    """Official metric (syn_datagen/reasoning.py)."""
    if predicted is None:
        return False
    stored, predicted = str(stored).strip(), str(predicted).strip()
    if re.fullmatch(r"[01]+", stored):
        return predicted.lower() == stored.lower()
    try:
        return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored.lower()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=os.path.join(HERE, "sft_train.jsonl"))
    ap.add_argument("--labels", default=os.path.join(HERE, "..", "data_generation",
                                                      "generated_cot", "problem_ids_matched.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "good_hard_categories.csv"))
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="HARD cap on the FULL training sequence (prompt + cot), in tokens")
    ap.add_argument("--tokenizer", default="", help="HF id/path for EXACT token count (else conservative estimate)")
    a = ap.parse_args()

    # exact tokenizer if given (e.g. on Kaggle), else conservative char/token estimate.
    # LaTeX/symbol traces tokenize denser than prose -> use 3.0 (not 3.5); +300 for the
    # system prompt + chat-template wrapper + boxed answer that get added at train time.
    _tok = None
    if a.tokenizer:
        try:
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
            print(f"[tok] exact token counting via {a.tokenizer}")
        except Exception as e:
            print(f"[tok] could not load tokenizer ({e}); using estimate")
    def seq_tokens(prompt_text, trace):
        if _tok is not None:
            return len(_tok(prompt_text + "\n" + trace, add_special_tokens=False)["input_ids"]) + 300
        return int((len(prompt_text) + len(trace)) / 3.0) + 300

    import pandas as pd
    lab = pd.read_csv(a.labels)
    # map canonical prompt -> (id, answer, type), restricted to target categories
    meta = {}
    for _, r in lab.iterrows():
        ty = str(r.get("type", "")).strip()
        if ty in TARGET:
            meta[norm(r["prompt"])] = (str(r.get("id", "")), str(r["answer"]), ty)
    print(f"labels: {len(lab)} rows -> {len(meta)} unique target-category problems")

    # scan andy traces; per problem keep the shortest CORRECT trace
    best = {}            # key -> (len, id, prompt, answer, type, trace)
    n_seen = n_match = n_correct = 0
    with open(a.jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_seen += 1
            try:
                msgs = json.loads(line)["messages"]
            except Exception:
                continue
            user = msgs[0]["content"]; trace = msgs[-1]["content"]
            prompt_text = strip_suffix(user)
            key = norm(prompt_text)
            m = meta.get(key)
            if not m:
                continue
            n_match += 1
            pid, ans, ty = m
            if seq_tokens(prompt_text, trace) > a.max_tokens:   # FULL seq (prompt+cot) <= 8192
                continue
            box = extract_boxed(trace)
            if not compare_answer(ans, box):
                continue
            n_correct += 1
            cur = best.get(key)
            if cur is None or len(trace) < cur[0]:
                # canonical prompt comes from the labels file
                prompt = lab.iloc[0]["prompt"] if False else None
                best[key] = (len(trace), pid, key, ans, ty, trace)

    # recover canonical prompt text from the labels file
    canon = {norm(p): str(p) for p in lab["prompt"]}
    rows = []
    for key, (_, pid, _, ans, ty, trace) in best.items():
        rows.append({"id": pid or key[:16], "type": ty,
                     "prompt": canon.get(key, ""), "answer": ans, "generated_cot": trace.strip()})

    rows.sort(key=lambda r: r["type"])
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "generated_cot"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ["id", "prompt", "answer", "generated_cot"]})

    from collections import Counter
    print(f"\nscanned {n_seen} traces | matched target cats {n_match} | correct {n_correct}")
    print(f"unique good problems written: {len(rows)}")
    print("per-category:", dict(Counter(r["type"] for r in rows)))
    print("wrote ->", a.out)

if __name__ == "__main__":
    main()
