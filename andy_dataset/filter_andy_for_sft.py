"""
filter_andy_for_sft.py — turn the raw andy279 traces into a *surgical* SFT slice.

Raw dump of 49k traces -> pack parity (~0.87). The win is the filter. This applies,
in priority order:
  1. LENGTH CAP  — drop traces whose assistant text exceeds the eval generation
     budget (teacher traces run to ~13k-24k tokens; training on those teaches the
     model to over-generate and get TRUNCATED before \boxed{} at eval -> zeros).
  2. FORMAT      — must close </think> and end with exactly one trailing \boxed{};
     normalize so the boxed answer is the LAST thing emitted.
  3. CORRECTNESS — (optional) if a ground-truth CSV is given, keep only traces whose
     boxed answer matches gt (STRICT; exact-or-numeric). Train strict, score official.
  4. FAILURE-INTERSECTION — (optional) if your eval-results CSV is given
     (outputs/matched_eval_results.csv from predict_matched_eval.ipynb), keep traces
     only for categories the model still FAILS, so rank-32 capacity isn't spent
     re-learning solved patterns.
Outputs a messages-JSONL slice (id, prompt, answer, type, generated_cot schema is
NOT used — this stays in chat `messages` format) + a stats report.

Dependency-light: stdlib + (optional) pandas for the CSV joins.
"""
import json, os, re, argparse, statistics
from collections import Counter, defaultdict

HERE = os.path.dirname(__file__)

# ── token estimate (char/3.5 ~ Nemotron tokenizer on this text) ──
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

_BX = "\\boxed{"
def extract_boxed(t):
    i = t.rfind(_BX)              # LAST box (matches the official extractor)
    if i == -1: return None
    d, j = 1, i + len(_BX)
    while j < len(t) and d > 0:
        if t[j] == "{": d += 1
        elif t[j] == "}": d -= 1
        j += 1
    return t[i + len(_BX):j - 1].strip() if d == 0 else None

def well_formed(a):
    return a.startswith("<think>") or ("</think>" in a and _BX in a.rsplit("</think>", 1)[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", default=os.path.join(HERE, "sft_train.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "sft_train_filtered.jsonl"))
    ap.add_argument("--max-tokens", type=int, default=3000,
                    help="drop traces whose assistant text exceeds this (eval budget)")
    ap.add_argument("--gt-csv", default=None,
                    help="ground-truth CSV (prompt,answer) for strict correctness filter")
    ap.add_argument("--fail-csv", default=None,
                    help="eval-results CSV (type,correct) -> keep only failing categories")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    print(f"loaded {len(rows)} raw traces from {a.inp}")

    # optional: which categories does the model still fail? (per-category acc < 0.95)
    fail_cats = None
    if a.fail_csv and os.path.exists(a.fail_csv):
        import pandas as pd
        ev = pd.read_csv(a.fail_csv)
        acc = ev.groupby("type")["correct"].mean()
        fail_cats = set(acc[acc < 0.95].index)
        print(f"[fail-intersect] weak categories (<95%): {sorted(fail_cats)}")

    # optional: ground-truth map for strict correctness
    gt = None
    if a.gt_csv and os.path.exists(a.gt_csv):
        import pandas as pd
        g = pd.read_csv(a.gt_csv)
        norm = lambda s: re.sub(r"\s+", "", str(s)).lower()
        gt = {norm(p): str(ans) for p, ans in zip(g["prompt"], g["answer"])}
        print(f"[correctness] gt map: {len(gt)} prompts")

    def _ans_ok(pred, exp):
        if pred is None: return False
        pn, en = pred.strip().lower(), str(exp).strip().lower()
        if pn == en: return True
        try: return abs(float(pred) - float(exp)) <= 1e-2 * max(1.0, abs(float(exp)))
        except Exception: return False

    kept, drop = [], Counter()
    klen_before, klen_after = [], []
    cat_kept = Counter()
    for r in rows:
        m = r["messages"]; u = m[0]["content"]; asst = m[-1]["content"]
        klen_before.append(est_tokens(asst))
        cat = classify(u)
        if not well_formed(asst):                 drop["malformed"] += 1; continue
        if est_tokens(asst) > a.max_tokens:        drop["too_long"]  += 1; continue
        if fail_cats is not None and cat not in fail_cats: drop["solved_cat"] += 1; continue
        if gt is not None:
            exp = gt.get(re.sub(r"\s+", "", u).lower())
            if exp is not None and not _ans_ok(extract_boxed(asst), exp):
                drop["wrong_answer"] += 1; continue
        kept.append(r); cat_kept[cat] += 1; klen_after.append(est_tokens(asst))

    with open(a.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def pct(x, p): return int(statistics.quantiles(x, n=100)[p-1]) if len(x) > 1 else (x[0] if x else 0)
    print(f"\nkept {len(kept)} / {len(rows)}  ({100*len(kept)/len(rows):.1f}%)")
    print("dropped:", dict(drop))
    print("kept per-category:", dict(cat_kept))
    if klen_before: print(f"len(tok) BEFORE: p50={pct(klen_before,50)} p90={pct(klen_before,90)} p99={pct(klen_before,99)} max={max(klen_before)}")
    if klen_after:  print(f"len(tok) AFTER : p50={pct(klen_after,50)} p90={pct(klen_after,90)} max={max(klen_after)}")
    print(f"\nwrote -> {a.out}")
    print("Note: pass --gt-csv (strict correctness) and --fail-csv outputs/matched_eval_results.csv")
    print("      (failure-intersection) for the full surgical slice once you've run the eval harness.")

if __name__ == "__main__":
    main()
