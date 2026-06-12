# Plan to 92% — using `andy_dataset/` surgically

**Context.** Stuck at 0.85. Field walls at ~0.87, current #1 ≈ 0.8946 (public LB).
`andy_dataset/` = andy279 teacher traces + two tools + the strategy session. This is
the analyzed, grounded plan to push toward the 88–92 band and win.

---

## Verdict: will training on a well-filtered andy set help?

- **Raw dump → NO** (pack parity ~0.87). It's public; everyone has it; the field walls
  at 0.87 *with* it. Public data is the floor, never the separation.
- **Surgically filtered → YES** (+1–3pp), and one filter dominates.

### Measured finding — the trace-length trap (the #1 filter)
Raw andy assistant traces (120B/GPT teacher) are HUGE:
`p50≈1,176 tok · p90≈4,608 · p99≈13,070 · max≈24,328 tok`. Eval context is 8192.
Train on >~3k-tok traces → model over-generates → **truncated before `\boxed{}` → 0**.
`filter_andy_for_sft.py --max-tokens 3000` drops **9,139 (18.5%)**, keeps **40,151**
(`sft_train_filtered.jsonl`, after-cap p90=2,276 / max=3,000). Raw dump can LOWER score.

### The four filters (priority order)
1. **Length-cap** ≤ ~3k tok (done — proven 18.5% poison removed).
2. **Strict correctness**, but **score with the OFFICIAL (buggy) metric**. For
   `bit_manipulation`, emit the exact binary string **with leading zeros** → hits the
   exact-match branch, dodges the numeric-tolerance bug.
3. **Failure-intersection** — pull andy traces only for categories/problems the 0.85
   model still FAILS (rank-32 capacity is tiny; don't relearn solved patterns). Needs
   the eval output.
4. **Dedup** vs the merged Tong+Sonnet corpus + 10–20% replay of working data.

---

## The folder's tools (worth more than the raw data)
- `rule_trace_generator.py` — rule **induction → retrieval**: enumerate candidate rules,
  test vs in-prompt examples (executed, not hallucinated), eliminate, apply, **verify**,
  box. Mechanically truthful. `est_tokens` (length filter) + `answer_span` (5–10×
  answer-token loss weight) built in. **The flat-85 reliability fix.** Validate it
  reconstructs train.csv answers ~100% per category before scaling synthetic data.
- `lora_merge.py` — TIES/DARE/linear soup → rank-32 via SVD, numpy-only, no base model,
  no torch (runs on the 4090). **Free points**: soup SFT/RAFT/GRPO/Tong adapters.
- `filter_andy_for_sft.py` — the surgical slicer (length + format + `--gt-csv` strict
  correctness + `--fail-csv` failure-intersection).

---

## Execution plan

### Step 0 — Measure (gates everything; do FIRST, no training)
- Run the fixed `outputs/predict_matched_eval.ipynb` (offline, transformers-only) on a
  held-out slice (`LIMIT=300` for a fast read, then full) → per-category accuracy +
  per-row pass/fail → `outputs/matched_eval_results.csv`.
- Build the 4-bucket failure ledger: **truncation / format-metric / wrong-rule(sub-cat) /
  execution-slip**. The bucket mix IS the strategy.
- Template/metric sanity: diff the training chat template + the official metric
  (replicate the buggy binary-string branch) so local eval tracks the LB.

### Step 1 — Surgical andy slice
```
python3 andy_dataset/filter_andy_for_sft.py \
  --max-tokens 3000 \
  --gt-csv   data_generation/generated_cot/problem_ids_matched.csv \
  --fail-csv outputs/matched_eval_results.csv \
  --out      andy_dataset/sft_slice.jsonl
```
→ andy traces for failing categories only, length-capped, strict-correct, dedupable.

### Step 2 — SFT (0.85 recipe + answer-token loss upweighting)
- Train `merged_sft_tong_sonnet.csv` + `sft_slice.jsonl` on the v21/v22 0.85 recipe
  (broad LoRA 888M, LR 2e-4 linear, 1 ep, max_grad_norm=1e9), 5–10× loss weight on the
  boxed-answer tokens (`answer_span`). Eval-gate per category vs 0.85.

### Step 3 — Reliability (the real lever to 90+)
- For weak sub-categories, generate `rule_trace_generator.py` traces (retrieval +
  in-trace verify). SFT a short focused pass. This converts near-misses → hits
  uniformly — the flat-85 cure.

### Step 4 — Soup (free)
- `lora_merge.py --mode ties --rank 32` over {best-SFT, RAFT, GRPO, Tong, category-experts}.
  Eval-gate each soup locally; merges are hypotheses, not upgrades.

### Step 5 — Select & submit
- Rank candidates on a large held-out set (LB noise ±2–3 problems at ~250). Ship 2:
  best local-eval adapter + most-diversified soup (decorrelated bets for the private set).
  Verify `submission.zip` = two files at root, rank ≤ 32.

---

## Honest calibration
- Template/metric fixes + filtered-andy surgery: **0.85 → 0.87–0.89**.
- Past 0.89: needs Step 3 cracking specific weak sub-categories + verification traces.
- **0.92 = all of it landing at once** — play for it, but select finals for the 88–90 band.

## Don't waste time on
- Raw-dumping andy or any generic public reasoning data (GSM8K/MATH/LogiQA) — off-distribution, crowds out the rank-32 budget.
- New LoRA target modules / higher rank (maxed at 888M; rank capped 32).
- Standing up a new RL framework (NeMo-RL) with days left — your GRPO stack only, and only if a category shows decent pass@k but low pass@1.

## Sources / refs
- `andy_dataset/Claude-Kaggle NVIDIA reasoning challenge 2026 strategy.md` (the session)
- `andy279/nemotron-reasoning-challenge` (HF) — teacher traces + `is_correct_official`
- Tong Hui Kang Progress-Prize writeup + `nemotron.huikang.dev` per-problem visualizer
- `work/2026-06-08_path-to-90.md` (per-category + LoRA-target analysis)
