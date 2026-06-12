# Nemotron Reasoning Challenge — Day 2–4 Tooling

Two tested, dependency-light tools (numpy only) for the final push.
Deadline: June 15. Slot these in once your Day-0/1 failure ledger exists.

## 1. `rule_trace_generator.py` — rule-enumeration SFT traces

Converts rule *induction* into rule *retrieval*: traces enumerate a finite
candidate-rule library, test each candidate against the in-prompt examples,
eliminate failures, apply the survivor, verify, and emit `\boxed{}`.
Every claim in a trace is computed by executing the rule functions —
traces are mechanically truthful, never LLM-hallucinated.

**To use on a real category (per weak category in your ledger):**
1. Copy 2–3 real prompts from train.csv into the file and write
   `parse(prompt) -> (example_pairs, query_input)` (regex).
2. Write `canon(answer)` to reproduce ground-truth formatting EXACTLY
   (leading zeros for bit strings; fixed decimals for ±1e-2 categories).
   Exact-string match is your insurance against the buggy numeric branch
   of the official metric.
3. Fill `rules` from Hui Kang's visualizer notes (nemotron.huikang.dev,
   per-problem rule-found/unknown status) and andy279's solver-discovered
   rules (the raw-traces dataset metadata).
4. Run `build_sft_dataset` over that category's train.csv rows.

**The stats dict is your generator validator.** Demand ~100% `written` on
train.csv for a category before generating synthetic data at scale:
- `no_rule_fits_examples` → your library is missing a rule.
- `rule_fits_examples_but_not_answer` → your rule or parser is WRONG —
  this is the cryptarithm-operator trap; investigate before training.
- `ambiguous_multiple_rules_fit` → two rules agree on the examples but the
  builder picked the one matching ground truth; check which wins on the
  *private* convention before trusting it broadly.

**Integration:** output is messages-JSONL in Nemotron thinking format
(`<think>…</think>\boxed{}`). `answer_span` gives char offsets of the final
boxed answer — wire it to 5–10x loss weighting on those tokens in your SFT
stack. `est_tokens` lets you filter under the eval generation budget you
extracted from the official demo notebook.

**`provisional_box` is OFF by default.** Enable only after you confirm in
the metric code that extraction prefers the LAST box and scans the full
generation (truncation insurance depends on it).

Run `python3 rule_trace_generator.py` for a working two-category toy demo.

## 2. `lora_merge.py` — TIES / DARE / linear adapter merging

Merges adapters trained on the same base (your SFT/GRPO/RAFT runs,
category-expert adapters, Tong's public adapter) **without the base model
and without torch** — runs on your 4090 box or a laptop. Reconstructs each
module's ΔW = scale·B·A, combines (linear soup, TIES sign-election, or
DARE drop-and-rescale), then SVD-refactorizes to a uniform rank-32 adapter
with alpha=rank (effective scale exactly 1.0).

```
python3 lora_merge.py \
  --adapter runs/sft_best:0.5 --adapter runs/grpo:0.3 --adapter runs/expert_crypt:0.2 \
  --mode ties --density 0.5 --rank 32 --out merged/
```

- Handles BF16/F16/F32 adapters and mismatched key styles across training
  stacks; output verified byte-compatible with the official safetensors lib.
- Watch the **energy report**: mean/min near 1.0 = the rank-32 truncation
  kept the signal; low-energy modules mean the merged update didn't fit in
  rank 32 — fewer or more-similar adapters, or accept the loss knowingly.
- Grid to try first: {ties d=0.5, ties d=0.7, dare_ties p=0.3, linear} ×
  2–3 weightings. Cheap enough to run all before your Day-3 eval sweep.

## Non-negotiables

- **Never submit a merge you haven't locally evaluated** with the
  official-metric replica (buggy binary-string comparison included).
  Merges are hypotheses.
- Train-data filtering: strict correctness. Self-measurement: official
  buggy metric. Different jobs, different flags.
- If using Tong's public adapter in a soup, re-check the competition rules
  on publicly shared artifacts first (it was shared on the forum, which is
  normally fine — verify).
- `modules_to_save` tensors are skipped with a warning by design — full
  embedding/head saves risk failing the server-side vLLM load.
