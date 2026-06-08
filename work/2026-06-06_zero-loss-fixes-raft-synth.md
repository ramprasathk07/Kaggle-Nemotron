# Work Log — 2026-06-06 — Zero-loss SFT fixes, v8 recipe port, RAFT-on-synthetic

Big debugging + build day. Chased the persistent **loss = 0.0** bug across four
notebooks to root cause, ported the proven **0.85 recipe** into the clean SFT
notebook, and built a new **RAFT/Reinforce-Rej on synthetic + train.csv**
notebook warm-started from the 0.85 adapter. Also fixed a Kaggle **disk-full**
crash and several data-format/escaping bugs. Every code cell `ast.parse`-clean.

---

## The zero-loss bug — root causes (THE big lesson)

Multiple notebooks logged `loss = 0.0` forever. There were **two independent
mechanisms**, plus a silent guard hiding both:

1. **Assistant target missing a closing `</think>`.** Old `build_assistant_text`,
   when the cot already contained `\boxed{}`, returned the raw cot with `<think>`
   prepended but **no `</think>`** and the box buried inside. Under
   `enable_thinking=True` the Nemotron/Qwen3 chat template mangles an assistant
   turn that opens `<think>` and never closes it → strips the content →
   `full_text == prefix_text` → the prompt-prefix mask masks **every** token →
   every row dropped by the unmasked-token filter → empty dataset → trainer logs
   `0.0`. Only triggered on datasets whose cots all contain `\boxed{`.

2. **TRL `SFTTrainer` + `dataset_kwargs={"skip_prepare_dataset": True}`** could
   drop the masked `labels` → `nll[mask].mean()` over an **empty mask** → `nan`
   → the silent `loss = logits.sum()*0.0` guard turned `nan` into a clean `0.0`.

**Key deduction:** a *clean* `0.0` (not `nan`) can only come from an empty mask
(finite logits, NaN reduction → guard → 0.0). If logits were NaN the guard gives
`nan`, not 0. That told us it was a **labels/masking** problem, not numerics.

### Fixes applied (nemo-v19, v20, v21, v18-raft)
- **Rebuild every target** as `<think>\n{reasoning}\n</think>\n\boxed{ans}` with a
  guaranteed closing tag + single trailing box from the answer column.
- **Drop TRL `SFTTrainer` → vanilla `transformers.Trainer`** with our own collator
  so `input_ids`/`labels` reach `compute_loss` verbatim.
- **Grad-path hardening:** `model.enable_input_require_grads()` + `use_cache=False`
  (reentrant gradient checkpointing needs an input requiring grad, or grads are 0).
- **Loud loss:** fp32 reductions, sanitise non-finite logits and recompute (real
  grad, not a dead zero), `nonfinite`/`dead` counters printed.
- **PRE-FLIGHT cell:** one real forward+backward asserting loss finite & > 0.3 and
  ≥ 1 LoRA tensor has non-zero grad — run before burning compute. A zero-loss run
  is now impossible to start unnoticed.

---

## Brace-y answers bug (cipher / symbolic)

Dataset has answers that literally contain braces (e.g. `(/&{`). The inline-box
stripper `re.sub(r'\\boxed\{[^{}]*\}', ...)` can't match a box whose **content**
has a `{` → the cot's own box survives → wrapper adds a 2nd → the
`count("\\boxed{") != 1` check raised "malformed targets" on ~107/7830 rows.

**Fix:** brace-balanced `_strip_boxed()` walk + **presence-based** format check
(`</think>` exists AND `\boxed{` after the last `</think>`) — never count boxes,
never require a trailing `}`. Later switched the hard `raise` to **drop + warn**
(1.4% dropped is harmless; don't crash 7700 good rows).

---

## The 0.53 vs 0.85 analysis (why the rewrite regressed)

User's v21 (clean rewrite) scored **0.53**; the saved `nvidia-nemotron-v8-85%`
scored **0.85**. v21 had changed ~6 things at once off the known-good recipe:

| Knob | v8 (0.85) | v21 (0.53) |
|---|---|---|
| Dataset | Tong `problem_ids_matched.csv` | different `new_dataset_filtered2.csv` |
| Rows | all 7830 | 3000 subset |
| LoRA | q/k/v/o + in/out/up/down + **lm_head** + 128 experts → **888M** | attn + shared experts only, no lm_head |
| Loss | plain mean NLL, full-sequence | blend (minmax worst-token), masked |
| LR/sched | 2e-4 linear, 1 ep | 8e-5 cosine, 2 ep |
| max_grad_norm | **1e9 (no clip)** | 1.0 |

**Lessons (saved to memory `winning-sft-recipe.md`):**
- `lm_head` + 128 routable experts in LoRA **empirically help** here (888M, 2.74%)
  — directly contradicts the old CLAUDE.md "avoid lm_head/experts" note.
- minmax/blend loss **hurts** vs plain mean SFT from cold.
- Tong's `max_grad_norm=1e9` (no clipping) is deliberate.
- 90% = data (Tong augmentation + golden synthetic) + RAFT, **not** loss tweaks.

---

## Files changed / created

### `sft/nemo-v21-minmax-sft-clean.ipynb` — ported to the 0.85 recipe
- Dataset → Tong `problem_ids_matched.csv`, `SUBSET_N=None` (all 7830).
- LoRA → Unsloth `get_peft_model` broad targets incl `lm_head` (≈888M).
- LR 2e-4 linear, 1 epoch, eff-batch 16, `max_grad_norm=1e9`, `weight_decay=0`.
- **Safe late-blend logprob:** `WARMUP_MEAN_FRAC=0.8` (pure mean 80% of run, then
  `blend` `BLEND_ALPHA=0.8`, `TOPK_MIN=8`). `WARMUP_MEAN_FRAC=1.0` = pure recipe.
- Brace-balanced data cell (drop malformed, don't raise).
- **Disk fix:** `save_strategy="no"` (no mid-run 4 GB checkpoints) + package zips
  the adapter **in place** (killed the 4 GB duplicate copy that threw
  `No space left on device`). Peak disk ≈ 7.6 GB now.

### `post-training/nemo-v20-raft-synth.ipynb` — NEW (RAFT on synthetic + train.csv)
- Reuses v19's setup / adapter-loader / **synthetic generators** / **verifiers**
  verbatim; swaps the GRPO trainer for the v18 rejection-sampling engine.
- Warm-starts the 0.85 adapter (`/kaggle/input/models/ramkan07/nemotron-lora-adaptor/pytorch/default/1`).
- Loop: sample 8 rollouts/prompt (Mamba-safe per-prompt, NaN guard) → verify with
  the type-aware `\boxed{}` checker → reinforce-rej filter (keep correct from
  **mixed** groups) → vanilla-Trainer mean-NLL self-distill.
- **Important:** the model trains on **its own verified-correct CoT**, not on the
  synthetic answer strings. Synthetic answers are only the verifier's ground truth.
  Synthetic *problems* are ephemeral (regenerated from `SEED=3407`); only the
  **rollouts** are cached (`raft_rollouts.jsonl`).
- **Bug found + fixed after build:** cell order — reused `triton`/`ptxas` cells ran
  before the config cell that defines `os`/`sys`/`TRAIN_ON_KAGGLE` → `NameError`.
  Reordered config first.

### `post-training/nemo-v18-raft-reinforce-rej.ipynb` — hardened
- Vanilla `Trainer` migration, grad hardening, empty-corpus + empty-dataset guards,
  pre-flight, robust adapter resolver (Kaggle Model slug → dataset dir → glob).
- Tuned for a strong base: `N_ROLLOUTS=8`, `KEEP_FORMAT_VALID=True`, `RAFT_LR=3e-5`.
- Gen-length fix: `SMOKE_GEN_MAX_NEW` 256→1024 (256 truncated before `\boxed{}` →
  0 correct rollouts), real `GEN_MAX_NEW` 1024→2048; added `[rollout-dbg]` print.

### `data/build_golden_cot_dataset.py` — NEW (earlier in the day)
- 250-per-category synthetic golden-CoT generator for the 10 reasoning families;
  deterministic solver writes the reasoning → perfect labels. Output schema matches
  the SFT pipeline. 729 spot-checks, 0 label mismatches. (memory: `golden-cot-dataset.md`)

---

## Mamba / generation gotchas (re-confirmed)

- **Nemotron-3-Nano Mamba-2 NaNs on LEFT-padded batches** → `probability tensor
  contains inf/nan` at `multinomial`. Fix everywhere: **per-prompt generation**
  (one `model.generate` per prompt, N rollouts via `num_return_sequences` → all
  same length → no padding). Plus a `SanitizeLogits` NaN safety net (must read 0).
- Submission `submission.zip` accepted iff: at `/kaggle/working` root, **two files
  at zip root** (arcname = bare filename, no subfolder), `adapter_config.json`
  points at base model, rank ≤ 32. v8 (accepted) and v21/v20 produce identical
  zip structure.

---

## Memory written

- `winning-sft-recipe.md` — the 0.85 config vs the 0.53 regression + why.
- `zero-loss-sft-trap.md` — the four mechanisms + pre-flight prevention.
- `golden-cot-dataset.md` — the synthetic generator.

---

## Open items / next

- **Reproduce 0.85 first** with v21 (recipe + `WARMUP_MEAN_FRAC=1.0`), confirm,
  upload adapter to Kaggle Models.
- Then **v20 RAFT** warm-started from it: smoke (check `[rollout-dbg] ok=True`,
  `hit-rate>0`, `NaN-guard=0`, corpus non-empty), then real run, **eval-gate vs
  0.85** before submitting (RAFT can regress a strong base).
- Push past 90: raise `N_ROLLOUTS`→16, curriculum (easy synthetic first),
  oversample weak categories (per-label hit-rate tells which), 2–3 RAFT rounds,
  SDPG dense reward (v19) only for the stubborn residual.
- Consider mixing `golden_cot_2500.csv` + Tong augmentation into the SFT base
  (v8 skipped augmentation — ~5pp left on the table).
