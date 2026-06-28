# NVIDIA Nemotron-3 Reasoning Challenge — LoRA Fine-Tuning

Competition entry for the **[NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)**.
Goal: fine-tune a **rank-32 LoRA adapter** on `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
to maximize accuracy on ~7,500 deterministic logic/math puzzles, on a single GPU, with
LoRA-only submissions (no full fine-tune, no merged weights).

**Best leaderboard score: 0.85** (SFT). This repo documents the full journey — what was
tried, what worked, what regressed, and why.

---

## TL;DR — what actually moved the score

| Lever | Effect | Verdict |
|---|---|---|
| Broad LoRA (888M: attn + Mamba `in_proj` + **all 128 MoE experts** + `lm_head`) | **0.85** | ✅ The win. Capacity matters here. |
| Plain mean-NLL loss, full-sequence, LR 2e-4, 1 epoch, no grad clip | baseline of 0.85 | ✅ Boring beats clever. |
| Single-teacher (Tong) CoT, full 7,830 rows | 0.85 | ✅ Consistency > volume. |
| **Multi-teacher** CoT mix (andy + Sonnet + Tong) | **0.67** | ❌ Contradictory traces regress the model. |
| Narrow LoRA + minmax/blend loss + 3k-row subset (6 changes at once) | **0.53** | ❌ Never change 6 vars at once. |
| LoRA-target / rank tweaks (rank capped at 32) | ≪1pp | ➖ Already maxed; not the path to 0.90. |
| RL (GRPO/Dr.GRPO/CISPO) under HF generate | infeasible | ⚠️ ~1 tok/s — Mamba has no incremental cache off-vLLM. |

**The path to 0.90 is data + per-category coverage, not architecture tweaks.** 0.85 overall
hides ~95% on deterministic-arithmetic families (numeral / gravity / unit / base) and
~60–75% on the hard tail (cipher long-phrase, complex bit-manipulation, symbolic equation,
cryptarithm). The remaining points live in those bottom categories.

---

## Hard submission constraints (non-negotiable)

| Constraint | Value |
|---|---|
| `max_lora_rank` | **32** (adapter rejected if higher) |
| Decoding | greedy, `temperature=0.0` |
| `max_tokens` | 7680 (model max 8192) |
| `max_num_seqs` | 64 |
| Answer format | must end with `\boxed{...}` |
| Submission | `submission.zip` = `adapter_config.json` + `adapter_model.safetensors` |
| Inference | vLLM with LoRA, greedy |

---

## The model (why targeting is unusual)

Nemotron-3-Nano-30B-A3B is a **hybrid Mamba-Transformer MoE**:
- 52 layers: **6 GQA self-attention** + **46 Mamba-2** SSM layers
- **128 routable experts** (top-6 active/token) + shared expert(s) always active
- No positional embeddings, no dropout, no linear bias, untied embedding & `lm_head`,
  RMSNorm, squared-ReLU MoE activation, sigmoid-gated aux-loss-free router

Three findings drove every adapter we shipped (details in [`sft/README.md`](sft/README.md)):

1. **`out_proj` LoRA is dead under Unsloth.** The fused Mamba scan output lacks
   `requires_grad`; the checkpoint backward guard zeroes its gradient. `in_proj` is the
   only live Mamba LoRA target. (`x_proj`/`dt_proj` are fused into `in_proj`.)
2. **`prepare_model_for_training` silently freezes** LoRA params whose names don't match
   the dotted `.lora_A.`/`.lora_B.` pattern → always audit `requires_grad` after wrapping.
3. **Fused-expert keys ≠ eval keys.** Unsloth saves *fused* expert LoRA, but the Kaggle
   evaluator exposes **128 per-expert `nn.Linear`** (`experts.{j}.up_proj`). Submitting the
   fused keys silently drops ~856M of trained expert weights at eval. Every submission
   must **split fused → per-expert keys + rename to the eval namespace** before zipping.

---

## Data format (identical across SFT / RL / eval)

```
<think>
{step-by-step reasoning}
</think>
\boxed{final_answer}
```

- `<think>` opening tag required. `\boxed{}` extraction uses a **brace-balanced parser**
  (nested `\boxed{\frac{1}{2}}` must round-trip), not flat regex.
- Validation checks the response **ends** with `\boxed{}` — not "exactly one occurrence"
  (CoT may reference boxed earlier).

---

## Repository layout

```
Kaggle-Nemotron/
├── README.md                    # this file
├── docs/
│   ├── BLOG_TECHNICAL.md        # deep technical write-up
│   └── BLOG_APPROACH.md         # the narrative / approach blog
├── sft/                         # supervised fine-tuning notebooks  (see sft/README.md)
├── post-training/               # RL + RAFT notebooks               (see post-training/README.md)
├── data/                        # raw competition data + CoT snapshots
├── data_manipulation/           # corpus balancing/merging scripts (balance_merge.py)
├── data_generation/             # CoT collection pipelines
├── syn_datagen/                 # deterministic per-category "reasoners" (solver-perfect CoT)
├── RLVR/                        # verifiable-reward solvers (the reward oracle for RL)
├── gpt5_trace_gen/              # GPT-5.5-pro trace generator (aisa.one /v1/responses)
├── andy_dataset/                # external trace corpus + extraction tools
├── reference/                   # huikang/Tong reference notebooks (rank ~89)
├── EDA/ + EDA.ipynb             # exploratory analysis
├── tools/                       # misc utilities
├── work/                        # planning docs, incl. WINNING_PLAN.md (phased strategy)
└── outputs/                     # (gitignored) trained adapters, scored notebooks
```

---

## Reproduce the 0.85 baseline

1. **Data**: single-teacher Tong CoT, all rows, format `<think>…</think>\boxed{}`.
2. **Model**: Unsloth `FastLanguageModel.get_peft_model`, rank 32 / alpha 64, targets
   `q/k/v/o, in_proj, gate/up/down, lm_head` → Unsloth auto-adds the 128 routable experts
   → **888M trainable (~2.74%)**. Drop `out_proj` (dead).
3. **Loss**: plain mean-NLL, full-sequence (no assistant-only masking).
4. **Optim**: LR 2e-4 linear, warmup 0, 1 epoch, eff. batch 16, `max_grad_norm=1e9`
   (clipping intentionally off), `wd=0`, `adam_beta2=0.95`, `max_length=8192`,
   grad-checkpoint `use_reentrant=False`. ~4h on one RTX 6000 Pro.
5. **Submit**: split fused experts → per-expert keys, rename to eval namespace, zip.

Canonical SFT notebooks: `sft/nemo-v21-minmax-sft-clean.ipynb` (clean baseline),
`sft/nemo-v32-submittable-weighted.ipynb` (broad targets + weighted loss + repackage).

---

## Leaderboard history (our submissions)

| Score | Run | What it was |
|---|---|---|
| **0.85** | best SFT | Tong-only CoT, 888M broad LoRA, plain mean-NLL |
| 0.84 | v20 warm-start | reference-style adapter, barely trained |
| 0.67 | v24 | mixed-teacher CoT regression (data, not config) |
| 0.53 | v21 over-changed | narrow LoRA + minmax loss + 3k subset — six changes at once |

> Reference (not our submission): huikang's SVD-recompose notebooks suggest ~0.87 is
> reachable by cleanly recomposing strong adapters. See `nemotron-eval-namespace-svd`.

**Note on accuracies:** the Kaggle CLI authenticates with an API token (`kaggle.json`), not
a password. Live leaderboard pulls were not run here; the numbers above are from our own
scored submissions during the event. Drop a `kaggle.json` in the repo to fetch live scores.

---

## Key gotchas (learned the hard way)

1. **Zero-loss SFT trap** — TRL `SFTTrainer` label handling produced `loss=0.0`; the fix
   was a vanilla `Trainer` + manual labels + a pre-flight cell that greedy-decodes 3
   samples and asserts a real loss > 0.3 before any multi-hour run.
2. **`max_completion_length` too short in RL** — Nemotron prepends `<think>`; with 256 the
   completion truncates before `\boxed{}`, all rewards = 0, loss = 0. Use ≥512.
3. **No fantasy prompt wrappers** — competition tests rule-following; use clean
   `{problem}\nPlease put your final answer inside \boxed{}`.
4. **Mamba + HF `generate` = ~1 tok/s** without `NemotronHHybridDynamicCache` (O(n²)
   decode). RL needs vLLM-speed gen; the offline Kaggle env has no vLLM wheel → RL was not
   viable. Every RL notebook now has a **30-second gen-speed probe that aborts** before a
   multi-hour run if `tok/s < threshold`.
5. **Avoid `--no-verify` / `git push --force`** — no destructive git ops.

---

## Tech stack

Unsloth · PEFT/LoRA · TRL · PyTorch · bitsandbytes (AdamW8bit) · vLLM (eval only) ·
single RTX 6000 Pro (Blackwell, 96 GB) for local training; Kaggle GPU for eval.

See [`docs/BLOG_TECHNICAL.md`](docs/BLOG_TECHNICAL.md) for the full technical deep-dive and
[`docs/BLOG_APPROACH.md`](docs/BLOG_APPROACH.md) for the approach narrative.
