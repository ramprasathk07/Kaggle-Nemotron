# CLAUDE.md — Kaggle Nemotron-3 Reasoning Challenge

Working notes for AI assistants on this repo. Read before editing.

## What this repo is

Kaggle competition entry: fine-tune `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` with a LoRA adapter to maximize accuracy on a ~7,500-problem logical reasoning benchmark (bit manipulation, algebraic equations, number-base conversion, gravitational constant, text encryption, unit conversion, equation transformation).

- Strategy doc: `WINNING_PLAN.md` (Dr.GRPO + DAPO + RAFT, 4-week phased)
- Background notes: `README.md`
- Tech report reference: arXiv 2512.20848 (Nemotron 3 Nano)

## Hard submission constraints (non-negotiable)

| Constraint | Value | Why it matters |
|---|---|---|
| `max_lora_rank` | 32 | Adapter rejected if higher. Use rank=32 to max capacity. |
| `temperature` | 0.0 (greedy) | No sampling tricks at eval. Model must be confidently correct. |
| `max_tokens` | 7680 | Prompts must leave room (max_model_len=8192). |
| `max_num_seqs` | 64 | Batch ceiling at inference. |
| Answer format | `\boxed{...}` | Extractor: boxed → heuristic → last numeric. Train so model *always* ends with `\boxed{}`. |
| Submission | `submission.zip` w/ `adapter_config.json` + `adapter_model.safetensors` | LoRA only — no full FT, no merged weights. |

## Model architecture facts that drive LoRA targeting

Nemotron-3-Nano-30B-A3B is **MoE + hybrid Mamba-Transformer**:
- 52 layers: **6 GQA self-attention** + **46 Mamba-2** SSM layers
- 128 routable experts (6 active/token) + 2 shared experts always active
- No positional embeddings, no dropout, no linear bias
- Untied embedding & lm_head, RMSNorm, squared-ReLU MoE activation

### LoRA target priority (from quantization sensitivity in §4.2 of paper)

| Target group | Suitability | Reason |
|---|---|---|
| `self_attn` (q/k/v/o on 6 layers) | **High** | Quantization-sensitive → high impact under perturbation. |
| Mamba layers immediately preceding attention (6 layers, x_proj/dt_proj) | **High** | Also kept in BF16 by NVIDIA → sensitive. |
| Shared experts (gate/up/down × 2 per layer) | **Good** | Always active; see every token. |
| Remaining 40 Mamba layers | **Low ROI** | Robust to perturbation — LoRA budget wasted here. |
| Routable experts (128/layer) | **Avoid** | Only 6/128 active per token → sparse gradient signal. |
| Router / gate | **Avoid** | NVIDIA freezes router during RL — modifying destabilizes routing. |
| `embedding` / `lm_head` | **Risky** | Untied; large; can destabilize. v9 excluded `lm_head`. |

To find the 12 sensitive layers programmatically: scan the model, the Mamba layer at index `i-1` where layer `i` is `self_attn` is "pre-attention Mamba."

## Data format (must match across SFT, GRPO, eval)

Assistant response template:
```
<think>
{step-by-step reasoning}
</think>
\boxed{final_answer}
```

- `<think>` opening tag is required. v1_data_prep.py had it; old notebook cell missed it — keep consistent.
- `\boxed{}` extraction uses brace-balanced parser, not flat regex (nested braces like `\boxed{\frac{1}{2}}` must round-trip).
- Validation: response must *end* with `\boxed{}` — do not assert "exactly one occurrence" since CoT may reference boxed earlier.

## Training stack & file layout

```
data/
  src/{train,test}.csv                 # raw competition data
  generated_cot/                       # CoT-augmented snapshots
  merged_cot_final.csv                 # current SFT corpus
  golden_cot_1000.csv                  # high-quality teacher-distilled subset
  v5_balanced_cot.py                   # canonical CoT collector
  collect_golden_cot.py                # teacher CoT generator
  merge_cot_datasets.py                # corpus merger

sft/
  nemo-v9-svd-lora.ipynb               # current SFT baseline (SVD-init LoRA)
  nemo-v8-*.ipynb                      # older variants: RsLoRA, DoRA, TensorBoard

post-training/
  nemo-v9-1-Grpo.ipynb                 # baseline GRPO (TRL)
  nemo-v10-grpo-traincsv.ipynb         # GRPO on train.csv directly
  nemo-v11-DrGrpo.ipynb                # Dr.GRPO migration (Phase 1)
  nemo-v12-DAPO.ipynb                  # DAPO upgrades on top of Dr.GRPO (Phase 2)

WINNING_PLAN.md                        # strategy
README.md                              # competition overview
```

## Phase plan summary (from WINNING_PLAN.md)

| Phase | Days | Output | Target |
|---|---|---|---|
| 0 | 1–2 | Clean data + baseline SFT (v9) submission | leaderboard score `S0` |
| 1 | 3–6 | Dr.GRPO migration (v11): LR=1e-6, β=0, scale_rewards=False, num_generations=8, T=1.0 | stable RL |
| 2 | 7–10 | DAPO upgrades on v11: clip-higher (0.2/0.28), dynamic sampling, token-level loss, overlong shaping | `S1 ≥ S0 + 8pp` |
| 3 | 3–10 (parallel) | RAFT distill: N=16 rollouts/problem, top-k SFT refresh | data engine |
| 4 | 11–14 | Curriculum: bucket by pass@8, hard set → GRPO, easy → SFT | iteration |
| 5 | 15–17 (optional) | EGGROLL-LoRA wild card (rank-1 perturbation, pop=128, σ=0.01) | wild card |
| 6 | 18–20 | Inference tune + best-of-2 LoRA submission | final |

## Critical gotchas (do not repeat)

1. **`max_completion_length=256` was too short** — Nemotron prepends `<think>` reasoning, completions get truncated before `\boxed{}`, all rewards = 0, loss = 0. **Use ≥512.** Better: `max_seq_length − max_prompt_length`.
2. **`mask_truncated_completions=True`** is required in GRPOConfig — DAPO paper standard, excludes truncated samples from loss.
3. **`scale_rewards`**: default `"group"` (std-norm) has difficulty bias. Set `False` (Dr.GRPO) or `"batch"` (PPO Lite).
4. **`beta=0.0`** (no KL, no reference model) saves ~50% memory. Standard post-DeepSeek-R1. v9 used `beta=0.04` — drop in v11.
5. **Effective batch must be divisible by `num_generations`**: `num_processes × per_device_batch_size × grad_accum_steps` must be a multiple of `num_generations`.
6. **TRL ≥0.16 required** for `loss_type="dr_grpo"`. Kaggle has no internet — bundle wheel in dataset.
7. **Reward function signature**: must accept `prompts`, `completions`, `completion_ids`, `trainer_state`, `log_extra`, `log_metric`, and dataset columns via `**kwargs`. Conversational dataset → `completions` is `list[list[dict]]` — handle with `isinstance(completion, list)`.
8. **Reward hacking**: model may emit fake early `\boxed{}`. Format reward must require `<think>...</think>` *before* `\boxed{}`.
9. **vLLM + LoRA + Mamba is fragile**: test `submission.zip` end-to-end early in every phase, not just at the end.
10. **No Wonderland prompt wrappers**: the competition tests rule-following, not fantasy lore. Use clean `{problem}\nPlease put your final answer inside \\boxed{}`.
11. **Skip the lm_head LoRA target.** v9 confirmed it destabilizes; untied embeddings mean changes don't propagate the way you'd expect.
12. **Avoid `--no-verify` / hook bypass / `git push --force`.** No destructive git ops without explicit ask.

## Reward-function metrics to log

- `reward` — primary learning signal
- `completions/clipped_ratio` — truncation rate (should drop over training)
- `completions/mean_length` — watch for length explosion
- `frac_reward_zero_std` — diversity within group (1.0 = all same reward = no learning signal)

## Reference settings cheat sheet

### Dr.GRPO (Phase 1)
```python
learning_rate=1e-6
beta=0.0
epsilon=0.2
num_generations=8
temperature=1.0
top_p=1.0
top_k=0
max_prompt_length=3072
max_completion_length=3000
loss_type="dr_grpo"
scale_rewards=False
mask_truncated_completions=True
per_device_train_batch_size=1
gradient_accumulation_steps=16
max_grad_norm=1.0
optim="adamw_8bit"
bf16=True
gradient_checkpointing=True
```

### DAPO add-ons (Phase 2)
- `epsilon_low=0.2`, `epsilon_high=0.28` (asymmetric clip → entropy preservation)
- Dynamic sampling: oversample 2× then drop groups with `mean(reward) ∈ {0, max}` and refill
- Token-level loss (sum of token log-probs × advantage, no per-sequence mean)
- Overlong soft penalty: `-min(0, (L_max + 512 − |o|) / 512)` between `L_max` and `L_max+512`, hard cutoff above

### NVIDIA's own GRPO (reference only — needs cluster compute)
- 128 prompts/step × 16 generations × batch 2048
- Max generation length 49,152 tokens
- Router frozen, aux-loss-free expert-bias load balancing (update 1e-3)

## When making changes

- Match data format across SFT and RL (chat template, `<think>...</think>\boxed{}`).
- Use `from peft import LoraConfig` target_modules patterns that match Nemotron module names — verify with `model.named_modules()` first; Nemotron has `mamba.x_proj`, `mamba.dt_proj`, `shared_expert.*`, `self_attn.{q,k,v,o}_proj`.
- Before launching multi-hour training, do a 3-sample sanity check: greedy-decode, confirm `\boxed{}` is emitted, confirm answer extraction works.
- Never edit the eval pipeline assumptions: greedy decoding, `max_tokens=7680`, vLLM with LoRA — local training must match this.

## Open items / not-yet-decided

- Whether to retrain on `golden_cot_1000.csv` vs. `merged_cot_final.csv` as SFT base for Phase 1
- Whether DAPO clip-higher needs a custom monkey-patch (TRL flag may not exist) — confirm on current `trl` version before Phase 2
- EGGROLL (Phase 5) only if there's spare compute — PyTorch port doesn't exist yet, JAX-only ref impl