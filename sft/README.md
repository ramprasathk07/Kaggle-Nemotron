# `sft/` — Supervised Fine-Tuning notebooks

Every SFT experiment for the Nemotron-3 reasoning challenge. The **0.85 leaderboard score
came from this folder** (broad LoRA + single-teacher CoT + plain mean-NLL). Read the root
[`README.md`](../README.md) first for the constraints and the model facts.

## What worked (the recipe to reproduce)

- **Broad LoRA, 888M trainable (~2.74%)**: `q/k/v/o`, Mamba `in_proj`, MoE `gate/up/down`
  (shared + all 128 routable via Unsloth), `lm_head`. **Drop `out_proj`** — it's a dead
  LoRA target under Unsloth (fused Mamba scan output has no `requires_grad`).
- **Plain mean-NLL, full sequence**, LR 2e-4 linear, 1 epoch, eff. batch 16,
  `max_grad_norm=1e9` (no clip), `adam_beta2=0.95`, `max_length=8192`.
- **Single-teacher CoT** (Tong), all rows. Consistency beats volume.
- **Always repackage before submit**: split fused expert keys → per-expert
  (`experts.{j}.up_proj`) + rename to eval namespace, else ~856M of expert LoRA is unused
  at eval.

## What regressed (don't repeat)

- **Multi-teacher CoT mix** (andy + Sonnet + Tong) → **0.67**. Contradictory reasoning
  styles per problem confuse the model. Pick one teacher, or generate solver-perfect CoT.
- **minmax / "blend" worst-token loss** → hurts vs plain mean SFT.
- **Narrow LoRA** (no `lm_head`, no routable experts) → loses the capacity that earns 0.85.
- **Changing 6 things at once** (dataset + targets + loss + LR + epochs + clip) → **0.53**,
  and you can't tell which change did it. **Change one variable per run.**

## Notebook index

| Notebook | Idea | Outcome |
|---|---|---|
| `nvidia-nemotron-v7-5.ipynb` | First Unsloth SFT + LoRA + submission packaging | Pipeline established |
| `nemo-25-apr-1108-fixed.ipynb` | Early fixed-format run | superseded |
| `nemo-v8-1-tensorboard.ipynb` | SFT + TensorBoard logging | tooling |
| `nemo-v8-2-RsLoRA.ipynb` | Rank-stabilized LoRA (RsLoRA) | neutral |
| `nemo-v8-3-DoRA.ipynb` | Weight-decomposed LoRA (DoRA) | neutral, slower |
| `nemo-v9-svd-lora.ipynb` | SVD-initialized LoRA baseline | baseline variant |
| `nemo-v13-sft-rtx6000.ipynb` | Single-GPU RTX 6000 SFT config | infra baseline |
| `nemo-v13-sft-grpo-rtx6000*.ipynb` | SFT→GRPO combined attempt | infra |
| `nemo-v14-sft-optimized.ipynb` | Speed/memory-tuned SFT | infra |
| `nemo-v19-minmax-sft.ipynb` | minmax (worst-token) loss | regressed vs mean |
| `nemo-v20-sft-minmax-logprob.ipynb` | minmax + logprob shaping | regressed |
| `nemo-v21-minmax-sft-clean.ipynb` | **Clean reproducible baseline loader** | the canonical clean run; reference for 0.85 recipe |
| `nemo-v21-sft-standard.ipynb` | Standard SFT (no exotic loss) | baseline |
| `nemo-v22-mambapeft-lora.ipynb` | MambaPEFT-style targeting; **fixed** to drop dead `out_proj`/`x_proj`/`dt_proj`, keep `in_proj`; `requires_grad` audit added | corrected targeting |
| `nemo-v23-sft-9500-even.ipynb` | Balanced 9,500-row corpus | 7.9h run (μ=1 token-bound); led to speed fixes |
| `nemo-v24-sft-9500-even.ipynb` | v23 + speed/hygiene fixes | **0.67** — exposed the mixed-teacher data regression |
| `nemo-v26-spatialclaw-verify-revise.ipynb` | Verify-then-revise trace generation | data engine |
| `nemo-v28-sft-spatialclaw-mix.ipynb` | SpatialClaw-mixed corpus | experiment |
| `nemo-v31-stratified-refine.ipynb` | Stratified batching + **eval-best checkpoint** | hygiene win |
| `nemo-v32-submittable-weighted.ipynb` | **Broad targets + weighted loss + repackage** | the submittable all-layers build (below) |

## The "submittable, all-layers" build — `nemo-v32-submittable-weighted.ipynb`

The most complete SFT notebook. It combines everything learned:

- **Broad targets** `[q_proj, k_proj, v_proj, o_proj, in_proj, up_proj, down_proj, lm_head]`
  (no `out_proj`) → ~888M trainable, meeting the ">800M" capacity goal under rank 32.
- **`requires_grad` audit** — re-enables any LoRA params silently frozen by
  `prepare_model_for_training`.
- **Weighted loss** blending three signals (warmup pure-CE for the first half of training):
  - `W_CE · CE` — standard cross-entropy (the workhorse).
  - `W_DFT · DFT` — reward-rectified, *down*-weights already-easy/confident tokens.
  - `W_HARD · maxmin` — *up*-weights the worst (top-k lowest-logprob) tokens.
  - DFT and maxmin pull opposite directions by design; the pure-CE warmup
    (`WARMUP_MEAN_FRAC=0.5`) stabilizes early training before the shaping kicks in.
  - Defaults: `W_CE=1.0, W_DFT=0.3, W_HARD=0.3, TOPK_MIN=8`.
- **Repackage cell** — diagnoses the saved adapter (counts fused vs per-expert vs
  `out_proj` keys), splits fused routed-expert LoRA into 128 per-expert keys, strips dead
  `out_proj`, renames to the eval namespace, and writes `submission.zip`.
- Config note: requires `DROP_OVERLONG = True` in the config cell (the reused tokenize cell
  references it).

## Data

SFT corpora are built by `../data_manipulation/balance_merge.py` (pools Tong + andy + GPT-5.5
traces, standardizes to `id/type/prompt/answer/generated_cot`, dedups exact
`(prompt, cot)`, caps per type). Solver-perfect CoT comes from `../syn_datagen/`. The format
contract (`<think>…</think>\boxed{}`) is identical to RL and eval.

## Pre-flight (every run, before multi-hour training)

Greedy-decode 3 samples → confirm `\boxed{}` is emitted → confirm the brace-balanced
extractor recovers the answer → confirm pre-flight loss is real (> 0.3, not 0.0). This
single cell caught the zero-loss trap and saved many wasted GPU-hours.
