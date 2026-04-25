# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Submission for the Kaggle **NVIDIA Nemotron Model Reasoning Challenge**. The deliverable is a LoRA adapter (`adapter_config.json` + `adapter_model.safetensors`) packaged as `submission.zip`, trained on top of `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (a hybrid Mamba SSM + Attention + MoE model, ~3B active params from 30B). Inference at evaluation time runs through vLLM on Kaggle.

Scoring: answer must appear inside `\boxed{...}`. Exact string match for text, ±1e-2 tolerance for numbers.

## Hard constraints (do not violate)

- **`max_lora_rank = 32`** — competition rule. No per-module LoRA matrix may have rank > 32. When using `rank_pattern`, every entry must be ≤ 32.
- The submission must contain only the two adapter files; `adapter_config.json` must have `base_model_name_or_path = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"` and `inference_mode = True` (the packaging cell rewrites these).

## Active notebook vs. README drift

The README still references `nvidia-nemotron-v7-5.ipynb`, but the **actually active training notebook is `nvidia-nemotron-v6-updated.ipynb`** (per recent commits, e.g. `638b218 Replace v8-improved notebook with v6-updated notebook`). Edit v6-updated unless explicitly told otherwise. The README's "Top 5 Things to Try" / status table is aspirational — trust git + the v6-updated notebook for current state.

## Pipeline architecture (the big picture)

The single training notebook drives everything via two top-level mode flags and two stage flags (cell `c2`):

```
TRAIN_ON_KAGGLE + USE_PRETRAINED == 1     # mutually exclusive (asserted)
RUN_SFT, RUN_GRPO                         # within TRAIN_ON_KAGGLE
```

- **Mode A (`TRAIN_ON_KAGGLE=1`)** — train from scratch on Kaggle GPU. Goes through Triton/ptxas environment fixes → offline pip install from `/kaggle/input/datasets/mayukh18/nemotron-packages/packages` (no network on Kaggle) → load base model with Unsloth at 8-bit → build LoRA adapter (cell `c10`) → optional SFT (cell `c14`) → optional GRPO (cell `cell-16`) → package submission.
- **Mode B (`USE_PRETRAINED=1`)** — skip training, just package an adapter from `PRETRAINED_ADAPTER_DATASET_PATH` into `submission.zip`.

The submission packaging cell (`c19`) picks the source adapter automatically: GRPO output if `RUN_GRPO`, else SFT output, else the pretrained path.

### Two-stage training: why both

- **SFT** (`StratifiedSFTTrainer` in `c14`) teaches the *format* (`<think>...</think>\boxed{...}`) and broad reasoning. Uses curriculum learning (sort by CoT length, easy→hard) and stratified-by-`type` batching so each effective batch sees a mix of problem categories. Quality-filters CoT length to `[100, 6000]` chars and dedups by prompt MD5.
- **GRPO** (`cell-16`) teaches the model to be *correct*. Reward = `accuracy_reward` (1.0 if `\boxed{}` matches ground truth, with float tolerance) + `format_reward` (small bonuses for `<think>` and `\boxed{}` presence). Generates 4 completions per prompt, ranks, updates policy. KL `beta=0.04`.

## Correctness landmines

These have already burned past versions; preserve them:

1. **Brace-balanced `\boxed{}` parsing.** The naive regex `\\boxed\{([^}]*)\}` breaks on nested braces like `\boxed{\frac{1}{2}}`. The notebook defines `extract_boxed()` / `remove_boxed()` in cell `c9` that handle nesting; use these everywhere, including in reward functions and any new data-prep code.
2. **Opening `<think>` tag must be present in training targets.** v6 (pre-update) was missing it — assistant content must be `f"<think>\n{cot_cleaned}\n</think>\n\\boxed{{{answer}}}"`. The chat template is applied with `enable_thinking=True`.
3. **Tokenizer pad token.** Set to `eos_token` if missing (cell `c8`).

## v1_data_prep.py — known to be partially deprecated

This script wraps math problems in **"Alice in Wonderland" themed prompts**. Per the README, this is counter-productive: the actual benchmark tests bit manipulation / custom operators / logical puzzles, not Wonderland fantasy. Don't extend the Wonderland wrappers — if generating new SFT data, mirror the format the notebook actually uses:

```
{problem}\nPlease put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
```

The script also has a broken `\boxed{}` regex (same nested-brace issue as #1 above) and a too-strict `count("\\boxed{") == 1` validator. Fix in place if you touch it; don't propagate the broken regex into new code.

## Running things

This codebase is **not run locally** — it's executed in Kaggle's notebook environment, which provides the `/kaggle/input/...` datasets, the GPU, and the Triton/Mamba/causal-conv1d wheels. There is no local build/test/lint toolchain. Concretely:

- **Training**: open `nvidia-nemotron-v6-updated.ipynb` on Kaggle, set the mode flags in cell `c2`, Run All. Reference SFT run is ~7h on Kaggle GPU.
- **`v1_data_prep.py`**: standalone CLI, runs locally if `datasets` is installed.
  - Dry run (no HF download): `python v1_data_prep.py --dry-run`
  - List datasets: `python v1_data_prep.py --list-datasets`
  - Real run: `python v1_data_prep.py --datasets numina --max-per-dataset 50000 --output data/augmented/numina_50k.jsonl`

There are no unit tests in this repo.

## Paths to know

- `BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"` — referenced when rewriting `adapter_config.json` for submission.
- `DATASET_PATH = "/kaggle/input/datasets/dgxchen/nemotron-cot-tong/problem_ids_matched.csv"` — the SFT/GRPO source. Schema: `prompt`, `answer`, `generated_cot`, `type`.
- `PRETRAINED_ADAPTER_DATASET_PATH` — used only in Mode B.
- Training outputs: `/kaggle/working/sft_adapter`, `/kaggle/working/grpo_adapter`, packaged into `/kaggle/working/submission.zip`.

## Common tasks

- **Tuning LoRA capacity**: edit cell `c10`. Honor the rank-32 cap. Heterogeneous ranks via `rank_pattern` are supported by the Unsloth/PEFT path; pair with `alpha_pattern` so RSLoRA scaling stays consistent (alpha = 2·rank per module).
- **Tuning training schedule**: `SFTConfig` in cell `c14`, `GRPOConfig` in `cell-16`. Effective SFT batch = `per_device_train_batch_size * gradient_accumulation_steps` (currently 2×4=8); the stratified sampler builds an order based on this number, so changing either invalidates the precomputed order — leave the stratified-order build in sync.
- **Changing reward shaping**: `accuracy_reward` / `format_reward` in `cell-16`. Both must use `extract_boxed()` (not regex) for correctness on nested braces.
- **Adding a new data source**: write an adapter in `v1_data_prep.py` following the existing per-dataset pattern, register it in `DATASET_CONFIGS`. Output schema must be `{"messages": [{"role":"user",...},{"role":"assistant","content":"<think>...</think>\\boxed{...}"}]}`.
