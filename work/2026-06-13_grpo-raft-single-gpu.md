# Post-training on ONE RTX 6000 Pro — RAFT/Dr.GRPO analysis + decoupled pipeline

Date: 2026-06-13. Goal: efficient RL/RAFT on top of the 0.85 checkpoint, single 96 GB GPU,
after the old post-training notebooks (v6/v18/v25) "failed or took too long / got stuck."

## Why the old code was slow / stuck
1. **Generation in HF `generate`, eager attention.** FA2 is rejected on the trust_remote_code
   Nemotron-H path → eager → ~600 tok/s on the 30B MoE. RAFT/GRPO need thousands of rollouts →
   hours.
2. **Mamba-2 NaNs on left-padded batches** → forced per-prompt generation (no batching) → even slower.
3. **Online GRPO (TRL)** regenerates every step (on-policy) interleaved with backprop, model in
   train mode → generation + training fight for 96 GB; TRL+PEFT+Mamba is fragile; ref forward (if
   beta>0) doubles cost.
4. Net: the bottleneck is **rollout generation**, and the old design put it on the slowest engine
   in the hottest loop.

## The fix: DECOUPLE, generate with vLLM
- **vLLM is the competition's own inference stack** — paged-attention batched generation on this
  exact model + LoRA (that is how submissions are scored). ~10–20× faster than HF eager, and it
  handles batching/padding internally (no Mamba NaN problem).
- Split into two notebooks so generation and training **never coexist in memory** and so training
  can be re-run (different filters/params/LR) **without regenerating**:
  - **Notebook A** — load 0.85 adapter into vLLM, sample G rollouts/prompt, score against
    ground truth (the official `compare_answer` metric), save groups to JSONL.
  - **Notebook B** — RAFT filter the JSONL → SFT the 0.85 adapter on the kept rollouts.

## RAFT vs Dr.GRPO on one GPU
- **RAFT / Reinforce-Rejection** (chosen): off-policy within a round → vLLM-batchable; training is
  plain SFT (proven v24 spine). Run 1–3 rounds (regenerate after each SFT) to approximate on-policy.
- **Dr.GRPO (online)**: more sample-efficient, but needs on-policy gen every step → can't batch
  ahead; vLLM-colocate + 30B training on one GPU = memory contention + fragile. The sample-efficiency
  edge does **not** beat the generation bottleneck on a single GPU. Use only if you get multi-GPU.
- **GRPO-flavored middle ground (offline, free):** keep RAFT's structure but **weight kept rollouts
  by group difficulty** `w = max(0.1, 1 − pass_rate)` → upweight correct answers from HARD groups,
  downweight easy ones. This reproduces GRPO's group-relative-advantage intuition with binary
  rewards, at zero extra cost. (`MODE="curriculum"` in Notebook B.)

## Compute budget (single RTX 6000 Pro, 96 GB)
- vLLM gen: ~2000 prompts × G=8 × ~1k tok ≈ 16M tok. At ~3–5k tok/s aggregate (vLLM, MoE) ≈
  **~1–1.5 h/round**. (HF eager fallback ≈ 8–10 h — only if vLLM unavailable.)
- RAFT-SFT on kept corpus (a few thousand short rows): **~20–40 min**.
- Total per round ≈ **~1.5–2 h**. Far under the old multi-hour stuck runs.

## Key settings
- **Gen:** temperature 1.0, top_p 1.0, n=G(8), max_tokens ≈ 2048 (puzzles are short; long CoT is
  truncated/penalized anyway). `gpu_memory_utilization=0.90`, `max_lora_rank=32`, `max_model_len=8192`.
- **Reward:** binary, official `compare_answer` (numeric 1e-2 rel-tol, else exact). Binary is the
  most stable GRPO/RAFT signal (Unsloth/DAPO guidance).
- **RAFT filter:** drop all-wrong (no signal) and all-correct (too easy, `reinforce_rej`); keep the
  shortest correct rollout(s) from MIXED groups; optional perfect-CoT rescue on all-wrong.
- **SFT:** warm-start 0.85, LR 1e-5–5e-5 (gentle refine, below from-scratch 2e-4), 1 epoch, cosine,
  masked-gather CE, `max_grad_norm=1.0`, disk-safe save. Rank ≤ 32 (submission cap).
- **Profile the reward before a big run** (Unsloth): if pass@8 is ~0 everywhere the verifier/pool is
  wrong; if ~1 everywhere there is no learning signal. Want lots of MIXED groups.

## Files
- `post-training/nemo-v30-rollout-gen-vllm.ipynb` — Notebook A (gen + score → `rollouts.jsonl`).
- `post-training/nemo-v30b-raft-train.ipynb` — Notebook B (RAFT/curriculum SFT on the JSONL).

## Escalation path
If a multi-GPU box appears: true Dr.GRPO via NeMo-RL (see `nemo-v27-nemo-rl-grpo.ipynb`) with the
same `compare_answer` reward; otherwise iterate Notebook A→B 2–3 rounds (re-point A's adapter at the
newest B output each round).
