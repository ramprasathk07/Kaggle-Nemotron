# Kaggle Nemotron — Winning Plan (Deep RL Stack)

Generated: 2026-05-13. Deadline target: ~2026-06-10.

## Method Digest

### 1. GRPO (baseline — already in `post-training/nemo-v9-1-Grpo.ipynb`)
- Group of G completions per prompt. Advantage = `(R_i − mean(R)) / std(R)`. Clipped PPO surrogate + KL beta.
- Two known biases (Sea AI Lab):
  - Response-length norm `1/|o_i|` → favors short correct, long wrong.
  - Std norm → over-weights easy/hard questions inverse to difficulty variance.

### 2. Dr. GRPO ("Done Right")
- Drop both norms. Advantage = `R_i − mean(R)` only. Use constant max-len norm or sum-loss.
- Oat-Zero-7B: AIME'24 43.3%, MATH500 80.0%. 27h on 8×A100.
- Recipe: LR=1e-6 constant, β(KL)=0, temp=1.0, group=8, max_completion=3000, clip=0.2, grad-clip=1.0.
- Hard constraint: model-template match. Nemotron → R1-style `<think>` template (already used).

### 3. DAPO (ByteDance, Qwen2.5-32B → AIME 50)
Four upgrades on top of GRPO:
- **Clip-Higher**: asymmetric clip ε_low=0.2, ε_high=0.28 → prevents entropy collapse.
- **Dynamic Sampling**: filter groups where reward all-1 or all-0. Resample until batch full.
- **Token-level loss** (not seq-level) → long-CoT stable.
- **Overlong reward shaping**: soft penalty between L_max and L_max+overflow, hard cutoff after.

### 4. RAFT / RAFT++ / Reinforce-Rej (RLHFlow)
- RAFT: sample N, keep top-k by reward, SFT on those. Off-policy, batch-friendly, cheap.
- RAFT++: add PPO importance-ratio clipping → narrows gap to GRPO.
- Reinforce-Rej: REINFORCE + drop all-correct + all-wrong groups ≈ GRPO logic, simpler.
- Verdict: RAFT++ converges fast early, plateaus (entropy collapse). Use as warm-start before GRPO.

### 5. RLOO (REINFORCE Leave-One-Out)
- Advantage_i = R_i − mean(R_{j≠i}). Same math as GRPO without clip + std-norm.
- Often beats PPO/DPO/RAFT. Equivalent to Dr.GRPO with N/(N−1) correction.

### 6. REINFORCE++
- REINFORCE + per-token KL + reward normalization + PPO clip. Critic-free. Stable under noisy reward.
- Cheaper than GRPO (one rollout per prompt). Useful when group sampling too expensive.

### 7. EGGROLL (arXiv 2511.16652, Nov 2025 — newest)
- Gradient-free ES. Each member i: `W_i = W + σ · A_i B_iᵀ`, A_i ∈ R^(m×r), B_i ∈ R^(n×r), r=1.
- Aggregate: `ΔW = (1/Nσ) · Σ (F_i − F̄) · A_i B_iᵀ`. Sum of N rank-1 = high-rank update.
- Population N up to 10^6. Throughput 91% of pure-inference on H100.
- Beats GRPO same-base on GSM8K with RWKV 1.5B. Forward-only, no backprop.
- Code: github.com/ESHyperscale/HyperscaleES (JAX). No PyTorch ref yet.

---

## Strategic Verdict

Constraints:
- Submission = LoRA adapter (rank ≤32) only. Full FT useless — can't ship.
- Base = Nemotron-3-Nano-30B-A3B (MoE+Mamba hybrid, BF16). Mamba layers need `x_proj`, `dt_proj` LoRA.
- ~7500 problems with answers. Reward = boxed-match (already coded). Verifiable RL = ideal fit.
- Kaggle GPU budget tight. ~2-4 wk runway.

**Recommend**: Dr.GRPO + DAPO upgrades on LoRA. Skip EGGROLL primary (JAX-only, pop=10^6 needs B200 cluster). RAFT as offline data-gen booster. EGGROLL = wild-card Phase 5.

---

## Phased Plan (4 weeks)

### Phase 0 — Cleanup + Baseline (Days 1–2)
- Confirm `data/v5_balanced_cot.py` produces clean `<think>…</think>\boxed{ans}` traces.
- `merged_cot_final.csv` deduped, CoT length 200–4000 chars.
- SFT v9 LoRA r=32 α=64, RSLoRA, target attn+MLP+Mamba projs (no `lm_head`).
- Lock submission script. Baseline leaderboard score = **S0**.

### Phase 1 — Dr.GRPO Migration (Days 3–6)
New notebook `post-training/nemo-v11-DrGrpo.ipynb`, clone v9-1, swap config:

```python
GRPOConfig(
    learning_rate=1e-6,            # was 5e-6 — drop
    beta=0.0,                      # was 0.04 — drop KL (rule reward, no drift risk)
    epsilon=0.2,                   # symmetric clip baseline
    num_generations=8,             # was 4 — Dr.GRPO recipe minimum
    temperature=1.0,               # was 0.8 — raise exploration
    top_p=1.0, top_k=0,
    max_prompt_length=3072,
    max_completion_length=3000,
    loss_type="dr_grpo",           # trl >=0.16 supports
    scale_rewards=False,           # disable std norm
    mask_truncated_completions=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    max_grad_norm=1.0,             # was 0.1 — too tight
    optim="adamw_8bit",
    bf16=True,
    gradient_checkpointing=True,
)
```

If TRL version lacks `loss_type="dr_grpo"` → monkey-patch loss: divide by `max_completion_length` constant instead of `|o_i|`, remove `std(R)` divide.

Reward = `format_reward + accuracy_reward` (already in notebook). Add DAPO overlong soft penalty: `-min(0, (L_max+512−|o|)/512)`. Cosine length penalty hurts — skip.

### Phase 2 — DAPO Upgrades (Days 7–10)
Stack on Dr.GRPO:
1. **Clip-Higher**: patch trl `GRPOTrainer._compute_loss` → ε_low=0.2, ε_high=0.28. Verl has flag; trl needs ~10 LOC patch.
2. **Dynamic sampling**: oversample 2× then drop groups where `mean(reward)==0` or `==max`. Refill batch. Infinite stream loader.
3. **Token-level loss**: sum log-probs × advantage over tokens, no per-sequence mean.
4. **Overlong shaping** (above).

1 epoch on 7500 problems. Validate every 50 steps on 200-problem held-out. Target **S1 > S0 + 8pp**.

### Phase 3 — Data Engine via RAFT (Parallel, Days 3–10)
Offline — runs while GRPO trains on separate GPU/box:
- Sample N=16 completions per problem from current best LoRA, temp=1.0, top_p=0.95.
- Keep top-1 (or top-k=2) by reward.
- Build `data/raft_distilled.csv`.
- Re-run `sft/nemo-v9-svd-lora.ipynb` on `merged_cot_final.csv ∪ raft_distilled.csv`.
- Best-of-N → distill. Cheap, complements GRPO.
- Reinforce-Rej variant: keep both top + bottom for contrastive — DPO fallback.

### Phase 4 — Curriculum + Iteration (Days 11–14)
- Bucket problems by current model's pass@8 rate.
- **Hard set** (pass@8 < 0.25): rerun GRPO with N=16 generations, more samples.
- **Easy set** (pass@8 > 0.75): drop from RL, keep in SFT.
- Avoid language-mix collapse: always keep ≥20% easy/medium problems mixed in batch.
- Iterate: SFT → Dr.GRPO+DAPO → RAFT-distill → SFT-refresh → Dr.GRPO again.

### Phase 5 — Wild Card (Days 15–17, optional)
One of:
- **EGGROLL-LoRA hack**: perturb only LoRA-A, LoRA-B matrices (rank ≤32). Pop N=128, σ=0.01, antithetic. Fitness = combined_reward. Single H100, ~6h. Feasible since LoRA params ≪ full model. If beats Dr.GRPO best → ship.
- **REINFORCE++**: cheaper single-rollout RL. Use when budget tight late.

### Phase 6 — Inference + Ensembling (Days 18–20)
- vLLM serve: temp=0.6, top_p=0.95, max_tokens=4096, stop=`</think>` for budget kills.
- Try Maj@8 if Kaggle eval allows (check rules).
- Pick best 2 LoRA checkpoints (different seeds), submit both, keep best.

---

## Code Action Items

| File | Change |
|------|--------|
| `post-training/nemo-v11-DrGrpo.ipynb` | New: clone v9-1, swap to Dr.GRPO recipe |
| `post-training/dapo_patch.py` | New: monkey-patch trl GRPOTrainer for clip-higher + token-level + dynamic sampling |
| `data/raft_collect.py` | New: rollout N=16 per problem, top-k filter, write CSV |
| `data/curriculum_bucket.py` | New: bucket problems by pass@8 |
| `sft/nemo-v9-svd-lora.ipynb` | Re-run on merged + RAFT-distilled set |
| `post-training/eggroll_lora.py` | Phase 5 only: rank-1 perturbation over LoRA-A/B, fitness = reward |

---

## Risk Register

- **Nemotron MoE+Mamba quirks**: x_proj/dt_proj LoRA may interact badly with RL gradients → if loss diverges, exclude Mamba targets from RL phase (keep in SFT).
- **TRL ≤0.15 lacks Dr.GRPO flag**: pin `trl>=0.16` in offline wheels. Verify on Kaggle (no internet).
- **KL=0 + lm_head excluded** → safe vs format drift, confirmed in v9.
- **Length explosion**: train at target len from start. No progressive 8k→16k→24k.
- **Reward hacking**: Nemotron may emit fake `\boxed{}` early. Require `<think>...</think>` before `\boxed{}` in format reward.
- **vLLM merge with LoRA + Mamba**: known unstable. Test submission.zip end-to-end early.

---

## Sources
- Dr. GRPO: https://arxiv.org/html/2503.20783v2
- DAPO: https://arxiv.org/html/2503.14476v1
- EGGROLL: https://arxiv.org/abs/2511.16652
- EGGROLL site: https://eshyperscale.github.io/
- RAFT++ / Reinforce-Rej: https://arxiv.org/html/2504.11343v1
- REINFORCE++: https://arxiv.org/pdf/2501.03262v6
- Kaggle DeepSeek-R1 math recipe: https://hav4ik.github.io/improving-deepseek-r1/
- RLHF Book policy gradients: https://rlhfbook.com/c/06-policy-gradients
- RLHFlow Minimal-RL: https://github.com/rlhflow/minimal-rl
- DAPO verl recipe: https://verl.readthedocs.io/en/latest/algo/dapo.html
