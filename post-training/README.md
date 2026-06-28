# `post-training/` — RL & RAFT notebooks

Reinforcement-learning and rejection-sampling (RAFT) experiments on top of the 0.85 SFT
adapter. Read the root [`README.md`](../README.md) first.

## The headline finding: RL is gated by generation speed, not by the algorithm

All RL here uses **verifiable rewards** — the `../RLVR/` solvers deterministically check
each rollout's `\boxed{}` answer (no neural reward model, no reference model with `beta=0`).
That part is cheap. The blocker is **rollout generation**:

> Nemotron-3 is a hybrid Mamba model. Under HuggingFace `generate` without
> `NemotronHHybridDynamicCache`, Mamba decode is **O(n²)** → measured **~1 token/s** →
> a single RL step of a few prompts × a few rollouts takes hours. One real run produced
> **12 rollout groups in 12 hours.** vLLM does this at ~600 tok/s, but the offline Kaggle
> environment has no vLLM wheel for this model.

**Consequence:** online RL (GRPO/Dr.GRPO/CISPO) is impractical on the available hardware.
The feasible post-training path is **decoupled RAFT** — generate rollouts in a separate
pass, filter to correct ones, and run them as ordinary SFT.

Every RL notebook now opens with a **30-second gen-speed probe** that generates 200 tokens,
measures tok/s, and **asserts `tok/s ≥ MIN_TOK_S` — aborting before any multi-hour run** if
generation is too slow. This is the single most important no-stuck guard in the folder.

## Algorithm notes (what we'd use if gen were fast)

- **Dr.GRPO** — advantage = `reward − group_mean` (no std-norm → removes difficulty bias),
  token-level loss, `beta=0` (no reference model → ~50% less memory). `loss_type="dr_grpo"`
  needs TRL ≥0.16.
- **DAPO add-ons** — asymmetric clip-higher (`eps_low=0.2`, `eps_high=0.28`) for entropy
  preservation, dynamic sampling (drop all-0 / all-max groups), overlong soft penalty,
  `mask_truncated_completions=True`.
- **CISPO** (MiniMax M1) — clips the importance-sampling *weight*, not the token update:
  `loss = −Σ_t sg(clip(ρ_t, 1−ε_lo, 1+ε_hi))·Â·logπ_t`. At a single on-policy update
  (`ratio=1`) it reduces exactly to Dr.GRPO — which is the minimal-VRAM default.

## Reward-function discipline

- Signature must accept `prompts, completions, completion_ids, trainer_state, log_extra,
  log_metric, **kwargs`. Conversational dataset → `completions` is `list[list[dict]]`.
- **Format reward must require `<think>…</think>` *before* `\boxed{}`** to block reward
  hacking (model emitting a fake early `\boxed{}`).
- Log: `reward`, `completions/clipped_ratio`, `completions/mean_length`,
  `frac_reward_zero_std` (1.0 = all-same reward in a group = no learning signal → skip it).

## Notebook index

| Notebook | Idea | Outcome |
|---|---|---|
| `nemo-v9-1-Grpo.ipynb` | Baseline GRPO (TRL) | heavy, slow |
| `nemo-v10-grpo-traincsv.ipynb` | GRPO directly on `train.csv` | slow gen |
| `nemo-v11-DrGrpo.ipynb` | Dr.GRPO migration (β=0, scale_rewards=False, dr_grpo loss) | correct config, gen-bound |
| `nemo-v12-DAPO.ipynb` | DAPO upgrades on Dr.GRPO (clip-higher, dynamic sampling, overlong shaping) | correct config, gen-bound |
| `nemo-v13-drgrpo-rtx6000.ipynb` | Single-GPU RTX 6000 Dr.GRPO | infra |
| `nemo-v15-grpo.ipynb` | GRPO variant | gen-bound |
| `nemo-v18-raft-reinforce-rej.ipynb` | **RAFT / Reinforce-rejection** spine (warm-start 0.85, Mamba-safe per-prompt sampling, verify→filter→SFT) | the viable post-training pattern |
| `nemo-v19-sdpg-grpo.ipynb` | Dr.GRPO RLVR + optional SDPG self-distillation on synthetic | RL on synthetic |
| `nemo-v20-raft-synth.ipynb` | RAFT on synthetic problems | data engine |
| `nemo-v25-raft-synth-perfect.ipynb` | v18 RAFT spine + **solver-perfect synthetic CoT** + PERFECT_RESCUE (distill solver CoT on all-wrong groups) | the strongest RAFT design |
| `nemo-v27-nemo-rl-grpo.ipynb` | NVIDIA NeMo-RL GRPO (cluster-style) | off-Kaggle reference only |
| `nemo-v29-router-moe-study.ipynb` | Router/MoE study + NVIDIA-aligned SFT; expert-split repackage save | study + long-run SFT |
| `nemo-v30-rollout-gen-hf.ipynb` | **Decoupled rollout generation** (HF-only, no vLLM): `for_inference` + `num_return_sequences` + stop-on-`\boxed{}` + resumable | the gen half of decoupled RAFT |
| `nemo-v30b-raft-train.ipynb` | Train on the filtered rollouts from v30 | the train half of decoupled RAFT |

## Decoupled RAFT (the recommended post-training loop)

Because online gen is too slow, split it:

1. **Generate** (`nemo-v30-rollout-gen-hf.ipynb`): warm-start 0.85, sample N rollouts per
   prompt with `FastLanguageModel.for_inference`, `use_cache=True`, stop early once a
   balanced `\boxed{}` is emitted, **save every row** (resumable — survives interruption).
2. **Filter**: keep rollouts whose `\boxed{}` answer the RLVR solver verifies as correct
   (and, optionally, PERFECT_RESCUE — distill the solver's own perfect CoT on prompts the
   policy gets wrong on *all* rollouts).
3. **Train** (`nemo-v30b-raft-train.ipynb`): run the filtered correct rollouts as ordinary
   SFT, then repackage (split fused experts → per-expert keys) and submit.

Single-GPU budget (RTX 6000 Pro): `SYN_PER_CAT≈150`, `N_ROLLOUTS=4`, `GEN_MAX_NEW=1024`,
`RAFT_LR=2e-5` → ~1h/round.

## Verifiable-reward solvers — `../RLVR/`

`from RLVR.solver import solve` → `solve(prompt)` returns a `SolverResult(answer, category,
confidence, reasoning, verified)` with a `.solved` property. Deterministic CPU verifiers
cover bit_manipulation, cipher, numeral, unit_conversion, gravity, and equation. These are
the reward oracle for RL and the correctness filter for RAFT.
