# Fine-Tuning a 30B Hybrid-Mamba MoE on One GPU: A Technical Post-Mortem

*How we LoRA-tuned NVIDIA's Nemotron-3-Nano-30B-A3B for a deterministic-reasoning
benchmark, the three submission-killing bugs nobody warns you about, and why the model
architecture — not the optimizer — decided what was possible.*

---

## The setup

The NVIDIA Nemotron Model Reasoning Challenge asks one thing: take
`NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, train a **rank-≤32 LoRA adapter**, and maximize
accuracy on ~7,500 deterministic puzzles — bit manipulation, base conversion, unit
conversion, gravitational-constant arithmetic, text ciphers, symbolic equation
transformation, and cryptarithms. Answers are graded by a `\boxed{}` extractor: exact
string match for text, ±10⁻² for numbers. Evaluation is **greedy** (`temperature=0`), via
vLLM, LoRA-only — no full fine-tune, no merged weights.

That constraint set is deceptively simple. It hides the fact that this base model is one of
the least conventional architectures you can put a LoRA on.

---

## 1. The architecture decides everything

Nemotron-3-Nano-30B-A3B is a **hybrid Mamba-Transformer Mixture-of-Experts**:

- **52 layers**: only **6** are GQA self-attention; the other **46 are Mamba-2** state-space
  layers.
- **128 routable experts**, top-6 active per token, plus shared expert(s) always active.
- No positional embeddings, no dropout, no linear bias, RMSNorm, squared-ReLU MoE
  activation, sigmoid-gated **aux-loss-free** router, untied embedding and `lm_head`.

Two implications fall straight out of that:

**(a) Most of the parameters are in places LoRA can't help much.** The 46 Mamba layers carry
the bulk of the sequence modeling, but the SSM scan parameters (`x_proj`, `dt_proj`, `A`,
`D`, `conv1d`) are not amenable to low-rank adaptation — the literature (SSM-PEFT
arXiv:2410.09016, MambaPEFT arXiv:2411.03855) is clear that the LoRA value in a Mamba block
lives in the **linear projections** (`in_proj` / `out_proj`), not the scan. So the effective
LoRA surface is: the 6 attention layers, the Mamba `in_proj`, the MoE expert projections,
and `lm_head`.

**(b) The MoE is where the capacity is.** With rank hard-capped at 32, you cannot buy
capacity by raising rank. You buy it by **widening the target set** — and the single biggest
source of width is the 128 routable experts. Targeting them takes trainable parameters from
a narrow ~30M up to **888M (~2.74%)**.

This is the crux of the whole competition, and it contradicts the conventional "LoRA the
attention only" wisdom: **here, the experts and `lm_head` are what move the score.**

---

## 2. The 0.85 recipe — and why "boring" won

The best leaderboard score, **0.85**, came from the least clever configuration we tried:

```text
LoRA      rank 32, alpha 64
targets   q/k/v/o, mamba.in_proj, moe.{gate,up,down} (shared + 128 routable), lm_head
trainable 888M  (~2.74%)
loss      plain mean-NLL, full sequence (no assistant-only masking)
optimizer LR 2e-4 linear, warmup 0, 1 epoch, eff. batch 16
          max_grad_norm = 1e9   # i.e. clipping intentionally OFF
          weight_decay 0, adam_beta2 0.95, max_length 8192
data      single-teacher (Tong) chain-of-thought, all 7,830 rows
```

Every "improvement" we layered on top of this regressed it:

| Change | Score | Lesson |
|---|---|---|
| The recipe above | **0.85** | — |
| Multi-teacher CoT (andy + Sonnet + Tong) | **0.67** | Mixing reasoning styles per problem creates contradictory supervision. |
| minmax / "blend" worst-token loss | regressed | Fancy loss shaping hurts clean SFT. |
| Narrow LoRA (no `lm_head`, no experts) + 3k subset + cosine LR + 2 epochs (all at once) | **0.53** | Six simultaneous changes = uninterpretable + worse. |

The 0.67 result is the most instructive. It dropped **8 points purely on data** — same
config, but the corpus mixed three teachers' chain-of-thought, sometimes two different
reasoning traces for the *same* problem. The model learned to imitate an averaged,
self-contradictory reasoner. **Consistency beats volume.** A single clean teacher, or
deterministically-generated solver-perfect CoT, beats a bigger noisy mixture.

---

## 3. Three bugs that silently zero your submission

These cost the most time and would not show up in training metrics — your loss curve looks
perfect and your leaderboard score is mysteriously the *base model's*.

### Bug 1 — `out_proj` LoRA is dead under Unsloth

You add `out_proj` to your target list, training runs, the adapter saves. But its gradient
is **zero**. Mamba's fused scan output doesn't carry `requires_grad`, and Unsloth's
checkpoint backward guard zeroes the gradient that would flow into `out_proj`'s LoRA. The
parameters exist, consume budget, and never learn. **`in_proj` is the only live Mamba LoRA
target** (`x_proj`/`dt_proj` are fused into it). Drop `out_proj` entirely.

### Bug 2 — silent freezing by `prepare_model_for_training`

`prepare_model_for_training` (and friends) walk the parameter tree and freeze anything whose
name doesn't match the expected dotted `.lora_A.` / `.lora_B.` pattern. On a module tree as
unusual as Nemotron's, some of your LoRA params get caught in that net and silently set to
`requires_grad=False`. Always run a post-wrap audit:

```python
lora = [(n, p) for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]
frozen = [n for n, p in lora if not p.requires_grad]
for n, p in lora:
    if not p.requires_grad: p.requires_grad_(True)   # re-enable
print("re-enabled", len(frozen), "silently-frozen LoRA params")
```

### Bug 3 — fused expert keys ≠ the evaluator's per-expert keys (the big one)

This is the one that silently discards ~856M of your trained weights. Unsloth trains and
saves the 128 routable experts as **fused** tensors. The Kaggle evaluator, however, loads a
transformers-native model that exposes **128 separate `nn.Linear` modules** —
`...experts.{j}.up_proj`, `...experts.{j}.down_proj`, etc. — in a different namespace
(`backbone`/`model`, with `gate_proj+x_proj → in_proj` merges). If you submit the fused
keys, **none of the expert LoRA binds**, and you score the base model on everything the
experts learned.

Every submission must therefore be **repackaged**: split each fused expert tensor into 128
per-expert tensors, rename keys to the eval namespace, strip the dead `out_proj` keys, and
rewrite `adapter_config.json`'s `target_modules`. The split handles the three layouts
Unsloth can emit:

```python
if v.dim() == 3 and v.shape[0] == N_EXPERTS:            # [E, r, d]
    per = [v[j].contiguous() for j in range(N_EXPERTS)]
elif v.dim() == 2 and v.shape[0] % N_EXPERTS == 0:       # [E*r, d]
    per = list(v.reshape(N_EXPERTS, -1, v.shape[1]))
elif v.dim() == 2 and v.shape[1] % N_EXPERTS == 0:       # [d, E*r]
    per = list(v.reshape(v.shape[0], N_EXPERTS, -1).permute(1, 0, 2))
```

**Verify adapter keys against the eval model's module names before trusting any leaderboard
number.** A namespace mismatch is indistinguishable from "my fine-tune did nothing."

---

## 4. The zero-loss SFT trap

Early runs logged `loss = 0.0` from step one. The cause: TRL's `SFTTrainer` label handling
interacted badly with the chat template / masking on this model, producing all-`-100`
label rows — nothing to learn from, zero loss, a perfectly flat curve that *looks* like
instant convergence. The fix was to drop to a vanilla `Trainer` with explicit label
construction, plus a **pre-flight cell** that runs before every multi-hour job:

1. Greedy-decode 3 training samples.
2. Assert each emits a `\boxed{}`.
3. Assert the brace-balanced extractor recovers the answer.
4. Assert pre-flight loss is real (> 0.3, not 0.0).

That one cell paid for itself many times over. The general principle: **on an unusual
architecture, never trust the trainer's defaults — verify the loss is real before you spend
GPU-hours on it.**

---

## 5. Why reinforcement learning didn't happen

The plan (see `../work/WINNING_PLAN.md`) had a clean RL phase: Dr.GRPO with verifiable rewards.
No reward model, no reference model (`beta=0`) — the `RLVR/` solvers deterministically grade
each rollout's `\boxed{}` answer. Memory-light, algorithmically sound. We had Dr.GRPO, DAPO
clip-higher, and even a CISPO (MiniMax M1) variant ready.

It died on **generation speed.**

Nemotron-3 is mostly Mamba. Efficient autoregressive decoding requires a recurrent state
cache (`NemotronHHybridDynamicCache`). Under HuggingFace `generate` without it, Mamba decode
degrades to **O(n²)** in sequence length. Measured throughput: **~1 token/second.** One real
RL run produced **12 rollout groups in 12 hours.** RL needs thousands of rollouts; at
~1 tok/s that is weeks. vLLM serves this at ~600 tok/s — two-and-a-half orders of magnitude
faster — but the offline Kaggle training environment has no vLLM wheel for this model.

The honest engineering conclusion: **online RL is not viable on the available hardware/stack
for this model.** Two adaptations followed.

**(a) A gen-speed probe that aborts.** Every RL notebook now opens with a 30-second check —
generate 200 tokens, measure tok/s, `assert tok/s ≥ MIN_TOK_S`. If generation is slow, it
**aborts before** committing to a multi-hour run, with a message pointing to the vLLM-wheel
fix or to SFT instead. This guard is the direct lesson of the 12-hour/12-prompt run.

**(b) Decoupled RAFT.** Instead of online RL, split generation from training: generate
rollouts in a resumable HF-only pass (`for_inference`, `num_return_sequences`, stop early
on a balanced `\boxed{}`, save every row), filter to the solver-verified-correct ones, and
run those as ordinary SFT. Same "learn from your own correct reasoning" effect as RAFT,
without needing fast online rollouts. For the hardest prompts (policy wrong on *all*
rollouts), a PERFECT_RESCUE step distills the deterministic solver's own perfect CoT.

---

## 6. Where the remaining points actually are

0.85 overall is an average over very uneven categories. Measured roughly:

- **~95%** on deterministic-arithmetic families: numeral-base conversion, gravity, unit
  conversion. These are nearly solved.
- **~60–75%** on the hard tail: long-phrase ciphers, complex multi-gate bit manipulation,
  symbolic equation transformation, and cryptarithms.

The path to 0.90 is **not** more LoRA targets or a higher rank (rank is capped; targets are
maxed; both give ≪1pp). It is **per-category data work**: pull the bottom 2–3 categories up
15–20pp via targeted, *consistent*, verified CoT. Cryptarithm in particular was found to be
data-starved — one source's 627 "rows" deduplicated to **54 unique problems** (92%
duplicates). That gap is filled by deterministic synthetic generation (`syn_datagen/`
reasoners that forward-generate examples from a rule, run the rule, and use its boxed output
as ground truth) and by strong-teacher trace generation (GPT-5.5-pro via the
`gpt5_trace_gen/` pipeline).

---

## What this competition actually taught

1. **Architecture > optimizer.** Whether RL was even possible, which modules LoRA could
   touch, and how to package a submission were all decided by the hybrid-Mamba-MoE design,
   not by hyperparameters.
2. **Capacity in the right place.** Targeting the 128 MoE experts + `lm_head` (not just
   attention) is what earned 0.85 — the opposite of the usual LoRA advice.
3. **Verify the plumbing, not just the loss.** Three different silent failures (dead
   `out_proj`, silent freezing, fused-key mismatch) each produced perfect-looking training
   and a base-model score.
4. **Consistency beats volume.** Single-teacher / solver-perfect CoT beat a bigger
   multi-teacher mixture by 8 points.
5. **Measure the thing that limits you.** A 30-second gen-speed probe would have saved a
   12-hour run — and reframed the entire post-training strategy from "online RL" to
   "decoupled RAFT."

Companion read: [`BLOG_APPROACH.md`](BLOG_APPROACH.md) for the narrative version.
