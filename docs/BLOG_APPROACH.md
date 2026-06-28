# Chasing 0.90 on a 30B Reasoning Model: Our Approach, Honestly

*The strategy we walked in with, the wall we hit at 0.85, every regression we earned, and
what we'd do differently. A story about a Kaggle competition where the model fought back.*

---

## The goal

Fine-tune NVIDIA's `Nemotron-3-Nano-30B-A3B` with a tiny rank-32 LoRA adapter and squeeze
the most accuracy out of ~7,500 deterministic reasoning puzzles — number bases, bit
tricks, ciphers, unit and gravity arithmetic, equation transforms, cryptarithms. One GPU.
LoRA-only submissions. Greedy decoding. Answers wrapped in `\boxed{}`.

We set ourselves a target of **0.90**. We got to **0.85** and learned exactly why the last
five points are hard — which is its own kind of result.

---

## Phase 0 — get *something* on the board

The first job in any competition is a real submission, end to end. We built the Unsloth +
LoRA + packaging pipeline, trained a broad adapter on a single teacher's chain-of-thought
data, packaged it, submitted — and landed **0.85**.

That number turned out to be sticky. The interesting part of this competition wasn't getting
to 0.85; it was discovering that almost everything we tried *to beat it* made it worse.

---

## The wall at 0.85

Here's the scoreboard of our own ideas, in the order we believed in them:

- **"Broad LoRA is overkill, let's be surgical."** We narrowed the targets, dropped
  `lm_head` and the MoE experts, added a smarter loss, used a curated subset, switched to a
  cosine schedule, ran two epochs. Result: **0.53.** We'd changed six things at once and
  couldn't even say which one hurt. *Lesson burned in: change one variable per run.*

- **"More data, more teachers, more reasoning diversity."** We merged three teachers' CoT —
  sometimes two different solutions to the same problem. Surely the model would learn a
  richer reasoning prior. Result: **0.67.** It learned to imitate a committee that disagreed
  with itself. *Consistency beats volume.*

- **"Fancy loss shaping — up-weight the hard tokens."** minmax/worst-token objectives,
  logprob shaping. All of them regressed against plain mean cross-entropy. *Boring won.*

- **"More LoRA capacity — different targets, higher rank."** Rank is capped at 32. The
  target set was already maxed at 888M. These tweaks moved the score by less than a point.
  *The architecture had already given us all the LoRA surface there was.*

The pattern across all of it: the **0.85 recipe was already near the ceiling of what SFT
config could do**, and the remaining gains were hiding somewhere else entirely.

---

## The three silent traps

Part of why progress was slow is that this model has three ways to make a fine-tune *look*
perfect while scoring like the untrained base. Each one cost us a confused day:

1. **The dead Mamba projection.** We were training an `out_proj` LoRA whose gradient was
   always zero — Unsloth's checkpointing silently kills it on Mamba's fused scan. Budget
   spent, nothing learned.
2. **The silent freeze.** A model-prep step quietly froze some of our LoRA parameters
   because their names didn't match a pattern. They sat in the optimizer doing nothing.
3. **The submission that discarded itself.** This was the worst. Our trainer saved the 128
   experts as *fused* tensors; the evaluator expected *128 separate* ones in a different
   namespace. Submit as-is and ~856M of trained weights simply don't load. The leaderboard
   reports your base-model score, and you have no idea why.

None of these show up in a loss curve. The takeaway became a habit: **verify the plumbing,
not just the metric.** A pre-flight cell now greedy-decodes three samples and asserts a real
loss before any long run; a repackage cell splits and renames every adapter before it's
zipped.

---

## The reinforcement-learning detour

Our written plan had a beautiful RL phase: Dr.GRPO with verifiable rewards. We even built a
deterministic solver suite (`RLVR/`) so the reward was free and exact — no reward model, no
reference model, minimal memory. We had Dr.GRPO, DAPO's clip-higher, and a CISPO variant
ready to go.

Then we measured generation speed: **about one token per second.**

Nemotron is mostly a Mamba model, and without its specialized recurrent cache, decoding
collapses to quadratic time. One real run generated **12 rollout groups in 12 hours**. RL
wants thousands. vLLM would fix it instantly, but the offline environment had no wheel for
this model. The honest call: **online RL wasn't viable here.**

Two things came out of that wall, and both were arguably more valuable than the RL would
have been:

- **A 30-second abort probe.** Every RL notebook now checks tok/s up front and *refuses* to
  start a multi-hour run if generation is too slow. The 12-hour lesson, encoded so it never
  repeats.
- **Decoupled RAFT.** If we can't generate fast *online*, generate offline: sample rollouts
  in a resumable pass, keep only the ones the solver verifies as correct, and train on them
  as plain SFT. Same spirit as RL — learn from your own correct reasoning — without the
  speed requirement. For problems the model fails every time, distill the solver's own
  perfect solution.

---

## Where the last five points really live

The most useful thing we learned is *where* 0.90 is hiding. 0.85 is an average that conceals
a split personality:

- The arithmetic-flavored categories (base conversion, units, gravity) are **nearly solved**
  — around 95%.
- The hard tail — long ciphers, gnarly bit-gate problems, symbolic equations, cryptarithms —
  sits at **60–75%**.

So the route to 0.90 isn't a better optimizer or a cleverer adapter. It's **boring,
category-specific data work**: pull the bottom two or three categories up by 15–20 points
with *consistent, verified* reasoning traces. We found cryptarithm was outright
data-starved — one source's 627 rows collapsed to 54 unique problems after dedup. So we
built two data engines to fill the gaps: deterministic synthetic generators that produce
provably-correct CoT (`syn_datagen/`), and a strong-teacher trace pipeline using GPT-5.5-pro
(`gpt5_trace_gen/`).

---

## What we'd tell ourselves on day one

1. **Get a clean 0.85 first, then change exactly one thing at a time.** Most of our pain was
   self-inflicted multi-variable changes.
2. **The MoE experts are the capacity. Use them.** The standard "LoRA the attention" advice
   is wrong for this model.
3. **Measure your bottleneck before you build for it.** A 30-second speed check would have
   redirected the whole post-training plan on day one instead of after a 12-hour run.
4. **Trust nothing silent.** Dead gradients, frozen params, and mismatched submission keys
   each produced a flawless-looking run and a base-model score. Verify, decode, repackage,
   confirm.
5. **The leaderboard is a category average.** Optimize the categories that are actually
   losing you points, not the ones already at 95%.

We didn't hit 0.90. But we mapped exactly what stands between 0.85 and 0.90, built the data
engines to close it, and left behind a pipeline that won't repeat any of the silent failures
that ate our first two weeks. For the engineering details — the exact bugs, the repackaging
code, the architecture reasoning — see [`BLOG_TECHNICAL.md`](BLOG_TECHNICAL.md).
