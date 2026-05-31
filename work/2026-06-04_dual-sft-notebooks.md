# Work Log — 2026-06-04 — Dual SFT Notebooks (Standard + Advanced)

Built two new well-structured SFT notebooks from scratch following `sft/nvidia-nemotron-v7-5.ipynb` load/save patterns + every hardening lesson from `work/2026-05-29` and `work/2026-05-31`.

Both notebooks pass `nbformat.validate` + `ast.parse` on every code cell. 24 cells each.

---

## Files created

### `nvidia-nemotron-sft-standard.ipynb`
Standard mean-NLL SFT. The clean baseline. Use this for the first attempt + as
a reference against which the advanced notebook is compared.

### `nvidia-nemotron-sft-minmax-logprob.ipynb`
Advanced SFT with selectable loss objectives that target hard / low-confidence
tokens. Use this AFTER getting a working baseline from the standard notebook.

---

## Research synthesis (driving the loss menu)

Web search across GitHub + Kaggle + arXiv 2025/2026:

1. **Tong Huikang Progress Prize submission** ([tonghuikang/nemotron](https://github.com/tonghuikang/nemotron))
   — Won the NVIDIA Nemotron Reasoning Challenge Progress Prize. Their
   `loss_config.py` exposes 5 loss classes:
   - `CrossEntropyLossConfig` (vanilla)
   - `CrossEntropyWithWeightingLossConfig` with `branch_weight = min(1, |logp| / branch_logprob)` + epoch-0 `first_cutoff_weight`
   - `ImportanceSamplingLossConfig` (off-policy KL)
   - `PPOLossConfig`, `CISPOLossConfig` (clip variants)
   - `DROLossConfig` (KL² penalty)

   They train on Tinker backend (not HF SFTTrainer) with stratified-by-category
   batching, per-token mask weights, batch=64, max_seq=8192, lr 2e-4
   StepLinearDecay, beta1=0.9 beta2=0.95, weight_decay=0, LoRA rank=32.

2. **Critical Token Fine-Tuning** (Wang 2025, [arXiv:2510.10974](https://arxiv.org/abs/2510.10974))
   — Updates only ~12% of tokens identified as "functionally indispensable" via
   counterfactual perturbations. Consistently beats standard SFT on 11 math
   benchmarks across Qwen, OLMo, LLaMA. Specific gains not in abstract but
   "consistently outperforms".

3. **Beyond Log Likelihood** (Wu 2025, [arXiv:2510.00526](https://arxiv.org/abs/2510.00526))
   — Probability-based objectives across model capability continuum. Min-probability
   objective is a "prior-leaning" special case implemented via RL-style downweighting.

4. **SFTKey** (anon 2025, [arXiv:2512.21017](https://arxiv.org/abs/2512.21017))
   — Two-stage: stage 1 vanilla SFT, stage 2 fine-tune only the "Key portion"
   (answer-bearing tokens). +5pp avg gain over uniform NLL.

5. **SFT-GO** — grouped worst-group loss; informative groups (TFIDF, key
   answer tokens) trained until floor reached.

Common thread: **uniform-NLL SFT under-trains the hardest tokens**, which on
verifiable-reasoning benchmarks are exactly the tokens that determine score.

---

## Loss menu (advanced notebook)

`LOSS_MODE` switches between:

| mode | formula | when |
|---|---|---|
| `mean` | `mean(NLL)` over assistant tokens | safe baseline / warm-up |
| `minmax` | `mean_batch( max_seq(NLL) )` | pure worst-token; very sparse |
| `topk_min` | `mean_batch( mean(topk(NLL, K)) )` | smoothed worst-K, recommended advanced |
| `blend` | `alpha*mean + (1-alpha)*topk_min` | RECOMMENDED default — mean density + worst-K focus |
| `branch` | `sum(NLL * w_branch) / sum(w_branch)`, `w_branch = min(1, |logp|/branch_logprob)` | Tong Huikang trick |

All modes compute per-token NLL via fused `F.cross_entropy(reduction='none')` —
no fp32 `(seq × 131072)` `log_softmax` tensor. Same memory-safe path as `mean`.

`WARMUP_MEAN_STEPS` lets the trainer run vanilla mean-NLL for N steps before
switching to the advanced mode. Useful when starting cold — early worst tokens
are huge and noisy.

---

## Shared design (both notebooks)

| Aspect | Choice | Why |
|---|---|---|
| Model load | Unsloth `FastLanguageModel.from_pretrained` (kagglehub) | proven on v7-5 |
| LoRA | r=32 alpha=64 dropout=0.0 RSLoRA+**DoRA** | r=32 = competition cap; DoRA = +2-4pp typical |
| Targets | regex on `self_attn.{q,k,v,o}_proj` + `mamba.{in,out,x,dt,gate}_proj` + `shared_experts.{gate,up,down}_proj` | per CLAUDE.md priority; gate added for postprocess Mamba fuse |
| **Audit assert** | `50M ≤ trainable ≤ 400M` | catches "regex matched 0 → suffix fallback attached to 128 routable experts" failure (had this bug, wasted a week) |
| Trainer | Vanilla TRL `SFTTrainer` (purge unsloth from sys.modules + meta_path) | Unsloth fused loss double-projects lm_head on Nemotron-H |
| Loss | fused `F.cross_entropy(reduction='none')` | no fp32 full-vocab `log_softmax` (avoids ~4.3 GB/step spike that OOM'd 96 GB GPU late in long runs) |
| Padding | right (SFT convention) | matches collator |
| Chat format | `[system, user, assistant]` with `<think>...</think>\boxed{}` | matches eval; assistant-only label masking via prefix-render diff |
| Stratified batching | Pre-computed index order grouped by puzzle type | Tong Huikang `train_sft.py`; balanced category mix per effective batch |
| Knobs | `SMOKE_TEST` first (64 rows, 8 steps, ~10 min), then `SUBSET_N=3000` + 1-2 epochs + `TRAIN_MAX_LEN=4096` | quota-safe |
| Save | 3-tier: `trainer.model.save_pretrained` → `trainer.save_model` → copy from `checkpoint-N/` | survives crash/OOM/timeout |
| Recovery | `try/except` around `trainer.train()`, save in `finally`, re-raise after | adapter always on disk |
| Package | `WORKING = /kaggle/working if exists else OUTPUT_ROOT` then write `submission.zip` to WORKING | Kaggle eval reads working-root |
| Post-process | Subprocess call to `tools/postprocess_adapter.py` with `POSTPROC_BOOST` env var | vLLM-clean module names + optional top-50% SV-amplify (Tong Huikang trick) |
| Audit cell | first/last 5 trainable tensor names + total param count | sanity at training time, not eval time |

---

## Workflow

```
Standard notebook (SMOKE):
  1. Run all cells with SMOKE_TEST=1 (~10 min)
  2. Watch: SFTTrainer module = trl... (not unsloth), trainable count in [50M,400M],
            length p50/p90/p99, PEAK VRAM headroom, HAS_BOXED on sanity
  3. SMOKE_TEST=0, rerun from paths cell, real run (~3-6 h depending on SUBSET_N + epochs)
  4. Upload adapter -> submit

Advanced notebook (after baseline lands):
  1. Same smoke flow with LOSS_MODE='mean' (verify scaffold)
  2. Smoke with LOSS_MODE='blend' or 'topk_min' (verify selected loss runs)
  3. Real run with LOSS_MODE='blend' alpha=0.5 TOPK_MIN=16
  4. Compare vs standard adapter
  5. A/B postprocess: POSTPROC_BOOST=1.0 vs 1.12, pick winner
```

---

## Open questions

- DoRA + Unsloth model load combo not battle-tested by us — first smoke run
  will tell. Fallback: drop `use_dora=True` from LoraConfig and rerun.
- `branch_logprob=1.0` is a guess. Tong Huikang notebook ran multiple values;
  worth sweeping `[0.5, 1.0, 2.0]` once a baseline is locked.
- `lm_head` LoRA still excluded per CLAUDE.md (destabilizing). Reference 87% adapter
  used it. Cheap test once baseline lands: include with r=4 / alpha=4.
- Tong Huikang's `CategoryStratifiedSampler` shuffles category labels per-epoch.
  Our `build_stratified_index_order` does it once — close enough for SFT, not RL.

---

## Sources

- [tonghuikang/nemotron](https://github.com/tonghuikang/nemotron) — Progress Prize winning submission
- [Wang 2025 — Selective Critical Token Fine-Tuning](https://arxiv.org/abs/2510.10974)
- [Wu 2025 — Beyond Log Likelihood](https://arxiv.org/abs/2510.00526)
- [Anon 2025 — SFTKey](https://arxiv.org/abs/2512.21017)
- [Nemotron 3 Nano Technical Report](https://arxiv.org/abs/2512.20848)
- [acecloud.ai — Nemotron-3-Nano Multi-GPU Fine-Tuning](https://acecloud.ai/blog/nemotron-3-nano-multi-gpu-fine-tuning/)
- [Unsloth Discussion #3810 — Trouble fine-tuning Nemotron 3 Nano](https://github.com/unslothai/unsloth/discussions/3810)
