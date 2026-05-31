# SVD-Amplify Notebook — Analysis & Improvements for SFT / MinMax

Source: `amplifying-top-50-singular-values.ipynb` (huikang nemotron adapter
post-processor). 54 cells, post-training adapter surgery. NO training.

---

## What this notebook actually does (3 phases)

### Phase 1: Config alignment (cells 4-11)
Compares own adapter config to a reference `submission.zip` config; force-overrides
`target_modules` to the canonical 9 names the eval harness expects:
```python
["k_proj","o_proj","in_proj","q_proj","up_proj","v_proj","down_proj","out_proj","lm_head"]
```
**Key takeaway:** at submission time PEFT only loads adapters whose
`target_modules` strings match the modules vLLM exposes. Misnamed targets =
silent skip = no adapter effect.

### Phase 2: Key renaming + expert unfusing (cells 13, 24)
- `base_model.model.model` → `base_model.model.backbone` (vLLM-side naming).
- Fused MoE experts `experts.w1`/`experts.w2` (shape `(num_experts, ...)`) **unfused** into per-expert `experts.{i}.up_proj` / `experts.{i}.down_proj` LoRA pairs (broadcasts shared-dim 1 with `.expand().contiguous()`).
- Skips empty `experts.w3` placeholders.

### Phase 3: Mamba SVD-merge (cell 24, the core trick)
Mamba layer has separate trained `gate_proj` (rank 32) + `x_proj` (rank 32) LoRAs.
vLLM expects a **fused `in_proj`** (out_dim = gate_dim + x_dim).

Naive concat → rank 64, exceeds competition's r=32 cap. So:
```python
A_cat   = cat([gate_A, x_A], dim=0)        # (64, in_dim)
B_block = block_diag([gate_B, x_B])         # (out_dim, 64)
Q_B,R_B = qr(B_block);  Q_A,R_A = qr(A_cat.T)
core    = R_B @ R_A.T                        # 64x64
U,S,Vh  = svd(core)

# THE TRICK: amplify top-50% singular values by 1.12x
S_importance = S[:32].clone()
S_importance[:16] *= 1.12

new_B = (Q_B @ U[:, :32]) * S_importance
new_A = Vh[:32, :] @ Q_A.T
```
QR-then-SVD = exact best rank-32 of the rank-64 fused adapter
(Eckart-Young). Then **1.12× boost** to top-16 singular values shifts mass
to "foundational" directions (boxed format, basic arithmetic) at the cost of
the bottom-16 fine-detail directions.

Author's claim (in code comments): top SV's encode foundational rules
(numerals, formatting). Boost them → more confident greedy decoding.

---

## Improvements transferable to current SFT / MinMax pipeline

### A. Use this notebook AS-IS (post-processing, zero retrain) — DO FIRST
RAFT / MinMax SFT adapters likely fail eval not because reasoning is wrong
but because **module names don't match** what vLLM loads. Run this notebook
against your adapter:
- Confirms target_modules align.
- Renames `model` → `backbone`.
- Unfuses experts (if you accidentally trained on fused expert tensors).
- Fuses split Mamba LoRAs (`gate_proj`+`x_proj` → `in_proj`).

**Could be a +5-10pp leaderboard jump on the SAME adapter just from correct loading.**

### B. Train DIRECTLY on the final vLLM names (skip Phase 2/3 entirely)
Our LoRA target regex hits `mamba.{in,out,x,dt}_proj` separately. If vLLM
expects fused `in_proj`, train on the **fused module name** directly:

```python
target_regex = (
    r".*("
    r"self_attn\.(q|k|v|o)_proj"
    r"|mamba\.in_proj"                    # FUSED, not gate/x split
    r"|mamba\.out_proj"
    r"|shared_experts\.(gate|up|down)_proj"
    r")$"
)
```
**Verify on Kaggle:** print `model.named_modules()` filter for `mamba` to
confirm whether the loaded Nemotron module name is `in_proj` (fused) or
`gate_proj`+`x_proj` (split). If fused, skip the SVD-merge surgery; if split,
keep our current targeting but **run this notebook post-train**.

### C. Apply SV-amplification AT TRAIN TIME (better than post-hoc)
Author boosts top-50% SVs by 1.12 *after* training. Equivalent effect baked
into training:
- **DoRA** (Weight-Decomposed LoRA, arXiv 2402.09353) — explicitly decomposes
  weight magnitude + direction, prioritizes magnitude updates = same idea as
  SV importance scaling. PEFT supports `use_dora=True`.
- **SVD-init LoRA** (PiSSA, arXiv 2404.02948) — initialize lora_A/B from top-k
  SVD of base weight instead of random; converges to "important" directions
  faster. PEFT supports `init_lora_weights="pissa"`.
- We already use `use_rslora=True` (stabilizes scaling). **Add `use_dora=True`**
  to existing LoraConfig — almost free upgrade.

### D. Post-hoc SV amplification on OUR adapter (cheap A/B test)
Steal cells 24's SVD trick. After SFT, for each adapter pair `(A, B)`:
```python
U,S,Vh = torch.linalg.svd(B @ A, full_matrices=False)
S_boost = S.clone(); S_boost[:rank//2] *= 1.12        # tune 1.05-1.20
new_B = U[:, :rank] * S_boost[:rank]
new_A = Vh[:rank, :]
```
Wrap as `boost_adapter(adapter_path, boost=1.12, top_frac=0.5)` script.
Generate 3 variants (1.00 baseline, 1.10, 1.20). Smoke-eval each on 50
held-out prompts. Pick winner. ~30 min compute, ~1-3pp possible gain.

### E. `lm_head` LoRA — reconsider
Reference includes `lm_head` in `target_modules`. CLAUDE.md says "skip lm_head"
(v9 destabilization). But reference notebook works at 87% with it.
**Hypothesis:** lm_head LoRA is destabilizing during training but **safe to
include if you train without it then PEFT-init lm_head LoRA to zero** at
adapter-save time. Cheap test: add `lm_head` as target with very low rank
(r=4, alpha=4), see if eval improves.

### F. Adapter-key audit cell — add to every training notebook
Cells 14-22 do the diff. **Adopt this pattern** in our SFT/RAFT notebooks
right after save:
```python
# After save_pretrained
import safetensors
with safe_open(f"{ADAPTER_DIR}/adapter_model.safetensors","pt","cpu") as f:
    keys = sorted(f.keys())
print(f"adapter has {len(keys)} keys, sample:", keys[:5], keys[-5:])
# Print module names model has
print([n for n,_ in model.named_modules() if "mamba" in n][:10])
```
Catches name-mismatch bugs locally, not at eval-time on Kaggle.

---

## Priority for our run (87% target)

| # | Action | Cost | Expected |
|---|---|---|---|
| 1 | Run SVD-amplify notebook on existing 85% adapter | 5 min | +1-5pp (just from correct module mapping if currently wrong) |
| 2 | Add `use_dora=True` to LoraConfig for next SFT/RAFT round | retrain | +2-4pp |
| 3 | Train with FUSED Mamba `in_proj` target (no split) | retrain | +1-2pp (cleaner adapter, no SVD lossy merge) |
| 4 | Post-hoc SV-boost A/B test (1.10 vs 1.20) on best adapter | 30 min | +0-2pp |
| 5 | Include `lm_head` low-rank LoRA | retrain | +0-2pp uncertain |

**Quickest win = (1) — costs nothing, no retrain.** Then bake fix (3) into
next training round, layer (2) on top.

---

## Caveats
- 1.12× boost is **arbitrary heuristic**, not validated. Could hurt on
  out-of-distribution. A/B test.
- SVD merge of two independently-trained LoRAs (gate + x) is lossy in general
  — author papers over it with amplification. Cleaner solution = train fused
  from start (item 3).
- `experts.w3` skip is competition-specific (Nemotron's gated MoE has dead
  w3 placeholders) — won't apply elsewhere.

---

## Implementation (2026-05-31, applied)

Plan `abundant-baking-hopcroft.md` executed. Concrete changes landed:

### New: `tools/postprocess_adapter.py`
Standalone CLI. Inputs adapter dir, writes vLLM-clean adapter to `--out`. Steps:
1. Force `target_modules` to canonical 9 names.
2. Rename safetensors keys `base_model.model.model` → `...backbone`.
3. Unfuse `experts.w1` / `experts.w2` to per-expert `up_proj` / `down_proj`.
4. Detect+fuse split `mamba.gate_proj` + `mamba.x_proj` LoRA pairs via QR-then-SVD
   to single `mamba.in_proj` at rank 32. Optional SV boost via `--boost 1.12 --top-frac 0.5`
   (1.0 = no boost; toggled per submission for A/B).
5. `--verify-only` for dry-run, prints key diffs without writing.

### Notebook patches (both `minmax-sft` cell 2b0532d8/cell 8 and `raft-reinforce-rej` cell-8)
- LoRA target regex updated: `shared_experts` (plural) confirmed, added `mamba.gate_proj`
  to the regex (was missing — Nemotron Mamba layer has `gate_proj` alongside `x_proj`;
  the post-process script later fuses these into `in_proj`).
- `use_dora=True` added to `LoraConfig` (Weight-Decomposed LoRA, +2-4pp typical).
- **Audit block** appended: prints first/last 5 trainable param names + total
  count, asserts `50M ≤ total ≤ 400M` to catch the "regex matched 0 → suffix
  fallback attached LoRA to all 128 routable experts" failure mode at training
  time instead of at eval.

### Notebook packaging patches (`minmax-sft` cell-24, `raft-reinforce-rej` cell-29)
- Calls `tools/postprocess_adapter.py` as subprocess before zipping.
- Looks for script at `tools/`, `/kaggle/working/tools/`, and `os.getcwd()/tools/`.
- `POSTPROC_BOOST` env var (default `1.0`) controls SV-amplify for A/B.
- Falls back to raw adapter if script missing or fails (with explicit warning).

### Kaggle prep needed
For the script to be found on Kaggle, either:
- Upload `tools/postprocess_adapter.py` as a Kaggle dataset and add to inputs, or
- Copy it inline in a notebook cell before packaging, or
- Use `!wget` from a public location.

Recommended: bundle `tools/` into the notebook's git/dataset.

### Verification
Local: `python tools/postprocess_adapter.py --help` works; argparse exits 0.
Pending end-to-end: full smoke on Kaggle once next adapter trains, then a
boost=1.0 vs 1.12 A/B over 100 prompts.
