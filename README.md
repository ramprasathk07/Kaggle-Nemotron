# 🏆 Kaggle – NVIDIA Nemotron Model Reasoning Challenge

> **Goal:** Advance reasoning techniques using **NVIDIA Nemotron-3-Nano-30B-A3B-BF16** on a novel benchmark of ~7,500 logic/math/bit-manipulation puzzles.  
> **Submission:** LoRA adapter (`adapter_config.json` + `adapter_model.safetensors`) evaluated via vLLM inference on Kaggle.  
> **Metric:** Accuracy — answers must be inside `\boxed{}`. String matching for text, ±10⁻² tolerance for numbers.  
> **Prizes:** $106,388 total — 1st $25k, 2nd $15k, 3rd $5k + DGX Sparks + Contribution Awards (Best Data, Best RL, Best Fine-tuning).  
> **Deadline:** ~2 months remaining (Final: June 2025).

---

## 📁 Repository Structure

```
Kaggle-Nemotron/
├── nvidia-nemotron-v7-5.ipynb   # Kaggle training notebook (Unsloth + SFT + LoRA)
├── v1_data_prep.py              # HuggingFace dataset augmentation pipeline
├── README.md                    # This file
├── LICENSE
└── .gitignore
```

---

## 📊 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Base Model** | ✅ Working | Nemotron-3-Nano-30B-A3B loaded via Unsloth, BF16 |
| **LoRA Config** | ✅ Working | Rank=32, Alpha=64, RSLoRA, targets: q/k/v/o/gate/up/down/x_proj/dt_proj/lm_head |
| **SFT Training** | ✅ Working | 2 epochs, batch=2, grad_accum=4, cosine LR schedule, paged_adamw_8bit |
| **Submission Pipeline** | ✅ Working | Auto-packages adapter into submission.zip |
| **Data Prep (v1_data_prep.py)** | ⚠️ Needs Fixes | Wonderland prompt wrapper + HF dataset ingestion (see issues below) |
| **CoT Generation** | 🔲 Not Started | Need Mistral Premium API to generate CoT for competition data |
| **GRPO Training** | 🔲 Not Started | Reinforcement learning phase not yet implemented |
| **Leaderboard Score** | 🔲 TBD | Pending submission with improved data |

---

## 🔍 Code Review: `v1_data_prep.py`

### ✅ What's Good
- Clean modular design — separate adapters per dataset, good CLI interface
- Category auto-detection via keyword matching is reasonable
- `<think>...</think>\boxed{}` output format matches competition expectations
- Streaming support for large datasets (7.5M+ OpenMathReasoning)
- Dry-run verification mode — great for testing without downloads
- JSONL validation with `\boxed{}` assertion

### ⚠️ Issues & Fixes Needed

**1. ❌ Wonderland prompts are counter-productive for this competition**
The competition benchmark tests logical reasoning, bit manipulation, and algebraic rule-following — NOT "Alice in Wonderland" themed problems. Wrapping standard math problems in Wonderland lore will confuse the model and hurt generalization to the actual test set. The competition's own "Wonderland" problems have specific transformation rules that aren't captured by generic fantasy wrappers.

> **Fix:** Remove Wonderland prompt wrappers. Use clean, direct problem statements with just the `\boxed{}` instruction suffix. Match the format the competition uses:
> ```
> {problem}\nPlease put your final answer inside \boxed{}.
> ```

**2. ❌ Missing `<think>` opening tag in CoT response**
In `format_cot_response()`, the response starts with `<think>\n{cot_clean}\n</think>` which is correct. **BUT** in the notebook's training cell (line ~713), the assistant content is constructed as:
```python
assistant_content = cot_cleaned + f"\n</think>\n\\boxed{{{answer}}}"
```
This is missing the opening `<think>` tag! The data prep script has it right, but the notebook doesn't — make sure they're consistent.

**3. ⚠️ `\boxed{}` regex fails on nested braces**
The regex `r"\\boxed\{([^}]*)\}"` at line 142/168 will fail on answers containing nested braces like `\boxed{\frac{1}{2}}`. This loses the inner content.

> **Fix:** Use a brace-balanced parser:
> ```python
> def extract_boxed(text):
>     idx = text.find("\\boxed{")
>     if idx == -1:
>         return ""
>     depth, start = 1, idx + 7
>     for i in range(start, len(text)):
>         if text[i] == '{': depth += 1
>         elif text[i] == '}': depth -= 1
>         if depth == 0:
>             return text[start:i]
>     return text[start:]
> ```

**4. ⚠️ Validation check too strict — `count("\\boxed{") == 1` will break**
Line 373: Some legitimate CoT solutions reference `\boxed{}` multiple times in their reasoning before the final answer. This assertion will incorrectly reject valid data.

> **Fix:** Check that the response *ends* with a `\boxed{}` pattern, rather than counting occurrences.

**5. ⚠️ No deduplication or quality filtering**
The pipeline doesn't check for duplicate problems across datasets (e.g., NuminaMath and OpenMathReasoning share many problems). Also, no length-based filtering on CoT quality — very short or very long solutions are both problematic for LoRA training.

> **Fix:** Add hash-based dedup on problem text and filter to a target CoT length range (e.g., 200–4000 chars).

**6. ℹ️ Missing competition-relevant problem types**
The pipeline covers standard math categories but misses the key competition domains:
- **Bit manipulation** (XOR, AND, OR operations on binary numbers)
- **Custom operation rules** (Wonderland-defined operations like `a ★ b = ...`)
- **Pattern recognition** (sequences following non-standard rules)

> **Fix:** Add adapters for datasets that cover these domains, or create synthetic examples.

---

## 🎯 Winning Strategy (Final Plan)

### Phase 1: SFT with Augmented External Data
1. **Curate high-quality CoT datasets from HuggingFace:**
   - `AI-MO/NuminaMath-CoT` — 860k competition-level math with built-in CoT
   - `nvidia/OpenMathReasoning` (CoT subset) — 7.5M diverse math reasoning
   - `AI-MO/OlympiadBench` — olympiad-level proofs
   - `RUC-AIBOX/OlymMATH` — novel olympiad generation
2. **Augment with competition-specific techniques:**
   - Add bit manipulation problems (XOR/AND/OR operations, binary conversion)
   - Add custom operator problems (define `★`, `⊕` rules → compute)
   - Add pattern recognition / sequence inference problems
3. **Filter & format:** Clean format with `<think>CoT</think>\boxed{answer}`, deduplicate, target 20–50k high-quality samples

### Phase 2: Competition-Specific CoT Generation
4. **Take the ~7,500 competition training problems** (which have `prompt` + `answer` but limited/no CoT)
5. **Use Mistral Premium (Le Chat API) to generate detailed CoT** for each problem — get the model to reason through each problem step-by-step, ending with `\boxed{answer}`
6. **Validate generated CoT** — only keep examples where the CoT leads to the correct answer

### Phase 3: GRPO (Reinforcement Learning)
7. **After SFT on the combined data (Phase 1 + Phase 2),** apply GRPO (Group Relative Policy Optimization):
   - Use the competition problems as the RL training set
   - Reward function: does the generated answer match the ground truth `\boxed{}`?
   - This aligns the model specifically to the competition's distribution
8. **Iterate:** Run multiple GRPO rounds on hard examples the model still gets wrong

### Phase 4: Submission Optimization
9. **Tune inference parameters:** temperature, top_p, repetition penalty
10. **Ensemble/best-of-N:** If time permits, submit from multiple checkpoints

---

## 🚀 Top 5 Things to Try Next

### 1. 🔧 Fix `v1_data_prep.py` — Remove Wonderland Wrappers & Fix Regex
Strip out the Wonderland prompt templates. Use the exact competition format: `{problem}\nPlease put your final answer inside \boxed{}`. Fix the nested brace regex for `\boxed{}` extraction. This is the most critical blocker — your SFT data format must match what the model sees at inference.

### 2. 🧪 Generate CoT for Competition Data via Mistral Premium
Write a script that takes the ~7,500 `problem_ids_matched.csv` rows, sends each problem to Mistral Premium (or GPT-4/Claude), asks it to produce a step-by-step solution ending in `\boxed{answer}`, then validates the answer matches ground truth. This gives you gold-standard CoT on the *exact* competition distribution.

### 3. 🔄 Implement GRPO Training Loop
After SFT, build a GRPO (or DPO/RLOO) training pipeline on the competition problems. Use the `trl` library's `GRPOTrainer` with the Nemotron model. Reward = 1.0 if `\boxed{answer}` matches ground truth, 0.0 otherwise. Format reward to also bonus on valid `<think>` reasoning structure.

### 4. 📊 Add Competition-Specific Synthetic Data
Generate synthetic problems covering **bit manipulation**, **custom operations**, and **Wonderland rule-following** (not generic fantasy, but actual "in this world, operation X works like Y" patterns from the competition). Target 2–5k synthetic examples to fill the distributional gap.

### 5. 🎛️ Hyperparameter Sweep on LoRA + Training Config
Current LoRA config may not be optimal. Try:
- **Rank 64** (more capacity for reasoning)
- **Alpha 128** (2× rank ratio)
- **Learning rate annealing:** try 5e-5 and 1e-4
- **3 epochs** with early stopping based on validation loss
- **Max seq length 4096** if OOMing on 8192 (most competition answers fit in <4k tokens)

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| Base Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Fine-tuning | Unsloth + PEFT/LoRA + TRL (SFTTrainer) |
| Training Hardware | Kaggle T4/P100 GPU (or Blackwell B200 if available) |
| Data Sources | HuggingFace (NuminaMath, OpenMathReasoning, OlympiadBench) |
| CoT Generation | Mistral Premium API |
| RL Training | TRL GRPOTrainer (planned) |
| Inference | vLLM on Kaggle |

---

## 📝 Competition Quick Reference

- **Competition URL:** https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- **Model:** Nemotron-3-Nano-30B-A3B-BF16
- **Submission:** `submission.zip` with LoRA adapter files
- **Answer format:** `\boxed{answer}` (string match or ±10⁻² numerical)
- **Key domains:** Bit manipulation, algebraic rules, logical puzzles, custom operators, pattern recognition
- **Prize eligibility for Contribution Awards:** Must be in top 10% of leaderboard
- **Custom Dataset in Kaggle**: https://www.kaggle.com/datasets/ramkan07/v7-mix/data