"""Builder for the v13 RTX 6000 Pro notebook pair.

Run:  python build_v13.py

Emits:
  sft/nemo-v13-sft-rtx6000.ipynb              — SFT warm-start
  post-training/nemo-v13-drgrpo-rtx6000.ipynb — Dr. GRPO RLVR w/ verifiable rewards

Why a builder script: notebooks have lots of raw-string regex / LaTeX literals
that are painful to escape inside JSON-quoted strings. Easier to author the
sources as Python here, dump to JSON once.

References folded in:
  - jhyland01/kaggle_nemotron       — SFT->GRPO pipeline, per-type verifiers
  - WINNING_PLAN.md Phase 1         — Dr.GRPO recipe (LR=1e-6 const, beta=0,
                                      num_gen=8, T=1.0, max_completion=3000)
  - WINNING_PLAN.md Phase 2 (DAPO)  — clip-higher, mask_truncated, overlong shaping
  - CLAUDE.md gotchas               — max_completion >= 512, beta=0 saves mem
"""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


# ============================================================
# Helpers
# ============================================================

def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n") + "\n"}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n") + "\n",
    }


def write_nb(path: Path, cells):
    nb = make_nb(cells)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  wrote {path}  ({len(cells)} cells, {path.stat().st_size/1024:.1f} KB)")


# ============================================================
# Shared cell sources (used by both notebooks)
# ============================================================

# ---- Shared boilerplate: stdio + path config
def cell_paths(*, role: str):
    """role in {'sft', 'grpo'} — only differs in adapter dir name."""
    return code(rf"""
import os, sys

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="strict")

# Repo layout — override via env var if running elsewhere
REPO_ROOT = os.environ.get("KAGGLE_NEMO_REPO", r"F:/Hackathons/Kaggle-Nemotron")

BASE_MODEL_NAME   = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
SFT_DATA_PATH     = os.path.join(REPO_ROOT, "data", "merged_cot_final.csv")
GRPO_DATA_PATH    = os.path.join(REPO_ROOT, "data", "src", "train.csv")

OUTPUT_ROOT       = os.path.join(REPO_ROOT, "outputs", "rtx6000_v13")
SFT_ADAPTER_DIR   = os.path.join(OUTPUT_ROOT, "sft_adapter")
GRPO_ADAPTER_DIR  = os.path.join(OUTPUT_ROOT, "grpo_adapter")
SUBMISSION_DIR    = os.path.join(OUTPUT_ROOT, "submission_adapter_{role}")
TB_LOG_DIR        = os.path.join(OUTPUT_ROOT, "tb_logs_{role}")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(TB_LOG_DIR, exist_ok=True)

SEED          = 42
PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{{}}`. For example: `\\boxed{{your answer}}`'

print({{
    "REPO_ROOT": REPO_ROOT,
    "BASE_MODEL_NAME": BASE_MODEL_NAME,
    "SFT_DATA_PATH": SFT_DATA_PATH,
    "GRPO_DATA_PATH": GRPO_DATA_PATH,
    "SFT_ADAPTER_DIR": SFT_ADAPTER_DIR,
    "GRPO_ADAPTER_DIR": GRPO_ADAPTER_DIR,
}})
""")


CELL_INSTALL = code(r"""
# Run ONCE per environment, then restart kernel and set back to False.
INSTALL_DEPS = False

if INSTALL_DEPS:
    import subprocess, sys
    pkgs = [
        # Blackwell sm_120/sm_100: need recent torch with CUDA 12.4+.
        # If wheels missing for your Blackwell SKU, swap to nightly:
        #   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
        "torch>=2.4",
        "transformers>=4.46",
        "peft>=0.13",
        "trl>=0.16",            # required for loss_type='dr_grpo' / scale_rewards / mask_truncated
        "datasets>=3.0",
        "accelerate>=1.0",
        "bitsandbytes>=0.44",
        "tensorboard",
        "mamba-ssm",
        "causal-conv1d",
        "pandas", "numpy",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-U"] + pkgs, check=True)
    print("Install done. RESTART the kernel before continuing.")
else:
    print("INSTALL_DEPS=False — assuming env is ready.")
""")


CELL_MODEL_LOAD = code(r"""
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required (target: RTX 6000 Pro Blackwell).")

dev_props = torch.cuda.get_device_properties(0)
print(f"GPU: {torch.cuda.get_device_name()}  Total VRAM: {dev_props.total_memory/1e9:.1f} GB  CC: sm_{dev_props.major}{dev_props.minor}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model (bf16, eager attention)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    attn_implementation="eager",
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

total_b = sum(p.numel() for p in model.parameters()) / 1e9
print(f"Model loaded — {total_b:.2f}B params total.  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
""")


CELL_LORA_DISCOVERY = code(r"""
import re
from collections import Counter

linear_modules = []
for name, mod in model.named_modules():
    cls = mod.__class__.__name__
    if cls in ("Linear", "Linear4bit", "Linear8bitLt"):
        linear_modules.append(name)

suffix_counts = Counter(n.rsplit(".", 1)[-1] for n in linear_modules)
print("Linear suffix counts (top 20):")
for s, c in suffix_counts.most_common(20):
    print(f"  {s:30s} {c}")

parent_counts = Counter()
for n in linear_modules:
    parts = n.split(".")
    if len(parts) >= 2:
        parent_counts[parts[-2]] += 1
print("\nParent module counts (top 20):")
for p, c in parent_counts.most_common(20):
    print(f"  {p:30s} {c}")

print("\nSample names:")
for n in linear_modules[:3] + linear_modules[len(linear_modules)//2:len(linear_modules)//2+3] + linear_modules[-3:]:
    print(f"  {n}")
""")


CELL_LORA_BUILD = code(r"""
from peft import LoraConfig, get_peft_model, TaskType

LORA_RANK    = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.0

# Sensitive targets only — excludes routable experts (sparse), router (frozen by NVIDIA),
# lm_head / embeddings (destabilize). See CLAUDE.md LoRA priority table.
target_regex = (
    r".*("
    r"self_attn\.(q|k|v|o)_proj"
    r"|mamba\.(in|out|x|dt)_proj"
    r"|shared_expert\.(gate|up|down)_proj"
    r")$"
)

matched = [n for n in linear_modules if re.match(target_regex, n)]
print(f"Regex matched {len(matched)} modules.")
if not matched:
    print("[WARN] regex matched 0 modules — module names differ from expected.")
    print("       Falling back to suffix list (will also hit routable experts — wasteful).")
    target_modules = ["q_proj","k_proj","v_proj","o_proj","in_proj","out_proj","x_proj","dt_proj","gate_proj","up_proj","down_proj"]
else:
    print("Sample matches:", matched[:3], "...", matched[-2:])
    target_modules = target_regex

lora_config = LoraConfig(
    r              = LORA_RANK,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    bias           = "none",
    target_modules = target_modules,
    task_type      = TaskType.CAUSAL_LM,
    use_rslora     = True,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
""")


# ============================================================
# Notebook 1: SFT
# ============================================================
SFT_CELLS = [
    md(r"""
# Nemotron 3 Nano — SFT Warm-Start (RTX 6000 Pro v13)

Single-GPU SFT on `data/merged_cot_final.csv`. Goal: imprint chat template +
`<think>...</think>\boxed{}` format on the base Nemotron-3-Nano-30B model.

**Pair**: GRPO done-right runs after this in
`post-training/nemo-v13-drgrpo-rtx6000.ipynb` (loads the adapter saved here).

**Hardware**: 1× RTX 6000 Pro (Blackwell, ~96 GB VRAM). Bare HuggingFace —
no Unsloth (Blackwell PTXAS path is fragile).

## Pipeline position

```
THIS NOTEBOOK ─── SFT warm-start ──> SFT_ADAPTER_DIR
                                      │
                                      ▼
                                post-training/nemo-v13-drgrpo-rtx6000.ipynb
                                  (Dr.GRPO + verifiable rewards)
                                      │
                                      ▼
                                GRPO_ADAPTER_DIR ──> submission.zip
```
"""),
    md("## Path & Run Configuration"),
    cell_paths(role="sft"),
    md("""## One-time dependency install
Run once with `INSTALL_DEPS=True`, restart kernel, set back to `False`."""),
    CELL_INSTALL,
    md("## Model + Tokenizer Loading"),
    CELL_MODEL_LOAD,
    md("## LoRA Target Discovery"),
    CELL_LORA_DISCOVERY,
    md("## LoRA Config (rank=32, RSLoRA)"),
    CELL_LORA_BUILD,
    md(r"""
## SFT Dataset Prep — `merged_cot_final.csv`

Builds conversational records `[user, assistant]` where the assistant turn is
the canonical `<think>{cot}</think>\boxed{{answer}}` template. Strips any
pre-existing `\boxed{{}}` from the CoT before re-appending a clean one.
"""),
    code(r"""
import pandas as pd, re
from datasets import Dataset as HFDataset

df_sft = pd.read_csv(SFT_DATA_PATH)
print(f"SFT data: {len(df_sft)} rows.  Columns: {list(df_sft.columns)}")
required = {"prompt", "cot", "answer"}
missing = required - set(df_sft.columns)
if missing:
    raise ValueError(f"Missing required columns in {SFT_DATA_PATH}: {missing}")

df_sft = df_sft.dropna(subset=list(required)).reset_index(drop=True)
df_sft = df_sft.sample(frac=1, random_state=SEED).reset_index(drop=True)

_BOXED_STRIP = re.compile(r"\\boxed\{[^}]*\}")
def clean_cot(c):
    c = _BOXED_STRIP.sub("", str(c)).rstrip()
    return c.replace("<think>", "").replace("</think>", "").strip()

records, skipped = [], 0
for _, row in df_sft.iterrows():
    cot = clean_cot(row["cot"])
    if len(cot) < 5:
        skipped += 1
        continue
    user_content = str(row["prompt"]) + PROMPT_SUFFIX
    assistant_content = f"<think>\n{cot}\n</think>\n\\boxed{{{row['answer']}}}"
    records.append({"messages": [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]})

sft_dataset = HFDataset.from_list(records)
print(f"SFT records: {len(records)} (skipped {skipped} short / empty CoT)")
"""),
    md(r"""
## SFT Trainer

Config matches the v7-5 Kaggle settings but scaled for RTX 6000 Pro headroom:

| Param | Value | Note |
|---|---|---|
| `num_train_epochs` | 2 | enough to imprint format |
| `per_device_train_batch_size` | 2 | 96 GB VRAM allows |
| `gradient_accumulation_steps` | 4 | effective batch = 8 |
| `learning_rate` | 5e-5 | matches NVIDIA's SFT recipe |
| `lr_scheduler_type` | cosine | with 5% warmup |
| `max_length` | 4096 | bigger than v7-5's 2048; covers long CoT |
| `optim` | paged_adamw_8bit | memory-friendly |
| `packing` | True | densify short samples |
"""),
    code(r"""
import gc, time, torch
from trl import SFTTrainer, SFTConfig

MAX_SEQ_LEN_SFT = 4096

def sft_formatting(example):
    msgs = example["messages"]
    if msgs and isinstance(msgs[0], dict):
        convs = [msgs]
    else:
        convs = msgs
    out = []
    for c in convs:
        try:
            t = tokenizer.apply_chat_template(c, tokenize=False,
                                              add_generation_prompt=False,
                                              enable_thinking=True)
        except TypeError:
            t = tokenizer.apply_chat_template(c, tokenize=False,
                                              add_generation_prompt=False)
        out.append(t)
    return out

sft_args = SFTConfig(
    output_dir                   = os.path.join(OUTPUT_ROOT, "sft_run"),
    num_train_epochs             = 2,
    per_device_train_batch_size  = 2,
    gradient_accumulation_steps  = 4,         # effective batch = 8
    learning_rate                = 5e-5,
    lr_scheduler_type            = "cosine",
    warmup_ratio                 = 0.05,
    max_length                   = MAX_SEQ_LEN_SFT,
    packing                      = True,
    optim                        = "paged_adamw_8bit",
    adam_beta1                   = 0.9,
    adam_beta2                   = 0.95,
    adam_epsilon                 = 1e-8,
    weight_decay                 = 0.01,
    max_grad_norm                = 1.0,
    bf16                         = True,
    gradient_checkpointing       = True,
    gradient_checkpointing_kwargs= {"use_reentrant": False},
    logging_steps                = 10,
    logging_dir                  = TB_LOG_DIR,
    report_to                    = "tensorboard",
    save_strategy                = "no",
    dataloader_num_workers       = 4,
    remove_unused_columns        = False,
    seed                         = SEED,
    dataset_num_proc             = 4,
)

sft_trainer = SFTTrainer(
    model            = model,
    args             = sft_args,
    train_dataset    = sft_dataset,
    processing_class = tokenizer,
    formatting_func  = sft_formatting,
)

torch.cuda.empty_cache(); gc.collect()

print("Starting SFT warm-start...")
t0 = time.time()
sft_trainer.train()
print(f"SFT done in {(time.time()-t0)/60:.1f} min")

os.makedirs(SFT_ADAPTER_DIR, exist_ok=True)
model.save_pretrained(SFT_ADAPTER_DIR)
tokenizer.save_pretrained(SFT_ADAPTER_DIR)
print(f"SFT adapter saved -> {SFT_ADAPTER_DIR}")
"""),
    md(r"""
## Post-SFT Sanity Check (3-sample greedy generation)

Before launching multi-hour GRPO, verify the SFT adapter actually emits
`\boxed{}`. If this fails → fix data/formatter before running GRPO.
"""),
    code(r"""
import torch
model.eval()
sample_rows = df_sft.sample(3, random_state=0)[["prompt", "answer"]].values.tolist()
for p, ans in sample_rows:
    msgs = [{"role": "user", "content": str(p) + PROMPT_SUFFIX}]
    try:
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=True)
    except TypeError:
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    has_box = "\\boxed{" in gen
    tag = "BOX  " if has_box else "NOBOX"
    print(f"[{tag}] expected={str(ans)!r:30s}  tail={gen[-180:]!r}")
model.train()
"""),
    md(r"""
## (Optional) Package SFT-only submission

Useful for an early leaderboard read on what SFT alone achieves before GRPO.
Skip if you intend to run GRPO immediately.
"""),
    code(r"""
PACKAGE_SFT_SUBMISSION = False  # flip to True if you want a pre-GRPO submission

if PACKAGE_SFT_SUBMISSION:
    import json, shutil, zipfile
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    required = ["adapter_config.json", "adapter_model.safetensors"]
    for fname in required:
        sp = os.path.join(SFT_ADAPTER_DIR, fname)
        dp = os.path.join(SUBMISSION_DIR, fname)
        if not os.path.exists(sp):
            raise FileNotFoundError(f"Missing: {sp}")
        shutil.copy2(sp, dp)
        print(f"  copied {fname}  ({os.path.getsize(dp)/1024/1024:.1f} MB)")
    cfg_path = os.path.join(SUBMISSION_DIR, "adapter_config.json")
    with open(cfg_path) as f: cfg = json.load(f)
    cfg["base_model_name_or_path"] = BASE_MODEL_NAME
    cfg["inference_mode"] = True
    cfg["lora_dropout"]   = 0.0
    with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
    zip_path = os.path.join(OUTPUT_ROOT, "submission_sft.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in required:
            zf.write(os.path.join(SUBMISSION_DIR, fname), fname)
    print(f"\nsubmission_sft.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
else:
    print("PACKAGE_SFT_SUBMISSION=False — skipping. Run GRPO notebook next.")
"""),
    md(r"""
## Next step

Open `post-training/nemo-v13-drgrpo-rtx6000.ipynb` and run it. It will load
the SFT adapter from `SFT_ADAPTER_DIR`, apply Dr. GRPO with verifiable
per-puzzle-type rewards, and produce the final `submission.zip`.
"""),
]


# ============================================================
# Notebook 2: Dr. GRPO with verifiable rewards
# ============================================================
GRPO_CELLS = [
    md(r"""
# Nemotron 3 Nano — Dr. GRPO RLVR (RTX 6000 Pro v13)

Dr. GRPO (Sea AI Lab, arxiv:2503.20783) with verifiable per-puzzle-type
rewards. Loads the SFT adapter from `nemo-v13-sft-rtx6000.ipynb`.

## Why "Done Right"

Standard GRPO has two biases (Sea AI Lab):
1. **Response-length norm `1/|o_i|`** — favors short correct, long wrong responses
2. **Std normalization** — over-weights easy/hard questions inverse to difficulty variance

Dr. GRPO removes both:
- `loss_type="dr_grpo"` (or fallback `"dapo"`) — replaces length norm with a constant
- `scale_rewards=False` — removes std norm

## Verifiable Rewards (per puzzle type)

Ported from [jhyland01/kaggle_nemotron](https://github.com/jhyland01/kaggle_nemotron):

| Puzzle Type | Verifier Strategy |
|---|---|
| Number Base Conversion | Numeric match in target base |
| Gravitational Constant | Numeric tolerance (rel 1e-2) |
| Unit Conversion | Numeric tolerance (rel 1e-2) |
| Text Encryption | Case-insensitive string match |
| Bit Manipulation | 8-bit binary string match |
| Equation Transformation | String match (numeric or concat) |
| Unknown | Fallback: normalized string + numeric tolerance |

Each completion gets a binary `correct/incorrect` decision against its
known answer. Per-type accuracy logged to TensorBoard.

## Hard rules

- Reward requires `<think>...</think>` **before** `\boxed{}` — blocks naked-boxed reward hacking
- `\boxed{}` extraction is brace-balanced (handles `\boxed{\frac{1}{2}}`)
- Eval is greedy (T=0); training is T=1.0 for exploration — this is expected
"""),
    md("## Path & Run Configuration"),
    cell_paths(role="grpo"),
    md("## Dependency install (one-time, kernel restart required)"),
    CELL_INSTALL,
    md("## Base Model + Tokenizer"),
    CELL_MODEL_LOAD,
    md(r"""
## Load SFT Adapter (warm-start)

Loads the LoRA adapter from the SFT notebook in `is_trainable=True` mode so
Dr. GRPO continues training the same LoRA matrices. If `SFT_ADAPTER_DIR` is
missing, falls back to building a fresh LoRA (cold-start GRPO — slower
convergence but doable).
"""),
    code(r"""
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import re
from collections import Counter

# Inventory linear modules — for fallback path + sanity print
linear_modules = []
for name, mod in model.named_modules():
    if mod.__class__.__name__ in ("Linear", "Linear4bit", "Linear8bitLt"):
        linear_modules.append(name)

if os.path.isdir(SFT_ADAPTER_DIR) and os.path.exists(os.path.join(SFT_ADAPTER_DIR, "adapter_config.json")):
    print(f"Loading SFT adapter from {SFT_ADAPTER_DIR} (is_trainable=True)")
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_DIR, is_trainable=True)
    # Re-enable gradient checkpointing on the wrapped model
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
else:
    print(f"[WARN] No SFT adapter at {SFT_ADAPTER_DIR}.  Cold-start GRPO with fresh LoRA.")
    target_regex = (
        r".*("
        r"self_attn\.(q|k|v|o)_proj"
        r"|mamba\.(in|out|x|dt)_proj"
        r"|shared_expert\.(gate|up|down)_proj"
        r")$"
    )
    matched = [n for n in linear_modules if re.match(target_regex, n)]
    target_modules = target_regex if matched else [
        "q_proj","k_proj","v_proj","o_proj","in_proj","out_proj",
        "x_proj","dt_proj","gate_proj","up_proj","down_proj",
    ]
    lora_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
        target_modules=target_modules, task_type=TaskType.CAUSAL_LM, use_rslora=True,
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
"""),
    md(r"""
## Verifiable Reward Functions (per-type)

Implements the deterministic verifiers from the reference repo. Each
completion's `\boxed{}` answer is compared against the labeled ground truth
using a type-aware comparator (numeric tolerance for gravity / unit conversion,
binary normalization for bit manipulation, etc.).
"""),
    code(r"""
import re

# ---------- Puzzle type classifier (regex from jhyland01/kaggle_nemotron) ----------
_PUZZLE_PATTERNS = [
    ("Number Base Conversion", re.compile(r"numeral system|base[- ]?\d|number.*convert|radix|secret number", re.IGNORECASE)),
    ("Gravitational Constant", re.compile(r"gravit|gravity|falling|free.?fall|acceleration due to", re.IGNORECASE)),
    ("Equation Transformation", re.compile(r"transformation rule|equation.*transform|secret.*rule.*equation|rule.*applied.*equation", re.IGNORECASE)),
    ("Text Encryption",         re.compile(r"encrypt|cipher|secret.*code.*letter|coded.*message|secret.*text", re.IGNORECASE)),
    ("Bit Manipulation",        re.compile(r"bit.?manipul|binary|8.?bit|bitwise|bit.*transform", re.IGNORECASE)),
    ("Unit Conversion",         re.compile(r"unit.?conver|measurement|becomes.*\d|secret.*conver.*measur", re.IGNORECASE)),
]

def classify_puzzle(prompt: str) -> str:
    for label, pat in _PUZZLE_PATTERNS:
        if pat.search(prompt or ""):
            return label
    return "Unknown"

# ---------- Boxed extraction (brace-balanced) ----------
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def _content(c):
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return c[-1].get("content", "")
    return str(c)

def extract_boxed(text):
    idx = text.find("\\boxed{")
    if idx == -1:
        m = _BOXED_RE.search(text)
        return m.group(1).strip() if m else None
    depth, start = 1, idx + 7
    for i in range(start, len(text)):
        if text[i] == "{":   depth += 1
        elif text[i] == "}": depth -= 1
        if depth == 0:       return text[start:i].strip()
    return text[start:].strip()

def _norm(s):
    return str(s).strip().lower().replace(" ", "").replace(",", "")

# ---------- Type-aware verifier ----------
def _numeric_match(predicted, expected, rel_tol=1e-2, abs_tol=1e-4):
    try:
        pf = float(str(predicted).replace(",", "").strip())
        ef = float(str(expected).replace(",", "").strip())
    except (ValueError, TypeError):
        return False
    return abs(pf - ef) <= max(rel_tol * max(1.0, abs(ef)), abs_tol)

def _binary_match(predicted, expected):
    p, e = _norm(predicted), _norm(expected)
    # Strip 0b prefix, accept any length
    p = p.removeprefix("0b") if hasattr(p, "removeprefix") else (p[2:] if p.startswith("0b") else p)
    e = e.removeprefix("0b") if hasattr(e, "removeprefix") else (e[2:] if e.startswith("0b") else e)
    if not all(ch in "01" for ch in p) or not all(ch in "01" for ch in e):
        return False
    return int(p, 2) == int(e, 2)

def _number_base_match(predicted, expected):
    # If both parse as ints in any base 2-16 to same value, accept.
    p, e = _norm(predicted), _norm(expected)
    if p == e:
        return True
    # try base-10 first
    try:
        if int(p, 10) == int(e, 10):
            return True
    except (ValueError, TypeError):
        pass
    # try other bases
    for base in (2, 8, 16):
        try:
            if int(p, base) == int(e, base):
                return True
        except (ValueError, TypeError):
            continue
    return False

def verify_answer(predicted, expected, puzzle_type):
    if predicted is None:
        return False
    pn, en = _norm(predicted), _norm(expected)
    # Always accept exact normalized string match
    if pn == en:
        return True

    if puzzle_type in ("Gravitational Constant", "Unit Conversion"):
        return _numeric_match(predicted, expected, rel_tol=1e-2, abs_tol=1e-4)
    if puzzle_type == "Bit Manipulation":
        return _binary_match(predicted, expected) or _numeric_match(predicted, expected)
    if puzzle_type == "Number Base Conversion":
        return _number_base_match(predicted, expected) or _numeric_match(predicted, expected)
    if puzzle_type == "Text Encryption":
        # case-insensitive exact, plus whitespace-insensitive
        return pn.replace("'", "") == en.replace("'", "")
    if puzzle_type == "Equation Transformation":
        return _numeric_match(predicted, expected) or pn == en
    # Unknown / fallback: numeric tolerance + string match
    return _numeric_match(predicted, expected)

# ---------- Smoke tests ----------
assert verify_answer("42", "42", "Unknown")
assert verify_answer("9.81", "9.80", "Gravitational Constant")          # within 1e-2 rel
assert not verify_answer("9.81", "10.0", "Gravitational Constant")
assert verify_answer("00101010", "00101010", "Bit Manipulation")
assert verify_answer("00101010", "42", "Bit Manipulation")              # bin vs dec
assert verify_answer("0xFF", "255", "Number Base Conversion")
assert verify_answer("HELLO", "hello", "Text Encryption")
print("Verifier smoke tests pass.")
"""),
    md(r"""
## Reward Functions

Reward components (summed):

1. **Format** — `+0.5` for any `\boxed{}`, `+0.3` for `<think>...</think>`
   appearing **before** the boxed answer
2. **Verifiable accuracy** — `+2.0` if type-aware verifier says correct, else `0.0`
3. **DAPO overlong penalty** — soft `[0, -1]` between
   `max_completion + buffer` and `2x max_completion`

Per-type accuracy logged to TensorBoard via `log_metric`.
"""),
    code(r"""
GRPO_MAX_COMPLETION = 3000   # Dr. GRPO recipe (Table 6 of arxiv:2503.20783)
GRPO_MAX_PROMPT     = 3072
OVERLONG_BUFFER     = 512

# ---------- Format reward ----------
def format_reward(completions, **kwargs):
    rewards = []
    for c in completions:
        text = _content(c)
        has_boxed = bool(_BOXED_RE.search(text))
        has_think = bool(_THINK_RE.search(text))
        think_before_boxed = (
            has_think and has_boxed and
            text.find("</think>") < text.rfind("\\boxed{")
        )
        r = 0.0
        if has_boxed:           r += 0.5
        if think_before_boxed:  r += 0.3
        rewards.append(r)
    return rewards

# ---------- Verifiable accuracy reward (binary, type-aware) ----------
# Track per-type accuracy for logging
_per_type_stats = {}

def accuracy_reward(completions, answer, prompt=None, **kwargs):
    # 'prompt' column may or may not be passed depending on trainer state
    prompts_iter = prompt if prompt is not None else [None] * len(completions)
    rewards = []
    for c, expected, raw_prompt in zip(completions, answer, prompts_iter):
        text = _content(c)
        predicted = extract_boxed(text)
        # Classify (uses raw user prompt if available, else completion itself)
        ptype = classify_puzzle(raw_prompt or text)
        ok = verify_answer(predicted, expected, ptype)
        rewards.append(2.0 if ok else 0.0)

        # Track per-type stats
        s = _per_type_stats.setdefault(ptype, {"n": 0, "correct": 0})
        s["n"] += 1
        if ok:
            s["correct"] += 1
    return rewards

# ---------- DAPO overlong shaping ----------
def overlong_penalty(completions, **kwargs):
    rewards = []
    soft = GRPO_MAX_COMPLETION + OVERLONG_BUFFER
    hard = GRPO_MAX_COMPLETION * 2
    for c in completions:
        text = _content(c)
        n = len(text) / 4.0   # rough char->token estimate
        if   n <= soft: r = 0.0
        elif n >= hard: r = -1.0
        else:           r = -((n - soft) / (hard - soft))
        rewards.append(r)
    return rewards

def combined_reward(completions, answer, prompt=None, **kwargs):
    f = format_reward(completions)
    a = accuracy_reward(completions, answer=answer, prompt=prompt)
    o = overlong_penalty(completions)
    return [x + y + z for x, y, z in zip(f, a, o)]

# Smoke test
test_compl = ["<think>\nthink\n</think>\n\\boxed{42}",
              "no think tag here \\boxed{42}",
              "<think>\nblah\n</think>\n\\boxed{wrong}"]
test_ans = ["42", "42", "42"]
test_prompts = ["compute the result", "compute the result", "compute the result"]
print("Smoke combined_reward:", combined_reward(test_compl, answer=test_ans, prompt=test_prompts))
print("Per-type stats:", _per_type_stats)
_per_type_stats.clear()
print("Reward functions ready.")
"""),
    md(r"""
## GRPO Dataset Prep — Raw competition `train.csv`

Keeps **prompt + answer** only. The model generates rollouts; the verifier
scores them. No pre-computed CoT.
"""),
    code(r"""
import pandas as pd
from datasets import Dataset as HFDataset

df_grpo = pd.read_csv(GRPO_DATA_PATH)
print(f"GRPO data: {len(df_grpo)} rows.  Columns: {list(df_grpo.columns)}")
if not {"prompt", "answer"}.issubset(df_grpo.columns):
    raise ValueError(f"train.csv must have 'prompt' and 'answer'; got {list(df_grpo.columns)}")

df_grpo = df_grpo.dropna(subset=["prompt", "answer"]).reset_index(drop=True)
df_grpo = df_grpo.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Sample type distribution before training
type_dist = df_grpo["prompt"].sample(min(1000, len(df_grpo)), random_state=0).apply(classify_puzzle).value_counts()
print("\nSample puzzle type distribution (first 1000 prompts):")
print(type_dist)

records_g = []
for _, row in df_grpo.iterrows():
    raw_prompt = str(row["prompt"])
    msgs = [{"role": "user", "content": raw_prompt + PROMPT_SUFFIX}]
    try:
        prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                    add_generation_prompt=True,
                                                    enable_thinking=True)
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                    add_generation_prompt=True)
    records_g.append({
        "prompt": prompt_text,
        "answer": str(row["answer"]),
        # NOTE: keeping the raw user content in a separate column lets the reward
        # fn classify by puzzle type without re-stripping the chat template.
        "raw_prompt": raw_prompt,
    })

grpo_dataset = HFDataset.from_list(records_g)
print(f"\nGRPO records: {len(records_g)}")
"""),
    md(r"""
## Per-Type Accuracy Logging Callback

Wraps the trainer log step so per-type accuracy snapshots get pushed to
TensorBoard each logging interval. Tracks moving accuracy per puzzle type.
"""),
    code(r"""
from transformers import TrainerCallback

class PerTypeAccuracyCallback(TrainerCallback):
    def __init__(self, stats_dict):
        super().__init__()
        self.stats = stats_dict   # references _per_type_stats from reward cell

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        for ptype, s in self.stats.items():
            if s["n"] > 0:
                logs[f"acc/{ptype.replace(' ', '_').lower()}"] = s["correct"] / s["n"]
                logs[f"count/{ptype.replace(' ', '_').lower()}"] = s["n"]
        # Reset window every log step so we see per-window accuracy, not cumulative
        for s in self.stats.values():
            s["n"] = 0
            s["correct"] = 0
"""),
    md(r"""
## Dr. GRPO Trainer Config

Strict Dr. GRPO recipe from WINNING_PLAN.md Phase 1 + DAPO upgrades from
Phase 2 where TRL supports them.

| Param | Value | Why |
|---|---|---|
| `learning_rate` | 1e-6 | Dr. GRPO recipe (constant) |
| `lr_scheduler_type` | constant | Dr. GRPO recipe |
| `num_generations` | 8 | Dr. GRPO minimum |
| `temperature` | 1.0 | exploration during RL |
| `top_p` | 1.0 | no truncation |
| `beta` | 0.0 | no KL, no reference model — saves ~50% mem |
| `max_prompt_length` | 3072 | accommodates long puzzle prompts |
| `max_completion_length` | 3000 | Dr. GRPO recipe |
| `per_device_train_batch_size` | 1 | full group fits |
| `gradient_accumulation_steps` | 16 | effective batch = 16, divisible by num_gen=8 |
| `max_grad_norm` | 1.0 | not 0.1 (too tight) |
| `loss_type` | `dr_grpo` (or `dapo` fallback) | drop length norm |
| `scale_rewards` | False | drop std norm |
| `mask_truncated_completions` | True | exclude truncated from loss |
| `epsilon_high` | 0.28 | DAPO clip-higher |
"""),
    code(r"""
import inspect, gc, time, torch
from trl import GRPOTrainer, GRPOConfig

_cfg_params = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
HAS_SCALE_REWARDS = "scale_rewards" in _cfg_params
HAS_LOSS_TYPE     = "loss_type"     in _cfg_params
HAS_MASK_TRUNC    = "mask_truncated_completions" in _cfg_params
HAS_EPSILON_HIGH  = "epsilon_high"  in _cfg_params

print(f"TRL flags: scale_rewards={HAS_SCALE_REWARDS} loss_type={HAS_LOSS_TYPE} "
      f"mask_trunc={HAS_MASK_TRUNC} eps_high={HAS_EPSILON_HIGH}")

grpo_kwargs = dict(
    output_dir                   = os.path.join(OUTPUT_ROOT, "grpo_run"),
    num_train_epochs             = 1,
    per_device_train_batch_size  = 1,
    gradient_accumulation_steps  = 16,        # effective batch = 16
    learning_rate                = 1e-6,      # Dr. GRPO recipe
    lr_scheduler_type            = "constant",
    warmup_ratio                 = 0.0,
    max_prompt_length            = GRPO_MAX_PROMPT,
    max_completion_length        = GRPO_MAX_COMPLETION,
    num_generations              = 8,
    temperature                  = 1.0,
    top_p                        = 1.0,
    beta                         = 0.0,
    optim                        = "paged_adamw_8bit",
    adam_beta1                   = 0.9,
    adam_beta2                   = 0.999,
    weight_decay                 = 0.0,        # Dr. GRPO: no weight decay
    max_grad_norm                = 1.0,
    logging_steps                = 1,
    logging_dir                  = TB_LOG_DIR,
    report_to                    = "tensorboard",
    save_strategy                = "no",
    bf16                         = True,
    gradient_checkpointing       = True,
    gradient_checkpointing_kwargs= {"use_reentrant": False},
    seed                         = SEED,
    remove_unused_columns        = False,
)

# Apply Dr. GRPO / DAPO knobs where TRL supports them
if HAS_SCALE_REWARDS:
    grpo_kwargs["scale_rewards"] = False                  # Dr. GRPO: drop std norm
if HAS_LOSS_TYPE:
    # Prefer 'dr_grpo' if available, fall back to 'dapo' (both remove length bias)
    try:
        GRPOConfig(loss_type="dr_grpo")
        grpo_kwargs["loss_type"] = "dr_grpo"
    except Exception:
        grpo_kwargs["loss_type"] = "dapo"
if HAS_MASK_TRUNC:
    grpo_kwargs["mask_truncated_completions"] = True
if HAS_EPSILON_HIGH:
    grpo_kwargs["epsilon"]      = 0.2
    grpo_kwargs["epsilon_high"] = 0.28                    # DAPO clip-higher

grpo_config = GRPOConfig(**grpo_kwargs)

print("\n" + "=" * 60)
print("  Dr. GRPO RLVR CONFIG (RTX 6000 Pro v13)")
print("=" * 60)
print(f"  LR (const):     {grpo_config.learning_rate}")
print(f"  Num gen:        {grpo_config.num_generations}")
print(f"  Beta (KL):      {grpo_config.beta}")
print(f"  Max comp len:   {grpo_config.max_completion_length}")
print(f"  Temperature:    {grpo_config.temperature}")
print(f"  Batch:          {grpo_config.per_device_train_batch_size} x {grpo_config.gradient_accumulation_steps} = "
      f"{grpo_config.per_device_train_batch_size*grpo_config.gradient_accumulation_steps} eff")
print(f"  scale_rewards:  {getattr(grpo_config, 'scale_rewards', 'n/a')}")
print(f"  loss_type:      {getattr(grpo_config, 'loss_type', 'n/a')}")
print(f"  mask_trunc:     {getattr(grpo_config, 'mask_truncated_completions', 'n/a')}")
print(f"  epsilon_high:   {getattr(grpo_config, 'epsilon_high', 'n/a')}")
print("=" * 60 + "\n")
"""),
    md("## Launch GRPO Training"),
    code(r"""
import gc, time, torch

# The combined reward needs `prompt` access for puzzle-type classification.
# We pass raw prompts via the dataset column 'raw_prompt'; TRL forwards
# extra dataset columns to the reward fn via **kwargs. Wrap to inject.

def combined_reward_with_prompt(completions, answer, raw_prompt=None, **kwargs):
    return combined_reward(completions, answer=answer, prompt=raw_prompt)

grpo_trainer = GRPOTrainer(
    model            = model,
    args             = grpo_config,
    train_dataset    = grpo_dataset,
    reward_funcs     = [combined_reward_with_prompt],
    processing_class = tokenizer,
    callbacks        = [PerTypeAccuracyCallback(_per_type_stats)],
)

torch.cuda.empty_cache(); gc.collect()

print("Starting Dr. GRPO RLVR training...")
t0 = time.time()
grpo_trainer.train()
print(f"Dr. GRPO done in {(time.time()-t0)/60:.1f} min")

os.makedirs(GRPO_ADAPTER_DIR, exist_ok=True)
model.save_pretrained(GRPO_ADAPTER_DIR)
tokenizer.save_pretrained(GRPO_ADAPTER_DIR)
print(f"GRPO adapter saved -> {GRPO_ADAPTER_DIR}")
"""),
    md(r"""
## Package `submission.zip`

Copies adapter files from `GRPO_ADAPTER_DIR`, patches
`inference_mode=True`, `lora_dropout=0.0`, and base model name. Final
artifact: `outputs/rtx6000_v13/submission.zip`.
"""),
    code(r"""
import json, shutil, zipfile

os.makedirs(SUBMISSION_DIR, exist_ok=True)
required = ["adapter_config.json", "adapter_model.safetensors"]
for fname in required:
    sp = os.path.join(GRPO_ADAPTER_DIR, fname)
    dp = os.path.join(SUBMISSION_DIR, fname)
    if not os.path.exists(sp):
        raise FileNotFoundError(f"Missing: {sp}")
    shutil.copy2(sp, dp)
    print(f"  copied {fname}  ({os.path.getsize(dp)/1024/1024:.1f} MB)")

cfg_path = os.path.join(SUBMISSION_DIR, "adapter_config.json")
with open(cfg_path) as f: cfg = json.load(f)
cfg["base_model_name_or_path"] = BASE_MODEL_NAME
cfg["inference_mode"] = True
cfg["lora_dropout"]   = 0.0
with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)

zip_path = os.path.join(OUTPUT_ROOT, "submission.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in required:
        zf.write(os.path.join(SUBMISSION_DIR, fname), fname)
print(f"\nsubmission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB — ready.")
"""),
    md(r"""
## Notes

- **TensorBoard**: `tensorboard --logdir outputs/rtx6000_v13/tb_logs_grpo`
- **Per-type accuracy**: shows up as `acc/<puzzle_type>` metrics
- **Watch metrics**:
  - `reward` — should rise (overall learning)
  - `completions/clipped_ratio` — should drop (less truncation)
  - `frac_reward_zero_std` — should drop (more learning signal per group)
  - `acc/bit_manipulation`, `acc/text_encryption`, etc. — per-type progress
- **OOM fallback**: drop `num_generations` 8→4, `max_completion_length` 3000→1536, or `gradient_accumulation_steps` 16→8
- **Cold-start fallback**: if SFT adapter is missing, the load cell falls back to a fresh LoRA — slower but works
- **vLLM smoke test**: load the resulting `submission.zip` through a local vLLM + LoRA inference test before uploading (vLLM+LoRA+Mamba is fragile)
"""),
]


# ============================================================
# Emit notebooks
# ============================================================
def main():
    sft_path = REPO_ROOT / "sft" / "nemo-v13-sft-rtx6000.ipynb"
    grpo_path = REPO_ROOT / "post-training" / "nemo-v13-drgrpo-rtx6000.ipynb"

    print("Writing notebooks:")
    write_nb(sft_path, SFT_CELLS)
    write_nb(grpo_path, GRPO_CELLS)
    print("Done.")


if __name__ == "__main__":
    main()
