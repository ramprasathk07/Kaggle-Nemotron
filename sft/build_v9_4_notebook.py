"""Build sft/nemo-v9-4-SDLoRA.ipynb.

SDLoRA-only variant of v9-3 (PiSSA stripped). Per-layer rank from SVD
explained variance, but standard LoRA init (Kaiming A, zero B) — no PiSSA
warm-start and no PiSSA->LoRA conversion at save time. Avoids the PiSSA
init failure modes that may have killed the v9-3 Kaggle run silently.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELLS: list[dict] = []

# ----------------------------------------------------------------------------
CELLS.append(md("## Mode Selection"))

CELLS.append(code("""import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="strict")

TRAIN_ON_KAGGLE = 1
USE_PRETRAINED  = 0

assert (TRAIN_ON_KAGGLE + USE_PRETRAINED) == 1, \\
    "Set exactly one of TRAIN_ON_KAGGLE / USE_PRETRAINED to 1."

PRETRAINED_ADAPTER_DATASET_PATH = "/kaggle/input/datasets/konbu17/nemotron-sft-lora-cot-selection"
BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

print({"TRAIN_ON_KAGGLE": TRAIN_ON_KAGGLE, "USE_PRETRAINED": USE_PRETRAINED})"""))

# ----------------------------------------------------------------------------
CELLS.append(md("## Setup"))

CELLS.append(code("""import os, glob, sys, subprocess, site

candidates = glob.glob("/kaggle/input/**/*triton*.whl", recursive=True)
if not candidates:
    raise FileNotFoundError("No Triton wheel found under /kaggle/input")
wheel  = candidates[0]
target = "/kaggle/working/pydeps"
os.makedirs(target, exist_ok=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-deps",
     "--target", target, "--upgrade", "--ignore-installed", wheel],
    check=True,
)
if target not in sys.path:
    sys.path.insert(0, target)
site.addsitedir(target)

import importlib.util
print("triton spec:", importlib.util.find_spec("triton"))"""))

CELLS.append(code("""if TRAIN_ON_KAGGLE:
    import sys, os, shutil, stat
    sys.path.insert(0, '/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script')
    ptxas_src = '/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/triton/backends/nvidia/bin/ptxas-blackwell'
    ptxas_dst = '/tmp/ptxas-blackwell'
    if os.path.exists(ptxas_src) and not os.path.exists(ptxas_dst):
        shutil.copy2(ptxas_src, ptxas_dst)
        os.chmod(ptxas_dst, os.stat(ptxas_dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        src_bin = os.path.dirname(ptxas_src)
        dst_bin = '/tmp/triton_nvidia_bin'
        shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
        for f in os.listdir(dst_bin):
            fp = os.path.join(dst_bin, f)
            if os.path.isfile(fp):
                os.chmod(fp, os.stat(fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        os.environ['TRITON_PTXAS_BLACKWELL_PATH'] = ptxas_dst
        import triton.backends.nvidia as nv_backend
        nv_backend.__file__ = os.path.join(dst_bin, '..', '__init__.py')
        os.environ['TRITON_PTXAS_PATH'] = ptxas_dst
    import triton.backends.nvidia.compiler as nv_compiler
    nv_compiler.get_ptxas_version = lambda arch: '12.0'
    print('ptxas env fixes applied.')"""))

CELLS.append(code("""if TRAIN_ON_KAGGLE:
    import glob, os, subprocess, sys

    def recursive_wheels(pat):
        return sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True))

    packages_dir = "/kaggle/input/datasets/mayukh18/nemotron-packages/packages"
    all_mamba    = recursive_wheels("mamba_ssm-*.whl")
    all_causal   = recursive_wheels("causal*conv1d*.whl")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required.")
    if not os.path.isdir(packages_dir):
        raise FileNotFoundError(f"Wheel dir not found: {packages_dir}")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "--no-index", "--find-links", packages_dir,
         "unsloth", "trl", "peft", "transformers",
         "datasets", "accelerate", "bitsandbytes"],
        check=True,
    )
    def pick_last(ws): return ws[-1] if ws else None
    for w in filter(None, [pick_last(all_causal), pick_last(all_mamba)]):
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", w], check=True)
    print("Packages installed.")"""))

# ----------------------------------------------------------------------------
CELLS.append(md("## Model Loading"))

CELLS.append(code("""if TRAIN_ON_KAGGLE:
    import torch
    import kagglehub
    from unsloth import FastLanguageModel

    MAX_SEQ_LEN = 8192
    MODEL_PATH  = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
    print(f"Model path: {MODEL_PATH}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=False,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("Base model loaded. LoRA will be applied after SDLoRA calibration.")
else:
    print("USE_PRETRAINED=1: skipping model loading.")"""))

# ----------------------------------------------------------------------------
CELLS.append(md("""## SDLoRA Calibration — Per-Layer Rank via SVD Explained Variance

**Method (SDLoRA = Selective Decomposition LoRA):**

Standard LoRA assigns a uniform rank `r` to every layer. SDLoRA computes an
optimal, layer-specific rank from the weight matrix's singular value spectrum:

1. For each target linear layer `W ∈ ℝ^{m×n}`, compute the top-k singular values
   `S = [σ₁ ≥ σ₂ ≥ ... ≥ σₖ]` via fast randomized SVD (`torch.svd_lowrank`, niter=4).

2. Compute cumulative explained variance:
   `EV(r) = Σᵢ₌₁ʳ σᵢ² / Σᵢ₌₁ᵏ σᵢ²`

3. Choose minimum rank `r*` such that `EV(r*) ≥ threshold` (default 0.85).
   Clip to `[min_rank, max_rank]`.

**Rationale:**
- Layers with a steep singular value drop-off (fast EV saturation) carry most
  information in a low-dimensional subspace → small r suffices.
- Layers with gradual decay (information spread across many singular values)
  need higher r to cover the relevant subspace.
- This avoids both under-allocation (missed subspaces) and over-allocation
  (wasted parameters on noise directions).

**Init:** standard Kaiming-uniform for `A`, zero for `B` (no PiSSA). Removes the
PiSSA SVD init pass that was suspected of failing silently on v9-3, and removes
the need for a `pissa_init.safetensors` residual file at inference time."""))

CELLS.append(code("""if TRAIN_ON_KAGGLE:
    import torch

    # ============================================================
    # SDLoRA CONFIGURATION
    # ============================================================
    EV_THRESHOLD = 0.85   # explained variance required per layer
    MIN_RANK     = 4      # floor — keeps small adapters meaningful
    MAX_RANK     = 32     # competition ceiling
    DEFAULT_RANK = 16     # fallback for layers not found in calibration

    TARGET_MODULES = [
        # Attention (6 GQA blocks)
        "q_proj", "k_proj", "v_proj", "o_proj",
        # Mamba-2 SSM (23 blocks) — in_proj/out_proj handle input/output mixing
        "in_proj", "out_proj",
        # MoE FFN (23 blocks)
        "gate_proj", "up_proj", "down_proj",
        # lm_head EXCLUDED: drifts \\boxed{} token distribution
    ]

    def _svd_rank(weight: torch.Tensor, ev_threshold: float,
                  min_rank: int, max_rank: int) -> tuple[int, float]:
        \"\"\"
        Find minimum rank r such that top-r singular values explain
        >= ev_threshold of total variance. Returns (rank, achieved_ev).
        \"\"\"
        W = weight.detach().float().cpu()
        if W.dim() > 2:
            W = W.flatten(1)

        m, n   = W.shape
        k      = min(max_rank + 8, min(m, n))   # compute a few extra for EV curve

        try:
            _, S, _ = torch.svd_lowrank(W, q=k, niter=4)
        except Exception:
            S = torch.linalg.svdvals(W)[:k]

        S          = S.float()
        total_var  = (S ** 2).sum()

        if total_var < 1e-8:
            return min_rank, 0.0

        cumev = torch.cumsum(S ** 2, dim=0) / total_var
        hits  = (cumev >= ev_threshold).nonzero(as_tuple=True)[0]
        r     = int(hits[0].item()) + 1 if len(hits) > 0 else max_rank
        r     = max(min_rank, min(max_rank, r))
        return r, float(cumev[min(r, len(cumev)) - 1].item())

    def build_sdlora_rank_config(
        model,
        target_modules: list[str],
        ev_threshold: float = 0.85,
        min_rank: int       = 4,
        max_rank: int       = 32,
    ) -> tuple[dict, dict, list]:
        \"\"\"
        Iterate all target linear layers, compute SVD-based rank,
        return (rank_pattern, alpha_pattern, stats_rows).

        rank_pattern  : {full_module_name: rank} for LoraConfig
        alpha_pattern : {full_module_name: 2*rank}  alpha=2*rank (standard LoRA convention)
        stats_rows    : list of (name, rank, σ₁, σᵣ, ev) for reporting
        \"\"\"
        target_set   = set(target_modules)
        rank_pattern = {}
        stats        = []

        modules = list(model.named_modules())
        total   = sum(
            1 for name, mod in modules
            if name.split(".")[-1] in target_set
            and hasattr(mod, "weight") and mod.weight is not None
        )
        done = 0

        for name, module in modules:
            if name.split(".")[-1] not in target_set:
                continue
            if not hasattr(module, "weight") or module.weight is None:
                continue

            r, ev = _svd_rank(module.weight, ev_threshold, min_rank, max_rank)
            rank_pattern[name] = r

            W = module.weight
            stats.append({
                "name":  name,
                "shape": tuple(W.shape),
                "rank":  r,
                "ev":    ev,
            })

            done += 1
            if done % 20 == 0 or done == total:
                print(f"  Calibrated {done}/{total} layers ...")

        # Standard LoRA scaling: alpha = 2 * rank → effective scale = 2.0
        alpha_pattern = {k: 2 * v for k, v in rank_pattern.items()}
        return rank_pattern, alpha_pattern, stats

    print("SDLoRA calibration functions defined.")
    print(f"  EV threshold : {EV_THRESHOLD}")
    print(f"  Rank range   : [{MIN_RANK}, {MAX_RANK}]")
    print(f"  Targets      : {TARGET_MODULES}")"""))

CELLS.append(code("""if TRAIN_ON_KAGGLE:
    print("Running SVD calibration across all target layers ...")
    print("(Uses fast randomized SVD via torch.svd_lowrank, niter=4)")
    print()

    rank_pattern, alpha_pattern, stats = build_sdlora_rank_config(
        model,
        TARGET_MODULES,
        ev_threshold=EV_THRESHOLD,
        min_rank=MIN_RANK,
        max_rank=MAX_RANK,
    )

    # ---- Report ----
    from collections import Counter
    rank_counts = Counter(s["rank"] for s in stats)

    print(f"\\nSDLoRA calibration done — {len(stats)} layers")
    print(f"  Rank distribution : {dict(sorted(rank_counts.items()))}")
    print(f"  Avg rank          : {sum(s['rank'] for s in stats)/len(stats):.1f}")
    print(f"  Avg EV covered    : {sum(s['ev'] for s in stats)/len(stats):.3f}")
    print()

    # Sample: show first 12 layers
    print(f"{'Module':<60} {'Shape':<18} {'Rank':>6} {'EV':>6}")
    print("-" * 94)
    for s in stats[:12]:
        name_short = s["name"][-58:] if len(s["name"]) > 58 else s["name"]
        print(f"{name_short:<60} {str(s['shape']):<18} {s['rank']:>6} {s['ev']:>6.3f}")
    if len(stats) > 12:
        print(f"  ... and {len(stats)-12} more layers")
else:
    print("USE_PRETRAINED=1: skipping SDLoRA calibration.")"""))

# ----------------------------------------------------------------------------
CELLS.append(md("## LoRA Construction — Standard Init (no PiSSA)"))

CELLS.append(code("""if TRAIN_ON_KAGGLE:
    from peft import LoraConfig, get_peft_model, TaskType

    # ============================================================
    # SDLoRA LoRA CONFIG (standard init, no PiSSA)
    # ============================================================
    # - rank_pattern  : per-layer ranks from SVD calibration
    # - alpha_pattern : alpha = 2*rank → standard LoRA scaling = 2.0
    # - init_lora_weights=True : Kaiming-uniform A, zero B (PEFT default)
    # - use_rslora=False : keep alpha/rank scaling (not alpha/sqrt(rank))
    # ============================================================
    lora_config = LoraConfig(
        r=DEFAULT_RANK,
        lora_alpha=2 * DEFAULT_RANK,     # fallback alpha for unlisted layers
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        rank_pattern=rank_pattern,       # per-layer ranks from SVD calibration
        alpha_pattern=alpha_pattern,     # alpha = 2*rank per layer
        init_lora_weights=True,          # Kaiming A, zero B — standard LoRA
        bias="none",
        use_rslora=False,
        task_type=TaskType.CAUSAL_LM,
    )

    model.enable_input_require_grads()   # required for grad checkpointing with PEFT
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print()
    print("SDLoRA adapter initialized (standard LoRA init).")
else:
    print("USE_PRETRAINED=1: skipping LoRA construction.")"""))

# ----------------------------------------------------------------------------
CELLS.append(md("## SFT Training"))

CELLS.append(code("""# ============================================================
# MEMORY OPTIMIZATIONS
# ============================================================
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import gc
import re
import subprocess
import time

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

# ============================================================
# GPU METRICS CALLBACK (TensorBoard)
# ============================================================
class GPUMetricsCallback(TrainerCallback):
    def __init__(self, log_every_n_steps=2):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._last_step_time   = None
        self._last_global_step = 0

    def _query_smi(self):
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            if r.returncode != 0: return None
            p = [x.strip() for x in r.stdout.strip().split("\\n")[0].split(",")]
            return {
                "gpu/utilization_percent": float(p[0]),
                "gpu/memory_used_gb":      float(p[1]) / 1024.0,
                "gpu/temperature_celsius": float(p[3]),
                "gpu/power_watts":         float(p[4]) if p[4] != "[N/A]" else 0.0,
            }
        except Exception:
            return None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step % self.log_every_n_steps != 0:
            return
        smi = self._query_smi()
        if smi: logs.update(smi)
        if torch.cuda.is_available():
            logs["gpu/memory_allocated_gb"]     = torch.cuda.memory_allocated()     / (1024**3)
            logs["gpu/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        now = time.time()
        if self._last_step_time is not None:
            elapsed = now - self._last_step_time
            steps   = state.global_step - self._last_global_step
            if elapsed > 0 and steps > 0:
                sps = steps / elapsed
                logs["throughput/steps_per_sec"]   = sps
                logs["throughput/samples_per_sec"] = sps * args.per_device_train_batch_size
        self._last_step_time   = now
        self._last_global_step = state.global_step

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_step_time   = time.time()
        self._last_global_step = state.global_step
        smi = self._query_smi()
        if smi:
            print(f"[GPU] {smi['gpu/utilization_percent']:.0f}% util | "
                  f"{smi['gpu/memory_used_gb']:.1f} GB | {smi['gpu/temperature_celsius']:.0f}C")

    def on_train_end(self, args, state, control, **kwargs):
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[GPU] Done. Peak VRAM: {peak:.1f} GB")

print("Imports and GPUMetricsCallback ready.")"""))

CELLS.append(code("""if TRAIN_ON_KAGGLE:
    # ============================================================
    # DATA LOADING
    # ============================================================
    SEED = 42
    PROMPT_SUFFIX = "\\nPlease put your final answer inside `\\\\boxed{}`. For example: `\\\\boxed{your answer}`"

    DATASET_PATH = "/kaggle/input/datasets/dgxchen/nemotron-cot-tong/problem_ids_matched.csv"
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["prompt", "answer", "generated_cot"]).reset_index(drop=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"Dataset: {len(df)} rows")

    records, skipped = [], 0
    for _, row in df.iterrows():
        cot = str(row["generated_cot"])
        if cot == "nan" or len(cot.strip()) < 5 or len(cot.strip()) > 8100:
            skipped += 1
            continue
        cot_clean = re.sub(r'\\\\boxed\\{[^}]*\\}', '', cot).rstrip()
        records.append({
            "messages": [
                {"role": "user",      "content": str(row["prompt"]) + PROMPT_SUFFIX},
                {"role": "assistant", "content": f"<think>\\n{cot_clean}\\n</think>\\n\\\\boxed{{{str(row['answer'])}}}"},
            ]
        })

    dataset = HFDataset.from_list(records)
    print(f"SFT records: {len(records)} (skipped {skipped})")

    def formatting_prompts_func(example):
        messages = example["messages"]
        convos   = [messages] if isinstance(messages[0], dict) else messages
        texts    = []
        for convo in convos:
            try:
                t = tokenizer.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False, enable_thinking=True)
            except TypeError:
                t = tokenizer.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False)
            texts.append(t)
        return texts

    # ============================================================
    # TRAINING CONFIG — SDLoRA v9.4 (standard LoRA init)
    # ============================================================
    TB_LOG_DIR = "/kaggle/working/tb_logs"

    training_args = SFTConfig(
        output_dir="/kaggle/working/sft_output",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,          # standard LoRA LR (no PiSSA warm-start)
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=8192,
        optim="adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.005,
        max_grad_norm=0.5,
        neftune_noise_alpha=5.0,
        logging_steps=2,
        logging_dir=TB_LOG_DIR,
        report_to="tensorboard",
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=SEED,
        packing=False,
        dataset_num_proc=4,
    )

    print("\\n" + "=" * 60)
    print("  SDLoRA SFT CONFIG v9.4 (standard init, no PiSSA)")
    print("=" * 60)
    print(f"  LR:        {training_args.learning_rate}")
    print(f"  Warmup:    {training_args.warmup_ratio}")
    print(f"  NEFTune:   {training_args.neftune_noise_alpha}")
    bs = training_args.per_device_train_batch_size
    ga = training_args.gradient_accumulation_steps
    print(f"  Batch:     {bs} × {ga} = {bs * ga}")
    print(f"  Epochs:    {training_args.num_train_epochs}")
    print("=" * 60 + "\\n")

    # ============================================================
    # TRAIN
    # ============================================================
    torch.cuda.empty_cache()
    gc.collect()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
        callbacks=[GPUMetricsCallback(log_every_n_steps=2)],
    )

    print("Starting SDLoRA SFT training v9.4 ...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Training done in {elapsed / 60:.1f} min")

    # ============================================================
    # SAVE — standard LoRA adapter (no PiSSA residual needed)
    # ============================================================
    ADAPTER_DIR = "/kaggle/working/sft_adapter"
    model.save_pretrained(ADAPTER_DIR, safe_serialization=True)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Adapter saved to {ADAPTER_DIR}")
    FINAL_ADAPTER_DIR = ADAPTER_DIR
else:
    FINAL_ADAPTER_DIR = PRETRAINED_ADAPTER_DATASET_PATH
    print("USE_PRETRAINED=1: skipping SFT training.")"""))

# ----------------------------------------------------------------------------
CELLS.append(md("## Package TensorBoard Logs"))

CELLS.append(code("""if TRAIN_ON_KAGGLE:
    import os, zipfile
    from pathlib import Path

    TB_LOG_DIR = "/kaggle/working/tb_logs"
    ZIP_OUTPUT = "/kaggle/working/tensorboard_logs.zip"
    log_path   = Path(TB_LOG_DIR)
    if log_path.exists():
        files = [f for f in log_path.rglob("*") if f.is_file()]
        with zipfile.ZipFile(ZIP_OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in files:
                zf.write(fp, fp.relative_to(log_path.parent))
        print(f"TB logs => {ZIP_OUTPUT} ({os.path.getsize(ZIP_OUTPUT)/1024/1024:.2f} MB)")
    else:
        print(f"[WARN] No TB logs at {TB_LOG_DIR}")"""))

# ----------------------------------------------------------------------------
CELLS.append(md("## Create submission.zip"))

CELLS.append(code("""import json, os, shutil, zipfile

OUTPUT_DIR             = "/kaggle/working"
SUBMISSION_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "submission_adapter")
os.makedirs(SUBMISSION_ADAPTER_DIR, exist_ok=True)

required_files = ["adapter_config.json", "adapter_model.safetensors"]

print(f"Packaging adapter from: {FINAL_ADAPTER_DIR}")

copied = []
for fname in required_files:
    src = os.path.join(FINAL_ADAPTER_DIR, fname)
    dst = os.path.join(SUBMISSION_ADAPTER_DIR, fname)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing required: {src}")
    shutil.copy2(src, dst)
    copied.append(fname)
    print(f"  Copied {fname} ({os.path.getsize(dst)/1024/1024:.1f} MB)")

config_path = os.path.join(SUBMISSION_ADAPTER_DIR, "adapter_config.json")
with open(config_path) as f:
    cfg = json.load(f)
cfg["base_model_name_or_path"] = BASE_MODEL_NAME
cfg["inference_mode"]          = True
cfg["lora_dropout"]            = 0.0
with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)

zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in copied:
        fpath = os.path.join(SUBMISSION_ADAPTER_DIR, fname)
        zf.write(fpath, fname)
        print(f"  Zipped {fname}")

print(f"\\nsubmission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
print("Adapter type  : standard LoRA")
print("Done.")"""))

# ----------------------------------------------------------------------------
NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    out = Path(__file__).resolve().parent / "nemo-v9-4-SDLoRA.ipynb"
    out.write_text(json.dumps(NOTEBOOK, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
