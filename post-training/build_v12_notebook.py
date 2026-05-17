"""
Build nemo-v12-GRPO-submission-realistic.ipynb.

GRPO recipe aligned to the contest's actual scoring pipeline.

Hard constraints (per the competition rules):
  * Submission is ONLY a LoRA adapter zip (adapter_config.json + adapter_model.safetensors).
  * Backend evaluates with vLLM. Participants have NO control over inference-time
    knobs (temperature, top-p, best-of-N, etc.). Only the adapter weights ship.
  * Scoring metric: accuracy on `\\boxed{<answer>}`. Exact string match (text) or
    +/- 1e-2 tolerance (numeric).
  * `test.csv` contains `id, prompt` only — NO ground-truth answers. So test
    prompts are usable for monitoring/probing during training, NOT for reward.

Design implications baked into v12:
  1. Reward is correctness-dominated. Format / length / repetition signals are
     reduced to hard guardrails (penalties on degenerate generations) rather
     than bonuses that the model can game.
  2. A `TestSetProbeCallback` runs the live policy on a small sample of test
     prompts every N steps and records: % with extractable \\boxed{}, mean
     completion length, self-consistency rate. These metrics drive early stop /
     drift detection.
  3. Curriculum filter (where pass@1 ∈ [0.25, 0.85]) uses train.csv only —
     test.csv has no labels.
  4. Inference-time tricks are intentionally absent. The only output is a single
     adapter zip.

Pipeline:
  Stage 0  Triton / ptxas / offline wheel setup
  Stage 1  Load Nemotron base + LoRA, warm-start from SFT-85 adapter
  Stage 2  Optional micro-SFT on Sonnet 4.6 thinking traces (1 epoch, LR 1e-5)
  Stage 3  Curriculum filter on labeled train.csv
  Stage 4  GRPO with vLLM colocate + TestSetProbeCallback
  Stage 5  Final test-set probe report
  Stage 6  Save adapter + zip submission
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "nemo-v12-GRPO-submission-realistic.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = []

# -----------------------------------------------------------------------------
cells.append(md(
    "# Nemotron-3-Nano-30B — GRPO, Submission-Realistic (v12)\n"
    "\n"
    "GRPO polish for an 85 %% SFT baseline. Designed for **1× RTX 6000 Pro 96 GB**.\n"
    "\n"
    "## What ships\n"
    "- ONLY the LoRA adapter (`adapter_config.json` + `adapter_model.safetensors`)\n"
    "  inside `submission.zip`.\n"
    "- Backend runs vLLM inference. We do NOT control temperature, sampling, or\n"
    "  ensembling at submit-time.\n"
    "- Scoring: accuracy on `\\boxed{<answer>}` (string match for text,\n"
    "  ±1e-2 for numbers).\n"
    "\n"
    "## Consequences for training\n"
    "- Reward is **correctness-aligned** — format/length tweaks are guardrails,\n"
    "  not bonuses, to prevent reward hacking that wouldn't transfer.\n"
    "- `test.csv` has prompts only (no answers). Used here for **monitoring**\n"
    "  during training, not for reward.\n"
    "- Curriculum filter uses `train.csv` (labeled) only.\n"
    "- We probe the live policy on test prompts every N steps and watch:\n"
    "  `% extractable \\boxed{}`, mean completion length, self-consistency.\n"
    "  Drift on these = early-stop signal.\n"
    "\n"
    "## Stages\n"
    "1. Triton / ptxas / offline wheels (Kaggle).\n"
    "2. LoRA wrap matching SFT-85 layout; warm-start from SFT-85 adapter.\n"
    "3. (Optional) Micro-SFT on 4 k Sonnet 4.6 thinking traces, LR 1e-5, 1 epoch.\n"
    "4. Curriculum filter on labeled train.csv (pass@1 ∈ [0.25, 0.85]).\n"
    "5. GRPO with vLLM colocate + `TestSetProbeCallback`.\n"
    "6. Final test probe report.\n"
    "7. Save adapter + zip submission.\n"
))

# -----------------------------------------------------------------------------
cells.append(md("## Mode Selection"))

cells.append(code(
    'import os, sys\n'
    'os.environ["PYTHONIOENCODING"] = "utf-8"\n'
    'os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"\n'
    'os.environ["TORCH_COMPILE_DISABLE"] = "1"\n'
    'os.environ["TORCHINDUCTOR_DISABLE"] = "1"\n'
    'os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"\n'
    'if hasattr(sys.stdout, "reconfigure"):\n'
    '    sys.stdout.reconfigure(encoding="utf-8", errors="strict")\n'
    'if hasattr(sys.stderr, "reconfigure"):\n'
    '    sys.stderr.reconfigure(encoding="utf-8", errors="strict")\n'
    '\n'
    'TRAIN_ON_KAGGLE = 1\n'
    'USE_PRETRAINED  = 0\n'
    'assert (TRAIN_ON_KAGGLE + USE_PRETRAINED) == 1\n'
    '\n'
    '# ---- Stage toggles -------------------------------------------------------\n'
    'DO_MICRO_SFT_SONNET   = 1\n'
    'DO_CURRICULUM_FILTER  = 1\n'
    'DO_GRPO               = 1\n'
    'DO_TEST_PROBE         = 1   # Eval probes on test.csv during/after training\n'
    '\n'
    '# ---- Paths ---------------------------------------------------------------\n'
    'SFT85_ADAPTER_PATH = os.environ.get(\n'
    '    "SFT85_ADAPTER_PATH",\n'
    '    "/kaggle/input/datasets/your-username/nemotron-sft-85",     # ← edit\n'
    ')\n'
    'SONNET_TRACES_PATH = os.environ.get(\n'
    '    "SONNET_TRACES_PATH",\n'
    '    "/kaggle/input/datasets/your-username/sonnet46-thinking-4k.csv",  # ← edit\n'
    ')\n'
    '\n'
    '# Labeled training prompts (id, prompt, answer[, type, generated_cot])\n'
    'TRAIN_DATASET_PATH = os.environ.get(\n'
    '    "TRAIN_DATASET_PATH",\n'
    '    "/kaggle/input/datasets/dgxchen/nemotron-cot-tong/problem_ids_matched.csv",\n'
    ')\n'
    '# Official Kaggle test prompts (id, prompt) — answers NOT included.\n'
    'TEST_DATASET_PATH = os.environ.get(\n'
    '    "TEST_DATASET_PATH",\n'
    '    "/kaggle/input/nvidia-nemotron-model-reasoning-challenge/test.csv",\n'
    ')\n'
    '\n'
    'BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"\n'
    'PRETRAINED_ADAPTER_DATASET_PATH = SFT85_ADAPTER_PATH\n'
    '\n'
    'print({\n'
    '    "TRAIN_ON_KAGGLE": TRAIN_ON_KAGGLE,\n'
    '    "DO_MICRO_SFT_SONNET": DO_MICRO_SFT_SONNET,\n'
    '    "DO_CURRICULUM_FILTER": DO_CURRICULUM_FILTER,\n'
    '    "DO_GRPO": DO_GRPO,\n'
    '    "DO_TEST_PROBE": DO_TEST_PROBE,\n'
    '    "SFT85_ADAPTER_PATH": SFT85_ADAPTER_PATH,\n'
    '    "SONNET_TRACES_PATH": SONNET_TRACES_PATH,\n'
    '    "TRAIN_DATASET_PATH": TRAIN_DATASET_PATH,\n'
    '    "TEST_DATASET_PATH": TEST_DATASET_PATH,\n'
    '})\n'
))

# -----------------------------------------------------------------------------
cells.append(md("## Setup"))

cells.append(code(
    'import os, glob, sys, subprocess, site\n'
    '\n'
    'candidates = glob.glob("/kaggle/input/**/*triton*.whl", recursive=True)\n'
    'print("Found Triton wheels:", candidates)\n'
    'if not candidates:\n'
    '    raise FileNotFoundError("No Triton wheel found under /kaggle/input")\n'
    'wheel = candidates[0]\n'
    'target = "/kaggle/working/pydeps"\n'
    'os.makedirs(target, exist_ok=True)\n'
    'subprocess.run(\n'
    '    [sys.executable, "-m", "pip", "install", "--no-deps", "--target", target,\n'
    '     "--upgrade", "--ignore-installed", wheel],\n'
    '    check=True,\n'
    ')\n'
    'if target not in sys.path:\n'
    '    sys.path.insert(0, target)\n'
    'site.addsitedir(target)\n'
    'import importlib.util\n'
    'print("triton spec:", importlib.util.find_spec("triton"))\n'
))

cells.append(code(
    'if TRAIN_ON_KAGGLE:\n'
    '    import sys, os, shutil, stat\n'
    "    sys.path.insert(0, '/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script')\n"
    "    ptxas_src = '/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/triton/backends/nvidia/bin/ptxas-blackwell'\n"
    "    ptxas_dst = '/tmp/ptxas-blackwell'\n"
    '    if os.path.exists(ptxas_src) and not os.path.exists(ptxas_dst):\n'
    '        shutil.copy2(ptxas_src, ptxas_dst)\n'
    '        os.chmod(ptxas_dst, os.stat(ptxas_dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)\n'
    '        src_bin = os.path.dirname(ptxas_src)\n'
    "        dst_bin = '/tmp/triton_nvidia_bin'\n"
    '        shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)\n'
    '        for f in os.listdir(dst_bin):\n'
    '            fp = os.path.join(dst_bin, f)\n'
    '            if os.path.isfile(fp):\n'
    '                os.chmod(fp, os.stat(fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)\n'
    "        os.environ['TRITON_PTXAS_BLACKWELL_PATH'] = ptxas_dst\n"
    '        import triton.backends.nvidia as nv_backend\n'
    "        nv_backend.__file__ = os.path.join(dst_bin, '..', '__init__.py')\n"
    "        os.environ['TRITON_PTXAS_PATH'] = ptxas_dst\n"
    '    import triton.backends.nvidia.compiler as nv_compiler\n'
    "    nv_compiler.get_ptxas_version = lambda arch: '12.0'\n"
    "    print('Training environment fixes applied.')\n"
))

cells.append(code(
    'if TRAIN_ON_KAGGLE:\n'
    '    import glob, os, subprocess, sys\n'
    '    def recursive_wheels(pat):\n'
    '        return sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True))\n'
    '    packages_dir = "/kaggle/input/datasets/mayukh18/nemotron-packages/packages"\n'
    '    import torch\n'
    '    print("Torch:", torch.__version__, " CUDA:", torch.version.cuda)\n'
    '    if not torch.cuda.is_available():\n'
    '        raise RuntimeError("GPU required.")\n'
    '    if not os.path.isdir(packages_dir):\n'
    '        raise FileNotFoundError(packages_dir)\n'
    '    subprocess.run(\n'
    '        [sys.executable, "-m", "pip", "install", "-q",\n'
    '         "--no-index", "--find-links", packages_dir,\n'
    '         "unsloth", "trl", "peft", "transformers", "datasets",\n'
    '         "accelerate", "bitsandbytes"],\n'
    '        check=True,\n'
    '    )\n'
    '    VLLM_OK = False\n'
    '    try:\n'
    '        subprocess.run(\n'
    '            [sys.executable, "-m", "pip", "install", "-q",\n'
    '             "--no-index", "--find-links", packages_dir, "vllm"],\n'
    '            check=True,\n'
    '        )\n'
    '        import vllm  # noqa: F401\n'
    '        VLLM_OK = True\n'
    '        print("vLLM available — will use colocate rollouts.")\n'
    '    except Exception as e:\n'
    '        print(f"vLLM unavailable, fallback to HF generate: {e}")\n'
    '    all_mamba  = recursive_wheels("mamba_ssm-*.whl")\n'
    '    all_causal = recursive_wheels("causal*conv1d*.whl")\n'
    '    if all_causal:\n'
    '        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", all_causal[-1]], check=True)\n'
    '    if all_mamba:\n'
    '        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", all_mamba[-1]], check=True)\n'
    '    else:\n'
    '        raise FileNotFoundError("Missing mamba_ssm wheel.")\n'
    '    print("Offline package installation finished.")\n'
))

cells.append(code(
    'if TRAIN_ON_KAGGLE:\n'
    '    import torch, kagglehub\n'
    '    from unsloth import FastLanguageModel\n'
    '    try:\n'
    '        from unsloth import PatchFastRL\n'
    '        PatchFastRL("GRPO", FastLanguageModel)\n'
    '        print("PatchFastRL applied for GRPO.")\n'
    '    except (ImportError, AttributeError):\n'
    '        print("PatchFastRL not available; proceeding without it.")\n'
    '\n'
    '    MAX_SEQ_LEN = 8192\n'
    '    MODEL_PATH = kagglehub.model_download(\n'
    '        "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"\n'
    '    )\n'
    '    print("Model path:", MODEL_PATH)\n'
    '    model, tokenizer = FastLanguageModel.from_pretrained(\n'
    '        model_name=MODEL_PATH,\n'
    '        max_seq_length=MAX_SEQ_LEN,\n'
    '        load_in_4bit=False,\n'
    '        load_in_8bit=False,\n'
    '        full_finetuning=False,\n'
    '        trust_remote_code=True,\n'
    '        unsloth_force_compile=False,\n'
    '        attn_implementation="eager",\n'
    '        dtype=torch.bfloat16,\n'
    '    )\n'
    '    if tokenizer.pad_token is None:\n'
    '        tokenizer.pad_token = tokenizer.eos_token\n'
    '    print("Base model loaded.")\n'
))

# -----------------------------------------------------------------------------
cells.append(md("## LoRA Wrap + SFT-85 Warm-Start"))

cells.append(code(
    'if TRAIN_ON_KAGGLE:\n'
    '    from unsloth import FastLanguageModel\n'
    '    from safetensors.torch import load_file\n'
    '    import os, json\n'
    '\n'
    '    LORA_RANK    = 32\n'
    '    LORA_ALPHA   = 64\n'
    '    LORA_DROPOUT = 0.0\n'
    '    INCLUDE_MAMBA_LORA = True  # match your SFT-85 exactly\n'
    '\n'
    '    target_modules = [\n'
    '        "q_proj", "k_proj", "v_proj", "o_proj",\n'
    '        "in_proj", "out_proj",\n'
    '        "gate_proj", "up_proj", "down_proj",\n'
    '    ]\n'
    '    if INCLUDE_MAMBA_LORA:\n'
    '        target_modules += ["x_proj", "dt_proj"]\n'
    '\n'
    '    model = FastLanguageModel.get_peft_model(\n'
    '        model,\n'
    '        r=LORA_RANK,\n'
    '        lora_alpha=LORA_ALPHA,\n'
    '        lora_dropout=LORA_DROPOUT,\n'
    '        target_modules=target_modules,\n'
    '        bias="none",\n'
    '        use_gradient_checkpointing=True,\n'
    '        random_state=42,\n'
    '        use_rslora=True,\n'
    '    )\n'
    '    model.print_trainable_parameters()\n'
    '\n'
    '    adapter_file = os.path.join(SFT85_ADAPTER_PATH, "adapter_model.safetensors")\n'
    '    cfg_file     = os.path.join(SFT85_ADAPTER_PATH, "adapter_config.json")\n'
    '    if not os.path.exists(adapter_file):\n'
    '        raise FileNotFoundError(f"SFT-85 adapter not found: {adapter_file}")\n'
    '    with open(cfg_file) as f:\n'
    '        sft_cfg = json.load(f)\n'
    '    print("SFT-85 cfg:",\n'
    '          {k: sft_cfg.get(k) for k in ("r", "lora_alpha", "use_rslora", "lora_dropout", "target_modules")})\n'
    '    if int(sft_cfg.get("r", -1)) != LORA_RANK or int(sft_cfg.get("lora_alpha", -1)) != LORA_ALPHA:\n'
    '        print("WARNING: SFT adapter rank/alpha differs — reload may be partial.")\n'
    '\n'
    '    sd = load_file(adapter_file)\n'
    '    model_sd = model.state_dict()\n'
    '    renamed = {}\n'
    '    for k, v in sd.items():\n'
    '        if k in model_sd:\n'
    '            renamed[k] = v\n'
    '        else:\n'
    '            for adapt in ("lora_A", "lora_B"):\n'
    '                needle = f"{adapt}.weight"\n'
    '                if needle in k:\n'
    '                    cand = k.replace(needle, f"{adapt}.default.weight")\n'
    '                    if cand in model_sd:\n'
    '                        renamed[cand] = v\n'
    '                        break\n'
    '    missing, unexpected = model.load_state_dict(renamed, strict=False)\n'
    '    lora_missing    = [m for m in missing    if "lora_" in m]\n'
    '    lora_unexpected = [u for u in unexpected if "lora_" in u]\n'
    '    print(f"Warm-start: matched {len(renamed)} tensors, "\n'
    '          f"LoRA-missing={len(lora_missing)}, LoRA-unexpected={len(lora_unexpected)}")\n'
    '    if lora_missing:\n'
    '        print("First 5 missing LoRA params:", lora_missing[:5])\n'
))

# -----------------------------------------------------------------------------
cells.append(md(
    "## Stage 2 — (Optional) Micro-SFT on Sonnet 4.6 Thinking Traces\n"
    "\n"
    "Skip when `DO_MICRO_SFT_SONNET == 0`. One epoch, LR 1e-5 — grafts the\n"
    "reasoning style from premium traces *before* RL polish."
))

cells.append(code(
    'if TRAIN_ON_KAGGLE and DO_MICRO_SFT_SONNET:\n'
    '    import pandas as pd, gc, time, torch, re\n'
    '    from datasets import Dataset as HFDataset\n'
    '    from trl import SFTTrainer, SFTConfig\n'
    '\n'
    '    sdf = pd.read_csv(SONNET_TRACES_PATH)\n'
    '    cot_col = next(\n'
    '        (c for c in ("generated_cot", "reasoning", "cot", "thinking") if c in sdf.columns),\n'
    '        None,\n'
    '    )\n'
    '    if cot_col is None:\n'
    '        raise ValueError(f"Sonnet CSV missing CoT column. Cols: {list(sdf.columns)}")\n'
    '    sdf = sdf.dropna(subset=["prompt", "answer", cot_col]).reset_index(drop=True)\n'
    '    print(f"Sonnet traces: {len(sdf)} rows (CoT col={cot_col})")\n'
    '\n'
    '    PROMPT_SUFFIX = "\\nPlease put your final answer inside `\\\\boxed{}`. For example: `\\\\boxed{your answer}`"\n'
    '    SYSTEM_PROMPT = (\n'
    '        "You are an expert mathematics assistant. "\n'
    '        "Think carefully step by step inside <think>...</think> tags, "\n'
    '        "then give your final answer inside \\\\boxed{}."\n'
    '    )\n'
    '    records = []\n'
    '    for _, row in sdf.iterrows():\n'
    '        cot = str(row[cot_col]).strip()\n'
    '        if len(cot) < 20:\n'
    '            continue\n'
    '        ans = str(row["answer"]).strip()\n'
    '        if "<think>" not in cot:\n'
    '            cot = f"<think>\\n{cot}\\n</think>"\n'
    '        cot = re.sub(r"\\\\boxed\\{[^}]*\\}", "", cot).rstrip()\n'
    '        records.append({"messages": [\n'
    '            {"role": "system",    "content": SYSTEM_PROMPT},\n'
    '            {"role": "user",      "content": str(row["prompt"]) + PROMPT_SUFFIX},\n'
    '            {"role": "assistant", "content": f"{cot}\\n\\\\boxed{{{ans}}}"},\n'
    '        ]})\n'
    '    print(f"Sonnet SFT records: {len(records)}")\n'
    '    sonnet_ds = HFDataset.from_list(records)\n'
    '\n'
    '    def fmt(example):\n'
    '        msgs  = example["messages"]\n'
    '        convs = [msgs] if (msgs and isinstance(msgs[0], dict)) else msgs\n'
    '        out = []\n'
    '        for conv in convs:\n'
    '            try:\n'
    '                t = tokenizer.apply_chat_template(conv, tokenize=False,\n'
    '                                                  add_generation_prompt=False,\n'
    '                                                  enable_thinking=True)\n'
    '            except TypeError:\n'
    '                t = tokenizer.apply_chat_template(conv, tokenize=False,\n'
    '                                                  add_generation_prompt=False)\n'
    '            out.append(t)\n'
    '        return out\n'
    '\n'
    '    micro_sft_args = SFTConfig(\n'
    '        output_dir="/kaggle/working/sonnet_sft",\n'
    '        num_train_epochs=1,\n'
    '        per_device_train_batch_size=2,\n'
    '        gradient_accumulation_steps=4,\n'
    '        learning_rate=1e-5,\n'
    '        lr_scheduler_type="cosine",\n'
    '        warmup_ratio=0.05,\n'
    '        max_length=8192,\n'
    '        optim="paged_adamw_8bit",\n'
    '        weight_decay=0.01,\n'
    '        max_grad_norm=0.5,\n'
    '        logging_steps=10,\n'
    '        save_strategy="no",\n'
    '        bf16=True,\n'
    '        gradient_checkpointing=True,\n'
    '        gradient_checkpointing_kwargs={"use_reentrant": True},\n'
    '        remove_unused_columns=False,\n'
    '        report_to="none",\n'
    '        seed=42,\n'
    '    )\n'
    '    torch.cuda.empty_cache(); gc.collect()\n'
    '    print("Starting micro-SFT on Sonnet traces...")\n'
    '    t0 = time.time()\n'
    '    SFTTrainer(\n'
    '        model=model,\n'
    '        args=micro_sft_args,\n'
    '        train_dataset=sonnet_ds,\n'
    '        processing_class=tokenizer,\n'
    '        formatting_func=fmt,\n'
    '    ).train()\n'
    '    print(f"Micro-SFT done in {(time.time()-t0)/60:.1f} min.")\n'
    '    model.save_pretrained("/kaggle/working/sft_after_sonnet")\n'
    '    tokenizer.save_pretrained("/kaggle/working/sft_after_sonnet")\n'
    'else:\n'
    '    print("Skipping micro-SFT on Sonnet traces.")\n'
))

# -----------------------------------------------------------------------------
cells.append(md(
    "## Submission-Aligned Reward\n"
    "\n"
    "Backend metric: exact (or ±1e-2 numeric) match on the **contents of\n"
    "`\\boxed{}`**. Nothing else is scored.\n"
    "\n"
    "Therefore the reward here is correctness-dominated with **only hard guardrails**\n"
    "to prevent degeneracy that would zero out correctness anyway:\n"
    "\n"
    "| Component | Range | Purpose |\n"
    "|---|---|---|\n"
    "| `reward_correctness` | 0.0 or **+2.0** | Primary signal — mirrors backend |\n"
    "| `reward_extractable` | **-0.5** or 0.0 | Penalize when no `\\boxed{}` exists (un-scoreable) |\n"
    "| `reward_truncation` | **-0.3** or 0.0 | Penalize hitting `max_completion_length` w/o boxed |\n"
    "| `reward_repetition_guard` | **-0.3** or 0.0 | Penalize degenerate token loops |\n"
    "\n"
    "Format/length **bonuses** are intentionally removed — they invite reward\n"
    "hacking patterns that won't transfer to the held-out test set."
))

cells.append(code(
    'if TRAIN_ON_KAGGLE:\n'
    '    import re\n'
    '\n'
    '    def _extract_boxed(text):\n'
    '        pos = text.rfind(r"\\boxed{")\n'
    '        if pos == -1:\n'
    '            return None\n'
    '        start = pos + len(r"\\boxed{")\n'
    '        depth, i = 1, start\n'
    '        while i < len(text) and depth > 0:\n'
    '            if   text[i] == "{": depth += 1\n'
    '            elif text[i] == "}": depth -= 1\n'
    '            i += 1\n'
    '        return text[start:i-1].strip() if depth == 0 else None\n'
    '\n'
    '    def _normalize(ans):\n'
    '        if not ans: return ""\n'
    '        ans = ans.strip()\n'
    '        for tok in (r"\\\\,", r"\\,", r"\\\\!", r"\\!", r"\\\\ ", r"\\ ", r"\\text{", "}"):\n'
    '            ans = ans.replace(tok, "")\n'
    '        try:\n'
    '            val = float(ans.replace(",", ""))\n'
    '            if val == int(val) and abs(val) < 1e15:\n'
    '                return str(int(val))\n'
    '            return f"{val:.8g}"\n'
    '        except (ValueError, OverflowError):\n'
    '            return ans.lower().replace(" ", "")\n'
    '\n'
    '    def is_correct_boxed(pred_text, gold):\n'
    '        """Mirror of backend scoring: extract last \\\\boxed{} and compare\n'
    '        with string-equality OR numeric ±1e-2 tolerance."""\n'
    '        pred = _extract_boxed(pred_text)\n'
    '        if pred is None:\n'
    '            return False\n'
    '        p, g = _normalize(pred), _normalize(str(gold))\n'
    '        if p == g:\n'
    '            return True\n'
    '        try:\n'
    '            pv = float(p.replace(",", ""))\n'
    '            gv = float(g.replace(",", ""))\n'
    '            return abs(pv - gv) <= 1e-2\n'
    '        except (ValueError, OverflowError):\n'
    '            return False\n'
    '\n'
    '    def reward_correctness(completions, answer, **kwargs):\n'
    '        return [2.0 if is_correct_boxed(c, a) else 0.0\n'
    '                for c, a in zip(completions, answer)]\n'
    '\n'
    '    def reward_extractable(completions, **kwargs):\n'
    '        return [0.0 if _extract_boxed(c) is not None else -0.5 for c in completions]\n'
    '\n'
    '    # We do not know max_completion_length here, but a near-truncated\n'
    '    # completion without a boxed answer is almost certainly truncated.\n'
    '    # Use a length-only heuristic: very long output AND no boxed → penalty.\n'
    '    TRUNCATION_CHARS = 12000  # ~ 3 k tokens; tuned for max_completion_length=3072\n'
    '    def reward_truncation(completions, **kwargs):\n'
    '        out = []\n'
    '        for c in completions:\n'
    '            if len(c) >= TRUNCATION_CHARS and _extract_boxed(c) is None:\n'
    '                out.append(-0.3)\n'
    '            else:\n'
    '                out.append(0.0)\n'
    '        return out\n'
    '\n'
    '    def reward_repetition_guard(completions, **kwargs):\n'
    '        out = []\n'
    '        for c in completions:\n'
    '            w = c.split()\n'
    '            if len(w) < 40:\n'
    '                out.append(0.0); continue\n'
    '            tail = w[-min(len(w), 200):]\n'
    '            ngr = [tuple(tail[i:i+6]) for i in range(len(tail)-5)]\n'
    '            if not ngr:\n'
    '                out.append(0.0); continue\n'
    '            div = len(set(ngr)) / len(ngr)\n'
    '            out.append(-0.3 if div < 0.45 else 0.0)\n'
    '        return out\n'
    '\n'
    '    def reward_combined(completions, answer, **kwargs):\n'
    '        a = reward_correctness(completions, answer, **kwargs)\n'
    '        b = reward_extractable(completions, **kwargs)\n'
    '        c_ = reward_truncation(completions, **kwargs)\n'
    '        d = reward_repetition_guard(completions, **kwargs)\n'
    '        return [w + x + y + z for w, x, y, z in zip(a, b, c_, d)]\n'
    '\n'
    '    REWARD_FUNCS = [reward_combined]\n'
    '    print("Reward: correctness +2.0 / extractable -0.5 / truncation -0.3 / repetition -0.3")\n'
))

# -----------------------------------------------------------------------------
cells.append(md(
    "## Test-Set Probe Utilities\n"
    "\n"
    "Loads `test.csv` (prompts only — no answers). Defines `probe_test_set()` that\n"
    "samples completions and computes:\n"
    "\n"
    "- `% extractable`  — fraction with a parseable `\\boxed{}`\n"
    "- `mean_len_chars` — mean completion length\n"
    "- `self_consistency` — mean over prompts of *majority-vote share among k\n"
    "  sampled answers*. Higher = more confident policy.\n"
    "\n"
    "These are watch-only metrics. None of them backs into the reward."
))

cells.append(code(
    'if TRAIN_ON_KAGGLE and DO_TEST_PROBE:\n'
    '    import os, random, torch, gc, time\n'
    '    import pandas as pd\n'
    '    from collections import Counter\n'
    '\n'
    '    test_path = TEST_DATASET_PATH\n'
    '    if not os.path.exists(test_path):\n'
    '        print(f"WARNING: Test set not found at {test_path}. "\n'
    '              f"Test probe will be disabled.")\n'
    '        TEST_PROMPTS = []\n'
    '    else:\n'
    '        test_df = pd.read_csv(test_path)\n'
    '        assert "prompt" in test_df.columns, f"test.csv missing prompt column. cols={list(test_df.columns)}"\n'
    '        if "answer" in test_df.columns:\n'
    '            print("NOTE: test.csv has an answer column — using it ONLY for offline "\n'
    '                  "diagnostics, never for reward.")\n'
    '        TEST_PROMPTS = test_df["prompt"].astype(str).tolist()\n'
    '        print(f"Loaded {len(TEST_PROMPTS)} test prompts.")\n'
    '\n'
    '    PROMPT_SUFFIX = "\\nPlease put your final answer inside `\\\\boxed{}`. For example: `\\\\boxed{your answer}`"\n'
    '    SYSTEM_PROMPT = (\n'
    '        "You are an expert mathematics assistant. "\n'
    '        "Think carefully step by step inside <think>...</think> tags, "\n'
    '        "then give your final answer inside \\\\boxed{}."\n'
    '    )\n'
    '\n'
    '    def _render(prompt):\n'
    '        msgs = [\n'
    '            {"role": "system", "content": SYSTEM_PROMPT},\n'
    '            {"role": "user",   "content": prompt + PROMPT_SUFFIX},\n'
    '        ]\n'
    '        try:\n'
    '            return tokenizer.apply_chat_template(msgs, tokenize=False,\n'
    '                                                 add_generation_prompt=True,\n'
    '                                                 enable_thinking=True)\n'
    '        except TypeError:\n'
    '            return tokenizer.apply_chat_template(msgs, tokenize=False,\n'
    '                                                 add_generation_prompt=True)\n'
    '\n'
    '    @torch.no_grad()\n'
    '    def probe_test_set(prompts, k=2, n_prompts=32, max_new_tokens=1024, seed=0):\n'
    '        """Generate k samples for n_prompts random test prompts. Returns dict\n'
    '        of monitoring metrics. No reward, no labels — pure observation."""\n'
    '        if not prompts:\n'
    '            return {"extractable_rate": float("nan"),\n'
    '                    "mean_len_chars":   float("nan"),\n'
    '                    "self_consistency": float("nan"),\n'
    '                    "n_probed":         0}\n'
    '        rng = random.Random(seed)\n'
    '        sample = rng.sample(prompts, min(n_prompts, len(prompts)))\n'
    '        rendered = [_render(p) for p in sample]\n'
    '\n'
    '        was_training = model.training\n'
    '        model.eval()\n'
    '        extractable_hits = 0\n'
    '        lengths   = []\n'
    '        consist   = []\n'
    '        for text in rendered:\n'
    '            enc = tokenizer([text] * k, return_tensors="pt", padding=True,\n'
    '                            truncation=True, max_length=2048).to(model.device)\n'
    '            out = model.generate(\n'
    '                **enc, max_new_tokens=max_new_tokens, do_sample=True,\n'
    '                temperature=0.9, top_p=0.95,\n'
    '                pad_token_id=tokenizer.pad_token_id, use_cache=True,\n'
    '            )\n'
    '            gen = tokenizer.batch_decode(out[:, enc.input_ids.shape[1]:],\n'
    '                                         skip_special_tokens=True)\n'
    '            answers = []\n'
    '            for g in gen:\n'
    '                lengths.append(len(g))\n'
    '                box = _extract_boxed(g)\n'
    '                if box is not None:\n'
    '                    extractable_hits += 1\n'
    '                    answers.append(_normalize(box))\n'
    '            if answers:\n'
    '                cnt = Counter(answers).most_common(1)[0][1]\n'
    '                consist.append(cnt / k)\n'
    '            else:\n'
    '                consist.append(0.0)\n'
    '            del enc, out\n'
    '        if was_training:\n'
    '            model.train()\n'
    '        return {\n'
    '            "extractable_rate": extractable_hits / (len(sample) * k),\n'
    '            "mean_len_chars":   (sum(lengths) / len(lengths)) if lengths else 0.0,\n'
    '            "self_consistency": (sum(consist) / len(consist)) if consist else 0.0,\n'
    '            "n_probed":         len(sample),\n'
    '        }\n'
    '\n'
    '    # Baseline probe — record the pre-GRPO state of the model after Stage 2.\n'
    '    if TEST_PROMPTS:\n'
    '        torch.cuda.empty_cache(); gc.collect()\n'
    '        t0 = time.time()\n'
    '        baseline = probe_test_set(TEST_PROMPTS, k=2, n_prompts=32)\n'
    '        print(f"Pre-GRPO probe ({(time.time()-t0):.1f}s):", baseline)\n'
    '    else:\n'
    '        baseline = None\n'
    'else:\n'
    '    TEST_PROMPTS = []\n'
    '    baseline = None\n'
    '    def probe_test_set(*args, **kwargs):\n'
    '        return {"extractable_rate": float("nan"),\n'
    '                "mean_len_chars":   float("nan"),\n'
    '                "self_consistency": float("nan"),\n'
    '                "n_probed":         0}\n'
    '    print("Test probe disabled.")\n'
))

# -----------------------------------------------------------------------------
cells.append(md(
    "## Curriculum Filter on Labeled Training Data\n"
    "\n"
    "At 85 % accuracy ≥ 50 % of prompts give a correct rollout almost every time\n"
    "→ low gradient signal. We pre-score the current policy on a sample of\n"
    "**`train.csv` (labeled)** prompts and keep prompts whose pass@1 is in the\n"
    "informative band [0.25, 0.85].\n"
    "\n"
    "Test prompts are NOT used here — they have no labels."
))

cells.append(code(
    'if TRAIN_ON_KAGGLE and DO_CURRICULUM_FILTER:\n'
    '    import pandas as pd, torch, gc, time, random\n'
    '\n'
    '    df = pd.read_csv(TRAIN_DATASET_PATH).dropna(subset=["answer", "prompt"]).reset_index(drop=True)\n'
    '    print(f"Train (labeled) source: {len(df)} rows")\n'
    '\n'
    '    N_SCORE = min(1500, len(df))   # subsample for speed\n'
    '    K_SCORE = 3\n'
    '    rng     = random.Random(42)\n'
    '    idxs    = rng.sample(range(len(df)), N_SCORE)\n'
    '    sample  = df.iloc[idxs].reset_index(drop=True)\n'
    '\n'
    '    SYSTEM_PROMPT = (\n'
    '        "You are an expert mathematics assistant. "\n'
    '        "Think carefully step by step inside <think>...</think> tags, "\n'
    '        "then give your final answer inside \\\\boxed{}."\n'
    '    )\n'
    '    PROMPT_SUFFIX = "\\nPlease put your final answer inside `\\\\boxed{}`. For example: `\\\\boxed{your answer}`"\n'
    '\n'
    '    def _build(text):\n'
    '        msgs = [\n'
    '            {"role": "system", "content": SYSTEM_PROMPT},\n'
    '            {"role": "user",   "content": str(text) + PROMPT_SUFFIX},\n'
    '        ]\n'
    '        try:\n'
    '            return tokenizer.apply_chat_template(msgs, tokenize=False,\n'
    '                                                 add_generation_prompt=True,\n'
    '                                                 enable_thinking=True)\n'
    '        except TypeError:\n'
    '            return tokenizer.apply_chat_template(msgs, tokenize=False,\n'
    '                                                 add_generation_prompt=True)\n'
    '\n'
    '    @torch.no_grad()\n'
    '    def _pass_rate(prompt, gold, k):\n'
    '        t = _build(prompt)\n'
    '        enc = tokenizer([t] * k, return_tensors="pt", padding=True,\n'
    '                        truncation=True, max_length=2048).to(model.device)\n'
    '        out = model.generate(**enc, max_new_tokens=1024, do_sample=True,\n'
    '                             temperature=0.9, top_p=0.95,\n'
    '                             pad_token_id=tokenizer.pad_token_id, use_cache=True)\n'
    '        gen = tokenizer.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)\n'
    '        hits = sum(is_correct_boxed(g, gold) for g in gen)\n'
    '        return hits / k\n'
    '\n'
    '    pass_rates = []\n'
    '    model.eval(); torch.cuda.empty_cache(); gc.collect()\n'
    '    t0 = time.time()\n'
    '    for i, row in sample.iterrows():\n'
    '        try:\n'
    '            pr = _pass_rate(row["prompt"], row["answer"], K_SCORE)\n'
    '        except Exception:\n'
    '            pr = 0.5\n'
    '        pass_rates.append(pr)\n'
    '        if (i + 1) % 50 == 0:\n'
    '            print(f"  scored {i+1}/{N_SCORE}  ({(time.time()-t0)/60:.1f} min)")\n'
    '    sample["pass_rate"] = pass_rates\n'
    '    LO, HI = 0.25, 0.85\n'
    '    kept = sample[(sample["pass_rate"] >= LO) & (sample["pass_rate"] <= HI)].reset_index(drop=True)\n'
    '    print(f"Curriculum kept {len(kept)}/{N_SCORE} prompts in [{LO}, {HI}].")\n'
    '    grpo_df = kept[["prompt", "answer"]].copy()\n'
    '    grpo_df.to_csv("/kaggle/working/grpo_curriculum.csv", index=False)\n'
    'else:\n'
    '    import pandas as pd\n'
    '    df = pd.read_csv(TRAIN_DATASET_PATH).dropna(subset=["answer", "prompt"])\n'
    '    grpo_df = df[["prompt", "answer"]].sample(frac=1, random_state=42).reset_index(drop=True)\n'
    '    print(f"Skipping curriculum filter; using full labeled set: {len(grpo_df)} rows")\n'
))

# -----------------------------------------------------------------------------
cells.append(md(
    "## Stage 4 — GRPO Training\n"
    "\n"
    "Key knobs tuned for an 85 % starting point on 1× RTX 6000 Pro 96 GB:\n"
    "\n"
    "- **LR = 2e-6**, **β (KL) = 0.05** → cautious updates, strong anchor to SFT.\n"
    "- **`num_generations = 8`**, **`max_completion_length = 3072`** → low-variance,\n"
    "  full-CoT room.\n"
    "- **DAPO asymmetric clip** (`epsilon=0.20`, `epsilon_high=0.28`) — rewards\n"
    "  rare correct tokens without amplifying common-token noise.\n"
    "- **vLLM colocate** rollouts (3–5× faster than HF generate).\n"
    "- **Reference policy** via `model.disable_adapter()` — saves a 60 GB copy.\n"
    "- **`TestSetProbeCallback`** runs `probe_test_set()` every `PROBE_EVERY` steps."
))

cells.append(code(
    'if TRAIN_ON_KAGGLE and DO_GRPO:\n'
    '    import gc, time, torch\n'
    '    from datasets import Dataset as HFDataset\n'
    '    from trl import GRPOTrainer, GRPOConfig\n'
    '    from transformers import TrainerCallback\n'
    '\n'
    '    SEED = 42\n'
    '    PROBE_EVERY     = 25                 # steps between test probes\n'
    '    PROBE_N_PROMPTS = 32\n'
    '    PROBE_K_SAMPLES = 2\n'
    '    PROBE_MAX_NEW   = 1024\n'
    '    EARLY_STOP_DROP = 0.10               # stop if extractable_rate drops 10 pts\n'
    '\n'
    '    PROMPT_SUFFIX = "\\nPlease put your final answer inside `\\\\boxed{}`. For example: `\\\\boxed{your answer}`"\n'
    '    SYSTEM_PROMPT = (\n'
    '        "You are an expert mathematics assistant. "\n'
    '        "Think carefully step by step inside <think>...</think> tags, "\n'
    '        "then give your final answer inside \\\\boxed{}."\n'
    '    )\n'
    '\n'
    '    grpo_records = [\n'
    '        {\n'
    '            "prompt": [\n'
    '                {"role": "system", "content": SYSTEM_PROMPT},\n'
    '                {"role": "user",   "content": str(p) + PROMPT_SUFFIX},\n'
    '            ],\n'
    '            "answer": str(a),\n'
    '        }\n'
    '        for p, a in zip(grpo_df["prompt"], grpo_df["answer"])\n'
    '    ]\n'
    '    grpo_dataset = HFDataset.from_list(grpo_records)\n'
    '    print(f"GRPO dataset: {len(grpo_dataset)} prompts")\n'
    '\n'
    '    class GPUMetricsCallback(TrainerCallback):\n'
    '        def on_log(self, args, state, control, logs=None, **kwargs):\n'
    '            if logs is None or not torch.cuda.is_available():\n'
    '                return\n'
    '            logs["gpu/mem_alloc_gb"]    = torch.cuda.memory_allocated()     / 2**30\n'
    '            logs["gpu/mem_reserved_gb"] = torch.cuda.memory_reserved()      / 2**30\n'
    '            logs["gpu/mem_peak_gb"]     = torch.cuda.max_memory_allocated() / 2**30\n'
    '\n'
    '    class TestSetProbeCallback(TrainerCallback):\n'
    '        """Periodically probe the live policy on test prompts. Watch metrics\n'
    '        only; no reward feedback. Stops training if extractable_rate falls\n'
    '        by more than EARLY_STOP_DROP from the running maximum (reward hack\n'
    '        / format drift detection)."""\n'
    '        def __init__(self):\n'
    '            self.history = []\n'
    '            self.peak_extractable = 0.0\n'
    '            self.best_step = -1\n'
    '        def on_step_end(self, args, state, control, **kwargs):\n'
    '            if not DO_TEST_PROBE or not TEST_PROMPTS:\n'
    '                return\n'
    '            if state.global_step == 0 or state.global_step % PROBE_EVERY != 0:\n'
    '                return\n'
    '            torch.cuda.empty_cache(); gc.collect()\n'
    '            t0 = time.time()\n'
    '            m = probe_test_set(TEST_PROMPTS,\n'
    '                               k=PROBE_K_SAMPLES,\n'
    '                               n_prompts=PROBE_N_PROMPTS,\n'
    '                               max_new_tokens=PROBE_MAX_NEW,\n'
    '                               seed=state.global_step)\n'
    '            m["step"]    = state.global_step\n'
    '            m["probe_s"] = time.time() - t0\n'
    '            self.history.append(m)\n'
    '            print(f"[probe step={state.global_step}] "\n'
    '                  f"extractable={m[\'extractable_rate\']:.3f}  "\n'
    '                  f"self_consistency={m[\'self_consistency\']:.3f}  "\n'
    '                  f"mean_len={m[\'mean_len_chars\']:.0f}  "\n'
    '                  f"({m[\'probe_s\']:.0f}s)")\n'
    '            if m["extractable_rate"] > self.peak_extractable:\n'
    '                self.peak_extractable = m["extractable_rate"]\n'
    '                self.best_step        = state.global_step\n'
    '                model.save_pretrained("/kaggle/working/best_adapter")\n'
    '                print(f"  ↳ new peak extractable {self.peak_extractable:.3f} — adapter cached")\n'
    '            elif self.peak_extractable - m["extractable_rate"] > EARLY_STOP_DROP:\n'
    '                print(f"  ↳ extractable dropped {self.peak_extractable:.3f} → "\n'
    '                      f"{m[\'extractable_rate\']:.3f}; stopping early.")\n'
    '                control.should_training_stop = True\n'
    '\n'
    '    USE_VLLM = bool(globals().get("VLLM_OK", False))\n'
    '    grpo_kwargs = dict(\n'
    '        num_generations=8,\n'
    '        max_prompt_length=2048,\n'
    '        max_completion_length=3072,\n'
    '        temperature=0.9,\n'
    '        top_p=0.95,\n'
    '        beta=0.05,\n'
    '        epsilon=0.20,\n'
    '        epsilon_high=0.28,\n'
    '        learning_rate=2e-6,\n'
    '        lr_scheduler_type="cosine",\n'
    '        warmup_ratio=0.10,\n'
    '        num_train_epochs=1,\n'
    '        per_device_train_batch_size=2,\n'
    '        gradient_accumulation_steps=4,\n'
    '        optim="paged_adamw_8bit",\n'
    '        adam_beta1=0.9,\n'
    '        adam_beta2=0.999,\n'
    '        adam_epsilon=1e-8,\n'
    '        weight_decay=0.01,\n'
    '        max_grad_norm=0.3,\n'
    '        gradient_checkpointing=True,\n'
    '        bf16=True,\n'
    '        output_dir="/kaggle/working/grpo_v12_output",\n'
    '        logging_steps=1,\n'
    '        logging_dir="/kaggle/working/tb_logs_v12",\n'
    '        report_to="tensorboard",\n'
    '        save_strategy="no",\n'
    '        seed=SEED,\n'
    '        remove_unused_columns=False,\n'
    '        dataloader_num_workers=2,\n'
    '    )\n'
    '    if USE_VLLM:\n'
    '        grpo_kwargs.update(\n'
    '            use_vllm=True,\n'
    '            vllm_mode="colocate",\n'
    '            vllm_gpu_memory_utilization=0.35,\n'
    '        )\n'
    '    grpo_config = GRPOConfig(**grpo_kwargs)\n'
    '\n'
    '    eff = (grpo_config.per_device_train_batch_size\n'
    '           * grpo_config.gradient_accumulation_steps\n'
    '           * grpo_config.num_generations)\n'
    '    print("=" * 60)\n'
    '    print(f"  GRPO v12 — LR={grpo_config.learning_rate}, beta={grpo_config.beta},"\n'
    '          f" eps_low/high={grpo_config.epsilon}/{grpo_config.epsilon_high}")\n'
    '    print(f"  gens={grpo_config.num_generations}, max_completion={grpo_config.max_completion_length},"\n'
    '          f" vLLM={USE_VLLM}, effective batch={eff} completions/update")\n'
    '    print("=" * 60)\n'
    '\n'
    '    torch.cuda.empty_cache(); gc.collect()\n'
    '    probe_cb = TestSetProbeCallback()\n'
    '    trainer = GRPOTrainer(\n'
    '        model=model,\n'
    '        processing_class=tokenizer,\n'
    '        reward_funcs=REWARD_FUNCS,\n'
    '        args=grpo_config,\n'
    '        train_dataset=grpo_dataset,\n'
    '        callbacks=[GPUMetricsCallback(), probe_cb],\n'
    '    )\n'
    '    print("Starting GRPO v12 training...")\n'
    '    t0 = time.time()\n'
    '    trainer.train()\n'
    '    print(f"GRPO done in {(time.time()-t0)/60:.1f} min")\n'
    '\n'
    '    # If a best-extractable checkpoint was saved during probing, prefer it.\n'
    '    BEST_DIR    = "/kaggle/working/best_adapter"\n'
    '    ADAPTER_DIR = "/kaggle/working/sft_adapter"\n'
    '    import os, shutil\n'
    '    if os.path.isdir(BEST_DIR) and os.path.exists(os.path.join(BEST_DIR, "adapter_model.safetensors")):\n'
    '        print(f"Using best-step adapter (peak extractable @ step {probe_cb.best_step}, "\n'
    '              f"value={probe_cb.peak_extractable:.3f})")\n'
    '        if os.path.isdir(ADAPTER_DIR):\n'
    '            shutil.rmtree(ADAPTER_DIR)\n'
    '        shutil.copytree(BEST_DIR, ADAPTER_DIR)\n'
    '    else:\n'
    '        model.save_pretrained(ADAPTER_DIR)\n'
    '    tokenizer.save_pretrained(ADAPTER_DIR)\n'
    '    print(f"Adapter saved to {ADAPTER_DIR}")\n'
    '\n'
    '    # Persist probe history for offline analysis.\n'
    '    import json as _json\n'
    '    with open("/kaggle/working/probe_history.json", "w") as f:\n'
    '        _json.dump(probe_cb.history, f, indent=2)\n'
    'else:\n'
    '    print("Skipping GRPO stage.")\n'
))

# -----------------------------------------------------------------------------
cells.append(md(
    "## Final Test-Set Probe Report\n"
    "\n"
    "Runs `probe_test_set()` one more time on the **final** adapter (best-cached\n"
    "or end-of-training) so you have a final-state snapshot to compare against\n"
    "`baseline` (pre-GRPO) before zipping."
))

cells.append(code(
    'if TRAIN_ON_KAGGLE and DO_TEST_PROBE and TEST_PROMPTS:\n'
    '    import torch, gc, time, json\n'
    '    torch.cuda.empty_cache(); gc.collect()\n'
    '    t0 = time.time()\n'
    '    final = probe_test_set(TEST_PROMPTS, k=4, n_prompts=64,\n'
    '                           max_new_tokens=1024, seed=999)\n'
    '    print(f"Post-GRPO probe ({(time.time()-t0):.1f}s):", final)\n'
    '    if baseline is not None:\n'
    '        delta = {\n'
    '            "extractable_rate": final["extractable_rate"] - baseline["extractable_rate"],\n'
    '            "self_consistency": final["self_consistency"] - baseline["self_consistency"],\n'
    '            "mean_len_chars":   final["mean_len_chars"]   - baseline["mean_len_chars"],\n'
    '        }\n'
    '        print("Δ vs pre-GRPO baseline:", delta)\n'
    '    with open("/kaggle/working/final_probe.json", "w") as f:\n'
    '        json.dump({"baseline": baseline, "final": final}, f, indent=2)\n'
    'else:\n'
    '    print("Test probe disabled or no prompts loaded.")\n'
))

# -----------------------------------------------------------------------------
cells.append(md("## Mode B: Load Pre-trained LoRA"))

cells.append(code(
    'if USE_PRETRAINED:\n'
    '    import os\n'
    '    SRC_ADAPTER_DIR = PRETRAINED_ADAPTER_DATASET_PATH\n'
    '    required_files = ["adapter_config.json", "adapter_model.safetensors"]\n'
    '    print("Using pre-trained adapter from:", SRC_ADAPTER_DIR)\n'
    '    for fname in required_files:\n'
    '        fpath = os.path.join(SRC_ADAPTER_DIR, fname)\n'
    '        if not os.path.exists(fpath):\n'
    '            raise FileNotFoundError(f"Missing: {fpath}")\n'
    '        print(f"  {fname}: {os.path.getsize(fpath)/1024/1024:.1f} MB")\n'
))

# -----------------------------------------------------------------------------
cells.append(md("## Create submission.zip"))

cells.append(code(
    'import json, os, shutil, zipfile\n'
    '\n'
    'OUTPUT_DIR = "/kaggle/working"\n'
    'SUBMISSION_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "submission_adapter")\n'
    'os.makedirs(SUBMISSION_ADAPTER_DIR, exist_ok=True)\n'
    'required_files = ["adapter_config.json", "adapter_model.safetensors"]\n'
    '\n'
    'if TRAIN_ON_KAGGLE:\n'
    '    src_adapter_dir = "/kaggle/working/sft_adapter"\n'
    '    print("Packaging GRPO-trained adapter from:", src_adapter_dir)\n'
    'else:\n'
    '    src_adapter_dir = PRETRAINED_ADAPTER_DATASET_PATH\n'
    '    print("Packaging pre-trained adapter directly from:", src_adapter_dir)\n'
    '\n'
    'for fname in required_files:\n'
    '    src = os.path.join(src_adapter_dir, fname)\n'
    '    dst = os.path.join(SUBMISSION_ADAPTER_DIR, fname)\n'
    '    if not os.path.exists(src):\n'
    '        raise FileNotFoundError(f"Missing: {src}")\n'
    '    shutil.copy2(src, dst)\n'
    '    print(f"Copied {fname} ({os.path.getsize(dst)/1024/1024:.1f} MB)")\n'
    '\n'
    'config_path = os.path.join(SUBMISSION_ADAPTER_DIR, "adapter_config.json")\n'
    'with open(config_path) as f:\n'
    '    cfg = json.load(f)\n'
    'cfg["base_model_name_or_path"] = BASE_MODEL_NAME\n'
    'cfg["inference_mode"] = True\n'
    'cfg["lora_dropout"] = 0.0\n'
    'with open(config_path, "w") as f:\n'
    '    json.dump(cfg, f, indent=2)\n'
    '\n'
    'zip_path = os.path.join(OUTPUT_DIR, "submission.zip")\n'
    'with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:\n'
    '    for fname in required_files:\n'
    '        zf.write(os.path.join(SUBMISSION_ADAPTER_DIR, fname), fname)\n'
    '        print(f"  Added {fname}")\n'
    'print(f"\\nsubmission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")\n'
    'print("Done.")\n'
))

# -----------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NB_PATH.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {NB_PATH}")
