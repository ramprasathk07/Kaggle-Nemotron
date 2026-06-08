"""
Model loading utilities for Nemotron-3-Nano-30B.
Handles the custom Mamba architecture and LoRA setup.
"""

import os
import sys
import yaml
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


# ── Model loading ─────────────────────────────────────────────────────────────

def _get_model_path(config_path: str | None = None) -> str:
    """Resolve the model path from config or environment."""
    if config_path and os.path.exists(config_path):
        return config_path

    # Try environment variable
    env_path = os.environ.get("NEMOTRON_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Try Kaggle environment
    try:
        import kagglehub
        path = kagglehub.model_download(
            "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
        )
        return path
    except Exception:
        pass

    # Fallback to HuggingFace ID
    return "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def _setup_mamba_site_packages():
    """Add Kaggle's pre-installed Mamba/CUTLASS packages to the path."""
    cutlass_path = (
        "/kaggle/usr/lib/notebooks/ryanholbrook/"
        "nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"
    )
    if os.path.isdir(cutlass_path):
        import site
        site.addsitedir(cutlass_path)


def load_model_and_tokenizer(
    model_path: str | None = None,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    load_in_8bit: bool = False,
) -> tuple:
    """
    Load the Nemotron-3-Nano-30B model and tokenizer.
    Returns (model, tokenizer).
    """
    _setup_mamba_site_packages()

    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        print("Warning: mamba_ssm not found. Install it for full Nemotron support.")

    resolved_path = _get_model_path(model_path)
    print(f"Loading model from: {resolved_path}")

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="eager",  # Required for Nemotron
        load_in_8bit=load_in_8bit,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
    return model, tokenizer


def load_lora_config(config_yaml: str = "configs/lora_config.yaml") -> LoraConfig:
    """Load LoRA configuration from YAML file."""
    with open(config_yaml) as f:
        cfg = yaml.safe_load(f)

    lora_cfg = cfg["lora"]
    return LoraConfig(
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules"),
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )


def apply_lora(model, lora_config: LoraConfig):
    """Apply LoRA to the model and print parameter stats."""
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied: {trainable/1e6:.1f}M trainable / {total/1e9:.2f}B total ({100*trainable/total:.3f}%)")
    return model


def save_adapter(model, output_dir: str, tokenizer=None) -> str:
    """Save LoRA adapter weights to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")
    return output_dir


def load_adapter(base_model, adapter_path: str):
    """Load a saved LoRA adapter onto a base model."""
    return PeftModel.from_pretrained(base_model, adapter_path)
