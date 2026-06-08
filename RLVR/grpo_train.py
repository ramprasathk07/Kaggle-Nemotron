"""
GRPO (Group Relative Policy Optimization) training script.

Runs AFTER SFT warmup to reinforce correct answers via RL.

Usage:
  python training/grpo_train.py [--config configs/grpo_config.yaml]
                                [--sft-adapter adapters/sft_v1]
                                [--output-dir adapters/grpo_v1]

The GRPO reward is:
  +2.5 for correct answer with \\boxed{}
  +2.0 for correct answer without \\boxed{}
  -0.5 for wrong answer with \\boxed{}
  -1.0 for wrong answer without \\boxed{}
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.utils import (
    load_model_and_tokenizer,
    load_lora_config,
    apply_lora,
    save_adapter,
    load_adapter,
)
from training.reward_functions import make_grpo_reward_fn
from datagen.system_prompts import get_system_prompt
from solvers import classify_problem


def load_grpo_dataset(
    train_jsonl: str,
    tokenizer,
    prioritize_categories: list[str] | None = None,
    max_examples: int | None = None,
) -> Dataset:
    """
    Load training data for GRPO.
    Format: each example has 'prompt_text' (for generation) and 'ground_truth' (for reward).
    """
    examples = []
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            cat = ex.get("category", "unknown")
            messages = ex.get("messages", [])

            # Extract the problem prompt (system + user messages only, no assistant)
            user_messages = [m for m in messages if m["role"] in ("system", "user")]
            prompt_text = tokenizer.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            examples.append({
                "prompt": prompt_text,
                "ground_truth": ex.get("answer", ""),
                "category": cat,
            })

    # Prioritize hard categories by oversampling
    if prioritize_categories:
        hard = [e for e in examples if e["category"] in prioritize_categories]
        easy = [e for e in examples if e["category"] not in prioritize_categories]
        # 2x weight on hard categories
        examples = hard * 2 + easy
        import random
        random.shuffle(examples)

    if max_examples:
        examples = examples[:max_examples]

    print(f"GRPO dataset: {len(examples)} examples")
    return Dataset.from_list(examples)


def train_grpo(
    config_path: str = "configs/grpo_config.yaml",
    sft_adapter_path: str | None = None,
    output_dir: str | None = None,
    model_path: str | None = None,
):
    """Main GRPO training function."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    grpo_cfg = config["grpo"]
    data_cfg = config["data"]

    if output_dir:
        train_cfg["output_dir"] = output_dir
    if sft_adapter_path:
        train_cfg["sft_adapter_path"] = sft_adapter_path

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path=model_path)

    # Apply LoRA (fresh or from SFT checkpoint)
    sft_path = train_cfg.get("sft_adapter_path")
    if sft_path and os.path.exists(sft_path):
        print(f"Loading SFT adapter from {sft_path}")
        model = load_adapter(model, sft_path)
    else:
        print("No SFT adapter found, initializing fresh LoRA")
        lora_config = load_lora_config("configs/lora_config.yaml")
        model = apply_lora(model, lora_config)

    # Load dataset
    dataset = load_grpo_dataset(
        train_jsonl=data_cfg["train_jsonl"],
        tokenizer=tokenizer,
        prioritize_categories=data_cfg.get("prioritize_categories"),
    )

    # Reward function
    reward_fn = make_grpo_reward_fn(
        w_correct=config["rewards"].get("correct_answer", 2.0),
        w_format=config["rewards"].get("boxed_format", 0.5),
        w_wrong=config["rewards"].get("wrong_answer", -1.0),
    )

    # GRPO training config
    grpo_training_args = GRPOConfig(
        output_dir=train_cfg["output_dir"],
        max_steps=train_cfg.get("max_steps", 1000),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.1),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.1),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg.get("logging_steps", 1),
        save_steps=train_cfg.get("save_steps", 100),
        report_to=train_cfg.get("report_to", "none"),
        num_generations=grpo_cfg.get("num_generations", 8),
        max_prompt_length=grpo_cfg.get("max_prompt_length", 4096),
        max_completion_length=grpo_cfg.get("max_completion_length", 3584),
        temperature=grpo_cfg.get("temperature", 0.9),
        beta=grpo_cfg.get("beta", 0.01),
        epsilon=grpo_cfg.get("epsilon", 0.2),
        remove_unused_columns=False,
    )

    print(f"\nStarting GRPO training...")
    print(f"  Output dir: {train_cfg['output_dir']}")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Max steps: {train_cfg.get('max_steps', 1000)}")
    print(f"  Num generations: {grpo_cfg.get('num_generations', 8)}")

    trainer = GRPOTrainer(
        model=model,
        args=grpo_training_args,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        tokenizer=tokenizer,
    )

    trainer.train()

    adapter_path = save_adapter(model, train_cfg["output_dir"], tokenizer)
    print(f"GRPO training complete. Adapter saved to {adapter_path}")
    return adapter_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO training for Nemotron")
    parser.add_argument("--config", default="configs/grpo_config.yaml")
    parser.add_argument("--sft-adapter", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    train_grpo(
        config_path=args.config,
        sft_adapter_path=args.sft_adapter,
        output_dir=args.output_dir,
        model_path=args.model_path,
    )
