"""
Reward functions for GRPO training.

The reward signal teaches the model:
1. Correct answer → high reward
2. \\boxed{} format compliance → partial reward
3. Wrong answer → penalty

The reward is designed so that:
  - Getting the right answer in \\boxed{} → +1.5 (max)
  - Getting the right answer without \\boxed{} → +0.5 (partial)
  - \\boxed{} with wrong answer → 0.0
  - No \\boxed{}, wrong answer → -1.0

This incentivizes BOTH correctness AND format compliance.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.extract_answer import extract_final_answer, verify


_BOXED_RE = re.compile(r"\\boxed\{")


def reward_correct_answer(
    completions: list[str],
    ground_truths: list[str],
) -> list[float]:
    """
    Per-completion reward: 1.0 if correct, 0.0 otherwise.
    (Used as one component of the combined reward)
    """
    rewards = []
    for completion, truth in zip(completions, ground_truths):
        predicted = extract_final_answer(completion)
        rewards.append(1.0 if verify(predicted, truth) else 0.0)
    return rewards


def reward_boxed_format(
    completions: list[str],
    ground_truths: list[str],  # unused but kept for uniform interface
) -> list[float]:
    """
    Per-completion reward: 0.5 if \\boxed{} present, 0.0 otherwise.
    """
    return [
        0.5 if _BOXED_RE.search(completion) else 0.0
        for completion in completions
    ]


def reward_combined(
    completions: list[str],
    ground_truths: list[str],
    w_correct: float = 2.0,
    w_format: float = 0.5,
    w_wrong: float = -1.0,
) -> list[float]:
    """
    Combined reward:
      - Correct in \\boxed{}: w_correct + w_format
      - Correct without \\boxed{}: w_correct
      - \\boxed{} but wrong: w_format + w_wrong (slightly penalized)
      - Wrong, no \\boxed{}: w_wrong
    """
    rewards = []
    for completion, truth in zip(completions, ground_truths):
        predicted = extract_final_answer(completion)
        is_correct = verify(predicted, truth)
        has_boxed = bool(_BOXED_RE.search(completion))

        if is_correct and has_boxed:
            r = w_correct + w_format
        elif is_correct:
            r = w_correct
        elif has_boxed:
            r = w_format + w_wrong
        else:
            r = w_wrong

        rewards.append(r)
    return rewards


# ── TRL-compatible reward function wrappers ────────────────────────────────────
# TRL's GRPOTrainer expects reward functions with signature:
#   fn(completions: list[str], **kwargs) -> list[float]
# where kwargs includes any fields from the dataset (e.g. 'answer', 'ground_truth')

def make_grpo_reward_fn(
    w_correct: float = 2.0,
    w_format: float = 0.5,
    w_wrong: float = -1.0,
):
    """
    Factory for a TRL-compatible GRPO reward function.

    The returned function expects the dataset to have a 'ground_truth' column.
    """
    def _reward_fn(completions: list[str], ground_truth: list[str] = None, **kwargs):
        if ground_truth is None:
            # Try 'answer' column as fallback
            ground_truth = kwargs.get("answer", [""] * len(completions))
        return reward_combined(
            completions, ground_truth,
            w_correct=w_correct,
            w_format=w_format,
            w_wrong=w_wrong,
        )
    return _reward_fn
