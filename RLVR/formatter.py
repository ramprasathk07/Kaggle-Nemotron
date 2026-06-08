"""
Formatter for SFT training examples.

Converts (prompt, reasoning_trace, answer) tuples into the
ChatML / conversation format expected by the Nemotron tokenizer.

The training format is:
  [System prompt]
  [User: problem prompt]
  [Assistant: <think>\n...reasoning...\n</think>\n\\boxed{answer}]

The evaluation format used by Kaggle:
  [User: problem prompt]
  [Assistant: ... \\boxed{answer}]

Both formats produce answers the metric can extract.
"""

from .system_prompts import get_system_prompt, SYSTEM_PROMPT_GENERIC


def format_sft_example(
    prompt: str,
    answer: str,
    reasoning: str | None = None,
    category: str = "unknown",
    use_category_system_prompt: bool = True,
    use_thinking_tags: bool = True,
) -> dict:
    """
    Build a single SFT training example in chat-message format.

    Returns a dict with:
      - 'messages': list of {"role": ..., "content": ...} dicts
      - 'category': str
      - 'answer': str (the ground-truth answer)
    """
    system_content = (
        get_system_prompt(category) if use_category_system_prompt
        else SYSTEM_PROMPT_GENERIC
    )

    # Build assistant response
    if reasoning and use_thinking_tags:
        assistant_content = (
            f"<think>\n{reasoning.strip()}\n</think>\n\\boxed{{{answer}}}"
        )
    elif reasoning:
        assistant_content = f"{reasoning.strip()}\n\nTherefore, the answer is \\boxed{{{answer}}}"
    else:
        assistant_content = f"\\boxed{{{answer}}}"

    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        "category": category,
        "answer": answer,
    }


def format_sft_dataset(
    examples: list[dict],
    use_category_system_prompt: bool = True,
    use_thinking_tags: bool = True,
) -> list[dict]:
    """
    Format a list of example dicts into SFT training examples.

    Each input dict should have:
      - 'prompt': str
      - 'answer': str (verified correct answer)
      - 'reasoning': str | None (CoT trace)
      - 'category': str

    Returns list of formatted SFT dicts.
    """
    formatted = []
    for ex in examples:
        formatted.append(
            format_sft_example(
                prompt=ex["prompt"],
                answer=ex["answer"],
                reasoning=ex.get("reasoning"),
                category=ex.get("category", "unknown"),
                use_category_system_prompt=use_category_system_prompt,
                use_thinking_tags=use_thinking_tags,
            )
        )
    return formatted


def apply_chat_template(messages: list[dict], tokenizer) -> str:
    """
    Apply the tokenizer's chat template to a list of messages.
    Falls back to manual formatting if the tokenizer doesn't have a template.
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Manual fallback (Nemotron/ChatML format)
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        return "".join(parts)
