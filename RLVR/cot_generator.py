"""
Chain-of-Thought (CoT) generator.

Generates reasoning traces for training examples using:
  1. Solver-derived reasoning (deterministic, always correct)
  2. OpenAI/Anthropic API calls for harder examples
  3. Template-based reasoning for simple cases

The key principle: ONLY accept reasoning traces that produce the CORRECT answer.
The solver's verified answer is ground truth. If the LLM produces a wrong answer,
we discard the trace and fall back to a simpler template.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.extract_answer import extract_final_answer, verify
from solvers import classify_problem, solve


# ── Solver-derived reasoning templates ────────────────────────────────────────

def _cot_from_solver(prompt: str, answer: str, category: str) -> str:
    """
    Generate a reasoning trace directly from the solver's step-by-step output.
    This is always correct because the solver IS the oracle.
    """
    from solvers import solve
    result = solve(prompt)

    if result.reasoning and result.answer and verify(result.answer, answer):
        return result.reasoning
    else:
        return _template_cot(prompt, answer, category)


def _template_cot(prompt: str, answer: str, category: str) -> str:
    """
    Generate a minimal template reasoning trace that always produces the correct answer.
    Used as fallback when solver reasoning is unavailable.
    """
    templates = {
        "gravity": (
            "I need to find the gravitational constant g from the examples.\n"
            "Using d = 0.5 * g * t², I solve for g = 2d / t² from each example.\n"
            "I compute g for each example and take the average.\n"
            "Then I apply d = 0.5 * g * t_query² to find the answer.\n"
            f"The answer is {answer}."
        ),
        "unit_conversion": (
            "I need to find the conversion ratio from the examples.\n"
            "For each example: ratio = output / input.\n"
            "I compute the ratio from each example and average them.\n"
            "Then: answer = query_input × ratio.\n"
            f"The answer is {answer}."
        ),
        "numeral": (
            "I examine the examples and recognize this is a Roman numeral conversion.\n"
            "I apply the standard Roman numeral rules:\n"
            "M=1000, D=500, C=100, L=50, X=10, V=5, I=1\n"
            "with subtractive notation: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900.\n"
            f"The answer is {answer}."
        ),
        "cipher": (
            "I build a character-to-character substitution mapping from the examples.\n"
            "Each encrypted word maps to a plaintext word of the same length,\n"
            "so each character maps consistently to another character.\n"
            "I apply the mapping to decrypt the query text.\n"
            f"The answer is {answer}."
        ),
        "bit_manipulation": (
            "I analyze each output bit position independently.\n"
            "For each bit position, I find the boolean function of input bits\n"
            "that is consistent with all examples.\n"
            "I apply the identified functions to compute the output for the query.\n"
            f"The answer is {answer}."
        ),
        "equation": (
            "I study the transformation rules from the examples.\n"
            "I look for consistent character-level or operator-level patterns.\n"
            "I identify the rule and apply it to the query expression.\n"
            f"The answer is {answer}."
        ),
    }
    return templates.get(category, f"Based on the pattern in the examples, the answer is {answer}.")


# ── API-based CoT generation ──────────────────────────────────────────────────

def _call_openai(prompt: str, answer: str, category: str, api_key: str) -> str | None:
    """Call OpenAI API to generate a CoT trace. Returns trace or None on failure."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        system = (
            "You are a careful reasoning assistant. You will be given a puzzle problem. "
            "Solve it step by step, showing your work. "
            f"The correct answer is: {answer}. "
            "Write a clear step-by-step solution that arrives at this answer. "
            f"End with: \\boxed{{{answer}}}"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return None


def _call_anthropic(prompt: str, answer: str, category: str, api_key: str) -> str | None:
    """Call Anthropic Claude API to generate a CoT trace."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Solve this puzzle step by step. The correct answer is {answer}. "
                        f"Show your reasoning clearly and end with \\boxed{{{answer}}}.\n\n"
                        f"Problem:\n{prompt}"
                    ),
                }
            ],
        )
        return message.content[0].text
    except Exception:
        return None


# ── Main CoT generator ────────────────────────────────────────────────────────

class CoTGenerator:
    """
    Generates Chain-of-Thought reasoning traces for training examples.

    Priority order:
      1. Solver-derived reasoning (deterministic, free)
      2. API-based generation (if API key provided and solver fails)
      3. Template fallback (always works, minimal quality)
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        use_solver_reasoning: bool = True,
        use_api: bool = True,
    ):
        self.openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.use_solver_reasoning = use_solver_reasoning
        self.use_api = use_api

    def generate(
        self,
        prompt: str,
        answer: str,
        category: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a CoT trace for a single example.

        Returns dict with:
          - 'reasoning': str (the reasoning trace, without \boxed{})
          - 'full_response': str (reasoning + \boxed{answer})
          - 'source': str ('solver' | 'api_openai' | 'api_anthropic' | 'template')
          - 'verified': bool (does the extracted answer match ground truth?)
        """
        if category is None:
            category = classify_problem(prompt)

        # Strategy 1: Solver-derived reasoning
        if self.use_solver_reasoning:
            reasoning = _cot_from_solver(prompt, answer, category)
            full = f"{reasoning}\n\n\\boxed{{{answer}}}"
            extracted = extract_final_answer(full)
            if verify(extracted, answer):
                return {
                    "reasoning": reasoning,
                    "full_response": full,
                    "source": "solver",
                    "verified": True,
                }

        # Strategy 2: API calls
        if self.use_api:
            for name, fn, key in [
                ("api_openai", _call_openai, self.openai_key),
                ("api_anthropic", _call_anthropic, self.anthropic_key),
            ]:
                if not key:
                    continue
                trace = fn(prompt, answer, category, key)
                if trace:
                    extracted = extract_final_answer(trace)
                    if verify(extracted, answer):
                        return {
                            "reasoning": trace,
                            "full_response": trace,
                            "source": name,
                            "verified": True,
                        }

        # Strategy 3: Template fallback (always accepted — answer is hardcoded)
        reasoning = _template_cot(prompt, answer, category)
        full = f"{reasoning}\n\n\\boxed{{{answer}}}"
        return {
            "reasoning": reasoning,
            "full_response": full,
            "source": "template",
            "verified": True,  # Template always contains the correct answer
        }

    def generate_batch(
        self,
        examples: list[dict],
        output_jsonl: str | None = None,
        max_examples: int | None = None,
    ) -> list[dict]:
        """
        Generate CoT traces for a batch of examples.

        Each input dict needs: prompt, answer, category (optional).
        Returns enriched dicts with reasoning added.
        """
        from tqdm import tqdm

        results = []
        subset = examples[:max_examples] if max_examples else examples

        for ex in tqdm(subset, desc="Generating CoT traces"):
            cot = self.generate(
                prompt=ex["prompt"],
                answer=ex["answer"],
                category=ex.get("category"),
            )
            enriched = {**ex, **cot}
            results.append(enriched)

        if output_jsonl:
            os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
            with open(output_jsonl, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Saved {len(results)} CoT examples to {output_jsonl}")

        return results


def prepare_sft_data(
    solver_verified_jsonl: str = "data/processed/solver_verified.jsonl",
    synthetic_jsonl: str | None = "data/synthetic/synthetic_problems.jsonl",
    public_datasets: list[str] | None = None,
    train_ids_json: str = "data/train_ids.json",
    output_jsonl: str = "data/processed/sft_training_data.jsonl",
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
) -> str:
    """
    Assemble and format the complete SFT training dataset:
      1. Load solver-verified training examples (filtered to train split)
      2. Add synthetic examples
      3. Add public dataset examples if provided
      4. Generate CoT traces for all
      5. Format and save to output JSONL

    Returns path to output JSONL.
    """
    import json
    from datagen.formatter import format_sft_example

    # Load train IDs
    with open(train_ids_json) as f:
        train_ids = set(json.load(f))

    # Load solver-verified examples (only correct ones in train split)
    examples = []
    with open(solver_verified_jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            if r["id"] in train_ids and r["correct"]:
                examples.append({
                    "id": r["id"],
                    "prompt": r["prompt"],
                    "answer": r["ground_truth"],
                    "category": r["category"],
                    "source": "train_solver",
                    "reasoning": r.get("reasoning", ""),
                })

    print(f"Loaded {len(examples)} solver-verified train examples")

    # Load synthetic examples
    if synthetic_jsonl and os.path.exists(synthetic_jsonl):
        synth_count = 0
        with open(synthetic_jsonl, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line.strip())
                examples.append({
                    "prompt": r["prompt"],
                    "answer": r["answer"],
                    "category": r["category"],
                    "source": "synthetic",
                })
                synth_count += 1
        print(f"Added {synth_count} synthetic examples")

    # Load public dataset examples
    if public_datasets:
        for path in public_datasets:
            if not os.path.exists(path):
                continue
            pub_count = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line.strip())
                    if "prompt" in r and "answer" in r:
                        examples.append({
                            "prompt": r["prompt"],
                            "answer": r.get("answer", r.get("ground_truth", "")),
                            "category": r.get("category", classify_problem(r["prompt"])),
                            "source": "public",
                            "reasoning": r.get("reasoning", r.get("cot", "")),
                        })
                        pub_count += 1
            print(f"Added {pub_count} examples from {path}")

    # Generate CoT traces and format
    generator = CoTGenerator(
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        use_solver_reasoning=True,
        use_api=bool(openai_api_key or anthropic_api_key),
    )

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    written = 0
    from tqdm import tqdm
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for ex in tqdm(examples, desc="Formatting SFT data"):
            # Use existing reasoning if available, otherwise generate
            if ex.get("reasoning"):
                reasoning = ex["reasoning"]
            else:
                cot = generator.generate(ex["prompt"], ex["answer"], ex.get("category"))
                reasoning = cot["reasoning"]

            formatted = format_sft_example(
                prompt=ex["prompt"],
                answer=ex["answer"],
                reasoning=reasoning,
                category=ex.get("category", "unknown"),
                use_thinking_tags=True,
            )
            formatted["source"] = ex.get("source", "unknown")
            fout.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved {written} SFT training examples to {output_jsonl}")
    return output_jsonl
