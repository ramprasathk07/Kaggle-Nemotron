"""
Dataset Augmentation Pipeline for NVIDIA Nemotron Reasoning Challenge
======================================================================
Downloads standard math/reasoning datasets from HuggingFace and augments
them with Wonderland-style prompts + <think>...</think>\\boxed{} CoT format.

Supported datasets:
  - NuminaMath-CoT   (AI-MO/NuminaMath-CoT)
  - Nemotron-Math    (nvidia/OpenMathReasoning)
  - OlympiadBench    (AI-MO/OlympiadBench)
  - OlymMATH         (RUC-AIBOX/OlymMATH)

Output: JSONL with {"messages": [{"role":"user","content":...},{"role":"assistant","content":...}]}
"""

import json
import random
import re
import argparse
import sys
from pathlib import Path
from typing import Optional

# ── optional heavy deps (installed at runtime) ──────────────────────────────
try:
    from datasets import load_dataset, DownloadConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# WONDERLAND PROMPT TEMPLATES
# IMPORTANT FIX: Added boxed instruction to every prompt template
WONDERLAND_CATEGORY_PROMPTS = {
    "algebra": (
        "In Alice's Wonderland, mathematical laws work slightly differently. "
        "The equations below follow the Wonderland rules of algebra, where symbols "
        "and operations carry their usual meaning but the universe demands precise reasoning.\n\n"
        "Solve the following problem step-by-step, showing all your work:\n\n"
        "{problem}\n\n"
        "Please put your final answer inside \\boxed{}."
    ),
    "number_theory": (
        "In Alice's Wonderland, the Red Queen demands rigorous proofs about numbers. "
        "The Cheshire Cat has posed the following number theory problem. "
        "Reason carefully — the Queen is watching.\n\n"
        "{problem}\n\n"
        "Please put your final answer inside \\boxed{}."
    ),
    "combinatorics": (
        "In Alice's Wonderland, the Mad Hatter's tea party has infinite guests and "
        "finite seats. The following counting problem must be solved to arrange "
        "everyone properly.\n\n"
        "Determine the answer step by step:\n\n"
        "{problem}\n\n"
        "Please put your final answer inside \\boxed{}."
    ),
    "geometry": (
        "In Alice's Wonderland, space itself bends to mathematical laws. "
        "The following geometry problem was inscribed on the looking-glass wall. "
        "Solve it with full reasoning:\n\n"
        "{problem}\n\n"
        "Please put your final answer inside \\boxed{}."
    ),
    "calculus": (
        "In Alice's Wonderland, time flows like a river — sometimes fast, sometimes slow. "
        "The following problem involves rates of change that even the White Rabbit "
        "struggles to track.\n\n"
        "Solve step by step:\n\n"
        "{problem}\n\n"
        "Please put your final answer inside \\boxed{}."
    ),
    "probability": (
        "In Alice's Wonderland, the Cheshire Cat appears and disappears at random. "
        "Use probability theory to answer the following question that the Caterpillar "
        "has posed from atop his mushroom:\n\n"
        "{problem}\n\n"
        "Please put your final answer inside \\boxed{}."
    ),
    "olympiad": (
        "In Alice's Wonderland, the annual Mathematical Olympiad is held by the Queen of Hearts. "
        "Only those who reason with complete rigour may keep their heads. "
        "Solve the following competition problem:\n\n"
        "{problem}\n\n"
        "Please put your final answer inside \\boxed{}."
    ),
    "general": (
        "In Alice's Wonderland, every puzzle has a hidden answer that careful reasoning reveals. "
        "The following problem was left by the Mad Hatter. Solve it completely, "
        "showing every step of your thought process:\n\n"
        "{problem}\n\n"
        "Please put your final answer inside \\boxed{}."
    ),
}

SYSTEM_INSTRUCTION = (
    "You are a precise mathematical reasoner in Alice's Wonderland. "
    "Always think through the problem carefully inside <think>...</think> tags, "
    "then provide your final answer inside \\boxed{}."
)


# CATEGORY DETECTION (from problem text)
CATEGORY_KEYWORDS = {
    "algebra":       ["equation", "solve for", "simplify", "expression", "polynomial",
                      "factor", "roots", "quadratic", "linear", "system of"],
    "number_theory": ["prime", "divisible", "modulo", "gcd", "lcm", "remainder",
                      "integer", "divisor", "congruent", "coprime", "digit"],
    "combinatorics": ["how many ways", "permutation", "combination", "arrange",
                      "choose", "select", "count the", "committee", "paths"],
    "geometry":      ["triangle", "circle", "area", "perimeter", "angle", "radius",
                      "polygon", "square", "rectangle", "tangent", "hypotenuse"],
    "calculus":      ["derivative", "integral", "limit", "differentiate", "converge",
                      "series", "sum of", "sigma", "diverge", "function f(x)"],
    "probability":   ["probability", "expected", "random", "dice", "coin", "event",
                      "outcome", "distribution", "likelihood", "chance"],
    "olympiad":      ["prove that", "show that", "find all", "for all integers",
                      "exists", "maximum", "minimum", "inequality", "iff"],
}

def detect_math_category(text: str) -> str:
    text_lower = text.lower()
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[cat] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# COT FORMATTING
def format_cot_response(cot: str, answer: str) -> str:
    """Wrap CoT and answer into the required format."""
    # Clean up existing boxed if present
    answer_clean = re.sub(r"\\boxed\{([^}]*)\}", r"\1", answer).strip()

    # Strip any existing <think> tags from source CoT
    cot_clean = re.sub(r"</?think>", "", cot).strip()

    return f"<think>\n{cot_clean}\n</think>\n\\boxed{{{answer_clean}}}"

def build_wonderland_prompt(problem: str, category: str) -> str:
    template = WONDERLAND_CATEGORY_PROMPTS.get(category, WONDERLAND_CATEGORY_PROMPTS["general"])
    return template.format(problem=problem.strip())


# PER-DATASET ADAPTERS
def adapt_numina(row: dict) -> Optional[dict]:
    """
    NuminaMath-CoT: fields = {problem, solution}
    solution already contains step-by-step CoT ending in \\boxed{}.
    """
    problem = row.get("problem", "")
    solution = row.get("solution", "")
    if not problem or not solution:
        return None

    # Extract boxed answer
    boxed_match = re.search(r"\\boxed\{([^}]*)\}", solution)
    answer = boxed_match.group(1) if boxed_match else ""

    # Use the full solution as the CoT
    cot = solution

    category = detect_math_category(problem)
    prompt = build_wonderland_prompt(problem, category)
    response = format_cot_response(cot, answer)

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "_meta": {"source": "numina", "category": category},
    }

def adapt_nemotron(row: dict) -> Optional[dict]:
    """
    nvidia/OpenMathReasoning: fields vary by split.
    Common fields: {problem, solution, answer, difficulty}
    """
    problem = row.get("problem", "") or row.get("question", "")
    solution = row.get("solution", "") or row.get("generated_solution", "")
    answer = str(row.get("answer", "") or row.get("expected_answer", ""))
    if not problem:
        return None

    # Filter out low-quality samples
    if solution and len(solution.strip()) < 20:
        return None

    cot = solution if solution else f"Solving step by step.\nAnswer: {answer}"

    category = detect_math_category(problem)
    prompt = build_wonderland_prompt(problem, category)
    response = format_cot_response(cot, answer)

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "_meta": {"source": "nemotron", "category": category},
    }

def adapt_olympiadbench(row: dict) -> Optional[dict]:
    """
    AI-MO/OlympiadBench: fields = {problem, solution, answer, subject, level}
    """
    problem = row.get("problem", "")
    solution = row.get("solution", "")
    answer = str(row.get("answer", ""))
    if not problem:
        return None

    cot = solution if solution else f"Applying rigorous mathematical reasoning.\nAnswer: {answer}"

    category = detect_math_category(problem)
    prompt = build_wonderland_prompt(problem, "olympiad")  # always olympiad-level
    response = format_cot_response(cot, answer)

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "_meta": {"source": "olympiadbench", "category": "olympiad"},
    }

def adapt_olymmath(row: dict) -> Optional[dict]:
    """
    RUC-AIBOX/OlymMATH: fields = {problem, answer, solution, subject}
    """
    problem = row.get("problem", "")
    solution = row.get("solution", "")
    answer = str(row.get("answer", ""))
    if not problem:
        return None

    cot = solution if solution else f"Solving olympiad problem.\nAnswer: {answer}"

    category = detect_math_category(problem)
    prompt = build_wonderland_prompt(problem, "olympiad")
    response = format_cot_response(cot, answer)

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "_meta": {"source": "olymmath", "category": "olympiad"},
    }


# DATASET CONFIGS
DATASET_CONFIGS = {
    "numina": {
        "hf_path": "AI-MO/NuminaMath-CoT",
        "splits": ["train"],
        "adapter": adapt_numina,
        "description": "NuminaMath-CoT (860k+, competition to olympiad, built-in CoT)",
        "streaming": True,
    },
    "nemotron": {
        "hf_path": "nvidia/OpenMathReasoning",
        "splits": ["train"],
        "adapter": adapt_nemotron,
        "description": "Nemotron-Math / OpenMathReasoning (7.5M, olympiad+)",
        "streaming": True,
        "config_name": "cot",  # use the CoT subset
    },
    "olympiadbench": {
        "hf_path": "AI-MO/OlympiadBench",
        "splits": ["train"],
        "adapter": adapt_olympiadbench,
        "description": "OlympiadBench (582k+, pure olympiad-level)",
        "streaming": True,
    },
    "olymmath": {
        "hf_path": "RUC-AIBOX/OlymMATH",
        "splits": ["train"],
        "adapter": adapt_olymmath,
        "description": "OlymMATH (olympiad+, novel generation pipeline)",
        "streaming": True,
    },
}


# CORE PIPELINE
def process_dataset(
    name: str,
    config: dict,
    max_samples: Optional[int],
    rng: random.Random,
    skip_errors: bool = True,
) -> list:
    """Download and augment one dataset."""
    print(f"\n{'='*60}")
    print(f"  Dataset : {name}")
    print(f"  Source  : {config['hf_path']}")
    print(f"  Info    : {config['description']}")
    print(f"{'='*60}")

    adapter = config["adapter"]
    results = []
    errors = 0

    for split in config["splits"]:
        print(f"  Loading split '{split}'...")
        try:
            load_kwargs = dict(
                path=config["hf_path"],
                split=split,
                streaming=config.get("streaming", True),
                trust_remote_code=True,
            )
            if "config_name" in config:
                load_kwargs["name"] = config["config_name"]

            ds = load_dataset(**load_kwargs)

            count = 0
            for row in ds:
                if max_samples and count >= max_samples:
                    break
                try:
                    out = adapter(row)
                    if out:
                        results.append(out)
                        count += 1
                        if count % 5000 == 0:
                            print(f"    ... processed {count:,} samples")
                except Exception as e:
                    errors += 1
                    if not skip_errors and errors > 10:
                        raise
            print(f"  ✓ Collected {count:,} samples from '{split}'")

        except Exception as e:
            print(f"  ✗ Failed to load '{split}': {e}")
            if not skip_errors:
                raise

    if errors:
        print(f"  ⚠ Skipped {errors} malformed rows")

    return results

def validate_jsonl(path: str, sample_size=100):
    """Validate generated JSONL file."""
    print(f"\nValidating {path}...")
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            try:
                data = json.loads(line)
                assert "messages" in data, "Missing 'messages' field"
                assert len(data["messages"]) == 2, "Expected exactly 2 messages"
                assert data["messages"][1]["content"].count("\\boxed{") == 1, "Response missing \\boxed{}"
            except Exception as e:
                print(f"  ✗ Validation failed at line {i+1}: {e}")
                return False
    print(f"  ✓ Validation passed for {sample_size} samples")
    return True

def run_pipeline(
    datasets: list[str],
    output_path: str,
    max_per_dataset: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
    keep_meta: bool = False,
    validate: bool = True,
):
    rng = random.Random(seed)
    all_samples = []
    stats = {}

    for name in datasets:
        if name not in DATASET_CONFIGS:
            print(f"Unknown dataset '{name}', skipping. Valid: {list(DATASET_CONFIGS.keys())}")
            continue
        samples = process_dataset(name, DATASET_CONFIGS[name], max_per_dataset, rng)
        stats[name] = len(samples)
        all_samples.extend(samples)

    print(f"\n{'='*60}")
    print(f"  AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    for name, count in stats.items():
        print(f"  {name:<20} {count:>10,}")
    print(f"  {'TOTAL':<20} {len(all_samples):>10,}")

    if shuffle:
        print(f"\n  Shuffling {len(all_samples):,} samples (seed={seed})...")
        rng.shuffle(all_samples)

    # Write JSONL
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Writing to {out}...")
    with open(out, "w", encoding="utf-8") as f:
        for item in all_samples:
            write_item = item if keep_meta else {"messages": item["messages"]}
            f.write(json.dumps(write_item, ensure_ascii=False) + "\n")

    print(f"  ✓ Saved {len(all_samples):,} examples → {out}")

    # Write a small stats JSON
    stats_path = out.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "total": len(all_samples),
            "by_dataset": stats,
            "output": str(out),
            "seed": seed,
            "max_per_dataset": max_per_dataset,
        }, f, indent=2)
    print(f"  ✓ Stats → {stats_path}")

    # Validate the output
    if validate:
        validate_jsonl(out)

    return all_samples


# QUICK VERIFY (dry run without HuggingFace)
def dry_run_verify():
    """Verify prompt templates and adapter logic with fake data."""
    print("Running dry-run verification...\n")

    fake_rows = {
        "numina": {
            "problem": "Find the number of integers n such that 1 ≤ n ≤ 100 and n² + n + 1 is divisible by 5.",
            "solution": "We need n² + n + 1 ≡ 0 (mod 5).\nTesting n = 0,1,2,3,4 mod 5:\n- n≡0: 1 → not 0\n- n≡1: 3 → not 0\n- n≡2: 7≡2 → not 0\n- n≡3: 13≡3 → not 0\n- n≡4: 21≡1 → not 0\nSo no solution exists.\n\\boxed{0}",
        },
        "nemotron": {
            "problem": "Let f(x) = x³ - 3x. Find all local maxima and minima.",
            "solution": "f'(x) = 3x² - 3 = 3(x-1)(x+1). Critical points at x=±1.\nf(-1)=2 (local max), f(1)=-2 (local min).",
            "answer": "local max at x=-1 with value 2; local min at x=1 with value -2",
        },
        "olympiadbench": {
            "problem": "Prove that for any positive integers a, b, c: (a+b)(b+c)(c+a) ≥ 8abc.",
            "solution": "By AM-GM: a+b ≥ 2√(ab), b+c ≥ 2√(bc), c+a ≥ 2√(ca).\nMultiplying: (a+b)(b+c)(c+a) ≥ 8√(ab·bc·ca) = 8abc.",
            "answer": "Proved",
        },
        "olymmath": {
            "problem": "How many ways can you tile a 2×10 board with 2×1 dominoes?",
            "solution": "Let f(n) = ways to tile 2×n. f(1)=1, f(2)=2, f(n)=f(n-1)+f(n-2).\nf(10) = 89.",
            "answer": "89",
        },
    }

    adapters = {
        "numina": adapt_numina,
        "nemotron": adapt_nemotron,
        "olympiadbench": adapt_olympiadbench,
        "olymmath": adapt_olymmath,
    }

    for name, row in fake_rows.items():
        result = adapters[name](row)
        if result:
            print(f"[{name}]")
            print(f"  Category  : {result['_meta']['category']}")
            print(f"  Prompt    : {result['messages'][0]['content'][:120]}...")
            print(f"  Response  : {result['messages'][1]['content'][:120]}...")
            print()
        else:
            print(f"[{name}] FAILED - adapter returned None\n")

    print("Dry-run complete ✓")


# CLI
def main():
    parser = argparse.ArgumentParser(
        description="Augment math datasets into Wonderland-style Nemotron training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (no HuggingFace needed):
  python augment_pipeline.py --dry-run

  # Download NuminaMath only, 50k samples:
  python augment_pipeline.py --datasets numina --max-per-dataset 50000

  # All datasets, 20k each, output to custom path:
  python augment_pipeline.py --datasets numina nemotron olympiadbench olymmath \\
      --max-per-dataset 20000 --output data/augmented/all_wonderland.jsonl

  # Full run (warning: numina=860k, nemotron=7.5M — use max-per-dataset):
  python augment_pipeline.py --datasets numina nemotron \\
      --max-per-dataset 100000 --output data/augmented/large.jsonl
        """,
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["numina", "nemotron", "olympiadbench", "olymmath"],
        choices=list(DATASET_CONFIGS.keys()),
        help="Which datasets to process",
    )
    parser.add_argument(
        "--output", default="data/augmented/train_wonderland.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--max-per-dataset", type=int, default=None,
        help="Max samples per dataset (use for testing; omit for full run)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="Disable output shuffling",
    )
    parser.add_argument(
        "--keep-meta", action="store_true",
        help="Keep _meta field in output (source + category)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Test adapters with fake data (no HuggingFace download)",
    )
    parser.add_argument(
        "--list-datasets", action="store_true",
        help="List available datasets and exit",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip JSONL validation after generation",
    )

    args = parser.parse_args()

    if args.list_datasets:
        print("\nAvailable datasets:")
        for name, cfg in DATASET_CONFIGS.items():
            print(f"  {name:<20} {cfg['description']}")
        return

    if args.dry_run:
        dry_run_verify()
        return

    if not HF_AVAILABLE:
        print("ERROR: 'datasets' package not found. Install with:")
        print("  pip install datasets --break-system-packages")
        sys.exit(1)

    run_pipeline(
        datasets=args.datasets,
        output_path=args.output,
        max_per_dataset=args.max_per_dataset,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        keep_meta=args.keep_meta,
        validate=not args.no_validate,
    )

if __name__ == "__main__":
    main()