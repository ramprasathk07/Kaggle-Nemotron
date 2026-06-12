#!/usr/bin/env python3
"""
rule_trace_generator.py — Nemotron Reasoning Challenge

Generates SFT traces that turn rule INDUCTION into rule RETRIEVAL:
the trace enumerates candidate rules from a finite library, tests each
against the in-prompt examples (computed, never hallucinated), eliminates
failures, applies the surviving rule to the query, verifies, and emits
\\boxed{answer}.

Every claim inside a generated trace is produced by EXECUTING the rule
functions, so traces are mechanically truthful. A problem is only emitted
if the discovered rule reproduces the ground-truth answer exactly — so this
doubles as your generator-validation harness ("reconstruct train.csv at
100% before trusting synthetic data").

USAGE
  1. For each weak category in your failure ledger, add a Category with:
       - parse(prompt)  -> (example_pairs, query_input)   [regex on real prompts]
       - canon(answer)  -> str  (EXACT ground-truth formatting: leading zeros,
                                 decimal places for the ±1e-2 categories, etc.)
       - rules: the finite rule list (mine Hui Kang's visualizer notes and
                andy279's solver-discovered rules)
  2. build_sft_dataset(problems, ...) -> JSONL in Nemotron thinking format:
       {"messages":[{user},{assistant:"<think>...</think>\\boxed{a}"}],
        "answer_span":[s,e], ...}
     answer_span = char offsets of the final boxed answer in the assistant
     string, for wiring 5–10x answer-token loss upweighting in your SFT stack.
  3. Run this file directly for a working end-to-end demo on two toy
     categories (bit transforms + affine sequences).

CAVEAT (provisional_box): the early-box-inside-<think> insurance assumes the
official extractor scans the whole generation and prefers the LAST box.
Verify that in the metric code before enabling; default is OFF.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

# --------------------------------------------------------------------------
# Core data structures
# --------------------------------------------------------------------------


@dataclass
class Rule:
    rule_id: str
    description: str                      # natural language used inside the trace
    fn: Callable[[str], str]              # input string -> output string (raw)

    def apply(self, x: str) -> Optional[str]:
        try:
            return self.fn(x)
        except Exception:
            return None                   # rule not applicable to this input


@dataclass
class Category:
    name: str
    parse: Callable[[str], tuple[list[tuple[str, str]], str]]
    canon: Callable[[str], str]           # canonical ground-truth formatting
    rules: list[Rule]
    extra_verify: Optional[Callable[[str, str], str]] = None
    # extra_verify(query_input, answer) -> short sanity sentence, e.g. a
    # length check for bit strings. Keep it cheap and truthful.


@dataclass
class TraceConfig:
    n_distractors: int = 2        # distractor rules enumerated per trace
    shuffle_candidates: bool = True
    provisional_box: bool = False # early \boxed inside <think> (see caveat)
    verify: bool = True           # re-apply rule + extra_verify before final box
    max_chars: int = 6000         # hard cap; retries with fewer distractors
    seed: int = 0


# --------------------------------------------------------------------------
# Trace rendering — every number/string below is COMPUTED, never asserted
# --------------------------------------------------------------------------


def _test_rule(rule: Rule, pairs: list[tuple[str, str]], canon) -> tuple[bool, list[str]]:
    """Run rule on all example pairs. Returns (fits_all, narrated_check_lines)."""
    lines = []
    for i, (xin, xout) in enumerate(pairs, 1):
        got = rule.apply(xin)
        got_c = canon(got) if got is not None else None
        want_c = canon(xout)
        if got_c == want_c:
            lines.append(f"  example {i}: {rule.rule_id}({xin}) = {got_c} ✓ matches {want_c}")
        else:
            shown = got_c if got_c is not None else "<not applicable>"
            lines.append(f"  example {i}: {rule.rule_id}({xin}) = {shown} ✗ expected {want_c} — eliminated")
            return False, lines
    return True, lines


def render_trace(cat: Category, pairs, query: str, true_rule: Rule,
                 distractors: list[Rule], cfg: TraceConfig,
                 rng: random.Random) -> Optional[str]:
    candidates = [true_rule] + list(distractors)
    if cfg.shuffle_candidates:
        rng.shuffle(candidates)

    p = []
    p.append("<think>")
    p.append("I need to identify the hidden transformation rule from the "
             "given input→output examples, then apply it to the query.")
    p.append("")
    p.append("Parsed examples:")
    for xin, xout in pairs:
        p.append(f"  {xin} -> {cat.canon(xout)}")
    p.append(f"Query input: {query}")
    p.append("")
    p.append(f"Known rule family for this puzzle type ({len(candidates)} candidates):")
    for r in candidates:
        p.append(f"  [{r.rule_id}] {r.description}")
    p.append("")
    p.append("Testing candidates against all examples:")

    identified = None
    for r in candidates:
        fits, lines = _test_rule(r, pairs, cat.canon)
        p.append(f"Candidate [{r.rule_id}]:")
        p.extend(lines)
        if fits:
            identified = r
            p.append(f"[{r.rule_id}] fits every example. Rule identified: {r.description}")
            break
    if identified is None or identified.rule_id != true_rule.rule_id:
        return None  # shuffled distractor accidentally fits all pairs → ambiguous, skip

    raw = identified.apply(query)
    if raw is None:
        return None
    ans = cat.canon(raw)
    p.append("")
    p.append(f"Applying [{identified.rule_id}] to the query: "
             f"{identified.rule_id}({query}) = {ans}")

    if cfg.provisional_box:
        p.append(f"Provisional answer: \\boxed{{{ans}}}")

    if cfg.verify:
        recheck = cat.canon(identified.apply(query))
        ok = "consistent" if recheck == ans else "INCONSISTENT"
        p.append(f"Verification: independently re-applying the rule gives {recheck} — {ok}.")
        if recheck != ans:
            return None
        if cat.extra_verify is not None:
            p.append(f"Sanity check: {cat.extra_verify(query, ans)}")

    p.append(f"Final answer: {ans}")
    p.append("</think>")
    p.append(f"\\boxed{{{ans}}}")
    return "\n".join(p)


# --------------------------------------------------------------------------
# Dataset builder + library-coverage report
# --------------------------------------------------------------------------


def discover_rule(cat: Category, pairs, query: str, gt_answer: str
                  ) -> tuple[Optional[Rule], str]:
    """Find library rules fitting ALL pairs; require agreement with ground truth."""
    fitting = [r for r in cat.rules if _test_rule(r, pairs, cat.canon)[0]]
    if not fitting:
        return None, "no_rule_fits_examples"
    matching = [r for r in fitting
                if r.apply(query) is not None
                and cat.canon(r.apply(query)) == cat.canon(gt_answer)]
    if not matching:
        return None, "rule_fits_examples_but_not_answer"   # wrong/ambiguous rule → inspect!
    if len(fitting) > 1:
        return matching[0], "ambiguous_multiple_rules_fit"  # cryptarithm-operator-style trap
    return matching[0], "ok"


def _answer_span(assistant_text: str) -> tuple[int, int]:
    i = assistant_text.rfind("\\boxed{")
    s = i + len("\\boxed{")
    return s, assistant_text.find("}", s)


def build_sft_dataset(problems: Iterable[dict], categories: dict[str, Category],
                      cfg: TraceConfig, out_path: str) -> dict:
    """problems: dicts with keys problem_id, category, prompt, answer."""
    rng = random.Random(cfg.seed)
    stats: dict = {"total": 0, "written": 0, "skipped": {}, "ambiguous": 0,
                   "per_category": {}}
    with open(out_path, "w") as f:
        for prob in problems:
            stats["total"] += 1
            cname = prob["category"]
            pc = stats["per_category"].setdefault(cname, {"written": 0, "skipped": 0})
            cat = categories.get(cname)
            if cat is None:
                stats["skipped"]["no_category"] = stats["skipped"].get("no_category", 0) + 1
                pc["skipped"] += 1
                continue
            try:
                pairs, query = cat.parse(prob["prompt"])
            except Exception:
                stats["skipped"]["parse_error"] = stats["skipped"].get("parse_error", 0) + 1
                pc["skipped"] += 1
                continue

            rule, status = discover_rule(cat, pairs, query, prob["answer"])
            if rule is None:
                stats["skipped"][status] = stats["skipped"].get(status, 0) + 1
                pc["skipped"] += 1
                continue
            if status == "ambiguous_multiple_rules_fit":
                stats["ambiguous"] += 1

            trace = None
            for nd in range(cfg.n_distractors, -1, -1):       # shrink under the char cap
                pool = [r for r in cat.rules if r.rule_id != rule.rule_id]
                trace = render_trace(cat, pairs, query, rule,
                                     rng.sample(pool, min(nd, len(pool))), cfg, rng)
                if trace is not None and len(trace) <= cfg.max_chars:
                    break
                trace = None
            if trace is None:
                stats["skipped"]["render_failed_or_too_long"] = \
                    stats["skipped"].get("render_failed_or_too_long", 0) + 1
                pc["skipped"] += 1
                continue

            s, e = _answer_span(trace)
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": prob["prompt"]},
                    {"role": "assistant", "content": trace},
                ],
                "problem_id": prob.get("problem_id", ""),
                "category": cname,
                "rule_id": rule.rule_id,
                "discovery_status": status,
                "answer_span": [s, e],          # for answer-token loss upweighting
                "est_tokens": len(trace) // 3,  # conservative ~3 chars/token
            }) + "\n")
            stats["written"] += 1
            pc["written"] += 1
    return stats


# --------------------------------------------------------------------------
# TOY DEMO — two fully worked categories. Replace with real ones.
# --------------------------------------------------------------------------

_TOY_RE = re.compile(
    r"Examples:\n((?:\S+ -> \S+\n)+)Now apply the rule to: (\S+)", re.M)


def _toy_parse(prompt: str):
    m = _TOY_RE.search(prompt)
    pairs = [tuple(line.split(" -> ")) for line in m.group(1).strip().split("\n")]
    return pairs, m.group(2)


def _rotl(s: str) -> str: return s[1:] + s[0]
def _rotr(s: str) -> str: return s[-1] + s[:-1]


def toy_bit_category() -> Category:
    rules = [
        Rule("REV",  "reverse the bit string", lambda s: s[::-1]),
        Rule("INV",  "invert every bit (0↔1)",
             lambda s: "".join("1" if c == "0" else "0" for c in s)),
        Rule("ROTL", "rotate left by one position", _rotl),
        Rule("ROTR", "rotate right by one position", _rotr),
        Rule("REVINV", "reverse the string, then invert every bit",
             lambda s: "".join("1" if c == "0" else "0" for c in s[::-1])),
    ]
    return Category(
        name="toy_bits", parse=_toy_parse, canon=lambda s: s.strip(), rules=rules,
        extra_verify=lambda q, a: (
            f"output length {len(a)} matches input length {len(q)}"
            if len(a) == len(q) else f"LENGTH MISMATCH {len(a)} vs {len(q)}"),
    )


def _canon_num(s) -> str:
    v = float(s)
    return str(int(v)) if v == int(v) else f"{v:.2f}"   # ±1e-2 categories: fix decimals


def toy_affine_category() -> Category:
    mk = lambda f: (lambda s: _canon_num(f(float(s))))
    rules = [
        Rule("2X+1", "double the input and add one",        mk(lambda x: 2 * x + 1)),
        Rule("3X-2", "triple the input and subtract two",   mk(lambda x: 3 * x - 2)),
        Rule("SQ-1", "square the input and subtract one",   mk(lambda x: x * x - 1)),
        Rule("2X-3", "double the input and subtract three", mk(lambda x: 2 * x - 3)),
        Rule("10-X", "subtract the input from ten",         mk(lambda x: 10 - x)),
    ]
    return Category(name="toy_affine", parse=_toy_parse, canon=_canon_num, rules=rules)


def _toy_prompt(pairs, query) -> str:
    ex = "\n".join(f"{a} -> {b}" for a, b in pairs)
    return (f"Find the hidden rule.\nExamples:\n{ex}\n"
            f"Now apply the rule to: {query}\n"
            f"Please put your final answer inside \\boxed{{}}.")


def _demo_problems(rng: random.Random):
    probs = []
    bits = toy_bit_category()
    for i in range(4):
        rule = rng.choice(bits.rules)
        ins: list[str] = []
        while len(ins) < 3:
            s = "".join(rng.choice("01") for _ in range(rng.randint(5, 8)))
            if s not in ins:
                ins.append(s)
        pairs = [(x, rule.fn(x)) for x in ins[:2]]
        probs.append({"problem_id": f"bits_{i}", "category": "toy_bits",
                      "prompt": _toy_prompt(pairs, ins[2]),
                      "answer": rule.fn(ins[2])})
    aff = toy_affine_category()
    for i in range(4):
        rule = rng.choice(aff.rules)
        xs = rng.sample(range(2, 30), 3)
        pairs = [(str(x), rule.fn(str(x))) for x in xs[:2]]
        probs.append({"problem_id": f"aff_{i}", "category": "toy_affine",
                      "prompt": _toy_prompt(pairs, str(xs[2])),
                      "answer": rule.fn(str(xs[2]))})
    return probs


if __name__ == "__main__":
    rng = random.Random(42)
    cats = {c.name: c for c in (toy_bit_category(), toy_affine_category())}
    problems = _demo_problems(rng)
    cfg = TraceConfig(n_distractors=2, provisional_box=False, seed=42)
    stats = build_sft_dataset(problems, cats, cfg, "demo_sft.jsonl")
    print("=== coverage stats ===")
    print(json.dumps(stats, indent=2))
    with open("demo_sft.jsonl") as f:
        rec = json.loads(f.readline())
    print("\n=== sample assistant trace ===")
    print(rec["messages"][1]["content"])
    s, e = rec["answer_span"]
    print(f"\nanswer_span check -> '{rec['messages'][1]['content'][s:e]}'")
