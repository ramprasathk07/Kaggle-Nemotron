#!/usr/bin/env python3
"""
Read source.csv (cols: prompt, answer), classify into families, balance,
generate CoT via LLM (multiple API keys cycling), write final_output.csv.

Supports multiple AISA_API_KEY_1, AISA_API_KEY_2, ... for rate‑limit resilience.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, List

import pandas as pd
from dotenv import load_dotenv
from openai import APIError, APIStatusError, OpenAI, RateLimitError


# ---------------------------------------------------------------------------
# 1. Classification (unchanged)
# ---------------------------------------------------------------------------

LABELS = [
    "numeric-rule",
    "binary-transform",
    "roman-numeral",
    "mapping-symbolic",
    "sequence-rule",
    "string-transform",
    "logic-short-answer",
    "encoding-decoding",
    "table-like-rule",
    "mixed-template",
]

CLASSIFY_RULES: list[tuple[str, str]] = [
    ("binary-transform", r"(?:\bbit\s*m[ai]n[ip]*u?l?ation\b|\bbit\s*wis?e\b|\bbit\s*shift\b|\bbit\s*rotat\w*|\bxor\b|\bxnor\b|\b8[-\s]?bit\b|\bbinary\s+numbers?\b|\bmajority\s+function\b|\bchoice\s+function\b)"),
    ("roman-numeral", r"(?:\broma+n\s+numeral\w*|\broma+n\b(?=[^.]{0,80}\bnumber\b)|\bnumeral\s+system\b|\bwonderland\s+numeral\b)"),
    ("encoding-decoding", r"(?:\b(?:en|de)\s*c?rypt\w*|\b(?:en|de)\s*cod\w*|\bcaes[ae]r\b|\bciph[ae]r\w*|\bsubstitut\w*|\bsecret\s+encryption\b)"),
    ("mapping-symbolic", r"(?:transformation\s+rules?[^.\n]{0,80}equations?|`[^`\n]*[!@#$%^&*()\[\]{}<>'\"\\][^`\n]*`\s*=\s*`?|\bsymbolic\s+(?:rule|mapping|transform)\w*)"),
    ("numeric-rule", r"(?:\bgravitation\w*|\bunit\s+conversion\b|\bd\s*=\s*0?\.5\s*\*\s*g\s*\*\s*t\s*\^?\s*2|\bmeasurement\w*\s+(?:is\s+)?(?:secretly\s+)?(?:applied|converted)|\bm\s+becomes\s+\d|\bsecret\s+formula\b|\bsecretly\s+(?:changed|converted))"),
    ("sequence-rule", r"(?:\bsequence\b|\bnext\s+(?:term|number|element|value)\b|\bcontinue\s+the\s+(?:pattern|sequence)\b|\barithmetic\s+progression\b|\bgeometric\s+progression\b|\bnth\s+term\b)"),
    ("string-transform", r"(?:\banagram\w*|\bpalindrom\w*|\breverse\s+the\s+(?:string|text|word)|\bstring\s+(?:transform|manipulation)\w*|\bcase\s+(?:swap|flip)\b)"),
    ("logic-short-answer", r"(?:\blogic\s+puzzle\w*|\briddle\w*|\btrue\s+or\s+false\b|\byes\s+or\s+no\b|\bknights?\s+and\s+knaves\b)"),
    ("table-like-rule", r"(?:\btable\b|\bgrid\b|\bmatrix\b|\brows?\s+and\s+columns?\b|\bcell\s+at\s+row\b)"),
]

COMPILED_RULES = [(label, re.compile(pat, re.IGNORECASE | re.DOTALL)) for label, pat in CLASSIFY_RULES]

def classify(prompt: str) -> str:
    p = prompt or ""
    for label, rx in COMPILED_RULES:
        if rx.search(p):
            return label
    return "mixed-template"


# ---------------------------------------------------------------------------
# 2. LLM client with multiple API keys (round‑robin + rate‑limit rotation)
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.aisa.one/v1"
DEFAULT_MODEL = "deepseek-v3.2"

AISA_PROVIDER_NAME = "aisa"
DEFAULT_PROVIDERS: dict[str, dict[str, str]] = {
    AISA_PROVIDER_NAME: {"base_url": "https://api.aisa.one/v1", "api_key_env": "AISA_API_KEY"},
}

DEFAULT_ROUTES: dict[str, dict[str, str]] = {
    "binary-transform":   {"provider": "aisa", "model": "claude-opus-4-7"},
    "mapping-symbolic":   {"provider": "aisa", "model": "gpt-5.4"},
    "logic-short-answer": {"provider": "aisa", "model": "deepseek-r1"},
    "encoding-decoding":  {"provider": "aisa", "model": "gemini-3.1-pro-preview"},
    "numeric-rule":       {"provider": "aisa", "model": "claude-sonnet-4-6-thinking"},
    "sequence-rule":      {"provider": "aisa", "model": "kimi-k2-thinking"},
    "roman-numeral":      {"provider": "aisa", "model": "glm-5"},
    "string-transform":   {"provider": "aisa", "model": "gemini-2.5-flash"},
    "table-like-rule":    {"provider": "aisa", "model": "qwen3.6-plus"},
    "mixed-template":     {"provider": "aisa", "model": "deepseek-v3.2"},
    "_default":           {"provider": "aisa", "model": DEFAULT_MODEL},
}

_NO_EMOJI_RULE = (
    " Use plain ASCII / standard Latin characters only - absolutely NO emojis, "
    "pictographs, decorative Unicode symbols, or icon characters anywhere."
)

# New format: <think> ... </think> \n \boxed{answer}
_OUTPUT_FORMAT_RULE = (
    " OUTPUT FORMAT - the 'cot' field MUST decode to:\n"
    "<think>\n<your detailed step-by-step reasoning>\n</think>\n"
    "\\boxed{<VERIFIED ANSWER, exact characters>}\n"
    "Rules: (1) start with <think> on its own line; (2) include </think> on its own line after reasoning; "
    "(3) the next line MUST be \\boxed{...} with the answer; (4) inside JSON escape backslash as \\\\boxed; "
    "(5) reply ONLY with {\"cot\": \"<think>\\n...\\n</think>\\n\\\\boxed{<answer>}\"}."
)

# Longer, more thorough system prompts (no sentence limit)
SYSTEM_PROMPT_VARIANTS = [
    "You are an expert puzzle solver. Given a PUZZLE and its VERIFIED ANSWER, produce a detailed, step-by-step chain of reasoning (aim for 500–2000 tokens) that derives the verified answer from the evidence. Use only facts in the puzzle. Never mention the answer was given. End with the exact answer in \\boxed{}." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
    "You are a methodical analyst. Write a thorough step-by-step derivation (500–2000 tokens). First identify the rule that fits every example, then apply it to the query. Show calculations. Conclude with the exact answer." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
    "You are a patient tutor. Walk through the puzzle in 8–12 logical steps, referencing examples by value. Derive the answer exactly, without mentioning it was given." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
    "You are a careful reasoner. Propose a candidate rule, verify against examples, apply to query, compute result. Ensure final computed value matches verified answer exactly." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
    "You are a terse but thorough expert. Derive the answer in 8–12 clear steps: identify rule, apply, conclude with exact answer. No meta‑commentary." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
]

USER_TEMPLATE = (
    "PUZZLE:\n{prompt}\n\nVERIFIED ANSWER: {answer}\n\n"
    "Produce a detailed chain of thought inside <think>...</think> followed by \\boxed{{{answer}}}.\n"
    'Reply ONLY with: {{"cot": "<think>\\n...\\n</think>\\n\\\\boxed{{{answer}}}"}}'
)

_EMOJI_RX = re.compile(
    "["
    "\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    "\U0001F1E6-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF"
    "\U0001F000-\U0001F0FF\U0000FE0E\U0000FE0F]"
)

def _contains_emoji(text: str) -> bool:
    return bool(_EMOJI_RX.search(text or ""))


class MultiKeyClient:
    """Round‑robin client pool for a single provider with multiple API keys."""
    def __init__(self, base_url: str, api_keys: List[str]):
        self.base_url = base_url
        self.clients = [OpenAI(base_url=base_url, api_key=k) for k in api_keys]
        self.n = len(self.clients)
        self.idx = 0
        self.lock = random.Random()  # not thread‑safe, but fine for sequential usage

    def get_client(self) -> OpenAI:
        c = self.clients[self.idx % self.n]
        self.idx = (self.idx + 1) % self.n
        return c

    def rotate(self):
        """Manually rotate to next client (e.g., after rate limit)."""
        self.idx = (self.idx + 1) % self.n


class ClientPool:
    """Manages per‑provider client pools. For AIsa, collects all AISA_API_KEY_*."""
    def __init__(self, providers: dict[str, dict[str, str]]):
        self._pools: dict[str, MultiKeyClient] = {}
        for name, spec in providers.items():
            if name == AISA_PROVIDER_NAME:
                # Gather all AISA_API_KEY_1, _2, ... from environment
                keys = []
                i = 1
                while True:
                    key = os.environ.get(f"AISA_API_KEY_{i}")
                    if key:
                        keys.append(key)
                        i += 1
                    else:
                        break
                if not keys:
                    # fallback to single key from spec["api_key_env"]
                    single_key = os.environ.get(spec["api_key_env"])
                    if single_key:
                        keys = [single_key]
                if not keys:
                    raise RuntimeError(f"No AISA API keys found. Set AISA_API_KEY_1, _2, ... or {spec['api_key_env']}")
                self._pools[name] = MultiKeyClient(spec["base_url"], keys)
            else:
                # Other providers (not used here) – not implemented, but we require only aisa.
                pass

    def get(self, provider_name: str) -> OpenAI:
        if provider_name not in self._pools:
            raise KeyError(f"Provider '{provider_name}' not initialized. Only '{AISA_PROVIDER_NAME}' supported.")
        return self._pools[provider_name].get_client()

    def rotate_provider(self, provider_name: str):
        if provider_name in self._pools:
            self._pools[provider_name].rotate()


def load_routing_config(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Routing config {path} must be a JSON object.")
    providers = {**DEFAULT_PROVIDERS, **(cfg.get("providers") or {})}
    routes = {**DEFAULT_ROUTES, **(cfg.get("routes") or {})}
    return providers, routes

def write_sample_routing_config(path: Path) -> None:
    sample = {"providers": DEFAULT_PROVIDERS, "routes": DEFAULT_ROUTES}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    print(f"Sample routing config written to {path}")

def resolve_route(routes: dict[str, dict[str, str]], label: str) -> tuple[str, str]:
    r = routes.get(label) or routes.get("_default")
    if not r or "provider" not in r or "model" not in r:
        raise KeyError(f"No route for label '{label}' and no usable _default route.")
    return r["provider"], r["model"]

def _is_rate_limit(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    return "429" in str(exc) or "rate limit" in str(exc).lower()

def _is_transient(exc: Exception) -> bool:
    if _is_rate_limit(exc):
        return True
    if isinstance(exc, (APIError, APIStatusError)):
        s = getattr(exc, "status_code", None)
        if s is None or s >= 500 or s == 408:
            return True
    msg = str(exc).lower()
    return any(x in msg for x in ("timeout", "timed out", "connection", "503", "502", "504"))

def _parse_cot(raw: str, expected_answer: str) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    if start == -1:
        return None
    depth, end = 0, -1
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    try:
        obj = json.loads(s[start:end+1])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    cot = str(obj.get("cot") or "").strip()
    if "<think>" not in cot or "</think>" not in cot:
        return None
    lines = [ln.strip() for ln in cot.splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    if not (last.startswith(r"\boxed{") and last.endswith("}")):
        return None
    if expected_answer and expected_answer.strip():
        if expected_answer.strip() not in cot:
            return None
    if _contains_emoji(cot):
        return None
    return cot

def call_llm(
    client_pool: ClientPool,
    provider_name: str,
    model: str,
    prompt: str,
    answer: str,
    variation_idx: int,
    max_attempts: int = 6,
) -> str:
    sys_prompt = SYSTEM_PROMPT_VARIANTS[variation_idx % len(SYSTEM_PROMPT_VARIANTS)]
    user_msg = USER_TEMPLATE.format(prompt=str(prompt)[:6000], answer=str(answer)[:1000])

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        client = client_pool.get(provider_name)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.5 if variation_idx > 0 else 0.2,
                seed=1000 + variation_idx * 13 + attempt,
                max_tokens=8192,   # AIsa supports up to 8192
            )
        except Exception as e:
            last_exc = e
            if _is_rate_limit(e):
                # rotate to next key and retry
                client_pool.rotate_provider(provider_name)
                wait = min(2.0 ** (attempt - 1) + random.uniform(0, 1), 60.0)
                time.sleep(wait)
                continue
            if not _is_transient(e):
                raise
            time.sleep(min(2.0 ** (attempt - 1), 60.0) + random.uniform(0, 1))
            continue
        text = resp.choices[0].message.content or ""
        cot = _parse_cot(text, str(answer))
        if cot:
            return cot
        last_exc = RuntimeError(f"invalid output: {text[:120]}")
        time.sleep(1.0 + random.uniform(0, 0.5))
    raise RuntimeError(f"exhausted {max_attempts} attempts: {type(last_exc).__name__}: {last_exc}")


# ---------------------------------------------------------------------------
# 3. Balancing (unchanged)
# ---------------------------------------------------------------------------

def build_plan(df: pd.DataFrame, target: int, rng: random.Random) -> list[dict[str, Any]]:
    plan = []
    for lbl in LABELS:
        rows = df[df["label"] == lbl].to_dict("records")
        if not rows:
            continue
        if len(rows) >= target:
            picked = rng.sample(rows, target)
            for r in picked:
                plan.append({"prompt": r["prompt"], "answer": r["answer"], "label": lbl, "variation_idx": 0})
        else:
            for r in rows:
                plan.append({"prompt": r["prompt"], "answer": r["answer"], "label": lbl, "variation_idx": 0})
            need = target - len(rows)
            v, i = 1, 0
            while need > 0:
                r = rows[i % len(rows)]
                plan.append({"prompt": r["prompt"], "answer": r["answer"], "label": lbl, "variation_idx": v})
                i += 1
                if i % len(rows) == 0:
                    v += 1
                need -= 1
    return plan


# ---------------------------------------------------------------------------
# 4. Output / resume (unchanged)
# ---------------------------------------------------------------------------

CSV_FIELDS = ["prompt", "answer", "cot", "label"]

def row_key(prompt: str, answer: str, var: int) -> str:
    h = hashlib.md5(f"{prompt}|||{answer}".encode("utf-8")).hexdigest()[:16]
    return f"{h}#{var}"

def load_progress(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return set()

def save_progress(path: Path, keys: set[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(sorted(keys)), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Balanced CoT generation with multi‑key support.")
    ap.add_argument("--input", type=Path, default=Path("data/src/train.csv"))
    ap.add_argument("--output", type=Path, default=Path("final_output.csv"))
    ap.add_argument("--progress", type=Path, default=None)
    ap.add_argument("--routing-config", type=Path, default=None)
    ap.add_argument("--write-sample-routing", type=Path, default=None)
    ap.add_argument("--only-labels", default=None)
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--max-per-label", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()

def main() -> int:
    load_dotenv()
    args = parse_args()

    if args.write_sample_routing:
        write_sample_routing_config(args.write_sample_routing)
        return 0

    if args.progress is None:
        args.progress = args.output.with_suffix(args.output.suffix + ".progress.json")

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    csv.field_size_limit(10 * 1024 * 1024)
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"ERROR: failed to read {args.input}: {e}", file=sys.stderr)
        return 2

    required = {"prompt", "answer"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing columns {missing}", file=sys.stderr)
        return 2

    df = df.dropna(subset=["prompt", "answer"]).reset_index(drop=True)
    df["prompt"] = df["prompt"].astype(str)
    df["answer"] = df["answer"].astype(str)
    df["label"] = df["prompt"].map(classify)

    before = Counter(df["label"])
    print("\n=== Label counts BEFORE balancing ===")
    for lbl in LABELS:
        print(f"  {lbl:22s} {before.get(lbl, 0):>6d}")
    print(f"  {'TOTAL':22s} {sum(before.values()):>6d}")

    nonzero = [c for c in before.values() if c > 0]
    if not nonzero:
        print("ERROR: no rows after classification.", file=sys.stderr)
        return 2

    target = args.target if args.target > 0 else min(nonzero)
    if args.max_per_label > 0:
        target = min(target, args.max_per_label)
    print(f"\nTarget per label: {target}")

    rng = random.Random(args.seed)
    plan = build_plan(df, target, rng)

    if args.only_labels:
        wanted = {s.strip() for s in args.only_labels.split(",") if s.strip()}
        plan = [p for p in plan if p["label"] in wanted]
        print(f"[only-labels] filtered to {len(plan)} rows")

    after = Counter(item["label"] for item in plan)
    print("\n=== Label counts AFTER balancing (planned) ===")
    for lbl in LABELS:
        print(f"  {lbl:22s} {after.get(lbl, 0):>6d}")
    print(f"  {'TOTAL':22s} {sum(after.values()):>6d}")

    # Resolve routing
    if args.routing_config:
        if not args.routing_config.exists():
            print(f"ERROR: routing config not found: {args.routing_config}", file=sys.stderr)
            return 2
        providers, routes = load_routing_config(args.routing_config)
    else:
        providers = DEFAULT_PROVIDERS
        routes = DEFAULT_ROUTES

    # Enforce AIsa only
    bad = [(lbl, r.get("provider")) for lbl, r in routes.items() if r.get("provider") != AISA_PROVIDER_NAME]
    if bad:
        print(f"ERROR: all routes must use '{AISA_PROVIDER_NAME}'. Offending: {bad}", file=sys.stderr)
        return 2

    print("\n=== Per-label routing (all via AIsa) ===")
    for lbl in LABELS:
        if after.get(lbl, 0) == 0:
            continue
        prov, mdl = resolve_route(routes, lbl)
        print(f"  {lbl:22s} -> {mdl}")

    if args.dry_run:
        print("\n[dry-run] no API calls made.")
        return 0

    done_keys = load_progress(args.progress)
    if done_keys:
        print(f"[resume] {len(done_keys)} rows already done")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_exists = args.output.exists() and args.output.stat().st_size > 0
    out_f = args.output.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
    if not file_exists:
        writer.writeheader()
        out_f.flush()

    # Initialize client pool with multiple keys
    client_pool = ClientPool(providers)

    succ = 0
    fail = 0
    skipped = 0
    try:
        for i, item in enumerate(plan, 1):
            key = row_key(item["prompt"], item["answer"], item["variation_idx"])
            if key in done_keys:
                skipped += 1
                continue
            try:
                prov_name, model_id = resolve_route(routes, item["label"])
                cot = call_llm(
                    client_pool, prov_name, model_id,
                    item["prompt"], item["answer"], item["variation_idx"],
                    max_attempts=args.max_attempts,
                )
            except Exception as e:
                fail += 1
                print(f"[{i}/{len(plan)}][{item['label']}] FAIL: {type(e).__name__}: {e}", file=sys.stderr)
                continue

            writer.writerow({
                "prompt": item["prompt"],
                "answer": item["answer"],
                "cot": cot,
                "label": item["label"],
            })
            out_f.flush()
            done_keys.add(key)
            save_progress(args.progress, done_keys)
            succ += 1
            print(f"[{i}/{len(plan)}][{item['label']}] OK (var={item['variation_idx']})")
    finally:
        out_f.close()

    print(f"\nDONE. success={succ} fail={fail} skipped={skipped} -> {args.output}")
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())