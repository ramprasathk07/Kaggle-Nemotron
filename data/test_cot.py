#!/usr/bin/env python3
"""
Test CoT generation for one question per label.
Reuses classification and LLM logic from the main pipeline.
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
from openai import APIError, APIStatusError, BadRequestError, OpenAI, RateLimitError


# ---------------------------------------------------------------------------
# 1. Classification (same as main script)
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
# 2. LLM client with multiple API keys (same as main script)
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

_OUTPUT_FORMAT_RULE = (
    " OUTPUT FORMAT - the 'cot' field MUST decode to:\n"
    "<think>\n<your detailed step-by-step reasoning>\n</think>\n"
    "\\boxed{<VERIFIED ANSWER, exact characters>}\n"
    "Rules: (1) start with <think> on its own line; (2) include </think> on its own line after reasoning; "
    "(3) the next line MUST be \\boxed{...} with the answer; (4) inside JSON escape backslash as \\\\boxed; "
    "(5) reply ONLY with {\"cot\": \"<think>\\n...\\n</think>\\n\\\\boxed{<answer>}\"}."
)

SYSTEM_PROMPT_VARIANTS = [
    "You are an expert puzzle solver. Given a PUZZLE and its VERIFIED ANSWER, produce a detailed, step-by-step chain of reasoning (aim for 500–2000 tokens) that derives the verified answer from the evidence. Use only facts in the puzzle. Never mention the answer was given. End with the exact answer in \\boxed{}." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
    "You are a methodical analyst. Write a thorough step-by-step derivation (500–2000 tokens). First identify the rule that fits every example, then apply it to the query. Show calculations. Conclude with the exact answer." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
    "You are a patient tutor. Walk through the puzzle in 8–12 logical steps, referencing examples by value. Derive the answer exactly, without mentioning it was given." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
    "You are a careful reasoner. Propose a candidate rule, verify against examples, apply to query, compute result. Ensure final computed value matches verified answer exactly." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
    "You are a terse but thorough expert. Derive the answer in 8–12 clear steps: identify rule, apply, conclude with exact answer. No meta‑commentary." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
]

# Label-specific system prompts (customized for certain families)
_ENCDEC_CORE = (
    " You are decoding a substitution / shift / keyword cipher. Your reasoning MUST contain, "
    "in this exact order, with concrete values shown:\n"
    "  STEP 1 - List every (encrypted_word -> decrypted_word) example given in the puzzle.\n"
    "  STEP 2 - Align them letter-by-letter and extract every (cipher_letter -> plain_letter) pair. "
    "Write the pairs out, e.g. 'h->k, i->i, e->n, a->g'. Do this for ALL example pairs.\n"
    "  STEP 3 - Build the consolidated mapping table sorted by cipher letter. "
    "If the same cipher letter ever maps to two different plain letters across examples, "
    "say so explicitly and resolve it.\n"
    "  STEP 4 - Identify the cipher type (Caesar shift by N / atbash / keyword substitution / "
    "monoalphabetic / etc.) and verify it explains every pair in the table. State the shift number "
    "or keyword if applicable.\n"
    "  STEP 5 - Apply the mapping to the target ciphertext one letter at a time. Write each "
    "substitution: 'h -> ?, i -> ?, e -> ?, a -> ?, ...'. Preserve spaces and punctuation as-is.\n"
    "  STEP 6 - Read off the decoded plaintext and confirm it is a sensible English phrase.\n"
    "Do NOT skip any step. Do NOT write generalities like 'by analyzing the examples'. Show the "
    "actual letters, the actual mapping, and the actual letter-by-letter substitution."
)
LABEL_SYSTEM_PROMPTS: dict[str, list[str]] = {
    "encoding-decoding": [
        "You are a cryptanalyst deriving a cipher from worked examples." + _ENCDEC_CORE + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
        "You are a methodical codebreaker. Reconstruct the cipher rigorously from the examples before decoding the query." + _ENCDEC_CORE + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
        "You are a patient cipher tutor. Show every alignment, every (cipher->plain) pair, the consolidated table, and the per-letter decoding of the target." + _ENCDEC_CORE + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
    ],
}

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

    def get_client(self) -> OpenAI:
        c = self.clients[self.idx % self.n]
        self.idx = (self.idx + 1) % self.n
        return c

    def rotate(self):
        self.idx = (self.idx + 1) % self.n


class ClientPool:
    """Manages per‑provider client pools. For AIsa, collects all AISA_API_KEY_*."""
    def __init__(self, providers: dict[str, dict[str, str]]):
        self._pools: dict[str, MultiKeyClient] = {}
        for name, spec in providers.items():
            if name == AISA_PROVIDER_NAME:
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
                    single_key = os.environ.get(spec["api_key_env"])
                    if single_key:
                        keys = [single_key]
                if not keys:
                    raise RuntimeError(f"No AISA API keys found. Set AISA_API_KEY_1, _2, ... or {spec['api_key_env']}")
                self._pools[name] = MultiKeyClient(spec["base_url"], keys)
            else:
                pass  # only AIsa supported

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

def resolve_route(routes: dict[str, dict[str, str]], label: str) -> tuple[str, str]:
    r = routes.get(label) or routes.get("_default")
    if not r or "provider" not in r or "model" not in r:
        raise KeyError(f"No route for label '{label}' and no usable _default route.")
    return r["provider"], r["model"]

_NO_SAMPLING_PARAMS_MODELS: set[str] = set()

# Models whose names match these substrings are known to reject `temperature`
# and/or `seed` (reasoning / thinking / o-series style). We strip those params
# from the very first request to avoid burning an attempt on a guaranteed 400.
_NO_SAMPLING_NAME_PATTERNS = (
    "thinking",
    "claude-opus-4-7",
    "deepseek-r1",
    "o1-", "o3-", "o4-",
    "gpt-5",
)

def _model_rejects_sampling(model: str) -> bool:
    if model in _NO_SAMPLING_PARAMS_MODELS:
        return True
    m = model.lower()
    return any(p in m for p in _NO_SAMPLING_NAME_PATTERNS)

def _is_bad_request(exc: Exception) -> bool:
    return isinstance(exc, BadRequestError) or getattr(exc, "status_code", None) == 400

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
    label: str = "",
    max_attempts: int = 6,
) -> str:
    variants = LABEL_SYSTEM_PROMPTS.get(label) or SYSTEM_PROMPT_VARIANTS
    sys_prompt = variants[variation_idx % len(variants)]
    user_msg = USER_TEMPLATE.format(prompt=str(prompt)[:6000], answer=str(answer)[:1000])

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        client = client_pool.get(provider_name)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 8192,
        }
        sent_sampling = not _model_rejects_sampling(model)
        if sent_sampling:
            kwargs["temperature"] = 0.5 if variation_idx > 0 else 0.2
            kwargs["seed"] = 1000 + variation_idx * 13 + attempt
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            last_exc = e
            # Any 400 while we were sending sampling params: assume the model
            # rejects them (covers explicit "deprecated" messages AND opaque
            # "upstream error" ones), disable for this model, and retry.
            if _is_bad_request(e) and sent_sampling and model not in _NO_SAMPLING_PARAMS_MODELS:
                _NO_SAMPLING_PARAMS_MODELS.add(model)
                print(f"[info] 400 from '{model}'; disabling temperature/seed and retrying", file=sys.stderr)
                continue
            if _is_rate_limit(e):
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
# 3. Test main: one example per label
# ---------------------------------------------------------------------------

def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Test CoT for one example per label.")
    ap.add_argument("--input", type=Path, default=Path("data/src/train.csv"))
    ap.add_argument("--output-json", type=Path, default=Path("test_cot_output.json"))
    ap.add_argument("--routing-config", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    # Load data
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

    # Show label distribution
    label_counts = Counter(df["label"])
    print("\n=== Label counts in input ===")
    for lbl in LABELS:
        print(f"  {lbl:22s} {label_counts.get(lbl, 0):>6d}")

    # Gather one random example per label
    rng = random.Random(args.seed)
    examples = {}
    for lbl in LABELS:
        subset = df[df["label"] == lbl]
        if len(subset) == 0:
            print(f"Warning: no rows for label '{lbl}'")
            continue
        # row = subset.sample(n=1, random_state=rng).iloc[0]
        row = subset.sample(n=1, random_state=args.seed).iloc[0]
        examples[lbl] = {
            "prompt": row["prompt"],
            "answer": row["answer"],
        }

    # Setup routing and client pool
    if args.routing_config and args.routing_config.exists():
        providers, routes = load_routing_config(args.routing_config)
    else:
        providers = DEFAULT_PROVIDERS
        routes = DEFAULT_ROUTES

    # Enforce AIsa only
    bad = [(lbl, r.get("provider")) for lbl, r in routes.items() if r.get("provider") != AISA_PROVIDER_NAME]
    if bad:
        print(f"ERROR: all routes must use '{AISA_PROVIDER_NAME}'. Offending: {bad}", file=sys.stderr)
        return 2

    client_pool = ClientPool(providers)

    results = {}
    for lbl, data in examples.items():
        print(f"\n=== Generating CoT for label: {lbl} ===")
        print(f"Prompt (first 200 chars): {data['prompt'][:200]}...")
        print(f"Answer: {data['answer']}")
        try:
            prov_name, model_id = resolve_route(routes, lbl)
            cot = call_llm(
                client_pool,
                prov_name,
                model_id,
                data["prompt"],
                data["answer"],
                variation_idx=0,   # use first system prompt variant
                label=lbl,
                max_attempts=3,
            )
            results[lbl] = {
                "prompt": data["prompt"],
                "answer": data["answer"],
                "cot": cot,
                "model": model_id,
            }
            print("\n--- Generated CoT ---")
            print(cot)
            print("--- End of CoT ---")
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)
            results[lbl] = {"error": str(e), "prompt": data["prompt"], "answer": data["answer"]}

    # Save results
    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved results to {args.output_json}")

    # Summary
    print("\n=== SUMMARY ===")
    for lbl, res in results.items():
        if "error" in res:
            print(f"{lbl:22s} FAILED: {res['error'][:80]}")
        else:
            print(f"{lbl:22s} OK (model {res.get('model', '?')})")

    return 0

if __name__ == "__main__":
    sys.exit(main())