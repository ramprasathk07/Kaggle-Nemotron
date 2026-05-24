# #!/usr/bin/env python3
# """
# Lenient pipeline: parallel CoT generation with tolerant answer matching.
# Saves API credits by accepting small variations (e.g., missing leading '}').
# """

# from __future__ import annotations

# import argparse
# import csv
# import hashlib
# import json
# import os
# import random
# import re
# import sys
# import time
# import threading
# from collections import Counter, deque
# from pathlib import Path
# from typing import Any, List, Optional
# from concurrent.futures import ThreadPoolExecutor, as_completed

# import pandas as pd
# from dotenv import load_dotenv
# from openai import APIError, APIStatusError, BadRequestError, OpenAI, RateLimitError


# # ---------------------------------------------------------------------------
# # 1. Classification (unchanged)
# # ---------------------------------------------------------------------------

# LABELS = [
#     "numeric-rule",
#     "binary-transform",
#     "roman-numeral",
#     "mapping-symbolic",
#     "sequence-rule",
#     "string-transform",
#     "logic-short-answer",
#     "encoding-decoding",
#     "table-like-rule",
#     "mixed-template",
# ]

# CLASSIFY_RULES: list[tuple[str, str]] = [
#     ("binary-transform", r"(?:\bbit\s*m[ai]n[ip]*u?l?ation\b|\bbit\s*wis?e\b|\bbit\s*shift\b|\bbit\s*rotat\w*|\bxor\b|\bxnor\b|\b8[-\s]?bit\b|\bbinary\s+numbers?\b|\bmajority\s+function\b|\bchoice\s+function\b)"),
#     ("roman-numeral", r"(?:\broma+n\s+numeral\w*|\broma+n\b(?=[^.]{0,80}\bnumber\b)|\bnumeral\s+system\b|\bwonderland\s+numeral\b)"),
#     ("encoding-decoding", r"(?:\b(?:en|de)\s*c?rypt\w*|\b(?:en|de)\s*cod\w*|\bcaes[ae]r\b|\bciph[ae]r\w*|\bsubstitut\w*|\bsecret\s+encryption\b)"),
#     ("mapping-symbolic", r"(?:transformation\s+rules?[^.\n]{0,80}equations?|`[^`\n]*[!@#$%^&*()\[\]{}<>'\"\\][^`\n]*`\s*=\s*`?|\bsymbolic\s+(?:rule|mapping|transform)\w*)"),
#     ("numeric-rule", r"(?:\bgravitation\w*|\bunit\s+conversion\b|\bd\s*=\s*0?\.5\s*\*\s*g\s*\*\s*t\s*\^?\s*2|\bmeasurement\w*\s+(?:is\s+)?(?:secretly\s+)?(?:applied|converted)|\bm\s+becomes\s+\d|\bsecret\s+formula\b|\bsecretly\s+(?:changed|converted))"),
#     ("sequence-rule", r"(?:\bsequence\b|\bnext\s+(?:term|number|element|value)\b|\bcontinue\s+the\s+(?:pattern|sequence)\b|\barithmetic\s+progression\b|\bgeometric\s+progression\b|\bnth\s+term\b)"),
#     ("string-transform", r"(?:\banagram\w*|\bpalindrom\w*|\breverse\s+the\s+(?:string|text|word)|\bstring\s+(?:transform|manipulation)\w*|\bcase\s+(?:swap|flip)\b)"),
#     ("logic-short-answer", r"(?:\blogic\s+puzzle\w*|\briddle\w*|\btrue\s+or\s+false\b|\byes\s+or\s+no\b|\bknights?\s+and\s+knaves\b)"),
#     ("table-like-rule", r"(?:\btable\b|\bgrid\b|\bmatrix\b|\brows?\s+and\s+columns?\b|\bcell\s+at\s+row\b)"),
# ]

# COMPILED_RULES = [(label, re.compile(pat, re.IGNORECASE | re.DOTALL)) for label, pat in CLASSIFY_RULES]

# def classify(prompt: str) -> str:
#     p = prompt or ""
#     for label, rx in COMPILED_RULES:
#         if rx.search(p):
#             return label
#     return "mixed-template"


# # ---------------------------------------------------------------------------
# # 2. LLM client with multiple API keys (thread‑safe)
# # ---------------------------------------------------------------------------

# DEFAULT_BASE_URL = "https://api.aisa.one/v1"
# DEFAULT_MODEL = "deepseek-v3.2"

# AISA_PROVIDER_NAME = "aisa"
# DEFAULT_PROVIDERS: dict[str, dict[str, str]] = {
#     AISA_PROVIDER_NAME: {"base_url": "https://api.aisa.one/v1", "api_key_env": "AISA_API_KEY"},
# }

# DEFAULT_ROUTES: dict[str, dict[str, str]] = {
#     "binary-transform":   {"provider": "aisa", "model": "claude-opus-4-7"},
#     "mapping-symbolic":   {"provider": "aisa", "model": "gpt-5.4"},
#     "logic-short-answer": {"provider": "aisa", "model": "deepseek-r1"},
#     "encoding-decoding":  {"provider": "aisa", "model": "gemini-3.1-pro-preview"},
#     "numeric-rule":       {"provider": "aisa", "model": "claude-sonnet-4-6-thinking"},
#     "sequence-rule":      {"provider": "aisa", "model": "kimi-k2-thinking"},
#     "roman-numeral":      {"provider": "aisa", "model": "glm-5"},
#     "string-transform":   {"provider": "aisa", "model": "gemini-2.5-flash"},
#     "table-like-rule":    {"provider": "aisa", "model": "qwen3.6-plus"},
#     "mixed-template":     {"provider": "aisa", "model": "deepseek-v3.2"},
#     "_default":           {"provider": "aisa", "model": DEFAULT_MODEL},
# }

# _NO_EMOJI_RULE = (
#     " Use plain ASCII / standard Latin characters only - absolutely NO emojis, "
#     "pictographs, decorative Unicode symbols, or icon characters anywhere."
# )

# _OUTPUT_FORMAT_RULE = (
#     " OUTPUT FORMAT: Put your reasoning between <think> and </think> tags, "
#     "then on a new line put the final answer inside \\boxed{...}. "
#     "Example:\n<think>\nStep-by-step reasoning...\n</think>\n\\boxed{answer}\n"
#     "Do NOT add extra text outside this structure."
# )

# SYSTEM_PROMPT_VARIANTS = [
#     "You are an expert puzzle solver. Given a PUZZLE and its VERIFIED ANSWER, produce a detailed, step-by-step chain of reasoning that derives the verified answer from the evidence. Use only facts in the puzzle. Never mention the answer was given. End with the exact answer in \\boxed{}." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
#     "You are a methodical analyst. Write a thorough step-by-step derivation. First identify the rule that fits every example, then apply it to the query. Show calculations. Conclude with the exact answer in \\boxed{}." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
#     "You are a patient tutor. Walk through the puzzle in logical steps, referencing examples by value. Derive the answer exactly, without mentioning it was given. Put final answer in \\boxed{}." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
#     "You are a careful reasoner. Propose a candidate rule, verify against examples, apply to query, compute result. Ensure final computed value matches verified answer exactly. End with \\boxed{answer}." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
#     "You are a terse but thorough expert. Derive the answer in clear steps: identify rule, apply, conclude with exact answer inside \\boxed{}. No meta‑commentary." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
# ]

# _ENCDEC_CORE = (
#     " You are decoding a substitution / shift / keyword cipher. Your reasoning MUST contain concrete values:\n"
#     "  STEP 1 - List every (encrypted_word -> decrypted_word) example.\n"
#     "  STEP 2 - Align letter-by-letter and extract each (cipher_letter -> plain_letter) pair.\n"
#     "  STEP 3 - Build the consolidated mapping table.\n"
#     "  STEP 4 - Identify the cipher type and verify it explains all pairs.\n"
#     "  STEP 5 - Apply the mapping to the target ciphertext letter by letter.\n"
#     "  STEP 6 - State the decoded plaintext inside \\boxed{...}.\n"
#     "Do NOT skip steps. Show actual letters and mapping."
# )
# LABEL_SYSTEM_PROMPTS: dict[str, list[str]] = {
#     "encoding-decoding": [
#         "You are a cryptanalyst deriving a cipher from worked examples." + _ENCDEC_CORE + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
#         "You are a methodical codebreaker. Reconstruct the cipher rigorously." + _ENCDEC_CORE + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
#         "You are a patient cipher tutor. Show every alignment and mapping." + _ENCDEC_CORE + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
#     ],
# }

# USER_TEMPLATE = (
#     "PUZZLE:\n{prompt}\n\nVERIFIED ANSWER: {answer}\n\n"
#     "Produce a chain of thought inside <think>...</think> followed by \\boxed{{{answer}}}.\n"
#     "Do NOT output any extra text outside this structure."
# )

# _EMOJI_RX = re.compile(
#     "["
#     "\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
#     "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
#     "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
#     "\U0001F1E6-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF"
#     "\U0001F000-\U0001F0FF\U0000FE0E\U0000FE0F]"
# )

# def _contains_emoji(text: str) -> bool:
#     return bool(_EMOJI_RX.search(text or ""))


# class MultiKeyClient:
#     def __init__(self, base_url: str, api_keys: List[str]):
#         self.base_url = base_url
#         self.clients = [OpenAI(base_url=base_url, api_key=k) for k in api_keys]
#         self.n = len(self.clients)
#         self.idx = 0
#         self._lock = threading.Lock()

#     def get_client(self) -> OpenAI:
#         with self._lock:
#             c = self.clients[self.idx % self.n]
#             self.idx = (self.idx + 1) % self.n
#             return c

#     def rotate(self):
#         with self._lock:
#             self.idx = (self.idx + 1) % self.n


# class ClientPool:
#     def __init__(self, providers: dict[str, dict[str, str]]):
#         self._pools: dict[str, MultiKeyClient] = {}
#         for name, spec in providers.items():
#             if name == AISA_PROVIDER_NAME:
#                 keys = []
#                 i = 1
#                 while True:
#                     key = os.environ.get(f"AISA_API_KEY_{i}")
#                     if key:
#                         keys.append(key)
#                         i += 1
#                     else:
#                         break
#                 if not keys:
#                     single_key = os.environ.get(spec["api_key_env"])
#                     if single_key:
#                         keys = [single_key]
#                 if not keys:
#                     raise RuntimeError("No AISA API keys found.")
#                 self._pools[name] = MultiKeyClient(spec["base_url"], keys)
#             # Only AIsa supported

#     def get(self, provider_name: str) -> OpenAI:
#         if provider_name not in self._pools:
#             raise KeyError(f"Provider '{provider_name}' not initialized.")
#         return self._pools[provider_name].get_client()

#     def rotate_provider(self, provider_name: str):
#         if provider_name in self._pools:
#             self._pools[provider_name].rotate()


# def load_routing_config(path: Path):
#     cfg = json.loads(path.read_text(encoding="utf-8"))
#     if not isinstance(cfg, dict):
#         raise ValueError(f"Routing config {path} must be a JSON object.")
#     providers = {**DEFAULT_PROVIDERS, **(cfg.get("providers") or {})}
#     routes = {**DEFAULT_ROUTES, **(cfg.get("routes") or {})}
#     return providers, routes

# def write_sample_routing_config(path: Path):
#     sample = {"providers": DEFAULT_PROVIDERS, "routes": DEFAULT_ROUTES}
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
#     print(f"Sample routing config written to {path}")

# def resolve_route(routes: dict[str, dict[str, str]], label: str) -> tuple[str, str]:
#     r = routes.get(label) or routes.get("_default")
#     if not r or "provider" not in r or "model" not in r:
#         raise KeyError(f"No route for label '{label}' and no usable _default route.")
#     return r["provider"], r["model"]

# _NO_SAMPLING_PARAMS_MODELS: set[str] = set()
# _NO_SAMPLING_PARAMS_LOCK = threading.Lock()
# _NO_SAMPLING_NAME_PATTERNS = ("thinking", "claude-opus-4-7", "deepseek-r1", "o1-", "o3-", "o4-", "gpt-5")

# def _model_rejects_sampling(model: str) -> bool:
#     with _NO_SAMPLING_PARAMS_LOCK:
#         if model in _NO_SAMPLING_PARAMS_MODELS:
#             return True
#     m = model.lower()
#     return any(p in m for p in _NO_SAMPLING_NAME_PATTERNS)

# def _is_bad_request(exc: Exception) -> bool:
#     return isinstance(exc, BadRequestError) or getattr(exc, "status_code", None) == 400

# def _is_rate_limit(exc: Exception) -> bool:
#     if isinstance(exc, RateLimitError):
#         return True
#     if getattr(exc, "status_code", None) == 429:
#         return True
#     return "429" in str(exc) or "rate limit" in str(exc).lower()

# def _is_transient(exc: Exception) -> bool:
#     if _is_rate_limit(exc):
#         return True
#     if isinstance(exc, (APIError, APIStatusError)):
#         s = getattr(exc, "status_code", None)
#         if s is None or s >= 500 or s == 408:
#             return True
#     msg = str(exc).lower()
#     return any(x in msg for x in ("timeout", "timed out", "connection", "503", "502", "504"))


# # ---------------------------------------------------------------------------
# # Rate limit tracker
# # ---------------------------------------------------------------------------
# class RateLimitTracker:
#     def __init__(self, window_sec: float = 10.0, max_events: int = 5, cooldown_sec: float = 30.0):
#         self.window = window_sec
#         self.max_events = max_events
#         self.cooldown = cooldown_sec
#         self._events: deque[float] = deque()
#         self._lock = threading.Lock()
#         self._global_cooldown_until: float = 0.0

#     def is_in_cooldown(self) -> float:
#         with self._lock:
#             if time.time() < self._global_cooldown_until:
#                 return self._global_cooldown_until - time.time()
#             return 0.0

#     def record_rate_limit(self) -> None:
#         with self._lock:
#             now = time.time()
#             if now < self._global_cooldown_until:
#                 return
#             self._events.append(now)
#             while self._events and self._events[0] < now - self.window:
#                 self._events.popleft()
#             if len(self._events) >= self.max_events:
#                 self._global_cooldown_until = now + self.cooldown
#                 self._events.clear()
#                 print(f"[rate-limit] global cooldown for {self.cooldown}s", file=sys.stderr)

# _rate_tracker = RateLimitTracker()


# # ---------------------------------------------------------------------------
# # LENIENT CoT parser (the fix)
# # ---------------------------------------------------------------------------
# def _extract_boxed_answer(text: str) -> Optional[str]:
#     """Extract content from \boxed{...} allowing nested braces and escaped backslashes."""
#     pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
#     match = re.search(pattern, text)
#     if match:
#         return match.group(1).strip()
#     pattern2 = r"\\\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
#     match2 = re.search(pattern2, text)
#     if match2:
#         return match2.group(1).strip()
#     return None

# def _normalize_answer_loose(s: str) -> str:
#     """
#     Remove non-alphanumeric characters from beginning and end.
#     This allows '}38' to match '38' and vice versa.
#     """
#     s = s.strip()
#     # Strip leading/trailing non-alphanumeric (keep internal symbols if needed)
#     s = re.sub(r'^[^a-zA-Z0-9]+', '', s)
#     s = re.sub(r'[^a-zA-Z0-9]+$', '', s)
#     return s

# def _parse_cot(raw: str, expected_answer: str) -> Optional[str]:
#     """
#     Extract CoT with lenient answer matching.
#     Requires <think>...</think> and \boxed{}.
#     The answer inside \boxed{} must match expected_answer after loose normalization.
#     """
#     if not raw:
#         return None

#     s = raw.strip()
#     if s.startswith("```"):
#         s = re.sub(r"^```(?:json)?\s*\n?", "", s)
#         s = re.sub(r"\n?```$", "", s)

#     # Extract boxed answer
#     boxed = _extract_boxed_answer(s)
#     if boxed is None:
#         return None

#     # Lenient comparison
#     if _normalize_answer_loose(boxed) != _normalize_answer_loose(expected_answer):
#         return None

#     # Extract think block
#     think_start = s.find("<think>")
#     think_end = s.find("</think>", think_start) if think_start != -1 else -1
#     if think_start == -1 or think_end == -1:
#         return None

#     # Find the position of the boxed expression to include it in the output
#     boxed_pos = s.find("\\boxed{", think_end)
#     if boxed_pos == -1:
#         boxed_pos = s.find("\\\\boxed{", think_end)
#     if boxed_pos == -1:
#         return None

#     # Determine end position (after the closing brace)
#     depth = 0
#     i = boxed_pos + len("\\boxed{")
#     end_pos = len(s)
#     while i < len(s):
#         if s[i] == '{':
#             depth += 1
#         elif s[i] == '}':
#             if depth == 0:
#                 end_pos = i + 1
#                 break
#             depth -= 1
#         i += 1

#     cot = s[think_start:end_pos].strip()
#     if not cot:
#         return None
#     if _contains_emoji(cot):
#         return None

#     return cot


# # ---------------------------------------------------------------------------
# # LLM call with fewer retries (default 3)
# # ---------------------------------------------------------------------------
# def call_llm(
#     client_pool: ClientPool,
#     provider_name: str,
#     model: str,
#     prompt: str,
#     answer: str,
#     variation_idx: int,
#     label: str = "",
#     max_attempts: int = 3,
# ) -> str:
#     variants = LABEL_SYSTEM_PROMPTS.get(label) or SYSTEM_PROMPT_VARIANTS
#     sys_prompt = variants[variation_idx % len(variants)]
#     user_msg = USER_TEMPLATE.format(prompt=str(prompt)[:6000], answer=str(answer)[:1000])

#     for attempt in range(1, max_attempts + 1):
#         cooldown = _rate_tracker.is_in_cooldown()
#         if cooldown > 0:
#             print(f"[rate-limit] global cooldown for {cooldown:.1f}s", file=sys.stderr)
#             time.sleep(cooldown)
#             continue

#         client = client_pool.get(provider_name)
#         kwargs: dict[str, Any] = {
#             "model": model,
#             "messages": [
#                 {"role": "system", "content": sys_prompt},
#                 {"role": "user", "content": user_msg},
#             ],
#             "max_tokens": 8192,
#         }
#         sent_sampling = not _model_rejects_sampling(model)
#         if sent_sampling:
#             kwargs["temperature"] = 0.5 if variation_idx > 0 else 0.2
#             kwargs["seed"] = 1000 + variation_idx * 13 + attempt

#         try:
#             resp = client.chat.completions.create(**kwargs)
#         except Exception as e:
#             if _is_bad_request(e) and sent_sampling and model not in _NO_SAMPLING_PARAMS_MODELS:
#                 with _NO_SAMPLING_PARAMS_LOCK:
#                     _NO_SAMPLING_PARAMS_MODELS.add(model)
#                 print(f"[info] 400 from '{model}'; disabling temperature/seed and retrying", file=sys.stderr)
#                 continue
#             if _is_rate_limit(e):
#                 _rate_tracker.record_rate_limit()
#                 client_pool.rotate_provider(provider_name)
#                 wait = min(2.0 ** (attempt - 1) + random.uniform(0, 1), 60.0)
#                 print(f"[rate-limit] key rotated, sleeping {wait:.1f}s (attempt {attempt})", file=sys.stderr)
#                 time.sleep(wait)
#                 continue
#             if not _is_transient(e):
#                 raise
#             time.sleep(min(2.0 ** (attempt - 1), 60.0) + random.uniform(0, 1))
#             continue

#         text = resp.choices[0].message.content or ""
#         cot = _parse_cot(text, str(answer))
#         if cot:
#             return cot
#         print(f"[warning] attempt {attempt} invalid output (preview: {text[:200]})", file=sys.stderr)
#         time.sleep(1.0 + random.uniform(0, 0.5))

#     raise RuntimeError(f"exhausted {max_attempts} attempts")


# # ---------------------------------------------------------------------------
# # 3. Balancing (unchanged)
# # ---------------------------------------------------------------------------
# def build_plan(df: pd.DataFrame, target: int, rng: random.Random) -> list[dict[str, Any]]:
#     per_label: dict[str, list[dict[str, Any]]] = {}
#     for lbl in LABELS:
#         rows = df[df["label"] == lbl].to_dict("records")
#         if not rows:
#             continue
#         bucket: list[dict[str, Any]] = []
#         if len(rows) >= target:
#             picked = rng.sample(rows, target)
#             for r in picked:
#                 bucket.append({"prompt": r["prompt"], "answer": r["answer"], "label": lbl, "variation_idx": 0})
#         else:
#             for r in rows:
#                 bucket.append({"prompt": r["prompt"], "answer": r["answer"], "label": lbl, "variation_idx": 0})
#             need = target - len(rows)
#             v, i = 1, 0
#             while need > 0:
#                 r = rows[i % len(rows)]
#                 bucket.append({"prompt": r["prompt"], "answer": r["answer"], "label": lbl, "variation_idx": v})
#                 i += 1
#                 if i % len(rows) == 0:
#                     v += 1
#                 need -= 1
#         per_label[lbl] = bucket

#     plan: list[dict[str, Any]] = []
#     cursors = {lbl: 0 for lbl in per_label}
#     while True:
#         added = False
#         for lbl in LABELS:
#             bucket = per_label.get(lbl)
#             if not bucket:
#                 continue
#             idx = cursors[lbl]
#             if idx < len(bucket):
#                 plan.append(bucket[idx])
#                 cursors[lbl] = idx + 1
#                 added = True
#         if not added:
#             break
#     return plan


# # ---------------------------------------------------------------------------
# # 4. Output / resume
# # ---------------------------------------------------------------------------
# CSV_FIELDS = ["prompt", "answer", "cot", "label"]

# def row_key(prompt: str, answer: str, var: int) -> str:
#     h = hashlib.md5(f"{prompt}|||{answer}".encode("utf-8")).hexdigest()[:16]
#     return f"{h}#{var}"

# def load_progress(path: Path) -> set[str]:
#     if not path.exists():
#         return set()
#     try:
#         return set(json.loads(path.read_text(encoding="utf-8")))
#     except Exception:
#         return set()

# def save_progress(path: Path, keys: set[str]) -> None:
#     tmp = path.with_suffix(path.suffix + ".tmp")
#     tmp.write_text(json.dumps(sorted(keys)), encoding="utf-8")
#     tmp.replace(path)


# # ---------------------------------------------------------------------------
# # 5. Main
# # ---------------------------------------------------------------------------
# def parse_args() -> argparse.Namespace:
#     ap = argparse.ArgumentParser(description="Lenient parallel CoT generation to save credits.")
#     ap.add_argument("--input", type=Path, default=Path("data/src/train.csv"))
#     ap.add_argument("--output", type=Path, default=Path("final_output.csv"))
#     ap.add_argument("--progress", type=Path, default=None)
#     ap.add_argument("--routing-config", type=Path, default=None)
#     ap.add_argument("--write-sample-routing", type=Path, default=None)
#     ap.add_argument("--only-labels", default=None)
#     ap.add_argument("--target", type=int, default=0)
#     ap.add_argument("--max-per-label", type=int, default=0)
#     ap.add_argument("--max-attempts", type=int, default=3, help="Max attempts per row (default 3 to save credits)")
#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--dry-run", action="store_true")
#     ap.add_argument("--force-model", default=None)
#     ap.add_argument("--workers", type=int, default=0)
#     ap.add_argument("--rate-limit-cooldown", type=float, default=30.0)
#     ap.add_argument("--rate-limit-window", type=float, default=10.0)
#     ap.add_argument("--rate-limit-threshold", type=int, default=5)
#     return ap.parse_args()


# def main() -> int:
#     load_dotenv()
#     args = parse_args()

#     if args.write_sample_routing:
#         write_sample_routing_config(args.write_sample_routing)
#         return 0

#     if args.progress is None:
#         args.progress = args.output.with_suffix(args.output.suffix + ".progress.json")

#     if not args.input.exists():
#         print(f"ERROR: input not found: {args.input}", file=sys.stderr)
#         return 2

#     csv.field_size_limit(10 * 1024 * 1024)
#     try:
#         df = pd.read_csv(args.input)
#     except Exception as e:
#         print(f"ERROR: failed to read {args.input}: {e}", file=sys.stderr)
#         return 2

#     required = {"prompt", "answer"}
#     missing = required - set(df.columns)
#     if missing:
#         print(f"ERROR: missing columns {missing}", file=sys.stderr)
#         return 2

#     df = df.dropna(subset=["prompt", "answer"]).reset_index(drop=True)
#     df["prompt"] = df["prompt"].astype(str)
#     df["answer"] = df["answer"].astype(str)
#     df["label"] = df["prompt"].map(classify)

#     before = Counter(df["label"])
#     print("\n=== Label counts BEFORE balancing ===")
#     for lbl in LABELS:
#         print(f"  {lbl:22s} {before.get(lbl, 0):>6d}")
#     print(f"  {'TOTAL':22s} {sum(before.values()):>6d}")

#     nonzero = [c for c in before.values() if c > 0]
#     if not nonzero:
#         print("ERROR: no rows after classification.", file=sys.stderr)
#         return 2

#     target = args.target if args.target > 0 else min(nonzero)
#     if args.max_per_label > 0:
#         target = min(target, args.max_per_label)
#     print(f"\nTarget per label: {target}")

#     rng = random.Random(args.seed)
#     plan = build_plan(df, target, rng)

#     if args.only_labels:
#         wanted = {s.strip() for s in args.only_labels.split(",") if s.strip()}
#         plan = [p for p in plan if p["label"] in wanted]
#         print(f"[only-labels] filtered to {len(plan)} rows")

#     after = Counter(item["label"] for item in plan)
#     print("\n=== Label counts AFTER balancing (planned) ===")
#     for lbl in LABELS:
#         print(f"  {lbl:22s} {after.get(lbl, 0):>6d}")
#     print(f"  {'TOTAL':22s} {sum(after.values()):>6d}")

#     # Routing
#     if args.routing_config and args.routing_config.exists():
#         providers, routes = load_routing_config(args.routing_config)
#     else:
#         providers = DEFAULT_PROVIDERS
#         routes = DEFAULT_ROUTES

#     if args.force_model:
#         routes = {k: {**v, "model": args.force_model} for k, v in routes.items()}
#         print(f"\n[force-model] All labels overridden -> {args.force_model}")

#     bad = [(lbl, r.get("provider")) for lbl, r in routes.items() if r.get("provider") != AISA_PROVIDER_NAME]
#     if bad:
#         print(f"ERROR: all routes must use '{AISA_PROVIDER_NAME}'. Offending: {bad}", file=sys.stderr)
#         return 2

#     print("\n=== Per-label routing (all via AIsa) ===")
#     for lbl in LABELS:
#         if after.get(lbl, 0) == 0:
#             continue
#         prov, mdl = resolve_route(routes, lbl)
#         print(f"  {lbl:22s} -> {mdl}")

#     if args.dry_run:
#         print("\n[dry-run] no API calls made.")
#         return 0

#     # Configure rate tracker
#     global _rate_tracker
#     _rate_tracker = RateLimitTracker(
#         window_sec=args.rate_limit_window,
#         max_events=args.rate_limit_threshold,
#         cooldown_sec=args.rate_limit_cooldown
#     )

#     # Count API keys
#     key_count = 0
#     i = 1
#     while os.environ.get(f"AISA_API_KEY_{i}"):
#         key_count += 1
#         i += 1
#     if key_count == 0 and os.environ.get("AISA_API_KEY"):
#         key_count = 1
#     if key_count == 0:
#         print("ERROR: No AISA API keys found.", file=sys.stderr)
#         return 1

#     workers = args.workers if args.workers > 0 else key_count
#     print(f"\n[parallel] Using {workers} workers with {key_count} API keys.")
#     print(f"[info] max_attempts per row = {args.max_attempts} (reduced to save credits)")

#     done_keys = load_progress(args.progress)
#     if done_keys:
#         print(f"[resume] {len(done_keys)} rows already done")

#     pending_items = [item for item in plan if row_key(item["prompt"], item["answer"], item["variation_idx"]) not in done_keys]
#     total_pending = len(pending_items)
#     if total_pending == 0:
#         print("No pending rows. Exiting.")
#         return 0

#     # Prepare output CSV
#     args.output.parent.mkdir(parents=True, exist_ok=True)
#     file_exists = args.output.exists() and args.output.stat().st_size > 0
#     out_f = open(args.output, "a", encoding="utf-8", newline="")
#     writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
#     if not file_exists:
#         writer.writeheader()
#         out_f.flush()

#     write_lock = threading.Lock()
#     progress_lock = threading.Lock()
#     client_pool = ClientPool(providers)

#     for item in pending_items:
#         item['key'] = row_key(item["prompt"], item["answer"], item["variation_idx"])

#     def process_one(item: dict[str, Any]) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
#         try:
#             prov_name, model_id = resolve_route(routes, item["label"])
#             cot = call_llm(
#                 client_pool, prov_name, model_id,
#                 item["prompt"], item["answer"], item["variation_idx"],
#                 label=item["label"],
#                 max_attempts=args.max_attempts,
#             )
#             row = {
#                 "prompt": item["prompt"],
#                 "answer": item["answer"],
#                 "cot": cot,
#                 "label": item["label"],
#             }
#             return (True, row, None)
#         except Exception as e:
#             return (False, None, f"{type(e).__name__}: {e}")

#     succ = 0
#     fail = 0
#     with ThreadPoolExecutor(max_workers=workers) as executor:
#         future_to_item = {executor.submit(process_one, item): item for item in pending_items}
#         for future in as_completed(future_to_item):
#             item = future_to_item[future]
#             try:
#                 success, row, error = future.result()
#             except Exception as exc:
#                 success = False
#                 error = str(exc)
#                 row = None

#             if success:
#                 with write_lock:
#                     writer.writerow(row)
#                     out_f.flush()
#                 with progress_lock:
#                     done_keys.add(item['key'])
#                     save_progress(args.progress, done_keys)
#                 succ += 1
#                 print(f"[{succ+fail}/{total_pending}][{item['label']}] OK (var={item['variation_idx']})")
#             else:
#                 fail += 1
#                 print(f"[{succ+fail}/{total_pending}][{item['label']}] FAIL: {error}", file=sys.stderr)

#     out_f.close()
#     print(f"\nDONE. success={succ} fail={fail} skipped={len(plan)-total_pending} -> {args.output}")
#     return 0 if fail == 0 else 1


# if __name__ == "__main__":
#     raise SystemExit(main())

#!/usr/bin/env python3
"""
Lenient pipeline: parallel CoT generation with tolerant answer matching.
Saves API credits by accepting small variations (e.g., missing <think> tags)
and avoiding expensive retries when the correct answer is already present.
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
import threading
from collections import Counter, deque
from pathlib import Path
from typing import Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import APIError, APIStatusError, BadRequestError, OpenAI, RateLimitError


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
# 2. LLM client with multiple API keys (thread‑safe)
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
    " OUTPUT FORMAT: Put your reasoning between <think> and </think> tags, "
    "then on a new line put the final answer inside \\boxed{...}. "
    "Example:\n<think>\nStep-by-step reasoning...\n</think>\n\\boxed{answer}\n"
    "Do NOT add extra text outside this structure."
)

SYSTEM_PROMPT_VARIANTS = [
    "You are an expert puzzle solver. Given a PUZZLE and its VERIFIED ANSWER, produce a detailed, step-by-step chain of reasoning that derives the verified answer from the evidence. Use only facts in the puzzle. Never mention the answer was given. End with the exact answer in \\boxed{}." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
    "You are a methodical analyst. Write a thorough step-by-step derivation. First identify the rule that fits every example, then apply it to the query. Show calculations. Conclude with the exact answer in \\boxed{}." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
    "You are a patient tutor. Walk through the puzzle in logical steps, referencing examples by value. Derive the answer exactly, without mentioning it was given. Put final answer in \\boxed{}." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
    "You are a careful reasoner. Propose a candidate rule, verify against examples, apply to query, compute result. Ensure final computed value matches verified answer exactly. End with \\boxed{answer}." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
    "You are a terse but thorough expert. Derive the answer in clear steps: identify rule, apply, conclude with exact answer inside \\boxed{}. No meta‑commentary." + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
]

_ENCDEC_CORE = (
    " You are decoding a substitution / shift / keyword cipher. Your reasoning MUST contain concrete values:\n"
    "  STEP 1 - List every (encrypted_word -> decrypted_word) example.\n"
    "  STEP 2 - Align letter-by-letter and extract each (cipher_letter -> plain_letter) pair.\n"
    "  STEP 3 - Build the consolidated mapping table.\n"
    "  STEP 4 - Identify the cipher type and verify it explains all pairs.\n"
    "  STEP 5 - Apply the mapping to the target ciphertext letter by letter.\n"
    "  STEP 6 - State the decoded plaintext inside \\boxed{...}.\n"
    "Do NOT skip steps. Show actual letters and mapping."
)
LABEL_SYSTEM_PROMPTS: dict[str, list[str]] = {
    "encoding-decoding": [
        "You are a cryptanalyst deriving a cipher from worked examples." + _ENCDEC_CORE + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
        "You are a methodical codebreaker. Reconstruct the cipher rigorously." + _ENCDEC_CORE + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
        "You are a patient cipher tutor. Show every alignment and mapping." + _ENCDEC_CORE + _NO_EMOJI_RULE + _OUTPUT_FORMAT_RULE,
    ],
}

USER_TEMPLATE = (
    "PUZZLE:\n{prompt}\n\nVERIFIED ANSWER: {answer}\n\n"
    "Produce a chain of thought inside <think>...</think> followed by \\boxed{{{answer}}}.\n"
    "Do NOT output any extra text outside this structure."
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
    def __init__(self, base_url: str, api_keys: List[str]):
        self.base_url = base_url
        self.clients = [OpenAI(base_url=base_url, api_key=k) for k in api_keys]
        self.n = len(self.clients)
        self.idx = 0
        self._lock = threading.Lock()

    def get_client(self) -> OpenAI:
        with self._lock:
            c = self.clients[self.idx % self.n]
            self.idx = (self.idx + 1) % self.n
            return c

    def rotate(self):
        with self._lock:
            self.idx = (self.idx + 1) % self.n


class ClientPool:
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
                    raise RuntimeError("No AISA API keys found.")
                self._pools[name] = MultiKeyClient(spec["base_url"], keys)
            # Only AIsa supported

    def get(self, provider_name: str) -> OpenAI:
        if provider_name not in self._pools:
            raise KeyError(f"Provider '{provider_name}' not initialized.")
        return self._pools[provider_name].get_client()

    def rotate_provider(self, provider_name: str):
        if provider_name in self._pools:
            self._pools[provider_name].rotate()


def load_routing_config(path: Path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Routing config {path} must be a JSON object.")
    providers = {**DEFAULT_PROVIDERS, **(cfg.get("providers") or {})}
    routes = {**DEFAULT_ROUTES, **(cfg.get("routes") or {})}
    return providers, routes

def write_sample_routing_config(path: Path):
    sample = {"providers": DEFAULT_PROVIDERS, "routes": DEFAULT_ROUTES}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    print(f"Sample routing config written to {path}")

def resolve_route(routes: dict[str, dict[str, str]], label: str) -> tuple[str, str]:
    r = routes.get(label) or routes.get("_default")
    if not r or "provider" not in r or "model" not in r:
        raise KeyError(f"No route for label '{label}' and no usable _default route.")
    return r["provider"], r["model"]

_NO_SAMPLING_PARAMS_MODELS: set[str] = set()
_NO_SAMPLING_PARAMS_LOCK = threading.Lock()
_NO_SAMPLING_NAME_PATTERNS = ("thinking", "claude-opus-4-7", "deepseek-r1", "o1-", "o3-", "o4-", "gpt-5")

def _model_rejects_sampling(model: str) -> bool:
    with _NO_SAMPLING_PARAMS_LOCK:
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


# ---------------------------------------------------------------------------
# Rate limit tracker
# ---------------------------------------------------------------------------
class RateLimitTracker:
    def __init__(self, window_sec: float = 10.0, max_events: int = 5, cooldown_sec: float = 30.0):
        self.window = window_sec
        self.max_events = max_events
        self.cooldown = cooldown_sec
        self._events: deque[float] = deque()
        self._lock = threading.Lock()
        self._global_cooldown_until: float = 0.0

    def is_in_cooldown(self) -> float:
        with self._lock:
            if time.time() < self._global_cooldown_until:
                return self._global_cooldown_until - time.time()
            return 0.0

    def record_rate_limit(self) -> None:
        with self._lock:
            now = time.time()
            if now < self._global_cooldown_until:
                return
            self._events.append(now)
            while self._events and self._events[0] < now - self.window:
                self._events.popleft()
            if len(self._events) >= self.max_events:
                self._global_cooldown_until = now + self.cooldown
                self._events.clear()
                print(f"[rate-limit] global cooldown for {self.cooldown}s", file=sys.stderr)

_rate_tracker = RateLimitTracker()


# ---------------------------------------------------------------------------
# LENIENT CoT parser – the critical fix
# ---------------------------------------------------------------------------
def _extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content from \boxed{...} allowing nested braces and escaped backslashes."""
    pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    pattern2 = r"\\\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    match2 = re.search(pattern2, text)
    if match2:
        return match2.group(1).strip()
    return None

def _normalize_answer_loose(s: str) -> str:
    """
    Remove non-alphanumeric characters from beginning and end.
    This allows '}38' to match '38' and vice versa.
    """
    s = s.strip()
    # Strip leading/trailing non-alphanumeric (keep internal symbols if needed)
    s = re.sub(r'^[^a-zA-Z0-9]+', '', s)
    s = re.sub(r'[^a-zA-Z0-9]+$', '', s)
    return s

def _parse_cot(raw: str, expected_answer: str) -> Optional[str]:
    """
    Extract CoT with lenient answer matching.
    First tries strict format (<think>...</think> and \boxed{}).
    If that fails but the raw text contains the expected answer (after loose
    normalisation), accept the whole output as CoT to avoid wasting API credits
    on retries.
    """
    if not raw:
        return None

    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)

    # --- 1. Strict extraction (think tags + boxed) ---
    boxed = _extract_boxed_answer(s)
    if boxed is not None and _normalize_answer_loose(boxed) == _normalize_answer_loose(expected_answer):
        think_start = s.find("<think>")
        think_end = s.find("</think>", think_start) if think_start != -1 else -1
        if think_start != -1 and think_end != -1:
            boxed_pos = s.find("\\boxed{", think_end)
            if boxed_pos == -1:
                boxed_pos = s.find("\\\\boxed{", think_end)
            if boxed_pos != -1:
                depth = 0
                i = boxed_pos + len("\\boxed{")
                end_pos = len(s)
                while i < len(s):
                    if s[i] == '{':
                        depth += 1
                    elif s[i] == '}':
                        if depth == 0:
                            end_pos = i + 1
                            break
                        depth -= 1
                    i += 1
                cot = s[think_start:end_pos].strip()
                if cot and not _contains_emoji(cot):
                    return cot

    # --- 2. Lenient fallback: does the raw text contain the expected answer? ---
    norm_expected = _normalize_answer_loose(str(expected_answer))
    if not norm_expected:
        return None

    # Look for the normalised answer anywhere in the normalised text (case-insensitive)
    if norm_expected.lower() in _normalize_answer_loose(s).lower():
        if _contains_emoji(s):
            return None
        # Log a warning so you can inspect these informally formatted CoTs later
        print(f"[lenient] accepting non‑conforming output for answer='{norm_expected}'",
              file=sys.stderr)
        return s   # use the whole raw output

    return None


# ---------------------------------------------------------------------------
# LLM call with fewer retries (default 3)
# ---------------------------------------------------------------------------
def call_llm(
    client_pool: ClientPool,
    provider_name: str,
    model: str,
    prompt: str,
    answer: str,
    variation_idx: int,
    label: str = "",
    max_attempts: int = 3,
) -> str:
    variants = LABEL_SYSTEM_PROMPTS.get(label) or SYSTEM_PROMPT_VARIANTS
    sys_prompt = variants[variation_idx % len(variants)]
    user_msg = USER_TEMPLATE.format(prompt=str(prompt)[:6000], answer=str(answer)[:1000])

    for attempt in range(1, max_attempts + 1):
        cooldown = _rate_tracker.is_in_cooldown()
        if cooldown > 0:
            print(f"[rate-limit] global cooldown for {cooldown:.1f}s", file=sys.stderr)
            time.sleep(cooldown)
            continue

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
            if _is_bad_request(e) and sent_sampling and model not in _NO_SAMPLING_PARAMS_MODELS:
                with _NO_SAMPLING_PARAMS_LOCK:
                    _NO_SAMPLING_PARAMS_MODELS.add(model)
                print(f"[info] 400 from '{model}'; disabling temperature/seed and retrying", file=sys.stderr)
                continue
            if _is_rate_limit(e):
                _rate_tracker.record_rate_limit()
                client_pool.rotate_provider(provider_name)
                wait = min(2.0 ** (attempt - 1) + random.uniform(0, 1), 60.0)
                print(f"[rate-limit] key rotated, sleeping {wait:.1f}s (attempt {attempt})", file=sys.stderr)
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
        print(f"[warning] attempt {attempt} invalid output (preview: {text[:200]})", file=sys.stderr)
        time.sleep(1.0 + random.uniform(0, 0.5))

    raise RuntimeError(f"exhausted {max_attempts} attempts")


# ---------------------------------------------------------------------------
# 3. Balancing (unchanged)
# ---------------------------------------------------------------------------
def build_plan(df: pd.DataFrame, target: int, rng: random.Random) -> list[dict[str, Any]]:
    per_label: dict[str, list[dict[str, Any]]] = {}
    for lbl in LABELS:
        rows = df[df["label"] == lbl].to_dict("records")
        if not rows:
            continue
        bucket: list[dict[str, Any]] = []
        if len(rows) >= target:
            picked = rng.sample(rows, target)
            for r in picked:
                bucket.append({"prompt": r["prompt"], "answer": r["answer"], "label": lbl, "variation_idx": 0})
        else:
            for r in rows:
                bucket.append({"prompt": r["prompt"], "answer": r["answer"], "label": lbl, "variation_idx": 0})
            need = target - len(rows)
            v, i = 1, 0
            while need > 0:
                r = rows[i % len(rows)]
                bucket.append({"prompt": r["prompt"], "answer": r["answer"], "label": lbl, "variation_idx": v})
                i += 1
                if i % len(rows) == 0:
                    v += 1
                need -= 1
        per_label[lbl] = bucket

    plan: list[dict[str, Any]] = []
    cursors = {lbl: 0 for lbl in per_label}
    while True:
        added = False
        for lbl in LABELS:
            bucket = per_label.get(lbl)
            if not bucket:
                continue
            idx = cursors[lbl]
            if idx < len(bucket):
                plan.append(bucket[idx])
                cursors[lbl] = idx + 1
                added = True
        if not added:
            break
    return plan


# ---------------------------------------------------------------------------
# 4. Output / resume
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

def load_existing_prompts(output_path: Path) -> set[str]:
    """Return set of prompt strings already present in the output CSV with a non-empty CoT."""
    if not output_path.exists():
        return set()
    existing: set[str] = set()
    try:
        with open(output_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cot = (row.get("cot") or "").strip()
                prompt = (row.get("prompt") or "").strip()
                if prompt and cot:
                    existing.add(prompt)
    except Exception as e:
        print(f"[warn] could not read existing output: {e}", file=sys.stderr)
    return existing


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Lenient parallel CoT generation to save credits.")
    ap.add_argument("--input", type=Path, default=Path("data/src/train.csv"))
    ap.add_argument("--output", type=Path, default=Path("final_output.csv"))
    ap.add_argument("--progress", type=Path, default=None)
    ap.add_argument("--routing-config", type=Path, default=None)
    ap.add_argument("--write-sample-routing", type=Path, default=None)
    ap.add_argument("--only-labels", default=None)
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--max-per-label", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=3, help="Max attempts per row (default 3 to save credits)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-model", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--rate-limit-cooldown", type=float, default=30.0)
    ap.add_argument("--rate-limit-window", type=float, default=10.0)
    ap.add_argument("--rate-limit-threshold", type=int, default=5)
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

    # Routing
    if args.routing_config and args.routing_config.exists():
        providers, routes = load_routing_config(args.routing_config)
    else:
        providers = DEFAULT_PROVIDERS
        routes = DEFAULT_ROUTES

    if args.force_model:
        routes = {k: {**v, "model": args.force_model} for k, v in routes.items()}
        print(f"\n[force-model] All labels overridden -> {args.force_model}")

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

    # Configure rate tracker
    global _rate_tracker
    _rate_tracker = RateLimitTracker(
        window_sec=args.rate_limit_window,
        max_events=args.rate_limit_threshold,
        cooldown_sec=args.rate_limit_cooldown
    )

    # Count API keys
    key_count = 0
    i = 1
    while os.environ.get(f"AISA_API_KEY_{i}"):
        key_count += 1
        i += 1
    if key_count == 0 and os.environ.get("AISA_API_KEY"):
        key_count = 1
    if key_count == 0:
        print("ERROR: No AISA API keys found.", file=sys.stderr)
        return 1

    workers = args.workers if args.workers > 0 else key_count
    print(f"\n[parallel] Using {workers} workers with {key_count} API keys.")
    print(f"[info] max_attempts per row = {args.max_attempts} (reduced to save credits)")

    existing_prompts = load_existing_prompts(args.output)
    print(f"[resume] {len(existing_prompts)} prompts already in output CSV — will skip these")

    # Source of truth: every row in train.csv, not the sampled plan
    all_train_items = [
        {"prompt": row["prompt"], "answer": row["answer"], "label": row["label"], "variation_idx": 0}
        for _, row in df.iterrows()
    ]
    pending_items = [item for item in all_train_items if item["prompt"] not in existing_prompts]
    total_pending = len(pending_items)
    if total_pending == 0:
        print("No pending rows. Exiting.")
        return 0

    # Prepare output CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_exists = args.output.exists() and args.output.stat().st_size > 0
    out_f = open(args.output, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
    if not file_exists:
        writer.writeheader()
        out_f.flush()

    write_lock = threading.Lock()
    existing_prompts_lock = threading.Lock()
    client_pool = ClientPool(providers)

    def process_one(item: dict[str, Any]) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
        try:
            prov_name, model_id = resolve_route(routes, item["label"])
            cot = call_llm(
                client_pool, prov_name, model_id,
                item["prompt"], item["answer"], item["variation_idx"],
                label=item["label"],
                max_attempts=args.max_attempts,
            )
            row = {
                "prompt": item["prompt"],
                "answer": item["answer"],
                "cot": cot,
                "label": item["label"],
            }
            return (True, row, None)
        except Exception as e:
            return (False, None, f"{type(e).__name__}: {e}")

    succ = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {executor.submit(process_one, item): item for item in pending_items}
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                success, row, error = future.result()
            except Exception as exc:
                success = False
                error = str(exc)
                row = None

            if success:
                with write_lock:
                    writer.writerow(row)
                    out_f.flush()
                with existing_prompts_lock:
                    existing_prompts.add(item["prompt"])
                succ += 1
                print(f"[{succ+fail}/{total_pending}][{item['label']}] OK (var={item['variation_idx']})")
            else:
                fail += 1
                print(f"[{succ+fail}/{total_pending}][{item['label']}] FAIL: {error}", file=sys.stderr)

    out_f.close()
    print(f"\nDONE. success={succ} fail={fail} skipped={len(plan)-total_pending} -> {args.output}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())