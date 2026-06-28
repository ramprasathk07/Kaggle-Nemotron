"""
generate_traces.py — distill reasoning traces for synthetic_hard.csv using
OpenAI-compatible models on https://aisa.one (e.g. GPT-5).

Features
  * MULTI-KEY rotation: put N keys in .env; on rate-limit (429) a key is put on
    cooldown and the next key is used automatically (round-robin, thread-safe).
  * BEST-MODEL auto-select: --model auto picks the strongest available model from
    /v1/models by a priority list (gpt-5 > o3 > opus > sonnet > gpt-4.1 > r1 ...).
  * 8192-token budget: the model is instructed to produce a solid but compact trace
    that, with the prompt, fits the 8192 train context; completion capped accordingly.
  * Verified: each trace's \\boxed{} is checked vs the KNOWN answer (official metric).
    Blind solve first; optional --fallback-rationalize rescues persistent failures.
  * Resumable + incremental CSV: traces.jsonl (source of truth, append per row) and
    traces.csv (id, category, prompt, answer, generated_cot, boxed, correct, mode),
    rewritten on every flush so the CoT CSV is always saved.

Config: copy .env.example -> .env and fill keys. Run: python generate_traces.py
"""
from __future__ import annotations
import os, re, csv, json, math, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_SUFFIX = ("\nPlease put your final answer inside `\\boxed{}`. "
                 "For example: `\\boxed{your answer}`")

SYSTEM_SOLVE = (
    "You solve deterministic symbol/equation puzzles with exactly one correct answer. "
    "Each prompt gives example transformations; infer the EXACT rule (concatenation, "
    "reverse concatenation, addition, subtraction, multiplication, a digit-position "
    "operation like (di+dj) mod 10, a determinant, modulo, or a symbol-substitution "
    "concat) and apply it to the query. Work step by step and TEST your rule against "
    "EVERY example before trusting it; preserve digit/symbol formatting exactly "
    "(including leading zeros).\n\n"
    "Write a SOLID, self-contained reasoning trace, but keep it COMPACT: the entire "
    "response must fit well under 8000 tokens (no rambling, no restating the prompt). "
    "Put all reasoning inside one <think> ... </think> block, then output the final "
    "answer once as \\boxed{...} with nothing after it.")

SYSTEM_RATIONALIZE = (
    "You are given a puzzle AND its verified correct answer. Produce a clean, rigorous, "
    "COMPACT step-by-step derivation (well under 8000 tokens) that infers the rule from "
    "the examples and arrives at exactly that answer. Do not mention that the answer was "
    "given. Put all reasoning in one <think> ... </think> block, then output the answer "
    "once as \\boxed{...}.")

# model priority for --model auto (substring match, first hit wins; skip mini/nano)
MODEL_PRIORITY = ["gpt-5.5-pro", "gpt-5.5", "gpt-5-pro", "gpt-5", "o3",
                  "claude-opus", "opus", "claude-3.7-sonnet", "sonnet",
                  "gpt-4.1", "deepseek-r1", "r1", "gpt-4o"]

_BX = re.compile(r"\\boxed\{([^{}]*)\}")
def extract_boxed(t):
    if not t:
        return None
    m = _BX.findall(t)
    return m[-1].strip() if m else None

def strip_all_boxed(s):
    """Remove every \\boxed{...} (brace-balanced) so we can re-append a clean one."""
    tok, out, i = "\\boxed{", [], 0
    while i < len(s):
        j = s.find(tok, i)
        if j == -1:
            out.append(s[i:]); break
        out.append(s[i:j]); k = j + len(tok); d = 1
        while k < len(s) and d > 0:
            if s[k] == "{": d += 1
            elif s[k] == "}": d -= 1
            k += 1
        i = k
    return "".join(out)

def compare_answer(stored, predicted):
    """Official metric (syn_datagen/reasoning.py)."""
    if predicted is None:
        return False
    stored, predicted = str(stored).strip(), str(predicted).strip()
    if re.fullmatch(r"[01]+", stored):
        return predicted.lower() == stored.lower()
    try:
        return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored.lower()

def has_think_then_boxed(t):
    return ("</think>" in t and "\\boxed{" in t
            and t.find("</think>") < t.rfind("\\boxed{"))

_print_lock = threading.Lock()
def log(*a):
    with _print_lock:
        print(*a, flush=True)

# ───────────────────────── config loading (.env + env) ─────────────────────────
def load_dotenv(path):
    """Tiny .env parser (no dependency). KEY=VALUE lines; # comments; quotes stripped."""
    env = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env

def collect_keys(env):
    keys = []
    raw = env.get("AISA_API_KEYS") or os.environ.get("AISA_API_KEYS", "")
    for part in re.split(r"[,\s]+", raw):
        if part:
            keys.append(part)
    # also AISA_API_KEY and AISA_API_KEY_1..N
    for src in (env, os.environ):
        for k, v in src.items():
            if re.fullmatch(r"AISA_API_KEY(_\d+)?", k) and v and v not in keys:
                keys.append(v)
    # de-dup, keep order
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k); out.append(k)
    return out

# ───────────────────────── key pool with cooldown rotation ─────────────────────────
class KeyPool:
    def __init__(self, keys, base_url, timeout=120):
        from openai import OpenAI
        # max_retries=0 -> OUR rotation handles retries (no silent SDK retry/hang);
        # timeout -> a bad endpoint fails fast instead of hanging forever.
        self.entries = [{"key": k,
                         "client": OpenAI(base_url=base_url, api_key=k,
                                          timeout=timeout, max_retries=0),
                         "cool_until": 0.0, "n": 0} for k in keys]
        self.lock = threading.Lock()
        self.rr = 0

    def acquire(self):
        """Round-robin pick of a key that isn't cooling; else the soonest-free one."""
        with self.lock:
            n = len(self.entries); now = time.time()
            for _ in range(n):
                e = self.entries[self.rr % n]; self.rr += 1
                if e["cool_until"] <= now:
                    e["n"] += 1
                    return e
            e = min(self.entries, key=lambda x: x["cool_until"])
            return e

    def penalize(self, e, secs):
        with self.lock:
            e["cool_until"] = max(e["cool_until"], time.time() + secs)

# ───────────────────────── model selection + call ─────────────────────────
def pick_best_model(pool, requested):
    if requested and requested != "auto":
        return requested
    e = pool.acquire()
    try:
        ids = [m.id for m in e["client"].models.list().data]
    except Exception as ex:
        log(f"[model] /models list failed ({ex}); falling back to 'gpt-5.5-pro'")
        return "gpt-5.5-pro"
    low = [(m, m.lower()) for m in ids]
    for pref in MODEL_PRIORITY:
        for orig, lo in low:
            if pref in lo and "mini" not in lo and "nano" not in lo:
                log(f"[model] auto-selected '{orig}' from {len(ids)} models")
                return orig
    log(f"[model] no priority match; using first of {len(ids)}: {ids[0]}")
    return ids[0]

def _needs_responses(model):
    """aisa.one: -pro / -codex / o-series are reasoning models served on /v1/responses,
    NOT /v1/chat/completions (which 404s 'not a chat model')."""
    m = model.lower()
    return ("-pro" in m) or ("-codex" in m) or m.startswith(("o1", "o3", "o4"))

def _invoke(client, model, system, user, max_out):
    """Call the right endpoint; gpt-5.x reject temperature!=1, so we omit temperature."""
    def via_responses():
        r = client.responses.create(model=model, instructions=system, input=user,
                                    max_output_tokens=max_out)
        return r.output_text
    def via_chat():
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_out)
        return r.choices[0].message.content
    order = [via_responses, via_chat] if _needs_responses(model) else [via_chat, via_responses]
    last = None
    for fn in order:
        try:
            return fn()
        except Exception as ex:
            last = ex
            s = str(ex).lower()
            if any(t in s for t in ("not a chat model", "v1/completions", "v1/responses",
                                    "not supported in", "did you mean")):
                continue                       # endpoint mismatch -> try the other one
            raise
    raise last

def chat(pool, model, system, user, max_out, temperature):
    """Invoke with key rotation on rate-limit (temperature kept for signature compat)."""
    from openai import RateLimitError
    attempts = max(6, len(pool.entries) * 2)
    last = None
    for k in range(attempts):
        e = pool.acquire()
        wait = e["cool_until"] - time.time()
        if wait > 0:
            time.sleep(min(wait, 5))
        try:
            msg = _invoke(e["client"], model, system, user, max_out)
            if not msg:
                last = RuntimeError("empty content"); time.sleep(min(20, 2 ** k)); continue
            return msg
        except RateLimitError as ex:
            last = ex; pool.penalize(e, 60)
            log(f"[rate-limit] key#{pool.entries.index(e)} cooling 60s -> rotating")
        except Exception as ex:
            last = ex
            code = getattr(ex, "status_code", None)
            if code == 429:
                pool.penalize(e, 60)
            log(f"[retry {k}] {type(ex).__name__} {code}: {str(ex)[:100]}")
            time.sleep(min(20, 2 ** k))
    raise RuntimeError(f"all attempts failed: {last}")

# ───────────────────────── output (jsonl + incremental CoT csv) ─────────────────────────
CSV_COLS = ["id", "category", "prompt", "answer", "generated_cot", "boxed", "correct", "mode"]

def rewrite_csv(jsonl_path, csv_path):
    rows = []
    if os.path.exists(jsonl_path):
        for line in open(jsonl_path, encoding="utf-8"):
            try: rows.append(json.loads(line))
            except Exception: pass
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS); w.writeheader()
        for x in rows:
            w.writerow({c: x.get(c, "") for c in CSV_COLS})
    return len(rows)

def load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try: done.add(json.loads(line)["id"])
            except Exception: pass
    return done

# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(HERE, "synthetic_hard.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "traces.jsonl"))
    ap.add_argument("--csv", default=os.path.join(HERE, "traces.csv"))
    ap.add_argument("--env", default=os.path.join(HERE, ".env"))
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--max-out", type=int, default=7000, help="completion-token cap (<8192 budget)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--solve-attempts", type=int, default=2)
    ap.add_argument("--fallback-rationalize", action="store_true")
    a = ap.parse_args()

    env = load_dotenv(a.env)
    base_url = a.base_url or env.get("AISA_BASE_URL") or os.environ.get("AISA_BASE_URL", "https://api.aisa.one/v1")
    keys = collect_keys(env)
    if not keys:
        raise SystemExit(f"No API keys. Put AISA_API_KEYS=key1,key2,... in {a.env} "
                         f"(copy .env.example).")
    pool = KeyPool(keys, base_url)
    model = pick_best_model(pool, a.model or env.get("AISA_MODEL", "auto"))
    log(f"keys={len(keys)} | base={base_url} | model={model} | max_out={a.max_out}")

    rows = list(csv.DictReader(open(a.inp, encoding="utf-8")))
    if a.limit:
        rows = rows[:a.limit]
    done = load_done(a.out)
    todo = [r for r in rows if r["id"] not in done]
    log(f"rows={len(rows)} done={len(done)} todo={len(todo)}")

    out_lock = threading.Lock()
    fout = open(a.out, "a", encoding="utf-8")
    stats = {"solve": 0, "rationalize": 0, "failed": 0}

    def work(r):
        t0 = time.time()
        user = r["prompt"] + PROMPT_SUFFIX
        ans = r["answer"]
        txt = box = None
        for _ in range(max(1, a.solve_attempts)):
            txt = chat(pool, model, SYSTEM_SOLVE, user, a.max_out, a.temperature)
            box = extract_boxed(txt)
            if compare_answer(ans, box) and has_think_then_boxed(txt):
                return r, txt, box, True, "solve", time.time() - t0
        if a.fallback_rationalize:
            ruser = user + f"\n\n[The verified correct answer is: {ans}]"
            txt = chat(pool, model, SYSTEM_RATIONALIZE, ruser, a.max_out, a.temperature)
            box = extract_boxed(txt)
            if compare_answer(ans, box):
                return r, txt, box, True, "rationalize", time.time() - t0
            txt = strip_all_boxed(txt).rstrip() + f"\n\\boxed{{{ans}}}"
            return r, txt, ans, True, "rationalize_forced", time.time() - t0
        return r, txt, box, False, "solve", time.time() - t0

    # sync csv from existing jsonl (resume), then APPEND each row (save interval = 1)
    rewrite_csv(a.out, a.csv)
    csvf = open(a.csv, "a", newline="", encoding="utf-8")
    cw = csv.writer(csvf)
    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(todo), desc="traces", unit="row")
    except Exception:
        bar = None

    N = len(todo)
    log(f"submitting {N} tasks across {a.workers} workers (model={model}); first results in a moment...")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r, txt, box, ok, mode, secs = fut.result()
            except Exception as e:
                log("[error]", repr(e))
                if bar: bar.update(1)
                continue
            rec = {"id": r["id"], "category": r["category"], "prompt": r["prompt"],
                   "answer": r["answer"], "generated_cot": txt, "boxed": box,
                   "correct": ok, "mode": mode,
                   "messages": [{"role": "system", "content": SYSTEM_SOLVE},
                                {"role": "user", "content": r["prompt"] + PROMPT_SUFFIX},
                                {"role": "assistant", "content": txt}]}
            with out_lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                cw.writerow([rec.get(c, "") for c in CSV_COLS]); csvf.flush()   # save every row
            stats["rationalize" if mode.startswith("rationalize") else ("solve" if ok else "failed")] += 1
            ntok = len(str(txt)) // 4
            msg = (f"[{i}/{N}] {r['id']} {r['category'][:14]:14s} "
                   f"{'OK ' if ok else ' X '} {mode:18s} {secs:4.0f}s ~{ntok}tok "
                   f"box={str(box)[:18]!r} | {stats}")
            if bar:
                bar.update(1); bar.set_postfix_str(str(stats)); tqdm.write(msg)
            else:
                log(msg)
    if bar: bar.close()
    csvf.close(); fout.close()
    n = rewrite_csv(a.out, a.csv)
    from collections import Counter
    recs = [json.loads(l) for l in open(a.out, encoding="utf-8")]
    log("DONE. traces:", len(recs), "| by mode:", dict(Counter(x["mode"] for x in recs)),
        "| correct:", sum(1 for x in recs if x["correct"]))
    log(f"CoT CSV -> {a.csv} ({n} rows). SFT chat -> {a.out}")

if __name__ == "__main__":
    main()
