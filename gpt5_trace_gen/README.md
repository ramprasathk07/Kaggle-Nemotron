# gpt5_trace_gen

Synthetic, **solver-verified** problems for the hard equation categories +
GPT-5 (via aisa.one) trace distillation, to train the model further on the
categories where the deterministic syn_datagen solvers are weak.

| Category | solver acc (comp) | here |
|---|---|---|
| cryptarithm_deduce | 8.2% | symbol-concat (fwd), solver-verified |
| cryptarithm_guess | 6.7% | symbol-concat (rev), solver-verified |
| equation_numeric_deduce | 90.6% | visible-digit, common op, solver-verified |
| equation_numeric_guess | low | visible-digit, rare/digit op, solver-verified |

**Why this is safe:** every problem is generated from one chosen rule, then the matching
syn_datagen reasoner is run and we keep ONLY rows whose solver answer equals the
constructed answer. So the `answer` column is ground truth — GPT-5 traces are checked
against it.

## Workflow

```bash
# 1) generate solver-verified problems (Alice-style prompt + correct answer)
python generate_problems.py --per-cat 200          # -> synthetic_hard.csv (800 rows)

# 2) build the spec doc for GPT-5 context (PDF + Markdown)
pip install fpdf2
python build_spec_pdf.py                            # -> category_spec.pdf + .md

# 3) distill GPT-5 reasoning traces via aisa.one (OpenAI-compatible)
pip install openai
export AISA_API_KEY=sk-...                           # from your aisa.one dashboard
export AISA_BASE_URL=https://aisa.one/v1             # confirm exact base_url
export AISA_MODEL=gpt-5                              # confirm exact model id
python generate_traces.py --workers 4 --fallback-rationalize
#   -> traces.jsonl  (chat messages, SFT-ready)  +  traces.csv (summary)
```

## Files
- `generate_problems.py` — forward generators + solver verification → `synthetic_hard.csv`.
- `build_spec_pdf.py` — `category_spec.pdf` / `.md` (rules, format, examples, answer contract).
- `generate_traces.py` — aisa.one client; blind-solve + verify, optional rationalize-fallback;
  resumable; outputs `traces.jsonl` (system/user/assistant) ready for SFT/RAFT.

## Notes
- `generate_problems.py` copies `../syn_datagen` into `./reasoners/` (the modules import
  as `from reasoners.X`). That dir is git-ignored and regenerated on each run.
- `generate_traces.py` assumes aisa.one is OpenAI-compatible (`/v1/chat/completions`).
  If the base_url/model differ, set them via env/flags. The boxed answer is always
  validated with the **official metric** (numeric 1e-2 rel-tol, else exact string).
- Keep only `correct=True` traces for training. `mode=solve` = honest blind solve;
  `mode=rationalize*` = answer was provided (use sparingly; lower-quality reasoning).
