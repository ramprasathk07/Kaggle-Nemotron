"""
build_spec_pdf.py — render a category-spec doc (PDF + Markdown) describing the four
hard equation categories, their rules, prompt format, worked examples (pulled from
synthetic_hard.csv), and the answer-format contract. Hand this to GPT-5 as context
when generating reasoning traces.

PDF via fpdf2 (`pip install fpdf2`). Always also writes category_spec.md as a fallback.
"""
import os, csv
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "synthetic_hard.csv")
PDF_PATH = os.path.join(HERE, "category_spec.pdf")
MD_PATH  = os.path.join(HERE, "category_spec.md")

# Solver accuracy on the competition's own problems (from syn_datagen/reasoning.py).
SOLVER_ACC = [
    ("cryptarithm_deduce",        "54 / 659",  "8.2%"),
    ("cryptarithm_guess",         "11 / 164",  "6.7%"),
    ("equation_numeric_deduce",   "540 / 596", "90.6%"),
    ("equation_numeric_guess",    "—",          "low"),
]

CATEGORY_DOC = {
    "equation_numeric_deduce": (
        "Visible two-digit operands joined by a single SYMBOL operator, e.g. `13{45 = 1345`. "
        "Each operator symbol stands for ONE arithmetic/string operation that is consistent "
        "across all examples sharing that symbol. 'deduce' uses COMMON operations: "
        "concatenation, reverse concatenation, addition, absolute difference, subtraction, "
        "reverse subtraction, multiplication. Infer the operation from the examples, then "
        "apply it to the query."),
    "equation_numeric_guess": (
        "Same visible-digit format, but the operation is RARE/digit-wise: digit add mod10, "
        "digit sub mod10, digit multiply, cross multiply, determinant (d1*d4 - d2*d3), "
        "modulo, integer division. Harder to spot; test digit-position operations explicitly. "
        "Example `48/69 = 07` is digit add mod10: (4+6)%10=0, (8+9)%10=7."),
    "cryptarithm_deduce": (
        "Like the equation format, but every DIGIT is replaced by a fixed unique SYMBOL "
        "(a 10-symbol substitution). The rule here is FORWARD concatenation of the two "
        "encoded operands. You do NOT need to decode the digits — operate on the symbols "
        "directly: output = left-operand-symbols followed by right-operand-symbols."),
    "cryptarithm_guess": (
        "Symbol-encoded operands with REVERSE concatenation: output = right-operand-symbols "
        "followed by left-operand-symbols. Operate on the symbols directly."),
}

ANSWER_CONTRACT = (
    "Output format (mandatory): put all reasoning inside one <think> ... </think> block, "
    "then immediately the final answer ONCE as \\boxed{...} with nothing after it. The boxed "
    "content is the answer only, in the exact form the prompt expects (a number, a digit "
    "string with leading zeros preserved, or the symbol string) — no spaces, no operator, "
    "no extra words. Answers are checked with the official metric: numeric within 1e-2 "
    "relative tolerance, otherwise exact case-insensitive string match.")

def load_examples():
    ex = defaultdict(list)
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if len(ex[r["category"]]) < 2:
                    ex[r["category"]].append(r)
    return ex

def build_markdown(ex):
    L = ["# Hard equation categories — spec for GPT-5 trace generation\n",
         "These four categories are where the deterministic syn_datagen solvers are weak, "
         "so we distill reasoning traces from a stronger model (GPT-5). Every problem below "
         "was generated from a known rule and **solver-verified**, so the `answer` is ground "
         "truth — your trace's `\\boxed{}` must match it.\n",
         "## Solver accuracy on the competition problems\n",
         "| Category | rule_found / total | accuracy |", "|---|---|---|"]
    for c, frac, acc in SOLVER_ACC:
        L.append(f"| {c} | {frac} | {acc} |")
    L.append("")
    L.append("## Categories\n")
    for cat, desc in CATEGORY_DOC.items():
        L.append(f"### {cat}\n{desc}\n")
        for r in ex.get(cat, []):
            L.append("```")
            L.append(r["prompt"])
            L.append(f"ANSWER: {r['answer']}")
            L.append("```")
        L.append("")
    L.append("## Answer contract\n" + ANSWER_CONTRACT + "\n")
    return "\n".join(L)

def write_pdf(ex):
    try:
        from fpdf import FPDF
    except Exception:
        print("[pdf] fpdf2 not installed (`pip install fpdf2`); wrote category_spec.md only.")
        return False
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    def H(t, sz=15):
        pdf.set_font("Helvetica", "B", sz); pdf.multi_cell(0, 8, t); pdf.ln(1)
    def P(t, sz=10, font="Helvetica", style=""):
        pdf.set_font(font, style, sz)
        pdf.multi_cell(0, 5, t.encode("latin-1", "replace").decode("latin-1")); pdf.ln(1)
    H("Hard equation categories - GPT-5 trace spec")
    P("Generated from known rules and solver-verified; the 'answer' is ground truth. "
      "Your trace's boxed answer must match it.")
    H("Solver accuracy on competition problems", 12)
    for c, frac, acc in SOLVER_ACC:
        P(f"  {c:28s} {frac:12s} {acc}", font="Courier", sz=9)
    for cat, desc in CATEGORY_DOC.items():
        H(cat, 12); P(desc)
        for r in ex.get(cat, []):
            P(r["prompt"] + "\nANSWER: " + r["answer"], font="Courier", sz=8)
    H("Answer contract", 12); P(ANSWER_CONTRACT)
    pdf.output(PDF_PATH)
    print("[pdf] wrote", PDF_PATH)
    return True

def main():
    ex = load_examples()
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(build_markdown(ex))
    print("[md] wrote", MD_PATH)
    write_pdf(ex)

if __name__ == "__main__":
    main()
