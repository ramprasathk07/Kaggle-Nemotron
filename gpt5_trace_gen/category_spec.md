# Hard equation categories — spec for GPT-5 trace generation

These four categories are where the deterministic syn_datagen solvers are weak, so we distill reasoning traces from a stronger model (GPT-5). Every problem below was generated from a known rule and **solver-verified**, so the `answer` is ground truth — your trace's `\boxed{}` must match it.

## Solver accuracy on the competition problems

| Category | rule_found / total | accuracy |
|---|---|---|
| cryptarithm_deduce | 54 / 659 | 8.2% |
| cryptarithm_guess | 11 / 164 | 6.7% |
| equation_numeric_deduce | 540 / 596 | 90.6% |
| equation_numeric_guess | — | low |

## Categories

### equation_numeric_deduce
Visible two-digit operands joined by a single SYMBOL operator, e.g. `13{45 = 1345`. Each operator symbol stands for ONE arithmetic/string operation that is consistent across all examples sharing that symbol. 'deduce' uses COMMON operations: concatenation, reverse concatenation, addition, absolute difference, subtraction, reverse subtraction, multiplication. Infer the operation from the examples, then apply it to the query.

```
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
13{45 = 1345
41{38 = 4138
27{23 = 2723
96{79 = 9679
Now, determine the result for: 21{85
ANSWER: 2185
```
```
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
13|21 = 8
37|39 = 2
74|87 = 13
13|81 = 68
Now, determine the result for: 35|93
ANSWER: 58
```

### equation_numeric_guess
Same visible-digit format, but the operation is RARE/digit-wise: digit add mod10, digit sub mod10, digit multiply, cross multiply, determinant (d1*d4 - d2*d3), modulo, integer division. Harder to spot; test digit-position operations explicitly. Example `48/69 = 07` is digit add mod10: (4+6)%10=0, (8+9)%10=7.

```
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
48/69 = 07
96/79 = 65
64/78 = 32
58/39 = 87
Now, determine the result for: 41/68
ANSWER: 09
```
```
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
86{14 = 72
67{86 = 81
96{16 = 80
41{15 = 36
Now, determine the result for: 61{66
ANSWER: 05
```

### cryptarithm_deduce
Like the equation format, but every DIGIT is replaced by a fixed unique SYMBOL (a 10-symbol substitution). The rule here is FORWARD concatenation of the two encoded operands. You do NOT need to decode the digits — operate on the symbols directly: output = left-operand-symbols followed by right-operand-symbols.

```
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
`=#** = `=**
|+#%* = |+%*
({#;` = ({;`
*`#=` = *`=`
Now, determine the result for: `;#;`
ANSWER: `;;`
```
```
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
}#]#( = }##(
(*],` = (*,`
+(]+; = +(+;
=`]~; = =`~;
Now, determine the result for: ++]+#
ANSWER: +++#
```

### cryptarithm_guess
Symbol-encoded operands with REVERSE concatenation: output = right-operand-symbols followed by left-operand-symbols. Operate on the symbols directly.

```
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
;@|*^ = *^;@
^*|.; = .;^*
*_|@. = @.*_
$$|;] = ;]$$
Now, determine the result for: $;|$@
ANSWER: $@$;
```
```
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
&[})[ = )[&[
[#}[, = [,[#
.)}&; = &;.)
[&}.; = .;[&
Now, determine the result for: )&}_)
ANSWER: _))&
```

## Answer contract
Output format (mandatory): put all reasoning inside one <think> ... </think> block, then immediately the final answer ONCE as \boxed{...} with nothing after it. The boxed content is the answer only, in the exact form the prompt expects (a number, a digit string with leading zeros preserved, or the symbol string) — no spaces, no operator, no extra words. Answers are checked with the official metric: numeric within 1e-2 relative tolerance, otherwise exact case-insensitive string match.
