# SSL Reasoner

SSL Reasoner is a math-only research prototype for testing whether a model can learn
structured arithmetic reasoning through latent and joint-embedding style training. It is
not a pretrained LLM and it does not call external math tools.

The project currently focuses on synthetic integer arithmetic with `+`, `-`, `*`, and
safe parenthesized expression parsing.

## Current Status

The repo has three main solver paths:

| Path | Status |
| --- | --- |
| Structured parser / trace renderer | Reliable for supported expressions; used as the deterministic reference path. |
| Learned operation / variable trace path | Strong when constrained to legal reduction order and candidate-scored arithmetic. |
| Raw learned arithmetic transition | Experimental; still weak on larger numbers and long chains. |

The best practical math solver is still the structured trace path. The standalone learned
transition can solve many in-distribution reductions, but larger-number arithmetic is not
solved yet.

## Architecture

### Original Latent Answer Path

```text
problem text
  -> tokenizer + math feature extractor
  -> problem encoder
  -> answer-slot predictor
  -> predicted answer latent slots
  -> scratch transformer readout decoder
  -> answer tokens
```

The decoder is trained from scratch inside this repo. It is not a pretrained language
model. Diagnostics show that the decoder can read good target latents, but the predicted
answer latents are often not accurate enough.

### Structured Reasoning Path

```text
problem text
  -> tokenizer + math feature extractor
  -> problem encoder / trace predictor
  -> operation or reduction policy
  -> structured fields: lhs, op, rhs, result
  -> trace renderer
  -> final answer
```

Example:

```text
input:  "What is 2+3*4-1?"
trace:  3*4=12,2+12=14,14-1=13,13
answer: 13
```

### Latent Reasoning Sequence

This path is the first implementation of the joint-embedding reasoning idea:

```text
problem math features
  -> latent reasoning sequence predictor
  -> z_step1, z_step2, ..., z_final
  -> structured decoder heads
  -> lhs/op/rhs/result fields + final answer
```

Target step embeddings are built from the structured trace fields:

```text
lhs, op, rhs, result, step position -> z_target_step
final result, final position        -> z_target_final
```

Training uses latent matching, in-batch contrastive loss, hard negatives, active/stop
prediction, operation decoding, bounded value-class decoding, and a digit value decoder.
This is now implemented as a trainable baseline, but it is not yet the strongest solver.

### Experimental Standalone Transition

The standalone transition tries to learn a single arithmetic reduction:

```text
(lhs, op, rhs) -> result
```

Implemented value modes:

| Mode | Description |
| --- | --- |
| `class` | Bounded integer class from `-1000` to `10000`. |
| `digit` | Sign plus fixed decimal digits. |
| `factor` | Sign plus magnitude bucket plus offset. |
| `decomposed` | Place-wise output digit prediction from lhs/rhs digits and operator features. |
| `hybrid` | Uses class mode unless confidence/boundary checks switch to fallback. |

The strongest current standalone mode is still bounded `class` mode. The `digit`,
`factor`, and `decomposed` paths are useful baselines, but they have not fixed
larger-number generalization.

## Quick Start

Install the package in editable mode if needed:

```bash
pip install -e .
```

Run the default solver:

```bash
bash scripts/solve_math.sh "What is 2+3*4?"
```

Show debug information:

```bash
bash scripts/solve_math.sh --debug "What is 2+3*4?"
```

Show structured reasoning steps:

```bash
bash scripts/solve_math.sh --debug-reasoning "What is 49-5+19?"
```

Example debug-reasoning output:

```text
answer: 63
mode: operation
problem: What is 49-5+19?
reasoning_order: left_first
step1: lhs=49 op=- rhs=5 result=44
step2: lhs=44 op=+ rhs=19 result=63
final: 63
trace: 49-5=44,44+19=63,63
```

Emit JSON:

```bash
bash scripts/solve_math.sh --json "What is 2+3*4?"
```

## Checkpoints

The current default checkpoint is local and ignored by git:

```text
checkpoints/trace_ops_head_mixed/best.pt
```

Download the current best released checkpoint:

```bash
bash scripts/download_current_best.sh
```

Release page:

```text
https://github.com/akshai0296/ssl_reasoner/releases/tag/math-solver-v0.1.0
```

The checkpoint manifest is tracked at:

```text
reports/checkpoint_manifest.json
```

## Evaluation

Evaluate the default math solver:

```bash
bash scripts/eval_math_solver.sh
```

Evaluate another curriculum:

```bash
CURRICULUM=mixed_only bash scripts/eval_math_solver.sh
```

Evaluate variable-length reasoning:

```bash
bash scripts/eval_variable_reasoning.sh
```

Evaluate a specific checkpoint:

```bash
python -m ssl_reasoner.eval --checkpoint checkpoints/best.pt --samples 500
```

Evaluate standalone transition modes:

```bash
PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/standalone_transition_factor/best.pt \
  --standalone-learned-values \
  --preset all

PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/standalone_transition_factor/best.pt \
  --standalone-factor-values \
  --preset all

PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/standalone_transition_decomposed_smoke/best.pt \
  --standalone-decomposed-values \
  --preset all
```

Evaluate the latent reasoning sequence:

```bash
PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/latent_reasoning_sequence/best.pt \
  --latent-reasoning-values \
  --preset in_dist

PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/latent_reasoning_sequence_digit/best.pt \
  --latent-reasoning-digit-values \
  --preset in_dist

PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/latent_reasoning_sequence_process_conditioned_smoke/best.pt \
  --latent-reasoning-process-values \
  --preset in_dist
```

## Current Results

### Default Math Solver

Expected fixed 500-sample results for the current local best checkpoint:

| Curriculum | Accuracy |
| --- | ---: |
| `mixed` | `1.000` |
| `mixed_only` | `0.990` |
| `seen_mixed` | `1.000` |
| `unseen_mixed` | `0.998` |

### Variable-Length Structured Reasoning

The deterministic variable-length trace path is exact on the current supported
multi-step target:

| Metric | Score |
| --- | ---: |
| answer exact | `1.000` |
| trace exact | `1.000` |

The learned dynamic pointer with candidate-scored arithmetic is also strong under legal
reduction constraints. Without legal constraints, long expressions are weaker.

| Preset | Constrained final exact | Unconstrained final exact |
| --- | ---: | ---: |
| `in_dist` | `1.000` | `1.000` |
| `larger_numbers` | `1.000` | `1.000` |
| `longer_expr` | `1.000` | `1.000` |
| `no_multiply` | `1.000` | `0.996` |
| `many_multiply` | `1.000` | `1.000` |
| `length_8` | `0.988` | `0.984` |
| `length_16` | `0.750` | `0.448` |

### Standalone Learned Arithmetic

Recent 200-sample standalone transition results:

| Checkpoint / mode | In-dist | Larger numbers | Longer expr | No multiply | Many multiply | Length 8 | Length 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `standalone_transition_factor`, class | `0.750` | `0.005` | `0.700` | `1.000` | `0.620` | `0.395` | `0.125` |
| `standalone_transition_factor`, factor | `0.065` | `0.000` | `0.055` | `0.080` | `0.080` | `0.015` | `0.000` |
| `standalone_transition_factor`, hybrid | `0.690` | `0.000` | `0.645` | `1.000` | `0.565` | `0.305` | `0.070` |
| `standalone_transition_large`, class | `0.150` | `0.000` | `0.115` | `0.470` | `0.100` | `0.020` | `0.000` |
| `standalone_transition_decomposed_smoke`, decomposed | `0.475` | `0.005` | `0.390` | `0.720` | `0.370` | `0.150` | `0.045` |

Interpretation:

- The bounded class head is the best standalone transition so far.
- Larger-number generalization is effectively unsolved.
- Digit/factor/decomposed output formats alone are not enough.
- The next real improvement should add explicit carry/borrow supervision for `+`/`-` and
  partial-product accumulation states for `*`.

### Latent Reasoning Sequence

A 5000-step run confirms that the latent sequence architecture trains, but exact value
decoding remains weak:

| Metric | Step 1 | Step 5000 |
| --- | ---: | ---: |
| active accuracy | `0.606` | `1.000` |
| operation accuracy | `0.284` | `0.929` |
| value accuracy | `0.000` | `0.310` |
| final accuracy | `0.000` | `0.188` |

Public trace eval for `checkpoints/latent_reasoning_sequence/best.pt` is still low:

| Preset | Answer exact | Trace exact |
| --- | ---: | ---: |
| `in_dist` | `0.035` | `0.005` |
| `larger_numbers` | `0.000` | `0.000` |
| `longer_expr` | `0.005` | `0.000` |
| `no_multiply` | `0.020` | `0.000` |
| `many_multiply` | `0.060` | `0.030` |
| `length_8` | `0.005` | `0.000` |
| `length_16` | `0.000` | `0.000` |

Interpretation: the model learns active/stop and operation structure in latent space, but
the bounded whole-value decoder is not accurate enough.

The first digit-decoder experiment adds sign plus fixed decimal digits to each latent
reasoning state. Initializing from `checkpoints/latent_reasoning_sequence/best.pt` and
training for 2000 more steps gives:

| Metric | Step 1 | Step 2000 |
| --- | ---: | ---: |
| class value accuracy | `0.318` | `0.379` |
| class final accuracy | `0.141` | `0.156` |
| digit value accuracy | `0.000` | `0.228` |
| digit final accuracy | `0.000` | `0.016` |

Public in-dist eval on `checkpoints/latent_reasoning_sequence_digit/best.pt`:

| Decode mode | Answer exact | Trace exact |
| --- | ---: | ---: |
| class | `0.035` | `0.005` |
| digit | `0.025` | `0.000` |

Interpretation: direct digit decoding learns some value fields, but it is not better than
the class decoder yet. The next architectural step is explicit carry/borrow supervision
for addition/subtraction and partial-product supervision for multiplication.

The first arithmetic-process supervision pass adds per-place result digit targets and
carry/borrow targets for supported nonnegative `+`, `-`, and `*` reductions. Starting
from `checkpoints/latent_reasoning_sequence/best.pt` and training for 1000 steps gives:

| Metric | Step 1 | Step 1000 |
| --- | ---: | ---: |
| class value accuracy | `0.344` | `0.395` |
| class final accuracy | `0.141` | `0.219` |
| digit value accuracy | `0.000` | `0.198` |
| process digit accuracy | `0.095` | `0.578` |
| process carry accuracy | `0.002` | `0.878` |

Public eval for `checkpoints/latent_reasoning_sequence_process_smoke/best.pt` is still
low:

| Preset | Answer exact | Trace exact |
| --- | ---: | ---: |
| `in_dist` | `0.030` | `0.005` |
| `larger_numbers` | `0.000` | `0.000` |
| `longer_expr` | `0.015` | `0.000` |
| `no_multiply` | `0.020` | `0.000` |
| `many_multiply` | `0.055` | `0.040` |
| `length_8` | `0.000` | `0.000` |
| `length_16` | `0.000` | `0.000` |

Interpretation: the auxiliary arithmetic-process heads are learnable, especially carry
state, but they are not yet coupled tightly enough to force the final decoded value to be
correct. The next version should feed predicted carry/borrow states into the value
decoder, not only train them as auxiliary heads.

The process-conditioned value decoder feeds predicted result-digit and carry/borrow
probabilities into a second bounded value head. This tests whether the learned arithmetic
process state can improve the decoded `lhs`, `rhs`, and `result` fields instead of only
being an auxiliary target. Starting from
`checkpoints/latent_reasoning_sequence_process_smoke/best.pt` and training for 1000
steps gives:

| Metric | Step 1 | Step 1000 |
| --- | ---: | ---: |
| class value accuracy | `0.318` | `0.362` |
| class final accuracy | `0.125` | `0.094` |
| process-conditioned value accuracy | `0.000` | `0.324` |
| process-conditioned final accuracy | `0.000` | `0.125` |
| process digit accuracy | `0.557` | `0.577` |
| process carry accuracy | `0.871` | `0.875` |

Public 200-sample eval for
`checkpoints/latent_reasoning_sequence_process_conditioned_smoke/best.pt`:

| Preset | Class answer | Class trace | Process answer | Process trace |
| --- | ---: | ---: | ---: | ---: |
| `in_dist` | `0.015` | `0.005` | `0.010` | `0.000` |
| `larger_numbers` | `0.000` | `0.000` | `0.000` | `0.000` |
| `longer_expr` | `0.015` | `0.000` | `0.015` | `0.000` |
| `no_multiply` | `0.050` | `0.000` | `0.025` | `0.000` |
| `many_multiply` | `0.075` | `0.035` | `0.045` | `0.005` |
| `length_8` | `0.000` | `0.000` | `0.010` | `0.000` |
| `length_16` | `0.000` | `0.000` | `0.000` | `0.000` |

Interpretation: process-conditioned decoding is learnable, but it does not yet improve
end-to-end exact reasoning. The main remaining problem is that the predicted latent
states do not preserve enough operand identity and intermediate numeric state for exact
multi-step traces. The next useful direction is to make the latent reasoner recurrently
consume its previous predicted state and train step-local negatives that swap operands,
operators, and intermediate results.

The recurrent latent predictor makes each predicted reasoning-state embedding feed into
the next step prediction. This gives the latent sequence an explicit step-to-step state
path instead of predicting all reasoning states independently from parallel query slots.
Starting from the process-conditioned checkpoint with partial loading and training for
1000 steps gives:

| Metric | Step 1 | Step 1000 |
| --- | ---: | ---: |
| class value accuracy | `0.026` | `0.424` |
| class final accuracy | `0.016` | `0.203` |
| process-conditioned value accuracy | `0.029` | `0.382` |
| process-conditioned final accuracy | `0.031` | `0.172` |
| operation accuracy | `0.281` | `0.934` |

Public 200-sample eval for `checkpoints/latent_reasoning_sequence_recurrent_smoke/best.pt`:

| Preset | Class answer | Class trace | Process answer | Process trace |
| --- | ---: | ---: | ---: | ---: |
| `in_dist` | `0.030` | `0.005` | `0.040` | `0.005` |
| `larger_numbers` | `0.000` | `0.000` | `0.000` | `0.000` |
| `longer_expr` | `0.015` | `0.000` | `0.010` | `0.000` |
| `no_multiply` | `0.040` | `0.000` | `0.025` | `0.000` |
| `many_multiply` | `0.085` | `0.060` | `0.090` | `0.040` |
| `length_8` | `0.005` | `0.000` | `0.000` | `0.000` |
| `length_16` | `0.000` | `0.000` | `0.000` | `0.000` |

Interpretation: recurrence helps the latent value fields recover and gives a small
public answer improvement, especially in the process-conditioned path. It still does not
solve exact traces because the contrastive task is not yet hard enough: swapped operands,
wrong intermediate substitutions, and near-miss results are still too close in latent
space.

The step-local hard-negative update expands each latent step's contrastive negatives
from simple result/operator perturbations to near-miss reasoning states:

- result `+1` and `-1`
- wrong operator
- swapped `lhs`/`rhs`
- previous-step result substituted as `lhs`
- next-step result substituted as `rhs`
- previous/future step target at the same position

These negatives are encoded at the original reasoning-step position and the
hard-negative loss weight is increased from `0.2` to `0.5`. Starting from the recurrent
checkpoint and training for 1000 steps gives:

| Metric | Step 1 | Step 1000 |
| --- | ---: | ---: |
| class value accuracy | `0.403` | `0.467` |
| class final accuracy | `0.188` | `0.281` |
| process-conditioned value accuracy | `0.373` | `0.433` |
| process-conditioned final accuracy | `0.172` | `0.266` |
| operation accuracy | `0.935` | `0.941` |

Public 200-sample eval for `checkpoints/latent_reasoning_sequence_hardneg_smoke/best.pt`:

| Preset | Class answer | Class trace | Process answer | Process trace |
| --- | ---: | ---: | ---: | ---: |
| `in_dist` | `0.040` | `0.015` | `0.030` | `0.005` |
| `larger_numbers` | `0.000` | `0.000` | `0.000` | `0.000` |
| `longer_expr` | `0.015` | `0.000` | `0.015` | `0.005` |
| `no_multiply` | `0.090` | `0.000` | `0.105` | `0.000` |
| `many_multiply` | `0.115` | `0.075` | `0.115` | `0.060` |
| `length_8` | `0.000` | `0.000` | `0.000` | `0.000` |
| `length_16` | `0.000` | `0.000` | `0.000` | `0.000` |

Interpretation: hard negatives improve both internal final-value accuracy and public
answer accuracy on several splits. The model is still not exact because it needs a
state-transition objective that explicitly predicts the next expression state after each
reduction, not only isolated step tuples.

The state-transition objective adds auxiliary heads on each predicted latent reasoning
state. After every reduction, the latent state must predict the remaining expression's
value slots, operator slots, and active masks. This trains the representation to carry
the post-step expression state, not only the local `(lhs, op, rhs, result)` tuple.
Starting from `checkpoints/latent_reasoning_sequence_hardneg_smoke/best.pt` with partial
loading and training for 1000 steps gives:

| Metric | Step 1 | Step 1000 |
| --- | ---: | ---: |
| class value accuracy | `0.440` | `0.523` |
| class final accuracy | `0.312` | `0.484` |
| process-conditioned value accuracy | `0.416` | `0.498` |
| process-conditioned final accuracy | `0.312` | `0.500` |
| state value exact accuracy | `0.009` | `0.010` |
| state value active accuracy | `0.512` | `0.992` |
| state op accuracy | `0.242` | `0.496` |
| state op active accuracy | `0.571` | `0.992` |

Public 200-sample eval for
`checkpoints/latent_reasoning_sequence_state_scaled_smoke/best.pt`:

| Preset | Class answer | Class trace | Process answer | Process trace |
| --- | ---: | ---: | ---: | ---: |
| `in_dist` | `0.035` | `0.005` | `0.035` | `0.010` |
| `larger_numbers` | `0.000` | `0.000` | `0.000` | `0.000` |
| `longer_expr` | `0.020` | `0.000` | `0.025` | `0.005` |
| `no_multiply` | `0.100` | `0.000` | `0.125` | `0.000` |
| `many_multiply` | `0.115` | `0.070` | `0.115` | `0.070` |
| `length_8` | `0.000` | `0.000` | `0.000` | `0.000` |
| `length_16` | `0.000` | `0.000` | `0.000` | `0.000` |

Interpretation: the state-transition objective substantially improves internal final
value accuracy, and the external `no_multiply` process path improves to `0.125`. Exact
state-value reconstruction is still poor because it uses direct regression over large
integer states. The next refinement should decode state values with class or digit
heads, or use the predicted state directly to constrain the next step selection.

## Training

Smoke train:

```bash
python -m ssl_reasoner.train --steps 50 --train-size 512 --val-size 128
```

Train the variable reasoner:

```bash
bash scripts/train_variable_reasoner.sh
```

Train the latent reasoning sequence:

```bash
bash scripts/train_latent_reasoning_sequence.sh
```

Train standalone transition on reduction states:

```bash
bash scripts/train_standalone_transition_reductions.sh
```

Train the decomposed transition baseline and select checkpoints using decomposed
validation accuracy:

```bash
STANDALONE_TRANSITION_EVAL_MODE=decomposed \
CHECKPOINT=checkpoints/standalone_transition_factor/best.pt \
OUTPUT_DIR=checkpoints/standalone_transition_decomposed_smoke \
  bash scripts/train_standalone_transition_reductions.sh
```

Train with the larger-reduction curriculum:

```bash
STEPS=5000 TRAIN_SIZE=60000 VAL_SIZE=3000 \
TRAIN_CURRICULUM=large_reductions VAL_CURRICULUM=large_reductions \
CHECKPOINT=checkpoints/standalone_transition_factor/best.pt \
OUTPUT_DIR=checkpoints/standalone_transition_large \
  bash scripts/train_standalone_transition_reductions.sh
```

Rebuild the current best trace-ops checkpoint and write a reproduction report:

```bash
bash scripts/reproduce_trace_ops_head.sh
```

If the checkpoint already exists locally and you only want to regenerate the report:

```bash
SKIP_TRAIN=1 bash scripts/reproduce_trace_ops_head.sh
```

## Diagnostics

Latent answer diagnostics:

```bash
bash scripts/eval_latent_answer_diagnostics.sh
```

Important modes:

```text
target     true answer latent -> decoder -> answer tokens
pred       problem -> predicted answer latent -> decoder -> answer tokens
latent_nn  predicted answer latent -> nearest true answer latent over numeric values
```

Compositional split diagnostic:

```bash
scripts/eval_compositional_splits.sh checkpoints/predictor_structured_answer/best.pt
```

Available split curricula:

```text
seen_single, unseen_single, seen_mixed, unseen_mixed, compositional_train
```

## Known Limitations

- Not arbitrary-length math reasoning.
- Not general symbolic math.
- Not comparable to o1/4o-style reasoning models.
- Parentheses are handled by the safe parser fallback, not by a fully learned parenthesis
  policy.
- The strongest exact path still relies on structured constraints or deterministic
  parsing.
- Raw learned arithmetic over larger values remains the main open problem.
