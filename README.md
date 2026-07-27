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

PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/latent_reasoning_sequence_predslot_transition_smoke/best.pt \
  --latent-reasoning-slot-class-values \
  --preset in_dist

PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/latent_reasoning_sequence_predslot_transition_smoke/best.pt \
  --latent-reasoning-slot-process-values \
  --preset in_dist

PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/latent_reasoning_sequence_predslot_transition_smoke/best.pt \
  --latent-reasoning-copy-update-values \
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

The discrete state-value decoder replaces direct state-value regression with sign and
fixed-digit classification for each remaining expression value slot. This keeps the
state objective exact without creating a huge class tensor over every state slot.
Starting from `checkpoints/latent_reasoning_sequence_state_scaled_smoke/best.pt` with
partial loading and training for 1000 steps gives:

| Metric | Step 1 | Step 1000 |
| --- | ---: | ---: |
| class value accuracy | `0.472` | `0.480` |
| class final accuracy | `0.344` | `0.375` |
| process-conditioned value accuracy | `0.443` | `0.476` |
| process-conditioned final accuracy | `0.328` | `0.359` |
| state value exact accuracy | `0.000` | `0.061` |
| state value active accuracy | `0.991` | `0.996` |
| state op accuracy | `0.526` | `0.563` |
| state op active accuracy | `0.990` | `0.996` |

Public 200-sample eval for
`checkpoints/latent_reasoning_sequence_state_digit_smoke/best.pt`:

| Preset | Class answer | Class trace | Process answer | Process trace |
| --- | ---: | ---: | ---: | ---: |
| `in_dist` | `0.055` | `0.010` | `0.050` | `0.015` |
| `larger_numbers` | `0.000` | `0.000` | `0.000` | `0.000` |
| `longer_expr` | `0.035` | `0.000` | `0.025` | `0.000` |
| `no_multiply` | `0.090` | `0.000` | `0.110` | `0.000` |
| `many_multiply` | `0.125` | `0.085` | `0.110` | `0.070` |
| `length_8` | `0.000` | `0.000` | `0.000` | `0.000` |
| `length_16` | `0.000` | `0.000` | `0.000` | `0.000` |

Interpretation: discrete state decoding improves exact state-value reconstruction from
about `0.010` to `0.061` and gives the best class-mode public results so far. It lowers
the internal final-value metric compared with the regression checkpoint, so the next
step is to use predicted expression state as a constraint or feature for the next-step
decoder instead of treating it as only an auxiliary loss.

The state-conditioned step decoder feeds predicted expression-state distributions into
additional operation and value heads for each reasoning step. This tests whether the
model can use its predicted "remaining expression" state to choose the next operation
and decode the step values. Starting from
`checkpoints/latent_reasoning_sequence_state_digit_smoke/best.pt` with partial loading
and training for 1000 steps gives:

| Metric | Step 1 | Step 1000 |
| --- | ---: | ---: |
| class value accuracy | `0.469` | `0.508` |
| class final accuracy | `0.281` | `0.438` |
| process-conditioned value accuracy | `0.451` | `0.476` |
| process-conditioned final accuracy | `0.297` | `0.391` |
| state-conditioned op accuracy | `0.177` | `0.958` |
| state-conditioned value accuracy | `0.000` | `0.380` |
| state-conditioned final accuracy | `0.000` | `0.203` |
| state op accuracy | `0.365` | `0.605` |

Public 200-sample eval for
`checkpoints/latent_reasoning_sequence_state_conditioned_smoke/best.pt`:

| Preset | Class answer | Class trace | Process answer | Process trace | State answer | State trace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `in_dist` | `0.070` | `0.010` | `0.065` | `0.015` | `0.040` | `0.000` |
| `larger_numbers` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| `longer_expr` | `0.015` | `0.000` | `0.030` | `0.005` | `0.015` | `0.000` |
| `no_multiply` | `0.115` | `0.000` | `0.095` | `0.000` | `0.060` | `0.000` |
| `many_multiply` | `0.140` | `0.100` | `0.140` | `0.095` | `0.055` | `0.025` |
| `length_8` | `0.000` | `0.000` | `0.005` | `0.000` | `0.005` | `0.000` |
| `length_16` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |

Interpretation: state-conditioned operation decoding works well, but
state-conditioned value decoding is still weaker than the older class/process value
heads. Training the extra state-conditioned path still improves the shared latent
representation: class in-dist answer reaches `0.070` and many-multiply class trace
reaches `0.100`, the best public latent-sequence scores so far. The next step should
constrain value decoding using the predicted state slots rather than only concatenating
state probabilities into another value head.

The slot-selection decoder predicts which pre-reduction expression slots contain
`lhs`, `rhs`, and `op`, then decodes the result separately. For step `t`, the selector
uses the initial expression state for `t=0` and the predicted post-reduction state from
`t-1` for later steps. This directly tests whether the latent reasoner can stop
inventing operands and instead point to operands in the current expression state.
Starting from `checkpoints/latent_reasoning_sequence_state_conditioned_smoke/best.pt`
with partial loading and training for 1000 steps gives:

| Metric | Step 1 | Step 1000 |
| --- | ---: | ---: |
| slot lhs accuracy | `0.009` | `0.914` |
| slot rhs accuracy | `0.072` | `0.921` |
| slot op accuracy | `0.316` | `0.918` |
| slot pair accuracy | `0.000` | `0.911` |
| slot result accuracy | `0.000` | `0.194` |
| slot final accuracy | `0.000` | `0.188` |
| class final accuracy | `0.344` | `0.406` |
| process final accuracy | `0.375` | `0.453` |

Public 200-sample eval for `checkpoints/latent_reasoning_sequence_slot_smoke/best.pt`:

| Preset | Class answer | Class trace | Process answer | Process trace | Slot answer | Slot trace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `in_dist` | `0.065` | `0.020` | `0.065` | `0.010` | `0.020` | `0.000` |
| `larger_numbers` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| `longer_expr` | `0.040` | `0.000` | `0.025` | `0.000` | `0.005` | `0.000` |
| `no_multiply` | `0.075` | `0.000` | `0.100` | `0.000` | `0.025` | `0.000` |
| `many_multiply` | `0.130` | `0.095` | `0.110` | `0.070` | `0.045` | `0.000` |
| `length_8` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| `length_16` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |

Interpretation: slot selection works: the model learns to pick the correct operand and
operator slots with about `0.91` pair accuracy. The slot inference path is still worse
because result decoding from the selected slots is weak. This isolates the next
bottleneck: keep the slot selector, but replace the free slot result head with a
selected-operand arithmetic transition head.

The selected-operand transition head replaces the free slot result head with a learned
result predictor conditioned on the selected `lhs`, selected `rhs`, selected `op`, and
the latent step embedding. This is still a neural prediction path, but it gives the
result module direct access to the operands instead of asking it to infer the result
from an unconstrained latent state. Starting from
`checkpoints/latent_reasoning_sequence_slot_smoke/best.pt` with partial loading and
training for 1000 steps gives:

| Metric | Step 1 | Step 1000 |
| --- | ---: | ---: |
| slot pair accuracy | `0.876` | `0.907` |
| free slot result accuracy | `0.115` | `0.184` |
| free slot final accuracy | `0.141` | `0.141` |
| transition result accuracy | `0.002` | `0.452` |
| transition final accuracy | `0.000` | `0.328` |
| class final accuracy | `0.359` | `0.406` |
| process final accuracy | `0.453` | `0.438` |

Public 200-sample eval for
`checkpoints/latent_reasoning_sequence_slot_transition_smoke/best.pt`:

| Preset | Class answer | Class trace | Slot answer | Slot trace | Slot-transition answer | Slot-transition trace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `in_dist` | `0.040` | `0.005` | `0.020` | `0.000` | `0.000` | `0.000` |
| `larger_numbers` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| `longer_expr` | `0.010` | `0.000` | `0.010` | `0.000` | `0.000` | `0.000` |
| `no_multiply` | `0.075` | `0.000` | `0.050` | `0.000` | `0.000` | `0.000` |
| `many_multiply` | `0.120` | `0.075` | `0.050` | `0.000` | `0.000` | `0.000` |
| `length_8` | `0.000` | `0.000` | `0.010` | `0.000` | `0.000` | `0.000` |
| `length_16` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |

Interpretation: the selected-operand transition head fixes the teacher-forced result
bottleneck internally: transition result accuracy improves from about `0.19` to `0.45`.
However, the standalone slot-transition inference path is still unusable because it
depends on predicted state values and predicted slot choices at every step. The next
fix is to train the transition head under its actual inference distribution: feed it
detached predicted slot-selected operands during training, or add a hybrid fallback that
uses the slot selector for operands and the stronger class/process result head for the
result.

The predicted-slot transition objective adds that inference-like training signal. In
addition to the teacher-forced transition loss above, it feeds detached predicted
slot-selected `lhs`, `rhs`, and `op` into the same transition head and trains it toward
the true result. Starting from
`checkpoints/latent_reasoning_sequence_slot_transition_smoke/best.pt` and training for
1000 more steps gives:

| Metric | Step 1 | Step 1000 |
| --- | ---: | ---: |
| slot pair accuracy | `0.840` | `0.920` |
| teacher-forced transition result accuracy | `0.004` | `0.339` |
| teacher-forced transition final accuracy | `0.000` | `0.188` |
| predicted-slot transition result accuracy | `0.000` | `0.196` |
| predicted-slot transition final accuracy | `0.000` | `0.156` |
| class final accuracy | `0.375` | `0.438` |
| process final accuracy | `0.328` | `0.484` |

Public 200-sample eval for
`checkpoints/latent_reasoning_sequence_predslot_transition_smoke/best.pt`:

| Preset | Class answer | Class trace | Process answer | Process trace | Slot-transition answer | Slot-transition trace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `in_dist` | `0.065` | `0.015` | `0.050` | `0.015` | `0.030` | `0.000` |
| `larger_numbers` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| `longer_expr` | `0.030` | `0.000` | `0.020` | `0.000` | `0.000` | `0.000` |
| `no_multiply` | `0.095` | `0.000` | `0.060` | `0.000` | `0.045` | `0.000` |
| `many_multiply` | `0.150` | `0.090` | `0.125` | `0.080` | `0.060` | `0.000` |
| `length_8` | `0.005` | `0.000` | `0.000` | `0.000` | `0.025` | `0.000` |
| `length_16` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |

Two hybrid inference modes were added to separate the slot-selection problem from the
result-decoding problem:

| Mode | Operand/source fields | Result source |
| --- | --- | --- |
| `--latent-reasoning-slot-class-values` | selected `lhs`, `rhs`, and `op` slots | bounded class value head |
| `--latent-reasoning-slot-process-values` | selected `lhs`, `rhs`, and `op` slots | process-conditioned value head |
| `--latent-reasoning-slot-digit-values` | selected `lhs`, `rhs`, and `op` slots | sign/digit/carry result head |
| `--latent-reasoning-copy-update-values` | rolled copy/update state with selected slots | selected-operand transition head |

Public 200-sample eval for the same checkpoint:

| Preset | Slot-class answer | Slot-class trace | Slot-process answer | Slot-process trace |
| --- | ---: | ---: | ---: | ---: |
| `in_dist` | `0.065` | `0.000` | `0.050` | `0.000` |
| `larger_numbers` | `0.000` | `0.000` | `0.000` | `0.000` |
| `longer_expr` | `0.030` | `0.000` | `0.020` | `0.000` |
| `no_multiply` | `0.095` | `0.000` | `0.060` | `0.000` |
| `many_multiply` | `0.150` | `0.000` | `0.125` | `0.000` |
| `length_8` | `0.005` | `0.000` | `0.000` | `0.000` |
| `length_16` | `0.000` | `0.000` | `0.000` | `0.000` |

Interpretation: training on predicted slot-selected operands makes the slot-transition
path nonzero externally, but it is still weaker than class/process decoding. The hybrid
paths recover class/process answer accuracy while using slot-selected operands, but they
do not produce valid full traces. The main remaining issue is predicted expression-state
value quality: the selector can point to slots, but the values inside those predicted
slots are often wrong. The next improvement should train the expression-state value
decoder much more strongly before relying on it for slot-transition inference.

The state-focused objective increases supervision on the expression-state decoder and
adds losses exactly where inference uses the state:

```text
post-step state slot at reduction position -> result value
pre-step selected lhs/rhs slots            -> lhs/rhs values
pre-step selected op slot                  -> op id
```

Starting from
`checkpoints/latent_reasoning_sequence_predslot_transition_smoke/best.pt` with partial
loading and training for 1000 steps gives:

| Metric | Step 1 | Step 1000 |
| --- | ---: | ---: |
| slot pair accuracy | `0.889` | `0.932` |
| state value accuracy | `0.043` | `0.071` |
| result-in-state value accuracy | `0.022` | `0.060` |
| pre-slot operand value accuracy | `0.044` | `0.073` |
| pre-slot op accuracy | `0.707` | `0.871` |
| slot-transition result accuracy | `0.351` | `0.432` |
| predicted-slot transition result accuracy | `0.213` | `0.244` |
| process final accuracy | `0.438` | `0.500` |

Public 200-sample eval for
`checkpoints/latent_reasoning_sequence_state_focused_smoke/best.pt`:

| Preset | Class answer | Class trace | Slot-class answer | Slot-class trace | Slot-transition answer | Slot-transition trace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `in_dist` | `0.055` | `0.010` | `0.055` | `0.000` | `0.050` | `0.000` |
| `larger_numbers` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| `longer_expr` | `0.030` | `0.000` | `0.030` | `0.000` | `0.030` | `0.000` |
| `no_multiply` | `0.120` | `0.000` | `0.120` | `0.000` | `0.090` | `0.000` |
| `many_multiply` | `0.150` | `0.090` | `0.150` | `0.000` | `0.100` | `0.000` |
| `length_8` | `0.015` | `0.000` | `0.015` | `0.000` | `0.010` | `0.000` |
| `length_16` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |

Interpretation: the focused objective moves the right internal metrics, especially
slot-transition result accuracy and pre-slot op recovery, but exact decoded state values
remain far too low. The next architectural bottleneck is numeric state representation:
the current fixed digit/class heads do not preserve exact intermediate values well enough
for multi-step latent trace generation.

The factorized slot-digit transition head predicts result sign, result digits, and
per-place carry/borrow from the selected operands, selected operator, and latent step
embedding. This keeps the arithmetic path learned, but gives the model a structured
numeric representation instead of a single bounded integer class. Starting from
`checkpoints/latent_reasoning_sequence_state_focused_smoke/best.pt` with partial loading
and training for 1000 steps gives:

| Metric | Step 1 | Step 1000 |
| --- | ---: | ---: |
| slot pair accuracy | `0.938` | `0.926` |
| slot-transition result accuracy | `0.423` | `0.568` |
| predicted-slot transition result accuracy | `0.245` | `0.292` |
| slot-digit result accuracy | `0.000` | `0.249` |
| slot-digit final accuracy | `0.000` | `0.109` |
| slot-digit carry accuracy | `0.005` | `0.941` |
| predicted-slot digit result accuracy | `0.000` | `0.150` |
| predicted-slot digit carry accuracy | `0.004` | `0.897` |
| predicted-slot digit arithmetic-valid accuracy | `0.000` | `0.210` |
| state value accuracy | `0.054` | `0.083` |

Public 200-sample eval for
`checkpoints/latent_reasoning_sequence_slot_digit_smoke/best.pt`:

| Preset | Class answer | Class trace | Slot-transition answer | Slot-transition trace | Slot-digit answer | Slot-digit trace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `in_dist` | `0.065` | `0.015` | `0.040` | `0.000` | `0.015` | `0.000` |
| `larger_numbers` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| `longer_expr` | `0.050` | `0.000` | `0.020` | `0.000` | `0.005` | `0.000` |
| `no_multiply` | `0.115` | `0.000` | `0.080` | `0.000` | `0.070` | `0.000` |
| `many_multiply` | `0.135` | `0.085` | `0.070` | `0.000` | `0.040` | `0.000` |
| `length_8` | `0.005` | `0.000` | `0.005` | `0.000` | `0.005` | `0.000` |
| `length_16` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |

Interpretation: the factorized path learns carry/borrow structure quickly and improves
teacher-forced slot-transition metrics, but the new slot-digit inference mode is not yet
competitive. Exact result digits are still the limiting factor, so this path needs a
longer curriculum or staged warmup where digit/carry accuracy is trained before it is
used as the main inference decoder.

Slot-digit curriculum weights can be changed from the training CLI:

```bash
--slot-digit-weight 4.0 \
--slot-digit-carry-weight 1.0 \
--predicted-slot-digit-weight 0.25 \
--predicted-slot-digit-carry-weight 0.25
```

A teacher-forced warmup from
`checkpoints/latent_reasoning_sequence_slot_digit_smoke/best.pt` improved the clean
slot-digit path:

| Metric | Before | Teacher warmup |
| --- | ---: | ---: |
| slot-digit result accuracy | `0.249` | `0.333` |
| slot-digit final accuracy | `0.109` | `0.172` |
| slot-digit carry accuracy | `0.941` | `0.945` |
| predicted-slot digit result accuracy | `0.150` | `0.142` |
| predicted-slot digit arithmetic-valid accuracy | `0.210` | `0.339` |
| slot-transition result accuracy | `0.568` | `0.587` |

A second phase with high predicted-slot digit weight did not improve the predicted-slot
digit path:

| Metric | Teacher warmup | Predicted-slot warmup |
| --- | ---: | ---: |
| slot-digit result accuracy | `0.333` | `0.186` |
| predicted-slot digit result accuracy | `0.142` | `0.131` |
| predicted-slot digit arithmetic-valid accuracy | `0.339` | `0.159` |
| slot-transition result accuracy | `0.587` | `0.582` |

Public 200-sample slot-digit eval:

| Checkpoint | `in_dist` answer | `no_multiply` answer | `many_multiply` answer | trace |
| --- | ---: | ---: | ---: | ---: |
| slot-digit smoke | `0.015` | `0.070` | `0.040` | `0.000` |
| teacher warmup | `0.025` | `0.035` | `0.040` | `0.000` |
| predicted-slot warmup | `0.020` | `0.035` | `0.040` | `0.000` |

Interpretation: teacher-forced warmup works for clean operands, but hard switching to
predicted operands is too abrupt. The predicted state values/selected operands are still
too noisy, so the next training change should ramp predicted-slot digit weight gradually
or freeze the selector/state decoder while training the digit head.

A frozen-head stage was added for that isolation experiment:

```bash
--stages latent_reasoning_slot_digit_head
```

This freezes the latent predictor, state decoder, and slot selector, then trains only
`latent_reasoning_sequence.slot_digit_result_head`. Starting from the teacher-warmup
checkpoint with high predicted-slot digit weight gave:

| Metric | Teacher warmup | Frozen digit-head |
| --- | ---: | ---: |
| slot pair accuracy | `0.943` | `0.880` |
| slot-digit result accuracy | `0.333` | `0.156` |
| predicted-slot digit result accuracy | `0.142` | `0.112` |
| predicted-slot digit arithmetic-valid accuracy | `0.339` | `0.156` |
| slot-digit carry accuracy | `0.945` | `0.919` |

Public 200-sample eval for
`checkpoints/latent_reasoning_slot_digit_head_predicted/best.pt`:

| Mode | `in_dist` answer | `no_multiply` answer | `many_multiply` answer | trace |
| --- | ---: | ---: | ---: | ---: |
| slot-digit | `0.020` | `0.030` | `0.040` | `0.000` |
| slot-transition | `0.040` | `0.080` | `0.080` | `0.000` |
| class | `0.060` | `0.130` | `0.120` | up to `0.070` |

Interpretation: freezing the upstream path did not improve predicted-slot digit
decoding. The bottleneck is therefore not mainly moving-target training instability; it
is the quality of the predicted operands/state slots themselves. The next useful change
should directly improve state-slot numeric identity, for example with a copy/update
state model that preserves unreduced values and writes only the reduced result slot.

The copy/update state path is now implemented as a constrained diagnostic. Instead of
regenerating every next-state slot, it applies this update rule:

```text
pre-state values + selected lhs/rhs + result
  -> copy values before lhs
  -> write result into lhs slot
  -> shift values after rhs left

pre-state ops + selected op
  -> copy ops before selected op
  -> shift ops after selected op left
```

The model reports both oracle and predicted copy/update metrics:

| Metric | Diagnostic value |
| --- | ---: |
| oracle copy/update value accuracy | `1.000` |
| oracle copy/update op accuracy | `1.000` |
| predicted copy/update value accuracy | `0.233` |
| predicted copy/update op accuracy | `0.595` |
| direct decoded state value accuracy | `0.060` |
| copy/update value improvement | `+0.173` |

The copy/update rule is also exposed as a rollout inference mode:

```bash
PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/latent_reasoning_sequence_predslot_transition_smoke/best.pt \
  --latent-reasoning-copy-update-values \
  --preset all
```

This prints copy/update-specific diagnostics:

```text
copy_update_lhs_exact
copy_update_rhs_exact
copy_update_op_exact
copy_update_operand_pair_exact
copy_update_slot_triple_exact
copy_update_result_exact
copy_update_arithmetic_valid
copy_update_post_state_exact
```

Interpretation: the structural update rule works exactly when given correct slots and
results, and even with predicted slots/results it preserves state values much better
than direct state decoding. The remaining issue is now sharply isolated: improve the
predicted action/result quality that feeds the copy/update rule.

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

Fine-tune the latent reasoning sequence with copy/update rollout-focused loss weights:

```bash
CHECKPOINT=checkpoints/latent_reasoning_sequence_predslot_transition_smoke/best.pt \
  bash scripts/train_copy_update_rollout.sh
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
