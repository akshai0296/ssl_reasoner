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
prediction, operation decoding, and value decoding. This is now implemented as a
trainable baseline, but it is not yet the strongest solver.

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
the bounded whole-value decoder is not accurate enough. The next architectural step is a
digit/carry-aware decoder for latent reasoning states.

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
