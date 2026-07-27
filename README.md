# SSL Reasoner

SSL Reasoner is a math-only research prototype for testing whether a model can learn
structured arithmetic reasoning through latent and joint-embedding (JEPA) style training.
It is **not** a pretrained LLM and does not call external math tools.

Scope: synthetic integer arithmetic with `+`, `-`, `*`, and safe parenthesized
expression parsing.

## Current status

Three solver paths, from most to least reliable:

| Path | Status |
| --- | --- |
| Structured parser / trace renderer | Reliable for supported expressions; the deterministic reference path. |
| Learned operation / variable trace path | Strong when constrained to legal reduction order with candidate-scored arithmetic. |
| Raw learned arithmetic transition | Experimental; still weak on larger numbers and long chains. |

The best practical solver is the structured trace path. The standalone learned transition
handles many in-distribution reductions, but **larger-number arithmetic is unsolved**.

## Architecture

### Original latent answer path

```text
problem text
  -> tokenizer + math feature extractor
  -> problem encoder
  -> answer-slot predictor
  -> predicted answer latent slots
  -> scratch transformer readout decoder
  -> answer tokens
```

The decoder is trained from scratch (not a pretrained LM). Diagnostics show the decoder
reads good target latents well, but the *predicted* answer latents are often not accurate
enough.

### Structured reasoning path

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

### Latent reasoning sequence

The joint-embedding reasoning path: predict a recurrent sequence of latent reasoning
states instead of one answer latent.

```text
problem math features
  -> latent reasoning sequence predictor
  -> z_step1, z_step2, ..., z_final
  -> structured decoder heads
  -> lhs/op/rhs/result fields + final answer
```

Target step embeddings are built from the structured trace fields (`lhs, op, rhs, result,
step position -> z_target_step`). Training uses latent matching, in-batch contrastive loss
with hard negatives, active/stop prediction, operation decoding, bounded value-class
decoding, and a digit value decoder.

### Experimental standalone transition

Learns a single arithmetic reduction `(lhs, op, rhs) -> result` under several output
encodings:

| Mode | Description |
| --- | --- |
| `class` | Bounded integer class from `-1000` to `10000` (strongest so far). |
| `digit` | Sign plus fixed decimal digits. |
| `factor` | Sign plus magnitude bucket plus offset. |
| `decomposed` | Place-wise output digit prediction from lhs/rhs digits and operator features. |
| `hybrid` | `class` mode with a confidence/boundary fallback. |

## Quick start

```bash
pip install -e .
```

```bash
bash scripts/solve_math.sh "What is 2+3*4?"          # solve
bash scripts/solve_math.sh --debug "What is 2+3*4?"  # debug info
bash scripts/solve_math.sh --json "What is 2+3*4?"   # JSON output
```

Show structured reasoning steps:

```bash
bash scripts/solve_math.sh --debug-reasoning "What is 49-5+19?"
```

```text
answer: 63
mode: operation
reasoning_order: left_first
step1: lhs=49 op=- rhs=5 result=44
step2: lhs=44 op=+ rhs=19 result=63
trace: 49-5=44,44+19=63,63
```

## Training

```bash
python -m ssl_reasoner.train --steps 50 --train-size 512 --val-size 128  # smoke
bash scripts/train_variable_reasoner.sh                                   # variable reasoner
bash scripts/train_latent_reasoning_sequence.sh                           # latent sequence
bash scripts/train_standalone_transition_reductions.sh                    # standalone transition
```

Fine-tune the latent sequence with copy/update rollout-focused loss weights:

```bash
CHECKPOINT=checkpoints/latent_reasoning_sequence_predslot_transition_smoke/best.pt \
  bash scripts/train_copy_update_rollout.sh
```

Larger-reduction curriculum, and a decomposed-transition baseline selected on decomposed
validation accuracy:

```bash
STEPS=5000 TRAIN_SIZE=60000 VAL_SIZE=3000 \
TRAIN_CURRICULUM=large_reductions VAL_CURRICULUM=large_reductions \
CHECKPOINT=checkpoints/standalone_transition_factor/best.pt \
OUTPUT_DIR=checkpoints/standalone_transition_large \
  bash scripts/train_standalone_transition_reductions.sh

STANDALONE_TRANSITION_EVAL_MODE=decomposed \
CHECKPOINT=checkpoints/standalone_transition_factor/best.pt \
OUTPUT_DIR=checkpoints/standalone_transition_decomposed_smoke \
  bash scripts/train_standalone_transition_reductions.sh
```

Rebuild the best trace-ops checkpoint and write a reproduction report (add `SKIP_TRAIN=1`
to only regenerate the report):

```bash
bash scripts/reproduce_trace_ops_head.sh
```

## Evaluation

```bash
bash scripts/eval_math_solver.sh                       # default solver
CURRICULUM=mixed_only bash scripts/eval_math_solver.sh # another curriculum
bash scripts/eval_variable_reasoning.sh                # variable-length reasoning
python -m ssl_reasoner.eval --checkpoint checkpoints/best.pt --samples 500
```

Evaluate standalone transition and latent-sequence decode modes via
`ssl_reasoner.eval_variable_reasoning` (`--preset all` runs every OOD preset):

```bash
PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/standalone_transition_factor/best.pt \
  --standalone-factor-values --preset all

PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/latent_reasoning_sequence/best.pt \
  --latent-reasoning-values --preset in_dist

PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/latent_reasoning_sequence_predslot_transition_smoke/best.pt \
  --latent-reasoning-copy-update-values --preset all
```

Decode-mode flags: `--standalone-learned-values`, `--standalone-factor-values`,
`--standalone-decomposed-values`, `--latent-reasoning-values`,
`--latent-reasoning-digit-values`, `--latent-reasoning-process-values`,
`--latent-reasoning-slot-class-values`, `--latent-reasoning-slot-process-values`,
`--latent-reasoning-copy-update-values`.

## Diagnostics

```bash
bash scripts/eval_latent_answer_diagnostics.sh                                  # latent answer
scripts/eval_compositional_splits.sh checkpoints/predictor_structured_answer/best.pt
```

Latent answer modes:

```text
target     true answer latent -> decoder -> answer tokens
pred       problem -> predicted answer latent -> decoder -> answer tokens
latent_nn  predicted answer latent -> nearest true answer latent over numeric values
```

Split curricula: `seen_single`, `unseen_single`, `seen_mixed`, `unseen_mixed`,
`compositional_train`.

## Checkpoints

```bash
bash scripts/download_current_best.sh
```

- Release: `https://github.com/akshai0296/ssl_reasoner/releases/tag/math-solver-v0.1.0`
- Manifest: `reports/checkpoint_manifest.json`

## Results

### Default math solver

Fixed 500-sample results for the current local best checkpoint:

| Curriculum | `mixed` | `mixed_only` | `seen_mixed` | `unseen_mixed` |
| --- | ---: | ---: | ---: | ---: |
| accuracy | `1.000` | `0.990` | `1.000` | `0.998` |

### Variable-length structured reasoning

The deterministic trace path is exact (`answer exact = 1.000`, `trace exact = 1.000`).
The learned dynamic pointer with candidate-scored arithmetic is strong under legal
reduction constraints and degrades on long expressions without them:

| Preset | Constrained | Unconstrained |
| --- | ---: | ---: |
| `in_dist` / `larger_numbers` / `longer_expr` / `many_multiply` | `1.000` | ≥ `1.000` |
| `no_multiply` | `1.000` | `0.996` |
| `length_8` | `0.988` | `0.984` |
| `length_16` | `0.750` | `0.448` |

### Standalone learned arithmetic (200-sample)

| Checkpoint / mode | In-dist | Larger | Longer | No mult | Many mult | Len 8 | Len 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `factor`, class | `0.750` | `0.005` | `0.700` | `1.000` | `0.620` | `0.395` | `0.125` |
| `factor`, hybrid | `0.690` | `0.000` | `0.645` | `1.000` | `0.565` | `0.305` | `0.070` |
| `decomposed_smoke`, decomposed | `0.475` | `0.005` | `0.390` | `0.720` | `0.370` | `0.150` | `0.045` |

Takeaways: the bounded `class` head is the strongest standalone transition; larger-number
generalization is effectively unsolved; digit/factor/decomposed formats alone are not
enough. The next step is explicit carry/borrow supervision for `+`/`-` and partial-product
accumulation for `*`.

### Latent reasoning sequence

A 5000-step run trains the architecture, but exact value decoding stays weak:

| Metric | Step 1 | Step 5000 |
| --- | ---: | ---: |
| active accuracy | `0.606` | `1.000` |
| operation accuracy | `0.284` | `0.929` |
| value accuracy | `0.000` | `0.310` |
| final accuracy | `0.000` | `0.188` |

Findings from the iteration series (full history in git log):

- **Process-conditioned decoding** is learnable but does not by itself lift end-to-end
  exact reasoning.
- **Recurrent state** (feeding each predicted state into the next step) plus **step-local
  hard negatives** (swapped operands, wrong operator, near-miss results) improve internal
  metrics and give small public gains, but do not solve exact traces.
- A **factorized slot-digit transition head** (result sign, digits, per-place carry/borrow)
  gives a structured numeric representation instead of one bounded class. It reaches high
  carry accuracy but still low exact value accuracy.
- The bottleneck is the **numeric identity of the predicted operand/state slots**, not
  training instability — freezing the upstream path did not help predicted-slot decoding.

**Copy/update state rule.** Instead of regenerating every next-state slot, apply a
structural update: copy unreduced values, write the result into the reduced slot, shift
the rest left (same for ops). This isolates the remaining problem:

| Metric | Value |
| --- | ---: |
| oracle copy/update value / op accuracy | `1.000` / `1.000` |
| predicted copy/update value / op accuracy | `0.233` / `0.595` |
| direct decoded state value accuracy | `0.060` |
| copy/update value improvement | `+0.173` |

```bash
PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/latent_reasoning_sequence_predslot_transition_smoke/best.pt \
  --latent-reasoning-copy-update-values --preset all
```

The rule is exact given correct slots/results, and preserves state far better than direct
decoding even with predicted inputs. The remaining problem is sharply isolated: **improve
the predicted action/result quality that feeds the copy/update rule.**

## Known limitations

- Not arbitrary-length math reasoning, not general symbolic math, not comparable to
  o1/4o-style reasoning models.
- Parentheses use the safe parser fallback, not a fully learned parenthesis policy.
- The strongest exact path still relies on structured constraints or deterministic parsing.
- Raw learned arithmetic over larger values remains the main open problem.
