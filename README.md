# SSL Reasoner

A fresh math-only prototype for a self-supervised latent reasoner.

The first target is intentionally narrow and math-only:

1. Generate synthetic arithmetic problems.
2. Stage 0: warm up an answer target encoder + deterministic readout.
3. Stage 1: train a problem encoder + sequence predictor to match EMA answer slots.
4. Stage 2: freeze the predictor and train the deterministic readout on predicted slots.
5. Evaluate exact-match numeric answers.

The original answer readout predicts all token positions and answer length in one forward
pass. The newer step-by-step reasoning decoder is autoregressive, but it is still trained
from scratch inside this repo and is not a pretrained LLM.

## Architecture Flow

The model has two answer paths. The original latent readout path predicts an answer
embedding and decodes it into tokens with a decoder trained from scratch:

```text
problem text
  -> tokenizer + math feature extractor
  -> problem encoder
  -> answer-slot predictor
  -> predicted answer latent slots
  -> ParallelReadoutDecoder
  -> answer tokens
```

The decoder is not a pretrained language model. It is a small local transformer decoder
that learns to map answer latent slots to numeric answer strings during Stage 0, Stage 2,
and Stage 3.

The current best math solver uses the interpretable trace-operation path:

```text
problem text
  -> tokenizer + math feature extractor
  -> problem encoder
  -> trace predictor
  -> trace operation head
  -> predicted operation ids
  -> deterministic operation executor
  -> final answer
```

Operation ids are:

```text
0 = none
1 = +
2 = -
3 = *
```

Example:

```text
input:  "What is 2+3*4?"
ops:    [3, 1]
trace:  3*4 = 12, then 2+12 = 14
answer: 14
```

With debug output:

```bash
bash scripts/solve_math.sh --debug "What is 2+3*4?"
```

You can inspect:

```text
answer, mode, predicted operation ids, operation confidences, operation answer, readout answer
```

The learned operation head is trained for one-step and two-step arithmetic expressions.
Longer expressions, such as `30+10+5-5`, are not passed through the two-step operation
head as if they were a prefix. They are marked as `mode=parsed_expression` and handled
by the deterministic parser fallback.

## Latent Answer Diagnostics

The direct latent readout path is currently much weaker than the operation path. Use:

```bash
bash scripts/eval_latent_answer_diagnostics.sh
```

Important modes:

```text
target     true answer latent -> decoder -> answer tokens
pred       problem -> predicted answer latent -> decoder -> answer tokens
latent_nn  predicted answer latent -> nearest true answer latent over numeric values
```

On the current checkpoint, `target` is high while `pred` and `latent_nn` are much
lower. That means the decoder can read good answer latents, but the problem
encoder/predictor is not reliably landing on the correct answer latent.

There is also an optional numeric value head on predicted answer slots:

```bash
bash scripts/train_answer_value_head.sh
CHECKPOINT=checkpoints/answer_value_head/best.pt MODES="answer_value" \
  bash scripts/eval_latent_answer_diagnostics.sh
```

In the current experiment, that head also stays near the direct readout accuracy, which
supports the same conclusion: predicted answer latents are the bottleneck.

To train the problem encoder/predictor harder toward the correct answer latent
neighborhood, use:

```bash
bash scripts/finetune_predictor_answer_latent.sh
CHECKPOINT=checkpoints/predictor_answer_latent/best.pt \
  bash scripts/eval_latent_answer_diagnostics.sh
```

This fine-tune adds a supervised answer-latent contrastive loss. Samples with the same
numeric answer are treated as positives instead of false negatives, while different
answers remain negatives.

## Mixed Trace State Path

The operation solver uses learned operation predictions, then applies those operations
to the expression from the prompt. To inspect the more fully learned path, use the trace
state solver. It answers from the model's predicted intermediate/final trace values:

```bash
bash scripts/finetune_trace_state_head.sh
bash scripts/eval_trace_state_solver.sh
```

For a mixed expression such as `20+1*0`, the trace state target is:

```text
1*0=0 20+0=20
```

The state solver reads the final predicted state (`20`) instead of recomputing it from
the prompt.

The continuous state variant avoids the fixed value-class range by regressing the
intermediate and final numeric states directly:

```bash
bash scripts/train_trace_state_regression.sh
bash scripts/eval_trace_state_regression.sh
```

To fine-tune the trace predictor and continuous state head together:

```bash
bash scripts/finetune_trace_state_joint.sh
CHECKPOINT=checkpoints/trace_state_joint/best.pt bash scripts/eval_trace_state_regression.sh
```

Current result: this improves the mixed direct readout somewhat, but the learned
continuous final-state path is still not exact enough to replace the operation solver.

The step-wise state solver uses an operator-conditioned transition cell with learned
execution-order routing:

```bash
bash scripts/train_step_state_solver.sh
bash scripts/eval_step_state_solver.sh
CURRICULUM=mixed_only bash scripts/eval_step_state_solver.sh
```

Current result with `checkpoints/step_state_solver_mixed_only/best.pt`: `1.000` on
`mixed` and `1.000` on `mixed_only` for the supported one- and two-operation expression
format. This path answers from predicted intermediate/final states rather than the
decoder's answer-token readout.

The value-conditioned latent bridge connects that exact state result back to answer
latent slots and the decoder:

```bash
bash scripts/train_value_conditioned_joint.sh
bash scripts/eval_value_conditioned_latent.sh
CURRICULUM=mixed_only bash scripts/eval_value_conditioned_latent.sh
```

Current result with `checkpoints/value_conditioned_joint_wide/best.pt`: `0.974` on
`mixed` and `0.876` on `mixed_only`. This is much better than the original direct
answer-latent decoder path, but the step-state solver remains the exact path.

## Step-by-Step Reasoning Sequence

This path predicts structured reasoning states first, then renders those fields into
compact trace tokens. This avoids asking a character decoder to guess operands and
intermediate values directly.

```text
problem text
  -> tokenizer + math feature extractor
  -> problem encoder
  -> step-state solver
  -> predicted execution order + intermediate/final state values
  -> structured fields: lhs, op, rhs, result, final
  -> deterministic trace renderer
  -> trace text + final answer
```

For example:

```text
input:  "What is 20+1*0?"
states: [1*0=0, 20+0=20, 20]
tokens: "1*0=0,20+0=20,20"
```

Train and evaluate it with:

```bash
bash scripts/train_reasoning_sequence.sh
bash scripts/eval_reasoning_sequence.sh
CURRICULUM=mixed_only bash scripts/eval_reasoning_sequence.sh
```

Inspect the structured reasoning fields for one query with:

```bash
bash scripts/solve_math.sh --debug-reasoning "What is 49-5+19?"
```

When `--debug-reasoning` is used without `CHECKPOINT=...`, the script defaults to
`checkpoints/step_state_solver_mixed_only/best.pt` because that checkpoint contains the
trained step-state head needed for stable reasoning traces.

Example debug output:

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

The current checkpoint is:

```text
checkpoints/reasoning_sequence/best.pt
```

Current 500-sample result:

| Curriculum | Exact trace+answer match |
| --- | ---: |
| `mixed` | `1.000` |
| `mixed_only` | `1.000` |

The scratch autoregressive decoder still exists for experiments, but the default
reasoning-sequence eval now uses structured state decoding before text rendering. That is
the stronger path because exact fields are checked before producing the final trace
string.

## Variable-Length Structured Reasoning

For expressions longer than the learned two-step state solver, the debug/eval path now
builds a variable-length structured trace with operator precedence:

```bash
bash scripts/solve_math.sh --debug-reasoning "What is 2+3*4-1?"
```

Example:

```text
answer: 13
mode: parsed_expression
problem: What is 2+3*4-1?
reasoning_order: variable_precedence
step1: lhs=3 op=* rhs=4 result=12
step2: lhs=2 op=+ rhs=12 result=14
step3: lhs=14 op=- rhs=1 result=13
final: 13
trace: 3*4=12,2+12=14,14-1=13,13
```

Evaluate the variable-length trace path with:

```bash
bash scripts/eval_variable_reasoning.sh
```

Current 500-sample `multi_step` result:

| Metric | Score |
| --- | ---: |
| answer exact | `1.000` |
| trace exact | `1.000` |

This variable-length path is structured and deterministic. It establishes the multi-step
trace contract and eval target.

A first learned recurrent transition baseline is also available:

```bash
bash scripts/train_variable_reasoner.sh
CHECKPOINT=checkpoints/variable_reasoner/best.pt \
  bash scripts/eval_variable_reasoning.sh --learned
CHECKPOINT=checkpoints/variable_reasoner/best.pt \
  bash scripts/eval_variable_reasoning.sh --learned --learned-values
CHECKPOINT=checkpoints/variable_reasoner/best.pt \
  bash scripts/eval_variable_reasoning.sh --learned --learned-values --preset all
CHECKPOINT=checkpoints/variable_reasoner/best.pt \
  bash scripts/eval_variable_reasoning.sh --learned --learned-values --unconstrained --preset all
```

Current learned baseline result:

| Metric | Score |
| --- | ---: |
| answer exact | `1.000` |
| learned trace exact | `1.000` |
| learned trace equivalent exact | `1.000` |
| learned step value exact | `1.000` |
| learned final value exact | `1.000` |

The answer remains exact because the solver still uses the parser fallback for final
answers on longer expressions. The learned recurrent trace head now predicts which
adjacent operation to reduce next with a dynamic pointer over the currently remaining
operators. The `--learned-values` path now uses a structured transition head: it scores
the three candidate results `lhs+rhs`, `lhs-rhs`, and `lhs*rhs`, chooses one, and renders
the resulting `lhs/rhs/result` triples. Legal-position constraints keep inference on the
canonical precedence and left-to-right schedule.

OOD evaluation presets expose where the current model generalizes and where it is still
distribution-bound:

| Preset | Trace exact | Learned step value exact | Learned final value exact |
| --- | ---: | ---: | ---: |
| `in_dist` | `1.000` | `1.000` | `1.000` |
| `larger_numbers` | `1.000` | `1.000` | `1.000` |
| `longer_expr` | `1.000` | `1.000` | `1.000` |
| `no_multiply` | `1.000` | `1.000` | `1.000` |
| `many_multiply` | `1.000` | `1.000` | `1.000` |

The variable reasoner now trains with a balanced multi-step curriculum and a 34-token
math feature window, covering standard, larger-number, longer, no-multiply, and
many-multiply expressions. Checkpoint selection uses unconstrained trace-equivalence.

The dynamic-length Phase 1 path raises the variable reasoner to a 34-token math feature
window and up to 16 reduction steps. Length generalization currently measures as:

| Preset | Trace exact | Learned step value exact | Learned final value exact |
| --- | ---: | ---: | ---: |
| `length_3` | `1.000` | `1.000` | `1.000` |
| `length_5` | `1.000` | `1.000` | `1.000` |
| `length_8` | `0.988` | `0.998` | `0.988` |
| `length_16` | `0.750` | `0.917` | `0.750` |

Without legal/canonical inference constraints, the learned pointer policy is weaker but
still usually chooses valid reductions:

| Preset | Unconstrained trace equivalent exact | Unconstrained final value exact |
| --- | ---: | ---: |
| `in_dist` | `1.000` | `1.000` |
| `larger_numbers` | `1.000` | `1.000` |
| `longer_expr` | `1.000` | `1.000` |
| `no_multiply` | `0.996` | `0.996` |
| `many_multiply` | `1.000` | `1.000` |
| `length_5` | `1.000` | `1.000` |
| `length_8` | `0.984` | `0.984` |
| `length_16` | `0.448` | `0.448` |

General structure support has started with safe parenthesized expression parsing. The
debug solver can now render parenthesized traces through the parser fallback:

```bash
bash scripts/solve_math.sh --debug-reasoning "What is (3+8)*2?"
```

Expected trace:

```text
3+8=11,11*2=22,22
```

This is not yet a learned parenthesis policy in the variable reasoner. Parentheses are
handled by the safe parser while the learned dynamic pointer still covers flat `+`, `-`,
and `*` expressions.

An experimental raw learned value head is also available. It predicts the numeric result
of each reduction directly from the recurrent latent state plus `lhs/op/rhs`, instead of
choosing among symbolic candidates:

```bash
PYTHONPATH=src python -m ssl_reasoner.train \
  --stages variable_raw_value_head \
  --checkpoint checkpoints/variable_reasoner/best.pt \
  --output-dir checkpoints/variable_reasoner_raw_head \
  --variable-reasoner-steps 1000 \
  --train-curriculum multi_step_balanced \
  --val-curriculum multi_step_balanced \
  --max-math-len 34 \
  --max-variable-steps 16 \
  --use-math-features

PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/variable_reasoner_raw_head/best.pt \
  --raw-learned-values \
  --preset all
```

Current result: the raw regression loss trains, but exact arithmetic remains near zero.
The candidate-scored learned-value path is still the working path for exact results. This
keeps the raw path measurable without pretending it has solved learned arithmetic.

Two sharper transition-head baselines are now available for the "learn single-step
arithmetic first" direction:

```bash
bash scripts/train_variable_digit_value_head.sh
bash scripts/train_variable_class_value_head.sh
bash scripts/train_standalone_transition.sh
bash scripts/train_standalone_transition_reductions.sh
```

The digit head predicts sign plus fixed decimal digits. The class head predicts a bounded
integer class from `-1000` to `10000`. The standalone transition encoder is stronger:
it sees only `(lhs, op, rhs)` through learned number embeddings plus numeric features,
then predicts the bounded result class. These can be used in variable-reasoning eval:

```bash
PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/variable_digit_value_head_single/best.pt \
  --digit-learned-values \
  --preset in_dist

PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/variable_class_value_head_single/best.pt \
  --class-learned-values \
  --preset in_dist

PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/standalone_transition_single/best.pt \
  --standalone-learned-values \
  --preset in_dist

PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/standalone_transition_reductions/best.pt \
  --standalone-learned-values \
  --preset all

PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/standalone_transition_hybrid/best.pt \
  --standalone-digit-values \
  --preset all

PYTHONPATH=src python -m ssl_reasoner.eval_variable_reasoning \
  --checkpoint checkpoints/standalone_transition_hybrid/best.pt \
  --standalone-hybrid-values \
  --preset all
```

Measured single-step answer exact on 500 generated `single_op_balanced` examples:

| Transition path | Single-step exact |
| --- | ---: |
| candidate-scored value head | `0.978` |
| digit value head | `0.062` |
| bounded class value head | `0.298` |
| standalone transition encoder | `0.982` |

Standalone by-op breakdown on that eval:

| Op | Exact |
| --- | ---: |
| `+` | `0.958` |
| `-` | `0.988` |
| `*` | `1.000` |

Plugging the single-step standalone transition directly into multi-step in-distribution
reasoning gives `0.310` answer exact. The remaining gap is expected: multi-step traces
create intermediate negative and large states that are not covered well by the
single-step curriculum.

Training the same standalone transition on the full reduction-state distribution improves
multi-step reasoning substantially. Out-of-range transition targets outside
`-1000..10000` are masked during training/eval instead of clamped.

Current `checkpoints/standalone_transition_reductions/best.pt` results:

| Eval | Answer exact |
| --- | ---: |
| in-dist multi-step, 500 samples | `0.678` |
| in-dist multi-step, 200 samples | `0.650` |
| longer expressions | `0.585` |
| no multiply | `0.995` |
| many multiply | `0.520` |
| length 5 | `0.450` |
| length 8 | `0.320` |
| length 16 | `0.080` |
| larger numbers | `0.000` |

Single-step retention after reduction training is `0.914` overall:

| Op | Exact |
| --- | ---: |
| `+` | `0.844` |
| `-` | `0.898` |
| `*` | `1.000` |

The reduction-trained standalone transition is now useful, but it still fails on larger
numbers and long chains. The next improvement should expand the value representation
beyond the bounded class range or add a digit/residual fallback for out-of-range states.

The first out-of-range experiment adds a 10-digit standalone decoder and a hybrid
class-or-digit inference mode. It trains, but digit inference is still weaker than the
bounded class path, so class mode remains the best current setting:

| Checkpoint / mode | In-dist | Larger numbers | Length 8 | Length 16 |
| --- | ---: | ---: | ---: | ---: |
| `standalone_transition_reductions`, class, 200 samples | `0.650` | `0.000` | `0.320` | `0.080` |
| `standalone_transition_hybrid`, class, 200 samples | `0.740` | `0.010` | `0.370` | `0.100` |
| `standalone_transition_hybrid`, hybrid, 200 samples | `0.695` | `0.000` | `0.330` | `0.095` |
| `standalone_transition_hybrid`, digit, 200 samples | `0.165` | `0.000` | `0.035` | `0.015` |

On 500 in-dist samples, `standalone_transition_hybrid` in class mode reaches `0.764`.
This shows the extra reduction training helped, but the digit fallback is not yet the
right larger-number solution.

## Smoke Train

```bash
python -m ssl_reasoner.train --steps 50 --train-size 512 --val-size 128
```

The `--steps` value is split across Stage 0/1/2 as 20%/50%/30% by default.
When `--stages 0,1,2,3` is used, it is split 20%/50%/20%/10%. For explicit control:

```bash
python -m ssl_reasoner.train \
  --stage0-steps 200 \
  --stage1-steps 500 \
  --stage2-steps 300
```

Stage 3 joint fine-tuning follows the plan's combined latent + readout objective:

```bash
scripts/stage3_joint_from_structured_answer.sh
```

The Stage 3 script uses a conservative low learning rate and standard mixed curriculum to
avoid decoder drift while still updating the encoder, predictor, readout, and trace heads.
Stage 3 also includes an answer-latent reasoning target for the two operation ids plus the
intermediate and final values, so mixed-expression reasoning can be supervised directly.
To train only that head on frozen Stage 3 latents:

```bash
scripts/train_reasoning_head.sh
```

To train the predictor with reasoning supervision during Stage 1, then re-fit the readout:

```bash
scripts/predictor_reasoning_stage1.sh
```

The current best math-only recipe uses a mixed-heavy Stage 1 curriculum:

```bash
scripts/predictor_reasoning_stage1_mixed_heavy.sh
```

The strongest fixed-eval result so far adds stronger anti-collapse pressure:

```bash
scripts/predictor_reasoning_stage1_mixed_anticollapse.sh
```

Stage 4 trains the latent verifier on frozen Stage 3 latents:

```bash
scripts/stage4_verifier_from_stage3.sh
```

Evaluate verifier-ranked latent candidates with:

```bash
scripts/eval_stage4_verifier.sh
```

Verifier evaluation includes symbolic math candidates: parsed precedence result,
left-to-right result, intermediate operation values, and small variants around the readout
answer.

## Current Best Math Solver

The strongest math-only path uses the model's predicted trace operation ids, executes
those operations deterministically, and falls back to the readout only when no expression
can be parsed. The current default checkpoint is:

```text
checkpoints/trace_ops_head_mixed/best.pt
```

That checkpoint is local and ignored by git. The repo tracks the commands and manifest
needed to reproduce it, not the 28 MB weight file.

Download the current best checkpoint from the GitHub Release:

```bash
bash scripts/download_current_best.sh
```

Release page:

```text
https://github.com/akshai0296/ssl_reasoner/releases/tag/math-solver-v0.1.0
```

Solve examples with:

```bash
bash scripts/solve_math.sh "What is 2+3*4?" "Calculate 15+33*4."
```

Show operation ids, operation confidence, operation answer, and readout answer:

```bash
bash scripts/solve_math.sh --debug "What is 2+3*4?"
```

Emit JSON:

```bash
bash scripts/solve_math.sh --json "What is 2+3*4?"
```

Evaluate the default math solver:

```bash
bash scripts/eval_math_solver.sh
```

Evaluate another curriculum:

```bash
CURRICULUM=mixed_only bash scripts/eval_math_solver.sh
```

Expected fixed 500-sample results for the current local best checkpoint:

| Curriculum | Accuracy |
| --- | ---: |
| `mixed` | `1.000` |
| `mixed_only` | `0.990` |
| `seen_mixed` | `1.000` |
| `unseen_mixed` | `0.998` |

The checkpoint manifest is tracked at:

```text
reports/checkpoint_manifest.json
```

Rebuild the current best checkpoint and write a reproduction report:

```bash
bash scripts/reproduce_trace_ops_head.sh
```

If the checkpoint already exists locally and you only want to regenerate the report:

```bash
SKIP_TRAIN=1 bash scripts/reproduce_trace_ops_head.sh
```

## Evaluate a Checkpoint

```bash
python -m ssl_reasoner.eval --checkpoint checkpoints/best.pt --samples 500
```

## Compositional Split Diagnostic

The math generator includes seen/unseen operand-range splits for checking whether the
reasoner learned reusable arithmetic behavior instead of memorizing local ranges:

```bash
scripts/eval_compositional_splits.sh checkpoints/predictor_structured_answer/best.pt
```

Available split curricula are `seen_single`, `unseen_single`, `seen_mixed`,
`unseen_mixed`, and `compositional_train`.
