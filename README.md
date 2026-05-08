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
```

Current learned baseline result:

| Metric | Score |
| --- | ---: |
| answer exact | `1.000` |
| learned trace exact | `0.812` |
| learned trace equivalent exact | `1.000` |
| learned step value exact | `0.010` |
| learned final value exact | `0.000` |

The answer remains exact because the solver still uses the parser fallback for final
answers on longer expressions. The learned recurrent trace head now predicts which
adjacent operation to reduce next with a dynamic pointer over the currently remaining
operators, then an exact transition cell executes `+`, `-`, or `*`. It also receives
legal-position supervision for precedence. The strict trace metric requires the canonical
target order; the equivalent trace metric replays the predicted reductions and accepts
valid alternate orders that still reach the same final answer. The `--learned-values`
eval disables exact arithmetic rendering and uses the model's predicted `lhs/rhs/result`
triples; this exposes the current transition-value bottleneck.

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
