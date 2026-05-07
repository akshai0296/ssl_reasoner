# SSL Reasoner

A fresh math-only prototype for a self-supervised latent reasoner.

The first target is intentionally narrow and math-only:

1. Generate synthetic arithmetic problems.
2. Stage 0: warm up an answer target encoder + deterministic readout.
3. Stage 1: train a problem encoder + sequence predictor to match EMA answer slots.
4. Stage 2: freeze the predictor and train the deterministic readout on predicted slots.
5. Evaluate exact-match numeric answers.

No autoregressive generation is used. The readout predicts all token positions and answer
length in one forward pass.

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
