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

The `--steps` value is split across Stage 0/1/2 as 20%/50%/30%. For explicit control:

```bash
python -m ssl_reasoner.train \
  --stage0-steps 200 \
  --stage1-steps 500 \
  --stage2-steps 300
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
