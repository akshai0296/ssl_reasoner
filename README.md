# SSL Reasoner

A fresh math-only prototype for a self-supervised latent reasoner.

The first target is intentionally narrow:

1. Generate synthetic arithmetic problems.
2. Encode the problem into a latent vector.
3. Encode the answer into `K` target latent slots.
4. Train a predictor to map problem latent to answer latent slots.
5. Decode answer latent slots with a deterministic parallel readout decoder.

No autoregressive generation is used. The readout predicts all token positions and answer
length in one forward pass.

## Smoke Train

```bash
python -m ssl_reasoner.train --steps 50 --train-size 512 --val-size 128
```

## Evaluate a Checkpoint

```bash
python -m ssl_reasoner.eval --checkpoint checkpoints/best.pt --samples 500
```
