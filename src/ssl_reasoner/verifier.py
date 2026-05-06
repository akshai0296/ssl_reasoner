from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import CURRICULA, MATH_FEATURE_VOCAB_SIZE, MathDataset, generate_math_examples
from .model import LatentVerifier, MathJEPAReadout
from .tokenizer import build_math_tokenizer


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_reasoner(checkpoint_path: str, device: torch.device) -> tuple[MathJEPAReadout, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt["args"]
    tokenizer = build_math_tokenizer()
    model = MathJEPAReadout(
        vocab_size=tokenizer.vocab_size,
        max_problem_len=args["max_problem_len"],
        max_answer_len=args["max_answer_len"],
        d_model=args["d_model"],
        num_slots=args["num_slots"],
        encoder_layers=args.get("encoder_layers", 2),
        predictor_layers=args.get("predictor_layers", 3),
        readout_layers=args.get("readout_layers", 2),
        num_heads=args.get("num_heads", 4),
        predictor_type=args.get("predictor_type", "pooled"),
        use_math_features=args.get("use_math_features", False),
        math_vocab_size=MATH_FEATURE_VOCAB_SIZE,
        max_math_len=args.get("max_math_len", 8),
        use_reasoning_trace=args.get("use_reasoning_trace", False),
        max_trace_len=args.get("max_trace_len", 32),
        use_trace_fusion=args.get("use_trace_fusion", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, args


@torch.no_grad()
def verifier_batch(
    model: MathJEPAReadout,
    tokenizer,
    batch: dict,
    device: torch.device,
    noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    problem_ids = batch["problem_ids"].to(device)
    answer_ids = batch["answer_ids"].to(device)
    math_ids = batch["math_ids"].to(device)
    context = model.encode_context(problem_ids)
    pred_slots = model.predict_answer_slots(problem_ids, math_ids)
    true_slots = model.encode_target(answer_ids, use_ema=True)
    wrong_slots = true_slots.roll(shifts=1, dims=0)
    noisy_slots = pred_slots + noise_scale * torch.randn_like(pred_slots)
    pred_ids = model.readout.decode_ids(pred_slots, pad_id=tokenizer.pad_id)
    pred_labels = torch.tensor(
        [
            float(tokenizer.decode(ids).strip() == answer)
            for ids, answer in zip(pred_ids, batch["answer"])
        ],
        device=device,
    )

    contexts = torch.cat([context, context, context, context], dim=0)
    candidates = torch.cat([true_slots, pred_slots, wrong_slots, noisy_slots], dim=0)
    labels = torch.cat(
        [
            torch.ones(context.size(0), device=device),
            pred_labels,
            torch.zeros(context.size(0), device=device),
            torch.zeros(context.size(0), device=device),
        ],
        dim=0,
    )
    return contexts, candidates, labels


@torch.no_grad()
def evaluate_verifier(
    model: MathJEPAReadout,
    verifier: LatentVerifier,
    dataset: MathDataset,
    tokenizer,
    device: torch.device,
    batch_size: int,
    noise_scale: float,
) -> dict[str, float]:
    verifier.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    total = 0
    pos_scores = []
    neg_scores = []
    for batch in loader:
        contexts, candidates, labels = verifier_batch(
            model, tokenizer, batch, device, noise_scale
        )
        logits = verifier(contexts, candidates)
        preds = logits.sigmoid().ge(0.5).float()
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
        pos_scores.append(logits.sigmoid()[labels.eq(1)].mean().detach().cpu())
        neg_scores.append(logits.sigmoid()[labels.eq(0)].mean().detach().cpu())
    return {
        "verifier_acc": correct / max(total, 1),
        "pos_score": float(torch.stack(pos_scores).mean().item()),
        "neg_score": float(torch.stack(neg_scores).mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/stage3_joint_low_lr/best.pt")
    parser.add_argument("--output", default="checkpoints/verifier_v1.pt")
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--curriculum", choices=CURRICULA, default="mixed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = _device(args.device)
    tokenizer = build_math_tokenizer()
    model, train_args = load_reasoner(args.checkpoint, device)
    verifier = LatentVerifier(train_args["d_model"]).to(device)

    train_dataset = MathDataset(
        generate_math_examples(args.train_size, seed=args.seed, curriculum=args.curriculum),
        tokenizer,
        train_args["max_problem_len"],
        train_args["max_answer_len"],
        train_args.get("max_math_len", 8),
        train_args.get("max_trace_len", 32),
    )
    val_dataset = MathDataset(
        generate_math_examples(args.val_size, seed=args.seed + 1, curriculum=args.curriculum),
        tokenizer,
        train_args["max_problem_len"],
        train_args["max_answer_len"],
        train_args.get("max_math_len", 8),
        train_args.get("max_trace_len", 32),
    )
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(verifier.parameters(), lr=args.lr, weight_decay=0.01)
    best_acc = -1.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    step = 0
    print(f"=== stage4_verifier ({args.steps} steps) ===")
    while step < args.steps:
        for batch in loader:
            verifier.train()
            optimizer.zero_grad(set_to_none=True)
            contexts, candidates, labels = verifier_batch(
                model, tokenizer, batch, device, noise_scale=args.noise_scale
            )
            logits = verifier(contexts, candidates)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(verifier.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                stats = evaluate_verifier(
                    model,
                    verifier,
                    val_dataset,
                    tokenizer,
                    device,
                    args.batch_size,
                    noise_scale=args.noise_scale,
                )
                print(
                    f"stage4_verifier step={step} loss={loss.item():.4f} "
                    f"verifier_acc={stats['verifier_acc']:.3f} "
                    f"pos_score={stats['pos_score']:.3f} "
                    f"neg_score={stats['neg_score']:.3f}"
                )
                if stats["verifier_acc"] > best_acc:
                    best_acc = stats["verifier_acc"]
                    torch.save(
                        {
                            "verifier": verifier.state_dict(),
                            "checkpoint": args.checkpoint,
                            "args": vars(args),
                            "d_model": train_args["d_model"],
                            "best_acc": best_acc,
                        },
                        output,
                    )
            if step >= args.steps:
                break
    print(f"done best_verifier_acc={best_acc:.3f} checkpoint={output}")


if __name__ == "__main__":
    main()
