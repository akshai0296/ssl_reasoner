from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import MathDataset, generate_math_examples
from .model import MathJEPAReadout
from .tokenizer import build_math_tokenizer


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, dataset, tokenizer, device, batch_size: int) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    total = 0
    for batch in loader:
        problem_ids = batch["problem_ids"].to(device)
        decoded_ids = model.solve_ids(problem_ids, pad_id=tokenizer.pad_id)
        predictions = [tokenizer.decode(ids).strip() for ids in decoded_ids]
        for pred, answer in zip(predictions, batch["answer"]):
            correct += pred == answer
            total += 1
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--max-problem-len", type=int, default=64)
    parser.add_argument("--max-answer-len", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = _device(args.device)
    tokenizer = build_math_tokenizer()

    train_examples = generate_math_examples(args.train_size, seed=args.seed)
    val_examples = generate_math_examples(args.val_size, seed=args.seed + 1)
    train_dataset = MathDataset(
        train_examples, tokenizer, args.max_problem_len, args.max_answer_len
    )
    val_dataset = MathDataset(val_examples, tokenizer, args.max_problem_len, args.max_answer_len)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = MathJEPAReadout(
        vocab_size=tokenizer.vocab_size,
        max_problem_len=args.max_problem_len,
        max_answer_len=args.max_answer_len,
        d_model=args.d_model,
        num_slots=args.num_slots,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    step = 0

    while step < args.steps:
        for batch in loader:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = model(
                batch["problem_ids"].to(device),
                batch["answer_ids"].to(device),
                batch["answer_len"].to(device),
            )
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            if step == 1 or step % 50 == 0 or step == args.steps:
                acc = evaluate(model, val_dataset, tokenizer, device, args.batch_size)
                print(
                    f"step={step} loss={out['loss'].item():.4f} "
                    f"pred={out['pred_loss'].item():.4f} token={out['token_loss'].item():.4f} "
                    f"len={out['length_loss'].item():.4f} val_exact={acc:.3f}"
                )
                if acc > best_acc:
                    best_acc = acc
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "args": vars(args),
                            "vocab_chars": tokenizer.chars,
                            "best_acc": best_acc,
                        },
                        output_dir / "best.pt",
                    )
            if step >= args.steps:
                break

    print(f"done best_val_exact={best_acc:.3f} checkpoint={output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
