from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .data import MATH_FEATURE_VOCAB_SIZE, MathDataset, generate_math_examples
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mode", choices=["pred", "target"], default="pred")
    parser.add_argument(
        "--curriculum",
        choices=["mixed", "single_op_balanced", "mixed_only"],
        default="mixed",
    )
    args = parser.parse_args()

    device = _device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt["args"]
    tokenizer = build_math_tokenizer()
    model = MathJEPAReadout(
        vocab_size=tokenizer.vocab_size,
        max_problem_len=train_args["max_problem_len"],
        max_answer_len=train_args["max_answer_len"],
        d_model=train_args["d_model"],
        num_slots=train_args["num_slots"],
        encoder_layers=train_args.get("encoder_layers", 2),
        predictor_layers=train_args.get("predictor_layers", 3),
        readout_layers=train_args.get("readout_layers", 2),
        num_heads=train_args.get("num_heads", 4),
        predictor_type=train_args.get("predictor_type", "pooled"),
        use_math_features=train_args.get("use_math_features", False),
        math_vocab_size=MATH_FEATURE_VOCAB_SIZE,
        max_math_len=train_args.get("max_math_len", 8),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = MathDataset(
        generate_math_examples(args.samples, seed=args.seed, curriculum=args.curriculum),
        tokenizer,
        train_args["max_problem_len"],
        train_args["max_answer_len"],
        train_args.get("max_math_len", 8),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)
    correct = 0
    total = 0
    by_op: dict[str, list[int]] = {}
    shown = 0
    with torch.no_grad():
        for batch in loader:
            if args.mode == "pred":
                decoded = model.solve_ids(
                    batch["problem_ids"].to(device),
                    pad_id=tokenizer.pad_id,
                    math_ids=batch["math_ids"].to(device),
                )
            else:
                decoded = model.solve_ids_from_target(
                    batch["answer_ids"].to(device),
                    pad_id=tokenizer.pad_id,
                    use_ema=True,
                )
            predictions = [tokenizer.decode(ids).strip() for ids in decoded]
            for problem, pred, answer, op_label in zip(
                batch["problem"], predictions, batch["answer"], batch["op_label"]
            ):
                is_correct = pred == answer
                correct += is_correct
                total += 1
                counts = by_op.setdefault(str(op_label), [0, 0])
                counts[0] += int(is_correct)
                counts[1] += 1
                if shown < 10:
                    mark = "ok" if is_correct else "bad"
                    print(f"{mark}: {problem} -> pred={pred!r} target={answer!r}")
                    shown += 1
    print(f"{args.mode}_exact_match={correct / max(total, 1):.3f} ({correct}/{total})")
    for op_label, counts in sorted(by_op.items()):
        acc = counts[0] / max(counts[1], 1)
        print(f"{args.mode}_op_{op_label}_exact={acc:.3f} ({counts[0]}/{counts[1]})")


if __name__ == "__main__":
    main()
