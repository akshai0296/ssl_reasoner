from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .data import (
    CURRICULA,
    MATH_FEATURE_VOCAB_SIZE,
    MathDataset,
    class_to_value,
    generate_math_examples,
)
from .model import LatentVerifier, MathJEPAReadout
from .tokenizer import build_math_tokenizer
from .verifier import model_candidate_texts


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
    parser.add_argument(
        "--mode",
        choices=[
            "pred",
            "target",
            "trace",
            "trace_struct",
            "structured_answer",
            "fallback",
            "verifier",
        ],
        default="pred",
    )
    parser.add_argument("--fallback-confidence", type=float, default=0.8)
    parser.add_argument("--verifier-checkpoint", default=None)
    parser.add_argument("--verifier-candidates", type=int, default=8)
    parser.add_argument("--verifier-noise-scale", type=float, default=0.15)
    parser.add_argument(
        "--candidate-set",
        choices=["neural", "symbolic_no_oracle", "symbolic_full"],
        default="symbolic_full",
    )
    parser.add_argument(
        "--curriculum",
        choices=CURRICULA,
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
        use_reasoning_trace=train_args.get("use_reasoning_trace", False),
        max_trace_len=train_args.get("max_trace_len", 32),
        use_trace_fusion=train_args.get("use_trace_fusion", False),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    verifier = None
    if args.mode == "verifier":
        if args.verifier_checkpoint is None:
            raise ValueError("--verifier-checkpoint is required for --mode verifier")
        verifier_ckpt = torch.load(args.verifier_checkpoint, map_location=device)
        verifier = LatentVerifier(train_args["d_model"]).to(device)
        verifier.load_state_dict(verifier_ckpt["verifier"])
        verifier.eval()

    dataset = MathDataset(
        generate_math_examples(args.samples, seed=args.seed, curriculum=args.curriculum),
        tokenizer,
        train_args["max_problem_len"],
        train_args["max_answer_len"],
        train_args.get("max_math_len", 8),
        train_args.get("max_trace_len", 32),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)
    correct = 0
    total = 0
    by_op: dict[str, list[int]] = {}
    by_split: dict[str, list[int]] = {}
    shown = 0
    with torch.no_grad():
        for batch in loader:
            if args.mode == "trace_struct":
                pred_ops, pred_value_ids = model.predict_structured_trace(
                    batch["problem_ids"].to(device),
                    math_ids=batch["math_ids"].to(device),
                )
                op_ids = batch["trace_op_ids"].to(device)
                value_targets = batch["trace_value_ids"].to(device)
                value_mask = batch["trace_value_mask"].to(device)
                op_correct = int((pred_ops == op_ids).sum().item())
                op_total = int(op_ids.numel())
                value_correct = float(((pred_value_ids == value_targets).float() * value_mask).sum().item())
                value_total = float(value_mask.sum().item())
                correct += op_correct
                total += op_total
                counts = by_op.setdefault("value_acc", [0, 0])
                counts[0] += value_correct
                counts[1] += value_total
                continue
            if args.mode in {"structured_answer", "fallback"}:
                pred_ids, confidence = model.predict_structured_answer(
                    batch["problem_ids"].to(device),
                    math_ids=batch["math_ids"].to(device),
                )
                structured_predictions = [str(class_to_value(int(idx))) for idx in pred_ids.tolist()]
                if args.mode == "structured_answer":
                    predictions = structured_predictions
                else:
                    decoded = model.solve_ids(
                        batch["problem_ids"].to(device),
                        pad_id=tokenizer.pad_id,
                        math_ids=batch["math_ids"].to(device),
                    )
                    readout_predictions = [tokenizer.decode(ids).strip() for ids in decoded]
                    predictions = [
                        structured if float(conf) >= args.fallback_confidence else readout
                        for structured, readout, conf in zip(
                            structured_predictions, readout_predictions, confidence
                        )
                    ]
                references = batch["answer"]
                for problem, pred, answer, op_label, split_label in zip(
                    batch["problem"],
                    predictions,
                    references,
                    batch["op_label"],
                    batch["split_label"],
                ):
                    is_correct = pred == answer
                    correct += is_correct
                    total += 1
                    counts = by_op.setdefault(str(op_label), [0, 0])
                    counts[0] += int(is_correct)
                    counts[1] += 1
                    split_counts = by_split.setdefault(str(split_label), [0, 0])
                    split_counts[0] += int(is_correct)
                    split_counts[1] += 1
                    if shown < 10:
                        mark = "ok" if is_correct else "bad"
                        print(f"{mark}: {problem} -> pred={pred!r} target={answer!r}")
                        shown += 1
                continue
            if args.mode == "verifier":
                assert verifier is not None
                problem_ids = batch["problem_ids"].to(device)
                math_ids = batch["math_ids"].to(device)
                contexts = model.encode_context(problem_ids)
                candidates, candidate_texts = model_candidate_texts(
                    model,
                    tokenizer,
                    batch,
                    device,
                    noise_scale=args.verifier_noise_scale,
                    noise_candidates=max(args.verifier_candidates - 8, 0),
                    candidate_set=args.candidate_set,
                )
                stacked = torch.stack(candidates, dim=1)
                flat_slots = stacked.flatten(0, 1)
                flat_contexts = contexts.unsqueeze(1).expand(
                    -1, stacked.size(1), -1
                ).flatten(0, 1)
                scores = verifier(flat_contexts, flat_slots).view(
                    contexts.size(0), stacked.size(1)
                )
                best_idx = scores.argmax(dim=1)
                predictions = [
                    candidate_texts[int(idx)][row]
                    for row, idx in enumerate(best_idx.detach().cpu().tolist())
                ]
                references = batch["answer"]
                for problem, pred, answer, op_label, split_label in zip(
                    batch["problem"],
                    predictions,
                    references,
                    batch["op_label"],
                    batch["split_label"],
                ):
                    is_correct = pred == answer
                    correct += is_correct
                    total += 1
                    counts = by_op.setdefault(str(op_label), [0, 0])
                    counts[0] += int(is_correct)
                    counts[1] += 1
                    split_counts = by_split.setdefault(str(split_label), [0, 0])
                    split_counts[0] += int(is_correct)
                    split_counts[1] += 1
                    if shown < 10:
                        mark = "ok" if is_correct else "bad"
                        print(f"{mark}: {problem} -> pred={pred!r} target={answer!r}")
                        shown += 1
                continue
            if args.mode == "pred":
                decoded = model.solve_ids(
                    batch["problem_ids"].to(device),
                    pad_id=tokenizer.pad_id,
                    math_ids=batch["math_ids"].to(device),
                )
                references = batch["answer"]
            elif args.mode == "target":
                decoded = model.solve_ids_from_target(
                    batch["answer_ids"].to(device),
                    pad_id=tokenizer.pad_id,
                    use_ema=True,
                )
                references = batch["answer"]
            else:
                decoded = model.solve_trace_ids(
                    batch["problem_ids"].to(device),
                    pad_id=tokenizer.pad_id,
                    math_ids=batch["math_ids"].to(device),
                )
                references = batch["trace"]
            predictions = [tokenizer.decode(ids).strip() for ids in decoded]
            for problem, pred, answer, op_label, split_label in zip(
                batch["problem"],
                predictions,
                references,
                batch["op_label"],
                batch["split_label"],
            ):
                is_correct = pred == answer
                correct += is_correct
                total += 1
                counts = by_op.setdefault(str(op_label), [0, 0])
                counts[0] += int(is_correct)
                counts[1] += 1
                split_counts = by_split.setdefault(str(split_label), [0, 0])
                split_counts[0] += int(is_correct)
                split_counts[1] += 1
                if shown < 10:
                    mark = "ok" if is_correct else "bad"
                    print(f"{mark}: {problem} -> pred={pred!r} target={answer!r}")
                    shown += 1
    if args.mode == "trace_struct":
        value_counts = by_op.pop("value_acc", [0, 1])
        print(f"trace_struct_op_acc={correct / max(total, 1):.3f} ({correct}/{total})")
        print(f"trace_struct_value_acc={value_counts[0] / max(value_counts[1], 1):.3f}")
        return
    print(f"{args.mode}_exact_match={correct / max(total, 1):.3f} ({correct}/{total})")
    for op_label, counts in sorted(by_op.items()):
        acc = counts[0] / max(counts[1], 1)
        print(f"{args.mode}_op_{op_label}_exact={acc:.3f} ({counts[0]}/{counts[1]})")
    for split_label, counts in sorted(by_split.items()):
        acc = counts[0] / max(counts[1], 1)
        print(f"{args.mode}_split_{split_label}_exact={acc:.3f} ({counts[0]}/{counts[1]})")


if __name__ == "__main__":
    main()
