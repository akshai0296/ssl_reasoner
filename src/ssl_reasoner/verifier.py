from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
import torch.nn.functional as F
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

TRACE_ID_TO_OP = {0: "none", 1: "+", 2: "-", 3: "*"}


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
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, args


def _apply_op(lhs: int, op: str, rhs: int) -> int:
    if op == "+":
        return lhs + rhs
    if op == "-":
        return lhs - rhs
    if op == "*":
        return lhs * rhs
    raise ValueError(f"Unknown operator: {op}")


def operation_candidate_text(
    problem: str,
    op_ids: list[int],
) -> str | None:
    match = re.search(r"\d+[+\-*]\d+(?:[+\-*]\d+)?", problem)
    if match is None:
        return None
    expr = match.group(0)
    parts = re.split(r"([+\-*])", expr)
    ops = [TRACE_ID_TO_OP.get(int(idx), "none") for idx in op_ids]

    if len(parts) == 3:
        a, true_op, b = parts
        op = ops[0] if ops and ops[0] != "none" else true_op
        try:
            return str(_apply_op(int(a), op, int(b)))
        except ValueError:
            return None

    if len(parts) != 5:
        return None

    a_text, op1, b_text, op2, c_text = parts
    a = int(a_text)
    b = int(b_text)
    c = int(c_text)
    first_op = ops[0] if len(ops) > 0 and ops[0] != "none" else op1
    second_op = ops[1] if len(ops) > 1 and ops[1] != "none" else op2

    try:
        if op2 == "*" and first_op == op2:
            first_value = _apply_op(b, first_op, c)
            return str(_apply_op(a, second_op, first_value))
        first_value = _apply_op(a, first_op, b)
        return str(_apply_op(first_value, second_op, c))
    except ValueError:
        return None


def symbolic_candidate_texts(
    problem: str,
    base_prediction: str | None = None,
    include_oracle: bool = True,
) -> list[str]:
    match = re.search(r"\d+[+\-*]\d+(?:[+\-*]\d+)?", problem)
    if match is None:
        return []
    expr = match.group(0)
    parts = re.split(r"([+\-*])", expr)
    candidates: list[int] = []

    if len(parts) == 3:
        a, op, b = parts
        if include_oracle:
            candidates.append(_apply_op(int(a), op, int(b)))
    elif len(parts) == 5:
        a_text, op1, b_text, op2, c_text = parts
        a = int(a_text)
        b = int(b_text)
        c = int(c_text)
        if op2 == "*":
            first = _apply_op(b, op2, c)
            precedence = _apply_op(a, op1, first)
        else:
            first = _apply_op(a, op1, b)
            precedence = _apply_op(first, op2, c)
        left_first = _apply_op(a, op1, b)
        left_to_right = _apply_op(left_first, op2, c)
        right_first = _apply_op(b, op2, c)
        right_grouped = _apply_op(a, op1, right_first)
        if include_oracle:
            candidates.append(precedence)
        variants = [left_to_right, first, left_first, right_first, right_grouped]
        if not include_oracle:
            variants = [value for value in variants if value != precedence]
        candidates.extend(variants)

    if base_prediction is not None and re.fullmatch(r"-?\d+", base_prediction.strip()):
        base = int(base_prediction.strip())
        candidates.extend([base - 1, base + 1])

    deduped = []
    seen = set()
    for value in candidates:
        if value not in seen:
            seen.add(value)
            deduped.append(str(value))
    return deduped


@torch.no_grad()
def encode_answer_strings(
    model: MathJEPAReadout,
    tokenizer,
    texts: list[str],
    max_answer_len: int,
    device: torch.device,
) -> torch.Tensor:
    ids = torch.tensor(
        [tokenizer.encode(text, max_answer_len) for text in texts],
        dtype=torch.long,
        device=device,
    )
    return model.encode_target(ids, use_ema=True)


@torch.no_grad()
def model_candidate_texts(
    model: MathJEPAReadout,
    tokenizer,
    batch: dict,
    device: torch.device,
    noise_scale: float,
    noise_candidates: int,
    candidate_set: str = "symbolic_full",
) -> tuple[list[torch.Tensor], list[list[str]]]:
    problem_ids = batch["problem_ids"].to(device)
    math_ids = batch["math_ids"].to(device)
    base_slots = model.predict_answer_slots(problem_ids, math_ids)
    base_ids = model.readout.decode_ids(base_slots, pad_id=tokenizer.pad_id)
    base_texts = [tokenizer.decode(ids).strip() for ids in base_ids]

    candidate_slots = [base_slots]
    candidate_texts = [base_texts]
    max_answer_len = batch["answer_ids"].size(1)

    if candidate_set in {"symbolic_no_oracle", "symbolic_full"}:
        include_oracle = candidate_set == "symbolic_full"
        symbolic_rows = [
            symbolic_candidate_texts(
                problem, base_prediction=base, include_oracle=include_oracle
            )
            for problem, base in zip(batch["problem"], base_texts)
        ]
        max_symbolic = max((len(row) for row in symbolic_rows), default=0)
        for idx in range(max_symbolic):
            texts = [
                row[idx] if idx < len(row) else base
                for row, base in zip(symbolic_rows, base_texts)
            ]
            candidate_slots.append(
                encode_answer_strings(model, tokenizer, texts, max_answer_len, device)
            )
            candidate_texts.append(texts)

    if model.use_reasoning_trace:
        reasoning_op_ids, reasoning_value_ids = model.predict_reasoning_struct(
            problem_ids, math_ids=math_ids
        )
        reasoning_op_texts = [
            operation_candidate_text(problem, op_row)
            or base
            for problem, op_row, base in zip(
                batch["problem"],
                reasoning_op_ids.detach().cpu().tolist(),
                base_texts,
            )
        ]
        candidate_slots.append(
            encode_answer_strings(model, tokenizer, reasoning_op_texts, max_answer_len, device)
        )
        candidate_texts.append(reasoning_op_texts)

        reasoning_texts = [
            str(class_to_value(int(idx)))
            for idx in reasoning_value_ids[:, -1].detach().cpu().tolist()
        ]
        candidate_slots.append(
            encode_answer_strings(model, tokenizer, reasoning_texts, max_answer_len, device)
        )
        candidate_texts.append(reasoning_texts)

        structured_ids, _ = model.predict_structured_answer(problem_ids, math_ids=math_ids)
        structured_texts = [
            str(class_to_value(int(idx))) for idx in structured_ids.detach().cpu().tolist()
        ]
        candidate_slots.append(
            encode_answer_strings(model, tokenizer, structured_texts, max_answer_len, device)
        )
        candidate_texts.append(structured_texts)

        trace_op_ids, trace_value_ids = model.predict_structured_trace(
            problem_ids, math_ids=math_ids
        )
        trace_op_texts = [
            operation_candidate_text(problem, op_row)
            or base
            for problem, op_row, base in zip(
                batch["problem"],
                trace_op_ids.detach().cpu().tolist(),
                base_texts,
            )
        ]
        candidate_slots.append(
            encode_answer_strings(model, tokenizer, trace_op_texts, max_answer_len, device)
        )
        candidate_texts.append(trace_op_texts)

        trace_texts = [
            str(class_to_value(int(idx))) for idx in trace_value_ids[:, -1].detach().cpu().tolist()
        ]
        candidate_slots.append(
            encode_answer_strings(model, tokenizer, trace_texts, max_answer_len, device)
        )
        candidate_texts.append(trace_texts)

    for _ in range(noise_candidates):
        noisy_slots = base_slots + noise_scale * torch.randn_like(base_slots)
        noisy_ids = model.readout.decode_ids(noisy_slots, pad_id=tokenizer.pad_id)
        noisy_texts = [tokenizer.decode(ids).strip() for ids in noisy_ids]
        candidate_slots.append(noisy_slots)
        candidate_texts.append(noisy_texts)

    return candidate_slots, candidate_texts


@torch.no_grad()
def verifier_batch(
    model: MathJEPAReadout,
    tokenizer,
    batch: dict,
    device: torch.device,
    noise_scale: float,
    candidate_set: str = "symbolic_full",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    problem_ids = batch["problem_ids"].to(device)
    answer_ids = batch["answer_ids"].to(device)
    math_ids = batch["math_ids"].to(device)
    context = model.encode_context(problem_ids)
    true_slots = model.encode_target(answer_ids, use_ema=True)
    wrong_slots = true_slots.roll(shifts=1, dims=0)
    candidate_slots, candidate_texts = model_candidate_texts(
        model,
        tokenizer,
        batch,
        device,
        noise_scale=noise_scale,
        noise_candidates=1,
        candidate_set=candidate_set,
    )

    contexts = [context, context]
    candidates = [true_slots, wrong_slots]
    labels = [torch.ones(context.size(0), device=device), torch.zeros(context.size(0), device=device)]
    for slots, texts in zip(candidate_slots, candidate_texts):
        contexts.append(context)
        candidates.append(slots)
        labels.append(
            torch.tensor(
                [float(text == answer) for text, answer in zip(texts, batch["answer"])],
                device=device,
            )
        )
    contexts = torch.cat(contexts, dim=0)
    candidates = torch.cat(candidates, dim=0)
    labels = torch.cat(labels, dim=0)
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
    candidate_set: str = "symbolic_full",
) -> dict[str, float]:
    verifier.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    total = 0
    pos_scores = []
    neg_scores = []
    for batch in loader:
        contexts, candidates, labels = verifier_batch(
            model, tokenizer, batch, device, noise_scale, candidate_set=candidate_set
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
    parser.add_argument(
        "--candidate-set",
        choices=["neural", "symbolic_no_oracle", "symbolic_full"],
        default="symbolic_full",
    )
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
                model,
                tokenizer,
                batch,
                device,
                noise_scale=args.noise_scale,
                candidate_set=args.candidate_set,
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
                    candidate_set=args.candidate_set,
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
