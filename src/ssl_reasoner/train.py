from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import CURRICULA, MATH_FEATURE_VOCAB_SIZE, MathDataset, generate_math_examples
from .diagnostics import latent_health
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


def _set_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


@torch.no_grad()
def predict_batch(model, batch, tokenizer, device, mode: str = "pred") -> list[str]:
    if mode == "pred":
        math_ids = batch.get("math_ids")
        decoded_ids = model.solve_ids(
            batch["problem_ids"].to(device),
            pad_id=tokenizer.pad_id,
            math_ids=math_ids.to(device) if math_ids is not None else None,
        )
    elif mode == "target":
        decoded_ids = model.solve_ids_from_target(
            batch["answer_ids"].to(device),
            pad_id=tokenizer.pad_id,
            use_ema=True,
        )
    else:
        raise ValueError(f"Unknown prediction mode: {mode}")
    return [tokenizer.decode(ids).strip() for ids in decoded_ids]


@torch.no_grad()
def predict_trace_batch(model, batch, tokenizer, device) -> list[str]:
    decoded_ids = model.solve_trace_ids(
        batch["problem_ids"].to(device),
        pad_id=tokenizer.pad_id,
        math_ids=batch["math_ids"].to(device),
    )
    return [tokenizer.decode(ids).strip() for ids in decoded_ids]


@torch.no_grad()
def evaluate(model, dataset, tokenizer, device, batch_size: int, mode: str = "pred") -> float:
    return evaluate_with_breakdown(model, dataset, tokenizer, device, batch_size, mode=mode)[
        "overall"
    ]


@torch.no_grad()
def evaluate_with_breakdown(
    model, dataset, tokenizer, device, batch_size: int, mode: str = "pred"
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    total = 0
    by_op: dict[str, list[int]] = {}
    by_split: dict[str, list[int]] = {}
    for batch in loader:
        predictions = predict_batch(model, batch, tokenizer, device, mode=mode)
        for pred, answer, op_label, split_label in zip(
            predictions, batch["answer"], batch["op_label"], batch["split_label"]
        ):
            is_correct = int(pred == answer)
            correct += is_correct
            total += 1
            counts = by_op.setdefault(str(op_label), [0, 0])
            counts[0] += is_correct
            counts[1] += 1
            split_counts = by_split.setdefault(str(split_label), [0, 0])
            split_counts[0] += is_correct
            split_counts[1] += 1
    stats = {"overall": correct / max(total, 1)}
    for op_label, counts in sorted(by_op.items()):
        stats[f"op_{op_label}"] = counts[0] / max(counts[1], 1)
    for split_label, counts in sorted(by_split.items()):
        stats[f"split_{split_label}"] = counts[0] / max(counts[1], 1)
    return stats


@torch.no_grad()
def evaluate_trace(model, dataset, tokenizer, device, batch_size: int) -> float:
    if not model.use_reasoning_trace:
        return 0.0
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    total = 0
    for batch in loader:
        predictions = predict_trace_batch(model, batch, tokenizer, device)
        for pred, trace in zip(predictions, batch["trace"]):
            correct += int(pred == trace)
            total += 1
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_structured_trace(model, dataset, device, batch_size: int) -> dict[str, float]:
    if not model.use_reasoning_trace:
        return {"trace_struct_op_acc": 0.0, "trace_struct_value_acc": 0.0}
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    op_correct = 0
    op_total = 0
    value_correct = 0.0
    value_total = 0.0
    for batch in loader:
        pred_ops, pred_values = model.predict_structured_trace(
            batch["problem_ids"].to(device),
            math_ids=batch["math_ids"].to(device),
        )
        op_ids = batch["trace_op_ids"].to(device)
        value_targets = batch["trace_value_ids"].to(device)
        value_mask = batch["trace_value_mask"].to(device)
        op_correct += int((pred_ops == op_ids).sum().item())
        op_total += int(op_ids.numel())
        value_correct += float(((pred_values == value_targets).float() * value_mask).sum().item())
        value_total += float(value_mask.sum().item())
    return {
        "trace_struct_op_acc": op_correct / max(op_total, 1),
        "trace_struct_value_acc": value_correct / max(value_total, 1.0),
    }


@torch.no_grad()
def evaluate_structured_answer(model, dataset, device, batch_size: int) -> float:
    if not model.use_reasoning_trace:
        return 0.0
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    total = 0
    for batch in loader:
        pred_ids, _ = model.predict_structured_answer(
            batch["problem_ids"].to(device),
            math_ids=batch["math_ids"].to(device),
        )
        target_ids = batch["answer_value_id"].to(device)
        correct += int((pred_ids == target_ids).sum().item())
        total += int(target_ids.numel())
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_answer_value(model, dataset, device, batch_size: int) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    total = 0
    for batch in loader:
        pred_ids, _ = model.predict_answer_value(
            batch["problem_ids"].to(device),
            math_ids=batch["math_ids"].to(device),
        )
        target_ids = batch["answer_value_id"].to(device)
        correct += int((pred_ids == target_ids).sum().item())
        total += int(target_ids.numel())
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_trace_state_final(model, dataset, device, batch_size: int) -> float:
    if not model.use_reasoning_trace:
        return 0.0
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    total = 0
    for batch in loader:
        pred = model.predict_trace_state_values(
            batch["problem_ids"].to(device),
            math_ids=batch["math_ids"].to(device),
        ).round()
        values = batch["trace_state_values"].to(device)
        mask = batch["trace_state_mask"].to(device)
        final_target = torch.where(mask[:, 1].bool(), values[:, 1], values[:, 0])
        final_pred = torch.where(mask[:, 1].bool(), pred[:, 1], pred[:, 0])
        correct += int((final_pred == final_target).sum().item())
        total += int(final_target.numel())
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_step_state_final(model, dataset, device, batch_size: int) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    total = 0
    for batch in loader:
        pred = model.predict_step_state_values(batch["math_ids"].to(device)).round()
        values = batch["trace_state_values"].to(device)
        mask = batch["trace_state_mask"].to(device)
        final_target = torch.where(mask[:, 1].bool(), values[:, 1], values[:, 0])
        final_pred = torch.where(mask[:, 1].bool(), pred[:, 1], pred[:, 0])
        correct += int((final_pred == final_target).sum().item())
        total += int(final_target.numel())
    return correct / max(total, 1)


@torch.no_grad()
def format_samples(
    model,
    dataset,
    tokenizer,
    device,
    count: int = 5,
    mode: str = "pred",
) -> list[str]:
    model.eval()
    subset = [dataset[i] for i in range(min(count, len(dataset)))]
    batch = {
        "problem_ids": torch.stack([item["problem_ids"] for item in subset]),
        "answer_ids": torch.stack([item["answer_ids"] for item in subset]),
        "math_ids": torch.stack([item["math_ids"] for item in subset]),
        "answer": [item["answer"] for item in subset],
        "trace": [item["trace"] for item in subset],
        "problem": [item["problem"] for item in subset],
        "op_label": [item["op_label"] for item in subset],
        "split_label": [item["split_label"] for item in subset],
    }
    predictions = predict_batch(model, batch, tokenizer, device, mode=mode)
    rows = []
    for problem, pred, answer in zip(batch["problem"], predictions, batch["answer"]):
        mark = "ok" if pred == answer else "bad"
        rows.append(f"  {mark}: {problem} -> pred={pred!r} target={answer!r}")
    return rows


def save_checkpoint(model, args, tokenizer, output_dir: Path, best_acc: float) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "vocab_chars": tokenizer.chars,
            "best_acc": best_acc,
        },
        output_dir / "best.pt",
    )


def load_matching_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> int:
    current = model.state_dict()
    matched = {
        key: value
        for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }
    model.load_state_dict(matched, strict=False)
    return len(matched)


def run_stage(
    *,
    name: str,
    model: MathJEPAReadout,
    loader: DataLoader,
    val_dataset: MathDataset,
    tokenizer,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    steps: int,
    batch_size: int,
    eval_every: int,
    sample_count: int,
    output_dir: Path,
    args,
    best_acc: float,
) -> float:
    step = 0
    print(f"=== {name} ({steps} steps) ===")
    while step < steps:
        for batch in loader:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            problem_ids = batch["problem_ids"].to(device)
            answer_ids = batch["answer_ids"].to(device)
            answer_len = batch["answer_len"].to(device)
            math_ids = batch["math_ids"].to(device)
            trace_ids = batch["trace_ids"].to(device)
            trace_len = batch["trace_len"].to(device)
            trace_op_ids = batch["trace_op_ids"].to(device)
            trace_value_ids = batch["trace_value_ids"].to(device)
            trace_value_mask = batch["trace_value_mask"].to(device)
            trace_state_values = batch["trace_state_values"].to(device)
            trace_state_mask = batch["trace_state_mask"].to(device)
            answer_value_id = batch["answer_value_id"].to(device)

            if name == "stage0_target_warmup":
                out = model.stage0_target_autoencode(
                    answer_ids,
                    answer_len,
                    trace_ids=trace_ids,
                    trace_len=trace_len,
                    trace_weight=args.trace_weight,
                )
            elif name == "stage1_predictor":
                out = model.stage1_predictor(
                    problem_ids,
                    answer_ids,
                    math_ids=math_ids,
                    trace_ids=trace_ids,
                    trace_len=trace_len,
                    trace_op_ids=trace_op_ids,
                    trace_value_ids=trace_value_ids,
                    trace_value_mask=trace_value_mask,
                    trace_state_values=trace_state_values,
                    trace_state_mask=trace_state_mask,
                    answer_value_id=answer_value_id,
                    trace_weight=args.trace_weight,
                    trace_struct_weight=args.trace_struct_weight,
                    reasoning_struct_weight=args.reasoning_struct_weight,
                    structured_answer_weight=args.structured_answer_weight,
                    answer_value_weight=args.answer_value_weight,
                    answer_contrastive_weight=args.answer_contrastive_weight,
                    contrastive_weight=args.contrastive_weight,
                    vicreg_weight=args.vicreg_weight,
                    slot_diversity_weight=args.slot_diversity_weight,
                    batch_diversity_weight=args.batch_diversity_weight,
                )
            elif name == "stage2_readout":
                progress = step / max(steps - 1, 1)
                true_ratio = max(0.0, args.true_latent_ratio * (1.0 - progress))
                out = model.stage2_readout(
                    problem_ids,
                    answer_ids,
                    answer_len,
                    math_ids=math_ids,
                    true_latent_ratio=true_ratio,
                    target_readout_weight=args.target_readout_weight,
                )
            elif name == "stage3_joint":
                out = model.stage3_joint(
                    problem_ids,
                    answer_ids,
                    answer_len,
                    math_ids=math_ids,
                    trace_ids=trace_ids,
                    trace_len=trace_len,
                    trace_op_ids=trace_op_ids,
                    trace_value_ids=trace_value_ids,
                    trace_value_mask=trace_value_mask,
                    trace_state_values=trace_state_values,
                    trace_state_mask=trace_state_mask,
                    answer_value_id=answer_value_id,
                    pred_weight=args.joint_pred_weight,
                    contrastive_weight=args.joint_contrastive_weight,
                    vicreg_weight=args.joint_vicreg_weight,
                    token_weight=args.joint_token_weight,
                    length_weight=args.joint_length_weight,
                    trace_weight=args.trace_weight,
                    trace_struct_weight=args.trace_struct_weight,
                    reasoning_struct_weight=args.reasoning_struct_weight,
                    structured_answer_weight=args.structured_answer_weight,
                    answer_value_weight=args.answer_value_weight,
                    answer_contrastive_weight=args.answer_contrastive_weight,
                    slot_diversity_weight=args.slot_diversity_weight,
                    batch_diversity_weight=args.batch_diversity_weight,
                )
            elif name == "reasoning_head":
                with torch.no_grad():
                    slots = model.predict_answer_slots(problem_ids, math_ids)
                out = model.reasoning_struct_loss(
                    slots,
                    trace_op_ids,
                    trace_value_ids,
                    trace_value_mask,
                )
            elif name == "trace_ops_head":
                with torch.no_grad():
                    slots = model.predict_trace_slots(problem_ids, math_ids)
                out = model.structured_trace_loss(
                    slots,
                    trace_op_ids,
                    trace_value_ids,
                    trace_value_mask,
                )
                if args.trace_state_weight > 0:
                    state_out = model.trace_state_regression_loss(
                        slots,
                        trace_state_values,
                        trace_state_mask,
                    )
                    out["loss"] = out["loss"] + args.trace_state_weight * state_out["loss"]
                    out["trace_state_regression_loss"] = state_out[
                        "trace_state_regression_loss"
                    ]
                    out["trace_state_regression_acc"] = state_out[
                        "trace_state_regression_acc"
                    ]
                    out["trace_state_final_acc"] = state_out["trace_state_final_acc"]
            elif name == "trace_state_head":
                with torch.no_grad():
                    slots = model.predict_trace_slots(problem_ids, math_ids)
                out = model.trace_state_regression_loss(
                    slots,
                    trace_state_values,
                    trace_state_mask,
                )
            elif name == "step_state_head":
                out = model.step_state_loss(
                    math_ids,
                    trace_state_values,
                    trace_state_mask,
                    trace_op_ids,
                )
            elif name == "answer_value_head":
                with torch.no_grad():
                    slots = model.predict_answer_slots(problem_ids, math_ids)
                out = model.answer_value_loss(slots, answer_value_id)
            else:
                raise ValueError(f"Unknown stage: {name}")

            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip
            )
            optimizer.step()

            if name in {"stage0_target_warmup", "stage3_joint"}:
                model.ema_update_target_encoder(args.ema_decay)

            step += 1
            if step == 1 or step % eval_every == 0 or step == steps:
                pred_stats = evaluate_with_breakdown(
                    model, val_dataset, tokenizer, device, batch_size, mode="pred"
                )
                pred_acc = pred_stats["overall"]
                target_acc = evaluate(
                    model, val_dataset, tokenizer, device, batch_size, mode="target"
                )
                trace_acc = evaluate_trace(model, val_dataset, tokenizer, device, batch_size)
                trace_struct = evaluate_structured_trace(model, val_dataset, device, batch_size)
                structured_answer_acc = evaluate_structured_answer(
                    model, val_dataset, device, batch_size
                )
                answer_value_acc = evaluate_answer_value(
                    model, val_dataset, device, batch_size
                )
                trace_state_final_acc = evaluate_trace_state_final(
                    model, val_dataset, device, batch_size
                )
                step_state_final_acc = evaluate_step_state_final(
                    model, val_dataset, device, batch_size
                )
                metrics = " ".join(
                    f"{key}={value.item():.4f}"
                    for key, value in out.items()
                    if key.endswith("loss")
                    or key in {
                        "trace_struct_op_acc",
                        "trace_struct_value_acc",
                        "reasoning_struct_op_acc",
                        "reasoning_struct_value_acc",
                        "structured_answer_acc",
                        "trace_state_regression_acc",
                        "trace_state_final_acc",
                        "step_state_final_acc",
                        "answer_value_acc",
                    }
                )
                print(
                    f"{name} step={step} {metrics} "
                    f"pred_exact={pred_acc:.3f} target_exact={target_acc:.3f} "
                    f"trace_exact={trace_acc:.3f} "
                    f"trace_struct_op_acc={trace_struct['trace_struct_op_acc']:.3f} "
                    f"trace_struct_value_acc={trace_struct['trace_struct_value_acc']:.3f} "
                    f"structured_answer_acc={structured_answer_acc:.3f} "
                    f"answer_value_acc={answer_value_acc:.3f} "
                    f"trace_state_final_acc={trace_state_final_acc:.3f} "
                    f"step_state_final_acc={step_state_final_acc:.3f}"
                )
                op_metrics = " ".join(
                    f"{key}={value:.3f}"
                    for key, value in pred_stats.items()
                    if key.startswith("op_")
                )
                split_metrics = " ".join(
                    f"{key}={value:.3f}"
                    for key, value in pred_stats.items()
                    if key.startswith("split_")
                )
                breakdown = " ".join(part for part in [op_metrics, split_metrics] if part)
                if breakdown:
                    print(f"  pred_breakdown {breakdown}")
                if name in {
                    "stage1_predictor",
                    "stage3_joint",
                    "reasoning_head",
                    "trace_ops_head",
                    "trace_state_head",
                    "step_state_head",
                    "answer_value_head",
                }:
                    diag = latent_health(model, val_dataset, device, batch_size)
                    print(
                        "  latent_health "
                        f"mean_random_cosine={diag['mean_random_cosine']:.3f} "
                        f"effective_rank_ratio={diag['effective_rank_ratio']:.3f} "
                        f"passes_cosine={bool(diag['passes_cosine'])} "
                        f"passes_rank={bool(diag['passes_rank'])}"
                    )
                for row in format_samples(
                    model, val_dataset, tokenizer, device, sample_count, mode="pred"
                ):
                    print(row)
                if pred_acc > best_acc:
                    best_acc = pred_acc
                    save_checkpoint(model, args, tokenizer, output_dir, best_acc)
            if step >= steps:
                break
    if name == "stage0_target_warmup":
        model.sync_target_encoder_ema()
    return best_acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None, help="Total steps split 20/50/20/10.")
    parser.add_argument("--stage0-steps", type=int, default=None)
    parser.add_argument("--stage1-steps", type=int, default=None)
    parser.add_argument("--stage2-steps", type=int, default=None)
    parser.add_argument("--stage3-steps", type=int, default=None)
    parser.add_argument("--reasoning-head-steps", type=int, default=None)
    parser.add_argument("--trace-ops-head-steps", type=int, default=None)
    parser.add_argument("--answer-value-head-steps", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument(
        "--train-curriculum",
        choices=CURRICULA,
        default="mixed",
    )
    parser.add_argument(
        "--val-curriculum",
        choices=CURRICULA,
        default="mixed",
    )
    parser.add_argument(
        "--train-easy-ratio",
        type=float,
        default=0.75,
        help="For mixed curriculum, probability of single-op examples.",
    )
    parser.add_argument(
        "--val-easy-ratio",
        type=float,
        default=0.75,
        help="For mixed curriculum, probability of single-op validation examples.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--predictor-layers", type=int, default=3)
    parser.add_argument("--predictor-type", choices=["pooled", "cross_attn"], default="pooled")
    parser.add_argument("--readout-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-problem-len", type=int, default=64)
    parser.add_argument("--max-answer-len", type=int, default=16)
    parser.add_argument("--max-math-len", type=int, default=8)
    parser.add_argument("--max-trace-len", type=int, default=32)
    parser.add_argument("--use-math-features", action="store_true")
    parser.add_argument("--use-reasoning-trace", action="store_true")
    parser.add_argument("--use-trace-fusion", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to resume from.")
    parser.add_argument(
        "--partial-checkpoint",
        action="store_true",
        help="Load only matching tensors, useful when increasing predictor capacity.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--contrastive-weight", type=float, default=0.5)
    parser.add_argument("--vicreg-weight", type=float, default=0.05)
    parser.add_argument("--slot-diversity-weight", type=float, default=0.1)
    parser.add_argument("--batch-diversity-weight", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--true-latent-ratio", type=float, default=1.0)
    parser.add_argument("--target-readout-weight", type=float, default=1.0)
    parser.add_argument("--trace-weight", type=float, default=0.5)
    parser.add_argument("--trace-struct-weight", type=float, default=1.0)
    parser.add_argument("--trace-state-weight", type=float, default=0.0)
    parser.add_argument("--reasoning-struct-weight", type=float, default=0.1)
    parser.add_argument("--structured-answer-weight", type=float, default=1.0)
    parser.add_argument("--answer-value-weight", type=float, default=1.0)
    parser.add_argument("--answer-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--joint-pred-weight", type=float, default=1.0)
    parser.add_argument("--joint-contrastive-weight", type=float, default=0.1)
    parser.add_argument("--joint-vicreg-weight", type=float, default=0.05)
    parser.add_argument("--joint-token-weight", type=float, default=0.5)
    parser.add_argument("--joint-length-weight", type=float, default=0.1)
    parser.add_argument(
        "--stages",
        default="0,1,2",
        help="Comma-separated stage ids to run. Use 0,1,2,3 for the full plan pipeline.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    requested_stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    if args.steps is not None:
        args.stage0_steps = args.stage0_steps or max(1, int(args.steps * 0.2))
        args.stage1_steps = args.stage1_steps or max(1, int(args.steps * 0.5))
        if "3" in requested_stages:
            args.stage2_steps = args.stage2_steps or max(1, int(args.steps * 0.2))
            args.stage3_steps = args.stage3_steps or max(
                1, args.steps - args.stage0_steps - args.stage1_steps - args.stage2_steps
            )
        else:
            args.stage2_steps = args.stage2_steps or max(
                1, args.steps - args.stage0_steps - args.stage1_steps
            )
            args.stage3_steps = args.stage3_steps or 300
    else:
        args.stage0_steps = args.stage0_steps or 200
        args.stage1_steps = args.stage1_steps or 500
        args.stage2_steps = args.stage2_steps or 300
        args.stage3_steps = args.stage3_steps or 300
        args.reasoning_head_steps = args.reasoning_head_steps or 500
        args.trace_ops_head_steps = args.trace_ops_head_steps or 500
        args.answer_value_head_steps = args.answer_value_head_steps or 500

    torch.manual_seed(args.seed)
    device = _device(args.device)
    tokenizer = build_math_tokenizer()
    checkpoint = torch.load(args.checkpoint, map_location=device) if args.checkpoint else None
    if checkpoint is not None:
        ckpt_args = checkpoint.get("args", {})
        args.max_problem_len = ckpt_args.get("max_problem_len", args.max_problem_len)
        args.max_answer_len = ckpt_args.get("max_answer_len", args.max_answer_len)
        args.max_math_len = ckpt_args.get("max_math_len", args.max_math_len)
        args.max_trace_len = ckpt_args.get("max_trace_len", args.max_trace_len)
        args.d_model = ckpt_args.get("d_model", args.d_model)
        args.num_slots = ckpt_args.get("num_slots", args.num_slots)
        args.encoder_layers = ckpt_args.get("encoder_layers", args.encoder_layers)
        args.readout_layers = ckpt_args.get("readout_layers", args.readout_layers)
        args.num_heads = ckpt_args.get("num_heads", args.num_heads)
        if not args.partial_checkpoint:
            args.predictor_layers = ckpt_args.get("predictor_layers", args.predictor_layers)
            args.predictor_type = ckpt_args.get("predictor_type", args.predictor_type)
            args.use_math_features = ckpt_args.get("use_math_features", args.use_math_features)
            args.use_reasoning_trace = ckpt_args.get(
                "use_reasoning_trace", args.use_reasoning_trace
            )
            args.use_trace_fusion = ckpt_args.get("use_trace_fusion", args.use_trace_fusion)

    train_examples = generate_math_examples(
        args.train_size,
        seed=args.seed,
        difficulty_mix=(args.train_easy_ratio, 1.0 - args.train_easy_ratio),
        curriculum=args.train_curriculum,
    )
    val_examples = (
        train_examples
        if args.overfit
        else generate_math_examples(
            args.val_size,
            seed=args.seed + 1,
            difficulty_mix=(args.val_easy_ratio, 1.0 - args.val_easy_ratio),
            curriculum=args.val_curriculum,
        )
    )
    train_dataset = MathDataset(
        train_examples,
        tokenizer,
        args.max_problem_len,
        args.max_answer_len,
        args.max_math_len,
        args.max_trace_len,
    )
    val_dataset = MathDataset(
        val_examples,
        tokenizer,
        args.max_problem_len,
        args.max_answer_len,
        args.max_math_len,
        args.max_trace_len,
    )
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = MathJEPAReadout(
        vocab_size=tokenizer.vocab_size,
        max_problem_len=args.max_problem_len,
        max_answer_len=args.max_answer_len,
        d_model=args.d_model,
        num_slots=args.num_slots,
        encoder_layers=args.encoder_layers,
        predictor_layers=args.predictor_layers,
        readout_layers=args.readout_layers,
        num_heads=args.num_heads,
        predictor_type=args.predictor_type,
        use_math_features=args.use_math_features,
        math_vocab_size=MATH_FEATURE_VOCAB_SIZE,
        max_math_len=args.max_math_len,
        use_reasoning_trace=args.use_reasoning_trace,
        max_trace_len=args.max_trace_len,
        use_trace_fusion=args.use_trace_fusion,
    ).to(device)
    if checkpoint is not None:
        if args.partial_checkpoint:
            matched = load_matching_state_dict(model, checkpoint["model"])
            print(f"partially loaded checkpoint={args.checkpoint} tensors={matched}")
        else:
            missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
            print(f"loaded checkpoint={args.checkpoint}")
            if missing or unexpected:
                print(f"checkpoint_non_strict missing={len(missing)} unexpected={len(unexpected)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    if "0" in requested_stages:
        _set_trainable(model.problem_encoder, False)
        _set_trainable(model.predictor, False)
        _set_trainable(model.target_encoder, True)
        _set_trainable(model.readout, True)
        stage0_params = list(model.target_encoder.parameters()) + list(model.readout.parameters())
        if args.use_reasoning_trace:
            _set_trainable(model.trace_predictor, False)
            _set_trainable(model.trace_target_encoder, True)
            _set_trainable(model.trace_readout, True)
            stage0_params += list(model.trace_target_encoder.parameters())
            stage0_params += list(model.trace_readout.parameters())
        optimizer = torch.optim.AdamW(stage0_params, lr=args.lr, weight_decay=0.01)
        best_acc = run_stage(
            name="stage0_target_warmup",
            model=model,
            loader=loader,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            device=device,
            optimizer=optimizer,
            steps=args.stage0_steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            sample_count=args.sample_count,
            output_dir=output_dir,
            args=args,
            best_acc=best_acc,
        )

    if "1" in requested_stages:
        _set_trainable(model.problem_encoder, True)
        _set_trainable(model.predictor, True)
        _set_trainable(model.target_encoder, False)
        _set_trainable(model.readout, False)
        _set_trainable(model.reasoning_struct_head, True)
        _set_trainable(model.answer_value_head, True)
        stage1_params = (
            list(model.problem_encoder.parameters())
            + list(model.predictor.parameters())
            + list(model.reasoning_struct_head.parameters())
            + list(model.answer_value_head.parameters())
        )
        if args.use_math_features:
            stage1_params += list(model.math_embed.parameters())
            stage1_params += [model.math_pos_embed]
            stage1_params += list(model.math_norm.parameters())
        if args.use_reasoning_trace:
            _set_trainable(model.trace_target_encoder, False)
            _set_trainable(model.trace_readout, False)
            _set_trainable(model.trace_predictor, True)
            _set_trainable(model.trace_struct_head, True)
            _set_trainable(model.structured_answer_head, True)
            _set_trainable(model.trace_state_regression_head, True)
            stage1_params += list(model.trace_predictor.parameters())
            stage1_params += list(model.trace_struct_head.parameters())
            stage1_params += list(model.structured_answer_head.parameters())
            stage1_params += list(model.trace_state_regression_head.parameters())
            if args.use_trace_fusion:
                _set_trainable(model.trace_struct_summary, True)
                _set_trainable(model.trace_fusion, True)
                stage1_params += list(model.trace_struct_summary.parameters())
                stage1_params += list(model.trace_fusion.parameters())
        optimizer = torch.optim.AdamW(
            stage1_params,
            lr=args.lr,
            weight_decay=0.01,
        )
        best_acc = run_stage(
            name="stage1_predictor",
            model=model,
            loader=loader,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            device=device,
            optimizer=optimizer,
            steps=args.stage1_steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            sample_count=args.sample_count,
            output_dir=output_dir,
            args=args,
            best_acc=best_acc,
        )

    if "2" in requested_stages:
        _set_trainable(model.problem_encoder, False)
        _set_trainable(model.predictor, False)
        _set_trainable(model.target_encoder, False)
        _set_trainable(model.readout, True)
        _set_trainable(model.reasoning_struct_head, False)
        if args.use_reasoning_trace:
            _set_trainable(model.trace_target_encoder, False)
            _set_trainable(model.trace_predictor, False)
            _set_trainable(model.trace_readout, False)
            _set_trainable(model.trace_struct_head, False)
            _set_trainable(model.structured_answer_head, False)
            if args.use_trace_fusion:
                _set_trainable(model.trace_struct_summary, False)
                _set_trainable(model.trace_fusion, False)
        optimizer = torch.optim.AdamW(model.readout.parameters(), lr=args.lr, weight_decay=0.01)
        best_acc = run_stage(
            name="stage2_readout",
            model=model,
            loader=loader,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            device=device,
            optimizer=optimizer,
            steps=args.stage2_steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            sample_count=args.sample_count,
            output_dir=output_dir,
            args=args,
            best_acc=best_acc,
        )

    if "3" in requested_stages:
        _set_trainable(model.problem_encoder, True)
        _set_trainable(model.predictor, True)
        _set_trainable(model.target_encoder, False)
        _set_trainable(model.readout, True)
        _set_trainable(model.reasoning_struct_head, True)
        _set_trainable(model.answer_value_head, True)
        stage3_params = (
            list(model.problem_encoder.parameters())
            + list(model.predictor.parameters())
            + list(model.readout.parameters())
            + list(model.reasoning_struct_head.parameters())
            + list(model.answer_value_head.parameters())
        )
        if args.use_math_features:
            stage3_params += list(model.math_embed.parameters())
            stage3_params += [model.math_pos_embed]
            stage3_params += list(model.math_norm.parameters())
        if args.use_reasoning_trace:
            _set_trainable(model.trace_target_encoder, False)
            _set_trainable(model.trace_predictor, True)
            _set_trainable(model.trace_readout, True)
            _set_trainable(model.trace_struct_head, True)
            _set_trainable(model.structured_answer_head, True)
            _set_trainable(model.trace_state_regression_head, True)
            stage3_params += list(model.trace_predictor.parameters())
            stage3_params += list(model.trace_readout.parameters())
            stage3_params += list(model.trace_struct_head.parameters())
            stage3_params += list(model.structured_answer_head.parameters())
            stage3_params += list(model.trace_state_regression_head.parameters())
            if args.use_trace_fusion:
                _set_trainable(model.trace_struct_summary, True)
                _set_trainable(model.trace_fusion, True)
                stage3_params += list(model.trace_struct_summary.parameters())
                stage3_params += list(model.trace_fusion.parameters())
        optimizer = torch.optim.AdamW(stage3_params, lr=args.lr, weight_decay=0.01)
        best_acc = run_stage(
            name="stage3_joint",
            model=model,
            loader=loader,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            device=device,
            optimizer=optimizer,
            steps=args.stage3_steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            sample_count=args.sample_count,
            output_dir=output_dir,
            args=args,
            best_acc=best_acc,
        )

    if "reasoning_head" in requested_stages or "rh" in requested_stages:
        for module in model.children():
            _set_trainable(module, False)
        _set_trainable(model.reasoning_struct_head, True)
        optimizer = torch.optim.AdamW(
            model.reasoning_struct_head.parameters(), lr=args.lr, weight_decay=0.01
        )
        best_acc = run_stage(
            name="reasoning_head",
            model=model,
            loader=loader,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            device=device,
            optimizer=optimizer,
            steps=args.reasoning_head_steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            sample_count=args.sample_count,
            output_dir=output_dir,
            args=args,
            best_acc=best_acc,
        )

    if "trace_ops_head" in requested_stages or "toh" in requested_stages:
        if not args.use_reasoning_trace:
            raise ValueError("trace_ops_head requires --use-reasoning-trace")
        for module in model.children():
            _set_trainable(module, False)
        _set_trainable(model.trace_struct_head, True)
        _set_trainable(model.trace_state_regression_head, args.trace_state_weight > 0)
        params = list(model.trace_struct_head.parameters())
        if args.trace_state_weight > 0:
            params += list(model.trace_state_regression_head.parameters())
        optimizer = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=0.01
        )
        best_acc = run_stage(
            name="trace_ops_head",
            model=model,
            loader=loader,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            device=device,
            optimizer=optimizer,
            steps=args.trace_ops_head_steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            sample_count=args.sample_count,
            output_dir=output_dir,
            args=args,
            best_acc=best_acc,
        )

    if "trace_state_head" in requested_stages or "tsh" in requested_stages:
        if not args.use_reasoning_trace:
            raise ValueError("trace_state_head requires --use-reasoning-trace")
        for module in model.children():
            _set_trainable(module, False)
        _set_trainable(model.trace_state_regression_head, True)
        optimizer = torch.optim.AdamW(
            model.trace_state_regression_head.parameters(), lr=args.lr, weight_decay=0.01
        )
        best_acc = run_stage(
            name="trace_state_head",
            model=model,
            loader=loader,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            device=device,
            optimizer=optimizer,
            steps=args.trace_ops_head_steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            sample_count=args.sample_count,
            output_dir=output_dir,
            args=args,
            best_acc=best_acc,
        )

    if "step_state_head" in requested_stages or "ssh" in requested_stages:
        for module in model.children():
            _set_trainable(module, False)
        _set_trainable(model.step_state_head, True)
        optimizer = torch.optim.AdamW(
            model.step_state_head.parameters(), lr=args.lr, weight_decay=0.01
        )
        best_acc = run_stage(
            name="step_state_head",
            model=model,
            loader=loader,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            device=device,
            optimizer=optimizer,
            steps=args.trace_ops_head_steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            sample_count=args.sample_count,
            output_dir=output_dir,
            args=args,
            best_acc=best_acc,
        )

    if "answer_value_head" in requested_stages or "avh" in requested_stages:
        for module in model.children():
            _set_trainable(module, False)
        _set_trainable(model.answer_value_head, True)
        optimizer = torch.optim.AdamW(
            model.answer_value_head.parameters(), lr=args.lr, weight_decay=0.01
        )
        best_acc = run_stage(
            name="answer_value_head",
            model=model,
            loader=loader,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            device=device,
            optimizer=optimizer,
            steps=args.answer_value_head_steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            sample_count=args.sample_count,
            output_dir=output_dir,
            args=args,
            best_acc=best_acc,
        )

    save_checkpoint(model, args, tokenizer, output_dir, best_acc)
    print(f"done best_val_exact={best_acc:.3f} checkpoint={output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
