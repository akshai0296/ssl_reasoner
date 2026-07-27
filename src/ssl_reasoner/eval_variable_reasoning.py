from __future__ import annotations

import argparse
import random
import re

import torch

from .data import (
    MathExample,
    encode_math_features,
    extract_math_expression,
    generate_math_examples,
    make_trace,
    make_reasoning_text,
    make_variable_trace_fields,
    safe_eval_expression,
)
from .solver import _device, load_math_solver


TRACE_STEP_RE = re.compile(r"^(-?\d+)([+\-*])(-?\d+)=(-?\d+)$")
EVAL_PRESETS = (
    "in_dist",
    "larger_numbers",
    "longer_expr",
    "no_multiply",
    "many_multiply",
    "length_3",
    "length_5",
    "length_8",
    "length_16",
)


def trace_is_equivalent(expr: str, trace: str) -> bool:
    parts = re.split(r"([+\-*])", expr)
    values = [int(parts[idx]) for idx in range(0, len(parts), 2)]
    ops = [parts[idx] for idx in range(1, len(parts), 2)]
    trace_parts = [part for part in trace.split(",") if part]
    if len(trace_parts) != len(ops) + 1:
        return False

    for step_text in trace_parts[:-1]:
        match = TRACE_STEP_RE.match(step_text)
        if match is None:
            return False
        lhs, op, rhs, result = (
            int(match.group(1)),
            match.group(2),
            int(match.group(3)),
            int(match.group(4)),
        )
        if "*" in ops and op != "*":
            return False
        if op == "+":
            expected = lhs + rhs
        elif op == "-":
            expected = lhs - rhs
        elif op == "*":
            expected = lhs * rhs
        else:
            return False
        if result != expected:
            return False

        reduce_idx = None
        for idx, current_op in enumerate(ops):
            if values[idx] == lhs and current_op == op and values[idx + 1] == rhs:
                reduce_idx = idx
                break
        if reduce_idx is None:
            return False
        values[reduce_idx : reduce_idx + 2] = [result]
        del ops[reduce_idx]

    if len(values) != 1:
        return False
    try:
        final = int(trace_parts[-1])
    except ValueError:
        return False
    return final == values[0] == safe_eval_expression(expr)


def trace_final_value(trace: str) -> int | None:
    parts = [part for part in trace.split(",") if part]
    if not parts:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def trace_step_values(trace: str) -> list[list[int]]:
    values = []
    for part in [part for part in trace.split(",") if part][:-1]:
        match = TRACE_STEP_RE.match(part)
        if match is None:
            continue
        values.append([int(match.group(1)), int(match.group(3)), int(match.group(4))])
    return values


def trace_step_dicts(trace: str) -> list[dict[str, int | str]]:
    steps = []
    for part in [part for part in trace.split(",") if part][:-1]:
        match = TRACE_STEP_RE.match(part)
        if match is None:
            continue
        steps.append(
            {
                "lhs": int(match.group(1)),
                "op": match.group(2),
                "rhs": int(match.group(3)),
                "result": int(match.group(4)),
            }
        )
    return steps


def _expression_state(expr: str) -> tuple[list[int], list[str]]:
    parts = re.split(r"([+\-*])", expr)
    values = [int(parts[idx]) for idx in range(0, len(parts), 2)]
    ops = [parts[idx] for idx in range(1, len(parts), 2)]
    return values, ops


def _apply_trace_step(
    values: list[int],
    ops: list[str],
    step: dict[str, int | str],
) -> bool:
    lhs = int(step["lhs"])
    rhs = int(step["rhs"])
    op = str(step["op"])
    result = int(step["result"])
    for idx, current_op in enumerate(ops):
        if values[idx] == lhs and values[idx + 1] == rhs and current_op == op:
            values[idx : idx + 2] = [result]
            del ops[idx]
            return True
    return False


def trace_rollout_metrics(expr: str, pred_trace: str) -> dict[str, int]:
    target_steps = trace_step_dicts(make_reasoning_text(expr))
    pred_steps = trace_step_dicts(pred_trace)
    target_values, target_ops = _expression_state(expr)
    pred_values, pred_ops = _expression_state(expr)
    stats = {
        "steps": len(target_steps),
        "pred_steps": len(pred_steps),
        "lhs": 0,
        "rhs": 0,
        "op": 0,
        "operand_pair": 0,
        "slot_triple": 0,
        "result": 0,
        "arithmetic_valid": 0,
        "post_state": 0,
    }
    for target_step, pred_step in zip(target_steps, pred_steps):
        lhs_ok = pred_step["lhs"] == target_step["lhs"]
        rhs_ok = pred_step["rhs"] == target_step["rhs"]
        op_ok = pred_step["op"] == target_step["op"]
        result_ok = pred_step["result"] == target_step["result"]
        stats["lhs"] += int(lhs_ok)
        stats["rhs"] += int(rhs_ok)
        stats["op"] += int(op_ok)
        stats["operand_pair"] += int(lhs_ok and rhs_ok)
        stats["slot_triple"] += int(lhs_ok and rhs_ok and op_ok)
        stats["result"] += int(result_ok)
        lhs = int(pred_step["lhs"])
        rhs = int(pred_step["rhs"])
        op = str(pred_step["op"])
        result = int(pred_step["result"])
        expected = (
            lhs + rhs
            if op == "+"
            else lhs - rhs
            if op == "-"
            else lhs * rhs
            if op == "*"
            else None
        )
        stats["arithmetic_valid"] += int(expected == result)
        _apply_trace_step(target_values, target_ops, target_step)
        pred_applied = _apply_trace_step(pred_values, pred_ops, pred_step)
        stats["post_state"] += int(
            pred_applied and pred_values == target_values and pred_ops == target_ops
        )
    return stats


def make_eval_example(expr: str, rng: random.Random, split_label: str) -> MathExample:
    template = rng.choice(
        [
            "What is {expr}?",
            "Calculate {expr}.",
            "Find the value of {expr}.",
        ]
    )
    return MathExample(
        problem=template.format(expr=expr),
        answer=str(safe_eval_expression(expr)),
        op_label=split_label,
        difficulty=2,
        trace=make_trace(expr),
        split_label=split_label,
    )


def generate_ood_examples(samples: int, seed: int, preset: str) -> list[MathExample]:
    if preset == "in_dist":
        return generate_math_examples(samples, seed=seed, curriculum="multi_step")
    if preset not in EVAL_PRESETS:
        raise ValueError(f"Unknown eval preset: {preset!r}")

    rng = random.Random(seed)
    examples = []
    for _ in range(samples):
        if preset == "larger_numbers":
            operands = [rng.randint(51, 120)]
            operands.extend(rng.randint(21, 60) for _ in range(3))
            ops = [rng.choice(["+", "-", "*"]) for _ in range(3)]
            if "*" not in ops:
                ops[rng.randrange(len(ops))] = "*"
        elif preset == "longer_expr":
            operands = [rng.randint(0, 50)]
            operands.extend(rng.randint(0, 20) for _ in range(4))
            ops = [rng.choice(["+", "-", "*"]) for _ in range(4)]
            if "*" not in ops:
                ops[rng.randrange(len(ops))] = "*"
        elif preset == "no_multiply":
            operands = [rng.randint(0, 50)]
            operands.extend(rng.randint(0, 20) for _ in range(3))
            ops = [rng.choice(["+", "-"]) for _ in range(3)]
        elif preset == "many_multiply":
            operands = [rng.randint(0, 20) for _ in range(4)]
            ops = [rng.choice(["+", "-", "*"]) for _ in range(3)]
            multiply_positions = rng.sample(range(3), k=2)
            for idx in multiply_positions:
                ops[idx] = "*"
        elif preset.startswith("length_"):
            num_ops = int(preset.split("_", maxsplit=1)[1])
            operands = [rng.randint(0, 50)]
            operands.extend(rng.randint(0, 20) for _ in range(num_ops))
            ops = [rng.choice(["+", "-", "*"]) for _ in range(num_ops)]
            if "*" not in ops:
                ops[rng.randrange(len(ops))] = "*"
        expr = "".join(f"{value}{op}" for value, op in zip(operands, ops)) + str(
            operands[-1]
        )
        examples.append(make_eval_example(expr, rng, preset))
    return examples


def evaluate_plan_order(
    model,
    examples: list[MathExample],
    train_args: dict,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """Score the H-JEPA plan level on reduction *order*, ignoring numeric values.

    Isolates "did the plan pick the right operator sequence and trace length" from
    the separate, harder problem of computing exact intermediate values. A high
    plan-order score on the long presets is the signal that the hierarchy generalizes
    even when the numeric decode does not.
    """
    max_math_len = train_args.get("max_math_len", 8)
    max_steps = train_args.get("max_variable_steps", 4)
    all_math_ids = torch.tensor(
        [encode_math_features(example.problem, max_math_len) for example in examples],
        dtype=torch.long,
        device=device,
    )
    pred_ops_chunks = []
    pred_active_chunks = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch_math_ids = all_math_ids[start : start + batch_size]
            _, op_logits, active_logits = model.hjepa_reasoner(batch_math_ids)
            pred_ops_chunks.append(op_logits.argmax(dim=-1).cpu())
            pred_active_chunks.append((active_logits.sigmoid() >= 0.5).cpu())
    pred_ops = torch.cat(pred_ops_chunks, dim=0)
    pred_active = torch.cat(pred_active_chunks, dim=0)

    op_step_correct = 0
    op_step_total = 0
    length_exact = 0
    sequence_exact = 0
    for idx, example in enumerate(examples):
        expr = extract_math_expression(example.problem)
        target_op_ids, _, _, _, target_mask = make_variable_trace_fields(
            expr, max_steps=max_steps
        )
        target_ops = torch.tensor(target_op_ids)
        active = torch.tensor(target_mask).bool()
        row_ops = pred_ops[idx][: active.numel()]
        row_active = pred_active[idx][: active.numel()]
        op_matches = row_ops[active] == target_ops[active]
        op_step_correct += int(op_matches.sum().item())
        op_step_total += int(active.sum().item())
        length_ok = bool((row_active == active).all().item())
        length_exact += int(length_ok)
        sequence_exact += int(length_ok and bool(op_matches.all().item()))

    total = max(len(examples), 1)
    return {
        "plan_order_op_step_exact": op_step_correct / max(op_step_total, 1),
        "plan_order_length_exact": length_exact / total,
        "plan_order_sequence_exact": sequence_exact / total,
    }


def evaluate_preset(
    args: argparse.Namespace,
    model,
    tokenizer,
    train_args: dict,
    device: torch.device,
    preset: str,
) -> None:
    examples = generate_ood_examples(args.samples, args.seed, preset)

    answer_correct = 0
    trace_correct = 0
    trace_equiv_correct = 0
    learned_step_value_correct = 0
    learned_step_value_total = 0
    learned_final_value_correct = 0
    copy_update_lhs_correct = 0
    copy_update_rhs_correct = 0
    copy_update_op_correct = 0
    copy_update_operand_pair_correct = 0
    copy_update_slot_triple_correct = 0
    copy_update_result_correct = 0
    copy_update_arithmetic_valid = 0
    copy_update_post_state_correct = 0
    copy_update_step_total = 0
    copy_update_pred_step_total = 0
    shown = 0
    learned_traces: list[str] | None = None
    if (
        args.learned
        or args.learned_values
        or args.raw_learned_values
        or args.digit_learned_values
        or args.class_learned_values
        or args.standalone_learned_values
        or args.standalone_digit_values
        or args.standalone_factor_values
        or args.standalone_decomposed_values
        or args.standalone_hybrid_values
        or args.latent_reasoning_values
        or args.latent_reasoning_digit_values
        or args.latent_reasoning_process_values
        or args.latent_reasoning_state_values
        or args.latent_reasoning_slot_values
        or args.latent_reasoning_slot_transition_values
        or args.latent_reasoning_slot_class_values
        or args.latent_reasoning_slot_process_values
        or args.latent_reasoning_slot_digit_values
        or args.latent_reasoning_copy_update_values
        or args.hjepa_values
    ):
        max_math_len = train_args.get("max_math_len", 8)
        all_math_ids = torch.tensor(
            [encode_math_features(example.problem, max_math_len) for example in examples],
            dtype=torch.long,
            device=device,
        )
        learned_traces = []
        for start in range(0, len(examples), args.batch_size):
            batch_math_ids = all_math_ids[start : start + args.batch_size]
            learned_traces.extend(
                model.solve_variable_reasoning_texts(
                    batch_math_ids,
                    constrain_to_legal=not args.unconstrained,
                    learned_values=args.learned_values,
                    raw_learned_values=args.raw_learned_values,
                    digit_learned_values=args.digit_learned_values,
                    class_learned_values=args.class_learned_values,
                    standalone_learned_values=args.standalone_learned_values,
                    standalone_digit_values=args.standalone_digit_values,
                    standalone_factor_values=args.standalone_factor_values,
                    standalone_decomposed_values=args.standalone_decomposed_values,
                    standalone_hybrid_values=args.standalone_hybrid_values,
                    latent_reasoning_values=args.latent_reasoning_values,
                    latent_reasoning_digit_values=args.latent_reasoning_digit_values,
                    latent_reasoning_process_values=args.latent_reasoning_process_values,
                    latent_reasoning_state_values=args.latent_reasoning_state_values,
                    latent_reasoning_slot_values=args.latent_reasoning_slot_values,
                    latent_reasoning_slot_transition_values=(
                        args.latent_reasoning_slot_transition_values
                    ),
                    latent_reasoning_slot_class_values=(
                        args.latent_reasoning_slot_class_values
                    ),
                    latent_reasoning_slot_process_values=(
                        args.latent_reasoning_slot_process_values
                    ),
                    latent_reasoning_slot_digit_values=(
                        args.latent_reasoning_slot_digit_values
                    ),
                    latent_reasoning_copy_update_values=(
                        args.latent_reasoning_copy_update_values
                    ),
                    hjepa_values=args.hjepa_values,
                )
            )

    for idx, example in enumerate(examples):
        expr = extract_math_expression(example.problem)
        target_trace = make_reasoning_text(expr)
        pred_trace = learned_traces[idx] if learned_traces is not None else target_trace
        pred_answer = str(trace_final_value(pred_trace))
        answer_ok = pred_answer == example.answer
        trace_ok = pred_trace == target_trace
        trace_equiv_ok = trace_is_equivalent(expr, pred_trace)
        answer_correct += int(answer_ok)
        trace_correct += int(trace_ok)
        trace_equiv_correct += int(trace_equiv_ok)
        if args.learned_values:
            _, _, target_values, _, target_mask = make_variable_trace_fields(
                expr,
                max_steps=train_args.get("max_variable_steps", 4),
            )
            for pred_values, expected_values, is_active in zip(
                trace_step_values(pred_trace),
                target_values,
                target_mask,
            ):
                if not is_active:
                    continue
                learned_step_value_total += 3
                learned_step_value_correct += sum(
                    int(pred == round(expected))
                    for pred, expected in zip(pred_values, expected_values)
                )
            learned_final_value_correct += int(trace_final_value(pred_trace) == int(example.answer))
        if args.latent_reasoning_copy_update_values:
            rollout = trace_rollout_metrics(expr, pred_trace)
            copy_update_lhs_correct += rollout["lhs"]
            copy_update_rhs_correct += rollout["rhs"]
            copy_update_op_correct += rollout["op"]
            copy_update_operand_pair_correct += rollout["operand_pair"]
            copy_update_slot_triple_correct += rollout["slot_triple"]
            copy_update_result_correct += rollout["result"]
            copy_update_arithmetic_valid += rollout["arithmetic_valid"]
            copy_update_post_state_correct += rollout["post_state"]
            copy_update_step_total += rollout["steps"]
            copy_update_pred_step_total += rollout["pred_steps"]
        if (not answer_ok or not trace_equiv_ok) and shown < args.dump_errors:
            print(
                f"bad: {example.problem} -> answer={pred_answer!r}/{example.answer!r} "
                f"trace={pred_trace!r}/{target_trace!r}"
            )
            shown += 1

    total = max(len(examples), 1)
    print(f"preset={preset}")
    print(f"variable_reasoning_answer_exact={answer_correct / total:.3f} ({answer_correct}/{total})")
    print(f"variable_reasoning_trace_exact={trace_correct / total:.3f} ({trace_correct}/{total})")
    print(
        f"variable_reasoning_trace_equiv_exact="
        f"{trace_equiv_correct / total:.3f} ({trace_equiv_correct}/{total})"
    )
    if args.hjepa_plan_order:
        plan_stats = evaluate_plan_order(
            model, examples, train_args, device, args.batch_size
        )
        print(
            f"plan_order_op_step_exact={plan_stats['plan_order_op_step_exact']:.3f} "
            f"plan_order_length_exact={plan_stats['plan_order_length_exact']:.3f} "
            f"plan_order_sequence_exact={plan_stats['plan_order_sequence_exact']:.3f}"
        )
    if args.learned_values:
        value_total = max(learned_step_value_total, 1)
        print(
            f"variable_reasoning_learned_step_value_exact="
            f"{learned_step_value_correct / value_total:.3f} "
            f"({learned_step_value_correct}/{learned_step_value_total})"
        )
        print(
            f"variable_reasoning_learned_final_value_exact="
            f"{learned_final_value_correct / total:.3f} "
            f"({learned_final_value_correct}/{total})"
        )
    if args.latent_reasoning_copy_update_values:
        step_total = max(copy_update_step_total, 1)
        print(
            f"copy_update_step_count_ratio="
            f"{copy_update_pred_step_total / step_total:.3f} "
            f"({copy_update_pred_step_total}/{copy_update_step_total})"
        )
        print(
            f"copy_update_lhs_exact="
            f"{copy_update_lhs_correct / step_total:.3f} "
            f"({copy_update_lhs_correct}/{copy_update_step_total})"
        )
        print(
            f"copy_update_rhs_exact="
            f"{copy_update_rhs_correct / step_total:.3f} "
            f"({copy_update_rhs_correct}/{copy_update_step_total})"
        )
        print(
            f"copy_update_op_exact="
            f"{copy_update_op_correct / step_total:.3f} "
            f"({copy_update_op_correct}/{copy_update_step_total})"
        )
        print(
            f"copy_update_operand_pair_exact="
            f"{copy_update_operand_pair_correct / step_total:.3f} "
            f"({copy_update_operand_pair_correct}/{copy_update_step_total})"
        )
        print(
            f"copy_update_slot_triple_exact="
            f"{copy_update_slot_triple_correct / step_total:.3f} "
            f"({copy_update_slot_triple_correct}/{copy_update_step_total})"
        )
        print(
            f"copy_update_result_exact="
            f"{copy_update_result_correct / step_total:.3f} "
            f"({copy_update_result_correct}/{copy_update_step_total})"
        )
        print(
            f"copy_update_arithmetic_valid="
            f"{copy_update_arithmetic_valid / step_total:.3f} "
            f"({copy_update_arithmetic_valid}/{copy_update_step_total})"
        )
        print(
            f"copy_update_post_state_exact="
            f"{copy_update_post_state_correct / step_total:.3f} "
            f"({copy_update_post_state_correct}/{copy_update_step_total})"
        )


def evaluate_problem(
    args: argparse.Namespace,
    model,
    train_args: dict,
    device: torch.device,
) -> None:
    expr = extract_math_expression(args.problem)
    max_math_len = train_args.get("max_math_len", 8)
    math_ids = torch.tensor(
        [encode_math_features(args.problem, max_math_len)],
        dtype=torch.long,
        device=device,
    )
    traces = model.solve_variable_reasoning_texts(
        math_ids,
        constrain_to_legal=not args.unconstrained,
        learned_values=args.learned_values,
        raw_learned_values=args.raw_learned_values,
        digit_learned_values=args.digit_learned_values,
        class_learned_values=args.class_learned_values,
        standalone_learned_values=args.standalone_learned_values,
        standalone_digit_values=args.standalone_digit_values,
        standalone_factor_values=args.standalone_factor_values,
        standalone_decomposed_values=args.standalone_decomposed_values,
        standalone_hybrid_values=args.standalone_hybrid_values,
        latent_reasoning_values=args.latent_reasoning_values,
        latent_reasoning_digit_values=args.latent_reasoning_digit_values,
        latent_reasoning_process_values=args.latent_reasoning_process_values,
        latent_reasoning_state_values=args.latent_reasoning_state_values,
        latent_reasoning_slot_values=args.latent_reasoning_slot_values,
        latent_reasoning_slot_transition_values=args.latent_reasoning_slot_transition_values,
        latent_reasoning_slot_class_values=args.latent_reasoning_slot_class_values,
        latent_reasoning_slot_process_values=args.latent_reasoning_slot_process_values,
        latent_reasoning_slot_digit_values=args.latent_reasoning_slot_digit_values,
        latent_reasoning_copy_update_values=args.latent_reasoning_copy_update_values,
        hjepa_values=args.hjepa_values,
    )
    pred_trace = traces[0]
    final = trace_final_value(pred_trace)
    target_trace = make_reasoning_text(expr)
    print(f"answer: {final}")
    print("mode: variable_reasoner")
    print(f"problem: {args.problem}")
    for idx, step in enumerate(trace_step_dicts(pred_trace), start=1):
        print(
            f"step{idx}: lhs={step['lhs']} op={step['op']} "
            f"rhs={step['rhs']} result={step['result']}"
        )
    print(f"final: {final}")
    print(f"trace: {pred_trace}")
    print(f"target_answer: {safe_eval_expression(expr)}")
    print(f"target_trace: {target_trace}")
    print(f"trace_exact: {pred_trace == target_trace}")
    print(f"trace_equiv: {trace_is_equivalent(expr, pred_trace)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/variable_reasoner/best.pt",
    )
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dump-errors", type=int, default=10)
    parser.add_argument("--problem")
    parser.add_argument(
        "--preset",
        choices=EVAL_PRESETS + ("all",),
        default="in_dist",
    )
    parser.add_argument("--learned", action="store_true")
    parser.add_argument("--learned-values", action="store_true")
    parser.add_argument("--raw-learned-values", action="store_true")
    parser.add_argument("--digit-learned-values", action="store_true")
    parser.add_argument("--class-learned-values", action="store_true")
    parser.add_argument("--standalone-learned-values", action="store_true")
    parser.add_argument("--standalone-digit-values", action="store_true")
    parser.add_argument("--standalone-factor-values", action="store_true")
    parser.add_argument("--standalone-decomposed-values", action="store_true")
    parser.add_argument("--standalone-hybrid-values", action="store_true")
    parser.add_argument("--latent-reasoning-values", action="store_true")
    parser.add_argument("--latent-reasoning-digit-values", action="store_true")
    parser.add_argument("--latent-reasoning-process-values", action="store_true")
    parser.add_argument("--latent-reasoning-state-values", action="store_true")
    parser.add_argument("--latent-reasoning-slot-values", action="store_true")
    parser.add_argument("--latent-reasoning-slot-transition-values", action="store_true")
    parser.add_argument("--latent-reasoning-slot-class-values", action="store_true")
    parser.add_argument("--latent-reasoning-slot-process-values", action="store_true")
    parser.add_argument("--latent-reasoning-slot-digit-values", action="store_true")
    parser.add_argument("--latent-reasoning-copy-update-values", action="store_true")
    parser.add_argument(
        "--hjepa-values",
        action="store_true",
        help="Condition the latent reasoning decode on the predicted H-JEPA plan latent.",
    )
    parser.add_argument(
        "--hjepa-plan-order",
        action="store_true",
        help="Report plan-order accuracy (operator sequence + trace length), "
        "ignoring numeric values.",
    )
    parser.add_argument("--unconstrained", action="store_true")
    args = parser.parse_args()

    device = _device(args.device)
    model, tokenizer, train_args = load_math_solver(args.checkpoint, device)
    if args.problem:
        evaluate_problem(args, model, train_args, device)
        return
    presets = EVAL_PRESETS if args.preset == "all" else (args.preset,)
    for idx, preset in enumerate(presets):
        if idx:
            print()
        evaluate_preset(args, model, tokenizer, train_args, device, preset)


if __name__ == "__main__":
    main()
