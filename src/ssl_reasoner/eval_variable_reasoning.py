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
)
from .solver import _device, load_math_solver


TRACE_STEP_RE = re.compile(r"^(-?\d+)([+\-*])(-?\d+)=(-?\d+)$")
EVAL_PRESETS = (
    "in_dist",
    "larger_numbers",
    "longer_expr",
    "no_multiply",
    "many_multiply",
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
    return final == values[0] == int(eval(expr))


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
        answer=str(eval(expr)),
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
        expr = "".join(f"{value}{op}" for value, op in zip(operands, ops)) + str(
            operands[-1]
        )
        examples.append(make_eval_example(expr, rng, preset))
    return examples


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
    shown = 0
    learned_traces: list[str] | None = None
    if args.learned:
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
            _, _, target_values, _, target_mask = make_variable_trace_fields(expr)
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
    print(f"target_answer: {eval(expr)}")
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
