from __future__ import annotations

import argparse
import re

import torch

from .data import (
    encode_math_features,
    extract_math_expression,
    generate_math_examples,
    make_reasoning_text,
)
from .solver import _device, load_math_solver, solve_problem_texts


TRACE_STEP_RE = re.compile(r"^(-?\d+)([+\-*])(-?\d+)=(-?\d+)$")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/step_state_solver_mixed_only/best.pt",
    )
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dump-errors", type=int, default=10)
    parser.add_argument("--learned", action="store_true")
    parser.add_argument("--unconstrained", action="store_true")
    args = parser.parse_args()

    device = _device(args.device)
    model, tokenizer, train_args = load_math_solver(args.checkpoint, device)
    examples = generate_math_examples(
        args.samples,
        seed=args.seed,
        curriculum="multi_step",
    )
    results = solve_problem_texts(
        model,
        tokenizer,
        [example.problem for example in examples],
        device,
        train_args["max_problem_len"],
        train_args.get("max_math_len", 8),
        batch_size=args.batch_size,
    )

    answer_correct = 0
    trace_correct = 0
    trace_equiv_correct = 0
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
            learned_traces.extend(
                model.solve_variable_reasoning_texts(
                    all_math_ids[start : start + args.batch_size],
                    constrain_to_legal=not args.unconstrained,
                )
            )

    for idx, (example, result) in enumerate(zip(examples, results)):
        expr = extract_math_expression(example.problem)
        target_trace = make_reasoning_text(expr)
        answer_ok = result.answer == example.answer
        pred_trace = learned_traces[idx] if learned_traces is not None else result.reasoning_trace
        trace_ok = pred_trace == target_trace
        trace_equiv_ok = trace_is_equivalent(expr, pred_trace)
        answer_correct += int(answer_ok)
        trace_correct += int(trace_ok)
        trace_equiv_correct += int(trace_equiv_ok)
        if (not answer_ok or not trace_equiv_ok) and shown < args.dump_errors:
            print(
                f"bad: {example.problem} -> answer={result.answer!r}/{example.answer!r} "
                f"trace={pred_trace!r}/{target_trace!r}"
            )
            shown += 1

    total = max(len(examples), 1)
    print(f"variable_reasoning_answer_exact={answer_correct / total:.3f} ({answer_correct}/{total})")
    print(f"variable_reasoning_trace_exact={trace_correct / total:.3f} ({trace_correct}/{total})")
    print(
        f"variable_reasoning_trace_equiv_exact="
        f"{trace_equiv_correct / total:.3f} ({trace_equiv_correct}/{total})"
    )


if __name__ == "__main__":
    main()
