from __future__ import annotations

import argparse

from .data import extract_math_expression, generate_math_examples, make_reasoning_text
from .solver import _device, load_math_solver, solve_problem_texts


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
    shown = 0
    for example, result in zip(examples, results):
        target_trace = make_reasoning_text(extract_math_expression(example.problem))
        answer_ok = result.answer == example.answer
        trace_ok = result.reasoning_trace == target_trace
        answer_correct += int(answer_ok)
        trace_correct += int(trace_ok)
        if (not answer_ok or not trace_ok) and shown < args.dump_errors:
            print(
                f"bad: {example.problem} -> answer={result.answer!r}/{example.answer!r} "
                f"trace={result.reasoning_trace!r}/{target_trace!r}"
            )
            shown += 1

    total = max(len(examples), 1)
    print(f"variable_reasoning_answer_exact={answer_correct / total:.3f} ({answer_correct}/{total})")
    print(f"variable_reasoning_trace_exact={trace_correct / total:.3f} ({trace_correct}/{total})")


if __name__ == "__main__":
    main()
