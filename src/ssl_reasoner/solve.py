from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .solver import _device, load_math_solver, solve_problem_texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/trace_ops_head_mixed/best.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--operation-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("problems", nargs="+")
    args = parser.parse_args()

    device = _device(args.device)
    model, tokenizer, train_args = load_math_solver(args.checkpoint, device)
    results = solve_problem_texts(
        model,
        tokenizer,
        args.problems,
        device,
        train_args["max_problem_len"],
        train_args.get("max_math_len", 8),
        batch_size=args.batch_size,
        operation_confidence_threshold=args.operation_confidence_threshold,
    )

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
        return

    for result in results:
        if args.debug:
            print(
                f"{result.answer}\tmode={result.mode} ops={result.operation_ids} "
                f"op_conf={result.operation_confidences} "
                f"min_op_conf={result.min_operation_confidence:.3f} "
                f"op_answer={result.operation_answer!r} readout={result.readout_answer!r} "
                f"problem={result.problem!r}"
            )
        else:
            print(f"{result.answer}\t{result.problem}")


if __name__ == "__main__":
    main()
