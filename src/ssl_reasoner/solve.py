from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict

import torch

from .data import encode_math_features, extract_math_expression, make_reasoning_text
from .solver import _device, load_math_solver, solve_problem_texts


TRACE_STEP_RE = re.compile(r"^(-?\d+)([+\-*])(-?\d+)=(-?\d+)$")


def trace_final_value(trace: str) -> int | None:
    parts = [part for part in trace.split(",") if part]
    if not parts:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


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


@torch.no_grad()
def solve_variable_debug(
    model,
    problems: list[str],
    train_args: dict,
    device: torch.device,
    constrain_to_legal: bool = True,
    value_mode: str = "candidate",
) -> list[dict]:
    max_math_len = train_args.get("max_math_len", 8)
    math_ids = torch.tensor(
        [encode_math_features(problem, max_math_len) for problem in problems],
        dtype=torch.long,
        device=device,
    )
    traces = model.solve_variable_reasoning_texts(
        math_ids,
        constrain_to_legal=constrain_to_legal,
        learned_values=value_mode == "candidate",
        raw_learned_values=value_mode == "raw",
        digit_learned_values=value_mode == "digit",
        class_learned_values=value_mode == "class",
        standalone_learned_values=value_mode == "standalone",
        standalone_digit_values=value_mode == "standalone_digit",
        standalone_factor_values=value_mode == "standalone_factor",
        standalone_decomposed_values=value_mode == "standalone_decomposed",
        standalone_hybrid_values=value_mode == "standalone_hybrid",
        latent_reasoning_values=value_mode == "latent_reasoning",
    )
    rows = []
    for problem, trace in zip(problems, traces):
        try:
            expr = extract_math_expression(problem)
        except ValueError:
            expr = ""
        is_parenthesized = "(" in expr or ")" in expr
        if is_parenthesized:
            trace = make_reasoning_text(expr)
        final = trace_final_value(trace)
        rows.append(
            {
                "answer": None if final is None else str(final),
                "mode": "parsed_expression" if is_parenthesized else "variable_reasoner",
                "problem": problem,
                "reasoning_steps": trace_step_dicts(trace),
                "reasoning_final": final,
                "reasoning_trace": trace,
            }
        )
    return rows


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
    parser.add_argument("--debug-reasoning", action="store_true")
    parser.add_argument("--unconstrained", action="store_true")
    parser.add_argument(
        "--value-mode",
        choices=[
            "candidate",
            "raw",
            "digit",
            "class",
            "standalone",
            "standalone_digit",
            "standalone_factor",
            "standalone_decomposed",
            "standalone_hybrid",
            "latent_reasoning",
        ],
        default="candidate",
    )
    parser.add_argument("problems", nargs="+")
    args = parser.parse_args()

    device = _device(args.device)
    model, tokenizer, train_args = load_math_solver(args.checkpoint, device)
    use_variable_debug = args.debug_reasoning and "variable_reasoner" in args.checkpoint
    if use_variable_debug or (
        args.debug_reasoning and not train_args.get("_checkpoint_has_step_state_head", False)
    ):
        rows = solve_variable_debug(
            model,
            args.problems,
            train_args,
            device,
            constrain_to_legal=not args.unconstrained,
            value_mode=args.value_mode,
        )
        if args.json:
            print(json.dumps(rows, indent=2))
            return
        for row in rows:
            print(f"answer: {row['answer']}")
            print(f"mode: {row['mode']}")
            print(f"problem: {row['problem']}")
            for idx, step in enumerate(row["reasoning_steps"], start=1):
                print(
                    f"step{idx}: lhs={step['lhs']} op={step['op']} "
                    f"rhs={step['rhs']} result={step['result']}"
                )
            print(f"final: {row['reasoning_final']}")
            print(f"trace: {row['reasoning_trace']}")
            print()
        return
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
        if args.debug_reasoning:
            print(f"answer: {result.answer}")
            print(f"mode: {result.mode}")
            print(f"problem: {result.problem}")
            print(f"reasoning_order: {result.reasoning_order}")
            for idx, step in enumerate(result.reasoning_steps, start=1):
                print(
                    f"step{idx}: lhs={step['lhs']} op={step['op']} "
                    f"rhs={step['rhs']} result={step['result']}"
                )
            print(f"final: {result.reasoning_final}")
            print(f"trace: {result.reasoning_trace}")
            print()
            continue
        if args.debug:
            print(
                f"{result.answer}\tmode={result.mode} ops={result.operation_ids} "
                f"op_conf={result.operation_confidences} "
                f"min_op_conf={result.min_operation_confidence:.3f} "
                f"op_answer={result.operation_answer!r} "
                f"trace_state_answer={result.trace_state_answer!r} "
                f"trace_state_conf={result.trace_state_confidence:.3f} "
                f"parsed_answer={result.parsed_expression_answer!r} "
                f"readout={result.readout_answer!r} "
                f"problem={result.problem!r}"
            )
        else:
            print(f"{result.answer}\t{result.problem}")


if __name__ == "__main__":
    main()
