from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from .data import MATH_FEATURE_VOCAB_SIZE, encode_math_features
from .model import MathJEPAReadout
from .tokenizer import CharTokenizer, build_math_tokenizer
from .verifier import extract_expression, operation_candidate_text


@dataclass(frozen=True)
class MathSolveResult:
    problem: str
    answer: str
    mode: str
    operation_answer: str | None
    parsed_expression_answer: str | None
    readout_answer: str
    operation_ids: list[int]
    operation_confidences: list[float]
    min_operation_confidence: float


def operation_confidence_count(problem: str) -> int:
    expr = extract_expression(problem)
    if expr is None:
        return 0
    parts = re.split(r"([+\-*])", expr)
    if len(parts) == 3:
        return 1
    if len(parts) == 5:
        return 2
    return 0


def parsed_expression_answer(problem: str) -> str | None:
    expr = extract_expression(problem)
    if expr is None:
        return None
    parts = re.split(r"([+\-*])", expr)
    if len(parts) < 3 or len(parts) % 2 == 0:
        return None

    values = [int(parts[idx]) for idx in range(0, len(parts), 2)]
    ops = [parts[idx] for idx in range(1, len(parts), 2)]

    collapsed_values = [values[0]]
    collapsed_ops: list[str] = []
    for op, value in zip(ops, values[1:]):
        if op == "*":
            collapsed_values[-1] *= value
        else:
            collapsed_ops.append(op)
            collapsed_values.append(value)

    total = collapsed_values[0]
    for op, value in zip(collapsed_ops, collapsed_values[1:]):
        if op == "+":
            total += value
        elif op == "-":
            total -= value
        else:
            return None
    return str(total)


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_math_solver(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[MathJEPAReadout, CharTokenizer, dict]:
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
    return model, tokenizer, args


@torch.no_grad()
def predict_trace_operation_ids(
    model: MathJEPAReadout,
    problem_ids: torch.Tensor,
    math_ids: torch.Tensor,
) -> tuple[list[list[int]], list[list[float]]]:
    if not model.use_reasoning_trace:
        empty = [[] for _ in range(problem_ids.size(0))]
        return empty, empty
    slots = model.predict_trace_slots(problem_ids, math_ids)
    pred = model.trace_struct_head(slots.mean(dim=1))
    op_probs = pred[:, :8].view(-1, 2, 4).softmax(dim=-1)
    confidences, op_ids = op_probs.max(dim=-1)
    return op_ids.detach().cpu().tolist(), confidences.detach().cpu().tolist()


@torch.no_grad()
def solve_problem_texts(
    model: MathJEPAReadout,
    tokenizer: CharTokenizer,
    problems: list[str],
    device: torch.device,
    max_problem_len: int,
    max_math_len: int,
    batch_size: int = 64,
    operation_confidence_threshold: float = 0.0,
) -> list[MathSolveResult]:
    results: list[MathSolveResult] = []
    for start in range(0, len(problems), batch_size):
        batch_problems = problems[start : start + batch_size]
        problem_ids = torch.tensor(
            [tokenizer.encode(problem, max_problem_len) for problem in batch_problems],
            dtype=torch.long,
            device=device,
        )
        math_ids = torch.tensor(
            [encode_math_features(problem, max_math_len) for problem in batch_problems],
            dtype=torch.long,
            device=device,
        )
        decoded = model.solve_ids(problem_ids, pad_id=tokenizer.pad_id, math_ids=math_ids)
        readout_answers = [tokenizer.decode(ids).strip() for ids in decoded]

        op_rows, confidence_rows = predict_trace_operation_ids(model, problem_ids, math_ids)

        for problem, op_row, confidence_row, readout_answer in zip(
            batch_problems, op_rows, confidence_rows, readout_answers
        ):
            operation_answer = operation_candidate_text(problem, op_row)
            parsed_answer = (
                parsed_expression_answer(problem)
                if operation_answer is None
                else None
            )
            confidence_count = operation_confidence_count(problem)
            used_confidences = confidence_row[:confidence_count]
            min_confidence = min(used_confidences, default=0.0)
            if (
                operation_answer is None
                or min_confidence < operation_confidence_threshold
            ):
                if parsed_answer is not None:
                    answer = parsed_answer
                    mode = "parsed_expression"
                else:
                    answer = readout_answer
                    mode = "readout"
            else:
                answer = operation_answer
                mode = "operation"
            results.append(
                MathSolveResult(
                    problem=problem,
                    answer=answer,
                    mode=mode,
                    operation_answer=operation_answer,
                    parsed_expression_answer=parsed_answer,
                    readout_answer=readout_answer,
                    operation_ids=[int(idx) for idx in op_row],
                    operation_confidences=[float(conf) for conf in confidence_row],
                    min_operation_confidence=float(min_confidence),
                )
            )
    return results
