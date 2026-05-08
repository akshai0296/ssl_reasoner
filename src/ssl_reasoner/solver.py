from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from .data import (
    MATH_FEATURE_VOCAB_SIZE,
    TRACE_VALUE_CLASSES,
    class_to_value,
    encode_math_features,
    make_variable_trace_steps,
)
from .model import MathJEPAReadout
from .tokenizer import CharTokenizer, build_math_tokenizer
from .verifier import extract_expression, operation_candidate_text


@dataclass(frozen=True)
class MathSolveResult:
    problem: str
    answer: str
    mode: str
    operation_answer: str | None
    trace_state_answer: str | None
    parsed_expression_answer: str | None
    readout_answer: str
    operation_ids: list[int]
    operation_confidences: list[float]
    trace_state_confidence: float
    min_operation_confidence: float
    reasoning_trace: str | None
    reasoning_steps: list[dict[str, int | str]]
    reasoning_final: str | None
    reasoning_order: str | None


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


def trace_final_value_index(problem: str) -> int | None:
    expr = extract_expression(problem)
    if expr is None:
        return None
    parts = re.split(r"([+\-*])", expr)
    if len(parts) == 3:
        return 2
    if len(parts) == 5:
        return 5
    return None


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


def _op_text(op_id: int) -> str:
    if op_id == 1:
        return "+"
    if op_id == 2:
        return "-"
    if op_id == 3:
        return "*"
    return ""


def _reasoning_step(
    lhs: int,
    op_id: int,
    rhs: int,
    result: int,
) -> dict[str, int | str]:
    return {
        "lhs": lhs,
        "op": _op_text(op_id),
        "rhs": rhs,
        "result": result,
    }


@torch.no_grad()
def predict_structured_reasoning_details(
    model: MathJEPAReadout,
    math_ids: torch.Tensor,
) -> list[dict[str, object]]:
    empty = {
        "trace": None,
        "steps": [],
        "final": None,
        "order": None,
    }
    if not hasattr(model, "step_state_head"):
        return [empty.copy() for _ in range(math_ids.size(0))]

    state_out = model.step_state_head(math_ids)
    values = state_out["values"].detach().cpu().tolist()
    order_ids = state_out["order_logits"].argmax(dim=-1).detach().cpu().tolist()
    math_rows = math_ids.detach().cpu().tolist()

    rows: list[dict[str, object]] = []
    for math_row, state_row, order_id in zip(math_rows, values, order_ids):
        op_count = sum(1 for idx in range(1, len(math_row), 2) if math_row[idx] != 0)
        if op_count > 2:
            expr_parts: list[str] = []
            for idx, token_id in enumerate(math_row):
                if token_id == 0:
                    break
                if idx % 2 == 0:
                    expr_parts.append(str(MathJEPAReadout._math_value(token_id)))
                else:
                    expr_parts.append(_op_text(MathJEPAReadout._math_op_id(token_id)))
            steps = [
                _reasoning_step(lhs, {"+": 1, "-": 2, "*": 3}[op], rhs, result)
                for lhs, op, rhs, result in make_variable_trace_steps("".join(expr_parts))
            ]
            final = steps[-1]["result"] if steps else None
            trace = ",".join(
                f"{step['lhs']}{step['op']}{step['rhs']}={step['result']}"
                for step in steps
            )
            if final is not None:
                trace = f"{trace},{final}"
            rows.append(
                {
                    "trace": trace,
                    "steps": steps,
                    "final": str(final) if final is not None else None,
                    "order": "variable_precedence",
                }
            )
            continue

        a = MathJEPAReadout._math_value(math_row[0])
        op1_id = MathJEPAReadout._math_op_id(math_row[1])
        b = MathJEPAReadout._math_value(math_row[2])
        op2_id = MathJEPAReadout._math_op_id(math_row[3])
        c = MathJEPAReadout._math_value(math_row[4])
        if not op1_id:
            rows.append(empty.copy())
            continue
        first = int(round(float(state_row[0])))
        final = int(round(float(state_row[1]))) if op2_id else first

        if not op2_id:
            steps = [_reasoning_step(a, op1_id, b, first)]
            order = "single"
        elif int(order_id) == 1:
            steps = [
                _reasoning_step(b, op2_id, c, first),
                _reasoning_step(a, op1_id, first, final),
            ]
            order = "right_first"
        else:
            steps = [
                _reasoning_step(a, op1_id, b, first),
                _reasoning_step(first, op2_id, c, final),
            ]
            order = "left_first"

        rows.append(
            {
                "trace": MathJEPAReadout._render_step_state_row(
                    math_row,
                    state_row,
                    int(order_id),
                ),
                "steps": steps,
                "final": str(final),
                "order": order,
            }
        )
    return rows


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
    args["_checkpoint_has_step_state_head"] = any(
        key.startswith("step_state_head.") for key in ckpt["model"]
    )
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
        max_variable_steps=args.get("max_variable_steps", 4),
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
def predict_trace_state_answers(
    model: MathJEPAReadout,
    problem_ids: torch.Tensor,
    math_ids: torch.Tensor,
    problems: list[str],
) -> tuple[list[str | None], list[float]]:
    if not model.use_reasoning_trace:
        return [None for _ in problems], [0.0 for _ in problems]
    slots = model.predict_trace_slots(problem_ids, math_ids)
    pred = model.trace_struct_head(slots.mean(dim=1))
    value_probs = pred[:, 8:].view(-1, 6, TRACE_VALUE_CLASSES).softmax(dim=-1)
    value_confidences, value_ids = value_probs.max(dim=-1)
    answers: list[str | None] = []
    confidences: list[float] = []
    for problem, row, conf_row in zip(
        problems,
        value_ids.detach().cpu().tolist(),
        value_confidences.detach().cpu().tolist(),
    ):
        final_idx = trace_final_value_index(problem)
        if final_idx is None:
            answers.append(None)
            confidences.append(0.0)
            continue
        answers.append(str(class_to_value(int(row[final_idx]))))
        confidences.append(float(conf_row[final_idx]))
    return answers, confidences


@torch.no_grad()
def predict_trace_state_regression_answers(
    model: MathJEPAReadout,
    problem_ids: torch.Tensor,
    math_ids: torch.Tensor,
    problems: list[str],
) -> tuple[list[str | None], list[float]]:
    if not model.use_reasoning_trace:
        return [None for _ in problems], [0.0 for _ in problems]
    pred_values = model.predict_trace_state_values(problem_ids, math_ids)
    answers: list[str | None] = []
    confidences: list[float] = []
    for problem, values in zip(problems, pred_values.detach().cpu()):
        final_idx = trace_final_value_index(problem)
        if final_idx is None:
            answers.append(None)
            confidences.append(0.0)
            continue
        state_idx = 1 if final_idx == 5 else 0
        value = float(values[state_idx].item())
        answers.append(str(int(round(value))))
        confidences.append(1.0)
    return answers, confidences


@torch.no_grad()
def predict_step_state_answers(
    model: MathJEPAReadout,
    math_ids: torch.Tensor,
    problems: list[str],
) -> tuple[list[str | None], list[float]]:
    values = model.predict_step_state_values(math_ids)
    answers: list[str | None] = []
    confidences: list[float] = []
    for problem, row in zip(problems, values.detach().cpu()):
        final_idx = trace_final_value_index(problem)
        if final_idx is None:
            answers.append(None)
            confidences.append(0.0)
            continue
        state_idx = 1 if final_idx == 5 else 0
        answers.append(str(int(round(float(row[state_idx].item())))))
        confidences.append(1.0)
    return answers, confidences


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
        state_answers, state_confidences = predict_trace_state_answers(
            model,
            problem_ids,
            math_ids,
            batch_problems,
        )
        reasoning_details = predict_structured_reasoning_details(model, math_ids)

        for (
            problem,
            op_row,
            confidence_row,
            state_answer,
            state_confidence,
            readout_answer,
            reasoning_detail,
        ) in zip(
            batch_problems,
            op_rows,
            confidence_rows,
            state_answers,
            state_confidences,
            readout_answers,
            reasoning_details,
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
                    trace_state_answer=state_answer,
                    parsed_expression_answer=parsed_answer,
                    readout_answer=readout_answer,
                    operation_ids=[int(idx) for idx in op_row],
                    operation_confidences=[float(conf) for conf in confidence_row],
                    trace_state_confidence=float(state_confidence),
                    min_operation_confidence=float(min_confidence),
                    reasoning_trace=(
                        str(reasoning_detail["trace"])
                        if reasoning_detail["trace"] is not None
                        else None
                    ),
                    reasoning_steps=list(reasoning_detail["steps"]),
                    reasoning_final=(
                        str(reasoning_detail["final"])
                        if reasoning_detail["final"] is not None
                        else None
                    ),
                    reasoning_order=(
                        str(reasoning_detail["order"])
                        if reasoning_detail["order"] is not None
                        else None
                    ),
                )
            )
    return results
