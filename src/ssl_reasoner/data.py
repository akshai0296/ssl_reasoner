from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


@dataclass(frozen=True)
class MathExample:
    problem: str
    answer: str
    op_label: str
    difficulty: int
    trace: str
    split_label: str = "standard"


MATH_FEATURE_PAD_ID = 0
MATH_FEATURE_NUM_OFFSET = 1
MATH_FEATURE_PLUS_ID = 202
MATH_FEATURE_MINUS_ID = 203
MATH_FEATURE_TIMES_ID = 204
MATH_FEATURE_VOCAB_SIZE = 205
TRACE_OP_TO_ID = {"none": 0, "+": 1, "-": 2, "*": 3}
TRACE_VALUE_MIN = -200
TRACE_VALUE_MAX = 1200
TRACE_VALUE_CLASSES = TRACE_VALUE_MAX - TRACE_VALUE_MIN + 1
EXPR_RE = re.compile(r"\d+(?:[+\-*]\d+)+")


def encode_math_features(problem: str, max_len: int = 8) -> list[int]:
    """Extract expression-level operand/operator tokens without exposing the answer."""
    ids = []
    for token in re.findall(r"\d+|[+\-*]", problem):
        if token.isdigit():
            value = min(int(token), MATH_FEATURE_PLUS_ID - MATH_FEATURE_NUM_OFFSET - 1)
            ids.append(MATH_FEATURE_NUM_OFFSET + value)
        elif token == "+":
            ids.append(MATH_FEATURE_PLUS_ID)
        elif token == "-":
            ids.append(MATH_FEATURE_MINUS_ID)
        elif token == "*":
            ids.append(MATH_FEATURE_TIMES_ID)
    ids = ids[:max_len]
    ids.extend([MATH_FEATURE_PAD_ID] * (max_len - len(ids)))
    return ids


def extract_math_expression(problem: str) -> str:
    match = EXPR_RE.search(problem)
    if match is None:
        raise ValueError(f"No arithmetic expression found in: {problem!r}")
    return match.group(0)


def make_variable_trace_steps(expr: str) -> list[tuple[int, str, int, int]]:
    parts = re.split(r"([+\-*])", expr)
    values = [int(parts[idx]) for idx in range(0, len(parts), 2)]
    ops = [parts[idx] for idx in range(1, len(parts), 2)]
    steps: list[tuple[int, str, int, int]] = []

    while "*" in ops:
        idx = ops.index("*")
        lhs = values[idx]
        rhs = values[idx + 1]
        result = lhs * rhs
        steps.append((lhs, "*", rhs, result))
        values[idx : idx + 2] = [result]
        del ops[idx]

    while ops:
        lhs = values[0]
        rhs = values[1]
        op = ops[0]
        if op == "+":
            result = lhs + rhs
        elif op == "-":
            result = lhs - rhs
        else:
            raise ValueError(f"Unsupported operator: {op}")
        steps.append((lhs, op, rhs, result))
        values[:2] = [result]
        del ops[0]

    return steps


def make_trace(expr: str) -> str:
    return " ".join(
        f"{lhs}{op}{rhs}={result}"
        for lhs, op, rhs, result in make_variable_trace_steps(expr)
    )


def make_reasoning_text(expr: str) -> str:
    trace = make_trace(expr)
    return f"{trace.replace(' ', ',')},{eval(expr)}"


def make_reasoning_step_texts(expr: str) -> tuple[list[str], list[float]]:
    trace_parts = make_trace(expr).split()
    answer = str(eval(expr))
    if len(trace_parts) == 1:
        return [trace_parts[0], answer, ""], [1.0, 1.0, 0.0]
    return [trace_parts[0], trace_parts[1], answer], [1.0, 1.0, 1.0]


def make_variable_trace_fields(
    expr: str,
    max_steps: int = 4,
) -> tuple[list[int], list[list[float]], list[float]]:
    steps = make_variable_trace_steps(expr)[:max_steps]
    op_ids = [TRACE_OP_TO_ID[op] for _, op, _, _ in steps]
    values = [[float(lhs), float(rhs), float(result)] for lhs, _, rhs, result in steps]
    mask = [1.0] * len(steps)
    while len(op_ids) < max_steps:
        op_ids.append(TRACE_OP_TO_ID["none"])
        values.append([0.0, 0.0, 0.0])
        mask.append(0.0)
    return op_ids, values, mask


def _value_to_class(value: float) -> int:
    return int(max(TRACE_VALUE_MIN, min(TRACE_VALUE_MAX, int(value)))) - TRACE_VALUE_MIN


def value_to_class(value: int | str) -> int:
    return _value_to_class(float(value))


def class_to_value(class_id: int) -> int:
    return int(class_id) + TRACE_VALUE_MIN


def make_trace_fields(expr: str) -> tuple[list[int], list[int], list[float]]:
    parts = re.split(r"([+\-*])", expr)
    if len(parts) == 3:
        a, op, b = parts
        value = eval(expr)
        return (
            [TRACE_OP_TO_ID[op], TRACE_OP_TO_ID["none"]],
            [_value_to_class(v) for v in [float(a), float(b), float(value), 0.0, 0.0, 0.0]],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        )

    if len(parts) != 5:
        value = eval(expr)
        return (
            [TRACE_OP_TO_ID["none"], TRACE_OP_TO_ID["none"]],
            [_value_to_class(v) for v in [float(value), 0.0, float(value), 0.0, 0.0, 0.0]],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        )

    a, op1, b, op2, c = parts
    if op2 == "*":
        first_lhs = float(b)
        first_rhs = float(c)
        first_op = op2
        first_value = eval(f"{b}{op2}{c}")
        second_lhs = float(a)
        second_rhs = float(first_value)
        second_op = op1
    else:
        first_lhs = float(a)
        first_rhs = float(b)
        first_op = op1
        first_value = eval(f"{a}{op1}{b}")
        second_lhs = float(first_value)
        second_rhs = float(c)
        second_op = op2
    second_value = eval(f"{int(second_lhs)}{second_op}{int(second_rhs)}")
    return (
        [TRACE_OP_TO_ID[first_op], TRACE_OP_TO_ID[second_op]],
        [_value_to_class(v) for v in [
            first_lhs,
            first_rhs,
            float(first_value),
            second_lhs,
            second_rhs,
            float(second_value),
        ]],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )


def make_trace_state_targets(expr: str) -> tuple[list[float], list[float]]:
    parts = re.split(r"([+\-*])", expr)
    if len(parts) == 3:
        value = float(eval(expr))
        return [value, 0.0], [1.0, 0.0]
    if len(parts) != 5:
        value = float(eval(expr))
        return [value, 0.0], [1.0, 0.0]

    a, op1, b, op2, c = parts
    if op2 == "*":
        first_value = float(eval(f"{b}{op2}{c}"))
        second_value = float(eval(f"{a}{op1}{int(first_value)}"))
    else:
        first_value = float(eval(f"{a}{op1}{b}"))
        second_value = float(eval(f"{int(first_value)}{op2}{c}"))
    return [first_value, second_value], [1.0, 1.0]


def _rand_operand(rng: random.Random, bounds: tuple[int, int]) -> int:
    return rng.randint(bounds[0], bounds[1])


def _make_expression(
    rng: random.Random,
    difficulty: int,
    op: str | None = None,
    *,
    single_add_bounds: tuple[int, int] = (0, 100),
    single_mul_bounds: tuple[int, int] = (0, 20),
    mixed_ab_bounds: tuple[int, int] = (0, 50),
    mixed_c_bounds: tuple[int, int] = (0, 20),
) -> tuple[str, int, str]:
    if difficulty == 0:
        op = op or rng.choice(["+", "-", "*"])
        bounds = single_mul_bounds if op == "*" else single_add_bounds
        a = _rand_operand(rng, bounds)
        b = _rand_operand(rng, bounds)
        if op == "-":
            a, b = max(a, b), min(a, b)
        expr = f"{a}{op}{b}"
        return expr, eval(expr), op

    if difficulty == 2:
        operands = [_rand_operand(rng, (0, 50))]
        operands.extend(_rand_operand(rng, (0, 20)) for _ in range(3))
        expr_ops = [rng.choice(["+", "-", "*"]) for _ in range(3)]
        if "*" not in expr_ops:
            expr_ops[rng.randrange(len(expr_ops))] = "*"
        expr = "".join(
            f"{value}{op}" for value, op in zip(operands, expr_ops)
        ) + str(operands[-1])
        return expr, eval(expr), "multi_step"

    a = _rand_operand(rng, mixed_ab_bounds)
    b = _rand_operand(rng, mixed_ab_bounds)
    c = _rand_operand(rng, mixed_c_bounds)
    op1 = rng.choice(["+", "-"])
    op2 = rng.choice(["+", "-", "*"])
    expr = f"{a}{op1}{b}{op2}{c}"
    return expr, eval(expr), "mixed"


COMPOSITIONAL_CURRICULA = {
    "seen_single",
    "unseen_single",
    "seen_mixed",
    "unseen_mixed",
    "compositional_train",
}
CURRICULA = (
    "mixed",
    "single_op_balanced",
    "mixed_only",
    "multi_step",
    *sorted(COMPOSITIONAL_CURRICULA),
)


def _compositional_spec(curriculum: str, idx: int) -> tuple[int, str | None, str, dict]:
    if curriculum == "compositional_train":
        curriculum = "seen_single" if idx % 2 == 0 else "seen_mixed"

    if curriculum == "seen_single":
        return (
            0,
            ["+", "-", "*"][idx % 3],
            "seen_single",
            {"single_add_bounds": (0, 50), "single_mul_bounds": (0, 10)},
        )
    if curriculum == "unseen_single":
        return (
            0,
            ["+", "-", "*"][idx % 3],
            "unseen_single",
            {"single_add_bounds": (51, 100), "single_mul_bounds": (11, 20)},
        )
    if curriculum == "seen_mixed":
        return (
            1,
            None,
            "seen_mixed",
            {"mixed_ab_bounds": (0, 25), "mixed_c_bounds": (0, 10)},
        )
    if curriculum == "unseen_mixed":
        return (
            1,
            None,
            "unseen_mixed",
            {"mixed_ab_bounds": (26, 50), "mixed_c_bounds": (11, 20)},
        )
    raise ValueError(f"Unknown compositional curriculum: {curriculum}")


def generate_math_examples(
    n: int,
    seed: int = 0,
    difficulty_mix: tuple[float, float] = (0.75, 0.25),
    curriculum: str = "mixed",
) -> list[MathExample]:
    rng = random.Random(seed)
    examples = []
    ops = ["+", "-", "*"]
    for idx in range(n):
        split_label = "standard"
        expression_kwargs = {}
        if curriculum == "mixed":
            difficulty = 0 if rng.random() < difficulty_mix[0] else 1
            op = None
        elif curriculum == "single_op_balanced":
            difficulty = 0
            op = ops[idx % len(ops)]
        elif curriculum == "mixed_only":
            difficulty = 1
            op = None
        elif curriculum == "multi_step":
            difficulty = 2
            op = None
        elif curriculum in COMPOSITIONAL_CURRICULA:
            difficulty, op, split_label, expression_kwargs = _compositional_spec(
                curriculum, idx
            )
        else:
            raise ValueError(f"Unknown curriculum: {curriculum}")
        expr, value, op_label = _make_expression(
            rng, difficulty, op=op, **expression_kwargs
        )
        template = rng.choice(
            [
                "What is {expr}?",
                "Calculate {expr}.",
                "Find the value of {expr}.",
            ]
        )
        examples.append(
            MathExample(
                problem=template.format(expr=expr),
                answer=str(value),
                op_label=op_label,
                difficulty=difficulty,
                trace=make_trace(expr),
                split_label=split_label,
            )
        )
    return examples


class MathDataset(Dataset):
    def __init__(
        self,
        examples: list[MathExample],
        tokenizer: CharTokenizer,
        max_problem_len: int = 64,
        max_answer_len: int = 16,
        max_math_len: int = 8,
        max_trace_len: int = 32,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_problem_len = max_problem_len
        self.max_answer_len = max_answer_len
        self.max_math_len = max_math_len
        self.max_trace_len = max_trace_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        ex = self.examples[idx]
        problem_ids = self.tokenizer.encode(ex.problem, self.max_problem_len)
        answer_ids = self.tokenizer.encode(ex.answer, self.max_answer_len)
        math_ids = encode_math_features(ex.problem, self.max_math_len)
        trace_ids = self.tokenizer.encode(ex.trace, self.max_trace_len)
        expr = extract_math_expression(ex.problem)
        trace_op_ids, trace_values, trace_value_mask = make_trace_fields(expr)
        trace_state_values, trace_state_mask = make_trace_state_targets(expr)
        reasoning_text = make_reasoning_text(expr)
        reasoning_step_texts, reasoning_step_mask = make_reasoning_step_texts(expr)
        variable_trace_op_ids, variable_trace_values, variable_trace_mask = (
            make_variable_trace_fields(expr)
        )
        reasoning_step_ids = [
            self.tokenizer.encode(text, self.max_answer_len)
            for text in reasoning_step_texts
        ]
        reasoning_ids = self.tokenizer.encode(reasoning_text, self.max_trace_len)
        answer_len = min(len(ex.answer) + 2, self.max_answer_len)
        trace_len = min(len(ex.trace) + 2, self.max_trace_len)
        reasoning_len = min(len(reasoning_text) + 2, self.max_trace_len)
        return {
            "problem_ids": torch.tensor(problem_ids, dtype=torch.long),
            "math_ids": torch.tensor(math_ids, dtype=torch.long),
            "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
            "answer_len": torch.tensor(answer_len, dtype=torch.long),
            "trace_ids": torch.tensor(trace_ids, dtype=torch.long),
            "trace_len": torch.tensor(trace_len, dtype=torch.long),
            "trace_op_ids": torch.tensor(trace_op_ids, dtype=torch.long),
            "trace_value_ids": torch.tensor(trace_values, dtype=torch.long),
            "trace_value_mask": torch.tensor(trace_value_mask, dtype=torch.float),
            "trace_state_values": torch.tensor(trace_state_values, dtype=torch.float),
            "trace_state_mask": torch.tensor(trace_state_mask, dtype=torch.float),
            "variable_trace_op_ids": torch.tensor(variable_trace_op_ids, dtype=torch.long),
            "variable_trace_values": torch.tensor(variable_trace_values, dtype=torch.float),
            "variable_trace_mask": torch.tensor(variable_trace_mask, dtype=torch.float),
            "reasoning_step_ids": torch.tensor(reasoning_step_ids, dtype=torch.long),
            "reasoning_step_mask": torch.tensor(reasoning_step_mask, dtype=torch.float),
            "reasoning_ids": torch.tensor(reasoning_ids, dtype=torch.long),
            "reasoning_len": torch.tensor(reasoning_len, dtype=torch.long),
            "answer_value_id": torch.tensor(value_to_class(ex.answer), dtype=torch.long),
            "answer": ex.answer,
            "trace": ex.trace,
            "reasoning": reasoning_text,
            "problem": ex.problem,
            "op_label": ex.op_label,
            "difficulty": ex.difficulty,
            "split_label": ex.split_label,
        }
