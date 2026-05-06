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


def make_trace(expr: str) -> str:
    parts = re.split(r"([+\-*])", expr)
    if len(parts) != 5:
        return f"{expr}={eval(expr)}"

    a, op1, b, op2, c = parts
    if op2 == "*":
        first_expr = f"{b}{op2}{c}"
        first_value = eval(first_expr)
        second_expr = f"{a}{op1}{first_value}"
        second_value = eval(second_expr)
    else:
        first_expr = f"{a}{op1}{b}"
        first_value = eval(first_expr)
        second_expr = f"{first_value}{op2}{c}"
        second_value = eval(second_expr)
    return f"{first_expr}={first_value} {second_expr}={second_value}"


def _value_to_class(value: float) -> int:
    return int(max(TRACE_VALUE_MIN, min(TRACE_VALUE_MAX, int(value)))) - TRACE_VALUE_MIN


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


def _make_expression(
    rng: random.Random, difficulty: int, op: str | None = None
) -> tuple[str, int, str]:
    if difficulty == 0:
        op = op or rng.choice(["+", "-", "*"])
        limit = 20 if op == "*" else 100
        a = rng.randint(0, limit)
        b = rng.randint(0, limit)
        if op == "-":
            a, b = max(a, b), min(a, b)
        expr = f"{a}{op}{b}"
        return expr, eval(expr), op

    a = rng.randint(0, 50)
    b = rng.randint(0, 50)
    c = rng.randint(0, 20)
    op1 = rng.choice(["+", "-"])
    op2 = rng.choice(["+", "-", "*"])
    expr = f"{a}{op1}{b}{op2}{c}"
    return expr, eval(expr), "mixed"


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
        if curriculum == "mixed":
            difficulty = 0 if rng.random() < difficulty_mix[0] else 1
            op = None
        elif curriculum == "single_op_balanced":
            difficulty = 0
            op = ops[idx % len(ops)]
        elif curriculum == "mixed_only":
            difficulty = 1
            op = None
        else:
            raise ValueError(f"Unknown curriculum: {curriculum}")
        expr, value, op_label = _make_expression(rng, difficulty, op=op)
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
        trace_op_ids, trace_values, trace_value_mask = make_trace_fields(
            re.search(r"\d+[+\-*]\d+(?:[+\-*]\d+)?", ex.problem).group(0)
        )
        answer_len = min(len(ex.answer) + 2, self.max_answer_len)
        trace_len = min(len(ex.trace) + 2, self.max_trace_len)
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
            "answer": ex.answer,
            "trace": ex.trace,
            "problem": ex.problem,
            "op_label": ex.op_label,
            "difficulty": ex.difficulty,
        }
