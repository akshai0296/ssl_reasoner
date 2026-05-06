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


MATH_FEATURE_PAD_ID = 0
MATH_FEATURE_NUM_OFFSET = 1
MATH_FEATURE_PLUS_ID = 202
MATH_FEATURE_MINUS_ID = 203
MATH_FEATURE_TIMES_ID = 204
MATH_FEATURE_VOCAB_SIZE = 205


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
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_problem_len = max_problem_len
        self.max_answer_len = max_answer_len
        self.max_math_len = max_math_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        ex = self.examples[idx]
        problem_ids = self.tokenizer.encode(ex.problem, self.max_problem_len)
        answer_ids = self.tokenizer.encode(ex.answer, self.max_answer_len)
        math_ids = encode_math_features(ex.problem, self.max_math_len)
        answer_len = min(len(ex.answer) + 2, self.max_answer_len)
        return {
            "problem_ids": torch.tensor(problem_ids, dtype=torch.long),
            "math_ids": torch.tensor(math_ids, dtype=torch.long),
            "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
            "answer_len": torch.tensor(answer_len, dtype=torch.long),
            "answer": ex.answer,
            "problem": ex.problem,
            "op_label": ex.op_label,
            "difficulty": ex.difficulty,
        }
