from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


@dataclass(frozen=True)
class MathExample:
    problem: str
    answer: str


def _make_expression(rng: random.Random, difficulty: int) -> tuple[str, int]:
    if difficulty == 0:
        op = rng.choice(["+", "-", "*"])
        limit = 20 if op == "*" else 100
        a = rng.randint(0, limit)
        b = rng.randint(0, limit)
        if op == "-":
            a, b = max(a, b), min(a, b)
        expr = f"{a}{op}{b}"
        return expr, eval(expr)

    a = rng.randint(0, 50)
    b = rng.randint(0, 50)
    c = rng.randint(0, 20)
    op1 = rng.choice(["+", "-"])
    op2 = rng.choice(["+", "-", "*"])
    expr = f"{a}{op1}{b}{op2}{c}"
    return expr, eval(expr)


def generate_math_examples(
    n: int,
    seed: int = 0,
    difficulty_mix: tuple[float, float] = (0.75, 0.25),
) -> list[MathExample]:
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        difficulty = 0 if rng.random() < difficulty_mix[0] else 1
        expr, value = _make_expression(rng, difficulty)
        template = rng.choice(
            [
                "What is {expr}?",
                "Calculate {expr}.",
                "Find the value of {expr}.",
            ]
        )
        examples.append(MathExample(problem=template.format(expr=expr), answer=str(value)))
    return examples


class MathDataset(Dataset):
    def __init__(
        self,
        examples: list[MathExample],
        tokenizer: CharTokenizer,
        max_problem_len: int = 64,
        max_answer_len: int = 16,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_problem_len = max_problem_len
        self.max_answer_len = max_answer_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        ex = self.examples[idx]
        problem_ids = self.tokenizer.encode(ex.problem, self.max_problem_len)
        answer_ids = self.tokenizer.encode(ex.answer, self.max_answer_len)
        answer_len = min(len(ex.answer) + 2, self.max_answer_len)
        return {
            "problem_ids": torch.tensor(problem_ids, dtype=torch.long),
            "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
            "answer_len": torch.tensor(answer_len, dtype=torch.long),
            "answer": ex.answer,
            "problem": ex.problem,
        }
