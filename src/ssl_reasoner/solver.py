from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import MATH_FEATURE_VOCAB_SIZE, encode_math_features
from .model import MathJEPAReadout
from .tokenizer import CharTokenizer, build_math_tokenizer
from .verifier import operation_candidate_text


@dataclass(frozen=True)
class MathSolveResult:
    problem: str
    answer: str
    mode: str
    operation_answer: str | None
    readout_answer: str
    operation_ids: list[int]


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
def solve_problem_texts(
    model: MathJEPAReadout,
    tokenizer: CharTokenizer,
    problems: list[str],
    device: torch.device,
    max_problem_len: int,
    max_math_len: int,
    batch_size: int = 64,
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

        if model.use_reasoning_trace:
            op_ids, _ = model.predict_structured_trace(problem_ids, math_ids=math_ids)
            op_rows = op_ids.detach().cpu().tolist()
        else:
            op_rows = [[] for _ in batch_problems]

        for problem, op_row, readout_answer in zip(
            batch_problems, op_rows, readout_answers
        ):
            operation_answer = operation_candidate_text(problem, op_row)
            if operation_answer is None:
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
                    readout_answer=readout_answer,
                    operation_ids=[int(idx) for idx in op_row],
                )
            )
    return results
