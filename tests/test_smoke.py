import torch

from ssl_reasoner.data import MathDataset, generate_math_examples
from ssl_reasoner.model import MathJEPAReadout
from ssl_reasoner.tokenizer import build_math_tokenizer


def test_forward_shapes():
    tokenizer = build_math_tokenizer()
    dataset = MathDataset(generate_math_examples(4), tokenizer)
    batch = [dataset[i] for i in range(4)]
    problem_ids = torch.stack([item["problem_ids"] for item in batch])
    answer_ids = torch.stack([item["answer_ids"] for item in batch])
    answer_len = torch.stack([item["answer_len"] for item in batch])

    model = MathJEPAReadout(vocab_size=tokenizer.vocab_size)
    out = model(problem_ids, answer_ids, answer_len)

    assert out["loss"].ndim == 0
    assert out["pred_slots"].shape == (4, 8, 128)
    decoded = model.solve_ids(problem_ids, pad_id=tokenizer.pad_id)
    assert len(decoded) == 4
