import torch

from ssl_reasoner.data import MathDataset, generate_math_examples
from ssl_reasoner.diagnostics import latent_health
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
    assert out["slot_diversity_loss"].ndim == 0
    assert out["batch_diversity_loss"].ndim == 0
    decoded = model.solve_ids(problem_ids, pad_id=tokenizer.pad_id)
    target_decoded = model.solve_ids_from_target(answer_ids, pad_id=tokenizer.pad_id)
    assert len(decoded) == 4
    assert len(target_decoded) == 4


def test_latent_health_keys():
    tokenizer = build_math_tokenizer()
    dataset = MathDataset(generate_math_examples(8), tokenizer)
    model = MathJEPAReadout(vocab_size=tokenizer.vocab_size)
    stats = latent_health(model, dataset, torch.device("cpu"), batch_size=4)

    assert "mean_random_cosine" in stats
    assert "effective_rank_ratio" in stats
    assert "passes_cosine" in stats
    assert "passes_rank" in stats


def test_cross_attention_predictor_forward():
    tokenizer = build_math_tokenizer()
    dataset = MathDataset(generate_math_examples(4), tokenizer)
    batch = [dataset[i] for i in range(4)]
    problem_ids = torch.stack([item["problem_ids"] for item in batch])
    answer_ids = torch.stack([item["answer_ids"] for item in batch])
    answer_len = torch.stack([item["answer_len"] for item in batch])

    model = MathJEPAReadout(vocab_size=tokenizer.vocab_size, predictor_type="cross_attn")
    out = model(problem_ids, answer_ids, answer_len)

    assert out["loss"].ndim == 0
    assert out["pred_slots"].shape == (4, 8, 128)
