import torch

from ssl_reasoner.data import (
    MathDataset,
    encode_math_features,
    generate_math_examples,
    make_trace,
    make_trace_fields,
)
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
    math_ids = torch.stack([item["math_ids"] for item in batch])

    model = MathJEPAReadout(vocab_size=tokenizer.vocab_size)
    out = model(problem_ids, answer_ids, answer_len, math_ids=math_ids)

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
    math_ids = torch.stack([item["math_ids"] for item in batch])

    model = MathJEPAReadout(vocab_size=tokenizer.vocab_size, predictor_type="cross_attn")
    out = model(problem_ids, answer_ids, answer_len, math_ids=math_ids)

    assert out["loss"].ndim == 0
    assert out["pred_slots"].shape == (4, 8, 128)


def test_math_features_extract_expression_tokens():
    ids = encode_math_features("Find the value of 12+7*3.", max_len=8)
    assert ids[:5] == [13, 202, 8, 204, 4]
    assert ids[5:] == [0, 0, 0]


def test_single_op_balanced_curriculum_cycles_operations():
    examples = generate_math_examples(6, seed=0, curriculum="single_op_balanced")
    assert [example.op_label for example in examples] == ["+", "-", "*", "+", "-", "*"]
    assert {example.difficulty for example in examples} == {0}


def test_mixed_only_curriculum_uses_three_term_expressions():
    examples = generate_math_examples(6, seed=0, curriculum="mixed_only")
    assert {example.op_label for example in examples} == {"mixed"}
    assert {example.difficulty for example in examples} == {1}


def test_trace_respects_multiplication_precedence():
    assert make_trace("20+1*0") == "1*0=0 20+0=20"
    assert make_trace("30+41+12") == "30+41=71 71+12=83"


def test_structured_trace_fields_respect_precedence():
    op_ids, values, mask = make_trace_fields("20+1*0")
    assert op_ids == [3, 1]
    assert values == [201, 200, 200, 220, 200, 220]
    assert mask == [1.0] * 6


def test_cross_attention_with_math_features_forward():
    tokenizer = build_math_tokenizer()
    dataset = MathDataset(generate_math_examples(4), tokenizer)
    batch = [dataset[i] for i in range(4)]
    problem_ids = torch.stack([item["problem_ids"] for item in batch])
    answer_ids = torch.stack([item["answer_ids"] for item in batch])
    answer_len = torch.stack([item["answer_len"] for item in batch])
    math_ids = torch.stack([item["math_ids"] for item in batch])

    model = MathJEPAReadout(
        vocab_size=tokenizer.vocab_size,
        predictor_type="cross_attn",
        use_math_features=True,
    )
    out = model(problem_ids, answer_ids, answer_len, math_ids=math_ids)

    assert out["loss"].ndim == 0
    assert out["pred_slots"].shape == (4, 8, 128)


def test_reasoning_trace_forward():
    tokenizer = build_math_tokenizer()
    dataset = MathDataset(generate_math_examples(4), tokenizer)
    batch = [dataset[i] for i in range(4)]
    problem_ids = torch.stack([item["problem_ids"] for item in batch])
    answer_ids = torch.stack([item["answer_ids"] for item in batch])
    answer_len = torch.stack([item["answer_len"] for item in batch])
    math_ids = torch.stack([item["math_ids"] for item in batch])
    trace_ids = torch.stack([item["trace_ids"] for item in batch])
    trace_len = torch.stack([item["trace_len"] for item in batch])
    trace_op_ids = torch.stack([item["trace_op_ids"] for item in batch])
    trace_value_ids = torch.stack([item["trace_value_ids"] for item in batch])
    trace_value_mask = torch.stack([item["trace_value_mask"] for item in batch])

    model = MathJEPAReadout(
        vocab_size=tokenizer.vocab_size,
        predictor_type="cross_attn",
        use_math_features=True,
        use_reasoning_trace=True,
    )
    out = model(
        problem_ids,
        answer_ids,
        answer_len,
        math_ids=math_ids,
        trace_ids=trace_ids,
        trace_len=trace_len,
        trace_op_ids=trace_op_ids,
        trace_value_ids=trace_value_ids,
        trace_value_mask=trace_value_mask,
    )
    decoded_trace = model.solve_trace_ids(problem_ids, pad_id=tokenizer.pad_id, math_ids=math_ids)
    pred_ops, pred_values = model.predict_structured_trace(problem_ids, math_ids=math_ids)

    assert out["loss"].ndim == 0
    assert out["pred_slots"].shape == (4, 8, 128)
    assert out["trace_pred_loss"].ndim == 0
    assert out["trace_struct_op_loss"].ndim == 0
    assert len(decoded_trace) == 4
    assert pred_ops.shape == (4, 2)
    assert pred_values.shape == (4, 6)


def test_trace_fusion_forward():
    tokenizer = build_math_tokenizer()
    dataset = MathDataset(generate_math_examples(4), tokenizer)
    batch = [dataset[i] for i in range(4)]
    problem_ids = torch.stack([item["problem_ids"] for item in batch])
    answer_ids = torch.stack([item["answer_ids"] for item in batch])
    answer_len = torch.stack([item["answer_len"] for item in batch])
    math_ids = torch.stack([item["math_ids"] for item in batch])
    trace_ids = torch.stack([item["trace_ids"] for item in batch])
    trace_len = torch.stack([item["trace_len"] for item in batch])
    trace_op_ids = torch.stack([item["trace_op_ids"] for item in batch])
    trace_value_ids = torch.stack([item["trace_value_ids"] for item in batch])
    trace_value_mask = torch.stack([item["trace_value_mask"] for item in batch])

    model = MathJEPAReadout(
        vocab_size=tokenizer.vocab_size,
        predictor_type="cross_attn",
        use_math_features=True,
        use_reasoning_trace=True,
        use_trace_fusion=True,
    )
    out = model(
        problem_ids,
        answer_ids,
        answer_len,
        math_ids=math_ids,
        trace_ids=trace_ids,
        trace_len=trace_len,
        trace_op_ids=trace_op_ids,
        trace_value_ids=trace_value_ids,
        trace_value_mask=trace_value_mask,
    )
    decoded = model.solve_ids(problem_ids, pad_id=tokenizer.pad_id, math_ids=math_ids)

    assert out["loss"].ndim == 0
    assert out["pred_slots"].shape == (4, 8, 128)
    assert len(decoded) == 4
