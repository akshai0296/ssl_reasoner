import re

import torch

from ssl_reasoner.data import (
    MathDataset,
    encode_math_features,
    generate_math_examples,
    make_trace,
    make_trace_fields,
)
from ssl_reasoner.diagnostics import latent_health
from ssl_reasoner.model import LatentVerifier, MathJEPAReadout
from ssl_reasoner.solver import solve_problem_texts
from ssl_reasoner.tokenizer import build_math_tokenizer
from ssl_reasoner.verifier import operation_candidate_text, symbolic_candidate_texts


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


def test_compositional_single_splits_use_disjoint_operand_ranges():
    seen = generate_math_examples(9, seed=0, curriculum="seen_single")
    unseen = generate_math_examples(9, seed=0, curriculum="unseen_single")

    assert {example.split_label for example in seen} == {"seen_single"}
    assert {example.split_label for example in unseen} == {"unseen_single"}
    assert [example.op_label for example in seen] == ["+", "-", "*"] * 3

    for example in seen:
        operands = [int(token) for token in re.findall(r"\d+", example.problem)]
        limit = 10 if example.op_label == "*" else 50
        assert all(0 <= operand <= limit for operand in operands)

    for example in unseen:
        operands = [int(token) for token in re.findall(r"\d+", example.problem)]
        lower = 11 if example.op_label == "*" else 51
        assert all(lower <= operand for operand in operands)


def test_compositional_mixed_splits_use_disjoint_operand_ranges():
    seen = generate_math_examples(8, seed=1, curriculum="seen_mixed")
    unseen = generate_math_examples(8, seed=1, curriculum="unseen_mixed")

    assert {example.split_label for example in seen} == {"seen_mixed"}
    assert {example.split_label for example in unseen} == {"unseen_mixed"}
    assert {example.op_label for example in seen + unseen} == {"mixed"}
    assert {example.difficulty for example in seen + unseen} == {1}

    for example in seen:
        a, b, c = [int(token) for token in re.findall(r"\d+", example.problem)]
        assert 0 <= a <= 25
        assert 0 <= b <= 25
        assert 0 <= c <= 10

    for example in unseen:
        a, b, c = [int(token) for token in re.findall(r"\d+", example.problem)]
        assert 26 <= a <= 50
        assert 26 <= b <= 50
        assert 11 <= c <= 20


def test_compositional_train_mixes_seen_single_and_seen_mixed():
    examples = generate_math_examples(10, seed=2, curriculum="compositional_train")
    assert {example.split_label for example in examples} == {"seen_single", "seen_mixed"}


def test_trace_respects_multiplication_precedence():
    assert make_trace("20+1*0") == "1*0=0 20+0=20"
    assert make_trace("30+41+12") == "30+41=71 71+12=83"


def test_symbolic_candidate_texts_include_precedence_and_variants():
    candidates = symbolic_candidate_texts("Find the value of 20+1*0.", base_prediction="19")
    assert candidates[0] == "20"
    assert "0" in candidates
    assert "18" in candidates
    assert "20" in candidates


def test_symbolic_candidate_texts_include_left_to_right_for_mixed():
    candidates = symbolic_candidate_texts("What is 2+3*4?")
    assert "14" in candidates
    assert "20" in candidates
    assert "12" in candidates


def test_symbolic_candidate_texts_can_exclude_oracle_precedence_result():
    candidates = symbolic_candidate_texts("What is 2+3*4?", include_oracle=False)
    assert "14" not in candidates
    assert "20" in candidates
    assert "12" in candidates


def test_operation_candidate_text_executes_predicted_operation_order():
    assert operation_candidate_text("What is 2+3*4?", [3, 1]) == "14"
    assert operation_candidate_text("What is 2+3*4?", [1, 3]) == "20"
    assert operation_candidate_text("Calculate 9-4.", [2, 0]) == "5"
    assert operation_candidate_text("Calculate 9-4-2.", [2, 2]) == "3"
    assert operation_candidate_text("Calculate 9-4+2.", [2, 1]) == "7"
    assert operation_candidate_text("Find 30+10+5-5.", [1, 1]) is None


def test_solve_problem_texts_uses_operation_then_readout_fallback():
    tokenizer = build_math_tokenizer()

    class FakeModel:
        use_reasoning_trace = True

        def solve_ids(self, problem_ids, pad_id, math_ids=None):
            return [
                tokenizer.encode("999", max_len=16),
                tokenizer.encode("42", max_len=16),
            ]

        def predict_structured_trace(self, problem_ids, math_ids=None):
            return torch.tensor([[3, 1], [1, 0]]), None

        def predict_trace_slots(self, problem_ids, math_ids=None):
            return torch.zeros(problem_ids.size(0), 8, 128)

        def trace_struct_head(self, pooled):
            logits = torch.full((pooled.size(0), 8 + 6 * 1401), -10.0)
            logits[0, 3] = 10.0
            logits[0, 4 + 1] = 10.0
            if pooled.size(0) > 1:
                logits[1, 1] = 10.0
                logits[1, 4] = 10.0
            return logits

    results = solve_problem_texts(
        FakeModel(),
        tokenizer,
        ["What is 2+3*4?", "No expression here."],
        torch.device("cpu"),
        max_problem_len=64,
        max_math_len=8,
    )

    assert results[0].answer == "14"
    assert results[0].mode == "operation"
    assert results[0].min_operation_confidence > 0.99
    assert results[0].readout_answer == "999"
    assert results[1].answer == "42"
    assert results[1].mode == "readout"

    gated_results = solve_problem_texts(
        FakeModel(),
        tokenizer,
        ["What is 2+3*4?"],
        torch.device("cpu"),
        max_problem_len=64,
        max_math_len=8,
        operation_confidence_threshold=1.1,
    )
    assert gated_results[0].answer == "999"
    assert gated_results[0].mode == "readout"

    parsed_results = solve_problem_texts(
        FakeModel(),
        tokenizer,
        ["Find 30+10+5-5."],
        torch.device("cpu"),
        max_problem_len=64,
        max_math_len=8,
    )
    assert parsed_results[0].answer == "40"
    assert parsed_results[0].mode == "parsed_expression"
    assert parsed_results[0].operation_answer is None
    assert parsed_results[0].parsed_expression_answer == "40"


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
    answer_value_id = torch.stack([item["answer_value_id"] for item in batch])

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
        answer_value_id=answer_value_id,
    )
    decoded_trace = model.solve_trace_ids(problem_ids, pad_id=tokenizer.pad_id, math_ids=math_ids)
    pred_ops, pred_values = model.predict_structured_trace(problem_ids, math_ids=math_ids)
    reason_ops, reason_values = model.predict_reasoning_struct(problem_ids, math_ids=math_ids)
    answer_pred, answer_conf = model.predict_structured_answer(problem_ids, math_ids=math_ids)

    assert out["loss"].ndim == 0
    assert out["pred_slots"].shape == (4, 8, 128)
    assert out["trace_pred_loss"].ndim == 0
    assert out["trace_struct_op_loss"].ndim == 0
    assert len(decoded_trace) == 4
    assert pred_ops.shape == (4, 2)
    assert pred_values.shape == (4, 6)
    assert reason_ops.shape == (4, 2)
    assert reason_values.shape == (4, 2)
    assert answer_pred.shape == (4,)
    assert answer_conf.shape == (4,)


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
    answer_value_id = torch.stack([item["answer_value_id"] for item in batch])

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
        answer_value_id=answer_value_id,
    )
    decoded = model.solve_ids(problem_ids, pad_id=tokenizer.pad_id, math_ids=math_ids)

    assert out["loss"].ndim == 0
    assert out["pred_slots"].shape == (4, 8, 128)
    assert len(decoded) == 4


def test_stage3_joint_forward():
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
    answer_value_id = torch.stack([item["answer_value_id"] for item in batch])

    model = MathJEPAReadout(
        vocab_size=tokenizer.vocab_size,
        predictor_type="cross_attn",
        use_math_features=True,
        use_reasoning_trace=True,
        use_trace_fusion=True,
    )
    out = model.stage3_joint(
        problem_ids,
        answer_ids,
        answer_len,
        math_ids=math_ids,
        trace_ids=trace_ids,
        trace_len=trace_len,
        trace_op_ids=trace_op_ids,
        trace_value_ids=trace_value_ids,
        trace_value_mask=trace_value_mask,
        answer_value_id=answer_value_id,
    )

    assert out["loss"].ndim == 0
    assert out["pred_loss"].ndim == 0
    assert out["token_loss"].ndim == 0
    assert out["length_loss"].ndim == 0
    assert out["reasoning_struct_op_loss"].ndim == 0
    assert out["reasoning_struct_value_loss"].ndim == 0
    assert out["pred_slots"].shape == (4, 8, 128)


def test_latent_verifier_forward():
    verifier = LatentVerifier(d_model=128)
    context = torch.randn(4, 128)
    candidate_slots = torch.randn(4, 8, 128)
    logits = verifier(context, candidate_slots)

    assert logits.shape == (4,)
