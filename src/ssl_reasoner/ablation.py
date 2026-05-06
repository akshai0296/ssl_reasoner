from __future__ import annotations

import argparse
import copy

import torch
from torch.utils.data import DataLoader

from .data import MathDataset, generate_math_examples
from .model import MathJEPAReadout, ParallelReadoutDecoder
from .tokenizer import build_math_tokenizer


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def exact_match(model, dataset, tokenizer, device, batch_size: int) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    total = 0
    for batch in loader:
        decoded = model.solve_ids(batch["problem_ids"].to(device), pad_id=tokenizer.pad_id)
        predictions = [tokenizer.decode(ids).strip() for ids in decoded]
        for pred, answer in zip(predictions, batch["answer"]):
            correct += pred == answer
            total += 1
    return correct / max(total, 1)


def train_random_readout_projection_only(
    model: MathJEPAReadout,
    train_dataset,
    device,
    steps: int,
    batch_size: int,
    lr: float,
) -> None:
    """Approximate plan gate: random decoder gets only a small latent projection adapter."""
    d_model = model.readout.query_embed.size(-1)
    vocab_size = model.readout.lm_head.out_features
    max_answer_len = model.readout.max_answer_len
    model.readout = ParallelReadoutDecoder(d_model, vocab_size, max_answer_len).to(device)
    adapter = torch.nn.Linear(d_model, d_model).to(device)

    for module in [model.problem_encoder, model.target_encoder, model.target_encoder_ema, model.predictor, model.readout]:
        for param in module.parameters():
            param.requires_grad = False
    for param in adapter.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0.01)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    step = 0
    while step < steps:
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                slots = model.predict_slots(batch["problem_ids"].to(device))
            out = model.readout_loss(
                adapter(slots),
                batch["answer_ids"].to(device),
                batch["answer_len"].to(device),
            )
            out["loss"].backward()
            optimizer.step()
            step += 1
            if step >= steps:
                break

    original_solve_ids = model.solve_ids

    @torch.no_grad()
    def solve_ids_with_adapter(problem_ids: torch.Tensor, pad_id: int):
        slots = adapter(model.predict_slots(problem_ids))
        return model.readout.decode_ids(slots, pad_id=pad_id)

    model.solve_ids = solve_ids_with_adapter  # type: ignore[method-assign]
    model._original_solve_ids = original_solve_ids  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--ablation-train-size", type=int, default=256)
    parser.add_argument("--ablation-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = _device(args.device)
    tokenizer = build_math_tokenizer()
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt["args"]
    model = MathJEPAReadout(
        vocab_size=tokenizer.vocab_size,
        max_problem_len=train_args["max_problem_len"],
        max_answer_len=train_args["max_answer_len"],
        d_model=train_args["d_model"],
        num_slots=train_args["num_slots"],
        encoder_layers=train_args.get("encoder_layers", 2),
        predictor_layers=train_args.get("predictor_layers", 3),
        readout_layers=train_args.get("readout_layers", 2),
        num_heads=train_args.get("num_heads", 4),
        predictor_type=train_args.get("predictor_type", "pooled"),
    ).to(device)
    model.load_state_dict(ckpt["model"])

    eval_dataset = MathDataset(
        generate_math_examples(args.samples, seed=args.seed),
        tokenizer,
        train_args["max_problem_len"],
        train_args["max_answer_len"],
    )
    ablation_train = MathDataset(
        generate_math_examples(args.ablation_train_size, seed=args.seed + 1),
        tokenizer,
        train_args["max_problem_len"],
        train_args["max_answer_len"],
    )

    baseline = exact_match(model, eval_dataset, tokenizer, device, args.batch_size)
    ablated = copy.deepcopy(model)
    train_random_readout_projection_only(
        ablated,
        ablation_train,
        device,
        steps=args.ablation_steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    ablated_acc = exact_match(ablated, eval_dataset, tokenizer, device, args.batch_size)
    ratio = ablated_acc / max(baseline, 1e-8)

    print(f"baseline_exact={baseline:.3f}")
    print(f"random_decoder_projection_only_exact={ablated_acc:.3f}")
    print(f"ablation_ratio={ratio:.3f}")
    print(f"passes_ablation_gate={ratio < 0.8}")


if __name__ == "__main__":
    main()
