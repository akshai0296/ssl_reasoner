from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPoolEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, num_layers: int = 2):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        mask = ids.ne(pad_id)
        x = self.token_embed(ids) + self.pos_embed[:, : ids.size(1)]
        x = self.encoder(x, src_key_padding_mask=~mask)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.norm(pooled)


class SlotTargetEncoder(nn.Module):
    """Encode answer tokens into K latent slots via learned queries."""

    def __init__(self, vocab_size: int, d_model: int, max_len: int, num_slots: int):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        mask = ids.ne(pad_id)
        memory = self.token_embed(ids) + self.pos_embed[:, : ids.size(1)]
        queries = self.slot_queries.expand(ids.size(0), -1, -1)
        slots, _ = self.cross_attn(queries, memory, memory, key_padding_mask=~mask)
        slots = slots + self.ff(slots)
        return self.norm(slots)


class SequencePredictor(nn.Module):
    def __init__(self, d_model: int, num_slots: int, num_layers: int = 3):
        super().__init__()
        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.context_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        context_token = self.context_proj(context).unsqueeze(1)
        slots = self.slot_queries.expand(context.size(0), -1, -1)
        x = torch.cat([context_token, slots], dim=1)
        x = self.encoder(x)
        return self.norm(x[:, 1:])


class ParallelReadoutDecoder(nn.Module):
    """Deterministic latent-slot to answer-token readout."""

    def __init__(self, d_model: int, vocab_size: int, max_answer_len: int, num_layers: int = 2):
        super().__init__()
        self.max_answer_len = max_answer_len
        self.query_embed = nn.Parameter(torch.randn(1, max_answer_len, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.length_head = nn.Linear(d_model, max_answer_len)

    def forward(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        queries = self.query_embed.expand(slots.size(0), -1, -1)
        h = self.decoder(tgt=queries, memory=slots)
        return {
            "token_logits": self.lm_head(h),
            "length_logits": self.length_head(slots.mean(dim=1)),
        }

    @torch.no_grad()
    def decode_ids(self, slots: torch.Tensor, pad_id: int) -> list[list[int]]:
        out = self.forward(slots)
        token_ids = out["token_logits"].argmax(dim=-1)
        lengths = out["length_logits"].argmax(dim=-1).clamp(min=1)
        decoded = []
        for row, length in zip(token_ids, lengths):
            ids = row[: int(length.item())].tolist()
            ids.extend([pad_id] * (self.max_answer_len - len(ids)))
            decoded.append(ids)
        return decoded


class MathJEPAReadout(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_problem_len: int = 64,
        max_answer_len: int = 16,
        d_model: int = 128,
        num_slots: int = 8,
    ):
        super().__init__()
        self.problem_encoder = MeanPoolEncoder(vocab_size, d_model, max_problem_len)
        self.target_encoder = SlotTargetEncoder(vocab_size, d_model, max_answer_len, num_slots)
        self.predictor = SequencePredictor(d_model, num_slots)
        self.readout = ParallelReadoutDecoder(d_model, vocab_size, max_answer_len)

    def forward(self, problem_ids: torch.Tensor, answer_ids: torch.Tensor, answer_len: torch.Tensor):
        context = self.problem_encoder(problem_ids)
        with torch.no_grad():
            target_slots = self.target_encoder(answer_ids)
        pred_slots = self.predictor(context)
        readout = self.readout(pred_slots)
        pred_loss = F.smooth_l1_loss(pred_slots, target_slots)
        token_loss = F.cross_entropy(
            readout["token_logits"].transpose(1, 2),
            answer_ids,
            ignore_index=0,
        )
        length_loss = F.cross_entropy(readout["length_logits"], answer_len.clamp(max=answer_ids.size(1) - 1))
        total_loss = pred_loss + token_loss + 0.1 * length_loss
        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "token_loss": token_loss,
            "length_loss": length_loss,
            "pred_slots": pred_slots,
        }

    @torch.no_grad()
    def solve_ids(self, problem_ids: torch.Tensor, pad_id: int) -> list[list[int]]:
        context = self.problem_encoder(problem_ids)
        slots = self.predictor(context)
        return self.readout.decode_ids(slots, pad_id=pad_id)
