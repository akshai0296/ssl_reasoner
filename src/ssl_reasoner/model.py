from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPoolEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        encoded, mask = self.forward_sequence(ids, pad_id=pad_id)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(encoded.dtype)
        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.norm(pooled)

    def forward_sequence(
        self, ids: torch.Tensor, pad_id: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = ids.ne(pad_id)
        x = self.token_embed(ids) + self.pos_embed[:, : ids.size(1)]
        x = self.encoder(x, src_key_padding_mask=~mask)
        return self.norm(x), mask


class SlotTargetEncoder(nn.Module):
    """Encode answer tokens into K latent slots via learned queries."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        num_slots: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
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
    def __init__(
        self,
        d_model: int,
        num_slots: int,
        num_layers: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
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


class CrossAttentionSequencePredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_slots: int,
        num_layers: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.slot_queries = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "cross": nn.MultiheadAttention(
                            d_model, num_heads=num_heads, batch_first=True
                        ),
                        "self": nn.MultiheadAttention(
                            d_model, num_heads=num_heads, batch_first=True
                        ),
                        "ff": nn.Sequential(
                            nn.LayerNorm(d_model),
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),
                            nn.Linear(d_model * 4, d_model),
                        ),
                        "norm_cross": nn.LayerNorm(d_model),
                        "norm_self": nn.LayerNorm(d_model),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, context_tokens: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        slots = self.slot_queries.expand(context_tokens.size(0), -1, -1)
        key_padding_mask = ~context_mask
        for layer in self.layers:
            residual = slots
            cross_in = layer["norm_cross"](slots)
            cross_out, _ = layer["cross"](
                cross_in,
                context_tokens,
                context_tokens,
                key_padding_mask=key_padding_mask,
            )
            slots = residual + cross_out

            residual = slots
            self_in = layer["norm_self"](slots)
            self_out, _ = layer["self"](self_in, self_in, self_in)
            slots = residual + self_out
            slots = slots + layer["ff"](slots)
        return self.norm(slots)


class ParallelReadoutDecoder(nn.Module):
    """Deterministic latent-slot to answer-token readout."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_answer_len: int,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.max_answer_len = max_answer_len
        self.query_embed = nn.Parameter(torch.randn(1, max_answer_len, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
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
        encoder_layers: int = 2,
        predictor_layers: int = 3,
        readout_layers: int = 2,
        num_heads: int = 4,
        predictor_type: str = "pooled",
    ):
        super().__init__()
        self.predictor_type = predictor_type
        self.problem_encoder = MeanPoolEncoder(
            vocab_size, d_model, max_problem_len, num_layers=encoder_layers, num_heads=num_heads
        )
        self.target_encoder = SlotTargetEncoder(
            vocab_size, d_model, max_answer_len, num_slots, num_heads=num_heads
        )
        self.target_encoder_ema = copy.deepcopy(self.target_encoder)
        for param in self.target_encoder_ema.parameters():
            param.requires_grad = False
        if predictor_type == "pooled":
            self.predictor = SequencePredictor(
                d_model, num_slots, num_layers=predictor_layers, num_heads=num_heads
            )
        elif predictor_type == "cross_attn":
            self.predictor = CrossAttentionSequencePredictor(
                d_model, num_slots, num_layers=predictor_layers, num_heads=num_heads
            )
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type}")
        self.readout = ParallelReadoutDecoder(
            d_model, vocab_size, max_answer_len, num_layers=readout_layers, num_heads=num_heads
        )

    @torch.no_grad()
    def ema_update_target_encoder(self, decay: float = 0.996) -> None:
        for ema_param, param in zip(
            self.target_encoder_ema.parameters(), self.target_encoder.parameters()
        ):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)

    @torch.no_grad()
    def sync_target_encoder_ema(self) -> None:
        self.target_encoder_ema.load_state_dict(self.target_encoder.state_dict())

    def encode_context(self, problem_ids: torch.Tensor) -> torch.Tensor:
        return self.problem_encoder(problem_ids)

    def encode_context_sequence(
        self, problem_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.problem_encoder.forward_sequence(problem_ids)

    def encode_target(self, answer_ids: torch.Tensor, use_ema: bool = True) -> torch.Tensor:
        encoder = self.target_encoder_ema if use_ema else self.target_encoder
        return encoder(answer_ids)

    def predict_slots(self, problem_ids: torch.Tensor) -> torch.Tensor:
        if self.predictor_type == "cross_attn":
            tokens, mask = self.encode_context_sequence(problem_ids)
            return self.predictor(tokens, mask)
        return self.predictor(self.encode_context(problem_ids))

    def readout_loss(
        self,
        slots: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_len: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        readout = self.readout(slots)
        token_loss = F.cross_entropy(
            readout["token_logits"].transpose(1, 2),
            answer_ids,
            ignore_index=0,
        )
        length_loss = F.cross_entropy(
            readout["length_logits"], answer_len.clamp(max=answer_ids.size(1) - 1)
        )
        return {
            "loss": token_loss + 0.1 * length_loss,
            "token_loss": token_loss,
            "length_loss": length_loss,
        }

    @staticmethod
    def info_nce_loss(pred_slots: torch.Tensor, target_slots: torch.Tensor, temp: float = 0.07):
        pred = F.normalize(pred_slots.flatten(start_dim=1), dim=-1)
        target = F.normalize(target_slots.flatten(start_dim=1), dim=-1)
        logits = pred @ target.T / temp
        labels = torch.arange(logits.size(0), device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    @staticmethod
    def vicreg_loss(slots: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        x = slots.flatten(0, 1)
        x = x - x.mean(dim=0)
        std = torch.sqrt(x.var(dim=0) + eps)
        var_loss = torch.mean(F.relu(1.0 - std))
        cov = (x.T @ x) / max(x.size(0) - 1, 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = off_diag.pow(2).sum() / x.size(1)
        return var_loss + cov_loss

    @staticmethod
    def slot_diversity_loss(slots: torch.Tensor) -> torch.Tensor:
        """Penalize slots within each example becoming near-identical."""
        normed = F.normalize(slots, dim=-1)
        cosine = normed @ normed.transpose(1, 2)
        eye = torch.eye(cosine.size(1), dtype=torch.bool, device=cosine.device).unsqueeze(0)
        off_diag = cosine.masked_select(~eye)
        return off_diag.pow(2).mean()

    @staticmethod
    def batch_diversity_loss(slots: torch.Tensor) -> torch.Tensor:
        """Penalize different examples having overly similar pooled latents."""
        pooled = F.normalize(slots.mean(dim=1), dim=-1)
        cosine = pooled @ pooled.T
        if cosine.size(0) <= 1:
            return cosine.new_tensor(0.0)
        eye = torch.eye(cosine.size(0), dtype=torch.bool, device=cosine.device)
        off_diag = cosine.masked_select(~eye)
        return off_diag.pow(2).mean()

    def stage0_target_autoencode(
        self,
        answer_ids: torch.Tensor,
        answer_len: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_slots = self.encode_target(answer_ids, use_ema=False)
        out = self.readout_loss(target_slots, answer_ids, answer_len)
        return {
            "loss": out["loss"],
            "token_loss": out["token_loss"],
            "length_loss": out["length_loss"],
            "target_slots": target_slots.detach(),
        }

    def stage1_predictor(
        self,
        problem_ids: torch.Tensor,
        answer_ids: torch.Tensor,
        contrastive_weight: float = 0.1,
        vicreg_weight: float = 0.05,
        slot_diversity_weight: float = 0.1,
        batch_diversity_weight: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        pred_slots = self.predict_slots(problem_ids)
        with torch.no_grad():
            target_slots = self.encode_target(answer_ids, use_ema=True)
        pred_loss = F.smooth_l1_loss(pred_slots, target_slots)
        contrastive_loss = self.info_nce_loss(pred_slots, target_slots)
        vicreg = self.vicreg_loss(pred_slots)
        diversity_loss = self.slot_diversity_loss(pred_slots)
        batch_diversity = self.batch_diversity_loss(pred_slots)
        loss = (
            pred_loss
            + contrastive_weight * contrastive_loss
            + vicreg_weight * vicreg
            + slot_diversity_weight * diversity_loss
            + batch_diversity_weight * batch_diversity
        )
        return {
            "loss": loss,
            "pred_loss": pred_loss,
            "contrastive_loss": contrastive_loss,
            "vicreg_loss": vicreg,
            "slot_diversity_loss": diversity_loss,
            "batch_diversity_loss": batch_diversity,
            "pred_slots": pred_slots.detach(),
            "target_slots": target_slots.detach(),
        }

    def stage2_readout(
        self,
        problem_ids: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_len: torch.Tensor,
        true_latent_ratio: float = 0.5,
        target_readout_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            pred_slots = self.predict_slots(problem_ids)
            true_slots = self.encode_target(answer_ids, use_ema=True)
            use_true = torch.rand(
                pred_slots.size(0), 1, 1, device=problem_ids.device
            ) < true_latent_ratio
            slots = torch.where(use_true, true_slots, pred_slots)
        out = self.readout_loss(slots, answer_ids, answer_len)
        target_out = self.readout_loss(true_slots, answer_ids, answer_len)
        total_loss = out["loss"] + target_readout_weight * target_out["loss"]
        return {
            "loss": total_loss,
            "token_loss": out["token_loss"],
            "length_loss": out["length_loss"],
            "target_token_loss": target_out["token_loss"],
            "target_length_loss": target_out["length_loss"],
            "used_true_latent": use_true.float().mean(),
        }

    def forward(self, problem_ids: torch.Tensor, answer_ids: torch.Tensor, answer_len: torch.Tensor):
        pred_out = self.stage1_predictor(problem_ids, answer_ids)
        dec_out = self.readout_loss(pred_out["pred_slots"], answer_ids, answer_len)
        total_loss = pred_out["loss"] + dec_out["loss"]
        return {
            "loss": total_loss,
            "pred_loss": pred_out["pred_loss"],
            "contrastive_loss": pred_out["contrastive_loss"],
            "vicreg_loss": pred_out["vicreg_loss"],
            "slot_diversity_loss": pred_out["slot_diversity_loss"],
            "batch_diversity_loss": pred_out["batch_diversity_loss"],
            "token_loss": dec_out["token_loss"],
            "length_loss": dec_out["length_loss"],
            "pred_slots": pred_out["pred_slots"],
        }

    @torch.no_grad()
    def solve_ids(self, problem_ids: torch.Tensor, pad_id: int) -> list[list[int]]:
        slots = self.predict_slots(problem_ids)
        return self.readout.decode_ids(slots, pad_id=pad_id)

    @torch.no_grad()
    def solve_ids_from_target(
        self,
        answer_ids: torch.Tensor,
        pad_id: int,
        use_ema: bool = True,
    ) -> list[list[int]]:
        slots = self.encode_target(answer_ids, use_ema=use_ema)
        return self.readout.decode_ids(slots, pad_id=pad_id)
