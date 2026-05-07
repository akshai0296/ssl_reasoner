from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

TRACE_VALUE_CLASSES = 1401


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


class LatentVerifier(nn.Module):
    """Score whether a candidate answer latent is correct for a problem latent."""

    def __init__(self, d_model: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 2
        self.net = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context: torch.Tensor, candidate_slots: torch.Tensor) -> torch.Tensor:
        candidate = candidate_slots.mean(dim=1)
        features = torch.cat([context, candidate], dim=-1)
        return self.net(features).squeeze(-1)


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
        use_math_features: bool = False,
        math_vocab_size: int = 205,
        max_math_len: int = 8,
        use_reasoning_trace: bool = False,
        max_trace_len: int = 32,
        use_trace_fusion: bool = False,
    ):
        super().__init__()
        self.predictor_type = predictor_type
        self.use_math_features = use_math_features
        self.use_reasoning_trace = use_reasoning_trace
        self.use_trace_fusion = use_trace_fusion
        self.problem_encoder = MeanPoolEncoder(
            vocab_size, d_model, max_problem_len, num_layers=encoder_layers, num_heads=num_heads
        )
        if use_math_features:
            self.math_embed = nn.Embedding(math_vocab_size, d_model, padding_idx=0)
            self.math_pos_embed = nn.Parameter(torch.randn(1, max_math_len, d_model) * 0.02)
            self.math_norm = nn.LayerNorm(d_model)
        self.target_encoder = SlotTargetEncoder(
            vocab_size, d_model, max_answer_len, num_slots, num_heads=num_heads
        )
        self.target_encoder_ema = copy.deepcopy(self.target_encoder)
        for param in self.target_encoder_ema.parameters():
            param.requires_grad = False
        if use_reasoning_trace:
            self.trace_target_encoder = SlotTargetEncoder(
                vocab_size, d_model, max_trace_len, num_slots, num_heads=num_heads
            )
            self.trace_target_encoder_ema = copy.deepcopy(self.trace_target_encoder)
            for param in self.trace_target_encoder_ema.parameters():
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
        self.reasoning_struct_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 8 + 2 * TRACE_VALUE_CLASSES),
        )
        self.answer_value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, TRACE_VALUE_CLASSES),
        )
        if use_reasoning_trace:
            self.trace_predictor = CrossAttentionSequencePredictor(
                d_model, num_slots, num_layers=predictor_layers, num_heads=num_heads
            )
            self.trace_readout = ParallelReadoutDecoder(
                d_model, vocab_size, max_trace_len, num_layers=readout_layers, num_heads=num_heads
            )
            self.trace_struct_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 8 + 6 * TRACE_VALUE_CLASSES),
            )
            self.structured_answer_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, TRACE_VALUE_CLASSES),
            )
            if use_trace_fusion:
                self.trace_struct_summary = nn.Sequential(
                    nn.LayerNorm(8 + 6 * TRACE_VALUE_CLASSES),
                    nn.Linear(8 + 6 * TRACE_VALUE_CLASSES, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
                self.trace_fusion = nn.Sequential(
                    nn.LayerNorm(d_model * 3),
                    nn.Linear(d_model * 3, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )

    @torch.no_grad()
    def ema_update_target_encoder(self, decay: float = 0.996) -> None:
        for ema_param, param in zip(
            self.target_encoder_ema.parameters(), self.target_encoder.parameters()
        ):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
        if self.use_reasoning_trace:
            for ema_param, param in zip(
                self.trace_target_encoder_ema.parameters(),
                self.trace_target_encoder.parameters(),
            ):
                ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)

    @torch.no_grad()
    def sync_target_encoder_ema(self) -> None:
        self.target_encoder_ema.load_state_dict(self.target_encoder.state_dict())
        if self.use_reasoning_trace:
            self.trace_target_encoder_ema.load_state_dict(self.trace_target_encoder.state_dict())

    def encode_context(self, problem_ids: torch.Tensor) -> torch.Tensor:
        return self.problem_encoder(problem_ids)

    def encode_context_sequence(
        self, problem_ids: torch.Tensor, math_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, mask = self.problem_encoder.forward_sequence(problem_ids)
        if self.use_math_features and math_ids is not None:
            math_mask = math_ids.ne(0)
            math_tokens = self.math_embed(math_ids) + self.math_pos_embed[:, : math_ids.size(1)]
            math_tokens = self.math_norm(math_tokens)
            tokens = torch.cat([tokens, math_tokens], dim=1)
            mask = torch.cat([mask, math_mask], dim=1)
        return tokens, mask

    def encode_target(self, answer_ids: torch.Tensor, use_ema: bool = True) -> torch.Tensor:
        encoder = self.target_encoder_ema if use_ema else self.target_encoder
        return encoder(answer_ids)

    def encode_trace_target(self, trace_ids: torch.Tensor, use_ema: bool = True) -> torch.Tensor:
        encoder = self.trace_target_encoder_ema if use_ema else self.trace_target_encoder
        return encoder(trace_ids)

    def predict_slots(
        self, problem_ids: torch.Tensor, math_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.predictor_type == "cross_attn":
            tokens, mask = self.encode_context_sequence(problem_ids, math_ids)
            return self.predictor(tokens, mask)
        return self.predictor(self.encode_context(problem_ids))

    def predict_trace_slots(
        self, problem_ids: torch.Tensor, math_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        tokens, mask = self.encode_context_sequence(problem_ids, math_ids)
        return self.trace_predictor(tokens, mask)

    def fuse_answer_trace_slots(
        self, answer_slots: torch.Tensor, trace_slots: torch.Tensor
    ) -> torch.Tensor:
        struct_logits = self.trace_struct_head(trace_slots.mean(dim=1))
        struct_summary = self.trace_struct_summary(struct_logits).unsqueeze(1)
        struct_summary = struct_summary.expand(-1, answer_slots.size(1), -1)
        fused_input = torch.cat([answer_slots, trace_slots, struct_summary], dim=-1)
        return answer_slots + self.trace_fusion(fused_input)

    def predict_answer_slots(
        self, problem_ids: torch.Tensor, math_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        answer_slots = self.predict_slots(problem_ids, math_ids)
        if self.use_trace_fusion:
            trace_slots = self.predict_trace_slots(problem_ids, math_ids)
            answer_slots = self.fuse_answer_trace_slots(answer_slots, trace_slots)
        return answer_slots

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

    def trace_readout_loss(
        self,
        slots: torch.Tensor,
        trace_ids: torch.Tensor,
        trace_len: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        readout = self.trace_readout(slots)
        token_loss = F.cross_entropy(
            readout["token_logits"].transpose(1, 2),
            trace_ids,
            ignore_index=0,
        )
        length_loss = F.cross_entropy(
            readout["length_logits"], trace_len.clamp(max=trace_ids.size(1) - 1)
        )
        return {
            "loss": token_loss + 0.1 * length_loss,
            "trace_token_loss": token_loss,
            "trace_length_loss": length_loss,
        }

    def structured_trace_loss(
        self,
        slots: torch.Tensor,
        trace_op_ids: torch.Tensor,
        trace_value_ids: torch.Tensor,
        trace_value_mask: torch.Tensor,
        prefix: str = "trace_struct",
        head: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        head = head or self.trace_struct_head
        pred = head(slots.mean(dim=1))
        op_logits = pred[:, :8].view(-1, 2, 4)
        value_logits = pred[:, 8:].view(-1, 6, TRACE_VALUE_CLASSES)
        op_loss = F.cross_entropy(op_logits.reshape(-1, 4), trace_op_ids.reshape(-1))
        value_mask = trace_value_mask
        per_value_loss = F.cross_entropy(
            value_logits.reshape(-1, TRACE_VALUE_CLASSES),
            trace_value_ids.reshape(-1),
            reduction="none",
        ).view_as(value_mask)
        value_loss = (per_value_loss * value_mask).sum() / value_mask.sum().clamp(min=1.0)
        op_acc = (op_logits.argmax(dim=-1) == trace_op_ids).float().mean()
        value_acc = (
            (value_logits.argmax(dim=-1) == trace_value_ids).float() * value_mask
        ).sum() / value_mask.sum().clamp(min=1.0)
        return {
            "loss": op_loss + value_loss,
            f"{prefix}_op_loss": op_loss,
            f"{prefix}_value_loss": value_loss,
            f"{prefix}_op_acc": op_acc,
            f"{prefix}_value_acc": value_acc,
        }

    def reasoning_struct_loss(
        self,
        slots: torch.Tensor,
        trace_op_ids: torch.Tensor,
        trace_value_ids: torch.Tensor,
        trace_value_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred = self.reasoning_struct_head(slots.mean(dim=1))
        op_logits = pred[:, :8].view(-1, 2, 4)
        value_logits = pred[:, 8:].view(-1, 2, TRACE_VALUE_CLASSES)
        value_targets = trace_value_ids[:, [2, 5]]
        value_mask = trace_value_mask[:, [2, 5]]
        op_loss = F.cross_entropy(op_logits.reshape(-1, 4), trace_op_ids.reshape(-1))
        per_value_loss = F.cross_entropy(
            value_logits.reshape(-1, TRACE_VALUE_CLASSES),
            value_targets.reshape(-1),
            reduction="none",
        ).view_as(value_mask)
        value_loss = (per_value_loss * value_mask).sum() / value_mask.sum().clamp(min=1.0)
        op_acc = (op_logits.argmax(dim=-1) == trace_op_ids).float().mean()
        value_acc = (
            (value_logits.argmax(dim=-1) == value_targets).float() * value_mask
        ).sum() / value_mask.sum().clamp(min=1.0)
        return {
            "loss": op_loss + value_loss,
            "reasoning_struct_op_loss": op_loss,
            "reasoning_struct_value_loss": value_loss,
            "reasoning_struct_op_acc": op_acc,
            "reasoning_struct_value_acc": value_acc,
        }

    def structured_answer_loss(
        self, slots: torch.Tensor, answer_value_id: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        logits = self.structured_answer_head(slots.mean(dim=1))
        loss = F.cross_entropy(logits, answer_value_id)
        acc = (logits.argmax(dim=-1) == answer_value_id).float().mean()
        return {
            "loss": loss,
            "structured_answer_loss": loss,
            "structured_answer_acc": acc,
        }

    def answer_value_loss(
        self, slots: torch.Tensor, answer_value_id: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        logits = self.answer_value_head(slots.mean(dim=1))
        loss = F.cross_entropy(logits, answer_value_id)
        acc = (logits.argmax(dim=-1) == answer_value_id).float().mean()
        return {
            "loss": loss,
            "answer_value_loss": loss,
            "answer_value_acc": acc,
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
        trace_ids: torch.Tensor | None = None,
        trace_len: torch.Tensor | None = None,
        trace_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        target_slots = self.encode_target(answer_ids, use_ema=False)
        out = self.readout_loss(target_slots, answer_ids, answer_len)
        result = {
            "loss": out["loss"],
            "token_loss": out["token_loss"],
            "length_loss": out["length_loss"],
            "target_slots": target_slots.detach(),
        }
        if self.use_reasoning_trace and trace_ids is not None and trace_len is not None:
            trace_slots = self.encode_trace_target(trace_ids, use_ema=False)
            trace_out = self.trace_readout_loss(trace_slots, trace_ids, trace_len)
            result["loss"] = result["loss"] + trace_weight * trace_out["loss"]
            result["trace_token_loss"] = trace_out["trace_token_loss"]
            result["trace_length_loss"] = trace_out["trace_length_loss"]
            result["trace_target_slots"] = trace_slots.detach()
        return result

    def stage1_predictor(
        self,
        problem_ids: torch.Tensor,
        answer_ids: torch.Tensor,
        math_ids: torch.Tensor | None = None,
        trace_ids: torch.Tensor | None = None,
        trace_len: torch.Tensor | None = None,
        trace_op_ids: torch.Tensor | None = None,
        trace_value_ids: torch.Tensor | None = None,
        trace_value_mask: torch.Tensor | None = None,
        answer_value_id: torch.Tensor | None = None,
        trace_weight: float = 0.5,
        trace_struct_weight: float = 1.0,
        reasoning_struct_weight: float = 1.0,
        structured_answer_weight: float = 1.0,
        answer_value_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        vicreg_weight: float = 0.05,
        slot_diversity_weight: float = 0.1,
        batch_diversity_weight: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        raw_pred_slots = self.predict_slots(problem_ids, math_ids)
        trace_pred_slots = None
        pred_slots = raw_pred_slots
        if self.use_trace_fusion:
            trace_pred_slots = self.predict_trace_slots(problem_ids, math_ids)
            pred_slots = self.fuse_answer_trace_slots(raw_pred_slots, trace_pred_slots)
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
        result = {
            "loss": loss,
            "pred_loss": pred_loss,
            "contrastive_loss": contrastive_loss,
            "vicreg_loss": vicreg,
            "slot_diversity_loss": diversity_loss,
            "batch_diversity_loss": batch_diversity,
            "pred_slots": pred_slots.detach(),
            "raw_pred_slots": raw_pred_slots.detach(),
            "target_slots": target_slots.detach(),
        }
        if (
            trace_op_ids is not None
            and trace_value_ids is not None
            and trace_value_mask is not None
        ):
            reasoning_struct = self.reasoning_struct_loss(
                pred_slots,
                trace_op_ids,
                trace_value_ids,
                trace_value_mask,
            )
            result["loss"] = result["loss"] + reasoning_struct_weight * reasoning_struct["loss"]
            result["reasoning_struct_op_loss"] = reasoning_struct["reasoning_struct_op_loss"]
            result["reasoning_struct_value_loss"] = reasoning_struct[
                "reasoning_struct_value_loss"
            ]
            result["reasoning_struct_op_acc"] = reasoning_struct["reasoning_struct_op_acc"]
            result["reasoning_struct_value_acc"] = reasoning_struct[
                "reasoning_struct_value_acc"
            ]
        if answer_value_id is not None:
            answer_value = self.answer_value_loss(pred_slots, answer_value_id)
            result["loss"] = result["loss"] + answer_value_weight * answer_value["loss"]
            result["answer_value_loss"] = answer_value["answer_value_loss"]
            result["answer_value_acc"] = answer_value["answer_value_acc"]
        if self.use_reasoning_trace and trace_ids is not None:
            if trace_pred_slots is None:
                trace_pred_slots = self.predict_trace_slots(problem_ids, math_ids)
            with torch.no_grad():
                trace_target_slots = self.encode_trace_target(trace_ids, use_ema=True)
            trace_pred_loss = F.smooth_l1_loss(trace_pred_slots, trace_target_slots)
            trace_contrastive_loss = self.info_nce_loss(trace_pred_slots, trace_target_slots)
            trace_aux_loss = trace_pred_loss + contrastive_weight * trace_contrastive_loss
            if trace_len is not None:
                trace_dec = self.trace_readout_loss(trace_pred_slots, trace_ids, trace_len)
                trace_aux_loss = trace_aux_loss + 0.1 * trace_dec["loss"]
                result["trace_token_loss"] = trace_dec["trace_token_loss"]
                result["trace_length_loss"] = trace_dec["trace_length_loss"]
            if (
                trace_op_ids is not None
                and trace_value_ids is not None
                and trace_value_mask is not None
            ):
                trace_struct = self.structured_trace_loss(
                    trace_pred_slots,
                    trace_op_ids,
                    trace_value_ids,
                    trace_value_mask,
                )
                trace_aux_loss = trace_aux_loss + trace_struct_weight * trace_struct["loss"]
                result["trace_struct_op_loss"] = trace_struct["trace_struct_op_loss"]
                result["trace_struct_value_loss"] = trace_struct["trace_struct_value_loss"]
                result["trace_struct_op_acc"] = trace_struct["trace_struct_op_acc"]
                result["trace_struct_value_acc"] = trace_struct["trace_struct_value_acc"]
            if answer_value_id is not None:
                answer_struct = self.structured_answer_loss(trace_pred_slots, answer_value_id)
                trace_aux_loss = trace_aux_loss + structured_answer_weight * answer_struct["loss"]
                result["structured_answer_loss"] = answer_struct["structured_answer_loss"]
                result["structured_answer_acc"] = answer_struct["structured_answer_acc"]
            result["loss"] = result["loss"] + trace_weight * trace_aux_loss
            result["trace_pred_loss"] = trace_pred_loss
            result["trace_contrastive_loss"] = trace_contrastive_loss
            result["trace_pred_slots"] = trace_pred_slots.detach()
            result["trace_target_slots"] = trace_target_slots.detach()
        return result

    def stage2_readout(
        self,
        problem_ids: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_len: torch.Tensor,
        math_ids: torch.Tensor | None = None,
        trace_ids: torch.Tensor | None = None,
        trace_len: torch.Tensor | None = None,
        trace_op_ids: torch.Tensor | None = None,
        trace_value_ids: torch.Tensor | None = None,
        trace_value_mask: torch.Tensor | None = None,
        answer_value_id: torch.Tensor | None = None,
        trace_weight: float = 0.5,
        trace_struct_weight: float = 1.0,
        reasoning_struct_weight: float = 1.0,
        structured_answer_weight: float = 1.0,
        true_latent_ratio: float = 0.5,
        target_readout_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            pred_slots = self.predict_answer_slots(problem_ids, math_ids)
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

    def stage3_joint(
        self,
        problem_ids: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_len: torch.Tensor,
        math_ids: torch.Tensor | None = None,
        trace_ids: torch.Tensor | None = None,
        trace_len: torch.Tensor | None = None,
        trace_op_ids: torch.Tensor | None = None,
        trace_value_ids: torch.Tensor | None = None,
        trace_value_mask: torch.Tensor | None = None,
        answer_value_id: torch.Tensor | None = None,
        pred_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        vicreg_weight: float = 0.05,
        token_weight: float = 0.5,
        length_weight: float = 0.1,
        trace_weight: float = 0.5,
        trace_struct_weight: float = 1.0,
        reasoning_struct_weight: float = 1.0,
        structured_answer_weight: float = 1.0,
        answer_value_weight: float = 1.0,
        slot_diversity_weight: float = 0.1,
        batch_diversity_weight: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        pred_out = self.stage1_predictor(
            problem_ids,
            answer_ids,
            math_ids=math_ids,
            trace_ids=trace_ids,
            trace_len=trace_len,
            trace_op_ids=trace_op_ids,
            trace_value_ids=trace_value_ids,
            trace_value_mask=trace_value_mask,
            answer_value_id=answer_value_id,
            trace_weight=trace_weight,
            trace_struct_weight=trace_struct_weight,
            reasoning_struct_weight=reasoning_struct_weight,
            structured_answer_weight=structured_answer_weight,
            answer_value_weight=answer_value_weight,
            contrastive_weight=contrastive_weight,
            vicreg_weight=vicreg_weight,
            slot_diversity_weight=slot_diversity_weight,
            batch_diversity_weight=batch_diversity_weight,
        )
        slots = self.predict_answer_slots(problem_ids, math_ids)
        readout = self.readout(slots)
        token_loss = F.cross_entropy(
            readout["token_logits"].transpose(1, 2),
            answer_ids,
            ignore_index=0,
        )
        length_loss = F.cross_entropy(
            readout["length_logits"], answer_len.clamp(max=answer_ids.size(1) - 1)
        )
        latent_loss = (
            pred_weight * pred_out["pred_loss"]
            + contrastive_weight * pred_out["contrastive_loss"]
            + vicreg_weight * pred_out["vicreg_loss"]
            + slot_diversity_weight * pred_out["slot_diversity_loss"]
            + batch_diversity_weight * pred_out["batch_diversity_loss"]
        )
        aux_loss = pred_out["loss"] - (
            pred_out["pred_loss"]
            + contrastive_weight * pred_out["contrastive_loss"]
            + vicreg_weight * pred_out["vicreg_loss"]
            + slot_diversity_weight * pred_out["slot_diversity_loss"]
            + batch_diversity_weight * pred_out["batch_diversity_loss"]
        )
        loss = latent_loss + aux_loss + token_weight * token_loss + length_weight * length_loss
        result = {
            "loss": loss,
            "pred_loss": pred_out["pred_loss"],
            "contrastive_loss": pred_out["contrastive_loss"],
            "vicreg_loss": pred_out["vicreg_loss"],
            "slot_diversity_loss": pred_out["slot_diversity_loss"],
            "batch_diversity_loss": pred_out["batch_diversity_loss"],
            "token_loss": token_loss,
            "length_loss": length_loss,
            "pred_slots": slots.detach(),
        }
        for key in [
            "trace_pred_loss",
            "trace_contrastive_loss",
            "trace_token_loss",
            "trace_length_loss",
            "trace_pred_slots",
            "trace_struct_op_loss",
            "trace_struct_value_loss",
            "trace_struct_op_acc",
            "trace_struct_value_acc",
            "reasoning_struct_op_loss",
            "reasoning_struct_value_loss",
            "reasoning_struct_op_acc",
            "reasoning_struct_value_acc",
            "structured_answer_loss",
            "structured_answer_acc",
            "answer_value_loss",
            "answer_value_acc",
        ]:
            if key in pred_out:
                result[key] = pred_out[key]
        return result

    def forward(
        self,
        problem_ids: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_len: torch.Tensor,
        math_ids: torch.Tensor | None = None,
        trace_ids: torch.Tensor | None = None,
        trace_len: torch.Tensor | None = None,
        trace_op_ids: torch.Tensor | None = None,
        trace_value_ids: torch.Tensor | None = None,
        trace_value_mask: torch.Tensor | None = None,
        answer_value_id: torch.Tensor | None = None,
        trace_weight: float = 0.5,
        trace_struct_weight: float = 1.0,
        structured_answer_weight: float = 1.0,
        answer_value_weight: float = 1.0,
    ):
        pred_out = self.stage1_predictor(
            problem_ids,
            answer_ids,
            math_ids=math_ids,
            trace_ids=trace_ids,
            trace_len=trace_len,
            trace_op_ids=trace_op_ids,
            trace_value_ids=trace_value_ids,
            trace_value_mask=trace_value_mask,
            answer_value_id=answer_value_id,
            trace_weight=trace_weight,
            trace_struct_weight=trace_struct_weight,
            structured_answer_weight=structured_answer_weight,
            answer_value_weight=answer_value_weight,
        )
        dec_out = self.readout_loss(pred_out["pred_slots"], answer_ids, answer_len)
        total_loss = pred_out["loss"] + dec_out["loss"]
        result = {
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
        for key in [
            "trace_pred_loss",
            "trace_contrastive_loss",
            "trace_token_loss",
            "trace_length_loss",
            "trace_pred_slots",
            "trace_struct_op_loss",
            "trace_struct_value_loss",
            "trace_struct_op_acc",
            "trace_struct_value_acc",
            "reasoning_struct_op_loss",
            "reasoning_struct_value_loss",
            "reasoning_struct_op_acc",
            "reasoning_struct_value_acc",
            "structured_answer_loss",
            "structured_answer_acc",
            "answer_value_loss",
            "answer_value_acc",
        ]:
            if key in pred_out:
                result[key] = pred_out[key]
        return result

    @torch.no_grad()
    def predict_structured_trace(
        self, problem_ids: torch.Tensor, math_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots = self.predict_trace_slots(problem_ids, math_ids)
        pred = self.trace_struct_head(slots.mean(dim=1))
        op_ids = pred[:, :8].view(-1, 2, 4).argmax(dim=-1)
        value_ids = pred[:, 8:].view(-1, 6, TRACE_VALUE_CLASSES).argmax(dim=-1)
        return op_ids, value_ids

    @torch.no_grad()
    def predict_reasoning_struct(
        self, problem_ids: torch.Tensor, math_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots = self.predict_answer_slots(problem_ids, math_ids)
        pred = self.reasoning_struct_head(slots.mean(dim=1))
        op_ids = pred[:, :8].view(-1, 2, 4).argmax(dim=-1)
        value_ids = pred[:, 8:].view(-1, 2, TRACE_VALUE_CLASSES).argmax(dim=-1)
        return op_ids, value_ids

    @torch.no_grad()
    def predict_structured_answer(
        self, problem_ids: torch.Tensor, math_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trace_slots = self.predict_trace_slots(problem_ids, math_ids)
        logits = self.structured_answer_head(trace_slots.mean(dim=1))
        return logits.argmax(dim=-1), logits.softmax(dim=-1).max(dim=-1).values

    @torch.no_grad()
    def predict_answer_value(
        self, problem_ids: torch.Tensor, math_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots = self.predict_answer_slots(problem_ids, math_ids)
        logits = self.answer_value_head(slots.mean(dim=1))
        return logits.argmax(dim=-1), logits.softmax(dim=-1).max(dim=-1).values

    @torch.no_grad()
    def solve_ids(
        self, problem_ids: torch.Tensor, pad_id: int, math_ids: torch.Tensor | None = None
    ) -> list[list[int]]:
        slots = self.predict_answer_slots(problem_ids, math_ids)
        return self.readout.decode_ids(slots, pad_id=pad_id)

    @torch.no_grad()
    def solve_trace_ids(
        self, problem_ids: torch.Tensor, pad_id: int, math_ids: torch.Tensor | None = None
    ) -> list[list[int]]:
        slots = self.predict_trace_slots(problem_ids, math_ids)
        return self.trace_readout.decode_ids(slots, pad_id=pad_id)

    @torch.no_grad()
    def solve_ids_from_target(
        self,
        answer_ids: torch.Tensor,
        pad_id: int,
        use_ema: bool = True,
    ) -> list[list[int]]:
        slots = self.encode_target(answer_ids, use_ema=use_ema)
        return self.readout.decode_ids(slots, pad_id=pad_id)
