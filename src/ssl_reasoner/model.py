from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

TRACE_VALUE_CLASSES = 1401
TRACE_VALUE_MIN = -200
VALUE_CONDITIONED_MIN = -1000
VALUE_CONDITIONED_MAX = 1200
VALUE_CONDITIONED_CLASSES = VALUE_CONDITIONED_MAX - VALUE_CONDITIONED_MIN + 1
TRACE_STATE_SCALE = 100.0
VARIABLE_REASONING_SCALE = 100.0
RAW_VARIABLE_VALUE_LOSS_WEIGHT = 0.01
MAX_VARIABLE_REASONING_STEPS = 16
TRANSITION_DIGITS = 5
TRANSITION_VALUE_MIN = -1000
TRANSITION_VALUE_MAX = 10000
TRANSITION_VALUE_CLASSES = TRANSITION_VALUE_MAX - TRANSITION_VALUE_MIN + 1
MATH_FEATURE_NUM_OFFSET = 1
MATH_FEATURE_PLUS_ID = 202
MATH_FEATURE_MINUS_ID = 203
MATH_FEATURE_TIMES_ID = 204
TRACE_ID_TO_OP = ("", "+", "-", "*")


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


class AutoregressiveReadoutDecoder(nn.Module):
    """Scratch latent-slot to token decoder with teacher forcing."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_len: int,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
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

    def forward(self, slots: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        x = self.token_embed(input_ids) + self.pos_embed[:, :seq_len]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        h = self.decoder(tgt=x, memory=slots, tgt_mask=causal_mask)
        return self.lm_head(h)

    @torch.no_grad()
    def decode_ids(
        self,
        slots: torch.Tensor,
        bos_id: int,
        eos_id: int,
        pad_id: int,
    ) -> list[list[int]]:
        batch = slots.size(0)
        ids = torch.full((batch, 1), bos_id, device=slots.device, dtype=torch.long)
        finished = torch.zeros(batch, device=slots.device, dtype=torch.bool)
        for _ in range(self.max_len - 1):
            logits = self.forward(slots, ids)
            next_ids = logits[:, -1].argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, pad_id), next_ids)
            ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)
            finished = finished | next_ids.eq(eos_id)
            if finished.all():
                break
        if ids.size(1) < self.max_len:
            pad = torch.full(
                (batch, self.max_len - ids.size(1)),
                pad_id,
                device=slots.device,
                dtype=torch.long,
            )
            ids = torch.cat([ids, pad], dim=1)
        return ids.tolist()


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


class StepwiseArithmeticStateHead(nn.Module):
    def __init__(self, math_vocab_size: int, d_model: int, max_math_len: int):
        super().__init__()
        self.token_embed = nn.Embedding(math_vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, max_math_len, d_model) * 0.02)
        self.summary = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.op1_head = nn.Linear(d_model, 3)
        self.op2_head = nn.Linear(d_model, 3)
        self.order_head = nn.Linear(d_model, 2)

    @staticmethod
    def _numbers(math_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = (math_ids.float() - MATH_FEATURE_NUM_OFFSET).clamp(min=0.0)
        a = values[:, 0]
        b = values[:, 2]
        c = values[:, 4]
        return a, b, c

    @staticmethod
    def _apply_op(lhs: torch.Tensor, op_probs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        candidates = torch.stack([lhs + rhs, lhs - rhs, lhs * rhs], dim=-1)
        return (candidates * op_probs).sum(dim=-1)

    @staticmethod
    def _math_op_targets(math_ids: torch.Tensor) -> torch.Tensor:
        op1 = (math_ids[:, 1] - MATH_FEATURE_PLUS_ID).clamp(min=0, max=2)
        op2 = (math_ids[:, 3] - MATH_FEATURE_PLUS_ID).clamp(min=0, max=2)
        return torch.stack([op1, op2], dim=-1)

    @staticmethod
    def _order_targets(trace_op_ids: torch.Tensor) -> torch.Tensor:
        return (trace_op_ids[:, 0] == 3).long()

    def forward(self, math_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = math_ids.ne(0)
        tokens = self.token_embed(math_ids) + self.pos_embed[:, : math_ids.size(1)]
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(tokens.dtype)
        pooled = (tokens * mask.unsqueeze(-1)).sum(dim=1) / denom
        hidden = self.summary(pooled)

        op1_logits = self.op1_head(hidden)
        op2_logits = self.op2_head(hidden)
        order_logits = self.order_head(hidden)
        op_targets = self._math_op_targets(math_ids)
        op1_probs = F.one_hot(op_targets[:, 0], num_classes=3).to(hidden.dtype)
        op2_probs = F.one_hot(op_targets[:, 1], num_classes=3).to(hidden.dtype)
        order_probs = order_logits.softmax(dim=-1)

        a, b, c = self._numbers(math_ids)
        left_first = self._apply_op(a, op1_probs, b)
        left_final = self._apply_op(left_first, op2_probs, c)
        right_first = self._apply_op(b, op2_probs, c)
        right_final = self._apply_op(a, op1_probs, right_first)
        mixed = math_ids[:, 3].ne(0).to(left_first.dtype)
        first = torch.where(
            mixed.bool(),
            order_probs[:, 0] * left_first + order_probs[:, 1] * right_first,
            left_first,
        )
        final = torch.where(
            mixed.bool(),
            order_probs[:, 0] * left_final + order_probs[:, 1] * right_final,
            left_first,
        )
        values = torch.stack([first, final], dim=-1)
        return {
            "values": values,
            "op1_logits": op1_logits,
            "op2_logits": op2_logits,
            "order_logits": order_logits,
        }

    def loss(
        self,
        math_ids: torch.Tensor,
        trace_state_values: torch.Tensor,
        trace_state_mask: torch.Tensor,
        trace_op_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self(math_ids)
        values = out["values"]
        value_loss = F.smooth_l1_loss(
            values / TRACE_STATE_SCALE,
            trace_state_values / TRACE_STATE_SCALE,
            reduction="none",
        )
        value_loss = (value_loss * trace_state_mask).sum() / trace_state_mask.sum().clamp(min=1.0)
        op_targets = self._math_op_targets(math_ids)
        op_loss = F.cross_entropy(out["op1_logits"], op_targets[:, 0])
        has_second = trace_state_mask[:, 1].bool()
        if has_second.any():
            op_loss = op_loss + F.cross_entropy(
                out["op2_logits"][has_second],
                op_targets[:, 1][has_second],
            )
            order_loss = F.cross_entropy(
                out["order_logits"][has_second],
                self._order_targets(trace_op_ids)[has_second],
            )
        else:
            order_loss = value_loss.new_tensor(0.0)
        final_target = torch.where(
            trace_state_mask[:, 1].bool(),
            trace_state_values[:, 1],
            trace_state_values[:, 0],
        )
        final_pred = torch.where(trace_state_mask[:, 1].bool(), values[:, 1], values[:, 0])
        final_acc = (final_pred.round() == final_target).float().mean()
        return {
            "loss": value_loss + op_loss + order_loss,
            "step_state_value_loss": value_loss,
            "step_state_op_loss": op_loss,
            "step_state_order_loss": order_loss,
            "step_state_final_acc": final_acc,
        }


class VariableStructuredReasoner(nn.Module):
    def __init__(
        self,
        math_vocab_size: int,
        d_model: int,
        max_math_len: int,
        max_steps: int = MAX_VARIABLE_REASONING_STEPS,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.token_embed = nn.Embedding(math_vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, max_math_len, d_model) * 0.02)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(d_model + max_math_len),
            nn.Linear(d_model + max_math_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.step_embed = nn.Parameter(torch.randn(max_steps, d_model) * 0.02)
        self.cell = nn.GRUCell(d_model, d_model)
        self.active_head = nn.Linear(d_model, 1)
        self.op_head = nn.Linear(d_model, 4)
        self.reduction_policy_head = nn.Linear(d_model, 2)
        self.position_query = nn.Linear(d_model, d_model)
        self.position_key = nn.Linear(d_model + 5, d_model)
        self.result_head = nn.Sequential(
            nn.LayerNorm(d_model + 5),
            nn.Linear(d_model + 5, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.raw_result_head = nn.Sequential(
            nn.LayerNorm(d_model + 5),
            nn.Linear(d_model + 5, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.digit_result_head = nn.Sequential(
            nn.LayerNorm(d_model + 5),
            nn.Linear(d_model + 5, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 + TRANSITION_DIGITS * 10),
        )
        self.class_result_head = nn.Sequential(
            nn.LayerNorm(d_model + 5),
            nn.Linear(d_model + 5, d_model),
            nn.GELU(),
            nn.Linear(d_model, TRANSITION_VALUE_CLASSES),
        )

    def pointer_logits_for_ops(
        self,
        hidden: torch.Tensor,
        op_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        op_embeddings = self.token_embed(op_ids)
        op_values = (op_ids - MATH_FEATURE_PLUS_ID).clamp(min=0, max=2)
        op_features = torch.stack(
            [
                op_ids.ne(0).float(),
                op_ids.eq(MATH_FEATURE_PLUS_ID).float(),
                op_ids.eq(MATH_FEATURE_MINUS_ID).float(),
                op_ids.eq(MATH_FEATURE_TIMES_ID).float(),
                op_values.float() / 2.0,
            ],
            dim=-1,
        ).to(device)
        op_keys = self.position_key(torch.cat([op_embeddings, op_features], dim=-1))
        return (
            self.position_query(hidden).unsqueeze(1) * op_keys
        ).sum(dim=-1) / (op_keys.size(-1) ** 0.5)

    def encode_initial(self, math_ids: torch.Tensor) -> torch.Tensor:
        mask = math_ids.ne(0)
        tokens = self.token_embed(math_ids) + self.pos_embed[:, : math_ids.size(1)]
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(tokens.dtype)
        pooled = (tokens * mask.unsqueeze(-1)).sum(dim=1) / denom
        math_features = math_ids.float() / MATH_FEATURE_TIMES_ID
        return self.input_proj(torch.cat([pooled, math_features], dim=-1))

    def dynamic_position_logits(
        self,
        math_ids: torch.Tensor,
        position_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.encode_initial(math_ids)
        op_rows = math_ids[:, 1::2].detach().cpu().tolist()
        logits = []
        for step in range(self.max_steps):
            step_input = self.step_embed[step].unsqueeze(0).expand(math_ids.size(0), -1)
            hidden = self.cell(step_input, hidden)
            op_tensor = math_ids.new_zeros((math_ids.size(0), self.max_steps))
            for row_idx, row_ops in enumerate(op_rows):
                current_ops = row_ops[: self.max_steps]
                if current_ops:
                    op_tensor[row_idx, : len(current_ops)] = torch.tensor(
                        current_ops,
                        device=math_ids.device,
                        dtype=torch.long,
                    )
            step_logits = self.pointer_logits_for_ops(hidden, op_tensor, math_ids.device)
            step_logits = step_logits.masked_fill(op_tensor.eq(0), -1e4)
            logits.append(step_logits)
            if position_targets is not None:
                target_positions = position_targets[:, step].detach().cpu().tolist()
                for row_idx, target_position in enumerate(target_positions):
                    if 0 <= target_position < len(op_rows[row_idx]):
                        del op_rows[row_idx][target_position]
        return torch.stack(logits, dim=1)

    def result_logits_for_values(
        self,
        hidden: torch.Tensor,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        op_ids: torch.Tensor,
    ) -> torch.Tensor:
        features = self._transition_features(hidden, lhs, rhs, op_ids)
        return self.result_head(features)

    def _transition_features(
        self,
        hidden: torch.Tensor,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        op_ids: torch.Tensor,
    ) -> torch.Tensor:
        op_features = F.one_hot(op_ids.clamp(min=0, max=3), num_classes=4).to(hidden.dtype)
        value_features = torch.stack(
            [
                lhs.to(hidden.dtype) / 1000.0,
                rhs.to(hidden.dtype) / 1000.0,
            ],
            dim=-1,
        )
        # Drop the "none" op feature; active transition rows use +, -, or *.
        return torch.cat([hidden, value_features, op_features[:, 1:]], dim=-1)

    def raw_result_for_values(
        self,
        hidden: torch.Tensor,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        op_ids: torch.Tensor,
    ) -> torch.Tensor:
        features = self._transition_features(hidden, lhs, rhs, op_ids)
        return self.raw_result_head(features).squeeze(-1) * VARIABLE_REASONING_SCALE

    @staticmethod
    def value_to_sign_digits(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rounded = values.round().long()
        sign = rounded.lt(0).long()
        magnitude = rounded.abs().clamp(max=(10 ** TRANSITION_DIGITS) - 1)
        digits = []
        for place in reversed(range(TRANSITION_DIGITS)):
            divisor = 10 ** place
            digits.append((magnitude // divisor) % 10)
        return sign, torch.stack(digits, dim=-1)

    @staticmethod
    def sign_digits_to_value(sign_ids: torch.Tensor, digit_ids: torch.Tensor) -> torch.Tensor:
        multipliers = torch.tensor(
            [10 ** place for place in reversed(range(TRANSITION_DIGITS))],
            device=digit_ids.device,
            dtype=torch.long,
        )
        magnitude = (digit_ids.long() * multipliers).sum(dim=-1)
        return torch.where(sign_ids.long().eq(1), -magnitude, magnitude)

    def digit_logits_for_values(
        self,
        hidden: torch.Tensor,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        op_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.digit_result_head(self._transition_features(hidden, lhs, rhs, op_ids))
        sign_logits = logits[:, :2]
        digit_logits = logits[:, 2:].view(-1, TRANSITION_DIGITS, 10)
        return sign_logits, digit_logits

    def digit_result_for_values(
        self,
        hidden: torch.Tensor,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        op_ids: torch.Tensor,
    ) -> torch.Tensor:
        sign_logits, digit_logits = self.digit_logits_for_values(hidden, lhs, rhs, op_ids)
        return self.sign_digits_to_value(
            sign_logits.argmax(dim=-1),
            digit_logits.argmax(dim=-1),
        )

    @staticmethod
    def value_to_transition_class(values: torch.Tensor) -> torch.Tensor:
        return values.round().long().clamp(
            min=TRANSITION_VALUE_MIN,
            max=TRANSITION_VALUE_MAX,
        ) - TRANSITION_VALUE_MIN

    @staticmethod
    def transition_class_to_value(class_ids: torch.Tensor) -> torch.Tensor:
        return class_ids.long() + TRANSITION_VALUE_MIN

    def class_logits_for_values(
        self,
        hidden: torch.Tensor,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        op_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.class_result_head(self._transition_features(hidden, lhs, rhs, op_ids))

    def class_result_for_values(
        self,
        hidden: torch.Tensor,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        op_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.class_logits_for_values(hidden, lhs, rhs, op_ids)
        return self.transition_class_to_value(logits.argmax(dim=-1))

    @staticmethod
    def arithmetic_candidates(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.stack([lhs + rhs, lhs - rhs, lhs * rhs], dim=-1)

    def forward(self, math_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encode_initial(math_ids)

        states = []
        for step in range(self.max_steps):
            step_input = self.step_embed[step].unsqueeze(0).expand(math_ids.size(0), -1)
            hidden = self.cell(step_input, hidden)
            states.append(hidden)
        step_states = torch.stack(states, dim=1)
        position_logits = self.dynamic_position_logits(math_ids)
        return {
            "active_logits": self.active_head(step_states).squeeze(-1),
            "op_logits": self.op_head(step_states),
            "reduction_policy_logits": self.reduction_policy_head(step_states),
            "position_logits": position_logits,
            "states": step_states,
        }

    @torch.no_grad()
    def initial_hidden(self, math_ids: torch.Tensor) -> torch.Tensor:
        return self.encode_initial(math_ids)

    @torch.no_grad()
    def advance_hidden(self, hidden: torch.Tensor, step_idx: int) -> torch.Tensor:
        step_input = self.step_embed[step_idx].unsqueeze(0).expand(hidden.size(0), -1)
        return self.cell(step_input, hidden)

    def loss(
        self,
        math_ids: torch.Tensor,
        op_targets: torch.Tensor,
        position_targets: torch.Tensor,
        value_targets: torch.Tensor,
        legal_position_targets: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self(math_ids)
        position_logits = self.dynamic_position_logits(math_ids, position_targets)
        active_loss = F.binary_cross_entropy_with_logits(out["active_logits"], step_mask)
        op_loss = F.cross_entropy(
            out["op_logits"].reshape(-1, 4),
            op_targets.reshape(-1),
            reduction="none",
        ).view_as(step_mask)
        op_loss = (op_loss * step_mask).sum() / step_mask.sum().clamp(min=1.0)
        reduction_policy_targets = op_targets.ne(3).long()
        reduction_policy_loss = F.cross_entropy(
            out["reduction_policy_logits"].reshape(-1, 2),
            reduction_policy_targets.reshape(-1),
            reduction="none",
        ).view_as(step_mask)
        reduction_policy_loss = (
            reduction_policy_loss * step_mask
        ).sum() / step_mask.sum().clamp(min=1.0)
        position_loss = F.cross_entropy(
            position_logits.reshape(-1, self.max_steps),
            position_targets.reshape(-1),
            reduction="none",
        ).view_as(step_mask)
        position_loss = (position_loss * step_mask).sum() / step_mask.sum().clamp(min=1.0)
        legal_loss = F.binary_cross_entropy_with_logits(
            position_logits,
            legal_position_targets,
            reduction="none",
        ).mean(dim=-1)
        legal_loss = (legal_loss * step_mask).sum() / step_mask.sum().clamp(min=1.0)
        result_logits = self.result_logits_for_values(
            out["states"].reshape(-1, out["states"].size(-1)),
            value_targets[:, :, 0].reshape(-1),
            value_targets[:, :, 1].reshape(-1),
            op_targets.reshape(-1),
        ).view(math_ids.size(0), self.max_steps, 3)
        candidate_targets = (op_targets - 1).clamp(min=0, max=2)
        value_loss = F.cross_entropy(
            result_logits.reshape(-1, 3),
            candidate_targets.reshape(-1),
            reduction="none",
        ).view_as(step_mask)
        value_loss = (value_loss * step_mask).sum() / step_mask.sum().clamp(min=1.0)
        raw_values = self.raw_result_for_values(
            out["states"].reshape(-1, out["states"].size(-1)),
            value_targets[:, :, 0].reshape(-1),
            value_targets[:, :, 1].reshape(-1),
            op_targets.reshape(-1),
        ).view_as(step_mask)
        raw_value_loss = F.smooth_l1_loss(
            raw_values / VARIABLE_REASONING_SCALE,
            value_targets[:, :, 2] / VARIABLE_REASONING_SCALE,
            reduction="none",
        )
        raw_value_loss = (
            raw_value_loss * step_mask
        ).sum() / step_mask.sum().clamp(min=1.0)
        sign_logits, digit_logits = self.digit_logits_for_values(
            out["states"].reshape(-1, out["states"].size(-1)),
            value_targets[:, :, 0].reshape(-1),
            value_targets[:, :, 1].reshape(-1),
            op_targets.reshape(-1),
        )
        sign_targets, digit_targets = self.value_to_sign_digits(
            value_targets[:, :, 2].reshape(-1)
        )
        flat_mask = step_mask.reshape(-1)
        digit_sign_loss = F.cross_entropy(
            sign_logits,
            sign_targets,
            reduction="none",
        )
        digit_sign_loss = (
            digit_sign_loss * flat_mask
        ).sum() / flat_mask.sum().clamp(min=1.0)
        per_digit_loss = F.cross_entropy(
            digit_logits.reshape(-1, 10),
            digit_targets.reshape(-1),
            reduction="none",
        ).view(-1, TRANSITION_DIGITS)
        digit_value_loss = (
            per_digit_loss.mean(dim=-1) * flat_mask
        ).sum() / flat_mask.sum().clamp(min=1.0)
        class_logits = self.class_logits_for_values(
            out["states"].reshape(-1, out["states"].size(-1)),
            value_targets[:, :, 0].reshape(-1),
            value_targets[:, :, 1].reshape(-1),
            op_targets.reshape(-1),
        )
        class_targets = self.value_to_transition_class(
            value_targets[:, :, 2].reshape(-1)
        )
        class_value_loss = F.cross_entropy(
            class_logits,
            class_targets,
            reduction="none",
        )
        class_value_loss = (
            class_value_loss * flat_mask
        ).sum() / flat_mask.sum().clamp(min=1.0)
        active_pred = out["active_logits"].sigmoid().ge(0.5).float()
        active_acc = (active_pred == step_mask).float().mean()
        op_acc = (
            (out["op_logits"].argmax(dim=-1) == op_targets).float() * step_mask
        ).sum() / step_mask.sum().clamp(min=1.0)
        reduction_policy_acc = (
            (
                out["reduction_policy_logits"].argmax(dim=-1)
                == reduction_policy_targets
            ).float()
            * step_mask
        ).sum() / step_mask.sum().clamp(min=1.0)
        position_acc = (
            (position_logits.argmax(dim=-1) == position_targets).float() * step_mask
        ).sum() / step_mask.sum().clamp(min=1.0)
        legal_pred = position_logits.sigmoid().ge(0.5).float()
        legal_acc = (
            (legal_pred == legal_position_targets).float().mean(dim=-1) * step_mask
        ).sum() / step_mask.sum().clamp(min=1.0)
        candidate_values = self.arithmetic_candidates(
            value_targets[:, :, 0],
            value_targets[:, :, 1],
        )
        value_acc = (
            (
                candidate_values.gather(
                    dim=-1,
                    index=result_logits.argmax(dim=-1, keepdim=True),
                ).squeeze(-1)
                == value_targets[:, :, 2].round()
            ).float()
            * step_mask
        ).sum() / step_mask.sum().clamp(min=1.0)
        raw_value_acc = (
            (raw_values.round() == value_targets[:, :, 2].round()).float() * step_mask
        ).sum() / step_mask.sum().clamp(min=1.0)
        digit_values = self.sign_digits_to_value(
            sign_logits.argmax(dim=-1),
            digit_logits.argmax(dim=-1),
        ).view_as(step_mask)
        digit_value_acc = (
            (digit_values == value_targets[:, :, 2].round().long()).float() * step_mask
        ).sum() / step_mask.sum().clamp(min=1.0)
        class_values = self.transition_class_to_value(
            class_logits.argmax(dim=-1)
        ).view_as(step_mask)
        class_value_acc = (
            (class_values == value_targets[:, :, 2].round().long()).float() * step_mask
        ).sum() / step_mask.sum().clamp(min=1.0)
        return {
            "loss": (
                active_loss
                + op_loss
                + reduction_policy_loss
                + position_loss
                + legal_loss
                + value_loss
                + RAW_VARIABLE_VALUE_LOSS_WEIGHT * raw_value_loss
                + digit_sign_loss
                + digit_value_loss
                + class_value_loss
            ),
            "variable_active_loss": active_loss,
            "variable_op_loss": op_loss,
            "variable_reduction_policy_loss": reduction_policy_loss,
            "variable_position_loss": position_loss,
            "variable_legal_loss": legal_loss,
            "variable_value_loss": value_loss,
            "variable_raw_value_loss": raw_value_loss,
            "variable_digit_sign_loss": digit_sign_loss,
            "variable_digit_value_loss": digit_value_loss,
            "variable_class_value_loss": class_value_loss,
            "variable_active_acc": active_acc,
            "variable_op_acc": op_acc,
            "variable_reduction_policy_acc": reduction_policy_acc,
            "variable_position_acc": position_acc,
            "variable_legal_acc": legal_acc,
            "variable_value_acc": value_acc,
            "variable_raw_value_acc": raw_value_acc,
            "variable_digit_value_acc": digit_value_acc,
            "variable_class_value_acc": class_value_acc,
        }


class StateConditionedSlotProjector(nn.Module):
    def __init__(self, d_model: int, num_slots: int):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.LayerNorm(d_model + 2),
            nn.Linear(d_model + 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_slots * d_model),
        )
        self.slot_norm = nn.LayerNorm(d_model)

    def forward(self, context: torch.Tensor, state_values: torch.Tensor) -> torch.Tensor:
        scaled_state = state_values / TRACE_STATE_SCALE
        flat_slots = self.net(torch.cat([context, scaled_state], dim=-1))
        slots = flat_slots.view(context.size(0), self.num_slots, self.d_model)
        return self.slot_norm(slots)


class ReasoningStateSequenceProjector(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_slots: int,
        max_math_len: int,
        num_states: int = 3,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        self.num_states = num_states
        self.max_math_len = max_math_len
        self.net = nn.Sequential(
            nn.LayerNorm(d_model + 2 + max_math_len),
            nn.Linear(d_model + 2 + max_math_len, d_model * 3),
            nn.GELU(),
            nn.Linear(d_model * 3, num_states * num_slots * d_model),
        )
        self.slot_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        context: torch.Tensor,
        state_values: torch.Tensor,
        math_ids: torch.Tensor,
    ) -> torch.Tensor:
        scaled_state = state_values / TRACE_STATE_SCALE
        math_features = math_ids[:, : self.max_math_len].float() / MATH_FEATURE_TIMES_ID
        flat = self.net(torch.cat([context, scaled_state, math_features], dim=-1))
        slots = flat.view(
            context.size(0),
            self.num_states,
            self.num_slots,
            self.d_model,
        )
        return self.slot_norm(slots)


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
        max_variable_steps: int = MAX_VARIABLE_REASONING_STEPS,
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
        self.step_state_head = StepwiseArithmeticStateHead(
            math_vocab_size, d_model, max_math_len
        )
        self.variable_structured_reasoner = VariableStructuredReasoner(
            math_vocab_size, d_model, max_math_len, max_steps=max_variable_steps
        )
        self.state_conditioned_projector = StateConditionedSlotProjector(
            d_model, num_slots
        )
        self.reasoning_sequence_projector = ReasoningStateSequenceProjector(
            d_model, num_slots, max_math_len
        )
        self.reasoning_sequence_readout = AutoregressiveReadoutDecoder(
            d_model,
            vocab_size,
            max_trace_len,
            num_layers=readout_layers,
            num_heads=num_heads,
        )
        self.reasoning_sequence_struct_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4 + 3 * TRACE_VALUE_CLASSES),
        )
        self.value_conditioned_slots = nn.Embedding(
            VALUE_CONDITIONED_CLASSES, num_slots * d_model
        )
        self.value_conditioned_norm = nn.LayerNorm(d_model)
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
            self.trace_state_regression_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2),
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

    def predict_state_conditioned_slots(
        self, problem_ids: torch.Tensor, math_ids: torch.Tensor
    ) -> torch.Tensor:
        context = self.encode_context(problem_ids)
        state_values = self.step_state_head(math_ids)["values"]
        return self.state_conditioned_projector(context, state_values)

    def predict_reasoning_state_slots(
        self, problem_ids: torch.Tensor, math_ids: torch.Tensor
    ) -> torch.Tensor:
        context = self.encode_context(problem_ids)
        state_values = self.step_state_head(math_ids)["values"]
        return self.reasoning_sequence_projector(context, state_values, math_ids)

    def predict_value_conditioned_slots_from_ids(
        self, answer_value_id: torch.Tensor
    ) -> torch.Tensor:
        value_conditioned_id = (
            answer_value_id + TRACE_VALUE_MIN - VALUE_CONDITIONED_MIN
        ).clamp(min=0, max=VALUE_CONDITIONED_CLASSES - 1)
        return self.predict_value_conditioned_slots_from_value_ids(value_conditioned_id)

    def predict_value_conditioned_slots_from_value_ids(
        self, value_conditioned_id: torch.Tensor
    ) -> torch.Tensor:
        flat = self.value_conditioned_slots(value_conditioned_id)
        slots = flat.view(
            value_conditioned_id.size(0),
            self.target_encoder.slot_queries.size(1),
            -1,
        )
        return self.value_conditioned_norm(slots)

    def predict_value_conditioned_slots(self, math_ids: torch.Tensor) -> torch.Tensor:
        state_values = self.step_state_head(math_ids)["values"]
        final_values = torch.where(
            math_ids[:, 3].ne(0),
            state_values[:, 1],
            state_values[:, 0],
        )
        value_ids = (final_values.round().long() - VALUE_CONDITIONED_MIN).clamp(
            min=0, max=VALUE_CONDITIONED_CLASSES - 1
        )
        return self.predict_value_conditioned_slots_from_value_ids(value_ids)

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

    def trace_state_regression_loss(
        self,
        slots: torch.Tensor,
        trace_state_values: torch.Tensor,
        trace_state_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred = self.trace_state_regression_head(slots.mean(dim=1))
        target = trace_state_values / TRACE_STATE_SCALE
        per_value = F.smooth_l1_loss(pred, target, reduction="none")
        loss = (per_value * trace_state_mask).sum() / trace_state_mask.sum().clamp(min=1.0)
        rounded = (pred * TRACE_STATE_SCALE).round()
        acc = (
            (rounded == trace_state_values).float() * trace_state_mask
        ).sum() / trace_state_mask.sum().clamp(min=1.0)
        final_target = torch.where(
            trace_state_mask[:, 1].bool(),
            trace_state_values[:, 1],
            trace_state_values[:, 0],
        )
        final_pred = torch.where(trace_state_mask[:, 1].bool(), rounded[:, 1], rounded[:, 0])
        final_acc = (final_pred == final_target).float().mean()
        return {
            "loss": loss,
            "trace_state_regression_loss": loss,
            "trace_state_regression_acc": acc,
            "trace_state_final_acc": final_acc,
        }

    def step_state_loss(
        self,
        math_ids: torch.Tensor,
        trace_state_values: torch.Tensor,
        trace_state_mask: torch.Tensor,
        trace_op_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.step_state_head.loss(
            math_ids,
            trace_state_values,
            trace_state_mask,
            trace_op_ids,
        )

    def variable_reasoning_loss(
        self,
        math_ids: torch.Tensor,
        variable_trace_op_ids: torch.Tensor,
        variable_trace_position_ids: torch.Tensor,
        variable_trace_values: torch.Tensor,
        variable_trace_legal_mask: torch.Tensor,
        variable_trace_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.variable_structured_reasoner.loss(
            math_ids,
            variable_trace_op_ids,
            variable_trace_position_ids,
            variable_trace_values,
            variable_trace_legal_mask,
            variable_trace_mask,
        )

    def state_conditioned_latent_loss(
        self,
        problem_ids: torch.Tensor,
        math_ids: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_len: torch.Tensor | None = None,
        answer_value_id: torch.Tensor | None = None,
        readout_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        pred_slots = self.predict_state_conditioned_slots(problem_ids, math_ids)
        with torch.no_grad():
            target_slots = self.encode_target(answer_ids, use_ema=True)
        pred_loss = F.smooth_l1_loss(pred_slots, target_slots)
        contrastive_loss = self.info_nce_loss(pred_slots, target_slots)
        loss = pred_loss + 0.2 * contrastive_loss
        result = {
            "loss": loss,
            "state_conditioned_pred_loss": pred_loss,
            "state_conditioned_contrastive_loss": contrastive_loss,
            "state_conditioned_slots": pred_slots.detach(),
            "target_slots": target_slots.detach(),
        }
        if answer_len is not None and readout_weight > 0:
            readout = self.readout_loss(pred_slots, answer_ids, answer_len)
            result["loss"] = result["loss"] + readout_weight * readout["loss"]
            result["state_conditioned_token_loss"] = readout["token_loss"]
            result["state_conditioned_length_loss"] = readout["length_loss"]
        if answer_value_id is not None:
            answer_value = self.answer_value_loss(pred_slots, answer_value_id)
            result["loss"] = result["loss"] + answer_value["loss"]
            result["answer_value_loss"] = answer_value["answer_value_loss"]
            result["answer_value_acc"] = answer_value["answer_value_acc"]
        return result

    def value_conditioned_latent_loss(
        self,
        answer_value_id: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_len: torch.Tensor | None = None,
        readout_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        pred_slots = self.predict_value_conditioned_slots_from_ids(answer_value_id)
        with torch.no_grad():
            target_slots = self.encode_target(answer_ids, use_ema=True)
        pred_loss = F.smooth_l1_loss(pred_slots, target_slots)
        contrastive_loss = self.info_nce_loss(pred_slots, target_slots)
        loss = pred_loss + 0.2 * contrastive_loss
        result = {
            "loss": loss,
            "value_conditioned_pred_loss": pred_loss,
            "value_conditioned_contrastive_loss": contrastive_loss,
            "value_conditioned_slots": pred_slots.detach(),
            "target_slots": target_slots.detach(),
        }
        if answer_len is not None and readout_weight > 0:
            readout = self.readout_loss(pred_slots, answer_ids, answer_len)
            result["loss"] = result["loss"] + readout_weight * readout["loss"]
            result["value_conditioned_token_loss"] = readout["token_loss"]
            result["value_conditioned_length_loss"] = readout["length_loss"]
        return result

    def reasoning_sequence_loss(
        self,
        problem_ids: torch.Tensor,
        math_ids: torch.Tensor,
        reasoning_step_ids: torch.Tensor,
        reasoning_step_mask: torch.Tensor,
        reasoning_ids: torch.Tensor,
        reasoning_len: torch.Tensor,
        trace_op_ids: torch.Tensor | None = None,
        trace_value_ids: torch.Tensor | None = None,
        trace_value_mask: torch.Tensor | None = None,
        answer_value_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_slots = self.predict_reasoning_state_slots(problem_ids, math_ids)
        batch, num_states, num_slots, d_model = pred_slots.shape
        flat_pred = pred_slots.reshape(batch * num_states, num_slots, d_model)
        flat_ids = reasoning_step_ids.reshape(batch * num_states, -1)
        flat_mask = reasoning_step_mask.reshape(batch * num_states)
        with torch.no_grad():
            target_slots = self.encode_target(flat_ids, use_ema=True)
        per_state = F.smooth_l1_loss(flat_pred, target_slots, reduction="none").mean(dim=(1, 2))
        latent_loss = (per_state * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)
        sequence_slots = pred_slots.reshape(batch, num_states * num_slots, d_model)
        readout = self.reasoning_sequence_readout(sequence_slots, reasoning_ids[:, :-1])
        token_loss = F.cross_entropy(
            readout.transpose(1, 2),
            reasoning_ids[:, 1:],
            ignore_index=0,
        )
        loss = latent_loss + token_loss
        result = {
            "loss": loss,
            "reasoning_state_latent_loss": latent_loss,
            "reasoning_sequence_token_loss": token_loss,
            "reasoning_sequence_length_loss": token_loss.new_tensor(0.0),
            "reasoning_state_slots": pred_slots.detach(),
        }
        if (
            trace_op_ids is not None
            and trace_value_ids is not None
            and trace_value_mask is not None
            and answer_value_id is not None
        ):
            struct = self.reasoning_sequence_struct_loss(
                pred_slots,
                trace_op_ids,
                trace_value_ids,
                trace_value_mask,
                answer_value_id,
            )
            result["loss"] = result["loss"] + 0.5 * struct["loss"]
            result["reasoning_sequence_struct_op_acc"] = struct[
                "reasoning_sequence_struct_op_acc"
            ]
            result["reasoning_sequence_struct_value_acc"] = struct[
                "reasoning_sequence_struct_value_acc"
            ]
            result["reasoning_sequence_struct_loss"] = struct["loss"]
        return result

    def reasoning_sequence_struct_loss(
        self,
        pred_slots: torch.Tensor,
        trace_op_ids: torch.Tensor,
        trace_value_ids: torch.Tensor,
        trace_value_mask: torch.Tensor,
        answer_value_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, num_states, _, d_model = pred_slots.shape
        pred = self.reasoning_sequence_struct_head(
            pred_slots.reshape(batch * num_states, -1, d_model).mean(dim=1)
        )
        pred = pred.view(batch, num_states, -1)
        op_logits = pred[:, :, :4]
        value_logits = pred[:, :, 4:].view(batch, num_states, 3, TRACE_VALUE_CLASSES)

        op_targets = trace_op_ids.new_zeros(batch, num_states)
        op_targets[:, :2] = trace_op_ids[:, :2]

        value_targets = trace_value_ids.new_zeros(batch, num_states, 3)
        value_masks = trace_value_mask.new_zeros(batch, num_states, 3)
        value_targets[:, 0] = trace_value_ids[:, :3]
        value_masks[:, 0] = trace_value_mask[:, :3]
        value_targets[:, 1] = trace_value_ids[:, 3:6]
        value_masks[:, 1] = trace_value_mask[:, 3:6]
        value_targets[:, 2, 2] = answer_value_id
        value_masks[:, 2, 2] = 1.0

        op_loss = F.cross_entropy(op_logits.reshape(-1, 4), op_targets.reshape(-1))
        per_value_loss = F.cross_entropy(
            value_logits.reshape(-1, TRACE_VALUE_CLASSES),
            value_targets.reshape(-1),
            reduction="none",
        ).view_as(value_masks)
        value_loss = (per_value_loss * value_masks).sum() / value_masks.sum().clamp(min=1.0)
        op_acc = (op_logits.argmax(dim=-1) == op_targets).float().mean()
        value_acc = (
            (value_logits.argmax(dim=-1) == value_targets).float() * value_masks
        ).sum() / value_masks.sum().clamp(min=1.0)
        return {
            "loss": op_loss + value_loss,
            "reasoning_sequence_struct_op_acc": op_acc,
            "reasoning_sequence_struct_value_acc": value_acc,
        }

    @staticmethod
    def _trace_class_to_value(class_id: int) -> int:
        return int(class_id) + TRACE_VALUE_MIN

    @staticmethod
    def _trace_op_to_text(op_id: int) -> str:
        if 0 <= int(op_id) < len(TRACE_ID_TO_OP):
            return TRACE_ID_TO_OP[int(op_id)]
        return ""

    @classmethod
    def _render_reasoning_struct_row(
        cls,
        op_ids: list[int],
        value_ids: list[list[int]],
        is_mixed: bool,
    ) -> str:
        lhs1 = cls._trace_class_to_value(value_ids[0][0])
        rhs1 = cls._trace_class_to_value(value_ids[0][1])
        result1 = cls._trace_class_to_value(value_ids[0][2])
        final = cls._trace_class_to_value(value_ids[2][2])
        op1 = cls._trace_op_to_text(op_ids[0])
        if not is_mixed:
            return f"{lhs1}{op1}{rhs1}={result1},{final}"

        lhs2 = cls._trace_class_to_value(value_ids[1][0])
        rhs2 = cls._trace_class_to_value(value_ids[1][1])
        result2 = cls._trace_class_to_value(value_ids[1][2])
        op2 = cls._trace_op_to_text(op_ids[1])
        return f"{lhs1}{op1}{rhs1}={result1},{lhs2}{op2}{rhs2}={result2},{final}"

    @staticmethod
    def _math_value(math_id: int) -> int:
        return max(0, int(math_id) - MATH_FEATURE_NUM_OFFSET)

    @staticmethod
    def _math_op_id(math_id: int) -> int:
        if int(math_id) == MATH_FEATURE_PLUS_ID:
            return 1
        if int(math_id) == MATH_FEATURE_MINUS_ID:
            return 2
        if int(math_id) == MATH_FEATURE_TIMES_ID:
            return 3
        return 0

    @classmethod
    def _render_step_state_row(
        cls,
        math_row: list[int],
        state_row: list[float],
        order_id: int,
    ) -> str:
        a = cls._math_value(math_row[0])
        op1_id = cls._math_op_id(math_row[1])
        b = cls._math_value(math_row[2])
        op2_id = cls._math_op_id(math_row[3])
        c = cls._math_value(math_row[4])
        first = int(round(float(state_row[0])))
        final = int(round(float(state_row[1]))) if op2_id else first

        op1 = cls._trace_op_to_text(op1_id)
        if not op2_id:
            return f"{a}{op1}{b}={first},{final}"

        op2 = cls._trace_op_to_text(op2_id)
        if int(order_id) == 1:
            return f"{b}{op2}{c}={first},{a}{op1}{first}={final},{final}"
        return f"{a}{op1}{b}={first},{first}{op2}{c}={final},{final}"

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
    def supervised_answer_contrastive_loss(
        pred_slots: torch.Tensor,
        target_slots: torch.Tensor,
        answer_value_id: torch.Tensor,
        temp: float = 0.07,
    ) -> torch.Tensor:
        pred = F.normalize(pred_slots.flatten(start_dim=1), dim=-1)
        target = F.normalize(target_slots.flatten(start_dim=1), dim=-1)
        logits = pred @ target.T / temp
        positive_mask = answer_value_id[:, None].eq(answer_value_id[None, :])

        pred_log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        pred_loss = -(
            pred_log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1)
            / positive_mask.sum(dim=1).clamp(min=1)
        ).mean()

        target_log_prob = logits.T - torch.logsumexp(logits.T, dim=1, keepdim=True)
        target_loss = -(
            target_log_prob.masked_fill(~positive_mask.T, 0.0).sum(dim=1)
            / positive_mask.T.sum(dim=1).clamp(min=1)
        ).mean()
        return (pred_loss + target_loss) / 2

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
        trace_state_values: torch.Tensor | None = None,
        trace_state_mask: torch.Tensor | None = None,
        answer_value_id: torch.Tensor | None = None,
        trace_weight: float = 0.5,
        trace_struct_weight: float = 1.0,
        reasoning_struct_weight: float = 1.0,
        structured_answer_weight: float = 1.0,
        answer_value_weight: float = 1.0,
        answer_contrastive_weight: float = 0.0,
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
            answer_contrastive = self.supervised_answer_contrastive_loss(
                pred_slots,
                target_slots,
                answer_value_id,
            )
            result["loss"] = (
                result["loss"] + answer_contrastive_weight * answer_contrastive
            )
            result["answer_contrastive_loss"] = answer_contrastive
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
            if trace_state_values is not None and trace_state_mask is not None:
                state_regression = self.trace_state_regression_loss(
                    trace_pred_slots,
                    trace_state_values,
                    trace_state_mask,
                )
                trace_aux_loss = trace_aux_loss + state_regression["loss"]
                result["trace_state_regression_loss"] = state_regression[
                    "trace_state_regression_loss"
                ]
                result["trace_state_regression_acc"] = state_regression[
                    "trace_state_regression_acc"
                ]
                result["trace_state_final_acc"] = state_regression[
                    "trace_state_final_acc"
                ]
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
        trace_state_values: torch.Tensor | None = None,
        trace_state_mask: torch.Tensor | None = None,
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
        trace_state_values: torch.Tensor | None = None,
        trace_state_mask: torch.Tensor | None = None,
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
        answer_contrastive_weight: float = 0.0,
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
            trace_state_values=trace_state_values,
            trace_state_mask=trace_state_mask,
            answer_value_id=answer_value_id,
            trace_weight=trace_weight,
            trace_struct_weight=trace_struct_weight,
            reasoning_struct_weight=reasoning_struct_weight,
            structured_answer_weight=structured_answer_weight,
            answer_value_weight=answer_value_weight,
            answer_contrastive_weight=answer_contrastive_weight,
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
            "trace_state_regression_loss",
            "trace_state_regression_acc",
            "trace_state_final_acc",
            "answer_contrastive_loss",
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
        trace_state_values: torch.Tensor | None = None,
        trace_state_mask: torch.Tensor | None = None,
        answer_value_id: torch.Tensor | None = None,
        trace_weight: float = 0.5,
        trace_struct_weight: float = 1.0,
        structured_answer_weight: float = 1.0,
        answer_value_weight: float = 1.0,
        answer_contrastive_weight: float = 0.0,
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
            trace_state_values=trace_state_values,
            trace_state_mask=trace_state_mask,
            answer_value_id=answer_value_id,
            trace_weight=trace_weight,
            trace_struct_weight=trace_struct_weight,
            structured_answer_weight=structured_answer_weight,
            answer_value_weight=answer_value_weight,
            answer_contrastive_weight=answer_contrastive_weight,
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
            "trace_state_regression_loss",
            "trace_state_regression_acc",
            "trace_state_final_acc",
            "answer_contrastive_loss",
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
    def predict_reasoning_sequence_struct(
        self,
        problem_ids: torch.Tensor,
        math_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots = self.predict_reasoning_state_slots(problem_ids, math_ids)
        batch, num_states, _, d_model = slots.shape
        pred = self.reasoning_sequence_struct_head(
            slots.reshape(batch * num_states, -1, d_model).mean(dim=1)
        )
        pred = pred.view(batch, num_states, -1)
        op_ids = pred[:, :, :4].argmax(dim=-1)
        value_ids = pred[:, :, 4:].view(
            batch, num_states, 3, TRACE_VALUE_CLASSES
        ).argmax(dim=-1)
        return op_ids, value_ids

    @torch.no_grad()
    def solve_reasoning_structured_texts(
        self,
        problem_ids: torch.Tensor,
        math_ids: torch.Tensor,
    ) -> list[str]:
        del problem_ids
        state_out = self.step_state_head(math_ids)
        order_ids = state_out["order_logits"].argmax(dim=-1).detach().cpu().tolist()
        state_rows = state_out["values"].detach().cpu().tolist()
        math_rows = math_ids.detach().cpu().tolist()
        return [
            self._render_step_state_row(math_row, state_row, order_id)
            for math_row, state_row, order_id in zip(math_rows, state_rows, order_ids)
        ]

    @torch.no_grad()
    def solve_reasoning_struct_head_texts(
        self,
        problem_ids: torch.Tensor,
        math_ids: torch.Tensor,
    ) -> list[str]:
        op_ids, value_ids = self.predict_reasoning_sequence_struct(problem_ids, math_ids)
        is_mixed = math_ids[:, 3].ne(0).detach().cpu().tolist()
        op_rows = op_ids.detach().cpu().tolist()
        value_rows = value_ids.detach().cpu().tolist()
        return [
            self._render_reasoning_struct_row(op_row, value_row, bool(mixed))
            for op_row, value_row, mixed in zip(op_rows, value_rows, is_mixed)
        ]

    @torch.no_grad()
    def solve_variable_reasoning_texts(
        self,
        math_ids: torch.Tensor,
        constrain_to_legal: bool = True,
        learned_values: bool = False,
        raw_learned_values: bool = False,
        digit_learned_values: bool = False,
        class_learned_values: bool = False,
    ) -> list[str]:
        math_rows = math_ids.detach().cpu().tolist()
        hidden_rows = self.variable_structured_reasoner.initial_hidden(math_ids)
        traces = []
        for row_idx, math_row in enumerate(math_rows):
            values = [
                self._math_value(math_row[idx])
                for idx in range(0, len(math_row), 2)
                if math_row[idx] != 0 or idx == 0
            ]
            ops = [
                self._math_op_id(math_row[idx])
                for idx in range(1, len(math_row), 2)
                if math_row[idx] != 0 and idx + 1 < len(math_row) and math_row[idx + 1] != 0
            ]
            parts = []
            final = None
            hidden = hidden_rows[row_idx : row_idx + 1]
            for step_idx in range(len(ops)):
                if not ops:
                    break
                hidden = self.variable_structured_reasoner.advance_hidden(hidden, step_idx)
                current_op_ids = torch.tensor(
                    [
                        MATH_FEATURE_PLUS_ID + op_id - 1
                        for op_id in ops
                    ],
                    device=math_ids.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                step_logits = self.variable_structured_reasoner.pointer_logits_for_ops(
                    hidden,
                    current_op_ids,
                    math_ids.device,
                )[0].detach().cpu().tolist()
                if constrain_to_legal:
                    legal_positions = (
                        [idx for idx, current_op in enumerate(ops) if current_op == 3]
                        if 3 in ops
                        else [0]
                    )
                    position = max(legal_positions, key=lambda idx: step_logits[idx])
                else:
                    position = max(range(len(step_logits)), key=lambda idx: step_logits[idx])
                lhs = values[position]
                rhs = values[position + 1]
                op_id = ops[position]
                if op_id == 1:
                    result = lhs + rhs
                elif op_id == 2:
                    result = lhs - rhs
                elif op_id == 3:
                    result = lhs * rhs
                else:
                    break
                if class_learned_values:
                    lhs_tensor = torch.tensor([lhs], device=math_ids.device)
                    rhs_tensor = torch.tensor([rhs], device=math_ids.device)
                    class_result = self.variable_structured_reasoner.class_result_for_values(
                        hidden,
                        lhs_tensor,
                        rhs_tensor,
                        torch.tensor([op_id], device=math_ids.device),
                    )
                    result = int(class_result[0].item())
                elif digit_learned_values:
                    lhs_tensor = torch.tensor([lhs], device=math_ids.device)
                    rhs_tensor = torch.tensor([rhs], device=math_ids.device)
                    digit_result = self.variable_structured_reasoner.digit_result_for_values(
                        hidden,
                        lhs_tensor,
                        rhs_tensor,
                        torch.tensor([op_id], device=math_ids.device),
                    )
                    result = int(digit_result[0].item())
                elif raw_learned_values:
                    lhs_tensor = torch.tensor([lhs], device=math_ids.device)
                    rhs_tensor = torch.tensor([rhs], device=math_ids.device)
                    raw_result = self.variable_structured_reasoner.raw_result_for_values(
                        hidden,
                        lhs_tensor,
                        rhs_tensor,
                        torch.tensor([op_id], device=math_ids.device),
                    )
                    result = int(round(float(raw_result[0].item())))
                elif learned_values:
                    lhs_tensor = torch.tensor([lhs], device=math_ids.device)
                    rhs_tensor = torch.tensor([rhs], device=math_ids.device)
                    result_logits = self.variable_structured_reasoner.result_logits_for_values(
                        hidden,
                        lhs_tensor,
                        rhs_tensor,
                        torch.tensor([op_id], device=math_ids.device),
                    )
                    candidates = self.variable_structured_reasoner.arithmetic_candidates(
                        lhs_tensor,
                        rhs_tensor,
                    )
                    result = int(
                        candidates.gather(
                            dim=-1,
                            index=result_logits.argmax(dim=-1, keepdim=True),
                        )[0, 0].item()
                    )
                op = self._trace_op_to_text(op_id)
                parts.append(f"{lhs}{op}{rhs}={result}")
                final = result
                values[position : position + 2] = [result]
                del ops[position]
            traces.append(",".join(parts + ([str(final)] if final is not None else [])))
        return traces

    @torch.no_grad()
    def predict_trace_state_values(
        self, problem_ids: torch.Tensor, math_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        trace_slots = self.predict_trace_slots(problem_ids, math_ids)
        return self.trace_state_regression_head(trace_slots.mean(dim=1)) * TRACE_STATE_SCALE

    @torch.no_grad()
    def predict_step_state_values(
        self, math_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.step_state_head(math_ids)["values"]

    @torch.no_grad()
    def solve_ids_from_state_conditioned(
        self,
        problem_ids: torch.Tensor,
        math_ids: torch.Tensor,
        pad_id: int,
    ) -> list[list[int]]:
        slots = self.predict_state_conditioned_slots(problem_ids, math_ids)
        return self.readout.decode_ids(slots, pad_id=pad_id)

    @torch.no_grad()
    def solve_ids_from_value_conditioned(
        self,
        math_ids: torch.Tensor,
        pad_id: int,
    ) -> list[list[int]]:
        slots = self.predict_value_conditioned_slots(math_ids)
        return self.readout.decode_ids(slots, pad_id=pad_id)

    @torch.no_grad()
    def solve_reasoning_sequence_ids(
        self,
        problem_ids: torch.Tensor,
        math_ids: torch.Tensor,
        bos_id: int,
        eos_id: int,
        pad_id: int,
    ) -> list[list[int]]:
        slots = self.predict_reasoning_state_slots(problem_ids, math_ids)
        batch, num_states, num_slots, d_model = slots.shape
        sequence_slots = slots.reshape(batch, num_states * num_slots, d_model)
        return self.reasoning_sequence_readout.decode_ids(
            sequence_slots,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
        )

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
