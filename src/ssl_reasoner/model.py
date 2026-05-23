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
LATENT_REASONING_TEMPERATURE = 0.07
TRANSITION_DIGITS = 5
LATENT_REASONING_CARRY_CLASSES = 100
TRANSITION_VALUE_MIN = -1000
TRANSITION_VALUE_MAX = 10000
TRANSITION_VALUE_CLASSES = TRANSITION_VALUE_MAX - TRANSITION_VALUE_MIN + 1
STANDALONE_TRANSITION_DIGITS = 10
FACTOR_BUCKET_BASES = (0, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000)
FACTOR_BUCKET_SIZES = (10, 90, 900, 9000, 90000, 900000, 9000000, 90000000, 900000000)
FACTOR_OFFSET_CLASSES = 1000
DECOMPOSED_DIGIT_FEATURES = 8
MATH_FEATURE_NUM_OFFSET = 1
MATH_FEATURE_PLUS_ID = 202
MATH_FEATURE_MINUS_ID = 203
MATH_FEATURE_TIMES_ID = 204
TRACE_ID_TO_OP = ("", "+", "-", "*")


class StandaloneArithmeticTransition(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.value_embed = nn.Embedding(TRANSITION_VALUE_CLASSES, d_model)
        self.lhs_proj = nn.Sequential(
            nn.Linear(d_model + 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.rhs_proj = nn.Sequential(
            nn.Linear(d_model + 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.op_embed = nn.Embedding(4, d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, TRANSITION_VALUE_CLASSES),
        )
        self.digit_head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 2 + STANDALONE_TRANSITION_DIGITS * 10),
        )
        self.factor_head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 2 + len(FACTOR_BUCKET_BASES) + FACTOR_OFFSET_CLASSES),
        )
        self.place_embed = nn.Embedding(STANDALONE_TRANSITION_DIGITS, d_model)
        self.input_digit_embed = nn.Embedding(10, d_model)
        self.decomposed_sign_head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.decomposed_digit_head = nn.Sequential(
            nn.LayerNorm(d_model * 7 + DECOMPOSED_DIGIT_FEATURES),
            nn.Linear(d_model * 7 + DECOMPOSED_DIGIT_FEATURES, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 10),
        )

    @staticmethod
    def _number_features(values: torch.Tensor) -> torch.Tensor:
        values = values.to(torch.float)
        return torch.stack(
            [
                values / 1000.0,
                values.abs().clamp(max=10000.0).log1p() / 10.0,
                values.lt(0).to(torch.float),
            ],
            dim=-1,
        )

    def _value_representation(self, values: torch.Tensor) -> torch.Tensor:
        class_ids = self.value_to_class(values)
        return torch.cat([self.value_embed(class_ids), self._number_features(values)], dim=-1)

    @staticmethod
    def value_to_class(values: torch.Tensor) -> torch.Tensor:
        return values.round().long().clamp(
            min=TRANSITION_VALUE_MIN,
            max=TRANSITION_VALUE_MAX,
        ) - TRANSITION_VALUE_MIN

    @staticmethod
    def class_to_value(class_ids: torch.Tensor) -> torch.Tensor:
        return class_ids.long() + TRANSITION_VALUE_MIN

    def forward(
        self,
        lhs: torch.Tensor,
        op_ids: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        return self.class_logits(lhs, op_ids, rhs)

    def encode_transition(
        self,
        lhs: torch.Tensor,
        op_ids: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        lhs_h = self.lhs_proj(self._value_representation(lhs))
        rhs_h = self.rhs_proj(self._value_representation(rhs))
        op_h = self.op_embed(op_ids.clamp(min=0, max=3).long())
        return torch.cat([lhs_h, op_h, rhs_h], dim=-1)

    def class_logits(
        self,
        lhs: torch.Tensor,
        op_ids: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(self.encode_transition(lhs, op_ids, rhs))

    @staticmethod
    def value_to_sign_digits(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rounded = values.round().long()
        sign = rounded.lt(0).long()
        magnitude = rounded.abs().clamp(max=(10 ** STANDALONE_TRANSITION_DIGITS) - 1)
        digits = []
        for place in reversed(range(STANDALONE_TRANSITION_DIGITS)):
            divisor = 10 ** place
            digits.append((magnitude // divisor) % 10)
        return sign, torch.stack(digits, dim=-1)

    @staticmethod
    def sign_digits_to_value(sign_ids: torch.Tensor, digit_ids: torch.Tensor) -> torch.Tensor:
        multipliers = torch.tensor(
            [10 ** place for place in reversed(range(STANDALONE_TRANSITION_DIGITS))],
            device=digit_ids.device,
            dtype=torch.long,
        )
        magnitude = (digit_ids.long() * multipliers).sum(dim=-1)
        return torch.where(sign_ids.long().eq(1), -magnitude, magnitude)

    def digit_logits(
        self,
        lhs: torch.Tensor,
        op_ids: torch.Tensor,
        rhs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.digit_head(self.encode_transition(lhs, op_ids, rhs))
        sign_logits = logits[:, :2]
        digit_logits = logits[:, 2:].view(-1, STANDALONE_TRANSITION_DIGITS, 10)
        return sign_logits, digit_logits

    @staticmethod
    def _abs_digit_matrix(values: torch.Tensor) -> torch.Tensor:
        rounded = values.round().long().abs().clamp(
            max=(10 ** STANDALONE_TRANSITION_DIGITS) - 1
        )
        digits = []
        for place in reversed(range(STANDALONE_TRANSITION_DIGITS)):
            divisor = 10 ** place
            digits.append((rounded // divisor) % 10)
        return torch.stack(digits, dim=-1)

    def decomposed_logits(
        self,
        lhs: torch.Tensor,
        op_ids: torch.Tensor,
        rhs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transition = self.encode_transition(lhs, op_ids, rhs)
        batch_size = transition.size(0)
        lhs_digits = self._abs_digit_matrix(lhs)
        rhs_digits = self._abs_digit_matrix(rhs)
        place_ids = torch.arange(
            STANDALONE_TRANSITION_DIGITS,
            device=transition.device,
            dtype=torch.long,
        )
        place_h = self.place_embed(place_ids).unsqueeze(0).expand(batch_size, -1, -1)
        lhs_digit_h = self.input_digit_embed(lhs_digits)
        rhs_digit_h = self.input_digit_embed(rhs_digits)
        op_h = self.op_embed(op_ids.clamp(min=0, max=3).long()).unsqueeze(1).expand(
            -1, STANDALONE_TRANSITION_DIGITS, -1
        )
        global_h = transition.unsqueeze(1).expand(-1, STANDALONE_TRANSITION_DIGITS, -1)
        place_scale = place_ids.to(torch.float).div(
            max(STANDALONE_TRANSITION_DIGITS - 1, 1)
        ).unsqueeze(0).expand(batch_size, -1)
        lhs_sign = lhs.lt(0).to(torch.float).unsqueeze(-1).expand(-1, STANDALONE_TRANSITION_DIGITS)
        rhs_sign = rhs.lt(0).to(torch.float).unsqueeze(-1).expand(-1, STANDALONE_TRANSITION_DIGITS)
        op_float = op_ids.to(torch.float).unsqueeze(-1).div(3.0).expand(
            -1, STANDALONE_TRANSITION_DIGITS
        )
        numeric_features = torch.stack(
            [
                lhs_digits.to(torch.float) / 9.0,
                rhs_digits.to(torch.float) / 9.0,
                place_scale,
                lhs_sign,
                rhs_sign,
                op_float,
                lhs.abs().clamp(max=100000.0).log1p().unsqueeze(-1).expand(
                    -1, STANDALONE_TRANSITION_DIGITS
                ) / 12.0,
                rhs.abs().clamp(max=100000.0).log1p().unsqueeze(-1).expand(
                    -1, STANDALONE_TRANSITION_DIGITS
                ) / 12.0,
            ],
            dim=-1,
        )
        digit_input = torch.cat(
            [global_h, place_h, lhs_digit_h, rhs_digit_h, op_h, numeric_features],
            dim=-1,
        )
        return self.decomposed_sign_head(transition), self.decomposed_digit_head(digit_input)

    @staticmethod
    def value_to_factor_targets(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rounded = values.round().long()
        sign = rounded.lt(0).long()
        max_magnitude = FACTOR_BUCKET_BASES[-1] + FACTOR_BUCKET_SIZES[-1] - 1
        magnitude = rounded.abs().clamp(max=max_magnitude)
        bucket = torch.zeros_like(magnitude)
        offset = torch.zeros_like(magnitude)
        for idx, (base, size) in enumerate(zip(FACTOR_BUCKET_BASES, FACTOR_BUCKET_SIZES)):
            if idx + 1 < len(FACTOR_BUCKET_BASES):
                is_bucket = magnitude.ge(base) & magnitude.lt(FACTOR_BUCKET_BASES[idx + 1])
            else:
                is_bucket = magnitude.ge(base) & magnitude.le(base + size - 1)
            relative = (magnitude - base).clamp(min=0, max=size - 1)
            scaled = (relative * FACTOR_OFFSET_CLASSES) // size
            bucket = torch.where(is_bucket, torch.full_like(bucket, idx), bucket)
            offset = torch.where(is_bucket, scaled.clamp(max=FACTOR_OFFSET_CLASSES - 1), offset)
        return sign, bucket, offset

    @staticmethod
    def factor_to_value(
        sign_ids: torch.Tensor,
        bucket_ids: torch.Tensor,
        offset_ids: torch.Tensor,
    ) -> torch.Tensor:
        bases = torch.tensor(FACTOR_BUCKET_BASES, device=bucket_ids.device, dtype=torch.long)
        sizes = torch.tensor(FACTOR_BUCKET_SIZES, device=bucket_ids.device, dtype=torch.long)
        bucket_ids = bucket_ids.long().clamp(min=0, max=len(FACTOR_BUCKET_BASES) - 1)
        base = bases[bucket_ids]
        size = sizes[bucket_ids]
        offset_ids = offset_ids.long().clamp(min=0, max=FACTOR_OFFSET_CLASSES - 1)
        relative = (offset_ids * size) // FACTOR_OFFSET_CLASSES
        magnitude = base + relative
        return torch.where(sign_ids.long().eq(1), -magnitude, magnitude)

    def factor_logits(
        self,
        lhs: torch.Tensor,
        op_ids: torch.Tensor,
        rhs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.factor_head(self.encode_transition(lhs, op_ids, rhs))
        sign_logits = logits[:, :2]
        bucket_end = 2 + len(FACTOR_BUCKET_BASES)
        bucket_logits = logits[:, 2:bucket_end]
        offset_logits = logits[:, bucket_end:]
        return sign_logits, bucket_logits, offset_logits

    def loss(
        self,
        lhs: torch.Tensor,
        op_ids: torch.Tensor,
        rhs: torch.Tensor,
        result: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits = self(lhs.reshape(-1), op_ids.reshape(-1), rhs.reshape(-1))
        flat_result = result.reshape(-1)
        targets = self.value_to_class(flat_result)
        in_range = flat_result.ge(TRANSITION_VALUE_MIN) & flat_result.le(TRANSITION_VALUE_MAX)
        flat_mask = mask.reshape(-1) * in_range.to(mask.dtype)
        per_row = F.cross_entropy(logits, targets, reduction="none")
        loss = (per_row * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)
        pred = self.class_to_value(logits.argmax(dim=-1))
        acc = (
            (pred == flat_result.round().long()).float() * flat_mask
        ).sum() / flat_mask.sum().clamp(min=1.0)
        coverage = flat_mask.sum() / mask.reshape(-1).sum().clamp(min=1.0)
        sign_logits, digit_logits = self.digit_logits(
            lhs.reshape(-1),
            op_ids.reshape(-1),
            rhs.reshape(-1),
        )
        sign_targets, digit_targets = self.value_to_sign_digits(flat_result)
        active_mask = mask.reshape(-1)
        digit_sign_loss = F.cross_entropy(
            sign_logits,
            sign_targets,
            reduction="none",
        )
        digit_sign_loss = (
            digit_sign_loss * active_mask
        ).sum() / active_mask.sum().clamp(min=1.0)
        digit_value_loss = F.cross_entropy(
            digit_logits.reshape(-1, 10),
            digit_targets.reshape(-1),
            reduction="none",
        ).view(-1, STANDALONE_TRANSITION_DIGITS)
        digit_value_loss = (
            digit_value_loss.mean(dim=-1) * active_mask
        ).sum() / active_mask.sum().clamp(min=1.0)
        digit_pred = self.sign_digits_to_value(
            sign_logits.argmax(dim=-1),
            digit_logits.argmax(dim=-1),
        )
        digit_acc = (
            (digit_pred == flat_result.round().long()).float() * active_mask
        ).sum() / active_mask.sum().clamp(min=1.0)
        factor_sign_logits, factor_bucket_logits, factor_offset_logits = self.factor_logits(
            lhs.reshape(-1),
            op_ids.reshape(-1),
            rhs.reshape(-1),
        )
        factor_sign_targets, factor_bucket_targets, factor_offset_targets = (
            self.value_to_factor_targets(flat_result)
        )
        factor_sign_loss = F.cross_entropy(
            factor_sign_logits,
            factor_sign_targets,
            reduction="none",
        )
        factor_bucket_loss = F.cross_entropy(
            factor_bucket_logits,
            factor_bucket_targets,
            reduction="none",
        )
        factor_offset_loss = F.cross_entropy(
            factor_offset_logits,
            factor_offset_targets,
            reduction="none",
        )
        factor_loss = (
            (factor_sign_loss + factor_bucket_loss + factor_offset_loss) * active_mask
        ).sum() / active_mask.sum().clamp(min=1.0)
        factor_pred = self.factor_to_value(
            factor_sign_logits.argmax(dim=-1),
            factor_bucket_logits.argmax(dim=-1),
            factor_offset_logits.argmax(dim=-1),
        )
        factor_acc = (
            (factor_pred == flat_result.round().long()).float() * active_mask
        ).sum() / active_mask.sum().clamp(min=1.0)
        decomposed_sign_logits, decomposed_digit_logits = self.decomposed_logits(
            lhs.reshape(-1),
            op_ids.reshape(-1),
            rhs.reshape(-1),
        )
        decomposed_sign_loss = F.cross_entropy(
            decomposed_sign_logits,
            sign_targets,
            reduction="none",
        )
        decomposed_sign_loss = (
            decomposed_sign_loss * active_mask
        ).sum() / active_mask.sum().clamp(min=1.0)
        decomposed_digit_loss = F.cross_entropy(
            decomposed_digit_logits.reshape(-1, 10),
            digit_targets.reshape(-1),
            reduction="none",
        ).view(-1, STANDALONE_TRANSITION_DIGITS)
        decomposed_digit_loss = (
            decomposed_digit_loss.mean(dim=-1) * active_mask
        ).sum() / active_mask.sum().clamp(min=1.0)
        decomposed_pred = self.sign_digits_to_value(
            decomposed_sign_logits.argmax(dim=-1),
            decomposed_digit_logits.argmax(dim=-1),
        )
        decomposed_acc = (
            (decomposed_pred == flat_result.round().long()).float() * active_mask
        ).sum() / active_mask.sum().clamp(min=1.0)
        return {
            "loss": (
                loss
                + digit_sign_loss
                + digit_value_loss
                + factor_loss
                + decomposed_sign_loss
                + decomposed_digit_loss
            ),
            "standalone_transition_loss": loss,
            "standalone_transition_acc": acc,
            "standalone_transition_coverage": coverage,
            "standalone_transition_digit_sign_loss": digit_sign_loss,
            "standalone_transition_digit_value_loss": digit_value_loss,
            "standalone_transition_digit_acc": digit_acc,
            "standalone_transition_factor_loss": factor_loss,
            "standalone_transition_factor_acc": factor_acc,
            "standalone_transition_decomposed_sign_loss": decomposed_sign_loss,
            "standalone_transition_decomposed_digit_loss": decomposed_digit_loss,
            "standalone_transition_decomposed_acc": decomposed_acc,
        }

    @torch.no_grad()
    def predict_value(
        self,
        lhs: torch.Tensor,
        op_ids: torch.Tensor,
        rhs: torch.Tensor,
        mode: str = "class",
    ) -> torch.Tensor:
        class_logits = self(lhs, op_ids, rhs)
        class_pred = self.class_to_value(class_logits.argmax(dim=-1))
        if mode == "class":
            return class_pred
        sign_logits, digit_logits = self.digit_logits(lhs, op_ids, rhs)
        digit_pred = self.sign_digits_to_value(
            sign_logits.argmax(dim=-1),
            digit_logits.argmax(dim=-1),
        )
        if mode == "digit":
            return digit_pred
        decomposed_sign_logits, decomposed_digit_logits = self.decomposed_logits(lhs, op_ids, rhs)
        decomposed_pred = self.sign_digits_to_value(
            decomposed_sign_logits.argmax(dim=-1),
            decomposed_digit_logits.argmax(dim=-1),
        )
        if mode == "decomposed":
            return decomposed_pred
        factor_sign_logits, factor_bucket_logits, factor_offset_logits = self.factor_logits(
            lhs, op_ids, rhs
        )
        factor_pred = self.factor_to_value(
            factor_sign_logits.argmax(dim=-1),
            factor_bucket_logits.argmax(dim=-1),
            factor_offset_logits.argmax(dim=-1),
        )
        if mode == "factor":
            return factor_pred
        if mode != "hybrid":
            raise ValueError(f"Unknown standalone transition mode: {mode}")
        class_probs = class_logits.softmax(dim=-1)
        class_conf = class_probs.max(dim=-1).values
        at_boundary = class_pred.eq(TRANSITION_VALUE_MIN) | class_pred.eq(TRANSITION_VALUE_MAX)
        use_factor = at_boundary | class_conf.lt(0.75)
        return torch.where(use_factor, factor_pred, class_pred)


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


class ReasoningStepTargetEncoder(nn.Module):
    def __init__(self, d_model: int, max_steps: int):
        super().__init__()
        self.max_steps = max_steps
        self.value_embed = nn.Embedding(TRANSITION_VALUE_CLASSES, d_model)
        self.op_embed = nn.Embedding(4, d_model)
        self.kind_embed = nn.Embedding(2, d_model)
        self.position_embed = nn.Embedding(max_steps + 1, d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model * 5 + 3),
            nn.Linear(d_model * 5 + 3, d_model * 3),
            nn.GELU(),
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def value_to_class(values: torch.Tensor) -> torch.Tensor:
        return values.round().long().clamp(
            min=TRANSITION_VALUE_MIN,
            max=TRANSITION_VALUE_MAX,
        ) - TRANSITION_VALUE_MIN

    @staticmethod
    def class_to_value(class_ids: torch.Tensor) -> torch.Tensor:
        return class_ids.long() + TRANSITION_VALUE_MIN

    def forward(
        self,
        op_ids: torch.Tensor,
        values: torch.Tensor,
        kind_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, states = op_ids.shape
        if position_ids is None:
            position_ids = torch.arange(states, device=op_ids.device).unsqueeze(0).expand(
                batch, -1
            )
        lhs = values[:, :, 0]
        rhs = values[:, :, 1]
        result = values[:, :, 2]
        features = torch.stack(
            [
                lhs / 1000.0,
                rhs / 1000.0,
                result / 1000.0,
            ],
            dim=-1,
        )
        encoded = torch.cat(
            [
                self.op_embed(op_ids.clamp(min=0, max=3).long()),
                self.value_embed(self.value_to_class(lhs)),
                self.value_embed(self.value_to_class(rhs)),
                self.value_embed(self.value_to_class(result)),
                self.kind_embed(kind_ids.clamp(min=0, max=1).long())
                + self.position_embed(position_ids.clamp(max=self.max_steps)),
                features,
            ],
            dim=-1,
        )
        return F.normalize(self.net(encoded), dim=-1)


class LatentReasoningSequencePredictor(nn.Module):
    def __init__(
        self,
        math_vocab_size: int,
        d_model: int,
        max_math_len: int,
        max_steps: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.token_embed = nn.Embedding(math_vocab_size, d_model, padding_idx=0)
        self.math_pos_embed = nn.Parameter(torch.randn(1, max_math_len, d_model) * 0.02)
        self.query_embed = nn.Parameter(torch.randn(1, max_steps + 1, d_model) * 0.02)
        self.kind_embed = nn.Embedding(2, d_model)
        self.start_state = nn.Parameter(torch.randn(1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.recurrent_in = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )
        self.step_cell = nn.GRUCell(d_model, d_model)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, math_ids: torch.Tensor) -> torch.Tensor:
        batch = math_ids.size(0)
        math_tokens = self.token_embed(math_ids) + self.math_pos_embed[:, : math_ids.size(1)]
        query_tokens = self.query_embed.expand(batch, -1, -1).clone()
        kind_ids = torch.zeros(
            self.max_steps + 1,
            device=math_ids.device,
            dtype=torch.long,
        )
        kind_ids[-1] = 1
        query_tokens = query_tokens + self.kind_embed(kind_ids).unsqueeze(0)
        tokens = torch.cat([math_tokens, query_tokens], dim=1)
        math_mask = math_ids.ne(0)
        query_mask = torch.ones(
            batch,
            self.max_steps + 1,
            device=math_ids.device,
            dtype=torch.bool,
        )
        mask = torch.cat([math_mask, query_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~mask)
        query_context = encoded[:, -self.max_steps - 1 :]
        prev = F.normalize(self.start_state.expand(batch, -1), dim=-1)
        hidden = query_context.new_zeros(batch, query_context.size(-1))
        reasoning = []
        for step in range(self.max_steps + 1):
            step_context = query_context[:, step]
            cell_input = self.recurrent_in(torch.cat([step_context, prev], dim=-1))
            hidden = self.step_cell(cell_input, hidden)
            current = F.normalize(
                self.out(torch.cat([step_context, hidden, prev], dim=-1)),
                dim=-1,
            )
            reasoning.append(current)
            prev = current
        return torch.stack(reasoning, dim=1)


class LatentReasoningSequence(nn.Module):
    def __init__(
        self,
        math_vocab_size: int,
        d_model: int,
        max_math_len: int,
        max_steps: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.target_encoder = ReasoningStepTargetEncoder(d_model, max_steps)
        self.predictor = LatentReasoningSequencePredictor(
            math_vocab_size,
            d_model,
            max_math_len,
            max_steps,
            num_heads=num_heads,
        )
        self.active_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.op_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 3 * TRANSITION_VALUE_CLASSES),
        )
        self.digit_value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 3 * (2 + TRANSITION_DIGITS * 10)),
        )
        self.process_digit_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, TRANSITION_DIGITS * 10),
        )
        self.process_carry_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, TRANSITION_DIGITS * LATENT_REASONING_CARRY_CLASSES),
        )
        process_feature_dim = TRANSITION_DIGITS * (
            10 + LATENT_REASONING_CARRY_CLASSES
        )
        self.process_conditioned_value_head = nn.Sequential(
            nn.LayerNorm(d_model + process_feature_dim),
            nn.Linear(d_model + process_feature_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 3 * TRANSITION_VALUE_CLASSES),
        )
        self.state_value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, (max_steps + 1) * (2 + TRANSITION_DIGITS * 10)),
        )
        self.state_value_active_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, max_steps + 1),
        )
        self.state_op_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, max_steps * 4),
        )
        self.state_op_active_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, max_steps),
        )
        state_feature_dim = (
            (max_steps + 1) * (2 + TRANSITION_DIGITS * 10)
            + (max_steps + 1)
            + max_steps * 4
            + max_steps
        )
        self.state_conditioned_op_head = nn.Sequential(
            nn.LayerNorm(d_model + state_feature_dim),
            nn.Linear(d_model + state_feature_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 4),
        )
        self.state_conditioned_value_head = nn.Sequential(
            nn.LayerNorm(d_model + state_feature_dim),
            nn.Linear(d_model + state_feature_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 3 * TRANSITION_VALUE_CLASSES),
        )
        self.slot_lhs_head = nn.Sequential(
            nn.LayerNorm(d_model + state_feature_dim),
            nn.Linear(d_model + state_feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, max_steps + 1),
        )
        self.slot_rhs_head = nn.Sequential(
            nn.LayerNorm(d_model + state_feature_dim),
            nn.Linear(d_model + state_feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, max_steps + 1),
        )
        self.slot_op_head = nn.Sequential(
            nn.LayerNorm(d_model + state_feature_dim),
            nn.Linear(d_model + state_feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, max_steps),
        )
        self.slot_result_head = nn.Sequential(
            nn.LayerNorm(d_model + state_feature_dim),
            nn.Linear(d_model + state_feature_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, TRANSITION_VALUE_CLASSES),
        )
        self.slot_transition_value_embed = nn.Embedding(TRANSITION_VALUE_CLASSES, d_model)
        self.slot_transition_op_embed = nn.Embedding(4, d_model)
        self.slot_transition_result_head = nn.Sequential(
            nn.LayerNorm(d_model * 4 + 3),
            nn.Linear(d_model * 4 + 3, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, TRANSITION_VALUE_CLASSES),
        )

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

    @staticmethod
    def _little_endian_digits(values: torch.Tensor) -> torch.Tensor:
        rounded = values.round().long().abs().clamp(max=(10 ** TRANSITION_DIGITS) - 1)
        digits = []
        for place in range(TRANSITION_DIGITS):
            digits.append((rounded // (10 ** place)) % 10)
        return torch.stack(digits, dim=-1)

    @classmethod
    def process_targets(
        cls,
        op_ids: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lhs = values[:, :, 0].round().long()
        rhs = values[:, :, 1].round().long()
        result = values[:, :, 2].round().long()
        lhs_digits = cls._little_endian_digits(lhs)
        rhs_digits = cls._little_endian_digits(rhs)
        result_digits = cls._little_endian_digits(result)
        carry_targets = torch.zeros_like(result_digits)
        process_mask = mask.unsqueeze(-1).expand_as(result_digits).clone()
        supported = (
            op_ids.ge(1)
            & op_ids.le(3)
            & lhs.ge(0)
            & rhs.ge(0)
            & result.ge(0)
            & result.lt(10 ** TRANSITION_DIGITS)
        )
        supported = supported & (
            op_ids.ne(2) | lhs.ge(rhs)
        )
        process_mask = process_mask * supported.unsqueeze(-1).to(process_mask.dtype)

        add_carry = torch.zeros_like(lhs)
        sub_borrow = torch.zeros_like(lhs)
        mul_carry = torch.zeros_like(lhs)
        for place in range(TRANSITION_DIGITS):
            add_total = lhs_digits[:, :, place] + rhs_digits[:, :, place] + add_carry
            add_carry = add_total // 10

            sub_diff = lhs_digits[:, :, place] - rhs_digits[:, :, place] - sub_borrow
            sub_borrow = sub_diff.lt(0).long()

            mul_total = mul_carry
            for left_place in range(place + 1):
                right_place = place - left_place
                mul_total = (
                    mul_total
                    + lhs_digits[:, :, left_place] * rhs_digits[:, :, right_place]
                )
            mul_carry = mul_total // 10

            place_carry = torch.where(
                op_ids.eq(1),
                add_carry,
                torch.where(op_ids.eq(2), sub_borrow, mul_carry),
            )
            carry_targets[:, :, place] = place_carry.clamp(
                max=LATENT_REASONING_CARRY_CLASSES - 1
            )
        return result_digits, carry_targets, process_mask

    @staticmethod
    def expression_state_targets(
        math_ids: torch.Tensor,
        position_ids: torch.Tensor,
        values: torch.Tensor,
        step_mask: torch.Tensor,
        max_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = math_ids.size(0)
        state_count = max_steps + 1
        state_values = values.new_zeros(batch, state_count, max_steps + 1)
        state_value_mask = values.new_zeros(batch, state_count, max_steps + 1)
        state_ops = position_ids.new_zeros(batch, state_count, max_steps)
        state_op_mask = values.new_zeros(batch, state_count, max_steps)
        math_rows = math_ids.detach().cpu().tolist()
        positions = position_ids.detach().cpu().tolist()
        value_rows = values.detach().cpu().tolist()
        mask_rows = step_mask.detach().cpu().tolist()

        for row_idx, math_row in enumerate(math_rows):
            current_values: list[float] = []
            current_ops: list[int] = []
            for token_idx, token in enumerate(math_row):
                if token == 0:
                    continue
                if token == MATH_FEATURE_PLUS_ID:
                    current_ops.append(1)
                elif token == MATH_FEATURE_MINUS_ID:
                    current_ops.append(2)
                elif token == MATH_FEATURE_TIMES_ID:
                    current_ops.append(3)
                elif token < MATH_FEATURE_PLUS_ID:
                    current_values.append(float(token - MATH_FEATURE_NUM_OFFSET))

            last_state_values = current_values[: max_steps + 1]
            last_state_ops = current_ops[:max_steps]
            for step_idx in range(max_steps):
                if mask_rows[row_idx][step_idx] < 0.5:
                    continue
                position = int(positions[row_idx][step_idx])
                if 0 <= position < len(current_ops) and position + 1 < len(current_values):
                    result = float(value_rows[row_idx][step_idx][2])
                    current_values[position : position + 2] = [result]
                    del current_ops[position]
                last_state_values = current_values[: max_steps + 1]
                last_state_ops = current_ops[:max_steps]
                value_len = len(last_state_values)
                op_len = len(last_state_ops)
                if value_len:
                    state_values[row_idx, step_idx, :value_len] = values.new_tensor(
                        last_state_values
                    )
                    state_value_mask[row_idx, step_idx, :value_len] = 1.0
                if op_len:
                    state_ops[row_idx, step_idx, :op_len] = position_ids.new_tensor(
                        last_state_ops
                    )
                    state_op_mask[row_idx, step_idx, :op_len] = 1.0

            value_len = len(last_state_values)
            op_len = len(last_state_ops)
            if value_len:
                state_values[row_idx, -1, :value_len] = values.new_tensor(
                    last_state_values
                )
                state_value_mask[row_idx, -1, :value_len] = 1.0
            if op_len:
                state_ops[row_idx, -1, :op_len] = position_ids.new_tensor(
                    last_state_ops
                )
                state_op_mask[row_idx, -1, :op_len] = 1.0
        return state_values, state_value_mask, state_ops, state_op_mask

    @staticmethod
    def initial_expression_state(
        math_ids: torch.Tensor,
        max_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = math_ids.size(0)
        values = torch.zeros(
            batch,
            max_steps + 1,
            device=math_ids.device,
            dtype=torch.float,
        )
        value_mask = torch.zeros_like(values)
        ops = torch.zeros(batch, max_steps, device=math_ids.device, dtype=torch.long)
        op_mask = torch.zeros(batch, max_steps, device=math_ids.device, dtype=torch.float)
        math_rows = math_ids.detach().cpu().tolist()
        for row_idx, math_row in enumerate(math_rows):
            row_values: list[float] = []
            row_ops: list[int] = []
            for token in math_row:
                if token == 0:
                    continue
                if token == MATH_FEATURE_PLUS_ID:
                    row_ops.append(1)
                elif token == MATH_FEATURE_MINUS_ID:
                    row_ops.append(2)
                elif token == MATH_FEATURE_TIMES_ID:
                    row_ops.append(3)
                elif token < MATH_FEATURE_PLUS_ID:
                    row_values.append(float(token - MATH_FEATURE_NUM_OFFSET))
            value_len = min(len(row_values), max_steps + 1)
            op_len = min(len(row_ops), max_steps)
            if value_len:
                values[row_idx, :value_len] = values.new_tensor(row_values[:value_len])
                value_mask[row_idx, :value_len] = 1.0
            if op_len:
                ops[row_idx, :op_len] = ops.new_tensor(row_ops[:op_len])
                op_mask[row_idx, :op_len] = 1.0
        return values, value_mask, ops, op_mask

    def _state_tensors_to_features(
        self,
        values: torch.Tensor,
        value_mask: torch.Tensor,
        ops: torch.Tensor,
        op_mask: torch.Tensor,
    ) -> torch.Tensor:
        sign_ids, digit_ids = self.value_to_sign_digits(values.reshape(-1))
        sign_features = F.one_hot(sign_ids, num_classes=2).float().view(
            values.size(0),
            values.size(1),
            2,
        )
        digit_features = F.one_hot(digit_ids, num_classes=10).float().view(
            values.size(0),
            values.size(1),
            TRANSITION_DIGITS,
            10,
        )
        op_features = F.one_hot(ops.clamp(min=0, max=3), num_classes=4).float()
        return torch.cat(
            [
                torch.cat(
                    [
                        sign_features.unsqueeze(2),
                        digit_features.flatten(start_dim=2).unsqueeze(2),
                    ],
                    dim=-1,
                ).flatten(start_dim=1),
                value_mask,
                op_features.flatten(start_dim=1),
                op_mask,
            ],
            dim=-1,
        )

    def _targets(
        self,
        op_ids: torch.Tensor,
        values: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = op_ids.size(0)
        final_index = step_mask.sum(dim=1).long().sub(1).clamp(min=0)
        final_result = values[
            torch.arange(batch, device=values.device),
            final_index,
            2,
        ]
        final_values = values.new_zeros(batch, 1, 3)
        final_values[:, 0, 2] = final_result
        all_values = torch.cat([values, final_values], dim=1)
        final_ops = op_ids.new_zeros(batch, 1)
        all_ops = torch.cat([op_ids, final_ops], dim=1)
        final_mask = step_mask.new_ones(batch, 1)
        all_mask = torch.cat([step_mask, final_mask], dim=1)
        kind_ids = op_ids.new_zeros(batch, self.max_steps + 1)
        kind_ids[:, -1] = 1
        return all_ops, all_values, all_mask, kind_ids

    def _decode(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        active_logits = self.active_head(embeddings).squeeze(-1)
        op_logits = self.op_head(embeddings)
        value_logits = self.value_head(embeddings).view(
            embeddings.size(0),
            embeddings.size(1),
            3,
            TRANSITION_VALUE_CLASSES,
        )
        digit_logits = self.digit_value_head(embeddings).view(
            embeddings.size(0),
            embeddings.size(1),
            3,
            2 + TRANSITION_DIGITS * 10,
        )
        sign_logits = digit_logits[:, :, :, :2]
        digit_value_logits = digit_logits[:, :, :, 2:].view(
            embeddings.size(0),
            embeddings.size(1),
            3,
            TRANSITION_DIGITS,
            10,
        )
        process_digit_logits = self.process_digit_head(embeddings).view(
            embeddings.size(0),
            embeddings.size(1),
            TRANSITION_DIGITS,
            10,
        )
        process_carry_logits = self.process_carry_head(embeddings).view(
            embeddings.size(0),
            embeddings.size(1),
            TRANSITION_DIGITS,
            LATENT_REASONING_CARRY_CLASSES,
        )
        process_features = torch.cat(
            [
                process_digit_logits.softmax(dim=-1).flatten(start_dim=2),
                process_carry_logits.softmax(dim=-1).flatten(start_dim=2),
            ],
            dim=-1,
        )
        process_value_logits = self.process_conditioned_value_head(
            torch.cat([embeddings, process_features], dim=-1)
        ).view(
            embeddings.size(0),
            embeddings.size(1),
            3,
            TRANSITION_VALUE_CLASSES,
        )
        return (
            active_logits,
            op_logits,
            value_logits,
            sign_logits,
            digit_value_logits,
            process_digit_logits,
            process_carry_logits,
            process_value_logits,
        )

    def _state_logits(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_value_logits = self.state_value_head(embeddings).view(
            embeddings.size(0),
            embeddings.size(1),
            self.max_steps + 1,
            2 + TRANSITION_DIGITS * 10,
        )
        state_value_active_logits = self.state_value_active_head(embeddings)
        state_op_logits = self.state_op_head(embeddings).view(
            embeddings.size(0),
            embeddings.size(1),
            self.max_steps,
            4,
        )
        state_op_active_logits = self.state_op_active_head(embeddings)
        return (
            state_value_logits,
            state_value_active_logits,
            state_op_logits,
            state_op_active_logits,
        )

    def _state_features(
        self,
        state_value_logits: torch.Tensor,
        state_value_active_logits: torch.Tensor,
        state_op_logits: torch.Tensor,
        state_op_active_logits: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                state_value_logits.softmax(dim=-1).flatten(start_dim=2),
                state_value_active_logits.sigmoid(),
                state_op_logits.softmax(dim=-1).flatten(start_dim=2),
                state_op_active_logits.sigmoid(),
            ],
            dim=-1,
        )

    def _state_conditioned_decode(
        self,
        embeddings: torch.Tensor,
        state_value_logits: torch.Tensor,
        state_value_active_logits: torch.Tensor,
        state_op_logits: torch.Tensor,
        state_op_active_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_features = self._state_features(
            state_value_logits,
            state_value_active_logits,
            state_op_logits,
            state_op_active_logits,
        )
        state_input = torch.cat([embeddings, state_features], dim=-1)
        state_conditioned_op_logits = self.state_conditioned_op_head(state_input)
        state_conditioned_value_logits = self.state_conditioned_value_head(
            state_input
        ).view(
            embeddings.size(0),
            embeddings.size(1),
            3,
            TRANSITION_VALUE_CLASSES,
        )
        return state_conditioned_op_logits, state_conditioned_value_logits

    def _pre_state_features(
        self,
        math_ids: torch.Tensor,
        state_value_logits: torch.Tensor,
        state_value_active_logits: torch.Tensor,
        state_op_logits: torch.Tensor,
        state_op_active_logits: torch.Tensor,
    ) -> torch.Tensor:
        initial_values, initial_value_mask, initial_ops, initial_op_mask = (
            self.initial_expression_state(math_ids, self.max_steps)
        )
        initial_features = self._state_tensors_to_features(
            initial_values,
            initial_value_mask,
            initial_ops,
            initial_op_mask,
        )
        post_features = self._state_features(
            state_value_logits,
            state_value_active_logits,
            state_op_logits,
            state_op_active_logits,
        )
        return torch.cat([initial_features.unsqueeze(1), post_features[:, :-1]], dim=1)

    def _pre_state_values_ops(
        self,
        math_ids: torch.Tensor,
        state_value_logits: torch.Tensor,
        state_op_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        initial_values, _initial_value_mask, initial_ops, _initial_op_mask = (
            self.initial_expression_state(math_ids, self.max_steps)
        )
        state_value_sign_logits = state_value_logits[:, :, :, :2]
        state_value_digit_logits = state_value_logits[:, :, :, 2:].view(
            state_value_logits.size(0),
            state_value_logits.size(1),
            self.max_steps + 1,
            TRANSITION_DIGITS,
            10,
        )
        post_values = self.sign_digits_to_value(
            state_value_sign_logits.argmax(dim=-1),
            state_value_digit_logits.argmax(dim=-1),
        )
        post_ops = state_op_logits.argmax(dim=-1)
        pre_values = torch.cat([initial_values.long().unsqueeze(1), post_values[:, :-1]], dim=1)
        pre_ops = torch.cat([initial_ops.unsqueeze(1), post_ops[:, :-1]], dim=1)
        return pre_values, pre_ops

    def _slot_decode(
        self,
        embeddings: torch.Tensor,
        math_ids: torch.Tensor,
        state_value_logits: torch.Tensor,
        state_value_active_logits: torch.Tensor,
        state_op_logits: torch.Tensor,
        state_op_active_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_features = self._pre_state_features(
            math_ids,
            state_value_logits,
            state_value_active_logits,
            state_op_logits,
            state_op_active_logits,
        )
        slot_input = torch.cat([embeddings, pre_features], dim=-1)
        lhs_slot_logits = self.slot_lhs_head(slot_input)
        rhs_slot_logits = self.slot_rhs_head(slot_input)
        op_slot_logits = self.slot_op_head(slot_input)
        result_logits = self.slot_result_head(slot_input)
        return lhs_slot_logits, rhs_slot_logits, op_slot_logits, result_logits

    def _slot_transition_result_logits(
        self,
        embeddings: torch.Tensor,
        lhs_values: torch.Tensor,
        rhs_values: torch.Tensor,
        op_ids: torch.Tensor,
    ) -> torch.Tensor:
        lhs = lhs_values.round()
        rhs = rhs_values.round()
        features = torch.stack(
            [
                lhs / 1000.0,
                rhs / 1000.0,
                op_ids.float() / 3.0,
            ],
            dim=-1,
        )
        transition_input = torch.cat(
            [
                embeddings,
                self.slot_transition_value_embed(
                    ReasoningStepTargetEncoder.value_to_class(lhs)
                ),
                self.slot_transition_value_embed(
                    ReasoningStepTargetEncoder.value_to_class(rhs)
                ),
                self.slot_transition_op_embed(op_ids.clamp(min=0, max=3).long()),
                features,
            ],
            dim=-1,
        )
        return self.slot_transition_result_head(transition_input)

    def loss(
        self,
        math_ids: torch.Tensor,
        op_ids: torch.Tensor,
        position_ids: torch.Tensor,
        values: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        all_ops, all_values, all_mask, kind_ids = self._targets(op_ids, values, step_mask)
        target = self.target_encoder(all_ops, all_values, kind_ids).detach()
        pred = self.predictor(math_ids)
        per_state = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
        latent_loss = (per_state * all_mask).sum() / all_mask.sum().clamp(min=1.0)
        cosine = (pred * target).sum(dim=-1)
        cosine_acc = (cosine * all_mask).sum() / all_mask.sum().clamp(min=1.0)

        flat_mask = all_mask.reshape(-1).bool()
        flat_pred = pred.reshape(-1, pred.size(-1))[flat_mask]
        flat_target = target.reshape(-1, target.size(-1))[flat_mask]
        logits = flat_pred @ flat_target.t() / LATENT_REASONING_TEMPERATURE
        labels = torch.arange(flat_pred.size(0), device=flat_pred.device)
        contrastive_loss = F.cross_entropy(logits, labels)

        hard_count = 8
        hard_values = all_values.unsqueeze(2).expand(-1, -1, hard_count, -1).clone()
        hard_ops = all_ops.unsqueeze(2).expand(-1, -1, hard_count).clone()

        hard_values[:, :, 0, 2] = hard_values[:, :, 0, 2] + 1.0
        hard_values[:, :, 1, 2] = hard_values[:, :, 1, 2] - 1.0
        hard_ops[:, :, 2] = torch.where(
            hard_ops[:, :, 2].eq(3),
            torch.ones_like(hard_ops[:, :, 2]),
            hard_ops[:, :, 2] + 1,
        )
        hard_values[:, :, 3, 0] = all_values[:, :, 1]
        hard_values[:, :, 3, 1] = all_values[:, :, 0]

        prev_values = torch.roll(all_values, shifts=1, dims=1)
        next_values = torch.roll(all_values, shifts=-1, dims=1)
        prev_values[:, 0] = all_values[:, 0]
        next_values[:, -1] = all_values[:, -1]
        hard_values[:, :, 4, 0] = prev_values[:, :, 2]
        hard_values[:, :, 5, 1] = next_values[:, :, 2]
        hard_values[:, :, 6] = prev_values
        hard_values[:, :, 7] = next_values

        flat_hard_values = hard_values.reshape(-1, hard_count, 3)[flat_mask]
        flat_hard_ops = hard_ops.reshape(-1, hard_count)[flat_mask]
        hard_kind = kind_ids.reshape(-1)[flat_mask].unsqueeze(1).expand(
            -1,
            hard_count,
        )
        hard_position_ids = torch.arange(
            all_ops.size(1),
            device=all_ops.device,
        ).unsqueeze(0).expand_as(all_ops)
        hard_positions = hard_position_ids.reshape(-1)[flat_mask].unsqueeze(1).expand(
            -1,
            hard_count,
        )
        hard_target = self.target_encoder(
            flat_hard_ops,
            flat_hard_values,
            hard_kind,
            position_ids=hard_positions,
        ).detach()
        hard_scores = (flat_pred.unsqueeze(1) * hard_target).sum(dim=-1)
        pos_scores = (flat_pred * flat_target).sum(dim=-1, keepdim=True)
        hard_logits = torch.cat([pos_scores, hard_scores], dim=1) / LATENT_REASONING_TEMPERATURE
        hard_loss = F.cross_entropy(
            hard_logits,
            torch.zeros(flat_pred.size(0), device=flat_pred.device, dtype=torch.long),
        )

        (
            active_logits,
            op_logits,
            value_logits,
            sign_logits,
            digit_value_logits,
            process_digit_logits,
            process_carry_logits,
            process_value_logits,
        ) = self._decode(pred)
        (
            state_value_logits,
            state_value_active_logits,
            state_op_logits,
            state_op_active_logits,
        ) = self._state_logits(pred)
        state_value_sign_logits = state_value_logits[:, :, :, :2]
        state_value_digit_logits = state_value_logits[:, :, :, 2:].view(
            pred.size(0),
            pred.size(1),
            self.max_steps + 1,
            TRANSITION_DIGITS,
            10,
        )
        state_conditioned_op_logits, state_conditioned_value_logits = (
            self._state_conditioned_decode(
                pred,
                state_value_logits,
                state_value_active_logits,
                state_op_logits,
                state_op_active_logits,
            )
        )
        slot_lhs_logits, slot_rhs_logits, slot_op_logits, slot_result_logits = (
            self._slot_decode(
                pred,
                math_ids,
                state_value_logits,
                state_value_active_logits,
                state_op_logits,
                state_op_active_logits,
            )
        )
        slot_transition_result_logits = self._slot_transition_result_logits(
            pred,
            all_values[:, :, 0],
            all_values[:, :, 1],
            all_ops,
        )
        pred_pre_values, pred_pre_ops = self._pre_state_values_ops(
            math_ids,
            state_value_logits,
            state_op_logits,
        )
        pred_lhs_slots = slot_lhs_logits.detach().argmax(dim=-1)
        pred_rhs_slots = slot_rhs_logits.detach().argmax(dim=-1)
        pred_op_slots = slot_op_logits.detach().argmax(dim=-1)
        pred_slot_lhs_values = pred_pre_values.gather(
            dim=2,
            index=pred_lhs_slots.unsqueeze(-1),
        ).squeeze(-1)
        pred_slot_rhs_values = pred_pre_values.gather(
            dim=2,
            index=pred_rhs_slots.unsqueeze(-1),
        ).squeeze(-1)
        pred_slot_op_ids = pred_pre_ops.gather(
            dim=2,
            index=pred_op_slots.unsqueeze(-1),
        ).squeeze(-1)
        predicted_slot_transition_result_logits = self._slot_transition_result_logits(
            pred,
            pred_slot_lhs_values.detach().float(),
            pred_slot_rhs_values.detach().float(),
            pred_slot_op_ids.detach(),
        )
        state_values, state_value_mask, state_ops, state_op_mask = (
            self.expression_state_targets(
                math_ids,
                position_ids,
                values,
                step_mask,
                self.max_steps,
            )
        )
        state_value_mask = state_value_mask * all_mask.unsqueeze(-1)
        state_op_mask = state_op_mask * all_mask.unsqueeze(-1)
        state_sign_targets, state_digit_targets = self.value_to_sign_digits(
            state_values.reshape(-1)
        )
        state_sign_targets = state_sign_targets.view_as(state_values).long()
        state_digit_targets = state_digit_targets.view(
            state_values.size(0),
            state_values.size(1),
            state_values.size(2),
            TRANSITION_DIGITS,
        )
        state_value_sign_loss = F.cross_entropy(
            state_value_sign_logits.reshape(-1, 2),
            state_sign_targets.reshape(-1),
            reduction="none",
        ).view_as(state_value_mask)
        state_value_sign_loss = (
            state_value_sign_loss * state_value_mask
        ).sum() / state_value_mask.sum().clamp(min=1.0)
        state_value_digit_loss = F.cross_entropy(
            state_value_digit_logits.reshape(-1, 10),
            state_digit_targets.reshape(-1),
            reduction="none",
        ).view(
            state_values.size(0),
            state_values.size(1),
            state_values.size(2),
            TRANSITION_DIGITS,
        )
        state_value_digit_loss = state_value_digit_loss.mean(dim=-1)
        state_value_digit_loss = (
            state_value_digit_loss * state_value_mask
        ).sum() / state_value_mask.sum().clamp(min=1.0)
        state_value_loss = state_value_sign_loss + state_value_digit_loss
        state_value_active_loss = F.binary_cross_entropy_with_logits(
            state_value_active_logits,
            state_value_mask,
        )
        state_op_loss = F.cross_entropy(
            state_op_logits.reshape(-1, 4),
            state_ops.reshape(-1),
            reduction="none",
        ).view_as(state_op_mask)
        state_op_loss = (state_op_loss * state_op_mask).sum() / state_op_mask.sum().clamp(min=1.0)
        state_op_active_loss = F.binary_cross_entropy_with_logits(
            state_op_active_logits,
            state_op_mask,
        )
        active_loss = F.binary_cross_entropy_with_logits(active_logits, all_mask)
        value_targets = ReasoningStepTargetEncoder.value_to_class(all_values)
        sign_targets, digit_targets = self.value_to_sign_digits(all_values.reshape(-1))
        sign_targets = sign_targets.view(all_values.size(0), all_values.size(1), 3)
        digit_targets = digit_targets.view(
            all_values.size(0),
            all_values.size(1),
            3,
            TRANSITION_DIGITS,
        )
        op_loss = F.cross_entropy(
            op_logits.reshape(-1, 4),
            all_ops.reshape(-1),
            reduction="none",
        ).view_as(all_mask)
        op_loss = (op_loss * all_mask).sum() / all_mask.sum().clamp(min=1.0)
        state_conditioned_op_loss = F.cross_entropy(
            state_conditioned_op_logits.reshape(-1, 4),
            all_ops.reshape(-1),
            reduction="none",
        ).view_as(all_mask)
        state_conditioned_op_loss = (
            state_conditioned_op_loss * all_mask
        ).sum() / all_mask.sum().clamp(min=1.0)
        slot_step_mask = step_mask[:, : self.max_steps]
        lhs_slot_targets = position_ids[:, : self.max_steps].clamp(
            min=0,
            max=self.max_steps,
        )
        rhs_slot_targets = (position_ids[:, : self.max_steps] + 1).clamp(
            min=0,
            max=self.max_steps,
        )
        op_slot_targets = position_ids[:, : self.max_steps].clamp(
            min=0,
            max=self.max_steps - 1,
        )
        slot_lhs_loss = F.cross_entropy(
            slot_lhs_logits[:, : self.max_steps].reshape(-1, self.max_steps + 1),
            lhs_slot_targets.reshape(-1),
            reduction="none",
        ).view_as(slot_step_mask)
        slot_lhs_loss = (
            slot_lhs_loss * slot_step_mask
        ).sum() / slot_step_mask.sum().clamp(min=1.0)
        slot_rhs_loss = F.cross_entropy(
            slot_rhs_logits[:, : self.max_steps].reshape(-1, self.max_steps + 1),
            rhs_slot_targets.reshape(-1),
            reduction="none",
        ).view_as(slot_step_mask)
        slot_rhs_loss = (
            slot_rhs_loss * slot_step_mask
        ).sum() / slot_step_mask.sum().clamp(min=1.0)
        slot_op_loss = F.cross_entropy(
            slot_op_logits[:, : self.max_steps].reshape(-1, self.max_steps),
            op_slot_targets.reshape(-1),
            reduction="none",
        ).view_as(slot_step_mask)
        slot_op_loss = (
            slot_op_loss * slot_step_mask
        ).sum() / slot_step_mask.sum().clamp(min=1.0)
        result_state_slots = position_ids[:, : self.max_steps].clamp(
            min=0,
            max=self.max_steps,
        )
        result_state_logits = state_value_logits[:, : self.max_steps].gather(
            dim=2,
            index=result_state_slots.unsqueeze(-1).unsqueeze(-1).expand(
                -1,
                -1,
                -1,
                state_value_logits.size(-1),
            ),
        ).squeeze(2)
        result_state_sign_logits = result_state_logits[:, :, :2]
        result_state_digit_logits = result_state_logits[:, :, 2:].view(
            result_state_logits.size(0),
            result_state_logits.size(1),
            TRANSITION_DIGITS,
            10,
        )
        result_state_sign_targets, result_state_digit_targets = self.value_to_sign_digits(
            all_values[:, : self.max_steps, 2].reshape(-1)
        )
        result_state_sign_targets = result_state_sign_targets.view(
            all_values.size(0),
            self.max_steps,
        )
        result_state_digit_targets = result_state_digit_targets.view(
            all_values.size(0),
            self.max_steps,
            TRANSITION_DIGITS,
        )
        result_state_sign_loss = F.cross_entropy(
            result_state_sign_logits.reshape(-1, 2),
            result_state_sign_targets.reshape(-1),
            reduction="none",
        ).view_as(slot_step_mask)
        result_state_digit_loss = F.cross_entropy(
            result_state_digit_logits.reshape(-1, 10),
            result_state_digit_targets.reshape(-1),
            reduction="none",
        ).view(
            all_values.size(0),
            self.max_steps,
            TRANSITION_DIGITS,
        )
        result_state_digit_loss = result_state_digit_loss.mean(dim=-1)
        result_state_value_loss = (
            (result_state_sign_loss + result_state_digit_loss) * slot_step_mask
        ).sum() / slot_step_mask.sum().clamp(min=1.0)
        if self.max_steps > 1:
            pre_slot_mask = slot_step_mask[:, 1:]
            pre_lhs_slots = lhs_slot_targets[:, 1:]
            pre_rhs_slots = rhs_slot_targets[:, 1:]
            pre_op_slots = op_slot_targets[:, 1:]
            pre_state_value_logits = state_value_logits[:, : self.max_steps - 1]
            pre_lhs_logits = pre_state_value_logits.gather(
                dim=2,
                index=pre_lhs_slots.unsqueeze(-1).unsqueeze(-1).expand(
                    -1,
                    -1,
                    -1,
                    state_value_logits.size(-1),
                ),
            ).squeeze(2)
            pre_rhs_logits = pre_state_value_logits.gather(
                dim=2,
                index=pre_rhs_slots.unsqueeze(-1).unsqueeze(-1).expand(
                    -1,
                    -1,
                    -1,
                    state_value_logits.size(-1),
                ),
            ).squeeze(2)
            pre_op_logits = state_op_logits[:, : self.max_steps - 1].gather(
                dim=2,
                index=pre_op_slots.unsqueeze(-1).unsqueeze(-1).expand(
                    -1,
                    -1,
                    -1,
                    state_op_logits.size(-1),
                ),
            ).squeeze(2)
            pre_value_logits = torch.stack([pre_lhs_logits, pre_rhs_logits], dim=2)
            pre_value_targets = all_values[:, 1 : self.max_steps, :2]
            pre_sign_targets, pre_digit_targets = self.value_to_sign_digits(
                pre_value_targets.reshape(-1)
            )
            pre_sign_targets = pre_sign_targets.view(
                all_values.size(0),
                self.max_steps - 1,
                2,
            )
            pre_digit_targets = pre_digit_targets.view(
                all_values.size(0),
                self.max_steps - 1,
                2,
                TRANSITION_DIGITS,
            )
            pre_sign_logits = pre_value_logits[:, :, :, :2]
            pre_digit_logits = pre_value_logits[:, :, :, 2:].view(
                all_values.size(0),
                self.max_steps - 1,
                2,
                TRANSITION_DIGITS,
                10,
            )
            pre_sign_loss = F.cross_entropy(
                pre_sign_logits.reshape(-1, 2),
                pre_sign_targets.reshape(-1),
                reduction="none",
            ).view(all_values.size(0), self.max_steps - 1, 2)
            pre_digit_loss = F.cross_entropy(
                pre_digit_logits.reshape(-1, 10),
                pre_digit_targets.reshape(-1),
                reduction="none",
            ).view(
                all_values.size(0),
                self.max_steps - 1,
                2,
                TRANSITION_DIGITS,
            )
            pre_digit_loss = pre_digit_loss.mean(dim=-1)
            pre_value_mask = pre_slot_mask.unsqueeze(-1).expand_as(pre_sign_loss)
            pre_slot_operand_value_loss = (
                (pre_sign_loss + pre_digit_loss) * pre_value_mask
            ).sum() / pre_value_mask.sum().clamp(min=1.0)
            pre_slot_op_loss = F.cross_entropy(
                pre_op_logits.reshape(-1, 4),
                all_ops[:, 1 : self.max_steps].reshape(-1),
                reduction="none",
            ).view_as(pre_slot_mask)
            pre_slot_op_loss = (
                pre_slot_op_loss * pre_slot_mask
            ).sum() / pre_slot_mask.sum().clamp(min=1.0)
        else:
            pre_slot_operand_value_loss = state_value_loss.new_tensor(0.0)
            pre_slot_op_loss = state_value_loss.new_tensor(0.0)
        value_loss = F.cross_entropy(
            value_logits.reshape(-1, TRANSITION_VALUE_CLASSES),
            value_targets.reshape(-1),
            reduction="none",
        ).view(all_mask.size(0), all_mask.size(1), 3)
        value_field_mask = all_mask.unsqueeze(-1).expand_as(value_loss).clone()
        value_field_mask[:, -1, :2] = 0.0
        value_loss = (value_loss * value_field_mask).sum() / value_field_mask.sum().clamp(min=1.0)
        state_conditioned_value_loss = F.cross_entropy(
            state_conditioned_value_logits.reshape(-1, TRANSITION_VALUE_CLASSES),
            value_targets.reshape(-1),
            reduction="none",
        ).view(all_mask.size(0), all_mask.size(1), 3)
        state_conditioned_value_loss = (
            state_conditioned_value_loss * value_field_mask
        ).sum() / value_field_mask.sum().clamp(min=1.0)
        slot_result_loss = F.cross_entropy(
            slot_result_logits.reshape(-1, TRANSITION_VALUE_CLASSES),
            value_targets[:, :, 2].reshape(-1),
            reduction="none",
        ).view_as(all_mask)
        slot_result_loss = (
            slot_result_loss * all_mask
        ).sum() / all_mask.sum().clamp(min=1.0)
        transition_mask = torch.cat(
            [
                step_mask[:, : self.max_steps],
                step_mask.new_zeros(step_mask.size(0), 1),
            ],
            dim=1,
        )
        slot_transition_result_loss = F.cross_entropy(
            slot_transition_result_logits.reshape(-1, TRANSITION_VALUE_CLASSES),
            value_targets[:, :, 2].reshape(-1),
            reduction="none",
        ).view_as(all_mask)
        slot_transition_result_loss = (
            slot_transition_result_loss * transition_mask
        ).sum() / transition_mask.sum().clamp(min=1.0)
        predicted_slot_transition_result_loss = F.cross_entropy(
            predicted_slot_transition_result_logits.reshape(-1, TRANSITION_VALUE_CLASSES),
            value_targets[:, :, 2].reshape(-1),
            reduction="none",
        ).view_as(all_mask)
        predicted_slot_transition_result_loss = (
            predicted_slot_transition_result_loss * transition_mask
        ).sum() / transition_mask.sum().clamp(min=1.0)
        process_value_loss = F.cross_entropy(
            process_value_logits.reshape(-1, TRANSITION_VALUE_CLASSES),
            value_targets.reshape(-1),
            reduction="none",
        ).view(all_mask.size(0), all_mask.size(1), 3)
        process_value_loss = (
            process_value_loss * value_field_mask
        ).sum() / value_field_mask.sum().clamp(min=1.0)
        sign_loss = F.cross_entropy(
            sign_logits.reshape(-1, 2),
            sign_targets.reshape(-1),
            reduction="none",
        ).view_as(value_field_mask)
        sign_loss = (sign_loss * value_field_mask).sum() / value_field_mask.sum().clamp(min=1.0)
        digit_loss = F.cross_entropy(
            digit_value_logits.reshape(-1, 10),
            digit_targets.reshape(-1),
            reduction="none",
        ).view(
            all_mask.size(0),
            all_mask.size(1),
            3,
            TRANSITION_DIGITS,
        )
        digit_loss = digit_loss.mean(dim=-1)
        digit_loss = (digit_loss * value_field_mask).sum() / value_field_mask.sum().clamp(min=1.0)
        process_digit_targets, process_carry_targets, process_mask = self.process_targets(
            all_ops,
            all_values,
            all_mask,
        )
        process_digit_loss = F.cross_entropy(
            process_digit_logits.reshape(-1, 10),
            process_digit_targets.reshape(-1),
            reduction="none",
        ).view_as(process_mask)
        process_digit_loss = (
            process_digit_loss * process_mask
        ).sum() / process_mask.sum().clamp(min=1.0)
        process_carry_loss = F.cross_entropy(
            process_carry_logits.reshape(-1, LATENT_REASONING_CARRY_CLASSES),
            process_carry_targets.reshape(-1),
            reduction="none",
        ).view_as(process_mask)
        process_carry_loss = (
            process_carry_loss * process_mask
        ).sum() / process_mask.sum().clamp(min=1.0)

        op_acc = (
            (op_logits.argmax(dim=-1) == all_ops).float() * all_mask
        ).sum() / all_mask.sum().clamp(min=1.0)
        state_conditioned_op_acc = (
            (state_conditioned_op_logits.argmax(dim=-1) == all_ops).float() * all_mask
        ).sum() / all_mask.sum().clamp(min=1.0)
        slot_lhs_acc = (
            (slot_lhs_logits[:, : self.max_steps].argmax(dim=-1) == lhs_slot_targets).float()
            * slot_step_mask
        ).sum() / slot_step_mask.sum().clamp(min=1.0)
        slot_rhs_acc = (
            (slot_rhs_logits[:, : self.max_steps].argmax(dim=-1) == rhs_slot_targets).float()
            * slot_step_mask
        ).sum() / slot_step_mask.sum().clamp(min=1.0)
        slot_op_acc = (
            (slot_op_logits[:, : self.max_steps].argmax(dim=-1) == op_slot_targets).float()
            * slot_step_mask
        ).sum() / slot_step_mask.sum().clamp(min=1.0)
        slot_pair_acc = (
            (
                (slot_lhs_logits[:, : self.max_steps].argmax(dim=-1) == lhs_slot_targets)
                & (slot_rhs_logits[:, : self.max_steps].argmax(dim=-1) == rhs_slot_targets)
                & (slot_op_logits[:, : self.max_steps].argmax(dim=-1) == op_slot_targets)
            ).float()
            * slot_step_mask
        ).sum() / slot_step_mask.sum().clamp(min=1.0)
        active_acc = (active_logits.sigmoid().ge(0.5) == all_mask.bool()).float().mean()
        value_acc = (
            (value_logits.argmax(dim=-1) == value_targets).float() * value_field_mask
        ).sum() / value_field_mask.sum().clamp(min=1.0)
        final_acc = (
            value_logits[:, -1, 2].argmax(dim=-1) == value_targets[:, -1, 2]
        ).float().mean()
        state_conditioned_value_acc = (
            (state_conditioned_value_logits.argmax(dim=-1) == value_targets).float()
            * value_field_mask
        ).sum() / value_field_mask.sum().clamp(min=1.0)
        state_conditioned_final_acc = (
            state_conditioned_value_logits[:, -1, 2].argmax(dim=-1)
            == value_targets[:, -1, 2]
        ).float().mean()
        slot_result_acc = (
            (slot_result_logits.argmax(dim=-1) == value_targets[:, :, 2]).float() * all_mask
        ).sum() / all_mask.sum().clamp(min=1.0)
        slot_final_acc = (
            slot_result_logits[:, -1].argmax(dim=-1) == value_targets[:, -1, 2]
        ).float().mean()
        slot_transition_result_acc = (
            (slot_transition_result_logits.argmax(dim=-1) == value_targets[:, :, 2]).float()
            * transition_mask
        ).sum() / transition_mask.sum().clamp(min=1.0)
        final_step_index = step_mask.sum(dim=1).long().sub(1).clamp(min=0)
        slot_transition_final_pred = slot_transition_result_logits[
            torch.arange(pred.size(0), device=pred.device),
            final_step_index,
        ].argmax(dim=-1)
        slot_transition_final_acc = (
            slot_transition_final_pred == value_targets[:, -1, 2]
        ).float().mean()
        predicted_slot_transition_result_acc = (
            (
                predicted_slot_transition_result_logits.argmax(dim=-1)
                == value_targets[:, :, 2]
            ).float()
            * transition_mask
        ).sum() / transition_mask.sum().clamp(min=1.0)
        predicted_slot_transition_final_pred = predicted_slot_transition_result_logits[
            torch.arange(pred.size(0), device=pred.device),
            final_step_index,
        ].argmax(dim=-1)
        predicted_slot_transition_final_acc = (
            predicted_slot_transition_final_pred == value_targets[:, -1, 2]
        ).float().mean()
        process_value_acc = (
            (process_value_logits.argmax(dim=-1) == value_targets).float() * value_field_mask
        ).sum() / value_field_mask.sum().clamp(min=1.0)
        process_final_acc = (
            process_value_logits[:, -1, 2].argmax(dim=-1) == value_targets[:, -1, 2]
        ).float().mean()
        digit_values = self.sign_digits_to_value(
            sign_logits.argmax(dim=-1),
            digit_value_logits.argmax(dim=-1),
        )
        digit_value_acc = (
            (digit_values == all_values.round().long()).float() * value_field_mask
        ).sum() / value_field_mask.sum().clamp(min=1.0)
        digit_final_acc = (
            digit_values[:, -1, 2] == all_values[:, -1, 2].round().long()
        ).float().mean()
        process_digit_acc = (
            (process_digit_logits.argmax(dim=-1) == process_digit_targets).float()
            * process_mask
        ).sum() / process_mask.sum().clamp(min=1.0)
        process_carry_acc = (
            (process_carry_logits.argmax(dim=-1) == process_carry_targets).float()
            * process_mask
        ).sum() / process_mask.sum().clamp(min=1.0)
        rounded_state_values = self.sign_digits_to_value(
            state_value_sign_logits.argmax(dim=-1),
            state_value_digit_logits.argmax(dim=-1),
        )
        state_value_acc = (
            (rounded_state_values == state_values).float() * state_value_mask
        ).sum() / state_value_mask.sum().clamp(min=1.0)
        result_state_values = self.sign_digits_to_value(
            result_state_sign_logits.argmax(dim=-1),
            result_state_digit_logits.argmax(dim=-1),
        )
        result_state_value_acc = (
            (result_state_values == all_values[:, : self.max_steps, 2].round().long()).float()
            * slot_step_mask
        ).sum() / slot_step_mask.sum().clamp(min=1.0)
        if self.max_steps > 1:
            pre_pred_values = self.sign_digits_to_value(
                pre_sign_logits.argmax(dim=-1),
                pre_digit_logits.argmax(dim=-1),
            )
            pre_slot_operand_value_acc = (
                (pre_pred_values == pre_value_targets.round().long()).float()
                * pre_value_mask
            ).sum() / pre_value_mask.sum().clamp(min=1.0)
            pre_slot_op_acc = (
                (pre_op_logits.argmax(dim=-1) == all_ops[:, 1 : self.max_steps]).float()
                * pre_slot_mask
            ).sum() / pre_slot_mask.sum().clamp(min=1.0)
        else:
            pre_slot_operand_value_acc = state_value_acc.new_tensor(0.0)
            pre_slot_op_acc = state_value_acc.new_tensor(0.0)
        state_value_active_acc = (
            state_value_active_logits.sigmoid().ge(0.5) == state_value_mask.bool()
        ).float().mean()
        state_op_acc = (
            (state_op_logits.argmax(dim=-1) == state_ops).float() * state_op_mask
        ).sum() / state_op_mask.sum().clamp(min=1.0)
        state_op_active_acc = (
            state_op_active_logits.sigmoid().ge(0.5) == state_op_mask.bool()
        ).float().mean()
        loss = (
            latent_loss
            + 0.2 * contrastive_loss
            + 0.5 * hard_loss
            + 2.0 * state_value_loss
            + 1.5 * state_value_active_loss
            + 1.5 * state_op_loss
            + 1.5 * state_op_active_loss
            + 2.0 * result_state_value_loss
            + 2.0 * pre_slot_operand_value_loss
            + pre_slot_op_loss
            + active_loss
            + op_loss
            + state_conditioned_op_loss
            + slot_lhs_loss
            + slot_rhs_loss
            + slot_op_loss
            + value_loss
            + state_conditioned_value_loss
            + slot_result_loss
            + slot_transition_result_loss
            + predicted_slot_transition_result_loss
            + process_value_loss
            + sign_loss
            + digit_loss
            + process_digit_loss
            + process_carry_loss
        )
        return {
            "loss": loss,
            "latent_reasoning_latent_loss": latent_loss,
            "latent_reasoning_contrastive_loss": contrastive_loss,
            "latent_reasoning_hard_negative_loss": hard_loss,
            "latent_reasoning_state_value_loss": state_value_loss,
            "latent_reasoning_state_value_sign_loss": state_value_sign_loss,
            "latent_reasoning_state_value_digit_loss": state_value_digit_loss,
            "latent_reasoning_state_value_active_loss": state_value_active_loss,
            "latent_reasoning_state_op_loss": state_op_loss,
            "latent_reasoning_state_op_active_loss": state_op_active_loss,
            "latent_reasoning_result_state_value_loss": result_state_value_loss,
            "latent_reasoning_pre_slot_operand_value_loss": pre_slot_operand_value_loss,
            "latent_reasoning_pre_slot_op_loss": pre_slot_op_loss,
            "latent_reasoning_state_conditioned_op_loss": state_conditioned_op_loss,
            "latent_reasoning_state_conditioned_value_loss": state_conditioned_value_loss,
            "latent_reasoning_slot_lhs_loss": slot_lhs_loss,
            "latent_reasoning_slot_rhs_loss": slot_rhs_loss,
            "latent_reasoning_slot_op_loss": slot_op_loss,
            "latent_reasoning_slot_result_loss": slot_result_loss,
            "latent_reasoning_slot_transition_result_loss": slot_transition_result_loss,
            "latent_reasoning_predicted_slot_transition_result_loss": (
                predicted_slot_transition_result_loss
            ),
            "latent_reasoning_active_loss": active_loss,
            "latent_reasoning_op_loss": op_loss,
            "latent_reasoning_value_loss": value_loss,
            "latent_reasoning_process_value_loss": process_value_loss,
            "latent_reasoning_digit_sign_loss": sign_loss,
            "latent_reasoning_digit_value_loss": digit_loss,
            "latent_reasoning_process_digit_loss": process_digit_loss,
            "latent_reasoning_process_carry_loss": process_carry_loss,
            "latent_reasoning_cosine": cosine_acc,
            "latent_reasoning_active_acc": active_acc,
            "latent_reasoning_op_acc": op_acc,
            "latent_reasoning_state_conditioned_op_acc": state_conditioned_op_acc,
            "latent_reasoning_slot_lhs_acc": slot_lhs_acc,
            "latent_reasoning_slot_rhs_acc": slot_rhs_acc,
            "latent_reasoning_slot_op_acc": slot_op_acc,
            "latent_reasoning_slot_pair_acc": slot_pair_acc,
            "latent_reasoning_slot_result_acc": slot_result_acc,
            "latent_reasoning_slot_final_acc": slot_final_acc,
            "latent_reasoning_slot_transition_result_acc": slot_transition_result_acc,
            "latent_reasoning_slot_transition_final_acc": slot_transition_final_acc,
            "latent_reasoning_predicted_slot_transition_result_acc": (
                predicted_slot_transition_result_acc
            ),
            "latent_reasoning_predicted_slot_transition_final_acc": (
                predicted_slot_transition_final_acc
            ),
            "latent_reasoning_value_acc": value_acc,
            "latent_reasoning_final_acc": final_acc,
            "latent_reasoning_state_conditioned_value_acc": state_conditioned_value_acc,
            "latent_reasoning_state_conditioned_final_acc": state_conditioned_final_acc,
            "latent_reasoning_process_value_acc": process_value_acc,
            "latent_reasoning_process_final_acc": process_final_acc,
            "latent_reasoning_digit_value_acc": digit_value_acc,
            "latent_reasoning_digit_final_acc": digit_final_acc,
            "latent_reasoning_process_digit_acc": process_digit_acc,
            "latent_reasoning_process_carry_acc": process_carry_acc,
            "latent_reasoning_state_value_acc": state_value_acc,
            "latent_reasoning_state_value_active_acc": state_value_active_acc,
            "latent_reasoning_state_op_acc": state_op_acc,
            "latent_reasoning_state_op_active_acc": state_op_active_acc,
            "latent_reasoning_result_state_value_acc": result_state_value_acc,
            "latent_reasoning_pre_slot_operand_value_acc": pre_slot_operand_value_acc,
            "latent_reasoning_pre_slot_op_acc": pre_slot_op_acc,
        }

    @torch.no_grad()
    def predict_structured(
        self,
        math_ids: torch.Tensor,
        value_mode: str = "class",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self.predictor(math_ids)
        (
            active_logits,
            op_logits,
            value_logits,
            sign_logits,
            digit_value_logits,
            _process_digit_logits,
            _process_carry_logits,
            process_value_logits,
        ) = self._decode(pred)
        (
            state_value_logits,
            state_value_active_logits,
            state_op_logits,
            state_op_active_logits,
        ) = self._state_logits(pred)
        state_conditioned_op_logits, state_conditioned_value_logits = (
            self._state_conditioned_decode(
                pred,
                state_value_logits,
                state_value_active_logits,
                state_op_logits,
                state_op_active_logits,
            )
        )
        slot_lhs_logits, slot_rhs_logits, slot_op_logits, slot_result_logits = (
            self._slot_decode(
                pred,
                math_ids,
                state_value_logits,
                state_value_active_logits,
                state_op_logits,
                state_op_active_logits,
            )
        )
        if value_mode == "class":
            values = ReasoningStepTargetEncoder.class_to_value(value_logits.argmax(dim=-1))
            decoded_op_logits = op_logits
        elif value_mode == "process":
            values = ReasoningStepTargetEncoder.class_to_value(
                process_value_logits.argmax(dim=-1)
            )
            decoded_op_logits = op_logits
        elif value_mode == "state":
            values = ReasoningStepTargetEncoder.class_to_value(
                state_conditioned_value_logits.argmax(dim=-1)
            )
            decoded_op_logits = state_conditioned_op_logits
        elif value_mode == "slot":
            pre_values, pre_ops = self._pre_state_values_ops(
                math_ids,
                state_value_logits,
                state_op_logits,
            )
            lhs_slots = slot_lhs_logits.argmax(dim=-1)
            rhs_slots = slot_rhs_logits.argmax(dim=-1)
            op_slots = slot_op_logits.argmax(dim=-1)
            lhs_values = pre_values.gather(dim=2, index=lhs_slots.unsqueeze(-1)).squeeze(-1)
            rhs_values = pre_values.gather(dim=2, index=rhs_slots.unsqueeze(-1)).squeeze(-1)
            result_values = ReasoningStepTargetEncoder.class_to_value(
                slot_result_logits.argmax(dim=-1)
            )
            values = torch.stack([lhs_values, rhs_values, result_values], dim=-1)
            op_ids = pre_ops.gather(dim=2, index=op_slots.unsqueeze(-1)).squeeze(-1)
            return active_logits.sigmoid().ge(0.5), op_ids, values
        elif value_mode == "slot_transition":
            pre_values, pre_ops = self._pre_state_values_ops(
                math_ids,
                state_value_logits,
                state_op_logits,
            )
            lhs_slots = slot_lhs_logits.argmax(dim=-1)
            rhs_slots = slot_rhs_logits.argmax(dim=-1)
            op_slots = slot_op_logits.argmax(dim=-1)
            lhs_values = pre_values.gather(dim=2, index=lhs_slots.unsqueeze(-1)).squeeze(-1)
            rhs_values = pre_values.gather(dim=2, index=rhs_slots.unsqueeze(-1)).squeeze(-1)
            op_ids = pre_ops.gather(dim=2, index=op_slots.unsqueeze(-1)).squeeze(-1)
            slot_transition_logits = self._slot_transition_result_logits(
                pred,
                lhs_values.float(),
                rhs_values.float(),
                op_ids,
            )
            result_values = ReasoningStepTargetEncoder.class_to_value(
                slot_transition_logits.argmax(dim=-1)
            )
            active = active_logits.sigmoid().ge(0.5)
            final_index = active[:, :-1].float().sum(dim=1).long().sub(1).clamp(min=0)
            result_values[:, -1] = result_values[
                torch.arange(result_values.size(0), device=result_values.device),
                final_index,
            ]
            values = torch.stack([lhs_values, rhs_values, result_values], dim=-1)
            return active, op_ids, values
        elif value_mode in {"slot_class", "slot_process"}:
            pre_values, pre_ops = self._pre_state_values_ops(
                math_ids,
                state_value_logits,
                state_op_logits,
            )
            lhs_slots = slot_lhs_logits.argmax(dim=-1)
            rhs_slots = slot_rhs_logits.argmax(dim=-1)
            op_slots = slot_op_logits.argmax(dim=-1)
            lhs_values = pre_values.gather(dim=2, index=lhs_slots.unsqueeze(-1)).squeeze(-1)
            rhs_values = pre_values.gather(dim=2, index=rhs_slots.unsqueeze(-1)).squeeze(-1)
            op_ids = pre_ops.gather(dim=2, index=op_slots.unsqueeze(-1)).squeeze(-1)
            result_logits = (
                process_value_logits[:, :, 2]
                if value_mode == "slot_process"
                else value_logits[:, :, 2]
            )
            result_values = ReasoningStepTargetEncoder.class_to_value(
                result_logits.argmax(dim=-1)
            )
            values = torch.stack([lhs_values, rhs_values, result_values], dim=-1)
            return active_logits.sigmoid().ge(0.5), op_ids, values
        elif value_mode == "digit":
            values = self.sign_digits_to_value(
                sign_logits.argmax(dim=-1),
                digit_value_logits.argmax(dim=-1),
            )
            decoded_op_logits = op_logits
        else:
            raise ValueError(f"Unknown latent reasoning value_mode: {value_mode}")
        return active_logits.sigmoid().ge(0.5), decoded_op_logits.argmax(dim=-1), values


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
        self.latent_reasoning_sequence = LatentReasoningSequence(
            math_vocab_size,
            d_model,
            max_math_len,
            max_steps=max_variable_steps,
            num_heads=num_heads,
        )
        self.standalone_transition = StandaloneArithmeticTransition(d_model)
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

    def latent_reasoning_sequence_loss(
        self,
        math_ids: torch.Tensor,
        variable_trace_op_ids: torch.Tensor,
        variable_trace_position_ids: torch.Tensor,
        variable_trace_values: torch.Tensor,
        variable_trace_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.latent_reasoning_sequence.loss(
            math_ids,
            variable_trace_op_ids,
            variable_trace_position_ids,
            variable_trace_values,
            variable_trace_mask,
        )

    def standalone_transition_loss(
        self,
        variable_trace_op_ids: torch.Tensor,
        variable_trace_values: torch.Tensor,
        variable_trace_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.standalone_transition.loss(
            variable_trace_values[:, :, 0],
            variable_trace_op_ids,
            variable_trace_values[:, :, 1],
            variable_trace_values[:, :, 2],
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

    @classmethod
    def _render_latent_reasoning_row(
        cls,
        active: list[bool],
        op_ids: list[int],
        value_rows: list[list[int]],
    ) -> str:
        parts = []
        for is_active, op_id, values in zip(active[:-1], op_ids[:-1], value_rows[:-1]):
            if not is_active:
                continue
            op = cls._trace_op_to_text(op_id)
            if not op:
                continue
            lhs, rhs, result = values
            parts.append(f"{lhs}{op}{rhs}={result}")
        final = value_rows[-1][2]
        parts.append(str(final))
        return ",".join(parts)

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
        standalone_learned_values: bool = False,
        standalone_digit_values: bool = False,
        standalone_factor_values: bool = False,
        standalone_decomposed_values: bool = False,
        standalone_hybrid_values: bool = False,
        latent_reasoning_values: bool = False,
        latent_reasoning_digit_values: bool = False,
        latent_reasoning_process_values: bool = False,
        latent_reasoning_state_values: bool = False,
        latent_reasoning_slot_values: bool = False,
        latent_reasoning_slot_transition_values: bool = False,
        latent_reasoning_slot_class_values: bool = False,
        latent_reasoning_slot_process_values: bool = False,
    ) -> list[str]:
        if (
            latent_reasoning_values
            or latent_reasoning_digit_values
            or latent_reasoning_process_values
            or latent_reasoning_state_values
            or latent_reasoning_slot_values
            or latent_reasoning_slot_transition_values
            or latent_reasoning_slot_class_values
            or latent_reasoning_slot_process_values
        ):
            value_mode = (
                "slot_process"
                if latent_reasoning_slot_process_values
                else
                "slot_class"
                if latent_reasoning_slot_class_values
                else
                "slot_transition"
                if latent_reasoning_slot_transition_values
                else
                "slot"
                if latent_reasoning_slot_values
                else
                "state"
                if latent_reasoning_state_values
                else
                "process"
                if latent_reasoning_process_values
                else "digit"
                if latent_reasoning_digit_values
                else "class"
            )
            active, op_ids, values = self.latent_reasoning_sequence.predict_structured(
                math_ids,
                value_mode=value_mode,
            )
            active_rows = active.detach().cpu().tolist()
            op_rows = op_ids.detach().cpu().tolist()
            value_rows = values.detach().cpu().tolist()
            return [
                self._render_latent_reasoning_row(active_row, op_row, value_row)
                for active_row, op_row, value_row in zip(active_rows, op_rows, value_rows)
            ]
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
                if (
                    standalone_learned_values
                    or standalone_digit_values
                    or standalone_factor_values
                    or standalone_decomposed_values
                    or standalone_hybrid_values
                ):
                    lhs_tensor = torch.tensor([lhs], device=math_ids.device)
                    rhs_tensor = torch.tensor([rhs], device=math_ids.device)
                    standalone_result = self.standalone_transition.predict_value(
                        lhs_tensor,
                        torch.tensor([op_id], device=math_ids.device),
                        rhs_tensor,
                        mode=(
                            "digit"
                            if standalone_digit_values
                            else "factor"
                            if standalone_factor_values
                            else "decomposed"
                            if standalone_decomposed_values
                            else "hybrid"
                            if standalone_hybrid_values
                            else "class"
                        ),
                    )
                    result = int(standalone_result[0].item())
                elif class_learned_values:
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
