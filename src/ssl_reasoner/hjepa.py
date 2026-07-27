"""Hierarchical JEPA (H-JEPA) reasoning level.

The existing :class:`~ssl_reasoner.model.LatentReasoningSequence` predicts a flat
sequence of latent reasoning states ``z_1 -> z_2 -> ... -> z_final`` at a single
abstraction level. This module adds an *abstract plan level* (L2) on top of that
state level (L1):

* :class:`PlanTargetEncoder` encodes the ground-truth reduction plan -- the ordered
  ``(operator, reduced-position)`` decisions -- into one normalized plan latent.
  It deliberately ignores the operand *values*: the plan captures precedence /
  reduction order, not arithmetic. This is the "slow", coarse level.
* :class:`PlanPredictor` predicts that plan latent from the math features alone, and
  also emits a coarse decode (per-step operator + active/stop) so the plan is
  measurable.
* :class:`HJEPAReasoner` wraps an existing ``LatentReasoningSequence`` instance,
  trains the plan level with its own JEPA (latent-matching + InfoNCE) objective, and
  feeds the predicted plan latent into the L1 predictor as top-down conditioning via
  the ``plan_latent=`` hook on ``LatentReasoningSequencePredictor.forward``.

The reasoner owns only the *new* plan parameters; the wrapped L1 sequence is held by
reference (in a tuple) so it is not re-registered as a duplicate submodule when both
live inside ``MathJEPAReadout``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Kept local so this module never imports from .model at top level (avoids any
# import cycle with model.py, which constructs HJEPAReasoner lazily).
DEFAULT_PLAN_TEMPERATURE = 0.07


class PlanTargetEncoder(nn.Module):
    """Encode the ground-truth reduction plan into one normalized latent.

    Inputs are the per-step reduction decisions; operand values are intentionally
    excluded so the plan latent stays a pure abstraction of computation order.
    """

    def __init__(self, d_model: int, max_steps: int):
        super().__init__()
        self.max_steps = max_steps
        self.op_embed = nn.Embedding(4, d_model)
        self.position_embed = nn.Embedding(max_steps + 1, d_model)
        self.step_embed = nn.Embedding(max_steps, d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        op_ids: torch.Tensor,
        position_ids: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps = op_ids.shape
        step_ids = torch.arange(steps, device=op_ids.device).unsqueeze(0).expand(batch, -1)
        encoded = (
            self.op_embed(op_ids.clamp(min=0, max=3).long())
            + self.position_embed(position_ids.clamp(min=0, max=self.max_steps).long())
            + self.step_embed(step_ids.clamp(max=self.max_steps - 1))
        )
        mask = step_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(self.net(pooled), dim=-1)


class PlanPredictor(nn.Module):
    """Predict the plan latent (and a coarse decode) from math features alone."""

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
        # max_steps step-queries + one plan-summary query.
        self.query_embed = nn.Parameter(torch.randn(1, max_steps + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.op_head = nn.Linear(d_model, 4)
        self.active_head = nn.Linear(d_model, 1)
        self.plan_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self, math_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = math_ids.size(0)
        math_tokens = self.token_embed(math_ids) + self.math_pos_embed[:, : math_ids.size(1)]
        queries = self.query_embed.expand(batch, -1, -1)
        tokens = torch.cat([math_tokens, queries], dim=1)
        math_mask = math_ids.ne(0)
        query_mask = torch.ones(
            batch, self.max_steps + 1, device=math_ids.device, dtype=torch.bool
        )
        mask = torch.cat([math_mask, query_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~mask)
        query_context = encoded[:, -(self.max_steps + 1) :]
        step_context = query_context[:, : self.max_steps]
        summary = query_context[:, -1]
        op_logits = self.op_head(step_context)
        active_logits = self.active_head(step_context).squeeze(-1)
        plan_latent = F.normalize(self.plan_out(summary), dim=-1)
        return plan_latent, op_logits, active_logits


class HJEPAReasoner(nn.Module):
    """Add an L2 plan level on top of an existing L1 latent reasoning sequence."""

    def __init__(
        self,
        latent_reasoning_sequence: nn.Module,
        math_vocab_size: int,
        d_model: int,
        max_math_len: int,
        max_steps: int,
        num_heads: int = 4,
        temperature: float = DEFAULT_PLAN_TEMPERATURE,
        condition_detached: bool = True,
    ):
        super().__init__()
        # Held by reference (not a registered submodule) to avoid duplicating the L1
        # parameters, which MathJEPAReadout already owns as .latent_reasoning_sequence.
        self._l1 = (latent_reasoning_sequence,)
        self.temperature = temperature
        self.condition_detached = condition_detached
        self.plan_target_encoder = PlanTargetEncoder(d_model, max_steps)
        self.plan_predictor = PlanPredictor(
            math_vocab_size, d_model, max_math_len, max_steps, num_heads=num_heads
        )
        # Projects the plan latent into the L1 query space before conditioning.
        self.plan_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    @property
    def l1(self) -> nn.Module:
        return self._l1[0]

    def forward(
        self, math_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.plan_predictor(math_ids)

    def plan_conditioning(self, math_ids: torch.Tensor) -> torch.Tensor:
        """Predicted plan latent, projected into the L1 conditioning space."""
        plan_latent, _, _ = self.plan_predictor(math_ids)
        return self.plan_proj(plan_latent)

    def loss(
        self,
        math_ids: torch.Tensor,
        op_ids: torch.Tensor,
        position_ids: torch.Tensor,
        values: torch.Tensor,
        step_mask: torch.Tensor,
        *,
        plan_weight: float = 1.0,
        plan_contrastive_weight: float = 0.5,
        plan_op_weight: float = 1.0,
        plan_active_weight: float = 0.5,
        l1_kwargs: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        l1_kwargs = dict(l1_kwargs or {})

        # ---- L2: plan level ------------------------------------------------
        plan_target = self.plan_target_encoder(op_ids, position_ids, step_mask).detach()
        plan_latent, plan_op_logits, plan_active_logits = self.plan_predictor(math_ids)

        plan_latent_loss = F.smooth_l1_loss(plan_latent, plan_target)
        plan_cosine = (plan_latent * plan_target).sum(dim=-1).mean()

        logits = plan_latent @ plan_target.t() / self.temperature
        labels = torch.arange(plan_latent.size(0), device=plan_latent.device)
        plan_contrastive_loss = F.cross_entropy(logits, labels)

        flat_mask = step_mask.reshape(-1)
        plan_op_loss = F.cross_entropy(
            plan_op_logits.reshape(-1, 4), op_ids.reshape(-1), reduction="none"
        )
        plan_op_loss = (plan_op_loss * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)
        plan_op_acc = (
            (plan_op_logits.argmax(dim=-1) == op_ids).float() * step_mask
        ).sum() / step_mask.sum().clamp(min=1.0)

        plan_active_loss = F.binary_cross_entropy_with_logits(
            plan_active_logits, step_mask.to(plan_active_logits.dtype)
        )
        plan_active_acc = (
            (plan_active_logits.sigmoid().ge(0.5).float() == step_mask).float().mean()
        )

        plan_loss = (
            plan_weight * plan_latent_loss
            + plan_contrastive_weight * plan_contrastive_loss
            + plan_op_weight * plan_op_loss
            + plan_active_weight * plan_active_loss
        )

        # ---- L1: state level, conditioned top-down on the plan -------------
        conditioning = self.plan_proj(plan_latent)
        if self.condition_detached:
            conditioning = conditioning.detach()
        l1_out = self.l1.loss(
            math_ids,
            op_ids,
            position_ids,
            values,
            step_mask,
            plan_latent=conditioning,
            **l1_kwargs,
        )

        out = dict(l1_out)
        out["loss"] = l1_out["loss"] + plan_loss
        out["hjepa_plan_loss"] = plan_loss
        out["hjepa_plan_latent_loss"] = plan_latent_loss
        out["hjepa_plan_contrastive_loss"] = plan_contrastive_loss
        out["hjepa_plan_op_loss"] = plan_op_loss
        out["hjepa_plan_active_loss"] = plan_active_loss
        out["hjepa_plan_cosine"] = plan_cosine
        out["hjepa_plan_op_acc"] = plan_op_acc
        out["hjepa_plan_active_acc"] = plan_active_acc
        return out
