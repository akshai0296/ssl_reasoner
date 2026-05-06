from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def latent_health(model, dataset, device, batch_size: int = 64) -> dict[str, float]:
    """Plan gate: predicted latents should not collapse."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    slots = model.predict_slots(batch["problem_ids"].to(device))
    pooled = slots.mean(dim=1)

    normed = F.normalize(pooled, dim=-1)
    cosine = normed @ normed.T
    if cosine.size(0) > 1:
        off_diag = cosine[~torch.eye(cosine.size(0), dtype=torch.bool, device=cosine.device)]
        mean_random_cosine = off_diag.mean().item()
    else:
        mean_random_cosine = 1.0

    centered = pooled - pooled.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered.float())
    threshold = singular_values.max().clamp(min=1e-8) * 1e-3
    effective_rank = (singular_values > threshold).sum().item()
    max_rank = min(centered.shape)
    effective_rank_ratio = effective_rank / max(max_rank, 1)

    return {
        "mean_random_cosine": float(mean_random_cosine),
        "effective_rank": float(effective_rank),
        "effective_rank_ratio": float(effective_rank_ratio),
        "passes_cosine": float(mean_random_cosine < 0.3),
        "passes_rank": float(effective_rank_ratio > 0.6),
    }
