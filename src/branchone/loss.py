"""Numerically stable single-label choice training loss."""

import torch


def choice_cross_entropy(logits, targets):
    """Return summed cross-entropy while preserving confident-target gradients.

    Factor out the target exponential before evaluating the logarithm. This
    avoids subtracting a probability rounded to one in the target derivative.
    Padded candidates may have negative-infinite or sufficiently negative scores.
    """
    logits = logits.float()
    differences = logits - logits.gather(1, targets[:, None])
    target_mask = torch.nn.functional.one_hot(targets, logits.shape[-1]).bool()
    others = torch.logsumexp(differences.masked_fill(target_mask, -torch.inf), dim=-1)
    return torch.logaddexp(torch.zeros_like(others), others).sum()
