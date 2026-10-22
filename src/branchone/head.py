"""Shared candidate scorer; checkpoint parameter names are preserved."""

from torch import nn

TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
]


class CandidateHead(nn.Module):
    def __init__(self, hidden=1024, width=256, comparison=False):
        super().__init__()
        self.width = width
        self.comparison = comparison
        # Common layers are initialized in the same order for both variants.
        self.norm = nn.LayerNorm(hidden)
        self.project = nn.Linear(hidden, width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )
        self.final_norm = nn.LayerNorm(width)
        self.scorer = nn.Linear(width, 1)
        if comparison:
            self.set_norm = nn.LayerNorm(width)
            self.attention = nn.MultiheadAttention(
                width, 4, dropout=0.0, batch_first=True
            )
            # Start as the independent head; learn interactions without perturbing its initial logits.
            nn.init.zeros_(self.attention.out_proj.weight)
            nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, h, valid, return_features=False):
        u = self.project(self.norm(h.float()))
        if self.comparison:
            q = self.set_norm(u)
            delta, _ = self.attention(
                q, q, q, key_padding_mask=~valid, need_weights=False
            )
            u = u + delta
        u = u + self.ff(self.ff_norm(u))
        u = self.final_norm(u)
        z = self.scorer(u).squeeze(-1).masked_fill(~valid, -1e9)
        return (z, u) if return_features else z
