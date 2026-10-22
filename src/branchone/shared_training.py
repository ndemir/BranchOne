"""Experimental differentiable prefix reuse; acceptance requires gradient parity."""

import contextlib

import torch
from torch.utils.checkpoint import checkpoint
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer, LinearAttentionLayer


class GradLinearLayer(LinearAttentionLayer):
    def update_conv_state(
        self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs
    ):
        if not self.is_conv_states_initialized[state_idx]:
            self.device = conv_states.device
            self.dtype = conv_states.dtype
            self.conv_kernel_size[state_idx] = conv_kernel_size or conv_states.shape[-1]
            self.is_conv_states_initialized[state_idx] = True
        full = (
            torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
            if self.has_previous_state[state_idx]
            else conv_states
        )
        size = self.conv_kernel_size[state_idx]
        if full.shape[-1] < size:
            full = torch.nn.functional.pad(full, (size - full.shape[-1], 0))
        self.conv_states[state_idx] = full[..., -size:].clone()
        self.has_previous_state[state_idx] = True
        return full

    def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
        # Assignment preserves autograd history without mutating an initial state saved for backward.
        self.recurrent_states[state_idx] = recurrent_states
        self.is_recurrent_states_initialized[state_idx] = True
        return recurrent_states


def empty_cache(m):
    cache = DynamicCache(config=m.backbone.config.text_config)
    for i, layer in enumerate(cache.layers):
        if isinstance(layer, LinearAttentionLayer):
            assert not isinstance(layer, DynamicLayer)
            cache.layers[i] = GradLinearLayer()
    return cache


def flatten(cache):
    values = []
    for layer in cache.layers:
        if isinstance(layer, GradLinearLayer):
            values.extend([layer.conv_states[0], layer.recurrent_states[0]])
        else:
            values.extend([layer.keys, layer.values])
    return tuple(values)


def restore_cache(m, values):
    cache = empty_cache(m)
    for i, layer in enumerate(cache.layers):
        a, b = values[2 * i : 2 * i + 2]
        layer.device = a.device
        layer.dtype = a.dtype
        if isinstance(layer, GradLinearLayer):
            layer.conv_states[0] = a
            layer.recurrent_states[0] = b
            layer.conv_kernel_size[0] = a.shape[-1]
            layer.is_conv_states_initialized[0] = True
            layer.is_recurrent_states_initialized[0] = True
            layer.has_previous_state[0] = True
        else:
            layer.keys = a
            layer.values = b
            layer.is_initialized = True
    return cache


@contextlib.contextmanager
def cache_enabled(m):
    flags = [
        (x, x.gradient_checkpointing)
        for x in m.backbone.language_model.modules()
        if hasattr(x, "gradient_checkpointing")
    ]
    for x, v in flags:
        x.gradient_checkpointing = False
    try:
        yield
    finally:
        for x, v in flags:
            x.gradient_checkpointing = v


def shared_logits(m, row):
    p = m.prepared_row(row)
    v = m.visual_features(p)
    # Match the canonical DeltaNet64-token chunks. Move at most63 common tokens
    # into each independent continuation; no input tokens are changed or dropped.
    cut = (len(p["prefix"]) // 64) * 64
    if (
        cut
        and p["prefix"][:cut].count(m.backbone.config.image_token_id)
        == p["visual_tokens"]
    ):
        tail = p["prefix"][cut:]
        p = {
            **p,
            "prefix": p["prefix"][:cut],
            "suffixes": [tail + s for s in p["suffixes"]],
        }
    inputs, _ = m._inputs([p["prefix"]], p, v, pad_to_bucket=False)
    prefix_length = len(p["prefix"])
    k = len(p["suffixes"])
    assert min(map(len, p["suffixes"])) > 1

    def encode_prefix(embeds, mask, positions):
        cache = empty_cache(m)
        with cache_enabled(m), torch.autocast("cuda", dtype=torch.float16):
            m.backbone.language_model(
                inputs_embeds=embeds,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=cache,
                use_cache=True,
            )
        return flatten(cache)

    if torch.is_grad_enabled() and m.training:
        values = checkpoint(
            encode_prefix,
            inputs["inputs_embeds"],
            inputs["attention_mask"],
            inputs["position_ids"],
            use_reentrant=False,
        )
    else:
        values = encode_prefix(
            inputs["inputs_embeds"], inputs["attention_mask"], inputs["position_ids"]
        )
    assert all(torch.isfinite(x).all() for x in values)
    if torch.is_grad_enabled():
        assert all(x.requires_grad for x in values), "Prefix history detached"
    # Each branch has separate tensor storage; repeat_interleave retains gradients to the shared prefix.
    cache = restore_cache(m, tuple(x.repeat_interleave(k, dim=0) for x in values))
    assert cache.get_seq_length() == prefix_length
    lengths = torch.tensor(list(map(len, p["suffixes"])), device=m.device)
    width = ((int(lengths.max()) + 63) // 64) * 64
    ids = torch.full(
        (k, width), m.tokenizer.pad_token_id, dtype=torch.long, device=m.device
    )
    for i, suffix in enumerate(p["suffixes"]):
        ids[i, : len(suffix)] = torch.tensor(suffix, device=m.device)
    valid = torch.arange(width, device=m.device)[None] < lengths[:, None]
    mask = torch.cat(
        [torch.ones((k, prefix_length), dtype=torch.bool, device=m.device), valid],
        dim=1,
    )
    start = int(inputs["position_ids"].max()) + 1
    positions = (
        torch.arange(start, start + width, device=m.device)[None]
        .expand(k, -1)
        .masked_fill(~valid, 0)
        .unsqueeze(0)
        .expand(3, -1, -1)
    )
    with cache_enabled(m), torch.autocast("cuda", dtype=torch.float16):
        out = m.backbone.language_model(
            inputs_embeds=m.backbone.get_input_embeddings()(ids),
            attention_mask=mask.long(),
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
        ).last_hidden_state
    h = out[torch.arange(k, device=m.device), lengths - 1].float()
    with torch.autocast("cuda", enabled=False):
        return m.head(h[None], torch.ones((1, k), dtype=torch.bool, device=m.device))[0]
