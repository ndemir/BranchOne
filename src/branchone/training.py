"""FP16 training mathematics, with explicit runtime paths."""

import collections
import hashlib
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import get_peft_model_state_dict

from .io import append_jsonl, request_from_row
from .loss import choice_cross_entropy
from .model import BranchOne

# Allow the 2**24 initial scale to back off below one while replaying
# the same batch. Persistent nonfinite gradients still terminate the stage.
INITIAL_LOSS_SCALE = 16777216.0
AMP_MAX_RETRIES = 32


def settings(seed=42):
    # Seeds and restored RNG alone do not make CUDA backward reductions exact.
    # Configure cuBLAS before the first matrix operation, and reject conflicting
    # process settings rather than silently bypassing the restart contract.
    for key, value in {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CAUSAL_CONV1D_DETERMINISTIC": "1",
    }.items():
        if os.environ.setdefault(key, value) != value:
            raise ValueError(f"Reproducible training requires {key}={value}")
    torch.set_num_threads(4)
    # PyTorch permits setting this pool only once. Repeated API calls may reuse
    # an already correct pool; an incompatible initialized pool must still fail.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_hash(items):
    h = hashlib.sha256()
    for n, t in sorted(items):
        h.update(n.encode())
        h.update(str(t.dtype).encode())
        h.update(str(list(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def learned_hash(m):
    return tensor_hash(
        list(get_peft_model_state_dict(m.backbone.language_model).items())
        + [("head." + n, p) for n, p in m.head.state_dict().items()]
    )


def frozen_hash(m):
    return tensor_hash(
        [(n, p) for n, p in m.backbone.named_parameters() if "lora_" not in n]
    )


def parameters(m):
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    assert named and all("lora_" in n or n.startswith("head.") for n, p in named)
    assert all(p.dtype == torch.float32 for n, p in named)
    assert all(not p.requires_grad for p in m.backbone.visual.parameters())
    return [p for n, p in named if "lora_" in n], list(m.head.parameters())


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])


def permute(row, rng):
    order = list(range(len(row["candidates"])))
    rng.shuffle(order)
    return {
        **row,
        "candidates": [row["candidates"][i] for i in order],
        "suffix_ids": [row["suffix_ids"][i] for i in order],
        "branch_lengths": [row["branch_lengths"][i] for i in order],
        "gold_index": order.index(row["gold_index"]),
    }


class TrainModel(BranchOne):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_store = collections.OrderedDict()
        self.feature_dir = None
        self.image_root = None
        self.overflow_log = None

    def configure_runtime(self, *, image_root, cache_dir=None, overflow_log=None):
        self.image_root = Path(image_root).resolve()
        self.feature_dir = Path(cache_dir).resolve() if cache_dir is not None else None
        if self.feature_dir is not None:
            self.feature_dir.mkdir(parents=True, exist_ok=True)
        self.overflow_log = Path(overflow_log) if overflow_log is not None else None

    def prepared_row(self, row):
        if row.get("image"):
            prepared = self.prepare(request_from_row(row, self.image_root))
            assert (
                prepared["prefix"] == row["prefix_ids"]
                and prepared["suffixes"] == row["suffix_ids"]
            )
        else:
            prepared = {
                "paths": [row["prefix_ids"] + s for s in row["suffix_ids"]],
                "prefix": row["prefix_ids"],
                "suffixes": row["suffix_ids"],
                "image_hash": None,
                "pixel_values": None,
                "image_grid_thw": None,
                "token_counts": row["branch_lengths"],
                "visual_tokens": 0,
            }
        assert max(prepared["token_counts"]) <= self.max_length
        return prepared

    @torch.no_grad()
    def visual_features(self, prepared, reuse=True):
        if not prepared["image_hash"]:
            return None
        key = prepared["image_hash"]
        dest = (
            self.feature_dir / (key + ".pt") if self.feature_dir is not None else None
        )
        if reuse and key in self.feature_store:
            self.feature_store.move_to_end(key)
            return self.feature_store[key].to(self.device)
        if reuse and dest is not None and dest.exists():
            value = torch.load(dest, map_location="cpu", weights_only=True).to(
                self.device
            )
        else:
            value = super().visual_features(prepared, reuse=False)
            assert value.dtype == torch.float16 and bool(torch.isfinite(value).all())
            if reuse and dest is not None:
                torch.save(value.cpu(), dest)
        if reuse:
            self._lru_put(self.feature_store, key, value.cpu(), limit=4096)
        return value

    def chunks(self, rows, max_rows=8, token_budget=16384):
        prepared = [self.prepared_row(r) for r in rows]
        visual = [self.visual_features(p) for p in prepared]
        buckets = {}
        offset = 0
        for b, p in enumerate(prepared):
            for k, path in enumerate(p["paths"]):
                width = ((len(path) + 63) // 64) * 64
                buckets.setdefault(width, []).append((offset, b, k))
                offset += 1
        for width, indices in sorted(buckets.items()):
            bs = max(1, min(max_rows, token_budget // width))
            for start in range(0, len(indices), bs):
                group = indices[start : start + bs]
                inputs = []
                lengths = []
                for flat, b, k in group:
                    x, row_lengths = self._inputs(
                        [prepared[b]["paths"][k]], prepared[b], visual[b]
                    )
                    inputs.append(x)
                    lengths.append(row_lengths)
                pos = []
                for x in inputs:
                    p = x["position_ids"]
                    pos.append(p if p.ndim == 3 else p.unsqueeze(0).expand(3, -1, -1))
                merged = {
                    key: torch.cat([x[key] for x in inputs], dim=0)
                    for key in ["inputs_embeds", "attention_mask"]
                }
                merged["position_ids"] = torch.cat(pos, dim=1)
                yield [x[0] for x in group], merged, torch.cat(lengths)

    def encode_chunk(self, inputs, lengths):
        with torch.autocast("cuda", dtype=torch.float16):
            h = self.backbone.language_model(
                **inputs, use_cache=False
            ).last_hidden_state
        return h[torch.arange(len(lengths), device=self.device), lengths - 1].float()

    def score(self, hidden, rows):
        kmax = max(len(r["candidates"]) for r in rows)
        hs = []
        masks = []
        offset = 0
        for r in rows:
            k = len(r["candidates"])
            hs.append(
                torch.nn.functional.pad(
                    hidden[offset : offset + k], (0, 0, 0, kmax - k)
                )
            )
            masks.append([True] * k + [False] * (kmax - k))
            offset += k
        with torch.autocast("cuda", enabled=False):
            return self.head(torch.stack(hs), torch.tensor(masks, device=self.device))

    def independent_rows(self, rows, max_rows=8, token_budget=16384):
        h = [None] * sum(len(r["candidates"]) for r in rows)
        for indices, inputs, lengths in self.chunks(rows, max_rows, token_budget):
            for i, v in zip(indices, self.encode_chunk(inputs, lengths)):
                h[i] = v
        return self.score(torch.stack(h), rows)

    def forward_rows(self, rows, max_rows=8, token_budget=16384):
        threshold = getattr(self, "shared_prefix_min_tokens", None)
        if not self.training or threshold is None:
            return self.independent_rows(rows, max_rows, token_budget)
        from .shared_training import shared_logits

        result = [None] * len(rows)
        plain = []
        for i, row in enumerate(rows):
            if len(row["prefix_ids"]) >= threshold:
                result[i] = shared_logits(self, row)
            else:
                plain.append(i)
        if plain:
            z = self.independent_rows([rows[i] for i in plain], max_rows, token_budget)
            for j, i in enumerate(plain):
                result[i] = z[j, : len(rows[i]["candidates"])]
        width = max(map(len, result))
        return torch.stack(
            [
                torch.nn.functional.pad(z, (0, width - len(z)), value=-1e9)
                for z in result
            ]
        )


def optimizer(m, total):
    a, h = parameters(m)
    opt = torch.optim.AdamW(
        [{"params": a, "lr": 1e-4}, {"params": h, "lr": 3e-4}], weight_decay=0.01
    )
    warmup = max(1, math.ceil(total * 0.05))

    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.1 + 0.9 * 0.5 * (
            1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup)))
        )

    return (
        opt,
        torch.optim.lr_scheduler.LambdaLR(opt, scale),
        torch.amp.GradScaler("cuda", init_scale=INITIAL_LOSS_SCALE),
    )


def microgroups(rows, budget=16384):
    group = []
    total = 0
    for r in rows:
        size = sum(((n + 63) // 64) * 64 for n in r["branch_lengths"])
        if group and total + size > budget:
            yield group
            group = []
            total = 0
        group.append(r)
        total += size
    if group:
        yield group


def update(m, rows, opt, scheduler, scaler, config, check_grad=False, _attempt=0):
    saved_rng = rng_state()
    m.train()
    opt.zero_grad(set_to_none=True)
    start = time.time()
    loss_value = 0.0
    correct = 0
    for group in microgroups(rows, config["graph_token_budget"]):
        z = m.forward_rows(
            group, config["max_candidate_rows"], config["candidate_token_budget"]
        )
        y = torch.tensor([r["gold_index"] for r in group], device="cuda")
        loss = choice_cross_entropy(z, y)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite loss")
        scaler.scale(loss).backward()
        loss_value += float(loss.detach()) / len(rows)
        correct += int((z.argmax(-1) == y).sum())
    scaler.unscale_(opt)
    # Average in FP32 after backward, preserving the mean-loss gradient without
    # dividing low-magnitude FP16 intermediate gradients by the effective batch.
    for p in m.parameters():
        if p.grad is not None:
            p.grad.div_(len(rows))
    gradients = {}
    finite = True
    for tag, ps in zip(["lora", "head"], parameters(m)):
        gs = [p.grad for p in ps if p.grad is not None]
        total = sum(float(g.abs().sum()) for g in gs)
        if not gs:
            raise RuntimeError("Disconnected " + tag + " gradients")
        finite = finite and math.isfinite(total)
        if check_grad:
            gradients[tag] = total
    if not finite:
        oldscale = scaler.get_scale()
        if m.overflow_log is not None:
            append_jsonl(
                m.overflow_log,
                {
                    "step": scheduler.last_epoch,
                    "attempt": _attempt,
                    "scale": oldscale,
                    "ids": [r["id"] for r in rows],
                    "time": time.time(),
                },
            )
        if _attempt >= AMP_MAX_RETRIES or oldscale < 0.001:
            raise RuntimeError(
                "Persistent nonfinite gradients after bounded AMP retries"
            )
        scaler.update(new_scale=oldscale / 2)
        opt.zero_grad(set_to_none=True)
        restore_rng(saved_rng)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        result = update(
            m, rows, opt, scheduler, scaler, config, check_grad, _attempt + 1
        )
        result["seconds"] += elapsed
        result["amp_retries"] = result.get("amp_retries", 0) + 1
        return result
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in m.parameters() if p.requires_grad], 1.0
    )
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("Nonfinite unscaled gradient norm")
    oldscale = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    assert scaler.get_scale() >= oldscale
    scheduler.step()
    torch.cuda.synchronize()
    return {
        "loss": loss_value,
        "accuracy": correct / len(rows),
        "gradient_norm": float(norm),
        "seconds": time.time() - start,
        "scaler": scaler.get_scale(),
        "gradients": gradients,
        "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
    }
