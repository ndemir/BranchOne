"""Native Qwen vision, independent candidate paths, and a shared scalar head.

Public base weights remain external inputs. Checkpoints store the adapter,
head, processor, and immutable backbone revision.
"""

import copy
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoProcessor, DynamicCache, Qwen3_5Model

from .calibration import UNFITTED, checkpoint_calibration
from .head import TARGETS, CandidateHead
from .schema import validate

MODEL = "Qwen/Qwen3.5-0.8B-Base"
REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
PROMPT_VERSION = "branchone-multimodal-segments-v2"


class BranchOne(nn.Module):
    def __init__(
        self,
        *,
        device="cuda",
        model_id=MODEL,
        revision=REVISION,
        max_length=8192,
        image_tokens=256,
        max_rows=1,
        batch_token_budget=8192,
        adapter_path=None,
        head_path=None,
        trainable=False,
        lora_rank=8,
        processor_path=None,
    ):
        super().__init__()
        if not 2 <= max_length <= 32768:
            raise ValueError("Configured context must be between 2 and 32768 tokens")
        self.model_id, self.revision = model_id, revision
        self.max_length, self.image_tokens = max_length, image_tokens
        self.max_rows, self.batch_token_budget = max_rows, batch_token_budget
        self.temperature = 1.0
        self.calibration_status = UNFITTED
        self.cache_validated = False
        self.processor = AutoProcessor.from_pretrained(
            processor_path or model_id, revision=None if processor_path else revision
        )
        self.tokenizer = self.processor.tokenizer
        self.backbone, self.loading_info = Qwen3_5Model.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.float16,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        if any(
            self.loading_info.get(k)
            for k in ["missing_keys", "mismatched_keys", "error_msgs"]
        ):
            raise RuntimeError(
                f"Incomplete native multimodal weight load: {self.loading_info}"
            )
        self.backbone.requires_grad_(False)
        patch = self.backbone.config.vision_config.patch_size
        merge = self.backbone.config.vision_config.spatial_merge_size
        self.processor.image_processor.size = {
            "shortest_edge": min(64, image_tokens) * (patch * merge) ** 2,
            "longest_edge": image_tokens * (patch * merge) ** 2,
        }
        self.adapter_audit = {"warm_started": bool(adapter_path)}
        if adapter_path:
            self.backbone.language_model = PeftModel.from_pretrained(
                self.backbone.language_model, adapter_path, is_trainable=trainable
            )
            source = load_file(str(Path(adapter_path) / "adapter_model.safetensors"))
            actual = get_peft_model_state_dict(self.backbone.language_model)
            if source.keys() != actual.keys():
                raise RuntimeError("Warm-start adapter tensor names differ")
            error = max(
                float((actual[k].float().cpu() - v.float()).abs().max())
                for k, v in source.items()
            )
            if error != 0:
                raise RuntimeError(f"Warm-start adapter changed: {error}")
            self.adapter_audit.update(tensors=len(source), max_error=error)
        elif lora_rank:
            config = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                r=lora_rank,
                lora_alpha=2 * lora_rank,
                lora_dropout=0.05,
                target_modules=TARGETS,
            )
            self.backbone.language_model = get_peft_model(
                self.backbone.language_model, config
            )
            if not trainable:
                self.backbone.language_model.requires_grad_(False)
        self.lora_rank = lora_rank
        self.head = CandidateHead(hidden=self.backbone.config.text_config.hidden_size)
        if head_path:
            self.head.load_state_dict(load_file(str(head_path)), strict=True)
        self._vision_cache = OrderedDict()
        self._prepared_cache = OrderedDict()
        self.to(device)
        self.train(trainable)

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def train(self, mode=True):
        super().train(mode)
        self.backbone.visual.eval()  # frozen vision must stay deterministic
        if mode:
            self.temperature, self.calibration_status = 1.0, UNFITTED
        return self

    def enable_gradient_checkpointing(self):
        self.backbone.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    @staticmethod
    def _lru_put(cache, key, value, limit=64):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    @staticmethod
    def _image(value):
        if isinstance(value, Image.Image):
            im = ImageOps.exif_transpose(value).convert("RGB")
        elif isinstance(value, (str, Path)):
            if "://" in str(value):
                raise ValueError(
                    "Pass a local image path; URL fetching is not supported"
                )
            with Image.open(value) as opened:
                im = ImageOps.exif_transpose(opened).convert("RGB")
        else:
            raise ValueError("image must be a local path or a PIL image")
        content_hash = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        return im, content_hash

    def prepare(self, request):
        """Build CPU tensors, enforce expanded token limits, and keep IDs external."""
        validate(request)
        if "images" in request:
            raise ValueError(
                "Use the singular image field; exactly one image is supported"
            )
        image_value = request.get("image")
        image_hash = None
        if image_value is not None:
            im, image_hash = self._image(image_value)
        state = "State:\n" + request["state"] + "\n"
        key = hashlib.sha256(
            json.dumps(
                {
                    "state": state,
                    "question": request["question"],
                    "type": request.get("type", "choice"),
                    "image": image_hash,
                    "candidates": [c["description"] for c in request["candidates"]],
                    "max_length": self.max_length,
                    "image_tokens": self.image_tokens,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if key in self._prepared_cache:
            self._prepared_cache.move_to_end(key)
            return self._prepared_cache[key]

        def enc(text):
            return self.tokenizer.encode(text, add_special_tokens=False)

        pixels = grid = None
        if image_hash:
            text = (
                "State:\n<|vision_start|><|image_pad|><|vision_end|>\n"
                + request["state"]
                + "\n"
            )
            processed = self.processor(
                text=text, images=im, add_special_tokens=False, return_tensors="pt"
            )
            state_ids = processed["input_ids"][0].tolist()
            pixels, grid = processed["pixel_values"], processed["image_grid_thw"]
            visual_count = state_ids.count(self.backbone.config.image_token_id)
            if not 0 < visual_count <= self.image_tokens:
                raise ValueError(
                    f"Visual token count {visual_count} exceeds budget {self.image_tokens}"
                )
        else:
            state_ids, visual_count = enc(state), 0
        prefix = state_ids + enc(
            "Task: "
            + request.get("type", "choice")
            + "\nQuestion:\n"
            + request["question"]
            + "\n"
        )
        suffixes = [
            enc("Candidate:\n" + c["description"] + "\nDecision:")
            for c in request["candidates"]
        ]
        paths = [prefix + suffix for suffix in suffixes]
        lengths = [len(p) for p in paths]
        if max(lengths) > self.max_length:
            raise ValueError(
                f"Expanded branch has {max(lengths)} tokens; limit {self.max_length}; no truncation"
            )
        result = {
            "paths": paths,
            "prefix": prefix,
            "suffixes": suffixes,
            "pixel_values": pixels,
            "image_grid_thw": grid,
            "image_hash": image_hash,
            "token_counts": lengths,
            "visual_tokens": visual_count,
            "image_size": list(im.size) if image_hash else None,
        }
        self._lru_put(self._prepared_cache, key, result)
        return result

    @torch.no_grad()
    def visual_features(self, prepared, reuse=True):
        if not prepared["image_hash"]:
            return None
        key = prepared["image_hash"]
        if reuse and key in self._vision_cache:
            self._vision_cache.move_to_end(key)
            return self._vision_cache[key].to(self.device)
        out = self.backbone.get_image_features(
            prepared["pixel_values"].to(self.device),
            prepared["image_grid_thw"].to(self.device),
            return_dict=True,
        )
        features = torch.cat(out.pooler_output, dim=0).detach()
        if reuse:
            self._lru_put(self._vision_cache, key, features.cpu())
        return features

    def _inputs(self, paths, prepared, features, *, pad_to_bucket=True):
        lengths = torch.tensor([len(p) for p in paths], device=self.device)
        width = int(lengths.max())
        if pad_to_bucket:
            width = ((width + 63) // 64) * 64
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        ids = torch.full(
            (len(paths), width), pad_id, dtype=torch.long, device=self.device
        )
        for i, path in enumerate(paths):
            ids[i, : len(path)] = torch.tensor(path, device=self.device)
        mask = torch.arange(width, device=self.device)[None, :] < lengths[:, None]
        embeds = self.backbone.get_input_embeddings()(ids)
        if features is not None:
            is_image = ids == self.backbone.config.image_token_id
            assert int(is_image.sum()) == len(paths) * features.shape[0]
            embeds = embeds.masked_scatter(
                is_image[..., None], features.repeat(len(paths), 1).to(embeds.dtype)
            )
            grid = prepared["image_grid_thw"].to(self.device).repeat(len(paths), 1)
            positions, _ = self.backbone.get_rope_index(
                ids,
                mm_token_type_ids=is_image.long(),
                image_grid_thw=grid,
                attention_mask=mask.long(),
            )
        else:
            positions = torch.arange(width, device=self.device)[None, :].expand(
                len(paths), -1
            )
            positions = positions.masked_fill(~mask, 0)
        return {
            "inputs_embeds": embeds,
            "attention_mask": mask.long(),
            "position_ids": positions,
        }, lengths

    def representations(self, request, *, reuse_image=True, max_rows=None):
        prepared = self.prepare(request)
        features = self.visual_features(prepared, reuse=reuse_image)
        paths = prepared["paths"]
        buckets = {}
        for i, p in enumerate(paths):
            buckets.setdefault(((len(p) + 63) // 64) * 64, []).append(i)
        hidden = [None] * len(paths)
        for width, indices in sorted(buckets.items()):
            batch_size = max(
                1, min(max_rows or self.max_rows, self.batch_token_budget // width)
            )
            for start in range(0, len(indices), batch_size):
                group = indices[start : start + batch_size]
                inputs, lengths = self._inputs(
                    [paths[i] for i in group], prepared, features
                )
                with torch.autocast(
                    "cuda", dtype=torch.float16, enabled=self.device.type == "cuda"
                ):
                    out = self.backbone.language_model(
                        **inputs, use_cache=False
                    ).last_hidden_state
                leaves = out[
                    torch.arange(len(group), device=self.device), lengths - 1
                ].float()
                for i, leaf in zip(group, leaves):
                    hidden[i] = leaf
        return torch.stack(hidden), prepared

    def forward(
        self, request, *, return_features=False, reuse_image=True, max_rows=None
    ):
        hidden, prepared = self.representations(
            request, reuse_image=reuse_image, max_rows=max_rows
        )
        valid = torch.ones((1, len(hidden)), dtype=torch.bool, device=self.device)
        logits, features = self.head(hidden[None], valid, return_features=True)
        return (logits[0], features[0], prepared) if return_features else logits[0]

    @torch.inference_mode()
    def native_reference(self, request):
        """Native full multimodal forward with matching padding for integration checks."""
        prepared = self.prepare(request)
        hidden = []
        for path in prepared["paths"]:
            width = ((len(path) + 63) // 64) * 64
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            ids = torch.tensor(
                [path + [pad_id] * (width - len(path))], device=self.device
            )
            mask = (torch.arange(width, device=self.device)[None] < len(path)).long()
            kwargs = {"input_ids": ids, "attention_mask": mask, "use_cache": False}
            if prepared["image_hash"]:
                kwargs.update(
                    pixel_values=prepared["pixel_values"].to(self.device),
                    image_grid_thw=prepared["image_grid_thw"].to(self.device),
                    mm_token_type_ids=(
                        ids == self.backbone.config.image_token_id
                    ).long(),
                )
            hidden.append(
                self.backbone(**kwargs).last_hidden_state[0, len(path) - 1].float()
            )
        h = torch.stack(hidden)[None]
        return self.head(
            h, torch.ones(h.shape[:2], dtype=torch.bool, device=self.device)
        )[0]

    @torch.inference_mode()
    def cached_logits(self, request):
        """Experimental native-cache fork; callers must validate before serving."""
        prepared = self.prepare(request)
        features = self.visual_features(prepared)
        full, _ = self._inputs(
            prepared["paths"], prepared, features, pad_to_bucket=False
        )
        prefix_length = len(prepared["prefix"])
        positions = full["position_ids"]
        prefix_pos = (
            positions[:, :1, :prefix_length]
            if positions.ndim == 3
            else positions[:1, :prefix_length]
        )
        out = self.backbone.language_model(
            inputs_embeds=full["inputs_embeds"][:1, :prefix_length],
            attention_mask=full["attention_mask"][:1, :prefix_length],
            position_ids=prefix_pos,
            past_key_values=DynamicCache(config=self.backbone.config.text_config),
            use_cache=True,
        )
        prefix_cache = out.past_key_values
        hidden = []
        for i, path in enumerate(prepared["paths"]):
            length = len(path)
            pos = (
                positions[:, i : i + 1, prefix_length:length]
                if positions.ndim == 3
                else positions[i : i + 1, prefix_length:length]
            )
            branch = self.backbone.language_model(
                inputs_embeds=full["inputs_embeds"][i : i + 1, prefix_length:length],
                attention_mask=full["attention_mask"][i : i + 1, :length],
                position_ids=pos,
                past_key_values=copy.deepcopy(prefix_cache),
                use_cache=True,
            )
            hidden.append(branch.last_hidden_state[0, -1].float())
        if prefix_cache.get_seq_length() != prefix_length:
            raise RuntimeError("Shared prefix cache was mutated")
        h = torch.stack(hidden)[None]
        return self.head(
            h, torch.ones(h.shape[:2], dtype=torch.bool, device=self.device)
        )[0]

    @torch.inference_mode()
    def predict(self, request, *, cached=False, threshold=None):
        self.eval()
        if cached and not self.cache_validated:
            raise ValueError(
                "Prefix cache has not passed checkpoint-specific validation"
            )
        logits = self.cached_logits(request) if cached else self(request)
        p = (logits / self.temperature).softmax(-1).cpu()
        ids = [c["id"] for c in request["candidates"]]
        selected = int(p.argmax())
        prepared = self.prepare(request)
        result = {
            "selected_id": ids[selected],
            "candidate_ids": ids,
            "probabilities": p.tolist(),
            "logits": logits.cpu().tolist(),
            "temperature": self.temperature,
            "calibration_status": self.calibration_status,
            "model_version": "branchone-0.3.0-independent",
            "context_limit": self.max_length,
            "branch_token_counts": prepared["token_counts"],
            "visual_tokens": prepared["visual_tokens"],
            "image_size": prepared["image_size"],
            "action": "defer"
            if threshold is None
            else ("accept" if float(p.max()) >= threshold else "defer"),
            "threshold": threshold,
            "deferral_status": "no validated policy"
            if threshold is None
            else "caller-supplied policy",
        }
        if request.get("type") == "boolean":
            result.update(
                p_true=float(p[ids.index("true")]), value=ids[selected] == "true"
            )
        if request.get("type") == "score":
            result["expected_score"] = sum(
                float(x) * c["level"] for x, c in zip(p, request["candidates"])
            )
        return result

    def save(self, path):
        """Save trainable weights without carrying calibration to a new payload."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.language_model.save_pretrained(path / "adapter")
        save_file(
            {
                k: v.detach().cpu().contiguous()
                for k, v in self.head.state_dict().items()
            },
            str(path / "head.safetensors"),
        )
        self.processor.save_pretrained(path / "processor")
        meta = {
            "model_id": self.model_id,
            "revision": self.revision,
            "max_length": self.max_length,
            "image_tokens": self.image_tokens,
            "max_rows": self.max_rows,
            "batch_token_budget": self.batch_token_budget,
            "lora_rank": self.lora_rank,
            "prompt_version": PROMPT_VERSION,
            "temperature": 1.0,
            "calibration_status": UNFITTED,
            "cache_validated": False,
        }
        (path / "model_config.json").write_text(json.dumps(meta, indent=2))
        (path / "calibration.json").unlink(missing_ok=True)

    @classmethod
    def load(cls, path, *, device="cuda", trainable=False):
        path = Path(path)
        meta = json.loads((path / "model_config.json").read_text())
        if meta.pop("prompt_version") != PROMPT_VERSION:
            raise ValueError("Incompatible prompt version")
        temperature, status = checkpoint_calibration(path, meta, trainable=trainable)
        meta.pop("temperature")
        meta.pop("calibration_status")
        meta.pop("cache_validated", None)
        model = cls(
            **meta,
            device=device,
            trainable=trainable,
            adapter_path=path / "adapter",
            head_path=path / "head.safetensors",
            processor_path=path / "processor",
        )
        model.temperature, model.calibration_status = temperature, status
        return model
