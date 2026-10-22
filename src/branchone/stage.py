"""A complete supervised stage, from fresh or existing weights, with exact resume."""

import copy
import gc
import importlib.metadata
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from .checkpoint import (
    load_resume,
    resolve_checkpoint,
    restore_journals,
    save_checkpoint,
)
from .io import (
    append_jsonl,
    canonical_hash,
    dataset_identity,
    file_manifest,
    output_lock,
    read_jsonl,
    source_identity,
    write_json,
)
from .metrics import metrics
from .model import MODEL, REVISION
from .training import (
    INITIAL_LOSS_SCALE,
    TrainModel,
    frozen_hash,
    learned_hash,
    optimizer,
    permute,
    restore_rng,
    rng_state,
    settings,
    update,
)

DEFAULTS = {
    "epochs": 1,
    "effective_batch": 16,
    "seed": 42,
    "max_length": 8192,
    "image_tokens": 256,
    "lora_rank": 8,
    "max_candidate_rows": 32,
    "candidate_token_budget": 65536,
    "graph_token_budget": 16384,
    "shared_prefix_min_tokens": None,
    "initial_loss_scale": INITIAL_LOSS_SCALE,
    "save_every_updates": 100,
    "save_every_seconds": 900,
    "evaluate_every_epoch_fraction": 0.25,
    "max_work_seconds": 14400,
}


def validated_config(path=None):
    supplied = json.loads(Path(path).read_text()) if path else {}
    if set(supplied) - set(DEFAULTS):
        raise ValueError("Unknown training configuration keys")
    result = DEFAULTS | supplied
    for key in [
        "epochs",
        "effective_batch",
        "max_length",
        "image_tokens",
        "lora_rank",
        "max_candidate_rows",
        "candidate_token_budget",
        "graph_token_budget",
        "save_every_updates",
        "save_every_seconds",
        "max_work_seconds",
    ]:
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(result["seed"]) is not int or result["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        result["max_length"] != 8192
        or result["image_tokens"] != 256
        or result["lora_rank"] != 8
    ):
        raise ValueError(
            "This reproduction protocol uses 8192 tokens, 256 image tokens, rank 8"
        )
    if result["shared_prefix_min_tokens"] is not None:
        raise ValueError(
            "Shared-prefix training is not qualified for this reproduction; use null"
        )
    if result["evaluate_every_epoch_fraction"] != 0.25:
        raise ValueError("Development selection uses fixed quarter-epoch milestones")
    if (
        not math.isfinite(result["initial_loss_scale"])
        or result["initial_loss_scale"] <= 0
    ):
        raise ValueError("initial_loss_scale must be positive and finite")
    return result


def environment_identity():
    names = [
        "torch",
        "torchvision",
        "torchaudio",
        "torchcodec",
        "transformers",
        "peft",
        "accelerate",
        "flash-linear-attention",
        "fla-core",
        "numpy",
        "Pillow",
        "safetensors",
        "causal-conv1d",
    ]
    versions = {name: importlib.metadata.version(name) for name in names}
    if versions["torch"] != "2.11.0+cu128" or versions["causal-conv1d"] != "1.7.0":
        raise RuntimeError(
            "Training requires the recorded Torch/CUDA and convolution builds"
        )
    return {
        "packages": versions,
        "cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "causal_conv_deterministic": os.environ.get("CAUSAL_CONV1D_DETERMINISTIC"),
    }


def initialization_identity(checkpoint, seed):
    if checkpoint is None:
        return {
            "kind": "fresh",
            "base_model": MODEL,
            "revision": REVISION,
            "seed": seed,
        }
    checkpoint = Path(checkpoint)
    required = [
        "model_config.json",
        "head.safetensors",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "processor/tokenizer.json",
    ]
    for relative in required:
        if not (checkpoint / relative).is_file():
            raise ValueError(f"Incomplete model checkpoint: missing {relative}")
    manifest = file_manifest(checkpoint)
    return {
        "kind": "existing_weights_new_stage",
        "parent_files": manifest,
        "parent_manifest_sha256": canonical_hash(manifest),
        "optimizer": "fresh",
    }


def batch_at(rows, step, config):
    """The cursor identifies an optimizer update, including a partial final batch."""
    per_epoch = math.ceil(len(rows) / config["effective_batch"])
    epoch, offset = divmod(step, per_epoch)
    order = list(range(len(rows)))
    random.Random(config["seed"] + epoch).shuffle(order)
    start = offset * config["effective_batch"]
    indices = order[start : start + config["effective_batch"]]
    rng = random.Random(config["seed"] * 1_000_000 + step)
    return [permute(rows[i], rng) for i in indices], epoch, offset


@torch.inference_mode()
def evaluate_rows(model, rows, output):
    model.eval()
    records = []
    for row in rows:
        logits = model.forward_rows([row], max_rows=1)[
            0, : len(row["candidates"])
        ].float()
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Nonfinite development logits")
        record = {
            "id": row["id"],
            "gold_index": row["gold_index"],
            "candidate_ids": [c["id"] for c in row["candidates"]],
            "logits": logits.cpu().tolist(),
            "selected_index": int(logits.argmax()),
            "modality": row["modality"],
        }
        records.append(record)
    # Commit a complete prediction file; never use a partial evaluation for selection.
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    os.replace(temporary, path)
    return metrics(records)


def verify_epoch_coverage(rows, training_path, config):
    records = list(read_jsonl(training_path))
    per_epoch = math.ceil(len(rows) / config["effective_batch"])
    expected = per_epoch * config["epochs"]
    if [x["step"] for x in records] != list(range(1, expected + 1)):
        raise RuntimeError(
            "Training journal has missing or duplicate optimizer updates"
        )
    expected_ids = sorted(row["id"] for row in rows)
    for epoch in range(config["epochs"]):
        group = records[epoch * per_epoch : (epoch + 1) * per_epoch]
        if sorted(rid for row in group for rid in row["ids"]) != expected_ids:
            raise RuntimeError(
                "An epoch did not visit every training example exactly once"
            )
        if any(row["epoch"] != epoch for row in group):
            raise RuntimeError("Incorrect epoch cursor in journal")
    return {
        "epochs": config["epochs"],
        "examples_per_epoch": len(rows),
        "optimizer_updates": expected,
        "every_example_once_per_epoch": True,
    }


def run_stage(
    *,
    train_path,
    development_path,
    image_root,
    output,
    config_path=None,
    init_checkpoint=None,
    resume=False,
    expected_training_rows=None,
    max_updates=None,
):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The validated training implementation requires an NVIDIA GPU"
        )
    if resume and init_checkpoint:
        raise ValueError(
            "Choose exact stage resume or initialization from model weights"
        )
    config = validated_config(config_path)
    if max_updates is not None and (type(max_updates) is not int or max_updates <= 0):
        raise ValueError("max_updates must be a positive interruption boundary")
    output = Path(output).resolve()
    if init_checkpoint:
        parent = Path(init_checkpoint).resolve()
        if (
            output == parent
            or parent.is_relative_to(output)
            or output.is_relative_to(parent)
        ):
            raise ValueError(
                "The new stage output must be separate from its parent checkpoint"
            )
    with output_lock(output):
        if (output / "complete.json").exists():
            raise ValueError(
                "Completed stages are immutable; use a different output for new data"
            )
        if not resume and any(p.name != ".stage.lock" for p in output.iterdir()):
            raise ValueError("A fresh stage needs an empty output directory")
        settings(config["seed"])
        data, data_id = dataset_identity(
            train_path, development_path, image_root, config["max_length"]
        )
        rows, dev = data["train"], data["development"]
        if expected_training_rows is not None and len(rows) != expected_training_rows:
            raise ValueError("Training row count differs from the declared experiment")
        per_epoch = math.ceil(len(rows) / config["effective_batch"])
        total = per_epoch * config["epochs"]
        current = {
            "format_version": 1,
            "source": source_identity(),
            "config": config,
            "dataset": data_id,
            "environment": environment_identity(),
            "total_updates": total,
        }
        if resume:
            checkpoint, restored = load_resume(output)
            current["initialization"] = restored["identity"]["initialization"]
            if current != restored["identity"]:
                raise ValueError(
                    "Data, source, environment or schedule changed; start a new stage"
                )
            restore_journals(output, restored["journal_offsets"])
            model = TrainModel.load(checkpoint, trainable=True)
            state = restored["state"]
        else:
            current["initialization"] = initialization_identity(
                init_checkpoint, config["seed"]
            )
            model = (
                TrainModel.load(init_checkpoint, trainable=True)
                if init_checkpoint
                else TrainModel(
                    trainable=True,
                    max_length=config["max_length"],
                    image_tokens=config["image_tokens"],
                    lora_rank=config["lora_rank"],
                )
            )
            state = {
                "step": 0,
                "best_step": 0,
                "best_nll": float("inf"),
                "evaluated": [],
                "training_seconds": 0.0,
                "initial_learned_hash": learned_hash(model),
                "frozen_hash": frozen_hash(model),
            }
            write_json(output / "stage.json", current)
        if (
            model.max_length != config["max_length"]
            or model.lora_rank != config["lora_rank"]
        ):
            raise ValueError("Checkpoint architecture and stage configuration disagree")
        model.configure_runtime(
            image_root=image_root,
            cache_dir=output / "cache/vision",
            overflow_log=output / "amp-overflows.jsonl",
        )
        model.temperature = 1.0
        model.calibration_status = "unfitted-after-training"
        model.enable_gradient_checkpointing()
        model.shared_prefix_min_tokens = config["shared_prefix_min_tokens"]
        opt, scheduler, scaler = optimizer(model, total)
        scaler = torch.amp.GradScaler("cuda", init_scale=config["initial_loss_scale"])
        if resume:
            opt.load_state_dict(restored["optimizer"])
            scheduler.load_state_dict(restored["scheduler"])
            scaler.load_state_dict(restored["scaler"])
            restore_rng(restored["rng"])
        if frozen_hash(model) != state["frozen_hash"]:
            raise RuntimeError(
                "Frozen model tensors differ from the stage's initial tensors"
            )
        started, last_save = time.time(), time.time()
        milestones = {0, total} | {
            math.ceil(q * per_epoch / 4) for q in range(1, config["epochs"] * 4 + 1)
        }

        def save():
            nonlocal last_save
            save_checkpoint(
                output, model, opt, scheduler, scaler, copy.deepcopy(state), current
            )
            last_save = time.time()
            write_json(
                output / "status.json",
                {
                    "status": "training",
                    "step": state["step"],
                    "total_updates": total,
                    "time": time.time(),
                },
            )

        def develop():
            saved_rng = rng_state()
            score = evaluate_rows(
                model,
                dev,
                output / "predictions" / f"development-{state['step']:06d}.jsonl",
            )
            score.update(step=state["step"], epochs=state["step"] / per_epoch)
            if score["nll"] < state["best_nll"]:
                state["best_nll"], state["best_step"] = score["nll"], state["step"]
            append_jsonl(output / "development.jsonl", score)
            state["evaluated"].append(state["step"])
            restore_rng(saved_rng)
            save()
            print("DEVELOPMENT", json.dumps(score), flush=True)

        if state["step"] in milestones and state["step"] not in state["evaluated"]:
            develop()
        print("RESUME" if resume else "START", state["step"], "/", total, flush=True)
        updates_this_call = 0
        for step in range(state["step"], total):
            batch, epoch, offset = batch_at(rows, step, config)
            torch.cuda.reset_peak_memory_stats()
            result = update(
                model, batch, opt, scheduler, scaler, config, check_grad=(step == 0)
            )
            state["step"] = step + 1
            state["training_seconds"] += result["seconds"]
            result.update(
                step=step + 1,
                epoch=epoch,
                offset=offset,
                ids=[r["id"] for r in batch],
                learning_rates=scheduler.get_last_lr(),
            )
            append_jsonl(output / "training.jsonl", result)
            updates_this_call += 1
            if state["step"] in milestones and state["step"] not in state["evaluated"]:
                develop()
            elif (
                state["step"] % config["save_every_updates"] == 0
                or time.time() - last_save >= config["save_every_seconds"]
            ):
                save()
            if state["step"] % 10 == 0:
                print("TRAIN", state["step"], "/", total, result["loss"], flush=True)
            boundary = time.time() - started >= config["max_work_seconds"]
            boundary = boundary or (
                max_updates is not None and updates_this_call >= max_updates
            )
            if boundary and state["step"] < total:
                save()
                write_json(
                    output / "status.json",
                    {
                        "status": "resume_required",
                        "step": state["step"],
                        "total_updates": total,
                        "time": time.time(),
                    },
                )
                return 75
        save()
        after = learned_hash(model)
        if (
            after == state["initial_learned_hash"]
            or frozen_hash(model) != state["frozen_hash"]
        ):
            raise RuntimeError(
                "Learned weights did not change or frozen weights changed"
            )
        coverage = verify_epoch_coverage(rows, output / "training.jsonl", config)
        final_checkpoint, _ = resolve_checkpoint(output)
        opt = scheduler = scaler = model = None
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = TrainModel.load(final_checkpoint)
        if (
            learned_hash(reloaded) != after
            or frozen_hash(reloaded) != state["frozen_hash"]
        ):
            raise RuntimeError("Final checkpoint reload changed model tensors")
        best, best_pointer = resolve_checkpoint(output, "best")
        completion = {
            "status": "complete",
            "time": time.time(),
            "state": state,
            "coverage": coverage,
            "initial_learned_hash": state["initial_learned_hash"],
            "final_learned_hash": after,
            "frozen_hash_unchanged": True,
            "reload_hashes_match": True,
            "best_checkpoint": best.relative_to(output).as_posix(),
            "best_step": best_pointer["step"],
            "stage_identity_sha256": canonical_hash(current),
        }
        write_json(output / "complete.json", completion)
        write_json(output / "status.json", completion)
        print("COMPLETE", json.dumps(coverage), flush=True)
        return 0
