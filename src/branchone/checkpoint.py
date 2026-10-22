"""Verified checkpoint generations and exact within-stage restart state."""

import json
import os
import shutil
import uuid
from pathlib import Path

import torch

from .io import file_manifest, sha256, write_json
from .training import rng_state

FORMAT_VERSION = 1


def verify_checkpoint(directory):
    root = Path(directory).resolve()
    if root.is_symlink():
        raise ValueError("Checkpoint directory must not be a symlink")
    manifest = json.loads((root / "payload.json").read_text())
    actual = file_manifest(root)
    actual.pop("payload.json")
    if actual != manifest:
        raise ValueError("Checkpoint file manifest mismatch")
    return manifest


def resolve_checkpoint(stage, pointer="latest"):
    stage = Path(stage).resolve()
    value = json.loads((stage / (pointer + ".json")).read_text())
    relative = Path(value["checkpoint"])
    root = (stage / relative).resolve()
    if relative.is_absolute() or not root.is_relative_to(stage / "checkpoints"):
        raise ValueError("Checkpoint pointer escaped the stage directory")
    if sha256(root / "payload.json") != value["manifest_sha256"]:
        raise ValueError("Checkpoint pointer does not match its manifest")
    verify_checkpoint(root)
    return root, value


def save_checkpoint(stage, model, optimizer, scheduler, scaler, state, identity):
    stage = Path(stage)
    generations = stage / "checkpoints"
    generations.mkdir(exist_ok=True)
    name = f"step-{state['step']:06d}-{uuid.uuid4().hex[:12]}"
    temp = generations / (name + ".tmp")
    final = generations / name
    temp.mkdir()
    try:
        model.save(temp)
        offsets = {
            filename: (stage / filename).stat().st_size
            if (stage / filename).exists()
            else 0
            for filename in [
                "training.jsonl",
                "development.jsonl",
                "amp-overflows.jsonl",
            ]
        }
        obj = {
            "format_version": FORMAT_VERSION,
            "identity": identity,
            "state": state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": rng_state(),
            "journal_offsets": offsets,
        }
        torch.save(obj, temp / "resume.pt")
        write_json(temp / "payload.json", file_manifest(temp))
        os.replace(temp, final)
        value = {
            "checkpoint": final.relative_to(stage).as_posix(),
            "step": state["step"],
            "manifest_sha256": sha256(final / "payload.json"),
        }
        write_json(stage / "latest.json", value)
        if state["step"] == state["best_step"]:
            write_json(stage / "best.json", value)
        return final, value
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def load_resume(stage):
    root, pointer = resolve_checkpoint(stage)
    # Resume state contains Python/NumPy RNG state. Only load a trusted stage
    # produced by this trainer; hashes detect corruption, not malicious inputs.
    obj = torch.load(root / "resume.pt", map_location="cpu", weights_only=False)
    if obj.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            "Unsupported resume state; use a model checkpoint to start a new stage"
        )
    if obj["state"]["step"] != pointer["step"]:
        raise ValueError("Data cursor and checkpoint pointer disagree")
    return root, obj


def restore_journals(stage, offsets):
    """Discard writes after the last committed optimizer state, including partial JSON."""
    stage = Path(stage)
    allowed = {"training.jsonl", "development.jsonl", "amp-overflows.jsonl"}
    if set(offsets) != allowed:
        raise ValueError("Unexpected resume journal names")
    for filename, offset in offsets.items():
        if type(offset) is not int or offset < 0:
            raise ValueError("Invalid journal offset")
        path = stage / filename
        if not path.exists():
            if offset:
                raise ValueError("Missing committed training journal")
            continue
        if path.stat().st_size < offset:
            raise ValueError("Training journal is shorter than the committed state")
        with path.open("r+b") as stream:
            stream.truncate(offset)
