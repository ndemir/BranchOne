"""Validate calibration against every serving input before constructing a model."""

import json
import math
import re
from pathlib import Path

from .io import sha256

BACKEND = "hf-adapter-fp16"
UNFITTED = "unfitted for v2"
FITTED = "fitted on the supplied calibration split only"
SERVING_FILES = (
    "model_config.json",
    "head.safetensors",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/processor_config.json",
    "processor/chat_template.jinja",
)
LEGACY_FILES = (
    "head.safetensors",
    "adapter/adapter_model.safetensors",
    "model_config.json",
)


def positive_temperature(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("Calibration temperature must be finite and positive")
    return float(value)


def model_identity(checkpoint):
    """Legacy identity retained for compatibility with completed result audits."""
    return {name: sha256(Path(checkpoint) / name) for name in LEGACY_FILES}


def serving_identity(checkpoint):
    root = Path(checkpoint).resolve()
    files = {}
    for name in SERVING_FILES:
        path = root / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or not path.is_file()
        ):
            raise ValueError("Missing or unsafe serving file: " + name)
        files[name] = sha256(path)
    return {"version": 1, "files": files}


def validate_calibration(checkpoint, value, *, backend=BACKEND):
    required = {
        "temperature",
        "status",
        "model_files",
        "backend",
        "data_sha256",
        "n",
        "serving_identity",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            "Calibration requires versioned serving-file provenance; legacy "
            "metadata must be upgraded using verified evaluation evidence"
        )
    positive_temperature(value["temperature"])
    if backend != BACKEND or value["backend"] != backend:
        raise ValueError("Calibration backend is not qualified")
    if value["status"] != FITTED:
        raise ValueError("Calibration must record a fitted calibration split")
    if type(value["n"]) is not int or value["n"] <= 0:
        raise ValueError("Calibration example count must be a positive integer")
    if not isinstance(value["data_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", value["data_sha256"]
    ):
        raise ValueError("Calibration data identity must be a SHA256 digest")
    binding = value["serving_identity"]
    if (
        not isinstance(binding, dict)
        or set(binding) != {"version", "files"}
        or type(binding["version"]) is not int
        or binding["version"] != 1
    ):
        raise ValueError("Unsupported calibration serving identity")
    actual = serving_identity(checkpoint)
    if binding != actual:
        raise ValueError("Calibration was fitted for different serving files")
    if value["model_files"] != {name: actual["files"][name] for name in LEGACY_FILES}:
        raise ValueError("Calibration has a stale legacy model identity")
    return value


def load_calibration(checkpoint, path, *, backend=BACKEND):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Cannot read calibration metadata") from error
    return validate_calibration(checkpoint, value, backend=backend)


def checkpoint_calibration(checkpoint, config, *, trainable=False):
    """Training starts unfitted; serving requires either unit T or valid binding."""
    if trainable:
        return 1.0, UNFITTED
    path = Path(checkpoint) / "calibration.json"
    if path.exists():
        value = load_calibration(checkpoint, path)
        return positive_temperature(value["temperature"]), value["status"]
    temperature = positive_temperature(config.get("temperature"))
    status = config.get("calibration_status")
    if (
        temperature != 1.0
        or not isinstance(status, str)
        or not status.startswith("unfitted")
    ):
        raise ValueError("Serving calibration is missing its verified binding")
    return 1.0, UNFITTED
