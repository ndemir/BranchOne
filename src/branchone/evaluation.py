"""Reference evaluation and calibration, with identities tied to the weights/data."""

import json
from pathlib import Path

import torch

from .calibration import (
    BACKEND,
    FITTED,
    load_calibration,
    model_identity,
    serving_identity,
)
from .io import (
    append_jsonl,
    image_path,
    read_jsonl,
    sha256,
    validate_prepared,
    write_json,
)
from .metrics import fit_temperature, report
from .training import TrainModel, settings


@torch.inference_mode()
def evaluate(
    checkpoint, data, image_root, output, calibration=None, fit_calibration=False
):
    if calibration and fit_calibration:
        raise ValueError("Choose fitting or applying calibration")
    calibration_value = (
        load_calibration(checkpoint, calibration) if calibration else None
    )
    bound_serving = serving_identity(checkpoint)
    settings()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(data))
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Evaluation rows must be nonempty with unique IDs")
    for row in rows:
        validate_prepared(row)
    if fit_calibration and any(row.get("split") != "calibration" for row in rows):
        raise ValueError(
            "Fit temperature only on an explicitly separate calibration split"
        )
    images = {
        row["image"]: sha256(image_path(image_root, row["image"]))
        for row in rows
        if row.get("image")
    }
    identity = {
        "model_files": model_identity(checkpoint),
        "serving_identity": bound_serving,
        "data_sha256": sha256(data),
        "image_files": images,
        "backend": BACKEND,
        "candidate_rows": 1,
        "calibration": calibration_value,
        "fit_calibration": fit_calibration,
    }
    identity_path = output / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Evaluation inputs changed; use a new output directory")
    else:
        if any(output.iterdir()):
            raise ValueError(
                "Evaluation output contains files without a valid identity"
            )
        write_json(identity_path, identity)
    predictions = output / "predictions.jsonl"
    records = list(read_jsonl(predictions)) if predictions.exists() else []
    expected = [
        (r["id"], r["gold_index"], [c["id"] for c in r["candidates"]]) for r in rows
    ]
    actual = [(r["id"], r["gold_index"], r["candidate_ids"]) for r in records]
    if actual != expected[: len(actual)] or len(actual) > len(expected):
        raise ValueError("Saved predictions do not match the evaluation rows")
    model = TrainModel.load(checkpoint)
    model.configure_runtime(image_root=image_root, cache_dir=output / "vision-cache")
    model.eval()
    for row in rows[len(records) :]:
        z = model.forward_rows([row], max_rows=1)[0, : len(row["candidates"])].float()
        if not bool(torch.isfinite(z).all()):
            raise RuntimeError("Nonfinite evaluation logits")
        item = {
            key: row.get(key, "unspecified")
            for key in [
                "id",
                "split",
                "modality",
                "schema",
                "domain",
                "answerability",
                "group_hash",
                "branch_tokens",
                "gold_index",
            ]
        }
        item.update(
            candidate_ids=[c["id"] for c in row["candidates"]],
            logits=z.cpu().tolist(),
            selected_index=int(z.argmax()),
        )
        append_jsonl(predictions, item)
        records.append(item)
        if len(records) % 50 == 0:
            print("EVALUATE", len(records), "/", len(rows), flush=True)
    temperature = 1.0
    if calibration_value:
        temperature = calibration_value["temperature"]
    if fit_calibration:
        temperature = fit_temperature(records)
        write_json(
            output / "calibration.json",
            {
                "temperature": temperature,
                "status": FITTED,
                "model_files": identity["model_files"],
                "serving_identity": identity["serving_identity"],
                "backend": identity["backend"],
                "data_sha256": identity["data_sha256"],
                "n": len(records),
            },
        )
    raw = report(records)
    final = report(records, temperature)
    write_json(output / "raw-metrics.json", raw)
    write_json(output / "metrics.json", final)
    write_json(
        output / "complete.json",
        {
            "n": len(records),
            "backend": identity["backend"],
            "temperature": temperature,
            "prediction_sha256": sha256(predictions),
            "raw": raw["overall"],
            "reported": final["overall"],
            "deferral_policy": "none",
        },
    )
    print(json.dumps(final["overall"]), flush=True)
    return final
