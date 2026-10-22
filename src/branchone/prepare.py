"""Prepare user-owned Choice data without truncation or cross-split evidence reuse."""

import collections
import concurrent.futures
import gzip
import io
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoConfig, AutoProcessor

from .io import canonical_hash, file_manifest, image_path, read_jsonl, write_json
from .model import MODEL, PROMPT_VERSION, REVISION
from .schema import validate

PRIORITY = {"train": 0, "development": 1, "calibration": 2, "threshold": 3, "test": 4}


def normalize(raw, split):
    validate(raw, max_candidates=150)
    if raw.get("type", "choice") != "choice":
        raise ValueError("Training preparation supports single-label Choice tasks")
    if not isinstance(raw.get("id"), str) or not raw["id"]:
        raise ValueError("Each data row needs a stable, nonempty ID")
    gold = raw.get("gold_index")
    if type(gold) is not int or not 0 <= gold < len(raw["candidates"]):
        raise ValueError("gold_index must identify exactly one supplied candidate")
    metadata = raw.get("metadata", {})
    modality = "multimodal" if raw.get("image") else "text"
    schema = raw.get("schema", metadata.get("task_schema", "unspecified"))
    domain = raw.get("domain", metadata.get("domain", "unspecified"))
    # Preserve the earlier synthetic corpus's factual grouping when provided;
    # generic datasets may supply a group_id shared by related examples.
    facts = None
    if modality == "text" and metadata.get("verification", {}).get("facts"):
        facts = dict(metadata["verification"]["facts"])
        for field in ["source_group", "style", "length_bucket"]:
            facts.pop(field, None)
    elif modality == "multimodal" and metadata.get("scene_witness", {}).get("facts"):
        facts = metadata["scene_witness"]["facts"]

    def normalized(text):
        return " ".join(text.split()).casefold()

    input_hash = canonical_hash(
        [
            normalized(raw["state"]),
            normalized(raw["question"]),
            sorted(normalized(c["description"]) for c in raw["candidates"]),
            raw.get("image"),
        ]
    )
    group = raw.get("group_hash") or (
        canonical_hash([modality, schema, domain, facts])
        if facts is not None
        else canonical_hash(["group_id", raw["group_id"]])
        if "group_id" in raw
        else input_hash
    )
    row = {k: raw[k] for k in ["id", "state", "question", "candidates", "gold_index"]}
    row.update(
        type="choice",
        split=split,
        modality=modality,
        group_hash=group,
        input_hash=input_hash,
        schema=schema,
        domain=domain,
        answerability=raw.get(
            "answerability", metadata.get("answerability", "unspecified")
        ),
    )
    if raw.get("image"):
        row["image"] = raw["image"]
    return row


def prepare_data(split_files, image_root, output, max_length=8192, workers=4):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Prepared-data output must be empty; previous splits are immutable"
        )
    if not {"train", "development"}.issubset(split_files) or set(split_files) - set(
        PRIORITY
    ):
        raise ValueError(
            "Supply train/development and optional calibration/threshold/test splits"
        )
    if max_length != 8192:
        raise ValueError("This reproduction protocol uses 8192 expanded tokens")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION)
    patch = config.vision_config.patch_size * config.vision_config.spatial_merge_size
    processor.image_processor.size = {
        "shortest_edge": 64 * patch**2,
        "longest_edge": 256 * patch**2,
    }
    processor.tokenizer.model_max_length = 10**8
    all_rows, owners, ids = [], {}, set()
    for split in PRIORITY:
        for path in split_files.get(split, []):
            for raw in read_jsonl(path):
                row = normalize(raw, split)
                if row["id"] in ids:
                    raise ValueError("Duplicate example ID")
                ids.add(row["id"])
                for signature in [
                    "fact:" + row["group_hash"],
                    "input:" + row["input_hash"],
                ]:
                    owners[signature] = max(owners.get(signature, -1), PRIORITY[split])
                all_rows.append(row)

    def encode(row):
        image = None
        if row.get("image"):
            with Image.open(image_path(image_root, row["image"])) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
        prompt = (
            "State:\n"
            + (
                "<|vision_start|><|image_pad|><|vision_end|>\n"
                if image is not None
                else ""
            )
            + row["state"]
            + "\n"
        )
        kwargs = {
            "text": prompt,
            "return_tensors": "pt",
            "add_special_tokens": False,
            "truncation": False,
        }
        if image is not None:
            kwargs["images"] = image
        inputs = processor(**kwargs)
        prefix = inputs["input_ids"][0].tolist()
        visual = prefix.count(config.image_token_id)
        if (image is None and visual != 0) or (
            image is not None and not 0 < visual <= 256
        ):
            raise ValueError("Image-token expansion differs from the declared budget")
        prefix += processor.tokenizer.encode(
            "Task: choice\nQuestion:\n" + row["question"] + "\n",
            add_special_tokens=False,
        )
        suffixes = processor.tokenizer(
            [
                "Candidate:\n" + c["description"] + "\nDecision:"
                for c in row["candidates"]
            ],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        lengths = [len(prefix) + len(s) for s in suffixes]
        reasons = []
        if (
            owners["fact:" + row["group_hash"]] > PRIORITY[row["split"]]
            or owners["input:" + row["input_hash"]] > PRIORITY[row["split"]]
        ):
            reasons.append("duplicate_in_higher_priority_split")
        if max(lengths) > max_length:
            reasons.append("expanded_token_overflow")
        prepared = row | {
            "prefix_ids": prefix,
            "suffix_ids": suffixes,
            "branch_lengths": lengths,
            "branch_tokens": max(lengths),
            "visual_tokens": visual,
        }
        return prepared, reasons

    counts, exclusions, development = collections.Counter(), [], []
    streams, raw_streams = {}, []
    try:
        for split in split_files:
            binary = (output / (split + ".jsonl.gz")).open("wb")
            raw_streams.append(binary)
            streams[split] = io.TextIOWrapper(
                gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0),
                encoding="utf-8",
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(all_rows), 32):
                for row, reasons in pool.map(encode, all_rows[start : start + 32]):
                    if reasons:
                        exclusions.append(
                            {"id": row["id"], "split": row["split"], "reasons": reasons}
                        )
                        continue
                    streams[row["split"]].write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
                    counts[row["split"] + "/" + row["modality"]] += 1
                    if row["split"] == "development":
                        development.append(row)
    finally:
        for stream in streams.values():
            stream.close()
        for stream in raw_streams:
            stream.close()
    strata = collections.defaultdict(list)
    for row in development:
        bucket = next(n for n in [512, 2048, 4096, 8192] if row["branch_tokens"] <= n)
        strata[(row["modality"], row["schema"], bucket)].append(row)
    for values in strata.values():
        values.sort(key=lambda r: canonical_hash(["selection-panel-v1", r["id"]]))
    panel, level = [], 0
    while len(panel) < min(512, len(development)):
        for key in sorted(strata):
            if len(strata[key]) > level and len(panel) < 512:
                panel.append(strata[key][level])
        level += 1
    with (output / "development-panel.jsonl.gz").open("wb") as binary:
        with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8") as stream:
                for row in panel:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    processor.save_pretrained(output / "processor")
    report = {
        "source_rows": len(all_rows),
        "accepted_counts": dict(counts),
        "excluded": exclusions,
        "development_panel_ids": [r["id"] for r in panel],
        "base_model": MODEL,
        "revision": REVISION,
        "prompt_version": PROMPT_VERSION,
        "max_length": max_length,
        "truncation": False,
    }
    write_json(output / "admission.json", report)
    write_json(output / "prepared-files.json", file_manifest(output))
    print(
        json.dumps(
            {
                "source_rows": len(all_rows),
                "accepted_counts": dict(counts),
                "excluded": len(exclusions),
            }
        ),
        flush=True,
    )
    return report
