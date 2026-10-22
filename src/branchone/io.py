"""Portable data and provenance helpers; no model imports."""

import gzip
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .schema import validate


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at line {line_number}")
            yield value


def image_path(image_root, relative):
    if image_root is None:
        raise ValueError("An image root is required for image data")
    root = Path(image_root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or "://" in str(relative):
        raise ValueError("Dataset image paths must be relative to the image root")
    path = (root / rel).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Dataset image is missing or outside the image root")
    return path


def request_from_row(row, image_root):
    result = {key: row[key] for key in ["state", "question", "candidates"]}
    result["type"] = row.get("type", "choice")
    if row.get("image"):
        result["image"] = str(image_path(image_root, row["image"]))
    return result


def validate_prepared(row, max_length=8192):
    validate(row)
    if row.get("type", "choice") != "choice":
        raise ValueError("The supervised trainer currently supports Choice tasks")
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("Training rows need a nonempty ID")
    gold = row.get("gold_index")
    if type(gold) is not int or not 0 <= gold < len(row["candidates"]):
        raise ValueError("Invalid gold_index")
    prefix, suffixes = row.get("prefix_ids"), row.get("suffix_ids")
    if not isinstance(prefix, list) or not prefix or not isinstance(suffixes, list):
        raise ValueError("Prepared token segments are missing")
    if len(suffixes) != len(row["candidates"]) or any(not x for x in suffixes):
        raise ValueError("Candidate token segments do not match the candidates")
    if any(type(t) is not int or t < 0 for seq in [prefix, *suffixes] for t in seq):
        raise ValueError("Token IDs must be nonnegative integers")
    lengths = [len(prefix) + len(s) for s in suffixes]
    if row.get("branch_lengths") != lengths or row.get("branch_tokens") != max(lengths):
        raise ValueError("Stored branch lengths are inconsistent")
    if max(lengths) > max_length:
        raise ValueError("Prepared row exceeds the context window; no truncation")
    if bool(row.get("image")) != (row.get("modality") == "multimodal"):
        raise ValueError("Image and modality fields disagree")
    return row


def dataset_identity(train_path, development_path, image_root, max_length=8192):
    """Check IDs, split leakage and actual image bytes before a training stage."""
    all_ids = set()
    split_signatures = {}
    images = {}
    rows_by_split = {}
    for name, path in [("train", train_path), ("development", development_path)]:
        rows = list(read_jsonl(path))
        if not rows:
            raise ValueError(f"Empty {name} data")
        signatures = set()
        for row in rows:
            validate_prepared(row, max_length)
            if row["id"] in all_ids:
                raise ValueError("Duplicate example ID within or across splits")
            all_ids.add(row["id"])
            if row.get("group_hash"):
                signatures.add("group:" + row["group_hash"])
            signatures.add(
                "input:"
                + canonical_hash(
                    [
                        " ".join(row["state"].split()).casefold(),
                        " ".join(row["question"].split()).casefold(),
                        sorted(
                            " ".join(c["description"].split()).casefold()
                            for c in row["candidates"]
                        ),
                        row.get("image"),
                    ]
                )
            )
            if row.get("image") and row["image"] not in images:
                images[row["image"]] = sha256(image_path(image_root, row["image"]))
        split_signatures[name] = signatures
        rows_by_split[name] = rows
    if split_signatures["train"] & split_signatures["development"]:
        raise ValueError("Training/development evidence or input overlap")
    identity = {
        "train_sha256": sha256(train_path),
        "development_sha256": sha256(development_path),
        "image_files": dict(sorted(images.items())),
        "train_count": len(rows_by_split["train"]),
        "development_count": len(rows_by_split["development"]),
    }
    return rows_by_split, identity


def file_manifest(directory):
    root = Path(directory)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in checkpoint payloads")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    return result


def source_identity():
    root = Path(__file__).resolve().parent
    return {p.name: sha256(p) for p in sorted(root.glob("*.py"))}


@contextmanager
def output_lock(directory):
    """Protect a stage against two writers; an existing complete stage is immutable."""
    import fcntl

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".stage.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        yield
