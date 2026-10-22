"""Typed runtime schemas; candidate IDs never enter the encoder."""

import math


def validate(row, max_candidates=150):
    for name in ["state", "question"]:
        if not isinstance(row.get(name), str) or (
            name == "question" and not row[name].strip()
        ):
            raise ValueError(
                f"{name} must be a string"
                + ("" if name == "state" else " with content")
            )
    typ = row.get("type", "choice")
    if typ not in ["choice", "boolean", "score"]:
        raise ValueError("Unknown decision type")
    cs = row.get("candidates")
    if not isinstance(cs, list) or not 2 <= len(cs) <= max_candidates:
        raise ValueError(f"Expected 2..{max_candidates} candidates")
    ids = []
    for c in cs:
        if not isinstance(c, dict):
            raise ValueError("Candidate must be an object")
        if not isinstance(c.get("id"), str) or not c["id"].strip():
            raise ValueError("Candidate IDs must be nonempty strings")
        if not isinstance(c.get("description"), str) or not c["description"].strip():
            raise ValueError("Candidate descriptions must be nonempty strings")
        ids.append(c["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be unique")
    if typ == "boolean" and set(ids) != {"false", "true"}:
        raise ValueError("Boolean candidates must use false and true IDs")
    if typ == "score":
        levels = [c.get("level") for c in cs]
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in levels
        ):
            raise ValueError("Score candidates need finite numeric levels")
        if any(a >= b for a, b in zip(levels, levels[1:])):
            raise ValueError("Score levels must be strictly increasing")
    return row
