"""Standard-library metrics also used by independent local result verification."""

import collections
import math


def probabilities(z, temp=1.0):
    z = [x / temp for x in z]
    v = max(z)
    es = [math.exp(x - v) for x in z]
    s = sum(es)
    return [x / s for x in es]


def metrics(records, temp=1.0):
    totals = [0.0, 0.0, 0.0]
    bins = [[0, 0.0, 0] for _ in range(15)]
    for r in records:
        z = [x / temp for x in r["logits"]]
        m = max(z)
        p = probabilities(r["logits"], temp)
        y = r["gold_index"]
        pred = max(range(len(p)), key=p.__getitem__)
        correct = pred == y
        confidence = p[pred]
        totals[0] += correct
        totals[1] += m + math.log(sum(math.exp(x - m) for x in z)) - z[y]
        totals[2] += sum((v - (i == y)) ** 2 for i, v in enumerate(p))
        b = bins[min(14, int(confidence * 15))]
        b[0] += 1
        b[1] += confidence
        b[2] += correct
    n = len(records)
    return {
        "n": n,
        "accuracy": totals[0] / n,
        "nll": totals[1] / n,
        "brier": totals[2] / n,
        "ece_15": sum(abs(b[1] - b[2]) for b in bins) / n,
        "temperature": temp,
    }


def fit_temperature(records):
    # Convex objective in inverse temperature, searched over fixed positive range.
    lo, hi = math.log(0.05), math.log(100.0)
    ratio = (math.sqrt(5) - 1) / 2

    def objective(x):
        return metrics(records, math.exp(x))["nll"]

    c = hi - ratio * (hi - lo)
    d = lo + ratio * (hi - lo)
    fc, fd = objective(c), objective(d)
    for _ in range(80):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - ratio * (hi - lo)
            fc = objective(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + ratio * (hi - lo)
            fd = objective(d)
    t = math.exp((lo + hi) / 2)
    return t if metrics(records, t)["nll"] < metrics(records, 1.0)["nll"] else 1.0


def report(records, temp=1.0):
    result = {"overall": metrics(records, temp), "slices": {}}
    for field in ["modality", "schema", "domain", "answerability"]:
        groups = collections.defaultdict(list)
        for r in records:
            groups[r[field]].append(r)
        result["slices"][field] = {
            k: metrics(v, temp) for k, v in sorted(groups.items())
        }
    for field in ["candidate_count", "length"]:
        groups = collections.defaultdict(list)
        for r in records:
            value = (
                len(r["logits"])
                if field == "candidate_count"
                else next(x for x in [512, 2048, 4096, 8192] if r["branch_tokens"] <= x)
            )
            groups[str(value)].append(r)
        result["slices"][field] = {
            k: metrics(v, temp) for k, v in sorted(groups.items())
        }
    result["unique_fact_groups"] = len({r["group_hash"] for r in records})
    return result
