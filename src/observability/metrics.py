from __future__ import annotations

import math


def _safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return int(values[0])
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return int(ordered[int(k)])
    d0 = ordered[f] * (c - k)
    d1 = ordered[c] * (k - f)
    return int(d0 + d1)


def summarize_eval_results(rows: list[dict]) -> dict:
    total = len(rows)
    if total == 0:
        return {
            "total_cases": 0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
            "citation_accuracy": 0.0,
            "grounded_rate": 0.0,
            "hallucination_rate": 0.0,
            "latency_ms": {"avg": 0, "p50": 0, "p95": 0},
        }

    hit5 = 0
    hit10 = 0
    citation_ok = 0
    grounded_ok = 0
    hallucinated = 0
    reciprocal_sum = 0.0
    latency_values: list[int] = []

    for row in rows:
        if int(row.get("hit_top5") or row.get("hit_expected_source") or 0) == 1:
            hit5 += 1
        if int(row.get("hit_top10") or row.get("hit_expected_source") or 0) == 1:
            hit10 += 1

        rank = row.get("rank_of_first_correct")
        if rank is not None and int(rank) >= 1:
            reciprocal_sum += 1.0 / int(rank)

        if int(row.get("citation_correct") or 0) == 1:
            citation_ok += 1
        if int(row.get("grounded") or 0) == 1:
            grounded_ok += 1
        if int(row.get("hallucination") or 0) == 1:
            hallucinated += 1

        lat = row.get("total_latency_ms")
        if lat is not None:
            latency_values.append(int(lat))

    avg_latency = int(sum(latency_values) / len(latency_values)) if latency_values else 0

    return {
        "total_cases": total,
        "recall_at_5": round(_safe_rate(hit5, total), 4),
        "recall_at_10": round(_safe_rate(hit10, total), 4),
        "mrr": round(reciprocal_sum / total, 4),
        "citation_accuracy": round(_safe_rate(citation_ok, total), 4),
        "grounded_rate": round(_safe_rate(grounded_ok, total), 4),
        "hallucination_rate": round(_safe_rate(hallucinated, total), 4),
        "latency_ms": {
            "avg": avg_latency,
            "p50": _percentile(latency_values, 0.50),
            "p95": _percentile(latency_values, 0.95),
        },
    }

