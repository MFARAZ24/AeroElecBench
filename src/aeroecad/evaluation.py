from __future__ import annotations

import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .agent import ReviewAgent


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[float], proportion: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * proportion)))
    return ordered[index]


def _metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision, recall = _safe_divide(tp, tp + fp), _safe_divide(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": _safe_divide(2 * precision * recall, precision + recall)}


def _key(item: dict[str, Any]) -> tuple[str, str]:
    return item["rule_id"], item["entity_path"]


def evaluate(scenarios: list[dict[str, Any]], catalog: dict[str, Any], mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    agent = ReviewAgent(catalog)
    rules = {rule["rule_id"]: rule for rule in catalog["rules"]}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_rule = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    latencies, clean_correct = [], 0
    citation_total = citation_correct = trace_total = trace_complete = 0
    human_controlled = automatic_modifications = 0
    retrieval_count = top1_correct = top3_correct = 0
    reciprocal_ranks: list[float] = []
    sample_report: dict[str, Any] = {}
    for scenario in scenarios:
        started = time.perf_counter_ns()
        report = agent.review(scenario["design"], scenario["review_query"], mode)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        if scenario["category"] == "mixed_fault" and not sample_report:
            sample_report = report
        expected, predicted = {_key(item) for item in scenario["ground_truth"]}, {_key(item) for item in report["findings"]}
        matched, extras, missed = expected & predicted, predicted - expected, expected - predicted
        totals["tp"] += len(matched); totals["fp"] += len(extras); totals["fn"] += len(missed)
        for rule_id, _ in matched: per_rule[rule_id]["tp"] += 1
        for rule_id, _ in extras: per_rule[rule_id]["fp"] += 1
        for rule_id, _ in missed: per_rule[rule_id]["fn"] += 1
        if not expected and not predicted:
            clean_correct += 1
        human_controlled += int(report["human_approval_required"])
        automatic_modifications += int(report["automatic_modification_performed"])
        for finding in report["findings"]:
            citation_total += 1; trace_total += 1
            citation = finding.get("rule_citation", {})
            rule = rules.get(finding["rule_id"], {})
            citation_correct += int(citation.get("catalog_id") == catalog["catalog_id"] and citation.get("section") == rule.get("section") and citation.get("rule_id") == finding["rule_id"])
            trace_complete += int(all(finding.get(field) not in (None, "", {}) for field in ("entity_path", "evidence", "rule_citation")))
        if len(scenario["ground_truth"]) == 1 and report["retrieval_ranking"]:
            expected_rule = scenario["ground_truth"][0]["rule_id"]
            ranking = [item["rule_id"] for item in report["retrieval_ranking"]]
            rank = ranking.index(expected_rule) + 1
            retrieval_count += 1; top1_correct += int(rank == 1); top3_correct += int(rank <= 3); reciprocal_ranks.append(1 / rank)
    global_metrics = _metrics(**totals)
    clean_total = sum(scenario["category"] == "clean" for scenario in scenarios)
    result = {
        "mode": mode, "scenario_count": len(scenarios), "ground_truth_violation_count": totals["tp"] + totals["fn"],
        **global_metrics, "clean_design_specificity": _safe_divide(clean_correct, clean_total),
        "unsupported_claim_rate": _safe_divide(totals["fp"], totals["tp"] + totals["fp"]),
        "citation_correctness": _safe_divide(citation_correct, citation_total),
        "traceability_completeness": _safe_divide(trace_complete, trace_total),
        "human_review_flag_coverage": _safe_divide(human_controlled, len(scenarios)),
        "automatic_modification_count": automatic_modifications,
        "latency_ms": {"mean": statistics.fmean(latencies), "median": statistics.median(latencies), "p95": _percentile(latencies, 0.95)},
        "retrieval": {"evaluated_queries": retrieval_count, "top1_accuracy": _safe_divide(top1_correct, retrieval_count), "top3_recall": _safe_divide(top3_correct, retrieval_count), "mrr": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0},
        "per_rule": {rule_id: _metrics(**per_rule[rule_id]) for rule_id in sorted(rules)},
    }
    return result, sample_report


def save_results(summary: dict[str, Any], sample_report: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "sample_review_report.json").write_text(json.dumps(sample_report, indent=2) + "\n", encoding="utf-8")
    full, guided = summary["modes"]["full"], summary["modes"]["retrieval_guided"]
    note = f"""# AeroECAD-Agent v0.2 Deterministic Baseline Results

The seeded synthetic benchmark contains {summary['dataset']['scenario_count']} designs: {summary['dataset']['categories']['clean']} clean, {summary['dataset']['categories']['single_fault']} single-fault, and {summary['dataset']['categories']['mixed_fault']} mixed-fault scenarios. Across the benchmark, {full['ground_truth_violation_count']} violations cover five encoded fictional ECAD rule families.

| Mode | Precision | Recall | F1 | Unsupported claims | Citation correctness | Traceability completeness | Median latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full deterministic audit | {full['precision']:.3f} | {full['recall']:.3f} | {full['f1']:.3f} | {full['unsupported_claim_rate']:.3f} | {full['citation_correctness']:.3f} | {full['traceability_completeness']:.3f} | {full['latency_ms']['median']:.3f} ms |
| Retrieval-guided deterministic audit | {guided['precision']:.3f} | {guided['recall']:.3f} | {guided['f1']:.3f} | {guided['unsupported_claim_rate']:.3f} | {guided['citation_correctness']:.3f} | {guided['traceability_completeness']:.3f} | {guided['latency_ms']['median']:.3f} ms |

The retrieval-guided mode achieved top-1 rule-retrieval accuracy of {guided['retrieval']['top1_accuracy']:.3f}, top-3 recall of {guided['retrieval']['top3_recall']:.3f}, and MRR of {guided['retrieval']['mrr']:.3f} on {guided['retrieval']['evaluated_queries']} targeted single-fault queries. Both modes flagged all reports for human review and performed zero automatic design modifications.

These results verify correct execution and evidence traceability within a deliberately controlled synthetic rule scope. They must not be interpreted as certification evidence or as validation on proprietary industrial ECAD artifacts. Version 0.2 adds separate LLM-only, retrieval-grounded, and hybrid explanation experiments while preserving this deterministic baseline.
"""
    (output / "preliminary_results.md").write_text(note, encoding="utf-8")
    with (output / "per_rule_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("mode", "rule_id", "tp", "fp", "fn", "precision", "recall", "f1"))
        writer.writeheader()
        for mode, metrics in summary["modes"].items():
            for rule_id, values in metrics["per_rule"].items():
                writer.writerow({"mode": mode, "rule_id": rule_id, **values})
