from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .llm_review import LLMReviewAgent
from .ollama import OllamaClient

PROFILE_COUNTS = {
    "smoke": {"clean": 1, "single_per_rule": 1, "mixed_fault": 1},
    "pilot": {"clean": 5, "single_per_rule": 5, "mixed_fault": 10},
}


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[float], proportion: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * proportion)))]


def _metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision, recall = _safe_divide(tp, tp + fp), _safe_divide(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": _safe_divide(2 * precision * recall, precision + recall)}


def _finding_key(item: dict[str, Any]) -> tuple[str, str]:
    return str(item.get("rule_id", "")), str(item.get("entity_path", ""))


def select_scenarios(scenarios: list[dict[str, Any]], profile: str) -> list[dict[str, Any]]:
    if profile == "full":
        return scenarios
    if profile not in PROFILE_COUNTS:
        raise ValueError(f"Unknown profile '{profile}'. Choose smoke, pilot, or full.")
    counts, selected = PROFILE_COUNTS[profile], []
    selected.extend([item for item in scenarios if item["category"] == "clean"][:counts["clean"]])
    rules = sorted({truth["rule_id"] for item in scenarios for truth in item["ground_truth"]})
    for rule_id in rules:
        matches = [item for item in scenarios if item["category"] == "single_fault" and item["ground_truth"][0]["rule_id"] == rule_id]
        selected.extend(matches[:counts["single_per_rule"]])
    selected.extend([item for item in scenarios if item["category"] == "mixed_fault"][:counts["mixed_fault"]])
    return selected


def _scenario_hash(scenario: dict[str, Any]) -> str:
    payload = json.dumps(scenario, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_rows(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                rows[(row["model"], row["mode"], row["scenario_id"], row.get("scenario_sha256", ""))] = row
            except (json.JSONDecodeError, KeyError):
                continue
    return rows


def _evaluate_rows(rows: list[dict[str, Any]], scenarios: dict[str, dict[str, Any]], catalog: dict[str, Any]) -> dict[str, Any]:
    rules = {rule["rule_id"]: rule for rule in catalog["rules"]}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_rule = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    clean_correct = exact_matches = parsed = abstained = invalid_findings = raw_findings = 0
    citation_correct = citation_total = trace_complete = trace_total = automatic_modifications = 0
    latencies, prompt_tokens, output_tokens = [], 0, 0
    for row in rows:
        scenario, report = scenarios[row["scenario_id"]], row["report"]
        expected = {_finding_key(item) for item in scenario["ground_truth"]}
        predicted = {_finding_key(item) for item in report["findings"]}
        matched, extras, missed = expected & predicted, predicted - expected, expected - predicted
        invalid_count = report["diagnostics"]["invalid_finding_count"]
        totals["tp"] += len(matched); totals["fp"] += len(extras) + invalid_count; totals["fn"] += len(missed)
        for rule_id, _ in matched: per_rule[rule_id]["tp"] += 1
        for rule_id, _ in extras: per_rule[rule_id]["fp"] += 1
        for rule_id, _ in missed: per_rule[rule_id]["fn"] += 1
        successful_decision = report["diagnostics"]["parse_success"] and not report["abstained"] and not invalid_count
        exact_matches += int(expected == predicted and successful_decision)
        clean_correct += int(not expected and not predicted and successful_decision)
        parsed += int(report["diagnostics"]["parse_success"])
        abstained += int(report["abstained"])
        invalid_findings += invalid_count
        raw_findings += report["diagnostics"]["raw_finding_count"]
        automatic_modifications += int(report["automatic_modification_performed"])
        latencies.append(row["latency_ms"])
        prompt_tokens += int(report["ollama_metadata"].get("prompt_eval_count") or 0)
        output_tokens += int(report["ollama_metadata"].get("eval_count") or 0)
        for finding in report["findings"]:
            citation_total += 1; trace_total += 1
            citation, rule = finding.get("rule_citation", {}), rules.get(finding.get("rule_id"), {})
            citation_correct += int(citation.get("catalog_id") == catalog["catalog_id"] and citation.get("section") == rule.get("section") and citation.get("rule_id") == finding.get("rule_id"))
            trace_complete += int(all(finding.get(field) not in (None, "", {}) for field in ("entity_path", "entity_id", "explanation", "evidence", "rule_citation")))
    clean_total = sum(not scenarios[row["scenario_id"]]["ground_truth"] for row in rows)
    result = {
        "scenario_count": len(rows), **_metrics(**totals), "scenario_exact_match_accuracy": _safe_divide(exact_matches, len(rows)),
        "clean_design_specificity": _safe_divide(clean_correct, clean_total), "unsupported_claim_rate": _safe_divide(totals["fp"], totals["tp"] + totals["fp"]),
        "parse_success_rate": _safe_divide(parsed, len(rows)), "invalid_finding_rate": _safe_divide(invalid_findings, raw_findings), "abstention_rate": _safe_divide(abstained, len(rows)),
        "citation_correctness": _safe_divide(citation_correct, citation_total), "traceability_completeness": _safe_divide(trace_complete, trace_total),
        "automatic_modification_count": automatic_modifications, "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
        "latency_ms": {"mean": statistics.fmean(latencies) if latencies else 0.0, "median": statistics.median(latencies) if latencies else 0.0, "p95": _percentile(latencies, 0.95)},
        "per_rule": {rule_id: _metrics(**per_rule[rule_id]) for rule_id in sorted(rules)},
    }
    return result


def _save_outputs(summary: dict[str, Any], manifest: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "llm_benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "llm_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    fieldnames = ("model", "mode", "scenario_count", "precision", "recall", "f1", "scenario_exact_match_accuracy", "clean_design_specificity", "unsupported_claim_rate", "parse_success_rate", "abstention_rate", "citation_correctness", "traceability_completeness", "median_latency_ms")
    with (output_dir / "llm_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader()
        for model, modes in summary["models"].items():
            for mode, metrics in modes.items():
                writer.writerow({"model": model, "mode": mode, **{key: metrics[key] for key in fieldnames[2:-1]}, "median_latency_ms": metrics["latency_ms"]["median"]})
    table_rows = []
    for model, modes in summary["models"].items():
        for mode, metrics in modes.items():
            table_rows.append(f"| {model} | {mode} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['scenario_exact_match_accuracy']:.3f} | {metrics['unsupported_claim_rate']:.3f} | {metrics['citation_correctness']:.3f} | {metrics['parse_success_rate']:.3f} | {metrics['latency_ms']['median'] / 1000:.2f} s |")
    note = f"""# AeroECAD-Agent v0.2 Ollama Results

Profile: **{summary['profile']}**; scenarios per model/mode: **{summary['scenario_count']}**. Models ran locally through Ollama with temperature 0 and a fixed seed.

| Model | Mode | Precision | Recall | F1 | Exact scenario | Unsupported claims | Citation correctness | Parse success | Median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

`llm_only` receives rule identifiers, titles, and citation metadata but no rule criteria. `retrieval_grounded` receives the retrieved fictional rule criteria. `hybrid_explainer` receives deterministic candidate findings and relevant rules, then verifies and explains them. All modes require human approval and make no automatic design changes.

These controlled synthetic results measure executable behavior within five fictional ECAD defect families. They are not certification evidence and do not demonstrate performance on proprietary industrial data.
"""
    (output_dir / "llm_preliminary_results.md").write_text(note, encoding="utf-8")


def run_llm_experiment(scenarios: list[dict[str, Any]], catalog: dict[str, Any], models: list[str], modes: list[str], profile: str, output_dir: str | Path, base_url: str = "http://localhost:11434", timeout: float = 300.0, seed: int = 2027, max_tokens: int = 1200, benchmark_path: str | Path | None = None) -> dict[str, Any]:
    selected, output = select_scenarios(scenarios, profile), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    client = OllamaClient(base_url, timeout); client.ensure_models(models)
    agent, response_path = LLMReviewAgent(catalog, client), output / "ollama_responses.jsonl"
    existing, scenario_index = _load_rows(response_path), {item["scenario_id"]: item for item in selected}
    total = len(models) * len(modes) * len(selected)
    scenario_hashes = {item["scenario_id"]: _scenario_hash(item) for item in selected}
    already_count = sum((model, mode, item["scenario_id"], scenario_hashes[item["scenario_id"]]) in existing for model in models for mode in modes for item in selected)
    completed_now = 0
    with response_path.open("a", encoding="utf-8") as handle:
        for model in models:
            for scenario in selected:
                for mode in modes:
                    scenario_sha = scenario_hashes[scenario["scenario_id"]]
                    key = (model, mode, scenario["scenario_id"], scenario_sha)
                    if key in existing:
                        continue
                    position = already_count + completed_now + 1
                    print(f"[{position}/{total}] {model} | {mode} | {scenario['scenario_id']}", flush=True)
                    started = time.perf_counter()
                    report = agent.review(scenario["design"], scenario["review_query"], model, mode, seed=seed, max_tokens=max_tokens)
                    row = {"model": model, "mode": mode, "scenario_id": scenario["scenario_id"], "scenario_sha256": scenario_sha, "category": scenario["category"], "latency_ms": (time.perf_counter() - started) * 1000, "report": report}
                    handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n"); handle.flush()
                    existing[key] = row; completed_now += 1
    grouped: dict[str, dict[str, Any]] = {}
    for model in models:
        grouped[model] = {}
        for mode in modes:
            rows = [existing[(model, mode, item["scenario_id"], scenario_hashes[item["scenario_id"]])] for item in selected]
            grouped[model][mode] = _evaluate_rows(rows, scenario_index, catalog)
    summary = {"prototype": "AeroECAD-Agent", "version": "0.2.0", "profile": profile, "scenario_count": len(selected), "models": grouped, "interpretation": "Controlled synthetic execution study; not certification or proprietary-data evidence."}
    benchmark_sha = hashlib.sha256(Path(benchmark_path).read_bytes()).hexdigest() if benchmark_path and Path(benchmark_path).exists() else None
    manifest = {"seed": seed, "profile": profile, "scenario_ids": [item["scenario_id"] for item in selected], "models": models, "modes": modes, "temperature": 0, "max_tokens": max_tokens, "base_url": base_url, "benchmark_sha256": benchmark_sha}
    _save_outputs(summary, manifest, output)
    return summary
