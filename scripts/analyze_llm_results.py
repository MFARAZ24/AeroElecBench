from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

Z_95 = 1.959963984540054
PAIRINGS = (("llm_only", "retrieval_grounded"), ("llm_only", "hybrid_explainer"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a completed AeroElecBench LLM experiment without calling Ollama")
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark/synthetic_benchmark.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/rules.json"))
    parser.add_argument("--responses", type=Path, default=Path("results/llm/ollama_responses.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("results/llm/llm_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows, malformed = [], 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rows.append(row) if isinstance(row, dict) else None
                malformed += int(not isinstance(row, dict))
            except json.JSONDecodeError:
                malformed += 1
    return rows, malformed


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scenario_hash(scenario: dict[str, Any]) -> str:
    payload = json.dumps(scenario, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def finding_key(item: dict[str, Any]) -> tuple[str, str]:
    return str(item.get("rule_id", "")), str(item.get("entity_path", ""))


def divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def prf(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision, recall = divide(tp, tp + fp), divide(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": divide(2 * precision * recall, precision + recall)}


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * proportion)))] if ordered else 0.0


def wilson(successes: int, total: int) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    rate, z2 = successes / total, Z_95**2
    center = (rate + z2 / (2 * total)) / (1 + z2 / total)
    margin = Z_95 * math.sqrt(rate * (1 - rate) / total + z2 / (4 * total**2)) / (1 + z2 / total)
    return max(0.0, center - margin), min(1.0, center + margin)


def exact_mcnemar(improved: int, regressed: int) -> float:
    discordant, smaller = improved + regressed, min(improved, regressed)
    if not discordant:
        return 1.0
    return min(1.0, 2 * sum(math.comb(discordant, i) for i in range(smaller + 1)) / 2**discordant)


def select_completed_rows(raw_rows: list[dict[str, Any]], manifest: dict[str, Any], scenarios: dict[str, dict[str, Any]], malformed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    models, modes, scenario_ids = manifest["models"], manifest["modes"], manifest["scenario_ids"]
    expected = {(model, mode, scenario_id) for model in models for mode in modes for scenario_id in scenario_ids}
    current_hashes = {scenario_id: scenario_hash(scenarios[scenario_id]) for scenario_id in scenario_ids}
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicates = stale = extra = invalid_shape = 0
    for row in raw_rows:
        try:
            key = row["model"], row["mode"], row["scenario_id"]
            if key not in expected:
                extra += 1; continue
            if row.get("scenario_sha256", "") != current_hashes[key[2]]:
                stale += 1; continue
            duplicates += int(key in selected); selected[key] = row
        except (KeyError, TypeError):
            invalid_shape += 1
    missing = sorted(expected - set(selected))
    integrity = {
        "raw_nonempty_line_count": len(raw_rows) + malformed, "parsed_row_count": len(raw_rows), "malformed_line_count": malformed,
        "expected_combination_count": len(expected), "usable_combination_count": len(selected), "duplicate_current_row_count": duplicates,
        "stale_benchmark_row_count": stale, "out_of_manifest_row_count": extra, "invalid_row_shape_count": invalid_shape,
        "missing_combination_count": len(missing), "missing_combinations": [" | ".join(item) for item in missing],
    }
    return [selected[key] for key in sorted(selected)], integrity


def make_observation(row: dict[str, Any], scenario: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    report, rules = row.get("report", {}), {rule["rule_id"]: rule for rule in catalog["rules"]}
    findings = report.get("findings", []) if isinstance(report.get("findings", []), list) else []
    expected, predicted = {finding_key(item) for item in scenario["ground_truth"]}, {finding_key(item) for item in findings if isinstance(item, dict)}
    matched, extras, missed = expected & predicted, predicted - expected, expected - predicted
    diagnostics = report.get("diagnostics", {}) if isinstance(report.get("diagnostics", {}), dict) else {}
    invalid = int(diagnostics.get("invalid_finding_count") or 0)
    parse_success, abstained = bool(diagnostics.get("parse_success", False)), bool(report.get("abstained", False))
    citation_correct = trace_complete = 0
    for finding in findings:
        citation, rule = finding.get("rule_citation", {}), rules.get(finding.get("rule_id"), {})
        citation_correct += int(citation.get("catalog_id") == catalog["catalog_id"] and citation.get("section") == rule.get("section") and citation.get("rule_id") == finding.get("rule_id"))
        trace_complete += int(all(finding.get(field) not in (None, "", {}) for field in ("entity_path", "entity_id", "explanation", "evidence", "rule_citation")))
    metadata = report.get("ollama_metadata", {}) if isinstance(report.get("ollama_metadata", {}), dict) else {}
    valid_decision = parse_success and not abstained and not invalid
    return {
        "model": row["model"], "mode": row["mode"], "scenario_id": row["scenario_id"], "category": scenario["category"],
        "expected": expected, "predicted": predicted, "matched": matched, "extras": extras, "missed": missed,
        "tp": len(matched), "fp": len(extras) + invalid, "fn": len(missed), "exact": expected == predicted and valid_decision,
        "clean_correct": not expected and not predicted and valid_decision, "parse_success": parse_success, "abstained": abstained,
        "invalid": invalid, "raw_findings": int(diagnostics.get("raw_finding_count") or len(findings)), "citation_correct": citation_correct,
        "citation_total": len(findings), "trace_complete": trace_complete, "trace_total": len(findings),
        "human_approval": bool(report.get("human_approval_required", False)), "automatic_modification": bool(report.get("automatic_modification_performed", False)),
        "latency_ms": float(row.get("latency_ms") or 0), "prompt_tokens": int(metadata.get("prompt_eval_count") or 0), "output_tokens": int(metadata.get("eval_count") or 0),
    }


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    tp, fp, fn, count = sum(item["tp"] for item in items), sum(item["fp"] for item in items), sum(item["fn"] for item in items), len(items)
    exact_count, clean_total = sum(item["exact"] for item in items), sum(not item["expected"] for item in items)
    clean_correct, raw_findings = sum(item["clean_correct"] for item in items), sum(item["raw_findings"] for item in items)
    latency = [item["latency_ms"] for item in items]
    ci_low, ci_high = wilson(exact_count, count)
    return {
        "scenario_count": count, **prf(tp, fp, fn), "exact_match_count": exact_count, "scenario_exact_match_accuracy": divide(exact_count, count),
        "exact_match_ci95_low": ci_low, "exact_match_ci95_high": ci_high, "clean_scenario_count": clean_total,
        "clean_design_specificity": divide(clean_correct, clean_total), "unsupported_claim_rate": divide(fp, tp + fp),
        "parse_success_rate": divide(sum(item["parse_success"] for item in items), count), "invalid_finding_rate": divide(sum(item["invalid"] for item in items), raw_findings),
        "abstention_rate": divide(sum(item["abstained"] for item in items), count), "citation_correctness": divide(sum(item["citation_correct"] for item in items), sum(item["citation_total"] for item in items)),
        "traceability_completeness": divide(sum(item["trace_complete"] for item in items), sum(item["trace_total"] for item in items)),
        "human_review_flag_coverage": divide(sum(item["human_approval"] for item in items), count), "automatic_modification_count": sum(item["automatic_modification"] for item in items),
        "prompt_tokens": sum(item["prompt_tokens"] for item in items), "output_tokens": sum(item["output_tokens"] for item in items),
        "mean_latency_ms": statistics.fmean(latency) if latency else 0.0, "median_latency_ms": statistics.median(latency) if latency else 0.0, "p95_latency_ms": percentile(latency, 0.95),
    }


def grouped_metrics(observations: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        groups[tuple(str(item[field]) for field in fields)].append(item)
    return [{**dict(zip(fields, key)), **aggregate(items)} for key, items in sorted(groups.items())]


def per_rule_metrics(observations: list[dict[str, Any]], rule_ids: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        groups[item["model"], item["mode"]].append(item)
    rows = []
    for (model, mode), items in sorted(groups.items()):
        for rule_id in rule_ids:
            tp = sum(sum(key[0] == rule_id for key in item["matched"]) for item in items)
            fp = sum(sum(key[0] == rule_id for key in item["extras"]) for item in items)
            fn = sum(sum(key[0] == rule_id for key in item["missed"]) for item in items)
            rows.append({"model": model, "mode": mode, "rule_id": rule_id, "support": tp + fn, **prf(tp, fp, fn)})
    return rows


def paired_metrics(observations: list[dict[str, Any]], overall: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(item["model"], item["mode"], item["scenario_id"]): item for item in observations}
    metrics = {(item["model"], item["mode"]): item for item in overall}
    models, scenario_ids = sorted({item["model"] for item in observations}), sorted({item["scenario_id"] for item in observations})
    rows = []
    for model in models:
        for baseline, challenger in PAIRINGS:
            pairs = [(index[model, baseline, sid], index[model, challenger, sid]) for sid in scenario_ids if (model, baseline, sid) in index and (model, challenger, sid) in index]
            improved = sum(not first["exact"] and second["exact"] for first, second in pairs)
            regressed = sum(first["exact"] and not second["exact"] for first, second in pairs)
            base, test = metrics[model, baseline], metrics[model, challenger]
            rows.append({
                "model": model, "baseline_mode": baseline, "challenger_mode": challenger, "paired_scenario_count": len(pairs),
                "improved_scenarios": improved, "regressed_scenarios": regressed, "tied_scenarios": len(pairs) - improved - regressed,
                "exact_match_delta": test["scenario_exact_match_accuracy"] - base["scenario_exact_match_accuracy"], "f1_delta": test["f1"] - base["f1"],
                "precision_delta": test["precision"] - base["precision"], "recall_delta": test["recall"] - base["recall"],
                "unsupported_claim_rate_delta": test["unsupported_claim_rate"] - base["unsupported_claim_rate"],
                "median_latency_ratio": divide(test["median_latency_ms"], base["median_latency_ms"]), "mcnemar_exact_p": exact_mcnemar(improved, regressed),
            })
    return rows


def failure_rows(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in observations:
        if item["exact"]:
            continue
        format_keys = lambda values: "; ".join(f"{rule}@{path}" for rule, path in sorted(values))
        rows.append({
            "model": item["model"], "mode": item["mode"], "scenario_id": item["scenario_id"], "category": item["category"],
            "expected": format_keys(item["expected"]), "predicted": format_keys(item["predicted"]), "false_positives": format_keys(item["extras"]),
            "false_negatives": format_keys(item["missed"]), "invalid_finding_count": item["invalid"], "parse_success": item["parse_success"],
            "abstained": item["abstained"], "latency_ms": item["latency_ms"],
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def markdown_report(integrity: dict[str, Any], overall: list[dict[str, Any]], categories: list[dict[str, Any]], paired: list[dict[str, Any]], failures: list[dict[str, Any]], profile: str) -> str:
    ranked = sorted(overall, key=lambda item: (item["f1"], item["scenario_exact_match_accuracy"], -item["unsupported_claim_rate"]), reverse=True)
    best = ranked[0]
    overall_lines = [f"| {item['model']} | {item['mode']} | {fmt(item['precision'])} | {fmt(item['recall'])} | {fmt(item['f1'])} | {fmt(item['scenario_exact_match_accuracy'])} | {fmt(item['unsupported_claim_rate'])} | {fmt(item['clean_design_specificity'])} | {item['median_latency_ms'] / 1000:.2f} |" for item in ranked]
    paired_lines = [f"| {item['model']} | {item['challenger_mode']} | {item['exact_match_delta']:+.3f} | {item['f1_delta']:+.3f} | {item['unsupported_claim_rate_delta']:+.3f} | {item['improved_scenarios']} | {item['regressed_scenarios']} | {item['mcnemar_exact_p']:.4f} |" for item in paired]
    category_lines = [f"| {item['model']} | {item['mode']} | {item['category']} | {item['scenario_count']} | {fmt(item['f1'])} | {fmt(item['scenario_exact_match_accuracy'])} |" for item in categories]
    return f"""# AeroElecBench LLM Pilot Analysis

Profile: **{profile}**. Integrity check: **PASS** ({integrity['usable_combination_count']}/{integrity['expected_combination_count']} expected model-mode-scenario combinations analyzed). No Ollama calls were made.

Best observed configuration by micro-F1 was **{best['model']} / {best['mode']}** with F1 **{best['f1']:.3f}** and exact-scenario accuracy **{best['scenario_exact_match_accuracy']:.3f}**.

## Overall results

| Model | Mode | Precision | Recall | F1 | Exact | Unsupported | Clean specificity | Median latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(overall_lines)}

## Paired changes from LLM-only

Positive exact/F1 deltas favor the challenger; a negative unsupported-claim delta is better. The exact McNemar p-value uses the same pilot scenarios and is descriptive, unadjusted for multiple comparisons.

| Model | Challenger | Exact delta | F1 delta | Unsupported delta | Improved | Regressed | Exact p |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(paired_lines)}

## Category breakdown

| Model | Mode | Category | N | F1 | Exact |
|---|---|---|---:|---:|---:|
{chr(10).join(category_lines)}

## Error audit

There are **{len(failures)}** non-exact model-mode-scenario results. Inspect `failure_cases.csv` before interpreting aggregate gains. Per-rule results are in `per_rule_metrics.csv`.

## Interpretation boundary

These are controlled synthetic pilot results over five fictional defect families. They support comparison within this benchmark only; they are not certification evidence or validation on proprietary industrial ECAD data. The 40-scenario pilot is suitable for debugging and preliminary evidence, while confirmatory claims require a separately justified evaluation design.
"""


def main() -> None:
    args = parse_args()
    required = (args.benchmark, args.catalog, args.responses, args.manifest)
    missing_files = [str(path) for path in required if not path.exists()]
    if missing_files:
        raise SystemExit(f"Missing required file(s): {', '.join(missing_files)}")
    scenarios_list, malformed_benchmark = load_jsonl(args.benchmark)
    if malformed_benchmark:
        raise SystemExit(f"Benchmark contains {malformed_benchmark} malformed line(s)")
    scenarios = {item["scenario_id"]: item for item in scenarios_list}
    catalog, manifest = load_json(args.catalog), load_json(args.manifest)
    unknown_ids = sorted(set(manifest.get("scenario_ids", [])) - set(scenarios))
    if unknown_ids:
        raise SystemExit(f"Manifest references unknown scenario IDs: {', '.join(unknown_ids)}")
    recorded_benchmark_hash = manifest.get("benchmark_sha256")
    if recorded_benchmark_hash and recorded_benchmark_hash != sha256_file(args.benchmark):
        raise SystemExit("Benchmark SHA-256 does not match the completed experiment manifest")
    raw_rows, malformed = load_jsonl(args.responses)
    selected_rows, integrity = select_completed_rows(raw_rows, manifest, scenarios, malformed)
    if integrity["missing_combination_count"]:
        sample = ", ".join(integrity["missing_combinations"][:5])
        raise SystemExit(f"Experiment is incomplete: {integrity['missing_combination_count']} combination(s) missing. First missing: {sample}")
    observations = [make_observation(row, scenarios[row["scenario_id"]], catalog) for row in selected_rows]
    overall = grouped_metrics(observations, ("model", "mode"))
    categories = grouped_metrics(observations, ("model", "mode", "category"))
    rules = per_rule_metrics(observations, [rule["rule_id"] for rule in catalog["rules"]])
    paired, failures = paired_metrics(observations, overall), failure_rows(observations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "overall_metrics.csv", overall)
    write_csv(args.output_dir / "category_metrics.csv", categories)
    write_csv(args.output_dir / "per_rule_metrics.csv", rules)
    write_csv(args.output_dir / "paired_mode_comparisons.csv", paired)
    write_csv(args.output_dir / "failure_cases.csv", failures)
    summary = {"profile": manifest.get("profile", "unknown"), "integrity": integrity, "overall": overall, "categories": categories, "per_rule": rules, "paired_mode_comparisons": paired, "failure_case_count": len(failures)}
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = markdown_report(integrity, overall, categories, paired, failures, summary["profile"])
    (args.output_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    best = max(overall, key=lambda item: (item["f1"], item["scenario_exact_match_accuracy"], -item["unsupported_claim_rate"]))
    print(json.dumps({"status": "complete", "analyzed_combinations": len(observations), "output_dir": str(args.output_dir), "best_model": best["model"], "best_mode": best["mode"], "best_f1": round(best["f1"], 4), "failure_cases": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
