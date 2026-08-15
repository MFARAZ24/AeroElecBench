from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .catalog import load_catalog
from .impact_benchmark import IMPACT_CASE_TYPES, generate_impact_benchmark, read_impact_benchmark, write_impact_benchmark
from .impact_graph import analyze_change_impact, compute_version_diff
from .validator import validate_design

DEFAULT_BENCHMARK = Path("benchmark/v07/impact_development_24.jsonl")
DEFAULT_OUTPUT = Path("results/impact_v07_development")
DEFAULT_SEED = 7107
BENCHMARK_ID = "AEROELECBENCH-IMPACT-DEVELOPMENT-0.7"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_key(path: dict[str, Any]) -> tuple[Any, ...]:
    return path["root_id"], path["target_id"], tuple(path["node_ids"]), tuple(path["relations"])


def _predicted_action(status: str) -> str:
    return {"completed": "report", "abstained": "abstain"}.get(status, status)


def _expected_diff_operations(scenario: dict[str, Any]) -> set[tuple[str, str]]:
    return {(item["op"], item["path"]) for item in scenario["change_request"]["operations"]}


def _observed_diff_operations(scenario: dict[str, Any]) -> set[tuple[str, str]]:
    return {(item["op"], item["path"]) for item in compute_version_diff(scenario["before_design"], scenario["after_design"])}


def validate_impact_benchmark(scenarios: list[dict[str, Any]], rules: list[dict[str, Any]]) -> None:
    if len({item["scenario_id"] for item in scenarios}) != len(scenarios):
        raise ValueError("Impact scenario identifiers must be unique")
    for scenario in scenarios:
        scenario_id, case_type = scenario["scenario_id"], scenario["impact_case_type"]
        if case_type not in IMPACT_CASE_TYPES or scenario.get("split") != "development":
            raise ValueError(f"Invalid development scenario metadata in {scenario_id}")
        if validate_design(scenario["before_design"], rules):
            raise ValueError(f"Before design is invalid in {scenario_id}")
        if case_type != "missing_information" and validate_design(scenario["after_design"], rules):
            raise ValueError(f"After design is invalid in {scenario_id}")
        expected_diff, observed_diff = _expected_diff_operations(scenario), _observed_diff_operations(scenario)
        if case_type != "missing_information" and expected_diff != observed_diff:
            raise ValueError(f"Structured change record does not match the version diff in {scenario_id}")
        before_copy, after_copy = copy.deepcopy(scenario["before_design"]), copy.deepcopy(scenario["after_design"])
        report = analyze_change_impact(scenario["before_design"], scenario["after_design"], scenario["change_request"])
        oracle = scenario["impact_oracle"]
        predicted_action = _predicted_action(report["status"])
        if predicted_action != oracle["expected_action"]:
            raise ValueError(f"Oracle action mismatch in {scenario_id}")
        if report["affected_node_ids"] != oracle["affected_node_ids"]:
            raise ValueError(f"Oracle affected-set mismatch in {scenario_id}")
        if {_path_key(item) for item in report["impact_paths"]} != {_path_key(item) for item in oracle["impact_paths"]}:
            raise ValueError(f"Oracle path mismatch in {scenario_id}")
        if scenario["before_design"] != before_copy or scenario["after_design"] != after_copy:
            raise ValueError(f"Impact analysis mutated an input in {scenario_id}")


def prepare_impact_development(
    benchmark_path: str | Path = DEFAULT_BENCHMARK,
    catalog_path: str | Path | None = None,
    source_registry_path: str | Path = "data/source_registry.json",
    seed: int = DEFAULT_SEED,
    cases_per_type: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark, registry_path = Path(benchmark_path), Path(source_registry_path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Source registry not found: {registry_path}")
    catalog = load_catalog(catalog_path)
    scenarios = generate_impact_benchmark(seed, cases_per_type)
    validate_impact_benchmark(scenarios, catalog["rules"])
    write_impact_benchmark(scenarios, benchmark)
    manifest = {
        "benchmark_id": BENCHMARK_ID, "benchmark": benchmark.name, "benchmark_sha256": _sha256(benchmark),
        "catalog_id": catalog["catalog_id"], "catalog_version": catalog["version"],
        "catalog_sha256": _sha256(Path(catalog_path or "data/rules.json")),
        "source_registry_id": json.loads(registry_path.read_text(encoding="utf-8"))["registry_id"],
        "source_registry_sha256": _sha256(registry_path), "seed": seed,
        "scenario_count": len(scenarios),
        "case_type_counts": dict(sorted(Counter(item["impact_case_type"] for item in scenarios).items())),
        "split_counts": dict(sorted(Counter(item["split"] for item in scenarios).items())),
        "oracle_validation_rate": 1.0, "oracle_exposed_to_model": False,
        "development_only": True, "heldout": False, "posthoc_tuning_allowed": True,
        "production_modifications_allowed": False, "dataset_kind": "fictional_synthetic",
        "rule_classification": "research_only", "certification_evidence": False, "narrative_output": False,
    }
    benchmark.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return scenarios, manifest


def _load_frozen_development(benchmark_path: str | Path, catalog_path: str | Path | None) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    benchmark, manifest_path = Path(benchmark_path), Path(benchmark_path).with_name("manifest.json")
    if not benchmark.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Development benchmark not found at {benchmark}. Run 'aeroecad impact-prototype --prepare-only' first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("benchmark_id") != BENCHMARK_ID or _sha256(benchmark) != manifest.get("benchmark_sha256"):
        raise ValueError("Development benchmark identity or hash mismatch")
    scenarios, catalog = read_impact_benchmark(benchmark), load_catalog(catalog_path)
    validate_impact_benchmark(scenarios, catalog["rules"])
    return scenarios, manifest, catalog


def evaluate_impact_records(scenarios: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_index = {item["scenario_id"]: item for item in scenarios}
    if len(records) != len(scenarios) or {item["scenario_id"] for item in records} != set(scenario_index):
        raise ValueError("Impact records do not cover every scenario exactly once")
    modes = {item.get("mode") for item in records}
    if len(modes) != 1 or None in modes:
        raise ValueError("Impact records must belong to exactly one named mode")
    mode = next(iter(modes))
    counts = defaultdict(int)
    per_case = defaultdict(lambda: defaultdict(int))
    for row in records:
        scenario, report = scenario_index[row["scenario_id"]], row["report"]
        oracle, case_type = scenario["impact_oracle"], scenario["impact_case_type"]
        expected_action = oracle["expected_action"]
        predicted_action = _predicted_action(report["status"])
        expected_nodes, predicted_nodes = set(oracle["affected_node_ids"]), set(report["affected_node_ids"])
        expected_paths, predicted_paths = {_path_key(item) for item in oracle["impact_paths"]}, {_path_key(item) for item in report["impact_paths"]}
        exact_set = expected_nodes == predicted_nodes
        correct_action = predicted_action == expected_action and (expected_action != "report" or exact_set)
        counts["scenario_count"] += 1
        counts["correct_action_count"] += int(correct_action)
        counts["llm_call_count"] += int(row.get("llm_call_count", 0))
        counts["rejected_output_count"] += int(report["status"] == "rejected")
        diagnostics = row.get("diagnostics") or {}
        counts["invalid_node_count"] += int(diagnostics.get("invalid_node_count", 0))
        counts["duplicate_node_count"] += int(diagnostics.get("duplicate_node_count", 0))
        counts["invalid_edge_count"] += int(diagnostics.get("invalid_edge_count", 0))
        counts["valid_edge_count"] += int(diagnostics.get("valid_edge_count", 0))
        counts["raw_edge_count"] += int(diagnostics.get("raw_edge_count", 0))
        counts["input_immutable_count"] += int(report["input_designs_unchanged"])
        counts["production_modification_count"] += int(report["production_modification_performed"])
        if expected_action == "report":
            counts["report_scenario_count"] += 1
            counts["true_positive"] += len(expected_nodes & predicted_nodes)
            counts["false_positive"] += len(predicted_nodes - expected_nodes)
            counts["false_negative"] += len(expected_nodes - predicted_nodes)
            counts["exact_set_count"] += int(exact_set)
            counts["path_true_positive"] += len(expected_paths & predicted_paths)
            counts["path_false_positive"] += len(predicted_paths - expected_paths)
            counts["path_false_negative"] += len(expected_paths - predicted_paths)
            multi_hop_targets = {item["target_id"] for item in oracle["impact_paths"] if len(item["relations"]) >= 2}
            predicted_targets = {item["target_id"] for item in report["impact_paths"]}
            counts["multi_hop_total"] += len(multi_hop_targets)
            counts["multi_hop_found"] += len(multi_hop_targets & predicted_targets)
            if mode == "text_rag":
                retrieved_nodes = set(row.get("retrieved_entity_ids", []))
                expected_retrieval_nodes = {node.lower() for node in expected_nodes}
                counts["retrieval_expected_node_count"] += len(expected_nodes)
                counts["retrieval_found_node_count"] += len(expected_retrieval_nodes & retrieved_nodes)
        if expected_action == "abstain":
            counts["expected_abstention_count"] += 1
            counts["correct_abstention_count"] += int(predicted_action == "abstain")
        if predicted_action == "abstain":
            counts["predicted_abstention_count"] += 1
            counts["correct_predicted_abstention_count"] += int(expected_action == "abstain")
        if expected_action == "no_change":
            counts["clean_scenario_count"] += 1
            counts["clean_preserved_count"] += int(predicted_action == "no_change")
        case = per_case[case_type]
        case["count"] += 1
        case["correct"] += int(correct_action)

    precision_denominator = counts["true_positive"] + counts["false_positive"]
    recall_denominator = counts["true_positive"] + counts["false_negative"]
    precision = counts["true_positive"] / precision_denominator if precision_denominator else 0.0
    recall = counts["true_positive"] / recall_denominator if recall_denominator else 0.0
    path_precision_denominator = counts["path_true_positive"] + counts["path_false_positive"]
    path_recall_denominator = counts["path_true_positive"] + counts["path_false_negative"]
    path_precision = counts["path_true_positive"] / path_precision_denominator if path_precision_denominator else 0.0
    path_recall = counts["path_true_positive"] / path_recall_denominator if path_recall_denominator else 0.0
    return {
        "mode": mode, "scenario_count": counts["scenario_count"],
        "report_scenario_count": counts["report_scenario_count"],
        "llm_call_count": counts["llm_call_count"], "rejected_output_count": counts["rejected_output_count"],
        "invalid_node_count": counts["invalid_node_count"], "duplicate_node_count": counts["duplicate_node_count"],
        "invalid_edge_count": counts["invalid_edge_count"], "valid_edge_count": counts["valid_edge_count"],
        "invalid_edge_rate": counts["invalid_edge_count"] / counts["raw_edge_count"] if counts["raw_edge_count"] else 0.0,
        "retrieval_node_recall": counts["retrieval_found_node_count"] / counts["retrieval_expected_node_count"] if mode == "text_rag" and counts["retrieval_expected_node_count"] else None,
        "impact_set_precision": precision, "impact_set_recall": recall,
        "impact_set_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "impact_exact_scenario_accuracy": counts["exact_set_count"] / counts["report_scenario_count"] if counts["report_scenario_count"] else 0.0,
        "path_precision": path_precision, "path_recall": path_recall,
        "path_f1": 2 * path_precision * path_recall / (path_precision + path_recall) if path_precision + path_recall else 0.0,
        "multi_hop_recall": counts["multi_hop_found"] / counts["multi_hop_total"] if counts["multi_hop_total"] else 0.0,
        "abstention_precision": counts["correct_predicted_abstention_count"] / counts["predicted_abstention_count"] if counts["predicted_abstention_count"] else 0.0,
        "abstention_recall": counts["correct_abstention_count"] / counts["expected_abstention_count"] if counts["expected_abstention_count"] else 0.0,
        "clean_case_specificity": counts["clean_preserved_count"] / counts["clean_scenario_count"] if counts["clean_scenario_count"] else 0.0,
        "oracle_action_accuracy": counts["correct_action_count"] / counts["scenario_count"] if counts["scenario_count"] else 0.0,
        "production_modification_count": counts["production_modification_count"],
        "input_immutability_rate": counts["input_immutable_count"] / counts["scenario_count"] if counts["scenario_count"] else 0.0,
        "per_case_type": {
            name: {"count": values["count"], "correct": values["correct"], "accuracy": values["correct"] / values["count"]}
            for name, values in sorted(per_case.items())
        },
    }


def _save_evaluation(metrics: dict[str, Any], records: list[dict[str, Any]], manifest: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "impact_records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (output / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    fields = tuple(key for key, value in metrics.items() if key != "per_case_type" and not isinstance(value, dict))
    with (output / "evaluation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: metrics[key] for key in fields})
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_impact_development(
    benchmark_path: str | Path = DEFAULT_BENCHMARK,
    output_dir: str | Path = DEFAULT_OUTPUT,
    catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    scenarios, benchmark_manifest, _ = _load_frozen_development(benchmark_path, catalog_path)
    records = [{
        "scenario_id": scenario["scenario_id"], "mode": "graph_deterministic",
        "report": analyze_change_impact(scenario["before_design"], scenario["after_design"], scenario["change_request"]),
    } for scenario in scenarios]
    metrics = evaluate_impact_records(scenarios, records)
    manifest = {
        "evaluation_id": "AEROELECBENCH-IMPACT-EVALUATION-0.7",
        "benchmark_id": benchmark_manifest["benchmark_id"], "benchmark_sha256": benchmark_manifest["benchmark_sha256"],
        "mode": "graph_deterministic", "oracle_exposed_to_model": False,
        "development_only": True, "heldout": False, "narrative_output": False,
        "metric_generation": "deterministic", "production_modifications_allowed": False,
    }
    _save_evaluation(metrics, records, manifest, Path(output_dir))
    return metrics
