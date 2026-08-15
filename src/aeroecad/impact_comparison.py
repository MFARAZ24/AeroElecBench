from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .impact_agent import IMPACT_MODES, ImpactAgent
from .impact_evaluation import DEFAULT_BENCHMARK, _load_frozen_development, evaluate_impact_records
from .ollama import OllamaClient

DEFAULT_COMPARISON_OUTPUT = Path("results/impact_v071_comparison")
EVALUATION_ID = "AEROELECBENCH-IMPACT-COMPARISON-0.7.1"
PIPELINE_VERSION = "0.7.1"


def _select_scenarios(scenarios: list[dict[str, Any]], profile: str) -> list[dict[str, Any]]:
    if profile == "development":
        return scenarios
    if profile != "smoke":
        raise ValueError("profile must be smoke or development")
    selected, seen = [], set()
    for scenario in scenarios:
        if scenario["impact_case_type"] not in seen:
            selected.append(scenario)
            seen.add(scenario["impact_case_type"])
    return selected


def _read_existing(path: Path, mode: str, model: str, benchmark_sha256: str, allowed_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len({row.get("scenario_id") for row in rows}):
        raise ValueError(f"Duplicate scenario records in {path}")
    for row in rows:
        expected_model = None if mode == "graph_deterministic" else model
        if row.get("pipeline_version") != PIPELINE_VERSION or row.get("mode") != mode or row.get("model") != expected_model or row.get("benchmark_sha256") != benchmark_sha256 or row.get("scenario_id") not in allowed_ids:
            raise ValueError(f"Existing record provenance mismatch in {path}")
    return {row["scenario_id"]: row for row in rows}


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_outputs(output: Path, aggregate: dict[str, Any], manifest: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    fields = [
        "mode", "scenario_count", "report_scenario_count", "llm_call_count", "rejected_output_count",
        "invalid_node_count", "duplicate_node_count", "invalid_edge_count", "valid_edge_count", "invalid_edge_rate",
        "retrieval_node_recall",
        "impact_set_precision", "impact_set_recall", "impact_set_f1", "impact_exact_scenario_accuracy",
        "path_precision", "path_recall", "path_f1", "multi_hop_recall", "abstention_precision",
        "abstention_recall", "clean_case_specificity", "oracle_action_accuracy",
        "production_modification_count", "input_immutability_rate",
    ]
    with (output / "evaluation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode in aggregate["modes"]:
            writer.writerow({field: aggregate["modes"][mode][field] for field in fields})
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_impact_comparison(
    model: str = "qwen2.5:7b",
    modes: tuple[str, ...] = IMPACT_MODES,
    benchmark_path: str | Path = DEFAULT_BENCHMARK,
    output_dir: str | Path = DEFAULT_COMPARISON_OUTPUT,
    catalog_path: str | Path | None = None,
    base_url: str = "http://localhost:11434",
    timeout: float = 300.0,
    seed: int = 7107,
    max_tokens: int = 2500,
    retrieval_top_k: int = 12,
    profile: str = "development",
    client: OllamaClient | None = None,
) -> dict[str, Any]:
    unknown = sorted(set(modes) - set(IMPACT_MODES))
    if unknown:
        raise ValueError(f"Unknown impact modes: {', '.join(unknown)}")
    scenarios, benchmark_manifest, _ = _load_frozen_development(benchmark_path, catalog_path)
    selected = _select_scenarios(scenarios, profile)
    output, allowed_ids = Path(output_dir), {item["scenario_id"] for item in selected}
    runtime = client or OllamaClient(base_url, timeout)
    if any(mode != "graph_deterministic" for mode in modes):
        runtime.ensure_models([model])
    agent, metrics_by_mode = ImpactAgent(runtime), {}
    for mode in modes:
        record_path = output / mode / "impact_records.jsonl"
        existing = _read_existing(record_path, mode, model, benchmark_manifest["benchmark_sha256"], allowed_ids)
        for scenario in selected:
            if scenario["scenario_id"] in existing:
                continue
            record = agent.run(scenario, model, mode, seed, max_tokens, retrieval_top_k)
            record["benchmark_sha256"] = benchmark_manifest["benchmark_sha256"]
            record["pipeline_version"] = PIPELINE_VERSION
            _append(record_path, record)
            existing[scenario["scenario_id"]] = record
        records = [existing[scenario["scenario_id"]] for scenario in selected]
        metrics_by_mode[mode] = evaluate_impact_records(selected, records)
    aggregate = {
        "evaluation_id": EVALUATION_ID, "model": model, "profile": profile,
        "scenario_count": len(selected), "modes": {mode: metrics_by_mode[mode] for mode in modes},
    }
    manifest = {
        "evaluation_id": EVALUATION_ID, "benchmark_id": benchmark_manifest["benchmark_id"],
        "benchmark_sha256": benchmark_manifest["benchmark_sha256"], "model": model,
        "modes": list(modes), "profile": profile, "scenario_count": len(selected),
        "seed": seed, "temperature": 0, "max_tokens": max_tokens, "retrieval_top_k": retrieval_top_k,
        "oracle_exposed_to_model": False, "root_node_ids_exposed_to_model": False,
        "pipeline_version": PIPELINE_VERSION, "metric_generation": "deterministic",
        "text_retrieval_method": "paired_entity_iterative_lexical_link_expansion_v01",
        "model_output_contract": "affected_nodes_plus_evidence_edges", "path_reconstruction": "model_edges_only",
        "agent_policy": "bounded_ordered_tool_plan_v01", "agent_model_role": "ordered_tool_plan_selection_only",
        "agent_impact_result_source": "validated_deterministic_tool_execution",
        "development_only": True, "heldout": False,
        "posthoc_tuning_allowed": True, "production_modifications_allowed": False, "narrative_output": False,
    }
    _write_outputs(output, aggregate, manifest)
    return aggregate
