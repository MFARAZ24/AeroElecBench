from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from .impact_agent import IMPACT_MODES, ImpactAgent, impact_llm_call_required
from .impact_evaluation import BENCHMARK_ID as DEVELOPMENT_BENCHMARK_ID
from .impact_evaluation import DEFAULT_BENCHMARK, evaluate_impact_records, load_frozen_impact_benchmark
from .ollama import OllamaClient

DEFAULT_COMPARISON_OUTPUT = Path("results/impact_v071_comparison")
EVALUATION_ID = "AEROELECBENCH-IMPACT-COMPARISON-0.7.1"
PIPELINE_VERSION = "0.7.1"


def _select_scenarios(scenarios: list[dict[str, Any]], profile: str) -> list[dict[str, Any]]:
    if profile in {"development", "heldout"}:
        return scenarios
    if profile != "smoke":
        raise ValueError("profile must be smoke, development, or heldout")
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
    expected_benchmark_id: str = DEVELOPMENT_BENCHMARK_ID,
    expected_split: str = "development",
    evaluation_id: str = EVALUATION_ID,
    development_only: bool = True,
    heldout: bool = False,
    posthoc_tuning_allowed: bool = True,
) -> dict[str, Any]:
    unknown = sorted(set(modes) - set(IMPACT_MODES))
    if unknown:
        raise ValueError(f"Unknown impact modes: {', '.join(unknown)}")
    scenarios, benchmark_manifest, _ = load_frozen_impact_benchmark(benchmark_path, catalog_path, expected_benchmark_id, expected_split)
    selected = _select_scenarios(scenarios, profile)
    output, allowed_ids = Path(output_dir), {item["scenario_id"] for item in selected}
    runtime = client or OllamaClient(base_url, timeout)
    if any(mode != "graph_deterministic" for mode in modes):
        runtime.ensure_models([model])
    agent, metrics_by_mode = ImpactAgent(runtime), {}
    existing_by_mode = {
        mode: _read_existing(output / mode / "impact_records.jsonl", mode, model, benchmark_manifest["benchmark_sha256"], allowed_ids)
        for mode in modes
    }
    expected_calls = sum(impact_llm_call_required(scenario, mode) for mode in modes for scenario in selected)
    completed_calls = sum(int(row.get("llm_call_count", 0)) for rows in existing_by_mode.values() for row in rows.values())
    print(f"[impact] profile={profile} scenarios={len(selected)} Qwen calls: {completed_calls}/{expected_calls} complete, {expected_calls - completed_calls} remaining", flush=True)
    for mode in modes:
        record_path = output / mode / "impact_records.jsonl"
        existing = existing_by_mode[mode]
        mode_expected = sum(impact_llm_call_required(scenario, mode) for scenario in selected)
        mode_completed = sum(int(row.get("llm_call_count", 0)) for row in existing.values())
        print(f"[impact][{mode}] records={len(existing)}/{len(selected)} Qwen calls={mode_completed}/{mode_expected}", flush=True)
        for scenario_index, scenario in enumerate(selected, start=1):
            if scenario["scenario_id"] in existing:
                continue
            requires_call = impact_llm_call_required(scenario, mode)
            if requires_call:
                print(f"[impact][{mode}] scenario {scenario_index}/{len(selected)} {scenario['scenario_id']} - starting Qwen call {completed_calls + 1}/{expected_calls}; {expected_calls - completed_calls} including current", flush=True)
            else:
                print(f"[impact][{mode}] scenario {scenario_index}/{len(selected)} {scenario['scenario_id']} - deterministic path", flush=True)
            started = time.perf_counter()
            record = agent.run(scenario, model, mode, seed, max_tokens, retrieval_top_k)
            record["benchmark_sha256"] = benchmark_manifest["benchmark_sha256"]
            record["pipeline_version"] = PIPELINE_VERSION
            _append(record_path, record)
            existing[scenario["scenario_id"]] = record
            completed_calls += int(record.get("llm_call_count", 0))
            print(f"[impact][{mode}] completed {len(existing)}/{len(selected)} status={record['report']['status']} elapsed={time.perf_counter() - started:.1f}s; Qwen remaining={expected_calls - completed_calls}", flush=True)
        records = [existing[scenario["scenario_id"]] for scenario in selected]
        metrics_by_mode[mode] = evaluate_impact_records(selected, records)
    aggregate = {
        "evaluation_id": evaluation_id, "model": model, "profile": profile,
        "scenario_count": len(selected), "modes": {mode: metrics_by_mode[mode] for mode in modes},
    }
    manifest = {
        "evaluation_id": evaluation_id, "benchmark_id": benchmark_manifest["benchmark_id"],
        "benchmark_sha256": benchmark_manifest["benchmark_sha256"], "model": model,
        "modes": list(modes), "profile": profile, "scenario_count": len(selected),
        "seed": seed, "temperature": 0, "max_tokens": max_tokens, "retrieval_top_k": retrieval_top_k,
        "oracle_exposed_to_model": False, "root_node_ids_exposed_to_model": False,
        "pipeline_version": PIPELINE_VERSION, "metric_generation": "deterministic",
        "text_retrieval_method": "paired_entity_iterative_lexical_link_expansion_v01",
        "model_output_contract": "affected_nodes_plus_evidence_edges", "path_reconstruction": "model_edges_only",
        "agent_policy": "bounded_ordered_tool_plan_v01", "agent_model_role": "ordered_tool_plan_selection_only",
        "agent_impact_result_source": "validated_deterministic_tool_execution",
        "development_only": development_only, "heldout": heldout,
        "posthoc_tuning_allowed": posthoc_tuning_allowed,
        "production_modifications_allowed": False, "narrative_output": False,
    }
    _write_outputs(output, aggregate, manifest)
    return aggregate
