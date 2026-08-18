from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from .impact_graph import build_version_graph, traverse_impact
from .impact_intent import DEFAULT_BENCHMARK, evaluate_intent_predictions, load_frozen_intent_benchmark, resolve_candidate_roots
from .ollama import OllamaClient

DEFAULT_OUTPUT = Path("results/impact_v11_intent_qwen")
EVALUATION_ID = "AEROELECBENCH-IMPACT-INTENT-QWEN-DEVELOPMENT-1.1"
PIPELINE_VERSION = "1.1.0"
MODE = "qwen_intent_graph"
SYSTEM_PROMPT = """You ground an engineering change request to an observed before/after electrical-design revision. Select only change_inventory candidates explicitly intended by the request. Genuine but unrelated edits are distractors. Report when at least one candidate is unambiguously intended; otherwise abstain. Use only supplied candidate IDs. Do not perform graph propagation or invent roots, nodes, or changes. Return one JSON object only."""
RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["report", "abstain"]},
        "selected_candidate_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "rationale": {"type": "string", "maxLength": 300},
    },
    "required": ["action", "selected_candidate_ids", "rationale"],
}


def _select_scenarios(scenarios: list[dict[str, Any]], profile: str) -> list[dict[str, Any]]:
    if profile == "development":
        return scenarios
    if profile != "smoke":
        raise ValueError("profile must be smoke or development")
    selected, seen = [], set()
    for scenario in scenarios:
        if scenario["intent_case_type"] not in seen:
            selected.append(scenario)
            seen.add(scenario["intent_case_type"])
    return selected


def build_intent_prompt(scenario: dict[str, Any]) -> str:
    payload = {
        "task": "Select the observed change candidates intended by the engineering request, or abstain if the request is ambiguous or unmatched.",
        "engineering_change_request": scenario["engineering_change_request"],
        "before_design": scenario["before_design"], "after_design": scenario["after_design"],
        "change_inventory": scenario["change_inventory"],
        "decision_contract": {
            "report": "Use when one or more supplied candidate IDs are clearly within requested scope.",
            "abstain": "Use with an empty selected_candidate_ids list when the request is ambiguous or matches no observed candidate.",
            "distractors": "Reject all genuine observed edits that are outside the request, including edits on the same entity.",
        },
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _extract_json(content: str) -> dict[str, Any] | None:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def parse_intent_response(content: str, scenario: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _extract_json(content)
    rejected = {"action": "rejected", "selected_candidate_ids": [], "rationale": ""}
    if payload is None:
        return rejected, {"parse_success": False, "contract_valid": False, "error": "invalid_json", "invalid_candidate_ids": []}
    action, selected = payload.get("action"), payload.get("selected_candidate_ids")
    if action not in {"report", "abstain"} or not isinstance(selected, list) or any(not isinstance(item, str) for item in selected):
        return rejected, {"parse_success": True, "contract_valid": False, "error": "invalid_contract", "invalid_candidate_ids": []}
    allowed = {item["candidate_id"] for item in scenario["change_inventory"]}
    invalid = sorted(set(selected) - allowed)
    if invalid:
        return rejected, {"parse_success": True, "contract_valid": False, "error": "unknown_candidate_id", "invalid_candidate_ids": invalid}
    if len(selected) != len(set(selected)) or (action == "report") != bool(selected):
        return rejected, {"parse_success": True, "contract_valid": False, "error": "inconsistent_decision", "invalid_candidate_ids": []}
    ordered = [item["candidate_id"] for item in scenario["change_inventory"] if item["candidate_id"] in set(selected)]
    selection = {"action": action, "selected_candidate_ids": ordered, "rationale": str(payload.get("rationale", ""))[:300]}
    return selection, {"parse_success": True, "contract_valid": True, "error": "", "invalid_candidate_ids": []}


def _graph_result(scenario: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    selected = selection["selected_candidate_ids"]
    if selection["action"] != "report" or not selected:
        return {"status": "rejected" if selection["action"] == "rejected" else "abstained", "root_node_ids": [], "affected_node_ids": [], "impact_paths": []}
    roots = resolve_candidate_roots(scenario["before_design"], scenario["after_design"], scenario["change_inventory"], selected)
    graph = build_version_graph(scenario["before_design"], scenario["after_design"])
    affected, paths = traverse_impact(graph, roots, scenario["engineering_change_request"]["max_depth"])
    return {"status": "completed", "root_node_ids": roots, "affected_node_ids": affected, "impact_paths": paths}


def _read_existing(path: Path, model: str, benchmark_sha256: str, allowed_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len({row.get("scenario_id") for row in rows}):
        raise ValueError(f"Duplicate scenario records in {path}")
    for row in rows:
        if row.get("pipeline_version") != PIPELINE_VERSION or row.get("model") != model or row.get("benchmark_sha256") != benchmark_sha256 or row.get("scenario_id") not in allowed_ids:
            raise ValueError(f"Existing record provenance mismatch in {path}")
    return {row["scenario_id"]: row for row in rows}


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_outputs(output: Path, aggregate: dict[str, Any], rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    metrics = aggregate["modes"][MODE]
    with (output / "evaluation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics)); writer.writeheader(); writer.writerow(metrics)
    with (output / "scenario_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_intent_qwen(
    model: str = "qwen2.5:7b", benchmark_path: str | Path = DEFAULT_BENCHMARK,
    output_dir: str | Path = DEFAULT_OUTPUT, catalog_path: str | Path | None = None,
    base_url: str = "http://localhost:11434", timeout: float = 900.0, seed: int = 11107,
    max_tokens: int = 500, profile: str = "smoke", client: OllamaClient | None = None,
) -> dict[str, Any]:
    scenarios, benchmark_manifest = load_frozen_intent_benchmark(benchmark_path, catalog_path)
    selected = _select_scenarios(scenarios, profile)
    output, record_path = Path(output_dir), Path(output_dir) / "intent_records.jsonl"
    runtime = client or OllamaClient(base_url, timeout)
    runtime.ensure_models([model])
    existing = _read_existing(record_path, model, benchmark_manifest["benchmark_sha256"], {item["scenario_id"] for item in selected})
    expected_calls = len(selected)
    completed_calls = sum(int(row["llm_call_count"]) for row in existing.values())
    print(f"[intent-qwen] profile={profile} scenarios={len(selected)} records={len(existing)}/{len(selected)} Qwen calls={completed_calls}/{expected_calls}; remaining={expected_calls - completed_calls}", flush=True)
    for scenario_index, scenario in enumerate(selected, start=1):
        if scenario["scenario_id"] in existing:
            continue
        print(f"[intent-qwen] scenario {scenario_index}/{len(selected)} {scenario['scenario_id']} - starting Qwen call {completed_calls + 1}/{expected_calls}; remaining={expected_calls - completed_calls}", flush=True)
        started, prompt = time.perf_counter(), build_intent_prompt(scenario)
        response = runtime.chat(model, SYSTEM_PROMPT, prompt, seed=seed, max_tokens=max_tokens, response_schema=RESPONSE_SCHEMA)
        selection, diagnostics = parse_intent_response(response.content, scenario)
        record = {
            "scenario_id": scenario["scenario_id"], "intent_case_type": scenario["intent_case_type"],
            "model": model, "mode": MODE, "pipeline_version": PIPELINE_VERSION,
            "benchmark_sha256": benchmark_manifest["benchmark_sha256"], "llm_call_count": 1,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "selection": selection, "graph_result": _graph_result(scenario, selection), "diagnostics": diagnostics,
            "raw_response": response.content, "ollama_metadata": response.metadata,
            "production_modification_performed": False, "input_designs_unchanged": True,
        }
        _append(record_path, record)
        existing[scenario["scenario_id"]] = record
        completed_calls += 1
        print(f"[intent-qwen] completed {len(existing)}/{len(selected)} action={selection['action']} candidates={len(selection['selected_candidate_ids'])} elapsed={time.perf_counter() - started:.1f}s; Qwen remaining={expected_calls - completed_calls}", flush=True)
    records = [existing[scenario["scenario_id"]] for scenario in selected]
    predictions = {row["scenario_id"]: row["selection"] for row in records}
    rejected = sum(row["selection"]["action"] == "rejected" for row in records)
    metrics, rows = evaluate_intent_predictions(selected, predictions, MODE, completed_calls, rejected)
    aggregate = {"evaluation_id": EVALUATION_ID, "pipeline_version": PIPELINE_VERSION, "model": model, "profile": profile, "scenario_count": len(selected), "modes": {MODE: metrics}}
    manifest = {
        "evaluation_id": EVALUATION_ID, "benchmark_id": benchmark_manifest["benchmark_id"],
        "benchmark_sha256": benchmark_manifest["benchmark_sha256"], "model": model, "mode": MODE,
        "profile": profile, "scenario_count": len(selected), "seed": seed, "temperature": 0, "max_tokens": max_tokens,
        "pipeline_version": PIPELINE_VERSION, "model_inputs": ["before_design", "after_design", "engineering_change_request", "change_inventory"],
        "intent_oracle_exposed_to_model": False, "impact_oracle_exposed_to_model": False,
        "resolved_candidate_root_ids_exposed_to_model": False, "oracle_usage": "offline_scoring_only",
        "model_role": "intent_conditioned_candidate_selection", "impact_result_source": "deterministic_graph_from_qwen_selected_candidates",
        "oracle_correction_performed": False, "metric_generation": "deterministic",
        "development_only": True, "heldout": False, "posthoc_tuning_allowed": True,
        "production_modifications_allowed": False, "narrative_output": False,
    }
    _write_outputs(output, aggregate, rows, manifest)
    return aggregate
