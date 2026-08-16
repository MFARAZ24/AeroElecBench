from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from .impact_agent import IMPACT_RESPONSE_SCHEMA, parse_impact_content
from .impact_evaluation import BENCHMARK_ID, DEFAULT_BENCHMARK, evaluate_impact_records, load_frozen_impact_benchmark
from .impact_graph import analyze_change_impact, build_version_graph, compute_version_diff, resolve_change_roots, traverse_impact
from .ollama import OllamaClient, OllamaResponse

PIPELINE_VERSION = "0.9.1"
EVALUATION_ID = "AEROELECBENCH-IMPACT-ASSURANCE-V2-DEVELOPMENT-0.9.1"
DEFAULT_OUTPUT = Path("results/impact_v091_assurance_v2")
TOOLS = ("validate_change_evidence", "compute_version_diff", "build_version_graph", "resolve_change_roots", "traverse_dependencies")
DECISION_SYSTEM_PROMPT = """You are a bounded aerospace electrical change-impact tool controller. Select exactly one decision from permitted_decisions using the completed tool observations. A tool decision executes that tool; abstain and no_change are terminal. Never combine a terminal action with a tool and never predict affected nodes in a tool decision. Return one JSON object only."""
CANDIDATE_SYSTEM_PROMPT = """You are a bounded aerospace electrical change-impact reporting agent. Using only the completed tool observations, produce the candidate affected-node set and directed evidence edges. Include every node returned by dependency traversal and only graph edges needed to support the returned traversal paths. Return one JSON object only; do not invent identifiers or relations."""
ProgressCallback = Callable[[str, str, int, int, float], None]


def _public_change(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "change_id": request.get("change_id"), "change_type": request.get("change_type"),
        "operations": request.get("operations", []), "analysis_depth_limit": int(request.get("max_depth", 3)),
        "evidence_complete": bool(request.get("evidence_complete", True)),
        "missing_evidence": list(request.get("missing_evidence", [])),
    }


def _empty_report(status: str, reason: str = "", trace: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": status, "affected_node_ids": [], "impact_paths": [],
        "abstention_reason": reason if status == "abstained" else "",
        "rejection_reason": reason if status == "rejected" else "", "tool_trace": trace or [],
        "production_modification_performed": False, "input_designs_unchanged": True,
    }


def _diagnostics(parse_success: bool, error: str = "") -> dict[str, Any]:
    return {
        "parse_success": parse_success, "error": error, "invalid_node_count": 0,
        "duplicate_node_count": 0, "invalid_edge_count": 0, "valid_edge_count": 0, "raw_edge_count": 0,
    }


def _json_object(content: str) -> dict[str, Any] | None:
    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _decision_schema(permitted: list[str]) -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": permitted},
            "rationale": {"type": "string", "maxLength": 240},
        },
        "required": ["decision", "rationale"],
    }


def _parse_decision(content: str, permitted: list[str]) -> tuple[dict[str, str] | None, str]:
    payload = _json_object(content)
    if payload is None or set(payload) != {"decision", "rationale"}:
        return None, "invalid_decision_shape"
    if payload["decision"] not in permitted or not isinstance(payload["rationale"], str):
        return None, "invalid_decision_values"
    return payload, ""


def _permitted_decisions(executed: list[str], observations: dict[str, Any]) -> list[str]:
    if not executed:
        return ["validate_change_evidence"]
    latest = observations[executed[-1]]
    if executed[-1] == "validate_change_evidence":
        return ["compute_version_diff"] if latest["evidence_complete"] else ["abstain"]
    if executed[-1] == "compute_version_diff":
        return ["build_version_graph"] if latest["change_count"] else ["no_change"]
    if executed[-1] == "build_version_graph":
        return ["resolve_change_roots"] if not latest["missing_paths"] else ["abstain"]
    if executed[-1] == "resolve_change_roots":
        return ["traverse_dependencies"] if latest["root_ids"] and not latest["unavailable_root_ids"] else ["abstain"]
    return []


def _path_key(path: dict[str, Any]) -> tuple[Any, ...]:
    return path["root_id"], path["target_id"], tuple(path["node_ids"]), tuple(path["relations"])


def _action(report: dict[str, Any]) -> str:
    return {"completed": "report", "abstained": "abstain"}.get(report["status"], report["status"])


def validate_candidate(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    candidate_nodes, reference_nodes = set(candidate["affected_node_ids"]), set(reference["affected_node_ids"])
    candidate_paths = {_path_key(item) for item in candidate["impact_paths"]}
    reference_paths = {_path_key(item) for item in reference["impact_paths"]}
    action_match = _action(candidate) == _action(reference)
    exact = action_match and candidate_nodes == reference_nodes and candidate_paths == reference_paths
    return {
        "action_match": action_match, "exact": exact,
        "missing_node_ids": sorted(reference_nodes - candidate_nodes), "extra_node_ids": sorted(candidate_nodes - reference_nodes),
        "missing_path_count": len(reference_paths - candidate_paths), "extra_path_count": len(candidate_paths - reference_paths),
        "validation_action": "accepted" if exact else "corrected",
    }


def _tool_observation(name: str, scenario: dict[str, Any], observations: dict[str, Any]) -> dict[str, Any]:
    before, after, request = scenario["before_design"], scenario["after_design"], scenario["change_request"]
    if name == "validate_change_evidence":
        graph = build_version_graph(before, after)
        missing = sorted(set(graph["missing_paths"] + list(request.get("missing_evidence", []))))
        return {"evidence_complete": bool(request.get("evidence_complete", True) and not missing), "missing_paths": missing}
    if name == "compute_version_diff":
        changes = compute_version_diff(before, after)
        return {"change_count": len(changes), "changes": changes}
    if name == "build_version_graph":
        graph = build_version_graph(before, after)
        return {
            "node_count": len(graph["nodes"]), "edge_count": len(graph["edges"]), "missing_paths": graph["missing_paths"],
            "node_ids": sorted(graph["nodes"]), "edges": graph["edges"],
        }
    if name == "resolve_change_roots":
        roots = resolve_change_roots(before, after, list(request.get("operations", [])))
        known = set(observations["build_version_graph"]["node_ids"])
        return {"root_ids": roots, "unavailable_root_ids": sorted(set(roots) - known)}
    if name == "traverse_dependencies":
        graph = build_version_graph(before, after)
        roots = observations["resolve_change_roots"]["root_ids"]
        nodes, paths = traverse_impact(graph, roots, int(request.get("max_depth", 3)))
        return {"affected_node_ids": nodes, "impact_paths": paths}
    raise ValueError(f"Unknown assurance-v2 tool: {name}")


def expected_assurance_v2_calls(scenario: dict[str, Any]) -> int:
    evidence = _tool_observation("validate_change_evidence", scenario, {})
    if not evidence["evidence_complete"]:
        return 2
    if not compute_version_diff(scenario["before_design"], scenario["after_design"]):
        return 3
    graph = build_version_graph(scenario["before_design"], scenario["after_design"])
    if graph["missing_paths"]:
        return 4
    roots = resolve_change_roots(scenario["before_design"], scenario["after_design"], list(scenario["change_request"].get("operations", [])))
    if not roots or set(roots) - set(graph["nodes"]):
        return 5
    return 6


class AssuranceAgentV2:
    def __init__(self, client: OllamaClient):
        self.client = client

    def run(self, scenario: dict[str, Any], model: str, seed: int = 7107, max_tokens: int = 2500, progress: ProgressCallback | None = None) -> dict[str, Any]:
        observations: dict[str, Any] = {}
        executed: list[str] = []
        raw_responses: list[dict[str, Any]] = []
        prompt_sha256: list[str] = []
        call_budget, call_count, violations = expected_assurance_v2_calls(scenario), 0, 0

        def chat(label: str, system: str, prompt: str, response_schema: dict[str, Any], token_limit: int) -> OllamaResponse:
            nonlocal call_count
            call_count += 1
            if progress:
                progress("start", label, call_count, call_budget, 0.0)
            started = time.perf_counter()
            response = self.client.chat(model, system, prompt, seed=seed, max_tokens=token_limit, response_schema=response_schema)
            elapsed = time.perf_counter() - started
            if progress:
                progress("complete", label, call_count, call_budget, elapsed)
            prompt_sha256.append(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
            raw_responses.append({"stage": label, "content": response.content, "metadata": response.metadata})
            return response

        candidate, candidate_diagnostics = None, _diagnostics(True)
        while len(executed) < len(TOOLS):
            permitted = _permitted_decisions(executed, observations)
            prompt = json.dumps({
                "task": "Select exactly one permitted decision.", "change": _public_change(scenario["change_request"]),
                "tool_dependencies": {
                    "validate_change_evidence": None, "compute_version_diff": "validate_change_evidence",
                    "build_version_graph": "compute_version_diff", "resolve_change_roots": "build_version_graph",
                    "traverse_dependencies": "resolve_change_roots",
                },
                "executed_tools": executed, "latest_observation": observations[executed[-1]] if executed else None,
                "permitted_decisions": permitted,
                "policy": "Abstain for incomplete evidence or unavailable roots; return no_change only for an empty diff; otherwise continue in dependency order.",
            }, separators=(",", ":"), ensure_ascii=False)
            response = chat(f"decision_after_{executed[-1] if executed else 'start'}", DECISION_SYSTEM_PROMPT, prompt, _decision_schema(permitted), min(max_tokens, 220))
            decision, error = _parse_decision(response.content, permitted)
            if decision is None:
                violations += 1
                candidate, candidate_diagnostics = _empty_report("rejected", "The agent returned an invalid tool decision.", executed), _diagnostics(False, error)
                break
            expected_tool = TOOLS[len(executed)]
            selected = decision["decision"]
            if selected == "abstain":
                candidate = _empty_report("abstained", decision["rationale"], executed)
                break
            if selected == "no_change":
                candidate = _empty_report("no_change", trace=executed)
                break
            if selected != expected_tool:
                violations += 1
                candidate, candidate_diagnostics = _empty_report("rejected", f"Tool-order violation: expected {expected_tool}, received {selected}.", executed), _diagnostics(False, "tool_order_violation")
                break
            observations[expected_tool] = _tool_observation(expected_tool, scenario, observations)
            executed.append(expected_tool)
            if expected_tool == "traverse_dependencies":
                break

        if candidate is None and executed == list(TOOLS):
            prompt = json.dumps({
                "task": "Produce the candidate impact report from completed tool observations.",
                "change": _public_change(scenario["change_request"]), "tool_observations": observations,
                "node_id_formats": ["component:<id>", "pin:<component-id>:<pin-id>", "wire:<id>", "requirement:<id>", "verification:<id>"],
            }, separators=(",", ":"), ensure_ascii=False)
            response = chat("candidate_report", CANDIDATE_SYSTEM_PROMPT, prompt, IMPACT_RESPONSE_SCHEMA, max_tokens)
            candidate, candidate_diagnostics = parse_impact_content(response.content, scenario)
            candidate["tool_trace"] = executed.copy()

        reference = analyze_change_impact(scenario["before_design"], scenario["after_design"], scenario["change_request"])
        validation = validate_candidate(candidate, reference)
        final_report = reference
        final_report["tool_trace"] = [*executed, "validate_impact_report"]
        return {
            "scenario_id": scenario["scenario_id"], "mode": "assurance_agent_v2", "model": model,
            "model_role": "sequential_tool_control_and_candidate_synthesis",
            "candidate_result_source": "qwen_synthesis_from_observed_tool_outputs",
            "final_result_source": "mandatory_deterministic_validation_and_correction",
            "candidate_report": candidate, "final_report": final_report, "report": final_report,
            "candidate_diagnostics": candidate_diagnostics, "final_diagnostics": _diagnostics(True),
            "validation": validation, "tool_observations": observations, "tool_trace": executed,
            "analysis_tool_call_count": len(executed), "validator_call_count": 1,
            "tool_call_count": len(executed) + 1, "tool_order_violation_count": violations,
            "llm_call_count": call_count, "llm_call_budget": call_budget,
            "prompt_sha256": prompt_sha256, "raw_responses": raw_responses,
            "production_modification_performed": False,
        }


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


def _stage_records(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": row["scenario_id"], "mode": f"assurance_v2_{stage}", "model": row["model"],
            "report": row[f"{stage}_report"], "diagnostics": row[f"{stage}_diagnostics"],
            "llm_call_count": row["llm_call_count"],
        }
        for row in rows
    ]


def _write_outputs(output: Path, aggregate: dict[str, Any], manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    fields = [
        "mode", "scenario_count", "report_scenario_count", "llm_call_count", "rejected_output_count",
        "invalid_node_count", "invalid_edge_count", "invalid_edge_rate", "impact_set_precision", "impact_set_recall",
        "impact_set_f1", "impact_exact_scenario_accuracy", "path_precision", "path_recall", "path_f1",
        "multi_hop_recall", "abstention_recall", "clean_case_specificity", "oracle_action_accuracy",
        "production_modification_count", "input_immutability_rate",
    ]
    with (output / "evaluation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", *fields])
        writer.writeheader()
        for stage, metrics in aggregate["stages"].items():
            writer.writerow({"stage": stage, **{field: metrics[field] for field in fields}})
    with (output / "validation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario_id", "validation_action", "action_match", "missing_node_count", "extra_node_count", "missing_path_count", "extra_path_count", "llm_call_count", "analysis_tool_call_count", "validator_call_count", "tool_order_violation_count"])
        writer.writeheader()
        for row in rows:
            validation = row["validation"]
            writer.writerow({
                "scenario_id": row["scenario_id"], "validation_action": validation["validation_action"],
                "action_match": validation["action_match"], "missing_node_count": len(validation["missing_node_ids"]),
                "extra_node_count": len(validation["extra_node_ids"]), "missing_path_count": validation["missing_path_count"],
                "extra_path_count": validation["extra_path_count"], "llm_call_count": row["llm_call_count"],
                "analysis_tool_call_count": row["analysis_tool_call_count"], "validator_call_count": row["validator_call_count"],
                "tool_order_violation_count": row["tool_order_violation_count"],
            })
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_assurance_v2(
    model: str = "qwen2.5:7b", benchmark_path: str | Path = DEFAULT_BENCHMARK,
    output_dir: str | Path = DEFAULT_OUTPUT, catalog_path: str | Path | None = None,
    base_url: str = "http://localhost:11434", timeout: float = 900.0, seed: int = 9107,
    max_tokens: int = 2500, profile: str = "smoke", client: OllamaClient | None = None,
) -> dict[str, Any]:
    scenarios, benchmark_manifest, _ = load_frozen_impact_benchmark(benchmark_path, catalog_path, BENCHMARK_ID, "development")
    selected = _select_scenarios(scenarios, profile)
    output, record_path = Path(output_dir), Path(output_dir) / "assurance_v2_records.jsonl"
    runtime = client or OllamaClient(base_url, timeout)
    runtime.ensure_models([model])
    allowed_ids = {scenario["scenario_id"] for scenario in selected}
    existing = _read_existing(record_path, model, benchmark_manifest["benchmark_sha256"], allowed_ids)
    completed_calls = sum(int(row["llm_call_count"]) for row in existing.values())
    expected_calls = completed_calls + sum(expected_assurance_v2_calls(scenario) for scenario in selected if scenario["scenario_id"] not in existing)
    print(f"[assurance-v2] profile={profile} scenarios={len(selected)} records={len(existing)}/{len(selected)} Qwen calls={completed_calls}/{expected_calls}; remaining={expected_calls - completed_calls}", flush=True)
    agent = AssuranceAgentV2(runtime)
    for scenario_index, scenario in enumerate(selected, start=1):
        if scenario["scenario_id"] in existing:
            continue
        budget = expected_assurance_v2_calls(scenario)

        def progress(event: str, label: str, local_call: int, local_budget: int, elapsed: float) -> None:
            nonlocal completed_calls
            if event == "start":
                print(f"[assurance-v2] scenario {scenario_index}/{len(selected)} {scenario['scenario_id']} stage={label} - starting Qwen call {completed_calls + 1}/{expected_calls}; remaining={expected_calls - completed_calls}", flush=True)
            else:
                completed_calls += 1
                print(f"[assurance-v2] scenario {scenario_index}/{len(selected)} stage={label} completed local={local_call}/{local_budget} elapsed={elapsed:.1f}s; Qwen remaining={expected_calls - completed_calls}", flush=True)

        started = time.perf_counter()
        record = agent.run(scenario, model, seed, max_tokens, progress)
        unused_budget = budget - int(record["llm_call_count"])
        expected_calls -= max(0, unused_budget)
        record["benchmark_sha256"], record["pipeline_version"] = benchmark_manifest["benchmark_sha256"], PIPELINE_VERSION
        _append(record_path, record)
        existing[scenario["scenario_id"]] = record
        print(f"[assurance-v2] completed {len(existing)}/{len(selected)} candidate={record['candidate_report']['status']} validation={record['validation']['validation_action']} elapsed={time.perf_counter() - started:.1f}s; Qwen remaining={expected_calls - completed_calls}", flush=True)
    rows = [existing[scenario["scenario_id"]] for scenario in selected]
    candidate_metrics = evaluate_impact_records(selected, _stage_records(rows, "candidate"))
    final_metrics = evaluate_impact_records(selected, _stage_records(rows, "final"))
    accepted = sum(row["validation"]["validation_action"] == "accepted" for row in rows)
    aggregate = {
        "evaluation_id": EVALUATION_ID, "model": model, "profile": profile, "scenario_count": len(selected),
        "llm_call_count": sum(row["llm_call_count"] for row in rows),
        "stages": {"candidate": candidate_metrics, "final": final_metrics},
        "validation": {
            "accepted_count": accepted, "corrected_count": len(rows) - accepted,
            "acceptance_rate": accepted / len(rows), "intervention_rate": (len(rows) - accepted) / len(rows),
            "missing_node_count": sum(len(row["validation"]["missing_node_ids"]) for row in rows),
            "extra_node_count": sum(len(row["validation"]["extra_node_ids"]) for row in rows),
            "missing_path_count": sum(row["validation"]["missing_path_count"] for row in rows),
            "extra_path_count": sum(row["validation"]["extra_path_count"] for row in rows),
        },
        "tools": {
            "analysis_tool_execution_count": sum(row["analysis_tool_call_count"] for row in rows),
            "validator_execution_count": sum(row["validator_call_count"] for row in rows),
            "total_tool_execution_count": sum(row["tool_call_count"] for row in rows),
            "tool_order_violation_count": sum(row["tool_order_violation_count"] for row in rows),
        },
    }
    manifest = {
        "evaluation_id": EVALUATION_ID, "benchmark_id": benchmark_manifest["benchmark_id"],
        "benchmark_sha256": benchmark_manifest["benchmark_sha256"], "model": model, "profile": profile,
        "scenario_count": len(selected), "seed": seed, "temperature": 0, "max_tokens": max_tokens,
        "pipeline_version": PIPELINE_VERSION, "oracle_exposed_to_model": False,
        "oracle_root_node_ids_exposed_to_model": False, "resolved_tool_root_ids_exposed_to_model": True,
        "agent_model_role": "sequential_tool_control_and_candidate_synthesis",
        "candidate_result_source": "qwen_synthesis_from_observed_deterministic_tool_outputs",
        "final_result_source": "mandatory_independent_deterministic_validation_and_correction",
        "graph_validation_policy": "mandatory_for_every_positive_impact_report",
        "candidate_and_final_metrics_reported_separately": True, "development_only": True,
        "heldout": False, "posthoc_tuning_allowed": True, "production_modifications_allowed": False,
        "narrative_output": False,
    }
    _write_outputs(output, aggregate, manifest, rows)
    return aggregate
