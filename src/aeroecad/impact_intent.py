from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .catalog import load_catalog
from .generator import generate_benchmark
from .impact_benchmark import (
    IMPACT_PROVENANCE,
    _oracle_impact,
    _with_verification_activities,
    read_impact_benchmark,
    write_impact_benchmark,
)
from .impact_graph import (
    build_version_graph,
    compute_version_diff,
    resolve_change_roots,
    traverse_impact,
)
from .validator import validate_design

INTENT_CASE_TYPES = ("single_explicit", "single_paraphrased", "multi_intent", "same_entity_distractor", "ambiguous_request", "unmatched_request")
DEFAULT_BENCHMARK = Path("benchmark/v10/impact_intent_development_24.jsonl")
DEFAULT_OUTPUT = Path("results/impact_v10_intent_baselines")
DEFAULT_SEED = 10107
BENCHMARK_ID = "AEROELECBENCH-IMPACT-INTENT-DEVELOPMENT-1.0"
EVALUATION_ID = "AEROELECBENCH-IMPACT-INTENT-BASELINES-1.0"
PIPELINE_VERSION = "1.0.0"
BASELINE_MODES = ("all_diff_graph", "lexical_intent_graph", "oracle_root_graph")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "change", "changes", "design", "for", "from", "in", "is", "it", "of", "on", "only", "or", "the", "this", "to", "with"}


def _operation(op: str, path: str, before: Any, after: Any) -> dict[str, Any]:
    return {"op": op, "path": path, "before": copy.deepcopy(before), "after": copy.deepcopy(after)}


def _candidate(candidate_id: str, change_type: str, entity_ids: list[str], operations: list[dict[str, Any]]) -> dict[str, Any]:
    return {"candidate_id": candidate_id, "change_type": change_type, "entity_ids": sorted(entity_ids), "operations": copy.deepcopy(operations)}


def _component_index(design: dict[str, Any], component_id: str) -> int:
    return next(index for index, component in enumerate(design["components"]) if component["id"] == component_id)


def _apply_candidate_changes(before: dict[str, Any], case_index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    after = copy.deepcopy(before)
    pin_wire_index = case_index % len(after["wires"])
    pin_wire = after["wires"][pin_wire_index]
    target_component_id, old_pin_id = pin_wire["target"]["component_id"], pin_wire["target"]["pin_id"]
    target_component_index = _component_index(after, target_component_id)
    target_component = after["components"][target_component_index]

    old_part = target_component["part_number"]
    new_part = f"SYN-INTENT-PN-{case_index:03d}"
    target_component["part_number"] = new_part
    component_candidate = _candidate("CAND-01", "component_replacement", [target_component_id], [
        _operation("replace", f"components[{target_component_index}].part_number", old_part, new_part),
    ])

    gauge_wire_index = (case_index + 2) % len(after["wires"])
    gauge_wire = after["wires"][gauge_wire_index]
    old_gauge = gauge_wire["gauge_awg"]
    new_gauge = next(value for value in (20, 22, 24) if value != old_gauge)
    gauge_wire["gauge_awg"] = new_gauge
    gauge_candidate = _candidate("CAND-02", "wire_gauge_revision", [gauge_wire["id"]], [
        _operation("replace", f"wires[{gauge_wire_index}].gauge_awg", old_gauge, new_gauge),
    ])

    requirement_index = case_index % len(after["requirements"])
    requirement = after["requirements"][requirement_index]
    old_text = requirement["text"]
    new_text = f"{old_text} Authorized intent-conditioned verification update {case_index}."
    requirement["text"] = new_text
    requirement_candidate = _candidate("CAND-03", "requirement_revision", [requirement["id"]], [
        _operation("replace", f"requirements[{requirement_index}].text", old_text, new_text),
    ])

    old_pin = next(pin for pin in target_component["pins"] if pin["id"] == old_pin_id)
    new_pin = {**copy.deepcopy(old_pin), "id": f"{old_pin_id}_ALT_{case_index:03d}"}
    target_component["pins"].append(new_pin)
    pin_wire["target"]["pin_id"] = new_pin["id"]
    pin_candidate = _candidate("CAND-04", "pin_reassignment", [pin_wire["id"], target_component_id, old_pin_id, new_pin["id"]], [
        _operation("add", f"components[{target_component_index}].pins[{len(target_component['pins']) - 1}]", None, new_pin),
        _operation("replace", f"wires[{pin_wire_index}].target.pin_id", old_pin_id, new_pin["id"]),
    ])
    after["revision"] = "B"
    return after, [component_candidate, gauge_candidate, requirement_candidate, pin_candidate]


def _request(case_type: str, scenario_index: int, inventory: list[dict[str, Any]]) -> tuple[str, list[str], str]:
    by_id = {item["candidate_id"]: item for item in inventory}
    component_id = by_id["CAND-01"]["entity_ids"][0]
    component_part = by_id["CAND-01"]["operations"][0]["after"]
    requirement_id = by_id["CAND-03"]["entity_ids"][0]
    pin_entities = by_id["CAND-04"]["entity_ids"]
    pin_wire_id = next(value for value in pin_entities if value.startswith("W-"))
    old_pin, new_pin = by_id["CAND-04"]["operations"][1]["before"], by_id["CAND-04"]["operations"][1]["after"]
    if case_type == "single_explicit":
        return f"Assess the authorized replacement of installed unit {component_id} with part {component_part}. Ignore every other revision edit.", ["CAND-01"], "report"
    if case_type == "single_paraphrased":
        return "Evaluate only the conductor resized to improve electrical margin; hardware, connector, and requirement-document edits are outside this request.", ["CAND-02"], "report"
    if case_type == "multi_intent":
        return f"Assess the coordinated hardware update at {component_id} and the revised {requirement_id}; exclude wiring and connector housekeeping.", ["CAND-01", "CAND-03"], "report"
    if case_type == "same_entity_distractor":
        return f"For {pin_wire_id}, evaluate only the target-pin reassignment from {old_pin} to {new_pin}. Do not treat the separate part-number edit on the same installed unit as authorized scope.", ["CAND-04"], "report"
    if case_type == "ambiguous_request":
        return "Assess the approved electrical update and its downstream consequences.", [], "abstain"
    return f"Assess only the requested replacement of LRU-NOT-PRESENT-{scenario_index:03d} with SYN-UNOBSERVED-{scenario_index:03d}; no other edit is authorized.", [], "abstain"


def _inventory_operations(inventory: list[dict[str, Any]], selected_ids: list[str] | None = None) -> list[dict[str, Any]]:
    selected = set(selected_ids) if selected_ids is not None else {item["candidate_id"] for item in inventory}
    return [copy.deepcopy(operation) for item in inventory if item["candidate_id"] in selected for operation in item["operations"]]


def resolve_candidate_roots(before: dict[str, Any], after: dict[str, Any], inventory: list[dict[str, Any]], selected_ids: list[str]) -> list[str]:
    return resolve_change_roots(before, after, _inventory_operations(inventory, selected_ids))


def generate_intent_benchmark(seed: int = DEFAULT_SEED, cases_per_type: int = 4, split: str = "development") -> list[dict[str, Any]]:
    total = len(INTENT_CASE_TYPES) * cases_per_type
    bases = generate_benchmark(seed=seed, cases_per_rule=0, clean_cases=total, mixed_cases=0)
    scenarios, scenario_index = [], 0
    for case_type in INTENT_CASE_TYPES:
        for case_index in range(cases_per_type):
            before = _with_verification_activities(bases[scenario_index]["design"])
            after, inventory = _apply_candidate_changes(before, scenario_index)
            request_text, intended_ids, expected_action = _request(case_type, scenario_index, inventory)
            root_ids = resolve_candidate_roots(before, after, inventory, intended_ids)
            max_depth = 3
            affected, paths = _oracle_impact(before, after, root_ids, max_depth) if expected_action == "report" else ([], [])
            scenario_id = f"v10-intent-{case_type}-{case_index:03d}"
            scenarios.append({
                "scenario_id": scenario_id, "category": "intent_conditioned_change_impact", "intent_case_type": case_type,
                "split": split, "before_design": before, "after_design": after, "change_inventory": inventory,
                "engineering_change_request": {"request_id": f"ECR-{scenario_index + 1:04d}", "text": request_text, "max_depth": max_depth},
                "intent_oracle": {"version": "1.0", "expected_action": expected_action, "intended_candidate_ids": intended_ids, "root_node_ids": root_ids},
                "impact_oracle": {"version": "1.0", "expected_action": expected_action, "affected_node_ids": affected, "impact_paths": paths},
                "provenance": copy.deepcopy(IMPACT_PROVENANCE),
            })
            scenario_index += 1
    return scenarios


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _operation_key(operation: dict[str, Any]) -> tuple[str, str, str, str]:
    return operation["op"], operation["path"], json.dumps(operation.get("before"), sort_keys=True), json.dumps(operation.get("after"), sort_keys=True)


def _path_key(path: dict[str, Any]) -> tuple[Any, ...]:
    return path["root_id"], path["target_id"], tuple(path["node_ids"]), tuple(path["relations"])


def validate_intent_benchmark(scenarios: list[dict[str, Any]], rules: list[dict[str, Any]], expected_split: str = "development") -> None:
    if len({item["scenario_id"] for item in scenarios}) != len(scenarios):
        raise ValueError("Intent scenario identifiers must be unique")
    for scenario in scenarios:
        scenario_id, inventory = scenario["scenario_id"], scenario["change_inventory"]
        if scenario.get("intent_case_type") not in INTENT_CASE_TYPES or scenario.get("split") != expected_split:
            raise ValueError(f"Invalid {expected_split} intent metadata in {scenario_id}")
        if validate_design(scenario["before_design"], rules) or validate_design(scenario["after_design"], rules):
            raise ValueError(f"Invalid design version in {scenario_id}")
        if len(inventory) < 3 or len({item["candidate_id"] for item in inventory}) != len(inventory):
            raise ValueError(f"Invalid change inventory in {scenario_id}")
        if any("root_node_ids" in item for item in inventory):
            raise ValueError(f"Resolved graph roots leaked into the observable inventory in {scenario_id}")
        observed = {_operation_key(item) for item in compute_version_diff(scenario["before_design"], scenario["after_design"]) if item["path"] != "revision"}
        inventoried = {_operation_key(item) for item in _inventory_operations(inventory)}
        if observed != inventoried:
            raise ValueError(f"Change inventory does not cover the semantic version diff in {scenario_id}")
        candidate_ids = {item["candidate_id"] for item in inventory}
        intended_ids = scenario["intent_oracle"]["intended_candidate_ids"]
        if not set(intended_ids) <= candidate_ids:
            raise ValueError(f"Intent oracle references an unknown candidate in {scenario_id}")
        expected_action = scenario["intent_oracle"]["expected_action"]
        if (expected_action == "report") != bool(intended_ids):
            raise ValueError(f"Intent action and selected candidates disagree in {scenario_id}")
        roots = resolve_candidate_roots(scenario["before_design"], scenario["after_design"], inventory, intended_ids)
        if roots != scenario["intent_oracle"]["root_node_ids"]:
            raise ValueError(f"Intent-root oracle mismatch in {scenario_id}")
        if expected_action == "report":
            graph = build_version_graph(scenario["before_design"], scenario["after_design"])
            affected, paths = traverse_impact(graph, roots, scenario["engineering_change_request"]["max_depth"])
            if affected != scenario["impact_oracle"]["affected_node_ids"] or {_path_key(item) for item in paths} != {_path_key(item) for item in scenario["impact_oracle"]["impact_paths"]}:
                raise ValueError(f"Independent impact oracle mismatch in {scenario_id}")


def prepare_intent_development(benchmark_path: str | Path = DEFAULT_BENCHMARK, catalog_path: str | Path | None = None, source_registry_path: str | Path = "data/source_registry.json", seed: int = DEFAULT_SEED, cases_per_type: int = 4) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark, registry_path = Path(benchmark_path), Path(source_registry_path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Source registry not found: {registry_path}")
    catalog = load_catalog(catalog_path)
    scenarios = generate_intent_benchmark(seed, cases_per_type)
    validate_intent_benchmark(scenarios, catalog["rules"])
    write_impact_benchmark(scenarios, benchmark)
    manifest = {
        "benchmark_id": BENCHMARK_ID, "benchmark": benchmark.name, "benchmark_sha256": _sha256(benchmark),
        "catalog_id": catalog["catalog_id"], "catalog_version": catalog["version"], "catalog_sha256": _sha256(Path(catalog_path or "data/rules.json")),
        "source_registry_id": json.loads(registry_path.read_text(encoding="utf-8"))["registry_id"], "source_registry_sha256": _sha256(registry_path),
        "seed": seed, "scenario_count": len(scenarios), "case_type_counts": dict(sorted(Counter(item["intent_case_type"] for item in scenarios).items())),
        "split_counts": {"development": len(scenarios)}, "minimum_concurrent_semantic_changes": 4, "oracle_validation_rate": 1.0,
        "before_after_exposed_to_model": True, "natural_language_request_exposed_to_model": True, "change_inventory_exposed_to_model": True,
        "intent_oracle_exposed_to_model": False, "impact_oracle_exposed_to_model": False, "resolved_candidate_root_ids_exposed_to_model": False,
        "semantic_diff_excludes_paths": ["revision"], "development_only": True, "heldout": False, "posthoc_tuning_allowed": True,
        "dataset_kind": "fictional_synthetic", "production_modifications_allowed": False, "narrative_output": False,
    }
    benchmark.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return scenarios, manifest


def load_frozen_intent_benchmark(benchmark_path: str | Path = DEFAULT_BENCHMARK, catalog_path: str | Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark, manifest_path = Path(benchmark_path), Path(benchmark_path).with_name("manifest.json")
    if not benchmark.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Intent benchmark not found at {benchmark}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("benchmark_id") != BENCHMARK_ID or manifest.get("benchmark_sha256") != _sha256(benchmark):
        raise ValueError("Intent benchmark identity or hash mismatch")
    scenarios, catalog = read_impact_benchmark(benchmark), load_catalog(catalog_path)
    validate_intent_benchmark(scenarios, catalog["rules"])
    return scenarios, manifest


def _tokens(value: Any) -> set[str]:
    return {token for token in _TOKEN.findall(str(value).lower()) if token not in _STOPWORDS and len(token) > 1}


def _lexical_selection(scenario: dict[str, Any]) -> list[str]:
    request_tokens = _tokens(scenario["engineering_change_request"]["text"])
    scores = []
    for candidate in scenario["change_inventory"]:
        searchable = [candidate["change_type"], *candidate["entity_ids"]]
        for operation in candidate["operations"]:
            searchable.extend((operation["path"], operation.get("before"), operation.get("after")))
        scores.append((len(request_tokens & _tokens(" ".join(map(str, searchable)))), candidate["candidate_id"]))
    return sorted(candidate_id for score, candidate_id in scores if score >= 2)


def _selection_for(mode: str, scenario: dict[str, Any]) -> list[str]:
    if mode == "all_diff_graph":
        return [item["candidate_id"] for item in scenario["change_inventory"]]
    if mode == "lexical_intent_graph":
        return _lexical_selection(scenario)
    return list(scenario["intent_oracle"]["intended_candidate_ids"])


def _set_counts(expected: set[Any], predicted: set[Any]) -> tuple[int, int, int]:
    return len(expected & predicted), len(predicted - expected), len(expected - predicted)


def _ratio(numerator: int, denominator: int, empty: float = 0.0) -> float:
    return numerator / denominator if denominator else empty


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return precision, recall, _ratio(2 * precision * recall, precision + recall)


def evaluate_intent_predictions(
    scenarios: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], mode: str,
    llm_call_count: int = 0, rejected_output_count: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    missing = [scenario["scenario_id"] for scenario in scenarios if scenario["scenario_id"] not in predictions]
    if missing:
        raise ValueError(f"Missing intent predictions for: {', '.join(missing)}")
    candidate_tp = candidate_fp = candidate_fn = rejected_distractors = total_distractors = 0
    root_tp = root_fp = root_fn = impact_tp = impact_fp = impact_fn = path_tp = path_fp = path_fn = 0
    candidate_exact = root_exact = impact_exact = action_correct = 0
    rows = []
    for scenario in scenarios:
        prediction = predictions[scenario["scenario_id"]]
        predicted_ids = list(prediction["selected_candidate_ids"])
        expected_ids = scenario["intent_oracle"]["intended_candidate_ids"]
        predicted_roots = resolve_candidate_roots(scenario["before_design"], scenario["after_design"], scenario["change_inventory"], predicted_ids)
        expected_roots = scenario["intent_oracle"]["root_node_ids"]
        predicted_action = prediction["action"]
        if predicted_ids:
            graph = build_version_graph(scenario["before_design"], scenario["after_design"])
            predicted_impact, predicted_paths = traverse_impact(graph, predicted_roots, scenario["engineering_change_request"]["max_depth"])
        else:
            predicted_impact, predicted_paths = [], []
        expected_impact, expected_paths = scenario["impact_oracle"]["affected_node_ids"], scenario["impact_oracle"]["impact_paths"]
        ctp, cfp, cfn = _set_counts(set(expected_ids), set(predicted_ids))
        rtp, rfp, rfn = _set_counts(set(expected_roots), set(predicted_roots))
        itp, ifp, ifn = _set_counts(set(expected_impact), set(predicted_impact))
        ptp, pfp, pfn = _set_counts({_path_key(item) for item in expected_paths}, {_path_key(item) for item in predicted_paths})
        candidate_tp += ctp; candidate_fp += cfp; candidate_fn += cfn
        root_tp += rtp; root_fp += rfp; root_fn += rfn
        impact_tp += itp; impact_fp += ifp; impact_fn += ifn
        path_tp += ptp; path_fp += pfp; path_fn += pfn
        distractors = {item["candidate_id"] for item in scenario["change_inventory"]} - set(expected_ids)
        rejected_distractors += len(distractors - set(predicted_ids)); total_distractors += len(distractors)
        candidate_exact += set(expected_ids) == set(predicted_ids)
        root_exact += set(expected_roots) == set(predicted_roots)
        impact_exact += set(expected_impact) == set(predicted_impact)
        action_correct += predicted_action == scenario["intent_oracle"]["expected_action"]
        rows.append({
            "mode": mode, "scenario_id": scenario["scenario_id"], "intent_case_type": scenario["intent_case_type"],
            "expected_action": scenario["intent_oracle"]["expected_action"], "predicted_action": predicted_action,
            "expected_candidate_ids": "|".join(expected_ids), "predicted_candidate_ids": "|".join(predicted_ids),
            "candidate_exact": int(set(expected_ids) == set(predicted_ids)), "root_exact": int(set(expected_roots) == set(predicted_roots)),
            "impact_exact": int(set(expected_impact) == set(predicted_impact)),
        })
    candidate_precision, candidate_recall, candidate_f1 = _prf(candidate_tp, candidate_fp, candidate_fn)
    root_precision, root_recall, root_f1 = _prf(root_tp, root_fp, root_fn)
    impact_precision, impact_recall, impact_f1 = _prf(impact_tp, impact_fp, impact_fn)
    path_precision, path_recall, path_f1 = _prf(path_tp, path_fp, path_fn)
    count = len(scenarios)
    return {
        "mode": mode, "scenario_count": count, "llm_call_count": llm_call_count, "rejected_output_count": rejected_output_count,
        "candidate_precision": candidate_precision, "candidate_recall": candidate_recall, "candidate_f1": candidate_f1,
        "candidate_exact_scenario_accuracy": candidate_exact / count, "distractor_rejection_rate": _ratio(rejected_distractors, total_distractors, 1.0),
        "root_precision": root_precision, "root_recall": root_recall, "root_f1": root_f1, "root_exact_scenario_accuracy": root_exact / count,
        "impact_set_precision": impact_precision, "impact_set_recall": impact_recall, "impact_set_f1": impact_f1, "impact_exact_scenario_accuracy": impact_exact / count,
        "path_precision": path_precision, "path_recall": path_recall, "path_f1": path_f1,
        "oracle_action_accuracy": action_correct / count, "production_modification_count": 0,
    }, rows


def _evaluate_mode(mode: str, scenarios: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = {
        scenario["scenario_id"]: {
            "action": "report" if (selected := _selection_for(mode, scenario)) else "abstain",
            "selected_candidate_ids": selected,
        }
        for scenario in scenarios
    }
    return evaluate_intent_predictions(scenarios, predictions, mode)


def run_intent_baselines(benchmark_path: str | Path = DEFAULT_BENCHMARK, output_dir: str | Path = DEFAULT_OUTPUT, catalog_path: str | Path | None = None) -> dict[str, Any]:
    scenarios, benchmark_manifest = load_frozen_intent_benchmark(benchmark_path, catalog_path)
    metrics_by_mode, rows = {}, []
    for mode in BASELINE_MODES:
        metrics_by_mode[mode], mode_rows = _evaluate_mode(mode, scenarios)
        rows.extend(mode_rows)
    aggregate = {"evaluation_id": EVALUATION_ID, "pipeline_version": PIPELINE_VERSION, "scenario_count": len(scenarios), "modes": metrics_by_mode}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    fields = list(next(iter(metrics_by_mode.values())).keys())
    with (output / "evaluation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(metrics_by_mode.values())
    with (output / "scenario_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    evaluation_manifest = {
        "evaluation_id": EVALUATION_ID, "benchmark_id": benchmark_manifest["benchmark_id"], "benchmark_sha256": benchmark_manifest["benchmark_sha256"],
        "pipeline_version": PIPELINE_VERSION, "modes": list(BASELINE_MODES), "scenario_count": len(scenarios), "llm_calls_performed": 0,
        "intent_oracle_exposed_to_non_oracle_baselines": False, "oracle_root_graph_role": "upper_bound",
        "all_diff_graph_role": "intent_agnostic_control", "lexical_intent_graph_role": "non_llm_intent_baseline",
        "metric_generation": "deterministic", "production_modifications_allowed": False, "narrative_output": False,
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(evaluation_manifest, indent=2) + "\n", encoding="utf-8")
    return aggregate
