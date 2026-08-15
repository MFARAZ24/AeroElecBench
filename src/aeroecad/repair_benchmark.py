from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .generator import generate_benchmark

REPAIR_CASE_TYPES = ("clean", "automatic", "constrained", "ambiguous", "insufficient")
BENCHMARK_PROVENANCE = {
    "dataset_kind": "fictional_synthetic",
    "source_registry_id": "AEROELECBENCH-SOURCES-0.1",
    "representation_references": ["CPACS", "VEC-2.2.0", "WIREVIZ"],
    "open_design_references": ["PIXHAWK-STANDARDS", "ORESAT"],
    "rule_classification": "research_only",
    "direct_source_conversion": False,
    "certification_evidence": False,
}


def _component(design: dict[str, Any], component_id: str) -> dict[str, Any]:
    matches = [item for item in design["components"] if item["id"] == component_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one component named {component_id}; found {len(matches)}")
    return matches[0]


def _replace_pin(component: dict[str, Any], original_id: str, replacements: list[dict[str, Any]]) -> None:
    updated, replaced = [], False
    for pin in component["pins"]:
        if pin["id"] == original_id:
            updated.extend(copy.deepcopy(replacements))
            replaced = True
        else:
            updated.append(pin)
    if not replaced:
        raise ValueError(f"Pin {original_id} was not found on {component['id']}")
    component["pins"] = updated


def _operation(path: str, value: str) -> dict[str, str]:
    return {"op": "replace", "path": path, "value": value}


def _oracle(
    expected_action: str,
    expected_operation: dict[str, str] | None,
    expected_design: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "version": "0.2",
        "expected_action": expected_action,
        "expected_operation": copy.deepcopy(expected_operation),
        "expected_design": copy.deepcopy(expected_design),
    }


def _fault_truth(rule_id: str, path: str, wire_id: str) -> list[dict[str, str]]:
    return [{
        "rule_id": rule_id, "entity_path": path, "entity_id": wire_id,
        "defect": "invalid_pin_reference",
    }]


def _split(case_index: int) -> str:
    return "development" if case_index < 2 else "heldout"


def generate_repair_benchmark(seed: int = 4107, cases_per_type: int = 5) -> list[dict[str, Any]]:
    total = len(REPAIR_CASE_TYPES) * cases_per_type
    bases = generate_benchmark(seed=seed, cases_per_rule=0, clean_cases=total, mixed_cases=0)
    scenarios: list[dict[str, Any]] = []
    base_index = 0

    for case_type in REPAIR_CASE_TYPES:
        for case_index in range(cases_per_type):
            design = copy.deepcopy(bases[base_index]["design"])
            base_index += 1
            scenario_id = f"repair-{case_type}-{case_index:03d}"
            split = _split(case_index)

            if case_type == "clean":
                scenarios.append({
                    "scenario_id": scenario_id,
                    "category": "repair_benchmark",
                    "repair_case_type": case_type,
                    "split": split,
                    "review_query": "Perform a complete repair-oriented electrical-design review.",
                    "design": design,
                    "ground_truth": [],
                    "repair_oracle": _oracle("no_change", None, design),
                })
                continue

            wire_index = case_index % len(design["wires"])
            endpoint_name = "target" if case_index % 2 == 0 else "source"
            wire = design["wires"][wire_index]
            endpoint = wire[endpoint_name]
            path = f"wires[{wire_index}].{endpoint_name}.pin_id"
            original_pin_id = endpoint["pin_id"]
            target_component = _component(design, endpoint["component_id"])
            original_pin = next(pin for pin in target_component["pins"] if pin["id"] == original_pin_id)
            signal_class = original_pin["signal_class"]

            if case_type == "automatic":
                expected_design = copy.deepcopy(design)
                expected_value = original_pin_id
                expected_action = "automatic_repair"

            elif case_type == "constrained":
                primary_id, backup_id = f"{original_pin_id}_PRIMARY", f"{original_pin_id}_BACKUP"
                replacements = [
                    {"id": primary_id, "signal_class": signal_class, "interface_role": "primary"},
                    {"id": backup_id, "signal_class": signal_class, "interface_role": "backup"},
                ]
                _replace_pin(target_component, original_pin_id, replacements)
                endpoint["pin_id"] = primary_id
                for requirement in design["requirements"]:
                    if requirement["id"] in wire["requirement_ids"]:
                        requirement["text"] += " The connection shall use the primary interface on the affected component."
                expected_design = copy.deepcopy(design)
                expected_value = primary_id
                expected_action = "constrained_proposal"

            elif case_type == "ambiguous":
                option_a, option_b = f"{original_pin_id}_A", f"{original_pin_id}_B"
                replacements = [
                    {"id": option_a, "signal_class": signal_class, "interface_role": "equivalent"},
                    {"id": option_b, "signal_class": signal_class, "interface_role": "equivalent"},
                ]
                _replace_pin(target_component, original_pin_id, replacements)
                expected_design = None
                expected_value = None
                expected_action = "abstain"

            else:
                _replace_pin(target_component, original_pin_id, [])
                expected_design = None
                expected_value = None
                expected_action = "abstain"

            faulty_design = copy.deepcopy(design)
            faulty_design["wires"][wire_index][endpoint_name]["pin_id"] = "UNDECLARED_PIN"
            expected_operation = _operation(path, expected_value) if expected_value else None

            scenarios.append({
                "scenario_id": scenario_id,
                "category": "repair_benchmark",
                "repair_case_type": case_type,
                "split": split,
                "review_query": "Diagnose the invalid connector-pin reference and determine whether it can be safely repaired.",
                "design": faulty_design,
                "ground_truth": _fault_truth("AE-R003", path, wire["id"]),
                "repair_oracle": _oracle(expected_action, expected_operation, expected_design),
            })

    for scenario in scenarios:
        scenario["provenance"] = copy.deepcopy(BENCHMARK_PROVENANCE)
    return scenarios


def write_repair_benchmark(scenarios: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for scenario in scenarios:
            handle.write(json.dumps(scenario, sort_keys=True) + "\n")


def read_repair_benchmark(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
