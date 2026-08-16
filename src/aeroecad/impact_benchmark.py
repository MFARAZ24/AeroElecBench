from __future__ import annotations

import copy
import json
from collections import deque
from pathlib import Path
from typing import Any

from .generator import generate_benchmark
from .impact_graph import component_node, pin_node, requirement_node, verification_node, wire_node

IMPACT_CASE_TYPES = ("clean", "pin_reassignment", "component_replacement", "requirement_revision", "interacting_change", "missing_information")
IMPACT_PROVENANCE = {
    "dataset_kind": "fictional_synthetic", "source_registry_id": "AEROELECBENCH-SOURCES-0.1",
    "representation_references": ["CPACS", "VEC-2.2.0", "WIREVIZ"],
    "open_design_references": ["PIXHAWK-STANDARDS", "ORESAT"],
    "direct_source_conversion": False, "rule_classification": "research_only", "certification_evidence": False,
}


def _with_verification_activities(design: dict[str, Any]) -> dict[str, Any]:
    enriched = copy.deepcopy(design)
    activities = []
    for requirement in enriched["requirements"]:
        requirement_id = requirement["id"]
        component_ids = sorted({
            endpoint["component_id"] for wire in enriched["wires"] if requirement_id in wire["requirement_ids"]
            for endpoint in (wire["source"], wire["target"])
        })
        activities.append({
            "id": f"TEST-{requirement_id.removeprefix('REQ-')}",
            "title": f"Synthetic verification activity for {requirement_id}",
            "requirement_ids": [requirement_id], "component_ids": component_ids,
        })
    enriched["verification_activities"] = activities
    return enriched


def _component(design: dict[str, Any], component_id: str) -> tuple[int, dict[str, Any]]:
    matches = [(index, item) for index, item in enumerate(design["components"]) if item["id"] == component_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one component named {component_id}; found {len(matches)}")
    return matches[0]


def _oracle_edges(design: dict[str, Any]) -> set[tuple[str, str, str]]:
    edges: set[tuple[str, str, str]] = set()
    components = {item["id"]: item for item in design.get("components", [])}
    requirements = {item["id"] for item in design.get("requirements", [])}
    for component_id, component in components.items():
        for pin in component.get("pins", []):
            edges.add((component_node(component_id), pin_node(component_id, pin["id"]), "contains_pin"))
    for wire in design.get("wires", []):
        source = wire_node(wire["id"])
        for endpoint_name in ("source", "target"):
            endpoint = wire[endpoint_name]
            if endpoint["component_id"] in components:
                edges.add((source, component_node(endpoint["component_id"]), f"connected_{endpoint_name}_component"))
                if any(pin["id"] == endpoint["pin_id"] for pin in components[endpoint["component_id"]].get("pins", [])):
                    edges.add((pin_node(endpoint["component_id"], endpoint["pin_id"]), source, f"{endpoint_name}_endpoint_of"))
        for requirement_id in wire.get("requirement_ids", []):
            if requirement_id in requirements:
                edges.add((source, requirement_node(requirement_id), "traced_to_requirement"))
                edges.add((requirement_node(requirement_id), source, "allocated_wire"))
    for activity in design.get("verification_activities", []):
        target = verification_node(activity["id"])
        for requirement_id in activity.get("requirement_ids", []):
            if requirement_id in requirements:
                edges.add((requirement_node(requirement_id), target, "verified_by"))
        for component_id in activity.get("component_ids", []):
            if component_id in components:
                edges.add((component_node(component_id), target, "covered_by_verification"))
    return edges


def _oracle_nodes(design: dict[str, Any]) -> set[str]:
    nodes = {component_node(component["id"]) for component in design.get("components", [])}
    nodes |= {pin_node(component["id"], pin["id"]) for component in design.get("components", []) for pin in component.get("pins", [])}
    nodes |= {wire_node(wire["id"]) for wire in design.get("wires", [])}
    nodes |= {requirement_node(requirement["id"]) for requirement in design.get("requirements", [])}
    nodes |= {verification_node(activity["id"]) for activity in design.get("verification_activities", [])}
    return nodes


def _oracle_impact(before: dict[str, Any], after: dict[str, Any], roots: list[str], max_depth: int) -> tuple[list[str], list[dict[str, Any]]]:
    nodes = _oracle_nodes(before) | _oracle_nodes(after)
    adjacency: dict[str, list[tuple[str, str]]] = {node_id: [] for node_id in nodes}
    for source, target, relation in _oracle_edges(before) | _oracle_edges(after):
        adjacency.setdefault(source, []).append((target, relation))
    for values in adjacency.values():
        values.sort()
    discovered: dict[str, tuple[str, list[str], list[str]]] = {}
    queue: deque[tuple[str, str, list[str], list[str]]] = deque()
    for root in sorted(set(roots)):
        if root not in nodes:
            raise ValueError(f"Oracle root does not exist: {root}")
        discovered[root] = (root, [root], [])
        queue.append((root, root, [root], []))
    while queue:
        root, current, node_path, relations = queue.popleft()
        if len(relations) >= max_depth:
            continue
        for target, relation in adjacency.get(current, []):
            if target in discovered:
                continue
            next_nodes, next_relations = [*node_path, target], [*relations, relation]
            discovered[target] = (root, next_nodes, next_relations)
            queue.append((root, target, next_nodes, next_relations))
    paths = [
        {"root_id": root, "target_id": target, "node_ids": node_path, "relations": relations}
        for target, (root, node_path, relations) in sorted(discovered.items())
    ]
    return sorted(discovered), paths


def _change(change_id: str, change_type: str, roots: list[str], operations: list[dict[str, Any]], max_depth: int = 3, evidence_complete: bool = True, missing: list[str] | None = None) -> dict[str, Any]:
    return {
        "change_id": change_id, "change_type": change_type, "root_node_ids": sorted(roots),
        "operations": copy.deepcopy(operations), "max_depth": max_depth,
        "evidence_complete": evidence_complete, "missing_evidence": missing or [],
    }


def _operation(op: str, path: str, before: Any, after: Any) -> dict[str, Any]:
    return {"op": op, "path": path, "before": copy.deepcopy(before), "after": copy.deepcopy(after)}


def generate_impact_benchmark(seed: int = 7107, cases_per_type: int = 4, version: str = "v07", split: str = "development") -> list[dict[str, Any]]:
    total = len(IMPACT_CASE_TYPES) * cases_per_type
    bases = generate_benchmark(seed=seed, cases_per_rule=0, clean_cases=total, mixed_cases=0)
    scenarios: list[dict[str, Any]] = []
    base_index = 0
    for case_type in IMPACT_CASE_TYPES:
        for case_index in range(cases_per_type):
            before = _with_verification_activities(bases[base_index]["design"])
            after = copy.deepcopy(before)
            base_index += 1
            scenario_id, change_id = f"{version}-impact-{case_type}-{case_index:03d}", f"CHG-{base_index:04d}"

            if case_type == "clean":
                request = _change(change_id, "no_change", [], [], 0)
                expected_action = "no_change"
            elif case_type == "pin_reassignment":
                wire_index = (case_index + 2) % len(after["wires"])
                wire = after["wires"][wire_index]
                component_id, original_pin = wire["target"]["component_id"], wire["target"]["pin_id"]
                component_index, component = _component(after, component_id)
                original = next(pin for pin in component["pins"] if pin["id"] == original_pin)
                alternate_pin = {**copy.deepcopy(original), "id": f"{original_pin}_ALT_{case_index}"}
                component["pins"].append(alternate_pin)
                wire["target"]["pin_id"] = alternate_pin["id"]
                after["revision"] = "B"
                operations = [
                    _operation("add", f"components[{component_index}].pins[{len(component['pins']) - 1}]", None, alternate_pin),
                    _operation("replace", f"wires[{wire_index}].target.pin_id", original_pin, alternate_pin["id"]),
                    _operation("replace", "revision", "A", "B"),
                ]
                roots = [pin_node(component_id, original_pin), pin_node(component_id, alternate_pin["id"]), wire_node(wire["id"])]
                request, expected_action = _change(change_id, case_type, roots, operations, 3), "report"
            elif case_type == "component_replacement":
                component_index = (case_index + 1) % len(after["components"])
                component = after["components"][component_index]
                old_part, new_part = component["part_number"], f"SYN-REPLACEMENT-{case_index:03d}"
                component["part_number"], after["revision"] = new_part, "B"
                operations = [_operation("replace", f"components[{component_index}].part_number", old_part, new_part), _operation("replace", "revision", "A", "B")]
                request, expected_action = _change(change_id, case_type, [component_node(component["id"])], operations, 3), "report"
            elif case_type == "requirement_revision":
                requirement_index = case_index % len(after["requirements"])
                requirement = after["requirements"][requirement_index]
                old_text, new_text = requirement["text"], f"{requirement['text']} Revision B impact review required."
                requirement["text"], after["revision"] = new_text, "B"
                operations = [_operation("replace", f"requirements[{requirement_index}].text", old_text, new_text), _operation("replace", "revision", "A", "B")]
                request, expected_action = _change(change_id, case_type, [requirement_node(requirement["id"])], operations, 2), "report"
            elif case_type == "interacting_change":
                component_index, requirement_index = (case_index + 2) % len(after["components"]), case_index % len(after["requirements"])
                component, requirement = after["components"][component_index], after["requirements"][requirement_index]
                old_part, new_part = component["part_number"], f"SYN-INTERACT-{case_index:03d}"
                old_text, new_text = requirement["text"], f"{requirement['text']} Coordinated replacement verification required."
                component["part_number"], requirement["text"], after["revision"] = new_part, new_text, "B"
                operations = [
                    _operation("replace", f"components[{component_index}].part_number", old_part, new_part),
                    _operation("replace", f"requirements[{requirement_index}].text", old_text, new_text),
                    _operation("replace", "revision", "A", "B"),
                ]
                roots = [component_node(component["id"]), requirement_node(requirement["id"])]
                request, expected_action = _change(change_id, case_type, roots, operations, 3), "report"
            else:
                component_index = (case_index + 3) % len(after["components"])
                component = after["components"][component_index]
                old_part, new_part = component["part_number"], f"SYN-PARTIAL-{case_index:03d}"
                component["part_number"], after["revision"] = new_part, "B"
                after.pop("verification_activities")
                operations = [_operation("replace", f"components[{component_index}].part_number", old_part, new_part), _operation("replace", "revision", "A", "B")]
                request = _change(change_id, case_type, [component_node(component["id"])], operations, 3, False, ["after_design.verification_activities"])
                expected_action = "abstain"

            if expected_action == "report":
                affected, paths = _oracle_impact(before, after, request["root_node_ids"], request["max_depth"])
            else:
                affected, paths = [], []
            scenarios.append({
                "scenario_id": scenario_id, "category": "change_impact", "impact_case_type": case_type,
                "split": split, "before_design": before, "after_design": after,
                "change_request": request,
                "impact_oracle": {
                    "version": "0.1", "expected_action": expected_action,
                    "affected_node_ids": affected, "impact_paths": paths,
                },
                "provenance": copy.deepcopy(IMPACT_PROVENANCE),
            })
    return scenarios


def write_impact_benchmark(scenarios: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for scenario in scenarios:
            handle.write(json.dumps(scenario, sort_keys=True) + "\n")


def read_impact_benchmark(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
