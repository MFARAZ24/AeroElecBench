from __future__ import annotations

from typing import Any, Callable


def _finding(rule: dict[str, Any], path: str, entity_id: str, message: str, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "rule_id": rule["rule_id"], "severity": rule["severity"], "entity_path": path, "entity_id": entity_id,
        "message": message, "evidence": {"observed": observed, "expected": expected},
        "rule_citation": {"catalog_id": "AEROECAD-SYNTH-RULES-0.1", "section": rule["section"], "rule_id": rule["rule_id"]},
    }


def _required_attributes(design: dict[str, Any], rule: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    for index, component in enumerate(design["components"]):
        for attribute in rule["parameters"]["attributes"]:
            if attribute not in component or component[attribute] in (None, ""):
                findings.append(_finding(rule, f"components[{index}].{attribute}", component.get("id", f"index:{index}"), f"Required component attribute '{attribute}' is missing.", "missing", "non-empty value"))
    return findings


def _unique_ids(design: dict[str, Any], rule: dict[str, Any]) -> list[dict[str, Any]]:
    findings, first_seen = [], {}
    for index, component in enumerate(design["components"]):
        component_id = component.get("id")
        if component_id in first_seen:
            findings.append(_finding(rule, f"components[{index}].id", str(component_id), f"Component identifier '{component_id}' duplicates an earlier occurrence.", {"value": component_id, "first_path": f"components[{first_seen[component_id]}].id"}, "unique identifier"))
        else:
            first_seen[component_id] = index
    return findings


def _component_index(design: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for component in design["components"]:
        index.setdefault(component.get("id", ""), component)
    return index


def _resolve_endpoint(components: dict[str, dict[str, Any]], endpoint: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    component = components.get(endpoint.get("component_id"))
    if not component:
        return None, None
    pin = next((item for item in component.get("pins", []) if item.get("id") == endpoint.get("pin_id")), None)
    return component, pin


def _valid_references(design: dict[str, Any], rule: dict[str, Any]) -> list[dict[str, Any]]:
    findings, components = [], _component_index(design)
    for wire_index, wire in enumerate(design["wires"]):
        for endpoint_name in ("source", "target"):
            endpoint = wire[endpoint_name]
            component, pin = _resolve_endpoint(components, endpoint)
            if component is None:
                path = f"wires[{wire_index}].{endpoint_name}.component_id"
                findings.append(_finding(rule, path, wire["id"], "Wire endpoint references an undeclared component.", endpoint.get("component_id"), "declared component identifier"))
            elif pin is None:
                path = f"wires[{wire_index}].{endpoint_name}.pin_id"
                findings.append(_finding(rule, path, wire["id"], "Wire endpoint references an undeclared connector pin.", endpoint.get("pin_id"), f"declared pin on {component['id']}"))
    return findings


def _compatible_connections(design: dict[str, Any], rule: dict[str, Any]) -> list[dict[str, Any]]:
    findings, components = [], _component_index(design)
    for wire_index, wire in enumerate(design["wires"]):
        _, source_pin = _resolve_endpoint(components, wire["source"])
        _, target_pin = _resolve_endpoint(components, wire["target"])
        if not source_pin or not target_pin:
            continue
        source_class, target_class = source_pin.get("signal_class"), target_pin.get("signal_class")
        if source_class != target_class:
            findings.append(_finding(rule, f"wires[{wire_index}]", wire["id"], "Connected pins use incompatible signal classes.", {"source": source_class, "target": target_class}, "identical endpoint signal classes"))
    return findings


def _requirement_traceability(design: dict[str, Any], rule: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    known_requirements = {requirement["id"] for requirement in design["requirements"]}
    for wire_index, wire in enumerate(design["wires"]):
        references = wire.get("requirement_ids", [])
        unknown = sorted(set(references) - known_requirements)
        if not references:
            findings.append(_finding(rule, f"wires[{wire_index}].requirement_ids", wire["id"], "Wire has no requirement trace.", [], "one or more declared requirement identifiers"))
        elif unknown:
            findings.append(_finding(rule, f"wires[{wire_index}].requirement_ids", wire["id"], "Wire cites an undeclared requirement.", unknown, "declared requirement identifiers only"))
    return findings


Checker = Callable[[dict[str, Any], dict[str, Any]], list[dict[str, Any]]]
CHECKERS: dict[str, Checker] = {
    "required_component_attributes": _required_attributes,
    "unique_component_ids": _unique_ids,
    "valid_pin_references": _valid_references,
    "compatible_signal_classes": _compatible_connections,
    "wire_requirement_traceability": _requirement_traceability,
}


def validate_design(design: dict[str, Any], rules: list[dict[str, Any]], selected_rule_ids: set[str] | None = None) -> list[dict[str, Any]]:
    findings = []
    for rule in rules:
        if selected_rule_ids is not None and rule["rule_id"] not in selected_rule_ids:
            continue
        checker = CHECKERS.get(rule["checker"])
        if checker is None:
            raise ValueError(f"No checker registered for {rule['checker']}")
        findings.extend(checker(design, rule))
    return sorted(findings, key=lambda item: (item["rule_id"], item["entity_path"]))
