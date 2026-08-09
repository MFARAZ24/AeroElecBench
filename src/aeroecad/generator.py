from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

RULE_IDS = ("AE-R001", "AE-R002", "AE-R003", "AE-R004", "AE-R005")
QUERY_TEMPLATES = {
    "AE-R001": (
        "Check whether every installed unit includes all mandatory metadata.",
        "Find equipment records with incomplete part-number or zone attributes.",
        "Verify the required component attributes in this design package.",
    ),
    "AE-R002": (
        "Find reused equipment identifiers in the component list.",
        "Check the design for duplicate component IDs.",
        "Verify that every installed-unit identifier is unique.",
    ),
    "AE-R003": (
        "Verify that cable endpoints resolve to declared connector pins.",
        "Find dangling wires or unresolved component and pin references.",
        "Check whether every wire endpoint names an existing connector pin.",
    ),
    "AE-R004": (
        "Review whether connected pins use compatible electrical signal classes.",
        "Find signal-class mismatches across wired interfaces.",
        "Check the electrical compatibility of each connection.",
    ),
    "AE-R005": (
        "Confirm every wire is linked to an applicable design requirement.",
        "Find untraced wiring with no declared requirement allocation.",
        "Check wire-to-requirement traceability coverage.",
    ),
}
PIN_DEFINITIONS = (
    {"id": "PWR", "signal_class": "POWER_28V"},
    {"id": "RTN", "signal_class": "GROUND"},
    {"id": "DATA_H", "signal_class": "ARINC429_HIGH"},
    {"id": "DATA_L", "signal_class": "ARINC429_LOW"},
)
WIRE_PIN_SEQUENCE = ("PWR", "RTN", "DATA_H", "DATA_L", "DATA_H", "PWR", "DATA_L")


def _base_design(index: int, rng: random.Random) -> dict[str, Any]:
    component_count = rng.randint(7, 10)
    design_id = f"SYN-ECAD-{index:04d}"
    components = []
    component_types = ("flight_display", "data_concentrator", "power_controller", "navigation_unit")
    zones = ("cockpit", "avionics_bay", "forward_fuselage", "aft_fuselage")
    for position in range(component_count):
        components.append({
            "id": f"LRU-{index:04d}-{position:02d}", "type": component_types[position % len(component_types)],
            "part_number": f"SYN-PN-{rng.randint(1000, 9999)}", "zone": zones[position % len(zones)],
            "pins": copy.deepcopy(PIN_DEFINITIONS),
        })
    requirements = [
        {"id": "REQ-PWR", "text": "Fictional 28 VDC distribution continuity requirement."},
        {"id": "REQ-DATA", "text": "Fictional differential avionics-data continuity requirement."},
    ]
    wires = []
    for position, pin_id in enumerate(WIRE_PIN_SEQUENCE):
        source, target = components[position % component_count], components[(position + 1) % component_count]
        requirement_id = "REQ-PWR" if pin_id in {"PWR", "RTN"} else "REQ-DATA"
        wires.append({
            "id": f"W-{index:04d}-{position:02d}", "gauge_awg": rng.choice((20, 22, 24)),
            "source": {"component_id": source["id"], "pin_id": pin_id},
            "target": {"component_id": target["id"], "pin_id": pin_id}, "requirement_ids": [requirement_id],
        })
    return {
        "design_id": design_id, "revision": "A", "synthetic": True,
        "notice": "Fictional research artifact; not approved for engineering or certification use.",
        "requirements": requirements, "components": components, "wires": wires,
    }


def _ground_truth(rule_id: str, path: str, entity_id: str, defect: str) -> dict[str, str]:
    return {"rule_id": rule_id, "entity_path": path, "entity_id": entity_id, "defect": defect}


def _inject(design: dict[str, Any], rule_id: str) -> dict[str, str]:
    if rule_id == "AE-R001":
        component = design["components"][0]
        component.pop("part_number")
        return _ground_truth(rule_id, "components[0].part_number", component["id"], "missing_required_attribute")
    if rule_id == "AE-R002":
        components = design["components"]
        previous_id = components[1]["id"]
        components[1]["id"] = components[0]["id"]
        for wire in design["wires"]:
            for endpoint_name in ("source", "target"):
                if wire[endpoint_name]["component_id"] == previous_id:
                    wire[endpoint_name]["component_id"] = components[1]["id"]
        return _ground_truth(rule_id, "components[1].id", components[1]["id"], "duplicate_identifier")
    if rule_id == "AE-R003":
        wire = design["wires"][0]
        wire["target"]["pin_id"] = "UNDECLARED_PIN"
        return _ground_truth(rule_id, "wires[0].target.pin_id", wire["id"], "invalid_pin_reference")
    if rule_id == "AE-R004":
        wire = design["wires"][1]
        wire["target"]["pin_id"] = "PWR"
        return _ground_truth(rule_id, "wires[1]", wire["id"], "incompatible_signal_classes")
    if rule_id == "AE-R005":
        wire = design["wires"][2]
        wire["requirement_ids"] = []
        return _ground_truth(rule_id, "wires[2].requirement_ids", wire["id"], "missing_requirement_trace")
    raise ValueError(f"Unsupported synthetic defect rule: {rule_id}")


def generate_benchmark(seed: int = 2027, cases_per_rule: int = 20, clean_cases: int = 20, mixed_cases: int = 50) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    scenarios: list[dict[str, Any]] = []
    design_index = 0
    for clean_index in range(clean_cases):
        design = _base_design(design_index, rng)
        scenarios.append({
            "scenario_id": f"clean-{clean_index:03d}", "category": "clean", "review_query": "Perform a complete electrical-design compliance review.",
            "design": design, "ground_truth": [],
        })
        design_index += 1
    for rule_id in RULE_IDS:
        for case_index in range(cases_per_rule):
            design = _base_design(design_index, rng)
            truth = _inject(design, rule_id)
            templates = QUERY_TEMPLATES[rule_id]
            scenarios.append({
                "scenario_id": f"single-{rule_id.lower()}-{case_index:03d}", "category": "single_fault",
                "review_query": templates[case_index % len(templates)], "design": design, "ground_truth": [truth],
            })
            design_index += 1
    for mixed_index in range(mixed_cases):
        design = _base_design(design_index, rng)
        selected = rng.sample(RULE_IDS, rng.randint(2, len(RULE_IDS)))
        truth = [_inject(design, rule_id) for rule_id in selected]
        scenarios.append({
            "scenario_id": f"mixed-{mixed_index:03d}", "category": "mixed_fault", "review_query": "Perform a complete electrical-design compliance review.",
            "design": design, "ground_truth": sorted(truth, key=lambda item: item["rule_id"]),
        })
        design_index += 1
    return scenarios


def write_jsonl(scenarios: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for scenario in scenarios:
            handle.write(json.dumps(scenario, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
