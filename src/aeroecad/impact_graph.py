from __future__ import annotations

import copy
from collections import deque
from typing import Any

REQUIRED_COLLECTIONS = ("components", "wires", "requirements", "verification_activities")


def component_node(component_id: str) -> str:
    return f"component:{component_id}"


def pin_node(component_id: str, pin_id: str) -> str:
    return f"pin:{component_id}:{pin_id}"


def wire_node(wire_id: str) -> str:
    return f"wire:{wire_id}"


def requirement_node(requirement_id: str) -> str:
    return f"requirement:{requirement_id}"


def verification_node(activity_id: str) -> str:
    return f"verification:{activity_id}"


def _add_node(nodes: dict[str, dict[str, Any]], node_id: str, entity_type: str, path: str, value: dict[str, Any]) -> None:
    nodes[node_id] = {"node_id": node_id, "entity_type": entity_type, "entity_path": path, "value": copy.deepcopy(value)}


def build_design_graph(design: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in REQUIRED_COLLECTIONS if not isinstance(design.get(name), list)]
    if missing:
        return {"nodes": {}, "edges": [], "missing_paths": [f"design.{name}" for name in missing]}

    nodes: dict[str, dict[str, Any]] = {}
    edges: set[tuple[str, str, str]] = set()
    for component_index, component in enumerate(design["components"]):
        component_id = str(component.get("id", ""))
        source = component_node(component_id)
        _add_node(nodes, source, "component", f"components[{component_index}]", component)
        for pin_index, pin in enumerate(component.get("pins", [])):
            target = pin_node(component_id, str(pin.get("id", "")))
            _add_node(nodes, target, "pin", f"components[{component_index}].pins[{pin_index}]", pin)
            edges.add((source, target, "contains_pin"))

    component_ids = {str(item.get("id", "")) for item in design["components"]}
    requirement_ids = {str(item.get("id", "")) for item in design["requirements"]}
    for requirement_index, requirement in enumerate(design["requirements"]):
        requirement_id = str(requirement.get("id", ""))
        _add_node(nodes, requirement_node(requirement_id), "requirement", f"requirements[{requirement_index}]", requirement)

    for wire_index, wire in enumerate(design["wires"]):
        wire_id, source = str(wire.get("id", "")), wire_node(str(wire.get("id", "")))
        _add_node(nodes, source, "wire", f"wires[{wire_index}]", wire)
        for endpoint_name in ("source", "target"):
            endpoint = wire.get(endpoint_name, {})
            component_id, pin_id = str(endpoint.get("component_id", "")), str(endpoint.get("pin_id", ""))
            target = pin_node(component_id, pin_id)
            if target in nodes:
                edges.add((target, source, f"{endpoint_name}_endpoint_of"))
            if component_id in component_ids:
                edges.add((source, component_node(component_id), f"connected_{endpoint_name}_component"))
        for requirement_id in wire.get("requirement_ids", []):
            requirement_id = str(requirement_id)
            if requirement_id in requirement_ids:
                edges.add((source, requirement_node(requirement_id), "traced_to_requirement"))
                edges.add((requirement_node(requirement_id), source, "allocated_wire"))

    for activity_index, activity in enumerate(design["verification_activities"]):
        activity_id = str(activity.get("id", ""))
        target = verification_node(activity_id)
        _add_node(nodes, target, "verification", f"verification_activities[{activity_index}]", activity)
        for requirement_id in activity.get("requirement_ids", []):
            requirement_id = str(requirement_id)
            if requirement_id in requirement_ids:
                edges.add((requirement_node(requirement_id), target, "verified_by"))
        for component_id in activity.get("component_ids", []):
            component_id = str(component_id)
            if component_id in component_ids:
                edges.add((component_node(component_id), target, "covered_by_verification"))
    return {
        "nodes": {key: nodes[key] for key in sorted(nodes)},
        "edges": [{"source": source, "target": target, "relation": relation} for source, target, relation in sorted(edges)],
        "missing_paths": [],
    }


def build_version_graph(before_design: dict[str, Any], after_design: dict[str, Any]) -> dict[str, Any]:
    before, after = build_design_graph(before_design), build_design_graph(after_design)
    nodes: dict[str, dict[str, Any]] = {}
    for version, graph in (("before", before), ("after", after)):
        for node_id, node in graph["nodes"].items():
            record = nodes.setdefault(node_id, {"node_id": node_id, "entity_type": node["entity_type"], "versions": {}})
            record["versions"][version] = {"entity_path": node["entity_path"], "value": node["value"]}
    edges = {
        (edge["source"], edge["target"], edge["relation"])
        for graph in (before, after) for edge in graph["edges"]
    }
    return {
        "nodes": {key: nodes[key] for key in sorted(nodes)},
        "edges": [{"source": source, "target": target, "relation": relation} for source, target, relation in sorted(edges)],
        "missing_paths": [f"before_{path}" for path in before["missing_paths"]] + [f"after_{path}" for path in after["missing_paths"]],
    }


def compute_version_diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        changes = []
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}" if path else key
            if key not in before:
                changes.append({"op": "add", "path": child, "before": None, "after": copy.deepcopy(after[key])})
            elif key not in after:
                changes.append({"op": "remove", "path": child, "before": copy.deepcopy(before[key]), "after": None})
            else:
                changes.extend(compute_version_diff(before[key], after[key], child))
        return changes
    if isinstance(before, list) and isinstance(after, list):
        changes = []
        for index in range(max(len(before), len(after))):
            child = f"{path}[{index}]"
            if index >= len(before):
                changes.append({"op": "add", "path": child, "before": None, "after": copy.deepcopy(after[index])})
            elif index >= len(after):
                changes.append({"op": "remove", "path": child, "before": copy.deepcopy(before[index]), "after": None})
            else:
                changes.extend(compute_version_diff(before[index], after[index], child))
        return changes
    return [] if before == after else [{"op": "replace", "path": path, "before": copy.deepcopy(before), "after": copy.deepcopy(after)}]


def traverse_impact(graph: dict[str, Any], root_ids: list[str], max_depth: int) -> tuple[list[str], list[dict[str, Any]]]:
    adjacency: dict[str, list[tuple[str, str]]] = {node_id: [] for node_id in graph["nodes"]}
    for edge in graph["edges"]:
        adjacency.setdefault(edge["source"], []).append((edge["target"], edge["relation"]))
    for values in adjacency.values():
        values.sort()

    paths: dict[str, tuple[str, list[str], list[str]]] = {}
    queue: deque[tuple[str, str, list[str], list[str]]] = deque()
    for root_id in sorted(set(root_ids)):
        paths[root_id] = (root_id, [root_id], [])
        queue.append((root_id, root_id, [root_id], []))
    while queue:
        root_id, current, node_path, relations = queue.popleft()
        if len(relations) >= max_depth:
            continue
        for target, relation in adjacency.get(current, []):
            if target in paths:
                continue
            target_nodes, target_relations = [*node_path, target], [*relations, relation]
            paths[target] = (root_id, target_nodes, target_relations)
            queue.append((root_id, target, target_nodes, target_relations))
    records = [
        {"root_id": root, "target_id": target, "node_ids": nodes, "relations": relations}
        for target, (root, nodes, relations) in sorted(paths.items())
    ]
    return sorted(paths), records


def analyze_change_impact(before_design: dict[str, Any], after_design: dict[str, Any], change_request: dict[str, Any]) -> dict[str, Any]:
    before_copy, after_copy = copy.deepcopy(before_design), copy.deepcopy(after_design)
    graph = build_version_graph(before_design, after_design)
    observed_diff = compute_version_diff(before_design, after_design)
    missing = sorted(set(graph["missing_paths"] + list(change_request.get("missing_evidence", []))))
    base = {
        "change_id": change_request.get("change_id"), "change_type": change_request.get("change_type"),
        "observed_diff": observed_diff, "graph_node_count": len(graph["nodes"]), "graph_edge_count": len(graph["edges"]),
        "tool_trace": ["parse_versions", "compute_structured_diff", "build_typed_graph", "traverse_dependencies"],
        "production_modification_performed": False,
    }
    if not change_request.get("evidence_complete", True) or missing:
        return {
            **base, "status": "abstained", "affected_node_ids": [], "impact_paths": [],
            "abstention_reason": f"Required evidence is incomplete: {', '.join(missing) or 'unspecified evidence'}.",
            "input_designs_unchanged": before_design == before_copy and after_design == after_copy,
        }
    if change_request.get("change_type") == "no_change":
        status = "no_change" if not observed_diff else "abstained"
        return {
            **base, "status": status, "affected_node_ids": [], "impact_paths": [],
            "abstention_reason": "" if status == "no_change" else "Observed differences contradict the no-change request.",
            "input_designs_unchanged": before_design == before_copy and after_design == after_copy,
        }
    root_ids = sorted(set(change_request.get("root_node_ids", [])))
    unavailable = sorted(set(root_ids) - set(graph["nodes"]))
    if not root_ids or unavailable:
        return {
            **base, "status": "abstained", "affected_node_ids": [], "impact_paths": [],
            "abstention_reason": "No resolvable change root." if not root_ids else f"Unresolved change roots: {', '.join(unavailable)}.",
            "input_designs_unchanged": before_design == before_copy and after_design == after_copy,
        }
    affected, paths = traverse_impact(graph, root_ids, int(change_request.get("max_depth", 3)))
    return {
        **base, "status": "completed", "affected_node_ids": affected, "impact_paths": paths, "abstention_reason": "",
        "input_designs_unchanged": before_design == before_copy and after_design == after_copy,
    }
