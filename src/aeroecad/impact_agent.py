from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from typing import Any

from .impact_graph import analyze_change_impact, build_version_graph, compute_version_diff, resolve_change_roots
from .ollama import OllamaClient

IMPACT_MODES = ("llm_only", "text_rag", "graph_deterministic", "assurance_agent")
IMPACT_RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["report", "abstain", "no_change"]},
        "affected_node_ids": {"type": "array", "items": {"type": "string"}},
        "impact_edges": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "source": {"type": "string"}, "target": {"type": "string"}, "relation": {"type": "string"},
                },
                "required": ["source", "target", "relation"],
            },
        },
        "rationale": {"type": "string", "maxLength": 300},
    },
    "required": ["action", "affected_node_ids", "impact_edges", "rationale"],
}
PLAN_RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["analyze", "abstain", "no_change"]},
        "tool_plan": {
            "type": "array", "items": {"type": "string", "enum": [
                "validate_change_evidence", "compute_version_diff", "build_version_graph",
                "resolve_change_roots", "traverse_dependencies", "validate_impact_report",
            ]},
        },
        "rationale": {"type": "string", "maxLength": 240},
    },
    "required": ["action", "tool_plan", "rationale"],
}
REQUIRED_TOOL_PLAN = [
    "validate_change_evidence", "compute_version_diff", "build_version_graph",
    "resolve_change_roots", "traverse_dependencies", "validate_impact_report",
]
DIRECT_SYSTEM_PROMPT = """Analyze a fictional synthetic aerospace electrical-design version pair. Infer changed graph roots from the structured operations, then report every affected component, pin, wire, requirement, and verification node within the stated depth. Return exact node IDs and the directed evidence edges connecting them. Abstain when required evidence is incomplete. Return one JSON object only; do not invent identifiers or relations."""
RAG_SYSTEM_PROMPT = """Analyze a fictional synthetic aerospace electrical-design change using only the retrieved text chunks. Report every supported affected node and directed evidence edge within the stated depth. Do not assume omitted chunks are evidence. Abstain if the evidence is insufficient. Return one JSON object only; do not invent identifiers or relations."""
PLAN_SYSTEM_PROMPT = """You are a bounded change-impact planning agent. Select the complete ordered tool plan needed to validate evidence, compute a version diff, build a typed graph, resolve changed roots, traverse dependencies, and validate the final report. Do not produce affected nodes yourself. Return one JSON object only."""
_TOKENS = re.compile(r"[A-Za-z0-9_:-]+")


def _public_change(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "change_id": request.get("change_id"), "change_type": request.get("change_type"),
        "operations": copy.deepcopy(request.get("operations", [])),
        "analysis_depth_limit": int(request.get("max_depth", 3)),
        "evidence_complete": bool(request.get("evidence_complete", True)),
        "missing_evidence": list(request.get("missing_evidence", [])),
    }


def _chunks(design: dict[str, Any], version: str) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for component in design.get("components", []):
        node_id = f"component:{component.get('id', '')}"
        pins = [{"node_id": f"pin:{component.get('id', '')}:{pin.get('id', '')}", **pin} for pin in component.get("pins", [])]
        chunks.append({"chunk_id": f"{version}:{node_id}", "text": json.dumps({"version": version, "node_id": node_id, "part_number": component.get("part_number"), "pins": pins}, sort_keys=True)})
    for wire in design.get("wires", []):
        source, target = wire.get("source", {}), wire.get("target", {})
        data = {
            "version": version, "node_id": f"wire:{wire.get('id', '')}",
            "source_pin": f"pin:{source.get('component_id', '')}:{source.get('pin_id', '')}",
            "target_pin": f"pin:{target.get('component_id', '')}:{target.get('pin_id', '')}",
            "source_component": f"component:{source.get('component_id', '')}",
            "target_component": f"component:{target.get('component_id', '')}",
            "requirement_nodes": [f"requirement:{item}" for item in wire.get("requirement_ids", [])],
        }
        chunks.append({"chunk_id": f"{version}:{data['node_id']}", "text": json.dumps(data, sort_keys=True)})
    for requirement in design.get("requirements", []):
        node_id = f"requirement:{requirement.get('id', '')}"
        chunks.append({"chunk_id": f"{version}:{node_id}", "text": json.dumps({"version": version, "node_id": node_id, "text": requirement.get("text", "")}, sort_keys=True)})
    for activity in design.get("verification_activities", []):
        node_id = f"verification:{activity.get('id', '')}"
        data = {
            "version": version, "node_id": node_id, "title": activity.get("title", ""),
            "requirement_nodes": [f"requirement:{item}" for item in activity.get("requirement_ids", [])],
            "component_nodes": [f"component:{item}" for item in activity.get("component_ids", [])],
        }
        chunks.append({"chunk_id": f"{version}:{node_id}", "text": json.dumps(data, sort_keys=True)})
    return chunks


def retrieve_text_chunks(scenario: dict[str, Any], top_k: int = 12) -> list[dict[str, Any]]:
    corpus = _chunks(scenario["before_design"], "before") + _chunks(scenario["after_design"], "after")
    query = json.dumps(_public_change(scenario["change_request"]), sort_keys=True)
    query_terms = Counter(token.lower() for token in _TOKENS.findall(query))
    ranked = []
    for chunk in corpus:
        terms = Counter(token.lower() for token in _TOKENS.findall(chunk["text"]))
        overlap = sum(min(count, terms[token]) for token, count in query_terms.items())
        ranked.append({**chunk, "retrieval_score": overlap})
    return sorted(ranked, key=lambda item: (-item["retrieval_score"], item["chunk_id"]))[:max(1, top_k)]


def build_impact_prompt(scenario: dict[str, Any], mode: str, top_k: int = 12) -> tuple[str, list[dict[str, Any]]]:
    if mode not in {"llm_only", "text_rag"}:
        raise ValueError("Impact prompts are only defined for llm_only and text_rag")
    evidence = retrieve_text_chunks(scenario, top_k) if mode == "text_rag" else []
    payload: dict[str, Any] = {
        "task": "Predict the complete change-impact set and directed evidence paths, or abstain.",
        "change": _public_change(scenario["change_request"]),
        "node_id_formats": ["component:<id>", "pin:<component-id>:<pin-id>", "wire:<id>", "requirement:<id>", "verification:<id>"],
        "allowed_relations": ["contains_pin", "source_endpoint_of", "target_endpoint_of", "connected_source_component", "connected_target_component", "traced_to_requirement", "allocated_wire", "verified_by", "covered_by_verification"],
        "output_contract": "For report, affected_node_ids must be unique and impact_edges must be the directed evidence tree connecting inferred changed roots to affected nodes. For abstain or no_change, both arrays must be empty.",
    }
    if mode == "llm_only":
        payload.update({"before_design": scenario["before_design"], "after_design": scenario["after_design"]})
    else:
        payload["retrieved_text_chunks"] = evidence
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False), evidence


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


def _report(status: str, reason: str = "", nodes: list[str] | None = None, paths: list[dict[str, Any]] | None = None, trace: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": status, "affected_node_ids": nodes or [], "impact_paths": paths or [],
        "abstention_reason": reason if status == "abstained" else "", "rejection_reason": reason if status == "rejected" else "",
        "tool_trace": trace or [], "production_modification_performed": False, "input_designs_unchanged": True,
    }


def _paths_from_edges(roots: set[str], nodes: set[str], edges: set[tuple[str, str, str]], max_depth: int) -> list[dict[str, Any]]:
    adjacency: dict[str, list[tuple[str, str]]] = {node: [] for node in nodes}
    for source, target, relation in edges:
        adjacency.setdefault(source, []).append((target, relation))
    for values in adjacency.values():
        values.sort()
    discovered: dict[str, tuple[str, list[str], list[str]]] = {}
    queue: list[tuple[str, str, list[str], list[str]]] = []
    for root in sorted(roots & nodes):
        discovered[root] = (root, [root], [])
        queue.append((root, root, [root], []))
    while queue:
        root, current, node_path, relations = queue.pop(0)
        if len(relations) >= max_depth:
            continue
        for target, relation in adjacency.get(current, []):
            if target in discovered:
                continue
            next_nodes, next_relations = [*node_path, target], [*relations, relation]
            discovered[target] = (root, next_nodes, next_relations)
            queue.append((root, target, next_nodes, next_relations))
    return [
        {"root_id": root, "target_id": target, "node_ids": node_path, "relations": relations}
        for target, (root, node_path, relations) in sorted(discovered.items())
    ]


def parse_impact_content(content: str, scenario: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _extract_json(content)
    required = {"action", "affected_node_ids", "impact_edges", "rationale"}
    if payload is None or set(payload) != required:
        return _report("rejected", "Response is not one object with exactly the required fields."), {"parse_success": False, "error": "invalid_output_shape"}
    if payload["action"] not in {"report", "abstain", "no_change"} or not isinstance(payload["affected_node_ids"], list) or not isinstance(payload["impact_edges"], list) or not isinstance(payload["rationale"], str):
        return _report("rejected", "Response fields have invalid types or values."), {"parse_success": False, "error": "invalid_output_types"}
    if payload["action"] != "report":
        if payload["affected_node_ids"] or payload["impact_edges"]:
            return _report("rejected", "A non-report action cannot contain impact results."), {"parse_success": False, "error": "invalid_nonreport_shape"}
        status = "abstained" if payload["action"] == "abstain" else "no_change"
        return _report(status, payload["rationale"] if status == "abstained" else ""), {"parse_success": True, "error": ""}

    graph = build_version_graph(scenario["before_design"], scenario["after_design"])
    known_nodes = set(graph["nodes"])
    roots = set(resolve_change_roots(scenario["before_design"], scenario["after_design"], scenario["change_request"].get("operations", [])))
    nodes = payload["affected_node_ids"]
    if not all(isinstance(item, str) for item in nodes) or len(nodes) != len(set(nodes)) or not set(nodes) <= known_nodes:
        return _report("rejected", "Affected nodes contain duplicates or unknown identifiers."), {"parse_success": False, "error": "invalid_affected_nodes"}
    graph_edges = {(item["source"], item["target"], item["relation"]) for item in graph["edges"]}
    predicted_edges: set[tuple[str, str, str]] = set()
    for edge in payload["impact_edges"]:
        if not isinstance(edge, dict) or set(edge) != {"source", "target", "relation"} or not all(isinstance(edge[key], str) for key in edge):
            return _report("rejected", "Impact edge has an invalid shape."), {"parse_success": False, "error": "invalid_edge_shape"}
        item = edge["source"], edge["target"], edge["relation"]
        if item not in graph_edges or item[0] not in nodes or item[1] not in nodes or item in predicted_edges:
            return _report("rejected", "Impact edge is duplicated or unsupported by the version graph."), {"parse_success": False, "error": "invalid_impact_edge"}
        predicted_edges.add(item)
    paths = _paths_from_edges(roots, set(nodes), predicted_edges, int(scenario["change_request"].get("max_depth", 3)))
    return _report("completed", nodes=sorted(nodes), paths=paths), {"parse_success": True, "error": ""}


def _parse_plan(content: str) -> tuple[dict[str, Any] | None, str]:
    payload = _extract_json(content)
    if payload is None or set(payload) != {"action", "tool_plan", "rationale"}:
        return None, "invalid_plan_shape"
    if payload["action"] not in {"analyze", "abstain", "no_change"} or not isinstance(payload["tool_plan"], list) or not all(isinstance(item, str) for item in payload["tool_plan"]) or not isinstance(payload["rationale"], str):
        return None, "invalid_plan_types"
    return payload, ""


class ImpactAgent:
    def __init__(self, client: OllamaClient):
        self.client = client

    def run(self, scenario: dict[str, Any], model: str, mode: str, seed: int = 7107, max_tokens: int = 2500, retrieval_top_k: int = 12) -> dict[str, Any]:
        if mode not in IMPACT_MODES:
            raise ValueError(f"mode must be one of: {', '.join(IMPACT_MODES)}")
        base = {"scenario_id": scenario["scenario_id"], "mode": mode, "model": None if mode == "graph_deterministic" else model}
        if mode == "graph_deterministic":
            return {**base, "report": analyze_change_impact(scenario["before_design"], scenario["after_design"], scenario["change_request"]), "llm_call_count": 0, "prompt_sha256": "", "raw_content": "", "ollama_metadata": {}, "retrieved_chunk_ids": [], "plan": None}
        if mode == "assurance_agent":
            return {**base, **self._run_assurance(scenario, model, seed, max_tokens)}

        prompt, evidence = build_impact_prompt(scenario, mode, retrieval_top_k)
        system = DIRECT_SYSTEM_PROMPT if mode == "llm_only" else RAG_SYSTEM_PROMPT
        response = self.client.chat(model, system, prompt, seed=seed, max_tokens=max_tokens, response_schema=IMPACT_RESPONSE_SCHEMA)
        report, diagnostics = parse_impact_content(response.content, scenario)
        report["tool_trace"] = [] if mode == "llm_only" else ["retrieve_text_chunks"]
        return {
            **base, "report": report, "llm_call_count": 1,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_content": response.content, "ollama_metadata": response.metadata,
            "retrieved_chunk_ids": [item["chunk_id"] for item in evidence], "diagnostics": diagnostics, "plan": None,
        }

    def _run_assurance(self, scenario: dict[str, Any], model: str, seed: int, max_tokens: int) -> dict[str, Any]:
        request = scenario["change_request"]
        graph = build_version_graph(scenario["before_design"], scenario["after_design"])
        missing = sorted(set(graph["missing_paths"] + list(request.get("missing_evidence", []))))
        if not request.get("evidence_complete", True) or missing or not compute_version_diff(scenario["before_design"], scenario["after_design"]):
            report = analyze_change_impact(scenario["before_design"], scenario["after_design"], request)
            return {"report": report, "llm_call_count": 0, "prompt_sha256": "", "raw_content": "", "ollama_metadata": {}, "retrieved_chunk_ids": [], "diagnostics": {"parse_success": True, "error": ""}, "plan": None}
        prompt = json.dumps({
            "task": "Choose the ordered tool plan for this change-impact analysis.", "change": _public_change(request),
            "observed_diff_count": len(compute_version_diff(scenario["before_design"], scenario["after_design"])),
            "available_tools": [
                {"name": name, "required_before": REQUIRED_TOOL_PLAN[index - 1] if index else None}
                for index, name in enumerate(REQUIRED_TOOL_PLAN)
            ],
            "execution_policy": "All six tools are required in the listed dependency order for a complete report.",
        }, separators=(",", ":"), ensure_ascii=False)
        response = self.client.chat(model, PLAN_SYSTEM_PROMPT, prompt, seed=seed, max_tokens=min(max_tokens, 400), response_schema=PLAN_RESPONSE_SCHEMA)
        plan, error = _parse_plan(response.content)
        if plan is None:
            report = _report("rejected", "The agent returned an invalid tool plan.")
        elif plan["action"] == "abstain":
            report = _report("abstained", plan["rationale"])
        elif plan["action"] == "no_change":
            report = _report("no_change")
        elif plan["tool_plan"] != REQUIRED_TOOL_PLAN:
            report = _report("rejected", "The proposed plan omitted, reordered, or duplicated required assurance tools.", trace=plan["tool_plan"])
            error = "incomplete_tool_plan"
        else:
            report = analyze_change_impact(scenario["before_design"], scenario["after_design"], request)
        return {
            "report": report, "llm_call_count": 1, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_content": response.content, "ollama_metadata": response.metadata, "retrieved_chunk_ids": [],
            "diagnostics": {"parse_success": not error, "error": error}, "plan": plan,
        }
