from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from .ollama import OllamaClient
from .repair import PatchOperation, RepairProposal, RepairStatus, Repairability, classify_repairability, execute_repair
from .validator import validate_design

REPAIR_MODES = ("llm_direct", "tool_evidence_grounded", "deterministic_auto")
REPAIR_RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "operations": {
            "type": "array", "maxItems": 1,
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "op": {"type": "string", "enum": ["replace"]},
                    "path": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["op", "path", "value"],
            },
        },
        "rationale": {"type": "string", "maxLength": 240},
        "abstained": {"type": "boolean"},
        "abstention_reason": {"type": "string"},
    },
    "required": ["operations", "rationale", "abstained", "abstention_reason"],
}
DIRECT_SYSTEM_PROMPT = """You propose one bounded repair for a fictional synthetic aerospace electrical-design JSON. Apply the supplied rule to the design and finding. Never invent a path. If the correction is not supported, abstain. Keep the rationale under 25 words. Return exactly one JSON object and no markdown. A deterministic verifier decides whether the sandbox candidate is accepted."""
EVIDENCE_SYSTEM_PROMPT = """You propose one bounded repair for a fictional synthetic aerospace electrical-design JSON. Apply the supplied rule using the tool-extracted peer endpoint and every declared target-pin candidate. The evidence does not identify the correct candidate; you must reason over the supplied properties. Never invent a path or candidate. If zero or multiple candidates are supported, abstain. Keep the rationale under 25 words. Return exactly one JSON object and no markdown. A deterministic verifier decides whether the sandbox candidate is accepted."""
_RESPONSE_KEYS = {"operations", "rationale", "abstained", "abstention_reason"}
_PIN_PATH = re.compile(r"^wires\[(\d+)\]\.(source|target)\.pin_id$")


def _canonical_path(value: Any) -> str:
    path = str(value or "").strip()
    if path.startswith("/"):
        parts = [segment.replace("~1", "/").replace("~0", "~") for segment in path.split("/") if segment]
        if parts and parts[0] in {"design", "$"}:
            parts = parts[1:]
        path = ".".join(parts)
    path = path.replace("['", ".").replace("']", "").replace('["', ".").replace('"]', "")
    path = re.sub(r"^(?:\$\.|design\.)", "", path)
    return re.sub(r"\.(\d+)(?=\.|$)", r"[\1]", path)


def _failed(reason: str, error: str, raw_count: int = 0, invalid_count: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = {"operations": (), "rationale": "", "abstained": True, "abstention_reason": reason}
    diagnostics = {"parse_success": False, "raw_operation_count": raw_count, "invalid_operation_count": invalid_count, "error": error}
    return parsed, diagnostics


def _pin_evidence(design: dict[str, Any], finding: dict[str, Any]) -> dict[str, Any]:
    match = _PIN_PATH.fullmatch(finding.get("entity_path", ""))
    if not match:
        return {"supported": False, "reason": "Finding is not a supported pin-reference path.", "declared_candidate_ids": [], "deterministic_matches": []}

    wire_index, endpoint_name = int(match.group(1)), match.group(2)
    if wire_index >= len(design.get("wires", [])):
        return {"supported": False, "reason": "Finding references an unavailable wire.", "declared_candidate_ids": [], "deterministic_matches": []}

    wire = design["wires"][wire_index]
    peer_name = "target" if endpoint_name == "source" else "source"
    endpoint, peer = wire[endpoint_name], wire[peer_name]
    components = design.get("components", [])
    target_components = [item for item in components if item.get("id") == endpoint.get("component_id")]
    peer_components = [item for item in components if item.get("id") == peer.get("component_id")]

    peer_pins = [
        {"component_id": component.get("id"), "pin_id": pin.get("id"), "signal_class": pin.get("signal_class")}
        for component in peer_components
        for pin in component.get("pins", [])
        if pin.get("id") == peer.get("pin_id")
    ]
    peer_classes = {item["signal_class"] for item in peer_pins if item["signal_class"]}

    target_pin_sets = [
        {pin.get("id") for pin in component.get("pins", []) if pin.get("id")}
        for component in target_components
    ]
    declared_ids = sorted(set.intersection(*target_pin_sets)) if target_pin_sets else []
    target_pins = sorted({
        (pin.get("id"), pin.get("signal_class"), pin.get("interface_role"))
        for component in target_components for pin in component.get("pins", [])
        if pin.get("id") in declared_ids
    }, key=lambda item: (str(item[0]), str(item[1]), str(item[2])))
    deterministic_matches = []
    if len(peer_classes) == 1:
        required_class = next(iter(peer_classes))
        deterministic_matches = sorted({pin_id for pin_id, signal_class, _ in target_pins if signal_class == required_class})
    else:
        required_class = None

    requirement_ids = set(wire.get("requirement_ids", []))
    wire_requirements = [
        {"requirement_id": item.get("id"), "text": item.get("text", "")}
        for item in design.get("requirements", []) if item.get("id") in requirement_ids
    ]

    return {
        "supported": bool(target_components and peer_pins),
        "reason": "" if target_components and peer_pins else "Endpoint evidence could not be resolved.",
        "defective_endpoint": {
            "endpoint_name": endpoint_name, "component_id": endpoint.get("component_id"),
            "observed_pin_id": endpoint.get("pin_id"),
        },
        "peer_endpoint": {
            "endpoint_name": peer_name, "component_id": peer.get("component_id"),
            "pin_id": peer.get("pin_id"), "resolved_pins": peer_pins,
        },
        "target_declared_pins": [
            {"pin_id": pin_id, "signal_class": signal_class, "interface_role": interface_role}
            for pin_id, signal_class, interface_role in target_pins
        ],
        "wire_requirements": wire_requirements,
        "declared_candidate_ids": declared_ids,
        "peer_signal_class": required_class,
        "deterministic_matches": deterministic_matches,
    }


def _model_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "defective_endpoint": evidence.get("defective_endpoint"),
        "peer_endpoint": evidence.get("peer_endpoint"),
        "peer_signal_class": evidence.get("peer_signal_class"),
        "target_declared_pins": evidence.get("target_declared_pins", []),
        "wire_requirements": evidence.get("wire_requirements", []),
        "instruction": "Choose by applying the rule to the evidence. No candidate has been preselected.",
    }


def _has_unique_requirement_support(evidence: dict[str, Any]) -> bool:
    matches = set(evidence.get("deterministic_matches", []))
    if len(matches) == 1:
        return True
    requirement_text = " ".join(item.get("text", "") for item in evidence.get("wire_requirements", [])).lower()
    primary_matches = [
        item["pin_id"] for item in evidence.get("target_declared_pins", [])
        if item.get("pin_id") in matches and item.get("interface_role") == "primary"
    ]
    return "primary" in requirement_text and len(primary_matches) == 1


def parse_repair_content(content: str, allowed_paths: set[str], allowed_values: set[str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return _failed("unparseable_output", "No JSON object found")
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError as error:
            return _failed("unparseable_output", str(error))

    if not isinstance(payload, dict) or set(payload) != _RESPONSE_KEYS:
        return _failed("invalid_output_shape", "Response must contain exactly the required repair fields")
    if not isinstance(payload["operations"], list):
        return _failed("invalid_operations", "operations must be an array")
    if not isinstance(payload["rationale"], str) or not isinstance(payload["abstention_reason"], str) or not isinstance(payload["abstained"], bool):
        return _failed("invalid_output_types", "Repair response fields use invalid types", len(payload["operations"]))

    raw_operations = payload["operations"]
    if payload["abstained"]:
        if raw_operations:
            return _failed("invalid_abstention_shape", "An abstaining response cannot include operations", len(raw_operations), len(raw_operations))
        return (
            {"operations": (), "rationale": payload["rationale"], "abstained": True, "abstention_reason": payload["abstention_reason"] or "model_abstained"},
            {"parse_success": True, "raw_operation_count": 0, "invalid_operation_count": 0, "error": ""},
        )

    operations, invalid = [], 0
    for item in raw_operations:
        if not isinstance(item, dict) or set(item) != {"op", "path", "value"}:
            invalid += 1
            continue
        path, value = _canonical_path(item["path"]), item["value"]
        value_allowed = isinstance(value, str) and value and (allowed_values is None or value in allowed_values)
        if item["op"] != "replace" or path not in allowed_paths or not value_allowed:
            invalid += 1
            continue
        operations.append(PatchOperation("replace", path, value))

    if invalid or len(operations) != 1:
        return _failed("invalid_patch_operation", "Exactly one valid operation using the permitted path and candidates is required", len(raw_operations), invalid or abs(1 - len(operations)))
    return (
        {"operations": tuple(operations), "rationale": payload["rationale"], "abstained": False, "abstention_reason": ""},
        {"parse_success": True, "raw_operation_count": len(raw_operations), "invalid_operation_count": 0, "error": ""},
    )


class LLMRepairAgent:
    def __init__(self, catalog: dict[str, Any], client: OllamaClient):
        self.rules, self.client = catalog["rules"], client
        self.rule_index = {rule["rule_id"]: rule for rule in self.rules}

    def _prompt(self, design: dict[str, Any], finding: dict[str, Any], mode: str, evidence: dict[str, Any]) -> str:
        rule = self.rule_index[finding["rule_id"]]
        sections = {
            "task": "Propose one correction or abstain.",
            "repair_mode": mode,
            "finding": {key: finding.get(key) for key in ("rule_id", "entity_path", "entity_id", "message", "evidence")},
            "rule_criteria": {key: rule[key] for key in ("rule_id", "title", "section", "severity", "description", "parameters")},
            "allowed_patch_paths": [finding["entity_path"]],
            "allowed_operations": ["replace"],
            "output_contract": {
                "operations": "Exactly one replace operation, or an empty array when abstaining.",
                "rationale": "Maximum 25 words; state only the decisive evidence.",
            },
            "design": design,
        }
        if mode == "tool_evidence_grounded":
            sections["tool_evidence"] = _model_evidence(evidence)
        return json.dumps(sections, separators=(",", ":"), ensure_ascii=False)

    def repair(self, design: dict[str, Any], model: str, mode: str = "llm_direct", seed: int = 2027, max_tokens: int = 400) -> dict[str, Any]:
        if mode not in REPAIR_MODES:
            raise ValueError(f"mode must be one of: {', '.join(REPAIR_MODES)}")
        original, working = copy.deepcopy(design), copy.deepcopy(design)
        initial_findings = validate_design(working, self.rules)
        attempts, accepted_count = [], 0

        for finding in initial_findings:
            repairability = classify_repairability(finding)
            base = {"rule_id": finding["rule_id"], "entity_path": finding["entity_path"], "repairability": repairability.value, "repair_mode": mode}

            if repairability not in {Repairability.AUTOMATIC, Repairability.CONSTRAINED}:
                attempts.append({
                    **base, "status": RepairStatus.ABSTAINED.value, "reason": "Repair requires human engineering judgment.",
                    "llm_call_performed": False, "tool_evidence": None, "proposal": None,
                    "introduced_findings": [], "resolved_findings": [],
                    "diagnostics": {"parse_success": True, "raw_operation_count": 0, "invalid_operation_count": 0, "error": ""},
                    "prompt_sha256": "", "ollama_metadata": {}, "raw_content": "",
                })
                continue

            evidence = _pin_evidence(working, finding)
            if mode == "tool_evidence_grounded" and not _has_unique_requirement_support(evidence):
                attempts.append({
                    **base, "status": RepairStatus.ABSTAINED.value,
                    "reason": "Grounding policy found no uniquely supported candidate; human review required.",
                    "llm_call_performed": False, "tool_evidence": _model_evidence(evidence), "proposal": None,
                    "introduced_findings": [], "resolved_findings": [],
                    "diagnostics": {"parse_success": True, "raw_operation_count": 0, "invalid_operation_count": 0, "error": ""},
                    "prompt_sha256": "", "ollama_metadata": {}, "raw_content": "",
                })
                continue
            if mode == "deterministic_auto":
                matches = evidence["deterministic_matches"]
                if len(matches) != 1:
                    attempts.append({
                        **base, "status": RepairStatus.ABSTAINED.value,
                        "reason": f"Deterministic repair requires one compatible candidate; found {len(matches)}.",
                        "llm_call_performed": False, "tool_evidence": _model_evidence(evidence), "proposal": None,
                        "introduced_findings": [], "resolved_findings": [],
                        "diagnostics": {"parse_success": True, "raw_operation_count": 0, "invalid_operation_count": 0, "error": ""},
                        "prompt_sha256": "", "ollama_metadata": {}, "raw_content": "",
                    })
                    continue
                parsed = {
                    "operations": (PatchOperation("replace", finding["entity_path"], matches[0]),),
                    "rationale": "Unique compatible candidate selected by deterministic signal-class matching.",
                }
                diagnostics = {"parse_success": True, "raw_operation_count": 0, "invalid_operation_count": 0, "error": ""}
                response_metadata, raw_content, prompt_hash, llm_called = {}, "", "", False
            else:
                prompt = self._prompt(working, finding, mode, evidence)
                system = EVIDENCE_SYSTEM_PROMPT if mode == "tool_evidence_grounded" else DIRECT_SYSTEM_PROMPT
                response = self.client.chat(model, system, prompt, seed=seed, max_tokens=max_tokens, response_schema=REPAIR_RESPONSE_SCHEMA)
                allowed_values = set(evidence["declared_candidate_ids"]) if mode == "tool_evidence_grounded" else None
                parsed, diagnostics = parse_repair_content(response.content, {finding["entity_path"]}, allowed_values)
                response_metadata, raw_content = response.metadata, response.content
                prompt_hash, llm_called = hashlib.sha256(prompt.encode("utf-8")).hexdigest(), True
                if parsed["abstained"] or not diagnostics["parse_success"]:
                    status = RepairStatus.ABSTAINED if diagnostics["parse_success"] else RepairStatus.REJECTED
                    attempts.append({
                        **base, "status": status.value,
                        "reason": parsed["abstention_reason"] or diagnostics["error"] or "No valid repair proposal.",
                        "llm_call_performed": True,
                        "tool_evidence": _model_evidence(evidence) if mode == "tool_evidence_grounded" else None,
                        "proposal": None, "introduced_findings": [], "resolved_findings": [],
                        "diagnostics": diagnostics, "prompt_sha256": prompt_hash,
                        "ollama_metadata": response_metadata, "raw_content": raw_content,
                    })
                    continue

            proposal = RepairProposal(finding, parsed["operations"], parsed["rationale"])
            result = execute_repair(working, self.rules, proposal)
            proposal_record = {
                "operations": [{"op": item.op, "path": item.path, "value": item.value} for item in proposal.operations],
                "rationale": proposal.rationale,
            }
            attempts.append({
                **base, "status": result.status.value, "reason": result.reason,
                "llm_call_performed": llm_called,
                "tool_evidence": _model_evidence(evidence) if mode != "llm_direct" else None,
                "proposal": proposal_record, "introduced_findings": list(result.introduced_findings),
                "resolved_findings": list(result.resolved_findings), "diagnostics": diagnostics,
                "prompt_sha256": prompt_hash, "ollama_metadata": response_metadata, "raw_content": raw_content,
            })
            if result.status == RepairStatus.ACCEPTED:
                working, accepted_count = result.design, accepted_count + 1

        final_findings = validate_design(working, self.rules)
        if not initial_findings:
            status = "no_repair_required"
        elif not final_findings:
            status = "repaired_in_sandbox"
        elif accepted_count:
            status = "partial_repair_review_required"
        else:
            status = "review_required"

        return {
            "model": model, "repair_mode": mode, "design_id": design["design_id"], "status": status,
            "initial_findings": initial_findings, "attempts": attempts, "final_findings": final_findings,
            "repaired_design": working, "accepted_repair_count": accepted_count,
            "automatic_modification_performed": bool(accepted_count),
            "sandbox_modification_performed": bool(accepted_count),
            "production_modification_performed": False,
            "human_approval_required": bool(final_findings),
            "input_design_unchanged": design == original,
            "decision_source": "deterministic_validator",
            "proposal_source": "deterministic_tool" if mode == "deterministic_auto" else mode,
        }
