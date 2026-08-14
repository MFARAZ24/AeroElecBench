from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from .ollama import OllamaClient
from .repair import PatchOperation, RepairProposal, RepairStatus, Repairability, classify_repairability, execute_repair
from .validator import validate_design

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
        "rationale": {"type": "string"},
        "abstained": {"type": "boolean"},
        "abstention_reason": {"type": "string"},
    },
    "required": ["operations", "rationale", "abstained", "abstention_reason"],
}
REPAIR_SYSTEM_PROMPT = """You propose one bounded candidate repair for a fictional synthetic aerospace electrical-design JSON. Use only the supplied rule, finding, design, operation, and exact allowed path. Never invent a path, component, pin, requirement, or value. If the correction is not uniquely supported by the design, abstain. Return exactly one JSON object and no markdown. You do not modify the original design; a deterministic verifier decides whether the candidate is accepted."""
_RESPONSE_KEYS = {"operations", "rationale", "abstained", "abstention_reason"}


def _canonical_path(value: Any) -> str:
    path = str(value or "").strip().replace("['", ".").replace("']", "").replace('["', ".").replace('"]', "")
    path = re.sub(r"^(?:\$\.|design\.)", "", path)
    return re.sub(r"\.(\d+)(?=\.|$)", r"[\1]", path)


def _failed(reason: str, error: str, raw_count: int = 0, invalid_count: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = {"operations": (), "rationale": "", "abstained": True, "abstention_reason": reason}
    diagnostics = {"parse_success": False, "raw_operation_count": raw_count, "invalid_operation_count": invalid_count, "error": error}
    return parsed, diagnostics


def parse_repair_content(content: str, allowed_paths: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
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
        path = _canonical_path(item["path"])
        if item["op"] != "replace" or path not in allowed_paths or not isinstance(item["value"], str) or not item["value"]:
            invalid += 1
            continue
        operations.append(PatchOperation("replace", path, item["value"]))

    if invalid or len(operations) != 1:
        return _failed("invalid_patch_operation", "Exactly one valid operation on an allowed path is required", len(raw_operations), invalid or abs(1 - len(operations)))
    return (
        {"operations": tuple(operations), "rationale": payload["rationale"], "abstained": False, "abstention_reason": ""},
        {"parse_success": True, "raw_operation_count": len(raw_operations), "invalid_operation_count": 0, "error": ""},
    )


class LLMRepairAgent:
    def __init__(self, catalog: dict[str, Any], client: OllamaClient):
        self.rules, self.client = catalog["rules"], client
        self.rule_index = {rule["rule_id"]: rule for rule in self.rules}

    def _prompt(self, design: dict[str, Any], finding: dict[str, Any]) -> str:
        rule = self.rule_index[finding["rule_id"]]
        safe_finding = {key: finding.get(key) for key in ("rule_id", "entity_path", "entity_id", "message", "evidence")}
        sections = {
            "task": "Propose one bounded correction or abstain.",
            "finding": safe_finding,
            "rule_criteria": {key: rule[key] for key in ("rule_id", "title", "section", "severity", "description", "parameters")},
            "allowed_patch_paths": [finding["entity_path"]],
            "allowed_operations": ["replace"],
            "required_output": {
                "operations": [{"op": "replace", "path": "<exact allowed path>", "value": "<declared value>"}],
                "rationale": "short evidence-based reason", "abstained": False, "abstention_reason": "",
            },
            "design": design,
        }
        return json.dumps(sections, separators=(",", ":"), ensure_ascii=False)

    def repair(self, design: dict[str, Any], model: str, seed: int = 2027, max_tokens: int = 400) -> dict[str, Any]:
        original, working = copy.deepcopy(design), copy.deepcopy(design)
        initial_findings = validate_design(working, self.rules)
        attempts, accepted_count = [], 0

        for finding in initial_findings:
            repairability = classify_repairability(finding)
            base = {"rule_id": finding["rule_id"], "entity_path": finding["entity_path"], "repairability": repairability.value}

            if repairability not in {Repairability.AUTOMATIC, Repairability.CONSTRAINED}:
                attempts.append({
                    **base, "status": RepairStatus.ABSTAINED.value, "reason": "Repair requires human engineering judgment.",
                    "llm_call_performed": False, "proposal": None, "introduced_findings": [], "resolved_findings": [],
                    "diagnostics": {"parse_success": True, "raw_operation_count": 0, "invalid_operation_count": 0, "error": ""},
                    "prompt_sha256": "", "ollama_metadata": {}, "raw_content": "",
                })
                continue

            prompt = self._prompt(working, finding)
            response = self.client.chat(
                model, REPAIR_SYSTEM_PROMPT, prompt, seed=seed, max_tokens=max_tokens,
                response_schema=REPAIR_RESPONSE_SCHEMA,
            )
            parsed, diagnostics = parse_repair_content(response.content, {finding["entity_path"]})

            if parsed["abstained"] or not diagnostics["parse_success"]:
                status = RepairStatus.ABSTAINED if diagnostics["parse_success"] else RepairStatus.REJECTED
                attempts.append({
                    **base, "status": status.value,
                    "reason": parsed["abstention_reason"] or diagnostics["error"] or "No valid repair proposal.",
                    "llm_call_performed": True, "proposal": None, "introduced_findings": [], "resolved_findings": [],
                    "diagnostics": diagnostics, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "ollama_metadata": response.metadata, "raw_content": response.content,
                })
                continue

            proposal = RepairProposal(finding, parsed["operations"], parsed["rationale"])
            result = execute_repair(working, self.rules, proposal)
            proposal_record = {
                "operations": [{"op": item.op, "path": item.path, "value": item.value} for item in proposal.operations],
                "rationale": proposal.rationale,
            }
            attempts.append({
                **base, "status": result.status.value, "reason": result.reason, "llm_call_performed": True,
                "proposal": proposal_record, "introduced_findings": list(result.introduced_findings),
                "resolved_findings": list(result.resolved_findings), "diagnostics": diagnostics,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "ollama_metadata": response.metadata, "raw_content": response.content,
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
            "model": model, "design_id": design["design_id"], "status": status,
            "initial_findings": initial_findings, "attempts": attempts, "final_findings": final_findings,
            "repaired_design": working, "accepted_repair_count": accepted_count,
            "automatic_modification_performed": bool(accepted_count),
            "sandbox_modification_performed": bool(accepted_count),
            "production_modification_performed": False,
            "human_approval_required": bool(final_findings),
            "input_design_unchanged": design == original,
            "decision_source": "deterministic_validator",
            "proposal_source": "llm",
        }
