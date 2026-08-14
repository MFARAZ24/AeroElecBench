from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .validator import validate_design

_SEGMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?$")
_ALLOWED_PATHS = (
    re.compile(r"^components\[\d+\]\.(?:id|type|part_number|zone)$"),
    re.compile(r"^wires\[\d+\]\.(?:source|target)\.(?:component_id|pin_id)$"),
    re.compile(r"^wires\[\d+\]\.requirement_ids$"),
)


class Repairability(str, Enum):
    AUTOMATIC = "automatic"
    CONSTRAINED = "constrained"
    AMBIGUOUS = "ambiguous"
    INTENT_DEPENDENT = "intent_dependent"


class RepairStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ABSTAINED = "abstained"


@dataclass(frozen=True)
class PatchOperation:
    op: str
    path: str
    value: Any


@dataclass(frozen=True)
class RepairProposal:
    finding: dict[str, Any]
    operations: tuple[PatchOperation, ...]
    rationale: str = ""


@dataclass(frozen=True)
class RepairResult:
    status: RepairStatus
    repairability: Repairability
    design: dict[str, Any]
    baseline_findings: tuple[dict[str, Any], ...]
    candidate_findings: tuple[dict[str, Any], ...]
    introduced_findings: tuple[dict[str, Any], ...]
    resolved_findings: tuple[dict[str, Any], ...]
    automatic_modification_performed: bool
    human_approval_required: bool
    reason: str


def classify_repairability(finding: dict[str, Any]) -> Repairability:
    rule_id, path = finding.get("rule_id"), finding.get("entity_path", "")
    if rule_id == "AE-R003" and path.endswith(".pin_id"):
        return Repairability.CONSTRAINED
    if rule_id in {"AE-R002", "AE-R003", "AE-R004"}:
        return Repairability.AMBIGUOUS
    return Repairability.INTENT_DEPENDENT


def _finding_key(finding: dict[str, Any]) -> tuple[str, str]:
    return str(finding.get("rule_id", "")), str(finding.get("entity_path", ""))


def _path_tokens(path: str) -> list[str | int]:
    if not any(pattern.fullmatch(path) for pattern in _ALLOWED_PATHS):
        raise ValueError(f"Patch path is outside the repair allowlist: {path}")
    tokens: list[str | int] = []
    for segment in path.split("."):
        match = _SEGMENT.fullmatch(segment)
        if not match:
            raise ValueError(f"Invalid patch path segment: {segment}")
        tokens.append(match.group(1))
        if match.group(2) is not None:
            tokens.append(int(match.group(2)))
    return tokens


def _validate_value(operation: PatchOperation) -> None:
    if operation.path.endswith(".requirement_ids"):
        if not isinstance(operation.value, list) or not operation.value or not all(isinstance(item, str) and item for item in operation.value):
            raise ValueError("requirement_ids must be a non-empty list of strings")
    elif not isinstance(operation.value, str) or not operation.value:
        raise ValueError("Patched identifiers and attributes must be non-empty strings")


def _apply_operation(design: dict[str, Any], operation: PatchOperation) -> None:
    if operation.op not in {"add", "replace"}:
        raise ValueError(f"Unsupported patch operation: {operation.op}")
    _validate_value(operation)
    tokens = _path_tokens(operation.path)
    parent: Any = design
    for token in tokens[:-1]:
        if isinstance(token, int):
            if not isinstance(parent, list) or token >= len(parent):
                raise IndexError(f"Patch index is out of range: {operation.path}")
        elif not isinstance(parent, dict) or token not in parent:
            raise KeyError(f"Patch parent does not exist: {operation.path}")
        parent = parent[token]
    leaf = tokens[-1]
    if not isinstance(leaf, str) or not isinstance(parent, dict):
        raise ValueError(f"Patch must target an object attribute: {operation.path}")
    if operation.op == "add" and leaf in parent:
        raise ValueError(f"Add operation targets an existing attribute: {operation.path}")
    if operation.op == "replace" and leaf not in parent:
        raise ValueError(f"Replace operation targets a missing attribute: {operation.path}")
    parent[leaf] = copy.deepcopy(operation.value)


def _result(
    status: RepairStatus,
    repairability: Repairability,
    original: dict[str, Any],
    candidate: dict[str, Any],
    baseline: tuple[dict[str, Any], ...],
    candidate_findings: tuple[dict[str, Any], ...],
    introduced: tuple[dict[str, Any], ...],
    resolved: tuple[dict[str, Any], ...],
    reason: str,
) -> RepairResult:
    accepted = status == RepairStatus.ACCEPTED
    return RepairResult(
        status, repairability, candidate if accepted else copy.deepcopy(original), baseline, candidate_findings,
        introduced, resolved, accepted, not accepted, reason,
    )


def execute_repair(design: dict[str, Any], rules: list[dict[str, Any]], proposal: RepairProposal) -> RepairResult:
    baseline = tuple(validate_design(design, rules))
    baseline_keys = {_finding_key(item) for item in baseline}
    target_key = _finding_key(proposal.finding)
    repairability = classify_repairability(proposal.finding)
    original, candidate = copy.deepcopy(design), copy.deepcopy(design)

    if target_key not in baseline_keys:
        return _result(RepairStatus.REJECTED, repairability, original, candidate, baseline, baseline, (), (), "Proposal targets a finding that is not present.")
    if repairability not in {Repairability.AUTOMATIC, Repairability.CONSTRAINED}:
        return _result(RepairStatus.ABSTAINED, repairability, original, candidate, baseline, baseline, (), (), "Repair requires human engineering judgment.")
    if not proposal.operations:
        return _result(RepairStatus.REJECTED, repairability, original, candidate, baseline, baseline, (), (), "Proposal contains no patch operations.")

    try:
        for operation in proposal.operations:
            _apply_operation(candidate, operation)
    except (ValueError, TypeError, KeyError, IndexError) as error:
        return _result(RepairStatus.REJECTED, repairability, original, candidate, baseline, baseline, (), (), f"Patch validation failed: {error}")

    candidate_findings = tuple(validate_design(candidate, rules))
    candidate_keys = {_finding_key(item) for item in candidate_findings}
    introduced = tuple(item for item in candidate_findings if _finding_key(item) not in baseline_keys)
    resolved = tuple(item for item in baseline if _finding_key(item) not in candidate_keys)

    if target_key in candidate_keys:
        return _result(RepairStatus.REJECTED, repairability, original, candidate, baseline, candidate_findings, introduced, resolved, "Target finding remains after patching; transaction rolled back.")
    if introduced:
        return _result(RepairStatus.REJECTED, repairability, original, candidate, baseline, candidate_findings, introduced, resolved, "Patch introduced new violations; transaction rolled back.")
    return _result(RepairStatus.ACCEPTED, repairability, original, candidate, baseline, candidate_findings, introduced, resolved, "Target finding resolved without validator regressions.")
