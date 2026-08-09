from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .ollama import OllamaClient
from .retrieval import RuleRetriever, is_full_audit
from .validator import validate_design

LLM_MODES = ("llm_only", "retrieval_grounded", "hybrid_explainer")
SYSTEM_PROMPT = """You are a cautious engineer-facing reviewer of fictional synthetic aerospace electrical-design JSON. Never modify the design. Report only violations supported by the provided design. Return exactly one JSON object and no markdown. Every finding must use an exact rule_id and an exact entity_path from the design. If no supported violation exists, return an empty findings list. A human engineer must approve every conclusion."""
OUTPUT_SCHEMA = {
    "findings": [{
        "rule_id": "<provided rule id>", "entity_path": "<exact design path>", "entity_id": "<affected entity id>",
        "explanation": "short evidence-based explanation", "evidence": {"observed": "value or missing", "expected": "expected condition"},
        "rule_citation": {"catalog_id": "<provided catalog id>", "section": "<provided section>", "rule_id": "<same rule id>"},
    }],
    "abstained": False, "abstention_reason": "",
}
PATH_CONVENTIONS = [
    "Use zero-based array indices and canonical paths such as components[0].part_number or wires[2].requirement_ids.",
    "For a missing field, name the canonical path where that field should exist.",
    "For duplicate identifiers, report each repeated occurrence after the first occurrence.",
    "For an invalid endpoint reference, name its exact component_id or pin_id field; for a connection-level incompatibility, name the wire path.",
]


def _rule_metadata(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"rule_id": rule["rule_id"], "title": rule["title"], "section": rule["section"], "catalog_id": "AEROECAD-SYNTH-RULES-0.1"} for rule in rules]


def _grounding_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: rule[key] for key in ("rule_id", "title", "section", "severity", "description", "parameters")} for rule in rules]


def _canonical_path(value: Any) -> str:
    path = str(value or "").strip().replace("['", ".").replace("']", "").replace('["', ".").replace('"]', "")
    path = re.sub(r"^(?:\$\.|design\.)", "", path)
    path = re.sub(r"\.(\d+)(?=\.|$)", r"[\1]", path)
    return path


def parse_llm_content(content: str, allowed_rule_ids: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {"findings": [], "abstained": True, "abstention_reason": "unparseable_output"}, {"parse_success": False, "raw_finding_count": 0, "invalid_finding_count": 0, "error": "No JSON object found"}
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            return {"findings": [], "abstained": True, "abstention_reason": "unparseable_output"}, {"parse_success": False, "raw_finding_count": 0, "invalid_finding_count": 0, "error": str(exc)}
    if not isinstance(payload, dict):
        return {"findings": [], "abstained": True, "abstention_reason": "invalid_output_shape"}, {"parse_success": False, "raw_finding_count": 0, "invalid_finding_count": 0, "error": "Top-level JSON value must be an object"}
    raw_findings = payload.get("findings", [])
    if not isinstance(raw_findings, list):
        raw_findings = []
    findings, invalid = [], 0
    for item in raw_findings:
        if not isinstance(item, dict) or item.get("rule_id") not in allowed_rule_ids or not _canonical_path(item.get("entity_path")):
            invalid += 1
            continue
        finding = dict(item)
        finding["entity_path"] = _canonical_path(item["entity_path"])
        finding["entity_id"] = str(item.get("entity_id", ""))
        finding["explanation"] = str(item.get("explanation", ""))
        finding["evidence"] = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        finding["rule_citation"] = item.get("rule_citation") if isinstance(item.get("rule_citation"), dict) else {}
        findings.append(finding)
    report = {"findings": findings, "abstained": bool(payload.get("abstained", False)), "abstention_reason": str(payload.get("abstention_reason", ""))}
    diagnostics = {"parse_success": True, "raw_finding_count": len(raw_findings), "invalid_finding_count": invalid, "error": ""}
    return report, diagnostics


class LLMReviewAgent:
    def __init__(self, catalog: dict[str, Any], client: OllamaClient, retrieval_top_k: int = 3):
        self.catalog, self.rules, self.client = catalog, catalog["rules"], client
        self.rule_index = {rule["rule_id"]: rule for rule in self.rules}
        self.retriever, self.retrieval_top_k = RuleRetriever(self.rules), retrieval_top_k

    def _selected_rules(self, query: str, mode: str) -> list[dict[str, Any]]:
        if mode == "llm_only" or is_full_audit(query):
            return self.rules
        ranking = self.retriever.retrieve(query, top_k=self.retrieval_top_k)
        return [self.rule_index[item["rule_id"]] for item in ranking]

    def _prompt(self, design: dict[str, Any], query: str, mode: str) -> tuple[str, list[str]]:
        selected = self._selected_rules(query, mode)
        sections: dict[str, Any] = {"review_request": query, "required_output_schema": OUTPUT_SCHEMA, "schema_notice": "Angle-bracket values are placeholders; never copy them into a finding.", "entity_path_conventions": PATH_CONVENTIONS}
        if mode == "llm_only":
            sections["available_rule_labels_only"] = _rule_metadata(self.rules)
            sections["instruction"] = "Use general reasoning from the design and the rule labels. No rule criteria are supplied in this mode."
        elif mode == "retrieval_grounded":
            sections["retrieved_rule_criteria"] = _grounding_rules(selected)
            sections["instruction"] = "Apply every retrieved rule criterion exactly and cite its supplied metadata."
        else:
            deterministic = validate_design(design, self.rules)
            sections["deterministic_candidate_findings"] = deterministic
            relevant_ids = {item["rule_id"] for item in deterministic}
            relevant = [rule for rule in self.rules if rule["rule_id"] in relevant_ids] or selected
            sections["rule_criteria"] = _grounding_rules(relevant)
            sections["instruction"] = "Check the deterministic candidates against the design, retain only supported candidates, and add a concise explanation."
        sections["design"] = design
        return json.dumps(sections, separators=(",", ":"), ensure_ascii=False), [rule["rule_id"] for rule in selected]

    def review(self, design: dict[str, Any], query: str, model: str, mode: str, seed: int = 2027, max_tokens: int = 1200) -> dict[str, Any]:
        if mode not in LLM_MODES:
            raise ValueError(f"mode must be one of: {', '.join(LLM_MODES)}")
        prompt, selected_rule_ids = self._prompt(design, query, mode)
        response = self.client.chat(model, SYSTEM_PROMPT, prompt, seed=seed, max_tokens=max_tokens)
        parsed, diagnostics = parse_llm_content(response.content, set(self.rule_index))
        return {
            "model": model, "mode": mode, "selected_rule_ids": selected_rule_ids, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "findings": parsed["findings"], "abstained": parsed["abstained"], "abstention_reason": parsed["abstention_reason"],
            "diagnostics": diagnostics, "ollama_metadata": response.metadata, "raw_content": response.content,
            "human_approval_required": True, "automatic_modification_performed": False,
        }
