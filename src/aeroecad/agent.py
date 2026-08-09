from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .retrieval import RuleRetriever, is_full_audit
from .validator import validate_design


class ReviewAgent:
    def __init__(self, catalog: dict[str, Any]):
        self.catalog = catalog
        self.rules = catalog["rules"]
        self.retriever = RuleRetriever(self.rules)

    def review(self, design: dict[str, Any], query: str, mode: str = "full") -> dict[str, Any]:
        if mode not in {"full", "retrieval_guided"}:
            raise ValueError("mode must be 'full' or 'retrieval_guided'")
        selected = [rule["rule_id"] for rule in self.rules]
        ranking = []
        if mode == "retrieval_guided":
            ranking = self.retriever.retrieve(query, top_k=len(self.rules))
            if not is_full_audit(query):
                selected = [ranking[0]["rule_id"]]
        findings = validate_design(design, self.rules, set(selected))
        return {
            "report_id": f"REPORT-{design['design_id']}", "design_id": design["design_id"],
            "generated_at": datetime.now(timezone.utc).isoformat(), "mode": mode, "query": query,
            "status": "REVIEW_REQUIRED" if findings else "PASS_WITHIN_ENCODED_SCOPE",
            "summary": {"finding_count": len(findings), "critical": sum(item["severity"] == "critical" for item in findings), "major": sum(item["severity"] == "major" for item in findings)},
            "selected_rule_ids": selected, "retrieval_ranking": ranking, "findings": findings,
            "human_approval_required": True, "automatic_modification_performed": False,
            "scope_notice": "Result applies only to the fictional encoded rule catalog and requires engineer review.",
        }
