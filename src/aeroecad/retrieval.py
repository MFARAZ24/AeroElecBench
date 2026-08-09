from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class RuleRetriever:
    def __init__(self, rules: list[dict[str, Any]]):
        self.rules = rules
        documents = [" ".join([rule["title"], rule["description"], *rule.get("keywords", [])]) for rule in rules]
        self.term_counts = [Counter(_tokens(document)) for document in documents]
        document_frequency = Counter(token for counts in self.term_counts for token in counts)
        self.idf = {token: math.log((1 + len(rules)) / (1 + frequency)) + 1 for token, frequency in document_frequency.items()}
        self.vectors = [self._vector(counts) for counts in self.term_counts]

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        total = sum(counts.values()) or 1
        return {token: count / total * self.idf.get(token, 1.0) for token, count in counts.items()}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        numerator = sum(value * right.get(token, 0.0) for token, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query_vector = self._vector(Counter(_tokens(query)))
        ranked = sorted(
            ({"rule_id": rule["rule_id"], "score": self._cosine(query_vector, vector)} for rule, vector in zip(self.rules, self.vectors, strict=True)),
            key=lambda item: (-item["score"], item["rule_id"]),
        )
        return ranked[: max(1, min(top_k, len(ranked)))]


def is_full_audit(query: str) -> bool:
    normalized = " ".join(_tokens(query))
    return ("complete" in normalized or "full" in normalized) and ("review" in normalized or "audit" in normalized)
