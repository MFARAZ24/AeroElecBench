from __future__ import annotations

import unittest

from aeroecad.agent import ReviewAgent
from aeroecad.catalog import load_catalog
from aeroecad.evaluation import evaluate
from aeroecad.generator import RULE_IDS, generate_benchmark


class PrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()
        cls.agent = ReviewAgent(cls.catalog)
        cls.scenarios = generate_benchmark(seed=7, cases_per_rule=1, clean_cases=1, mixed_cases=2)

    def test_clean_design_has_no_findings(self) -> None:
        scenario = next(item for item in self.scenarios if item["category"] == "clean")
        report = self.agent.review(scenario["design"], scenario["review_query"], "full")
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["status"], "PASS_WITHIN_ENCODED_SCOPE")

    def test_every_single_fault_is_detected_exactly(self) -> None:
        singles = [item for item in self.scenarios if item["category"] == "single_fault"]
        self.assertEqual({item["ground_truth"][0]["rule_id"] for item in singles}, set(RULE_IDS))
        for scenario in singles:
            report = self.agent.review(scenario["design"], scenario["review_query"], "full")
            expected = {(item["rule_id"], item["entity_path"]) for item in scenario["ground_truth"]}
            actual = {(item["rule_id"], item["entity_path"]) for item in report["findings"]}
            self.assertEqual(actual, expected)

    def test_retrieval_routes_single_fault_queries(self) -> None:
        for scenario in (item for item in self.scenarios if item["category"] == "single_fault"):
            report = self.agent.review(scenario["design"], scenario["review_query"], "retrieval_guided")
            self.assertEqual(report["selected_rule_ids"], [scenario["ground_truth"][0]["rule_id"]])

    def test_mixed_faults_preserve_evidence_and_human_control(self) -> None:
        scenario = next(item for item in self.scenarios if item["category"] == "mixed_fault")
        report = self.agent.review(scenario["design"], scenario["review_query"], "full")
        self.assertTrue(report["human_approval_required"])
        self.assertFalse(report["automatic_modification_performed"])
        self.assertTrue(all(item["evidence"] and item["rule_citation"] and item["entity_path"] for item in report["findings"]))

    def test_evaluation_is_exact_for_encoded_scope(self) -> None:
        metrics, _ = evaluate(self.scenarios, self.catalog, "retrieval_guided")
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["retrieval"]["top1_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
