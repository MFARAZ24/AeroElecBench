from __future__ import annotations

import copy
import unittest
from collections import Counter

from aeroecad.catalog import load_catalog
from aeroecad.llm_repair import _pin_evidence
from aeroecad.repair import PatchOperation, _apply_operation
from aeroecad.repair_benchmark import REPAIR_CASE_TYPES, generate_repair_benchmark
from aeroecad.validator import validate_design


class RepairBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = load_catalog()["rules"]
        cls.scenarios = generate_repair_benchmark(seed=4107, cases_per_type=5)

    def test_balanced_case_types_and_frozen_splits(self) -> None:
        self.assertEqual(len(self.scenarios), 25)
        self.assertEqual(Counter(item["repair_case_type"] for item in self.scenarios), {name: 5 for name in REPAIR_CASE_TYPES})
        self.assertEqual(Counter(item["split"] for item in self.scenarios), {"development": 10, "heldout": 15})
        self.assertEqual(len({item["scenario_id"] for item in self.scenarios}), 25)

    def test_clean_and_faulty_designs_match_ground_truth(self) -> None:
        for scenario in self.scenarios:
            findings = validate_design(scenario["design"], self.rules)
            expected = {(item["rule_id"], item["entity_path"]) for item in scenario["ground_truth"]}
            actual = {(item["rule_id"], item["entity_path"]) for item in findings}
            self.assertEqual(actual, expected)
            self.assertNotIn("repair_oracle", scenario["design"])

    def test_expected_repairs_reconstruct_clean_design(self) -> None:
        repairable = {"automatic", "constrained"}
        for scenario in self.scenarios:
            if scenario["repair_case_type"] not in repairable:
                continue
            oracle = scenario["repair_oracle"]
            candidate = copy.deepcopy(scenario["design"])
            _apply_operation(candidate, PatchOperation(**oracle["expected_operation"]))
            self.assertEqual(candidate, oracle["expected_design"])
            self.assertEqual(validate_design(candidate, self.rules), [])

    def test_candidate_structure_matches_case_type(self) -> None:
        expected_matches = {"automatic": 1, "constrained": 2, "ambiguous": 2, "insufficient": 0}
        for scenario in self.scenarios:
            case_type = scenario["repair_case_type"]
            if case_type == "clean":
                continue
            finding = validate_design(scenario["design"], self.rules)[0]
            evidence = _pin_evidence(scenario["design"], finding)
            self.assertEqual(len(evidence["deterministic_matches"]), expected_matches[case_type])


if __name__ == "__main__":
    unittest.main()
