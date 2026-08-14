from __future__ import annotations

import copy
import unittest

from aeroecad.catalog import load_catalog
from aeroecad.generator import generate_benchmark
from aeroecad.repair import PatchOperation, _apply_operation, classify_repairability
from aeroecad.validator import validate_design


class RepairOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = load_catalog()["rules"]
        cls.scenarios = generate_benchmark(seed=31, cases_per_rule=1, clean_cases=1, mixed_cases=3)

    def test_oracle_is_complete_and_separate_from_agent_design(self) -> None:
        for scenario in self.scenarios:
            oracle = scenario["repair_oracle"]
            self.assertEqual(oracle["version"], "0.1")
            self.assertNotIn("repair_oracle", scenario["design"])
            self.assertEqual(
                {item["rule_id"] for item in oracle["repairs"]},
                {item["rule_id"] for item in scenario["ground_truth"]},
            )
            if scenario["category"] == "clean":
                self.assertEqual(scenario["design"], oracle["expected_design"])
            else:
                self.assertNotEqual(scenario["design"], oracle["expected_design"])

    def test_oracle_patches_reconstruct_exact_clean_design(self) -> None:
        for scenario in self.scenarios:
            candidate = copy.deepcopy(scenario["design"])
            oracle = scenario["repair_oracle"]
            truth_by_rule = {item["rule_id"]: item for item in scenario["ground_truth"]}

            for repair in oracle["repairs"]:
                expected_class = classify_repairability(truth_by_rule[repair["rule_id"]]).value
                self.assertEqual(repair["repairability"], expected_class)
                for operation in repair["operations"]:
                    _apply_operation(candidate, PatchOperation(**operation))

            self.assertEqual(candidate, oracle["expected_design"])
            self.assertEqual(validate_design(oracle["expected_design"], self.rules), [])


if __name__ == "__main__":
    unittest.main()
