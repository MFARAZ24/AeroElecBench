from __future__ import annotations

import copy
import unittest

from aeroecad.catalog import load_catalog
from aeroecad.generator import generate_benchmark
from aeroecad.repair import (
    PatchOperation,
    RepairProposal,
    RepairStatus,
    Repairability,
    classify_repairability,
    execute_repair,
)
from aeroecad.validator import validate_design


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()
        cls.rules = cls.catalog["rules"]
        cls.scenarios = generate_benchmark(seed=19, cases_per_rule=1, clean_cases=0, mixed_cases=0)

    def scenario(self, rule_id: str) -> dict:
        return next(item for item in self.scenarios if item["ground_truth"][0]["rule_id"] == rule_id)

    def test_constrained_pin_repair_is_accepted_without_mutating_input(self) -> None:
        scenario = self.scenario("AE-R003")
        original = copy.deepcopy(scenario["design"])
        finding = validate_design(original, self.rules)[0]
        source_pin = original["wires"][0]["source"]["pin_id"]
        proposal = RepairProposal(finding, (PatchOperation("replace", "wires[0].target.pin_id", source_pin),), "Match the valid source signal class.")
        result = execute_repair(original, self.rules, proposal)

        self.assertEqual(result.status, RepairStatus.ACCEPTED)
        self.assertEqual(result.candidate_findings, ())
        self.assertTrue(result.automatic_modification_performed)
        self.assertFalse(result.human_approval_required)
        self.assertEqual(original, scenario["design"])
        self.assertTrue(validate_design(original, self.rules))

    def test_regression_causes_transactional_rollback(self) -> None:
        scenario = self.scenario("AE-R003")
        original = copy.deepcopy(scenario["design"])
        finding = validate_design(original, self.rules)[0]
        proposal = RepairProposal(finding, (PatchOperation("replace", "wires[0].target.pin_id", "RTN"),), "Attempt a valid but incompatible pin.")
        result = execute_repair(original, self.rules, proposal)

        self.assertEqual(result.status, RepairStatus.REJECTED)
        self.assertEqual(result.design, original)
        self.assertFalse(result.automatic_modification_performed)
        self.assertTrue(any(item["rule_id"] == "AE-R004" for item in result.introduced_findings))

    def test_ambiguous_repair_abstains_before_patching(self) -> None:
        scenario = self.scenario("AE-R004")
        original = copy.deepcopy(scenario["design"])
        finding = validate_design(original, self.rules)[0]
        proposal = RepairProposal(finding, (PatchOperation("replace", "wires[1].target.pin_id", "RTN"),), "One plausible endpoint correction.")
        result = execute_repair(original, self.rules, proposal)

        self.assertEqual(result.status, RepairStatus.ABSTAINED)
        self.assertEqual(result.design, original)
        self.assertTrue(result.human_approval_required)
        self.assertFalse(result.automatic_modification_performed)

    def test_repairability_policy_is_conservative(self) -> None:
        expected = {
            "AE-R001": Repairability.INTENT_DEPENDENT,
            "AE-R002": Repairability.AMBIGUOUS,
            "AE-R003": Repairability.CONSTRAINED,
            "AE-R004": Repairability.AMBIGUOUS,
            "AE-R005": Repairability.INTENT_DEPENDENT,
        }
        for rule_id, classification in expected.items():
            path = "wires[0].target.pin_id" if rule_id == "AE-R003" else "synthetic.path"
            self.assertEqual(classify_repairability({"rule_id": rule_id, "entity_path": path}), classification)


if __name__ == "__main__":
    unittest.main()
