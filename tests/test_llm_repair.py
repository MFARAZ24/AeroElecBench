from __future__ import annotations

import json
import unittest

from aeroecad.catalog import load_catalog
from aeroecad.generator import generate_benchmark
from aeroecad.llm_repair import LLMRepairAgent
from aeroecad.ollama import OllamaResponse


class FakeRepairClient:
    def __init__(self, payload: dict | str):
        self.payload = payload
        self.calls = []

    def chat(self, model: str, system: str, user: str, seed: int = 2027, max_tokens: int = 400, response_schema: dict | None = None) -> OllamaResponse:
        self.calls.append({"model": model, "system": system, "user": user, "schema": response_schema})
        content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return OllamaResponse(content, {"prompt_eval_count": 20, "eval_count": 10})


class LLMRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()
        cls.scenarios = generate_benchmark(seed=43, cases_per_rule=1, clean_cases=1, mixed_cases=0)

    def scenario(self, rule_id: str) -> dict:
        return next(item for item in self.scenarios if item["ground_truth"] and item["ground_truth"][0]["rule_id"] == rule_id)

    def test_correct_patch_is_accepted_and_oracle_never_enters_prompt(self) -> None:
        scenario = self.scenario("AE-R003")
        oracle_operation = scenario["repair_oracle"]["repairs"][0]["operations"][0]
        payload = {"operations": [oracle_operation], "rationale": "The declared matching pin restores endpoint validity.", "abstained": False, "abstention_reason": ""}
        client = FakeRepairClient(payload)
        report = LLMRepairAgent(self.catalog, client).repair(scenario["design"], "qwen2.5:7b")

        self.assertEqual(report["status"], "repaired_in_sandbox")
        self.assertEqual(report["repaired_design"], scenario["repair_oracle"]["expected_design"])
        self.assertTrue(report["automatic_modification_performed"])
        self.assertFalse(report["production_modification_performed"])
        self.assertTrue(report["input_design_unchanged"])
        self.assertEqual(len(client.calls), 1)

        prompt_text = client.calls[0]["user"]
        prompt = json.loads(prompt_text)
        self.assertEqual(set(prompt), {"task", "finding", "rule_criteria", "allowed_patch_paths", "allowed_operations", "required_output", "design"})
        for forbidden in ("repair_oracle", "expected_design", "ground_truth"):
            self.assertNotIn(forbidden, prompt_text)

    def test_ambiguous_finding_abstains_without_calling_llm(self) -> None:
        scenario = self.scenario("AE-R004")
        client = FakeRepairClient({})
        report = LLMRepairAgent(self.catalog, client).repair(scenario["design"], "qwen2.5:7b")

        self.assertEqual(report["status"], "review_required")
        self.assertEqual(report["attempts"][0]["status"], "abstained")
        self.assertEqual(client.calls, [])
        self.assertTrue(report["human_approval_required"])
        self.assertFalse(report["automatic_modification_performed"])

    def test_regression_is_rejected_and_rolled_back(self) -> None:
        scenario = self.scenario("AE-R003")
        payload = {
            "operations": [{"op": "replace", "path": "wires[0].target.pin_id", "value": "RTN"}],
            "rationale": "Use a declared but electrically incompatible pin.",
            "abstained": False, "abstention_reason": "",
        }
        report = LLMRepairAgent(self.catalog, FakeRepairClient(payload)).repair(scenario["design"], "qwen2.5:7b")

        self.assertEqual(report["attempts"][0]["status"], "rejected")
        self.assertEqual(report["repaired_design"], scenario["design"])
        self.assertTrue(any(item["rule_id"] == "AE-R004" for item in report["attempts"][0]["introduced_findings"]))
        self.assertFalse(report["automatic_modification_performed"])

    def test_out_of_scope_path_is_rejected_before_application(self) -> None:
        scenario = self.scenario("AE-R003")
        payload = {
            "operations": [{"op": "replace", "path": "components[0].id", "value": "UNSAFE-ID"}],
            "rationale": "Unsupported path.",
            "abstained": False, "abstention_reason": "",
        }
        report = LLMRepairAgent(self.catalog, FakeRepairClient(payload)).repair(scenario["design"], "qwen2.5:7b")

        self.assertEqual(report["attempts"][0]["status"], "rejected")
        self.assertEqual(report["repaired_design"], scenario["design"])
        self.assertFalse(report["attempts"][0]["diagnostics"]["parse_success"])

    def test_clean_design_skips_llm_and_requires_no_repair(self) -> None:
        scenario = next(item for item in self.scenarios if item["category"] == "clean")
        client = FakeRepairClient({})
        report = LLMRepairAgent(self.catalog, client).repair(scenario["design"], "qwen2.5:7b")

        self.assertEqual(report["status"], "no_repair_required")
        self.assertEqual(report["initial_findings"], [])
        self.assertEqual(report["final_findings"], [])
        self.assertEqual(client.calls, [])
        self.assertFalse(report["automatic_modification_performed"])


if __name__ == "__main__":
    unittest.main()
