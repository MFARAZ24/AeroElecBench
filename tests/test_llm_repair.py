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

    @staticmethod
    def payload(value: str) -> dict:
        return {
            "operations": [{"op": "replace", "path": "wires[0].target.pin_id", "value": value}],
            "rationale": "Candidate selected by applying the compatibility rule.",
            "abstained": False, "abstention_reason": "",
        }

    def test_direct_llm_repair_accepts_correct_patch_without_oracle_leakage(self) -> None:
        scenario = self.scenario("AE-R003")
        client = FakeRepairClient(self.payload("PWR"))
        report = LLMRepairAgent(self.catalog, client).repair(scenario["design"], "qwen2.5:7b", "llm_direct")
        self.assertEqual(report["status"], "repaired_in_sandbox")
        self.assertEqual(report["repaired_design"], scenario["repair_oracle"]["expected_design"])
        prompt = client.calls[0]["user"]
        for forbidden in ("repair_oracle", "expected_design", "ground_truth"):
            self.assertNotIn(forbidden, prompt)

    def test_grounded_mode_exposes_all_candidates_but_not_correct_answer(self) -> None:
        scenario = self.scenario("AE-R003")
        client = FakeRepairClient(self.payload("PWR"))
        report = LLMRepairAgent(self.catalog, client).repair(scenario["design"], "qwen2.5:7b", "tool_evidence_grounded")
        prompt_text = client.calls[0]["user"]
        prompt = json.loads(prompt_text)
        candidates = prompt["tool_evidence"]["target_declared_pins"]
        self.assertEqual({item["pin_id"] for item in candidates}, {"PWR", "RTN", "DATA_H", "DATA_L"})
        self.assertEqual(prompt["tool_evidence"]["peer_signal_class"], "POWER_28V")
        self.assertNotIn("deterministic_matches", prompt_text)
        self.assertEqual(report["status"], "repaired_in_sandbox")

    def test_wrong_declared_candidate_reaches_verifier_and_is_rolled_back(self) -> None:
        scenario = self.scenario("AE-R003")
        report = LLMRepairAgent(self.catalog, FakeRepairClient(self.payload("DATA_H"))).repair(
            scenario["design"], "qwen2.5:7b", "tool_evidence_grounded"
        )
        attempt = report["attempts"][0]
        self.assertTrue(attempt["diagnostics"]["parse_success"])
        self.assertEqual(attempt["status"], "rejected")
        self.assertTrue(any(item["rule_id"] == "AE-R004" for item in attempt["introduced_findings"]))
        self.assertEqual(report["repaired_design"], scenario["design"])

    def test_deterministic_mode_repairs_without_calling_llm(self) -> None:
        scenario = self.scenario("AE-R003")
        client = FakeRepairClient({})
        report = LLMRepairAgent(self.catalog, client).repair(scenario["design"], "qwen2.5:7b", "deterministic_auto")
        self.assertEqual(report["status"], "repaired_in_sandbox")
        self.assertEqual(report["repaired_design"], scenario["repair_oracle"]["expected_design"])
        self.assertEqual(client.calls, [])
        self.assertFalse(report["attempts"][0]["llm_call_performed"])

    def test_nonrepairable_finding_abstains_without_llm(self) -> None:
        scenario = self.scenario("AE-R004")
        for mode in ("llm_direct", "tool_evidence_grounded", "deterministic_auto"):
            client = FakeRepairClient({})
            report = LLMRepairAgent(self.catalog, client).repair(scenario["design"], "qwen2.5:7b", mode)
            self.assertEqual(report["attempts"][0]["status"], "abstained")
            self.assertEqual(client.calls, [])

    def test_out_of_scope_path_is_rejected(self) -> None:
        scenario = self.scenario("AE-R003")
        payload = {
            "operations": [{"op": "replace", "path": "components[0].id", "value": "PWR"}],
            "rationale": "Unsupported path.", "abstained": False, "abstention_reason": "",
        }
        report = LLMRepairAgent(self.catalog, FakeRepairClient(payload)).repair(
            scenario["design"], "qwen2.5:7b", "tool_evidence_grounded"
        )
        self.assertEqual(report["attempts"][0]["status"], "rejected")
        self.assertFalse(report["attempts"][0]["diagnostics"]["parse_success"])

    def test_clean_design_skips_every_repair_mode(self) -> None:
        scenario = next(item for item in self.scenarios if item["category"] == "clean")
        for mode in ("llm_direct", "tool_evidence_grounded", "deterministic_auto"):
            client = FakeRepairClient({})
            report = LLMRepairAgent(self.catalog, client).repair(scenario["design"], "qwen2.5:7b", mode)
            self.assertEqual(report["status"], "no_repair_required")
            self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
