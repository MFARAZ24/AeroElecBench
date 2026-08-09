from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aeroecad.catalog import load_catalog
from aeroecad.generator import generate_benchmark
from aeroecad.llm_evaluation import run_llm_experiment, select_scenarios
from aeroecad.llm_review import LLMReviewAgent, parse_llm_content
from aeroecad.ollama import OllamaResponse


class FakeOllamaClient:
    def __init__(self, payload: dict | str = "http://localhost:11434", timeout: float = 300.0):
        self.payload = payload if isinstance(payload, dict) else {"findings": [], "abstained": False, "abstention_reason": ""}

    def ensure_models(self, requested: list[str]) -> list[str]:
        return requested

    def chat(self, model: str, system: str, user: str, seed: int = 2027, max_tokens: int = 1200) -> OllamaResponse:
        return OllamaResponse(json.dumps(self.payload), {"prompt_eval_count": 10, "eval_count": 5})


class LLMPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()
        cls.scenarios = generate_benchmark(seed=2027, cases_per_rule=20, clean_cases=20, mixed_cases=50)

    def test_run_profiles_are_balanced_and_reproducible(self) -> None:
        self.assertEqual(len(select_scenarios(self.scenarios, "smoke")), 7)
        self.assertEqual(len(select_scenarios(self.scenarios, "pilot")), 40)
        self.assertEqual(len(select_scenarios(self.scenarios, "full")), 170)
        pilot = select_scenarios(self.scenarios, "pilot")
        for rule_id in ("AE-R001", "AE-R002", "AE-R003", "AE-R004", "AE-R005"):
            self.assertEqual(sum(item["category"] == "single_fault" and item["ground_truth"][0]["rule_id"] == rule_id for item in pilot), 5)

    def test_parser_accepts_fenced_json_and_canonicalizes_paths(self) -> None:
        content = '```json\n{"findings":[{"rule_id":"AE-R001","entity_path":"$.components.0.part_number"}],"abstained":false}\n```'
        report, diagnostics = parse_llm_content(content, {"AE-R001"})
        self.assertTrue(diagnostics["parse_success"])
        self.assertEqual(report["findings"][0]["entity_path"], "components[0].part_number")

    def test_parser_drops_unknown_rule_ids(self) -> None:
        report, diagnostics = parse_llm_content('{"findings":[{"rule_id":"MADE-UP","entity_path":"wires[0]"}]}', {"AE-R001"})
        self.assertEqual(report["findings"], [])
        self.assertEqual(diagnostics["invalid_finding_count"], 1)

    def test_llm_agent_preserves_human_control_and_diagnostics(self) -> None:
        scenario = next(item for item in self.scenarios if item["category"] == "clean")
        agent = LLMReviewAgent(self.catalog, FakeOllamaClient({"findings": [], "abstained": False, "abstention_reason": ""}))
        report = agent.review(scenario["design"], scenario["review_query"], "qwen2.5:7b", "llm_only")
        self.assertTrue(report["diagnostics"]["parse_success"])
        self.assertTrue(report["human_approval_required"])
        self.assertFalse(report["automatic_modification_performed"])

    def test_offline_experiment_writes_resumable_result_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch("aeroecad.llm_evaluation.OllamaClient", FakeOllamaClient):
            summary = run_llm_experiment(self.scenarios, self.catalog, ["fake:7b"], ["llm_only"], "smoke", directory)
            self.assertEqual(summary["scenario_count"], 7)
            self.assertTrue((Path(directory) / "ollama_responses.jsonl").exists())
            self.assertTrue((Path(directory) / "llm_benchmark_summary.json").exists())
            self.assertTrue((Path(directory) / "llm_comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
