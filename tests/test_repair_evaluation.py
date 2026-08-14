from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aeroecad.catalog import load_catalog
from aeroecad.generator import generate_benchmark
from aeroecad.ollama import OllamaResponse
from aeroecad.repair_evaluation import run_repair_experiment


class FakeRepairClient:
    total_calls = 0

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 300.0):
        pass

    def ensure_models(self, requested: list[str]) -> list[str]:
        return requested

    def chat(self, model: str, system: str, user: str, seed: int = 2027, max_tokens: int = 400, response_schema: dict | None = None) -> OllamaResponse:
        type(self).total_calls += 1
        prompt = json.loads(user)
        path = prompt["allowed_patch_paths"][0]
        match = re.fullmatch(r"wires\[(\d+)\]\.(source|target)\.pin_id", path)
        if not match:
            raise AssertionError(f"Unexpected eligible repair path: {path}")
        wire = prompt["design"]["wires"][int(match.group(1))]
        other_endpoint = "target" if match.group(2) == "source" else "source"
        value = wire[other_endpoint]["pin_id"]
        payload = {
            "operations": [{"op": "replace", "path": path, "value": value}],
            "rationale": "Use the declared pin matching the opposite endpoint.",
            "abstained": False, "abstention_reason": "",
        }
        return OllamaResponse(json.dumps(payload), {"prompt_eval_count": 20, "eval_count": 10})


class RepairEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()
        cls.scenarios = generate_benchmark(seed=2027, cases_per_rule=20, clean_cases=20, mixed_cases=50)

    def test_mocked_smoke_experiment_writes_metrics_and_artifacts(self) -> None:
        FakeRepairClient.total_calls = 0
        with tempfile.TemporaryDirectory() as directory, patch("aeroecad.repair_evaluation.OllamaClient", FakeRepairClient):
            summary = run_repair_experiment(self.scenarios, self.catalog, ["fake:7b"], "smoke", directory)
            metrics = summary["models"]["fake:7b"]

            self.assertEqual(summary["scenario_count"], 7)
            self.assertEqual(metrics["verified_repair_success_rate"], 1.0)
            self.assertEqual(metrics["eligible_exact_restoration_rate"], 1.0)
            self.assertEqual(metrics["correct_abstention_rate"], 1.0)
            self.assertEqual(metrics["clean_preservation_rate"], 1.0)
            self.assertEqual(metrics["production_modification_count"], 0)
            self.assertGreater(FakeRepairClient.total_calls, 0)

            output = Path(directory)
            for name in (
                "repair_responses.jsonl", "repair_benchmark_summary.json",
                "repair_manifest.json", "repair_comparison.csv", "repair_results.md",
            ):
                self.assertTrue((output / name).exists())

    def test_repair_experiment_resumes_without_duplicate_calls(self) -> None:
        FakeRepairClient.total_calls = 0
        with tempfile.TemporaryDirectory() as directory, patch("aeroecad.repair_evaluation.OllamaClient", FakeRepairClient):
            run_repair_experiment(self.scenarios, self.catalog, ["fake:7b"], "smoke", directory)
            first_call_count = FakeRepairClient.total_calls
            run_repair_experiment(self.scenarios, self.catalog, ["fake:7b"], "smoke", directory)

            self.assertEqual(FakeRepairClient.total_calls, first_call_count)
            rows = (Path(directory) / "repair_responses.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 7)


if __name__ == "__main__":
    unittest.main()
