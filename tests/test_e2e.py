from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from aeroecad.catalog import load_catalog
from aeroecad.e2e import prepare_repair_prototype
from aeroecad.ollama import OllamaResponse
from aeroecad.repair_evaluation import run_repair_experiment


class CalibratedRepairClient:
    calls = 0

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 300.0):
        pass

    def ensure_models(self, requested: list[str]) -> list[str]:
        return requested

    def chat(self, model: str, system: str, user: str, seed: int = 2027, max_tokens: int = 300, response_schema: dict | None = None) -> OllamaResponse:
        type(self).calls += 1
        prompt = json.loads(user)
        path = prompt["allowed_patch_paths"][0]
        evidence = prompt["tool_evidence"]
        required_class = evidence["peer_signal_class"]
        candidates = [item["pin_id"] for item in evidence["target_declared_pins"] if item["signal_class"] == required_class]
        primary = [candidate for candidate in candidates if candidate.endswith("_PRIMARY")]
        selected = candidates[0] if len(candidates) == 1 else primary[0] if len(primary) == 1 else None
        payload = {
            "operations": [{"op": "replace", "path": path, "value": selected}] if selected else [],
            "rationale": "Selected the uniquely supported candidate." if selected else "Evidence does not support one candidate.",
            "abstained": selected is None,
            "abstention_reason": "No uniquely supported candidate." if selected is None else "",
        }
        return OllamaResponse(json.dumps(payload), {"prompt_eval_count": 20, "eval_count": 10})


def test_prepare_prototype_writes_balanced_validated_dataset() -> None:
    with tempfile.TemporaryDirectory() as directory:
        benchmark = Path(directory) / "repair.jsonl"
        scenarios, manifest = prepare_repair_prototype(benchmark)
        assert len(scenarios) == 25
        assert manifest["scenario_count"] == 25
        assert manifest["oracle_validation_rate"] == 1.0
        assert set(manifest["case_type_counts"].values()) == {5}
        assert benchmark.exists() and benchmark.with_name("manifest.json").exists()
        assert all(not item["provenance"]["certification_evidence"] for item in scenarios)


def test_dedicated_repair_benchmark_runs_end_to_end() -> None:
    CalibratedRepairClient.calls = 0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scenarios, _ = prepare_repair_prototype(root / "repair.jsonl")
        with patch("aeroecad.repair_evaluation.OllamaClient", CalibratedRepairClient):
            summary = run_repair_experiment(
                scenarios, load_catalog(), ["fake:7b"], "full", root / "results",
                repair_mode="tool_evidence_grounded",
            )
        metrics = summary["models"]["fake:7b"]
        assert metrics["eligible_repair_count"] == 10
        assert metrics["verified_repair_success_rate"] == 1.0
        assert metrics["eligible_exact_restoration_rate"] == 1.0
        assert metrics["correct_abstention_rate"] == 1.0
        assert metrics["clean_preservation_rate"] == 1.0
        assert metrics["oracle_action_accuracy"] == 1.0
        assert metrics["unsafe_accepted_abstention_count"] == 0
        assert metrics["production_modification_count"] == 0
        assert CalibratedRepairClient.calls == 20
