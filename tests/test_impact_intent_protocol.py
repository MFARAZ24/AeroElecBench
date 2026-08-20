from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aeroecad.impact_intent import prepare_intent_development
from aeroecad.impact_intent_protocol import (
    FORBIDDEN_INPUT_KEYS,
    INPUT_FILE,
    ORACLE_FILE,
    PREDICTION_FILE,
    PREDICTION_MANIFEST,
    prepare_oracle_separated_package,
    run_oracle_free_predictions,
    score_frozen_predictions,
)
from aeroecad.ollama import OllamaResponse


class PerfectIntentClient:
    def __init__(self) -> None:
        self.calls = 0

    def ensure_models(self, requested: list[str]) -> list[str]:
        return requested

    def chat(self, model: str, system: str, user: str, **_: object) -> OllamaResponse:
        self.calls += 1
        request = json.loads(user)["engineering_change_request"]
        if "authorized replacement" in request:
            selected = ["CAND-01"]
        elif "conductor resized" in request:
            selected = ["CAND-02"]
        elif "coordinated hardware" in request:
            selected = ["CAND-01", "CAND-03"]
        elif "target-pin reassignment" in request:
            selected = ["CAND-04"]
        else:
            selected = []
        return OllamaResponse(json.dumps({"action": "report" if selected else "abstain", "selected_candidate_ids": selected, "rationale": "Oracle-free grounded selection."}), {"eval_count": 12, "done_reason": "stop"})


def test_prediction_succeeds_without_oracle_file_and_scoring_is_separate() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        benchmark, package, predictions, scores = root / "benchmark" / "intent.jsonl", root / "package", root / "predictions", root / "scores"
        prepare_intent_development(benchmark, cases_per_type=1)
        package_manifest = prepare_oracle_separated_package(benchmark, package)
        inputs = [json.loads(line) for line in (package / INPUT_FILE).read_text(encoding="utf-8").splitlines()]
        serialized_inputs = json.dumps(inputs)
        assert package_manifest["oracle_fields_present_in_model_input"] is False
        assert all(set(row) == {"scenario_id", "before_design", "after_design", "engineering_change_request", "change_inventory"} for row in inputs)
        assert all(row["scenario_id"] == f"INTENT-CASE-{index:04d}" for index, row in enumerate(inputs, start=1))
        assert all(set(row["engineering_change_request"]) == {"text", "max_depth"} for row in inputs)
        assert not any(f'"{key}"' in serialized_inputs for key in FORBIDDEN_INPUT_KEYS)
        assert not any(case_type in serialized_inputs for case_type in ("single_explicit", "multi_intent", "ambiguous_request", "unmatched_request"))
        oracle_bytes = (package / ORACLE_FILE).read_bytes()
        (package / ORACLE_FILE).unlink()
        client = PerfectIntentClient()
        prediction_manifest = run_oracle_free_predictions(package_dir=package, output_dir=predictions, client=client)
        assert client.calls == 6 and prediction_manifest["oracle_file_read"] is False and prediction_manifest["metrics_generated"] is False
        assert set(path.name for path in predictions.iterdir()) == {PREDICTION_FILE, PREDICTION_MANIFEST}
        prediction_text = (predictions / PREDICTION_FILE).read_text(encoding="utf-8")
        assert "expected_action" not in prediction_text and "intent_case_type" not in prediction_text
        with pytest.raises(FileNotFoundError):
            score_frozen_predictions(package, predictions, scores)
        (package / ORACLE_FILE).write_bytes(oracle_bytes)
        metrics = score_frozen_predictions(package, predictions, scores)
        assert metrics["modes"]["qwen_intent_graph"]["candidate_f1"] == 1.0
        score_manifest = json.loads((scores / "evaluation_manifest.json").read_text(encoding="utf-8"))
        assert score_manifest["oracle_loaded_only_by_scoring_command"] is True
        assert score_manifest["prediction_file_unchanged_during_scoring"] is True
        assert score_manifest["oracle_correction_performed"] is False


def test_oracle_free_input_and_frozen_prediction_tampering_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        benchmark, package, predictions = root / "benchmark" / "intent.jsonl", root / "package", root / "predictions"
        prepare_intent_development(benchmark, cases_per_type=1)
        prepare_oracle_separated_package(benchmark, package)
        input_path = package / INPUT_FILE
        input_path.write_text(input_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="hash mismatch"):
            run_oracle_free_predictions(package_dir=package, output_dir=predictions, client=PerfectIntentClient())
        prepare_oracle_separated_package(benchmark, package)
        run_oracle_free_predictions(package_dir=package, output_dir=predictions, client=PerfectIntentClient())
        record_path = predictions / PREDICTION_FILE
        record_path.write_text(record_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="prediction identity or hash mismatch"):
            score_frozen_predictions(package, predictions, root / "scores")
