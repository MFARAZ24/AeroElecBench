from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from aeroecad.heldout import prepare_heldout_benchmark, run_heldout_evaluation


class NoCallClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 300.0):
        pass

    def ensure_models(self, requested: list[str]) -> list[str]:
        return requested

    def chat(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("deterministic_auto must not call the model")


def test_prepare_v06_is_all_heldout_and_hash_locked() -> None:
    with tempfile.TemporaryDirectory() as directory:
        benchmark = Path(directory) / "repair_heldout_25.jsonl"
        scenarios, manifest = prepare_heldout_benchmark(benchmark)
        assert len(scenarios) == 25
        assert manifest["benchmark_id"] == "AEROELECBENCH-REPAIR-HOLDOUT-0.6"
        assert manifest["split_counts"] == {"heldout": 25}
        assert manifest["frozen_before_model_run"] is True
        assert manifest["posthoc_tuning_allowed"] is False
        assert manifest["narrative_output"] is False
        assert all(item["scenario_id"].startswith("v06-heldout-") for item in scenarios)


def test_v06_writes_metrics_and_raw_rows_without_narrative_reports() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        benchmark, output = root / "benchmark" / "repair.jsonl", root / "results"
        prepare_heldout_benchmark(benchmark)
        with patch("aeroecad.repair_evaluation.OllamaClient", NoCallClient):
            metrics = run_heldout_evaluation(
                model="fake:7b", modes=("deterministic_auto",),
                benchmark_path=benchmark, output_dir=output,
            )
        assert metrics["scenario_count"] == 25
        assert metrics["split_counts"] == {"heldout": 25}
        assert (output / "evaluation_metrics.json").exists()
        assert (output / "evaluation_table.csv").exists()
        assert (output / "evaluation_manifest.json").exists()
        assert (output / "deterministic_auto" / "repair_responses.jsonl").exists()
        assert not list(output.rglob("*.md"))
        assert not (output / "deterministic_auto" / "repair_benchmark_summary.json").exists()
        saved = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
        assert saved["narrative_output"] is False
        assert saved["metric_generation"] == "deterministic"
