from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from aeroecad.impact_benchmark import IMPACT_CASE_TYPES
from aeroecad.impact_heldout import BENCHMARK_ID, prepare_impact_heldout, run_impact_heldout


def test_v08_heldout_is_balanced_frozen_and_idempotent() -> None:
    with tempfile.TemporaryDirectory() as directory:
        benchmark = Path(directory) / "v08" / "impact.jsonl"
        scenarios, manifest = prepare_impact_heldout(benchmark)
        assert len(scenarios) == 30
        assert Counter(item["impact_case_type"] for item in scenarios) == {name: 5 for name in IMPACT_CASE_TYPES}
        assert {item["split"] for item in scenarios} == {"heldout"}
        assert manifest["benchmark_id"] == BENCHMARK_ID
        assert manifest["heldout"] is True and manifest["frozen_before_model_run"] is True
        assert manifest["posthoc_tuning_allowed"] is False and manifest["narrative_output"] is False
        repeated, repeated_manifest = prepare_impact_heldout(benchmark)
        assert repeated == scenarios and repeated_manifest == manifest


def test_v08_heldout_refuses_tampered_freeze() -> None:
    with tempfile.TemporaryDirectory() as directory:
        benchmark = Path(directory) / "impact.jsonl"
        prepare_impact_heldout(benchmark)
        benchmark.write_text(benchmark.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="does not match"):
            prepare_impact_heldout(benchmark)


def test_v08_graph_run_uses_heldout_provenance_and_progress(capsys: pytest.CaptureFixture[str]) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        benchmark, output = root / "benchmark" / "impact.jsonl", root / "results"
        prepare_impact_heldout(benchmark, cases_per_type=1)
        metrics = run_impact_heldout(modes=("graph_deterministic",), benchmark_path=benchmark, output_dir=output)
        assert metrics["scenario_count"] == 6
        assert metrics["modes"]["graph_deterministic"]["oracle_action_accuracy"] == 1.0
        manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
        assert manifest["heldout"] is True and manifest["development_only"] is False
        assert manifest["posthoc_tuning_allowed"] is False and manifest["narrative_output"] is False
        progress = capsys.readouterr().out
        assert "Qwen calls: 0/0 complete" in progress and "Qwen remaining=0" in progress
        assert not list(output.rglob("*.md"))
