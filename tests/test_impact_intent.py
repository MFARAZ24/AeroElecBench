from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from aeroecad.impact_intent import (
    INTENT_CASE_TYPES,
    generate_intent_benchmark,
    prepare_intent_development,
    run_intent_baselines,
)


def test_intent_benchmark_is_balanced_and_contains_concurrent_distractors() -> None:
    scenarios = generate_intent_benchmark(seed=101, cases_per_type=2)
    assert len(scenarios) == 12
    assert Counter(item["intent_case_type"] for item in scenarios) == {name: 2 for name in INTENT_CASE_TYPES}
    assert all(len(item["change_inventory"]) == 4 for item in scenarios)
    assert all(len(item["intent_oracle"]["intended_candidate_ids"]) < len(item["change_inventory"]) for item in scenarios)
    assert all("root_node_ids" not in candidate for item in scenarios for candidate in item["change_inventory"])
    assert all(item["engineering_change_request"]["text"] for item in scenarios)


def test_intent_oracle_separates_semantic_selection_from_graph_propagation() -> None:
    scenarios = generate_intent_benchmark(seed=103, cases_per_type=1)
    report_cases = [item for item in scenarios if item["intent_oracle"]["expected_action"] == "report"]
    abstain_cases = [item for item in scenarios if item["intent_oracle"]["expected_action"] == "abstain"]
    assert report_cases and abstain_cases
    assert all(item["intent_oracle"]["root_node_ids"] and item["impact_oracle"]["affected_node_ids"] for item in report_cases)
    assert all(not item["intent_oracle"]["root_node_ids"] and not item["impact_oracle"]["affected_node_ids"] for item in abstain_cases)
    assert any(len(item["intent_oracle"]["intended_candidate_ids"]) == 2 for item in report_cases)


def test_deterministic_controls_show_why_intent_grounding_is_needed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        benchmark, output = root / "benchmark" / "intent.jsonl", root / "results"
        prepare_intent_development(benchmark, cases_per_type=2)
        metrics = run_intent_baselines(benchmark, output)
        all_diff = metrics["modes"]["all_diff_graph"]
        oracle = metrics["modes"]["oracle_root_graph"]
        assert all_diff["candidate_recall"] == 1.0 and all_diff["candidate_precision"] < 1.0
        assert all_diff["impact_set_recall"] == 1.0 and all_diff["impact_set_precision"] < 1.0
        assert all_diff["oracle_action_accuracy"] < 1.0
        assert oracle["candidate_f1"] == oracle["root_f1"] == oracle["impact_set_f1"] == oracle["path_f1"] == 1.0
        assert oracle["oracle_action_accuracy"] == 1.0
        assert {path.name for path in output.iterdir()} == {"evaluation_manifest.json", "evaluation_metrics.json", "evaluation_table.csv", "scenario_table.csv"}
        manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
        assert manifest["llm_calls_performed"] == 0 and manifest["oracle_root_graph_role"] == "upper_bound"
        assert not list(output.rglob("*.md"))


def test_frozen_intent_benchmark_rejects_tampering() -> None:
    with tempfile.TemporaryDirectory() as directory:
        benchmark = Path(directory) / "intent.jsonl"
        prepare_intent_development(benchmark, cases_per_type=1)
        benchmark.write_text(benchmark.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="identity or hash mismatch"):
            run_intent_baselines(benchmark, Path(directory) / "results")
