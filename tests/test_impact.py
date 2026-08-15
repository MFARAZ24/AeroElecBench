from __future__ import annotations

import copy
import json
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from aeroecad.impact_benchmark import IMPACT_CASE_TYPES, generate_impact_benchmark
from aeroecad.impact_evaluation import prepare_impact_development, run_impact_development
from aeroecad.impact_graph import analyze_change_impact, build_version_graph


def test_impact_benchmark_is_balanced_versioned_development_data() -> None:
    scenarios = generate_impact_benchmark(seed=17, cases_per_type=4)
    assert len(scenarios) == 24
    assert Counter(item["impact_case_type"] for item in scenarios) == {name: 4 for name in IMPACT_CASE_TYPES}
    assert {item["split"] for item in scenarios} == {"development"}
    assert all(item["before_design"]["design_id"] == item["after_design"]["design_id"] for item in scenarios)
    report_cases = [item for item in scenarios if item["impact_oracle"]["expected_action"] == "report"]
    assert report_cases and all(item["impact_oracle"]["affected_node_ids"] for item in report_cases)
    assert any(len(path["relations"]) >= 2 for item in report_cases for path in item["impact_oracle"]["impact_paths"])


def test_graph_analyzer_matches_independent_oracle_without_mutation() -> None:
    for scenario in generate_impact_benchmark(seed=23, cases_per_type=1):
        before, after = copy.deepcopy(scenario["before_design"]), copy.deepcopy(scenario["after_design"])
        report = analyze_change_impact(before, after, scenario["change_request"])
        oracle = scenario["impact_oracle"]
        action = {"completed": "report", "abstained": "abstain"}.get(report["status"], report["status"])
        assert action == oracle["expected_action"]
        assert report["affected_node_ids"] == oracle["affected_node_ids"]
        assert report["impact_paths"] == oracle["impact_paths"]
        assert report["input_designs_unchanged"] is True
        assert report["production_modification_performed"] is False
        assert before == scenario["before_design"] and after == scenario["after_design"]
        if scenario["impact_case_type"] == "missing_information":
            assert report["status"] == "abstained"


def test_version_graph_contains_typed_multilayer_entities() -> None:
    scenario = next(item for item in generate_impact_benchmark(seed=29, cases_per_type=1) if item["impact_case_type"] == "component_replacement")
    graph = build_version_graph(scenario["before_design"], scenario["after_design"])
    entity_types = {item["entity_type"] for item in graph["nodes"].values()}
    relations = {item["relation"] for item in graph["edges"]}
    assert entity_types == {"component", "pin", "wire", "requirement", "verification"}
    assert {"contains_pin", "allocated_wire", "traced_to_requirement", "verified_by"} <= relations


def test_development_evaluation_writes_metrics_without_narrative_output() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        benchmark, output = root / "benchmark" / "impact.jsonl", root / "results"
        scenarios, manifest = prepare_impact_development(benchmark)
        assert len(scenarios) == 24 and manifest["split_counts"] == {"development": 24}
        metrics = run_impact_development(benchmark, output)
        for name in (
            "impact_set_precision", "impact_set_recall", "impact_set_f1", "impact_exact_scenario_accuracy",
            "path_precision", "path_recall", "path_f1", "multi_hop_recall", "abstention_precision",
            "abstention_recall", "clean_case_specificity", "oracle_action_accuracy", "input_immutability_rate",
        ):
            assert metrics[name] == 1.0
        assert metrics["production_modification_count"] == 0
        assert len((output / "impact_records.jsonl").read_text(encoding="utf-8").splitlines()) == 24
        assert (output / "evaluation_metrics.json").exists()
        assert (output / "evaluation_table.csv").exists()
        evaluation_manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
        assert evaluation_manifest["development_only"] is True
        assert evaluation_manifest["narrative_output"] is False
        assert not list(output.rglob("*.md"))


def test_development_evaluation_rejects_benchmark_tampering() -> None:
    with tempfile.TemporaryDirectory() as directory:
        benchmark = Path(directory) / "impact.jsonl"
        prepare_impact_development(benchmark)
        benchmark.write_text(benchmark.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="hash mismatch"):
            run_impact_development(benchmark, Path(directory) / "results")
