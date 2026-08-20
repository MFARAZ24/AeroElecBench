from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from aeroecad.impact_intent import generate_intent_benchmark
from aeroecad.impact_intent_heldout_v15 import (
    BENCHMARK_ID,
    DEFAULT_SEED,
    generate_intent_heldout,
    load_frozen_intent_heldout,
    prepare_intent_heldout,
)
from aeroecad.impact_intent_protocol import INPUT_FILE, prepare_oracle_separated_package


def test_heldout_is_balanced_diverse_and_disjoint_from_development() -> None:
    scenarios = generate_intent_heldout()
    assert len(scenarios) == 30
    assert Counter(item["intent_case_type"] for item in scenarios) == {name: 5 for name in ("single_explicit", "single_paraphrased", "multi_intent", "same_entity_distractor", "ambiguous_request", "unmatched_request")}
    assert all(item["split"] == "heldout" and len(item["change_inventory"]) == 4 for item in scenarios)
    assert all(len(item["intent_oracle"]["intended_candidate_ids"]) == 2 for item in scenarios if item["intent_case_type"] == "multi_intent")
    intended = Counter(candidate for item in scenarios for candidate in item["intent_oracle"]["intended_candidate_ids"])
    assert intended == {"CAND-01": 6, "CAND-02": 6, "CAND-03": 6, "CAND-04": 7}
    heldout_requests = {item["engineering_change_request"]["text"] for item in scenarios}
    development_requests = {item["engineering_change_request"]["text"] for item in generate_intent_benchmark(cases_per_type=4)}
    assert len(heldout_requests) == 30 and heldout_requests.isdisjoint(development_requests)


def test_heldout_freeze_and_oracle_separated_package_are_hash_bound() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        benchmark, package = root / "benchmark" / "heldout.jsonl", root / "package"
        scenarios, manifest = prepare_intent_heldout(benchmark)
        assert manifest["benchmark_id"] == BENCHMARK_ID and manifest["scenario_count"] == 30
        assert manifest["heldout"] is True and manifest["posthoc_tuning_allowed"] is False
        assert manifest["exact_development_request_overlap_count"] == 0
        repeated, repeated_manifest = prepare_intent_heldout(benchmark)
        assert repeated == scenarios and repeated_manifest == manifest
        with pytest.raises(ValueError, match="frozen configuration"):
            prepare_intent_heldout(benchmark, seed=DEFAULT_SEED + 1)
        package_manifest = prepare_oracle_separated_package(benchmark, package)
        assert package_manifest["split"] == "heldout" and package_manifest["posthoc_tuning_allowed"] is False
        text = (package / INPUT_FILE).read_text(encoding="utf-8")
        assert all(value not in text for value in ("intent_case_type", "source_scenario_id", "ambiguous_request", "multi_intent", "v15-heldout"))
        benchmark.write_text(benchmark.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="identity or hash mismatch"):
            load_frozen_intent_heldout(benchmark)
