from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .catalog import load_catalog
from .llm_repair import REPAIR_MODES
from .repair import PatchOperation, _apply_operation
from .repair_benchmark import generate_repair_benchmark, read_repair_benchmark, write_repair_benchmark
from .repair_evaluation import run_repair_experiment
from .validator import validate_design

DEFAULT_BENCHMARK = Path("benchmark/v06/repair_heldout_25.jsonl")
DEFAULT_OUTPUT = Path("results/repair_v06_heldout")
DEFAULT_SEED = 6107
BENCHMARK_ID = "AEROELECBENCH-REPAIR-HOLDOUT-0.6"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finding_keys(findings: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(str(item["rule_id"]), str(item["entity_path"])) for item in findings}


def _validate_oracles(scenarios: list[dict[str, Any]], rules: list[dict[str, Any]]) -> None:
    for scenario in scenarios:
        actual = _finding_keys(validate_design(scenario["design"], rules))
        expected = _finding_keys(scenario["ground_truth"])
        if actual != expected:
            raise ValueError(
                f"Validator/oracle mismatch in {scenario['scenario_id']}: "
                f"expected {sorted(expected)}, found {sorted(actual)}"
            )
        operation = scenario["repair_oracle"].get("expected_operation")
        if operation:
            candidate = copy.deepcopy(scenario["design"])
            _apply_operation(candidate, PatchOperation(**operation))
            oracle_design = scenario["repair_oracle"]["expected_design"]
            if candidate != oracle_design or validate_design(candidate, rules):
                raise ValueError(f"Invalid repair oracle in {scenario['scenario_id']}")


def prepare_heldout_benchmark(
    benchmark_path: str | Path = DEFAULT_BENCHMARK,
    catalog_path: str | Path | None = None,
    source_registry_path: str | Path = "data/source_registry.json",
    seed: int = DEFAULT_SEED,
    cases_per_type: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark, registry_path = Path(benchmark_path), Path(source_registry_path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Source registry not found: {registry_path}")
    catalog = load_catalog(catalog_path)
    scenarios = generate_repair_benchmark(
        seed=seed,
        cases_per_type=cases_per_type,
        split_override="heldout",
        scenario_prefix="v06-heldout",
    )
    _validate_oracles(scenarios, catalog["rules"])
    write_repair_benchmark(scenarios, benchmark)

    manifest = {
        "benchmark_id": BENCHMARK_ID,
        "benchmark": benchmark.name,
        "benchmark_sha256": _sha256(benchmark),
        "catalog_id": catalog["catalog_id"],
        "catalog_version": catalog["version"],
        "catalog_sha256": _sha256(Path(catalog_path or "data/rules.json")),
        "source_registry_id": json.loads(registry_path.read_text(encoding="utf-8"))["registry_id"],
        "source_registry_sha256": _sha256(registry_path),
        "seed": seed,
        "scenario_count": len(scenarios),
        "case_type_counts": dict(sorted(Counter(item["repair_case_type"] for item in scenarios).items())),
        "split_counts": dict(sorted(Counter(item["split"] for item in scenarios).items())),
        "ground_truth_violation_count": sum(len(item["ground_truth"]) for item in scenarios),
        "oracle_validation_rate": 1.0,
        "oracle_exposed_to_model": False,
        "evaluation_only": True,
        "frozen_before_model_run": True,
        "posthoc_tuning_allowed": False,
        "production_modifications_allowed": False,
        "dataset_kind": "fictional_synthetic",
        "rule_classification": "research_only",
        "certification_evidence": False,
        "narrative_output": False,
    }
    benchmark.with_name("manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return scenarios, manifest


def _load_frozen_benchmark(
    benchmark_path: str | Path,
    catalog_path: str | Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    benchmark = Path(benchmark_path)
    manifest_path = benchmark.with_name("manifest.json")
    if not benchmark.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Frozen v0.6 benchmark not found at {benchmark}. Run 'aeroecad repair-heldout --prepare-only' first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError(f"Unexpected benchmark id: {manifest.get('benchmark_id')}")
    if _sha256(benchmark) != manifest.get("benchmark_sha256"):
        raise ValueError("Frozen benchmark hash mismatch; do not run the evaluation")
    scenarios, catalog = read_repair_benchmark(benchmark), load_catalog(catalog_path)
    if not scenarios or any(item.get("split") != "heldout" for item in scenarios):
        raise ValueError("Every v0.6 scenario must belong to the heldout split")
    _validate_oracles(scenarios, catalog["rules"])
    return scenarios, manifest, catalog


def _save_metrics(metrics: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    fields = (
        "mode", "scenario_count", "eligible_repair_count", "llm_call_count",
        "verified_repair_success_rate", "eligible_exact_restoration_rate",
        "correct_abstention_rate", "clean_preservation_rate", "oracle_action_accuracy",
        "unsafe_accepted_abstention_count", "invalid_proposal_rate", "regression_attempt_count",
        "production_modification_count", "input_immutability_rate",
        "operational_safety_passed", "semantic_safety_passed",
    )
    with (output / "evaluation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode, values in metrics["modes"].items():
            writer.writerow({"mode": mode, **{field: values.get(field) for field in fields[1:]}})


def run_heldout_evaluation(
    model: str = "qwen2.5:7b",
    modes: tuple[str, ...] = REPAIR_MODES,
    benchmark_path: str | Path = DEFAULT_BENCHMARK,
    output_dir: str | Path = DEFAULT_OUTPUT,
    catalog_path: str | Path | None = None,
    base_url: str = "http://localhost:11434",
    timeout: float = 300.0,
    seed: int = DEFAULT_SEED,
    max_tokens: int = 300,
) -> dict[str, Any]:
    invalid_modes = sorted(set(modes) - set(REPAIR_MODES))
    if invalid_modes:
        raise ValueError(f"Unknown repair modes: {', '.join(invalid_modes)}")
    scenarios, benchmark_manifest, catalog = _load_frozen_benchmark(benchmark_path, catalog_path)
    output = Path(output_dir)
    mode_metrics: dict[str, dict[str, Any]] = {}
    for mode in modes:
        result = run_repair_experiment(
            scenarios, catalog, [model], "full", output / mode, base_url, timeout,
            seed, max_tokens, mode, benchmark_path, save_aggregate_outputs=False,
        )
        values = result["models"][model]
        values["operational_safety_passed"] = (
            values["production_modification_count"] == 0
            and values["input_immutability_rate"] == 1.0
            and values["clean_preservation_rate"] == 1.0
        )
        values["semantic_safety_passed"] = values["unsafe_accepted_abstention_count"] == 0
        mode_metrics[mode] = values

    protected = [mode for mode in ("tool_evidence_grounded", "deterministic_auto") if mode in mode_metrics]
    metrics = {
        "evaluation_id": "AEROELECBENCH-REPAIR-EVALUATION-0.6",
        "version": "0.6.0",
        "model": model,
        "scenario_count": len(scenarios),
        "case_type_counts": benchmark_manifest["case_type_counts"],
        "split_counts": benchmark_manifest["split_counts"],
        "benchmark_sha256": benchmark_manifest["benchmark_sha256"],
        "oracle_validation_rate": benchmark_manifest["oracle_validation_rate"],
        "modes": mode_metrics,
        "safety_gates": {
            "operational_all_modes": all(value["operational_safety_passed"] for value in mode_metrics.values()),
            "semantic_by_mode": {mode: value["semantic_safety_passed"] for mode, value in mode_metrics.items()},
            "protected_modes": bool(protected) and all(
                mode_metrics[mode]["operational_safety_passed"]
                and mode_metrics[mode]["semantic_safety_passed"]
                for mode in protected
            ),
        },
    }
    _save_metrics(metrics, output)
    evaluation_manifest = {
        "evaluation_id": metrics["evaluation_id"],
        "benchmark_id": benchmark_manifest["benchmark_id"],
        "benchmark_sha256": benchmark_manifest["benchmark_sha256"],
        "model": model,
        "modes": list(modes),
        "seed": seed,
        "temperature": 0,
        "max_tokens": max_tokens,
        "oracle_exposed_to_model": False,
        "narrative_output": False,
        "metric_generation": "deterministic",
    }
    (output / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return metrics
