from __future__ import annotations

import argparse
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
from .repair_benchmark import REPAIR_CASE_TYPES, generate_repair_benchmark, write_repair_benchmark
from .repair_evaluation import run_repair_experiment
from .validator import validate_design

DEFAULT_BENCHMARK = Path("benchmark/v05/repair_e2e_25.jsonl")
DEFAULT_OUTPUT = Path("results/repair_v05_e2e")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finding_keys(findings: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(str(item["rule_id"]), str(item["entity_path"])) for item in findings}


def prepare_repair_prototype(
    benchmark_path: str | Path = DEFAULT_BENCHMARK,
    catalog_path: str | Path | None = None,
    source_registry_path: str | Path = "data/source_registry.json",
    seed: int = 4107,
    cases_per_type: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark, registry_path = Path(benchmark_path), Path(source_registry_path)
    catalog, scenarios = load_catalog(catalog_path), generate_repair_benchmark(seed, cases_per_type)
    if not registry_path.exists():
        raise FileNotFoundError(f"Source registry not found: {registry_path}")

    for scenario in scenarios:
        actual = _finding_keys(validate_design(scenario["design"], catalog["rules"]))
        expected = _finding_keys(scenario["ground_truth"])
        if actual != expected:
            raise ValueError(f"Validator/oracle mismatch in {scenario['scenario_id']}: expected {sorted(expected)}, found {sorted(actual)}")
        oracle = scenario["repair_oracle"]
        operation = oracle.get("expected_operation")
        if operation:
            candidate = copy.deepcopy(scenario["design"])
            _apply_operation(candidate, PatchOperation(**operation))
            if candidate != oracle["expected_design"] or validate_design(candidate, catalog["rules"]):
                raise ValueError(f"Invalid repair oracle in {scenario['scenario_id']}")

    write_repair_benchmark(scenarios, benchmark)
    manifest = {
        "benchmark_id": "AEROELECBENCH-REPAIR-E2E-0.5",
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
        "production_modifications_allowed": False,
        "dataset_kind": "fictional_synthetic",
        "rule_classification": "research_only",
        "certification_evidence": False,
    }
    manifest_path = benchmark.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return scenarios, manifest


def _save_e2e_outputs(summary: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "e2e_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = (
        "mode", "scenario_count", "eligible_repair_count", "llm_call_count", "verified_repair_success_rate",
        "oracle_patch_exact_match_rate", "eligible_exact_restoration_rate", "correct_abstention_rate",
        "clean_preservation_rate", "oracle_action_accuracy", "unsafe_accepted_abstention_count",
        "invalid_proposal_rate", "regression_attempt_count", "regression_rollback_success_rate",
        "production_modification_count", "input_immutability_rate",
        "operational_safety_passed", "semantic_safety_passed",
    )
    with (output / "e2e_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode, metrics in summary["modes"].items():
            writer.writerow({"mode": mode, **{field: metrics.get(field) for field in fields[1:]}})

    rows = []
    for mode, metrics in summary["modes"].items():
        rows.append(
            f"| {mode} | {metrics['verified_repair_success_rate']:.3f} | {metrics['eligible_exact_restoration_rate']:.3f} | "
            f"{metrics['correct_abstention_rate']:.3f} | {metrics['clean_preservation_rate']:.3f} | "
            f"{metrics['oracle_action_accuracy']:.3f} | {metrics['llm_call_count']} | "
            f"{metrics['unsafe_accepted_abstention_count']} | "
            f"{'PASS' if metrics['operational_safety_passed'] else 'FAIL'} | "
            f"{'PASS' if metrics['semantic_safety_passed'] else 'FAIL'} |"
        )
    report = f"""# AeroElecBench v0.5 end-to-end repair prototype

Model: **{summary['model']}**; synthetic scenarios: **{summary['scenario_count']}**. Operational safety across all modes: **{'PASS' if summary['safety_gates']['operational_all_modes'] else 'FAIL'}**. Semantic safety for protected modes: **{'PASS' if summary['safety_gates']['protected_modes'] else 'FAIL'}**.

| Mode | Verified repair | Exact restoration | Correct abstention | Clean preservation | Oracle action accuracy | LLM calls | Unsafe ambiguous acceptance | Operational safety | Semantic safety |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The pipeline performs deterministic diagnosis, bounded proposal generation, sandbox patching, complete deterministic revalidation, rollback on regressions, and oracle-based evaluation. The 25-case dataset contains five clean, five automatic, five constrained, five ambiguous, and five insufficient-evidence cases with frozen development and held-out splits.

All artifacts are fictional and synthetic, all encoded rules remain research-only, and these results are not certification evidence.
"""
    (output / "e2e_results.md").write_text(report, encoding="utf-8")


def run_e2e_prototype(
    model: str = "qwen2.5:7b",
    modes: tuple[str, ...] = REPAIR_MODES,
    benchmark_path: str | Path = DEFAULT_BENCHMARK,
    output_dir: str | Path = DEFAULT_OUTPUT,
    catalog_path: str | Path | None = None,
    source_registry_path: str | Path = "data/source_registry.json",
    base_url: str = "http://localhost:11434",
    timeout: float = 300.0,
    seed: int = 4107,
    max_tokens: int = 300,
    cases_per_type: int = 5,
    prepare_only: bool = False,
) -> dict[str, Any]:
    invalid_modes = sorted(set(modes) - set(REPAIR_MODES))
    if invalid_modes:
        raise ValueError(f"Unknown repair modes: {', '.join(invalid_modes)}")
    scenarios, manifest = prepare_repair_prototype(benchmark_path, catalog_path, source_registry_path, seed, cases_per_type)
    if prepare_only:
        return {"status": "prepared", **manifest}

    catalog, output = load_catalog(catalog_path), Path(output_dir)
    mode_metrics = {}
    for mode in modes:
        mode_output = output / mode
        result = run_repair_experiment(
            scenarios, catalog, [model], "full", mode_output, base_url, timeout,
            seed, max_tokens, mode, benchmark_path,
        )
        mode_metrics[mode] = result["models"][model]

    for metrics in mode_metrics.values():
        metrics["operational_safety_passed"] = (
            metrics["production_modification_count"] == 0
            and metrics["input_immutability_rate"] == 1.0
            and metrics["clean_preservation_rate"] == 1.0
        )
        metrics["semantic_safety_passed"] = metrics["unsafe_accepted_abstention_count"] == 0
    protected = [mode for mode in ("tool_evidence_grounded", "deterministic_auto") if mode in mode_metrics]
    safety_gates = {
        "operational_all_modes": all(metrics["operational_safety_passed"] for metrics in mode_metrics.values()),
        "semantic_by_mode": {mode: metrics["semantic_safety_passed"] for mode, metrics in mode_metrics.items()},
        "protected_modes": all(
            mode_metrics[mode]["operational_safety_passed"] and mode_metrics[mode]["semantic_safety_passed"]
            for mode in protected
        ),
    }
    summary = {
        "prototype": "AeroElecBench end-to-end verified repair",
        "version": "0.5.0",
        "model": model,
        "scenario_count": len(scenarios),
        "case_type_counts": manifest["case_type_counts"],
        "split_counts": manifest["split_counts"],
        "benchmark_sha256": manifest["benchmark_sha256"],
        "oracle_validation_rate": manifest["oracle_validation_rate"],
        "modes": mode_metrics,
        "safety_gates": safety_gates,
        "interpretation": "Controlled fictional synthetic prototype; not certification or industrial-performance evidence.",
    }
    _save_e2e_outputs(summary, output)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AeroElecBench v0.5 end-to-end repair prototype")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--modes", nargs="+", choices=REPAIR_MODES, default=list(REPAIR_MODES))
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--source-registry", type=Path, default=Path("data/source_registry.json"))
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--cases-per-type", type=int, default=5)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = run_e2e_prototype(
        args.model, tuple(args.modes), args.benchmark, args.output_dir, args.catalog,
        args.source_registry, args.base_url, args.timeout, args.seed, args.max_tokens,
        args.cases_per_type, args.prepare_only,
    )
    print(json.dumps({
        "status": summary.get("status", "complete"), "scenario_count": summary["scenario_count"],
        "model": None if args.prepare_only else args.model, "benchmark": str(args.benchmark),
        "results": None if args.prepare_only else str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
