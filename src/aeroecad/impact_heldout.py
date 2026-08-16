from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .catalog import load_catalog
from .impact_agent import IMPACT_MODES
from .impact_benchmark import generate_impact_benchmark, read_impact_benchmark, write_impact_benchmark
from .impact_comparison import PIPELINE_VERSION, run_impact_comparison
from .impact_evaluation import validate_impact_benchmark
from .ollama import OllamaClient

DEFAULT_BENCHMARK = Path("benchmark/v08/impact_heldout_30.jsonl")
DEFAULT_OUTPUT = Path("results/impact_v08_heldout")
DEFAULT_SEED = 8107
BENCHMARK_ID = "AEROELECBENCH-IMPACT-HELDOUT-0.8"
EVALUATION_ID = "AEROELECBENCH-IMPACT-EVALUATION-0.8"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_impact_heldout(
    benchmark_path: str | Path = DEFAULT_BENCHMARK,
    catalog_path: str | Path | None = None,
    source_registry_path: str | Path = "data/source_registry.json",
    seed: int = DEFAULT_SEED,
    cases_per_type: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark, manifest_path, registry_path = Path(benchmark_path), Path(benchmark_path).with_name("manifest.json"), Path(source_registry_path)
    if benchmark.exists() or manifest_path.exists():
        if not benchmark.exists() or not manifest_path.exists():
            raise ValueError("Incomplete v0.8 benchmark freeze; both benchmark and manifest are required")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("benchmark_id") != BENCHMARK_ID or manifest.get("benchmark_sha256") != _sha256(benchmark) or manifest.get("seed") != seed or manifest.get("cases_per_type") != cases_per_type:
            raise ValueError("Existing v0.8 benchmark does not match the requested frozen configuration")
        scenarios, catalog = read_impact_benchmark(benchmark), load_catalog(catalog_path)
        validate_impact_benchmark(scenarios, catalog["rules"], "heldout")
        return scenarios, manifest
    if not registry_path.exists():
        raise FileNotFoundError(f"Source registry not found: {registry_path}")
    catalog = load_catalog(catalog_path)
    scenarios = generate_impact_benchmark(seed, cases_per_type, "v08", "heldout")
    validate_impact_benchmark(scenarios, catalog["rules"], "heldout")
    write_impact_benchmark(scenarios, benchmark)
    manifest = {
        "benchmark_id": BENCHMARK_ID, "benchmark": benchmark.name, "benchmark_sha256": _sha256(benchmark),
        "catalog_id": catalog["catalog_id"], "catalog_version": catalog["version"],
        "catalog_sha256": _sha256(Path(catalog_path or "data/rules.json")),
        "source_registry_id": json.loads(registry_path.read_text(encoding="utf-8"))["registry_id"],
        "source_registry_sha256": _sha256(registry_path), "seed": seed, "cases_per_type": cases_per_type,
        "scenario_count": len(scenarios),
        "case_type_counts": dict(sorted(Counter(item["impact_case_type"] for item in scenarios).items())),
        "split_counts": {"heldout": len(scenarios)}, "oracle_validation_rate": 1.0,
        "oracle_exposed_to_model": False, "root_node_ids_exposed_to_model": False,
        "pipeline_version": PIPELINE_VERSION, "development_only": False, "heldout": True,
        "frozen_before_model_run": True, "posthoc_tuning_allowed": False,
        "production_modifications_allowed": False, "dataset_kind": "fictional_synthetic",
        "rule_classification": "research_only", "certification_evidence": False, "narrative_output": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return scenarios, manifest


def run_impact_heldout(
    model: str = "qwen2.5:7b",
    modes: tuple[str, ...] = IMPACT_MODES,
    benchmark_path: str | Path = DEFAULT_BENCHMARK,
    output_dir: str | Path = DEFAULT_OUTPUT,
    catalog_path: str | Path | None = None,
    base_url: str = "http://localhost:11434",
    timeout: float = 900.0,
    seed: int = DEFAULT_SEED,
    max_tokens: int = 2500,
    retrieval_top_k: int = 12,
    client: OllamaClient | None = None,
) -> dict[str, Any]:
    return run_impact_comparison(
        model, modes, benchmark_path, output_dir, catalog_path, base_url, timeout, seed,
        max_tokens, retrieval_top_k, "heldout", client, BENCHMARK_ID, "heldout", EVALUATION_ID,
        False, True, False,
    )
