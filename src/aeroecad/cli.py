from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .catalog import load_catalog
from .evaluation import evaluate, save_results
from .generator import generate_benchmark, read_jsonl, write_jsonl
from .llm_evaluation import run_llm_experiment
from .llm_review import LLM_MODES
from .ollama import OllamaClient
from .repair_evaluation import run_repair_experiment

DEFAULT_MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AeroECAD-Agent synthetic verification prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "experiment"):
        command = subparsers.add_parser(name)
        command.add_argument("--seed", type=int, default=2027)
        command.add_argument("--cases-per-rule", type=int, default=20)
        command.add_argument("--clean-cases", type=int, default=20)
        command.add_argument("--mixed-cases", type=int, default=50)
        command.add_argument("--benchmark-out", type=Path, default=Path("benchmark/synthetic_benchmark.jsonl"))
        if name == "experiment":
            command.add_argument("--catalog", type=Path, default=None)
            command.add_argument("--output-dir", type=Path, default=Path("results"))
    check = subparsers.add_parser("ollama-check", help="Verify the local Ollama service and installed models")
    check.add_argument("--base-url", default="http://localhost:11434")
    check.add_argument("--timeout", type=float, default=30.0)
    check.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    llm = subparsers.add_parser("llm-experiment", help="Run resumable local-LLM comparison experiments")
    llm.add_argument("--benchmark", type=Path, default=Path("benchmark/synthetic_benchmark.jsonl"))
    llm.add_argument("--catalog", type=Path, default=None)
    llm.add_argument("--output-dir", type=Path, default=Path("results/llm"))
    llm.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    llm.add_argument("--modes", nargs="+", choices=LLM_MODES, default=list(LLM_MODES))
    llm.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    llm.add_argument("--base-url", default="http://localhost:11434")
    llm.add_argument("--timeout", type=float, default=300.0)
    llm.add_argument("--seed", type=int, default=2027)
    llm.add_argument("--max-tokens", type=int, default=1200)
    repair = subparsers.add_parser("repair-experiment", help="Run resumable verified-repair experiments")
    repair.add_argument("--benchmark", type=Path, default=Path("benchmark/synthetic_benchmark.jsonl"))
    repair.add_argument("--catalog", type=Path, default=None)
    repair.add_argument("--output-dir", type=Path, default=Path("results/repair"))
    repair.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    repair.add_argument("--profile", choices=("smoke", "pilot", "full"), default="smoke")
    repair.add_argument("--base-url", default="http://localhost:11434")
    repair.add_argument("--timeout", type=float, default=300.0)
    repair.add_argument("--seed", type=int, default=2027)
    repair.add_argument("--max-tokens", type=int, default=400)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "ollama-check":
        client = OllamaClient(args.base_url, args.timeout)
        available = client.ensure_models(args.models)
        print(json.dumps({"status": "ready", "base_url": args.base_url, "requested_models": args.models, "installed_models": available}, indent=2))
        return
    if args.command == "repair-experiment":
        if not args.benchmark.exists():
            raise FileNotFoundError(f"Benchmark not found: {args.benchmark}. Run 'aeroecad generate' first.")
        summary = run_repair_experiment(read_jsonl(args.benchmark), load_catalog(args.catalog), args.models, args.profile, args.output_dir, args.base_url, args.timeout, args.seed, args.max_tokens, args.benchmark)
        print(json.dumps({"status": "complete", "profile": args.profile, "scenario_count": summary["scenario_count"], "models": args.models, "results": str(args.output_dir)}, indent=2))
        return
    if args.command == "llm-experiment":
        if not args.benchmark.exists():
            raise FileNotFoundError(f"Benchmark not found: {args.benchmark}. Run 'aeroecad generate' first.")
        summary = run_llm_experiment(read_jsonl(args.benchmark), load_catalog(args.catalog), args.models, args.modes, args.profile, args.output_dir, args.base_url, args.timeout, args.seed, args.max_tokens, args.benchmark)
        print(json.dumps({"status": "complete", "profile": args.profile, "scenario_count": summary["scenario_count"], "models": args.models, "modes": args.modes, "results": str(args.output_dir)}, indent=2))
        return
    scenarios = generate_benchmark(args.seed, args.cases_per_rule, args.clean_cases, args.mixed_cases)
    write_jsonl(scenarios, args.benchmark_out)
    manifest = {
        "benchmark": args.benchmark_out.name, "sha256": hashlib.sha256(args.benchmark_out.read_bytes()).hexdigest(),
        "seed": args.seed, "scenario_count": len(scenarios), "ground_truth_violation_count": sum(len(item["ground_truth"]) for item in scenarios),
        "cases_per_rule": args.cases_per_rule, "clean_cases": args.clean_cases, "mixed_cases": args.mixed_cases,
    }
    manifest_path = args.benchmark_out.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.command == "generate":
        print(json.dumps({"scenario_count": len(scenarios), "benchmark": str(args.benchmark_out), "manifest": str(manifest_path)}, indent=2))
        return
    catalog = load_catalog(args.catalog)
    full_metrics, full_report = evaluate(scenarios, catalog, "full")
    retrieval_metrics, retrieval_report = evaluate(scenarios, catalog, "retrieval_guided")
    categories = Counter(scenario["category"] for scenario in scenarios)
    summary = {
        "prototype": "AeroECAD-Agent", "version": "0.2.0",
        "configuration": {"seed": args.seed, "cases_per_rule": args.cases_per_rule, "clean_cases": args.clean_cases, "mixed_cases": args.mixed_cases},
        "dataset": {"scenario_count": len(scenarios), "categories": dict(sorted(categories.items())), "synthetic": True, "rule_count": len(catalog["rules"])},
        "modes": {"full": full_metrics, "retrieval_guided": retrieval_metrics},
        "interpretation": "Preliminary results validate execution within the encoded fictional rule scope; they do not establish certification or proprietary-data performance.",
    }
    save_results(summary, retrieval_report or full_report, args.output_dir)
    print(json.dumps({"benchmark": str(args.benchmark_out), "results": str(args.output_dir), "modes": {name: {key: value for key, value in metrics.items() if key in {"precision", "recall", "f1", "citation_correctness", "traceability_completeness"}} for name, metrics in summary["modes"].items()}}, indent=2))


if __name__ == "__main__":
    main()
