from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .catalog import load_catalog
from .e2e import run_e2e_prototype
from .evaluation import evaluate, save_results
from .generator import generate_benchmark, read_jsonl, write_jsonl
from .heldout import DEFAULT_BENCHMARK as HELDOUT_BENCHMARK
from .heldout import DEFAULT_OUTPUT as HELDOUT_OUTPUT
from .heldout import DEFAULT_SEED as HELDOUT_SEED
from .heldout import prepare_heldout_benchmark, run_heldout_evaluation
from .impact_agent import IMPACT_MODES
from .impact_assurance_v2 import DEFAULT_OUTPUT as ASSURANCE_V2_OUTPUT
from .impact_assurance_v2 import run_assurance_v2
from .impact_comparison import DEFAULT_COMPARISON_OUTPUT, run_impact_comparison
from .impact_evaluation import DEFAULT_BENCHMARK as IMPACT_BENCHMARK
from .impact_evaluation import DEFAULT_OUTPUT as IMPACT_OUTPUT
from .impact_evaluation import DEFAULT_SEED as IMPACT_SEED
from .impact_evaluation import prepare_impact_development, run_impact_development
from .impact_heldout import DEFAULT_BENCHMARK as IMPACT_HELDOUT_BENCHMARK
from .impact_heldout import DEFAULT_OUTPUT as IMPACT_HELDOUT_OUTPUT
from .impact_heldout import DEFAULT_SEED as IMPACT_HELDOUT_SEED
from .impact_heldout import prepare_impact_heldout, run_impact_heldout
from .impact_intent import DEFAULT_BENCHMARK as INTENT_BENCHMARK
from .impact_intent import DEFAULT_OUTPUT as INTENT_OUTPUT
from .impact_intent import DEFAULT_SEED as INTENT_SEED
from .impact_intent import prepare_intent_development, run_intent_baselines
from .impact_intent_llm import DEFAULT_OUTPUT as INTENT_QWEN_OUTPUT
from .impact_intent_llm import run_intent_qwen
from .llm_evaluation import run_llm_experiment
from .llm_repair import REPAIR_MODES
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
    repair.add_argument("--repair-mode", choices=REPAIR_MODES, default="llm_direct")
    repair.add_argument("--profile", choices=("smoke", "pilot", "full"), default="smoke")
    repair.add_argument("--base-url", default="http://localhost:11434")
    repair.add_argument("--timeout", type=float, default=300.0)
    repair.add_argument("--seed", type=int, default=2027)
    repair.add_argument("--max-tokens", type=int, default=400)
    prototype = subparsers.add_parser("repair-prototype", help="Run the 25-case end-to-end verified-repair prototype")
    prototype.add_argument("--model", default="qwen2.5:7b")
    prototype.add_argument("--modes", nargs="+", choices=REPAIR_MODES, default=list(REPAIR_MODES))
    prototype.add_argument("--benchmark", type=Path, default=Path("benchmark/v05/repair_e2e_25.jsonl"))
    prototype.add_argument("--output-dir", type=Path, default=Path("results/repair_v05_e2e"))
    prototype.add_argument("--catalog", type=Path, default=None)
    prototype.add_argument("--source-registry", type=Path, default=Path("data/source_registry.json"))
    prototype.add_argument("--base-url", default="http://localhost:11434")
    prototype.add_argument("--timeout", type=float, default=300.0)
    prototype.add_argument("--seed", type=int, default=4107)
    prototype.add_argument("--max-tokens", type=int, default=300)
    prototype.add_argument("--cases-per-type", type=int, default=5)
    prototype.add_argument("--prepare-only", action="store_true")
    heldout = subparsers.add_parser("repair-heldout", help="Run the frozen v0.6 held-out repair evaluation")
    heldout.add_argument("--model", default="qwen2.5:7b")
    heldout.add_argument("--modes", nargs="+", choices=REPAIR_MODES, default=list(REPAIR_MODES))
    heldout.add_argument("--benchmark", type=Path, default=HELDOUT_BENCHMARK)
    heldout.add_argument("--output-dir", type=Path, default=HELDOUT_OUTPUT)
    heldout.add_argument("--catalog", type=Path, default=None)
    heldout.add_argument("--source-registry", type=Path, default=Path("data/source_registry.json"))
    heldout.add_argument("--base-url", default="http://localhost:11434")
    heldout.add_argument("--timeout", type=float, default=300.0)
    heldout.add_argument("--seed", type=int, default=HELDOUT_SEED)
    heldout.add_argument("--max-tokens", type=int, default=300)
    heldout.add_argument("--cases-per-type", type=int, default=5)
    heldout.add_argument("--prepare-only", action="store_true")
    impact = subparsers.add_parser("impact-prototype", help="Run the v0.7 versioned change-impact development prototype")
    impact.add_argument("--benchmark", type=Path, default=IMPACT_BENCHMARK)
    impact.add_argument("--output-dir", type=Path, default=IMPACT_OUTPUT)
    impact.add_argument("--catalog", type=Path, default=None)
    impact.add_argument("--source-registry", type=Path, default=Path("data/source_registry.json"))
    impact.add_argument("--seed", type=int, default=IMPACT_SEED)
    impact.add_argument("--cases-per-type", type=int, default=4)
    impact.add_argument("--prepare-only", action="store_true")
    comparison = subparsers.add_parser("impact-comparison", help="Run the resumable v0.7 change-impact mode comparison")
    comparison.add_argument("--model", default="qwen2.5:7b")
    comparison.add_argument("--modes", nargs="+", choices=IMPACT_MODES, default=list(IMPACT_MODES))
    comparison.add_argument("--benchmark", type=Path, default=IMPACT_BENCHMARK)
    comparison.add_argument("--output-dir", type=Path, default=DEFAULT_COMPARISON_OUTPUT)
    comparison.add_argument("--catalog", type=Path, default=None)
    comparison.add_argument("--base-url", default="http://localhost:11434")
    comparison.add_argument("--timeout", type=float, default=300.0)
    comparison.add_argument("--seed", type=int, default=IMPACT_SEED)
    comparison.add_argument("--max-tokens", type=int, default=2500)
    comparison.add_argument("--retrieval-top-k", type=int, default=12)
    comparison.add_argument("--profile", choices=("smoke", "development"), default="development")
    impact_heldout = subparsers.add_parser("impact-heldout", help="Prepare or run the frozen v0.8 change-impact evaluation")
    impact_heldout.add_argument("--model", default="qwen2.5:7b")
    impact_heldout.add_argument("--modes", nargs="+", choices=IMPACT_MODES, default=list(IMPACT_MODES))
    impact_heldout.add_argument("--benchmark", type=Path, default=IMPACT_HELDOUT_BENCHMARK)
    impact_heldout.add_argument("--output-dir", type=Path, default=IMPACT_HELDOUT_OUTPUT)
    impact_heldout.add_argument("--catalog", type=Path, default=None)
    impact_heldout.add_argument("--source-registry", type=Path, default=Path("data/source_registry.json"))
    impact_heldout.add_argument("--base-url", default="http://localhost:11434")
    impact_heldout.add_argument("--timeout", type=float, default=900.0)
    impact_heldout.add_argument("--seed", type=int, default=IMPACT_HELDOUT_SEED)
    impact_heldout.add_argument("--max-tokens", type=int, default=2500)
    impact_heldout.add_argument("--retrieval-top-k", type=int, default=12)
    impact_heldout.add_argument("--cases-per-type", type=int, default=5)
    impact_heldout.add_argument("--prepare-only", action="store_true")
    assurance_v2 = subparsers.add_parser("impact-assurance-v2", help="Run the resumable v0.9 tool-observing assurance-agent development experiment")
    assurance_v2.add_argument("--model", default="qwen2.5:7b")
    assurance_v2.add_argument("--benchmark", type=Path, default=IMPACT_BENCHMARK)
    assurance_v2.add_argument("--output-dir", type=Path, default=ASSURANCE_V2_OUTPUT)
    assurance_v2.add_argument("--catalog", type=Path, default=None)
    assurance_v2.add_argument("--base-url", default="http://localhost:11434")
    assurance_v2.add_argument("--timeout", type=float, default=900.0)
    assurance_v2.add_argument("--seed", type=int, default=9107)
    assurance_v2.add_argument("--max-tokens", type=int, default=2500)
    assurance_v2.add_argument("--profile", choices=("smoke", "development"), default="smoke")
    intent = subparsers.add_parser("impact-intent", help="Prepare or run the deterministic v1.0 intent-grounding controls")
    intent.add_argument("--benchmark", type=Path, default=INTENT_BENCHMARK)
    intent.add_argument("--output-dir", type=Path, default=INTENT_OUTPUT)
    intent.add_argument("--catalog", type=Path, default=None)
    intent.add_argument("--source-registry", type=Path, default=Path("data/source_registry.json"))
    intent.add_argument("--seed", type=int, default=INTENT_SEED)
    intent.add_argument("--cases-per-type", type=int, default=4)
    intent.add_argument("--prepare-only", action="store_true")
    intent_qwen = subparsers.add_parser("impact-intent-qwen", help="Run the resumable v1.2 Qwen intent-grounding experiment")
    intent_qwen.add_argument("--model", default="qwen2.5:7b")
    intent_qwen.add_argument("--benchmark", type=Path, default=INTENT_BENCHMARK)
    intent_qwen.add_argument("--output-dir", type=Path, default=INTENT_QWEN_OUTPUT)
    intent_qwen.add_argument("--catalog", type=Path, default=None)
    intent_qwen.add_argument("--base-url", default="http://localhost:11434")
    intent_qwen.add_argument("--timeout", type=float, default=900.0)
    intent_qwen.add_argument("--seed", type=int, default=11107)
    intent_qwen.add_argument("--max-tokens", type=int, default=300)
    intent_qwen.add_argument("--profile", choices=("smoke", "development"), default="smoke")
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
        summary = run_repair_experiment(read_jsonl(args.benchmark), load_catalog(args.catalog), args.models, args.profile, args.output_dir, args.base_url, args.timeout, args.seed, args.max_tokens, args.repair_mode, args.benchmark)
        print(json.dumps({"status": "complete", "profile": args.profile, "repair_mode": args.repair_mode, "scenario_count": summary["scenario_count"], "models": args.models, "results": str(args.output_dir)}, indent=2))
        return
    if args.command == "repair-prototype":
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
        return
    if args.command == "repair-heldout":
        if args.prepare_only:
            scenarios, manifest = prepare_heldout_benchmark(
                args.benchmark, args.catalog, args.source_registry, args.seed, args.cases_per_type,
            )
            result = {
                "status": "prepared", "scenario_count": len(scenarios),
                "benchmark": str(args.benchmark), "benchmark_sha256": manifest["benchmark_sha256"],
            }
        else:
            metrics = run_heldout_evaluation(
                args.model, tuple(args.modes), args.benchmark, args.output_dir, args.catalog,
                args.base_url, args.timeout, args.seed, args.max_tokens,
            )
            result = {
                "status": "complete", "scenario_count": metrics["scenario_count"],
                "model": args.model, "results": str(args.output_dir),
            }
        print(json.dumps(result, indent=2))
        return
    if args.command == "impact-prototype":
        if args.prepare_only:
            scenarios, manifest = prepare_impact_development(
                args.benchmark, args.catalog, args.source_registry, args.seed, args.cases_per_type,
            )
            result = {
                "status": "prepared", "scenario_count": len(scenarios),
                "benchmark": str(args.benchmark), "benchmark_sha256": manifest["benchmark_sha256"],
                "split": "development",
            }
        else:
            metrics = run_impact_development(args.benchmark, args.output_dir, args.catalog)
            result = {
                "status": "complete", "scenario_count": metrics["scenario_count"],
                "mode": metrics["mode"], "results": str(args.output_dir), "split": "development",
            }
        print(json.dumps(result, indent=2))
        return
    if args.command == "impact-comparison":
        metrics = run_impact_comparison(
            args.model, tuple(args.modes), args.benchmark, args.output_dir, args.catalog,
            args.base_url, args.timeout, args.seed, args.max_tokens, args.retrieval_top_k, args.profile,
        )
        print(json.dumps({
            "status": "complete", "scenario_count": metrics["scenario_count"], "model": args.model,
            "modes": list(metrics["modes"]), "profile": args.profile, "results": str(args.output_dir),
        }, indent=2))
        return
    if args.command == "impact-heldout":
        if args.prepare_only:
            scenarios, manifest = prepare_impact_heldout(
                args.benchmark, args.catalog, args.source_registry, args.seed, args.cases_per_type,
            )
            result = {
                "status": "prepared", "scenario_count": len(scenarios), "benchmark": str(args.benchmark),
                "benchmark_sha256": manifest["benchmark_sha256"], "split": "heldout",
            }
        else:
            metrics = run_impact_heldout(
                args.model, tuple(args.modes), args.benchmark, args.output_dir, args.catalog,
                args.base_url, args.timeout, args.seed, args.max_tokens, args.retrieval_top_k,
            )
            result = {
                "status": "complete", "scenario_count": metrics["scenario_count"], "model": args.model,
                "modes": list(metrics["modes"]), "profile": "heldout", "results": str(args.output_dir),
            }
        print(json.dumps(result, indent=2))
        return
    if args.command == "impact-assurance-v2":
        metrics = run_assurance_v2(
            args.model, args.benchmark, args.output_dir, args.catalog, args.base_url,
            args.timeout, args.seed, args.max_tokens, args.profile,
        )
        print(json.dumps({
            "status": "complete", "scenario_count": metrics["scenario_count"], "model": args.model,
            "profile": args.profile, "stages": list(metrics["stages"]), "results": str(args.output_dir),
        }, indent=2))
        return
    if args.command == "impact-intent":
        if args.prepare_only:
            scenarios, manifest = prepare_intent_development(args.benchmark, args.catalog, args.source_registry, args.seed, args.cases_per_type)
            result = {"status": "prepared", "scenario_count": len(scenarios), "benchmark": str(args.benchmark), "benchmark_sha256": manifest["benchmark_sha256"], "split": "development", "llm_calls": 0}
        else:
            metrics = run_intent_baselines(args.benchmark, args.output_dir, args.catalog)
            result = {"status": "complete", "scenario_count": metrics["scenario_count"], "modes": list(metrics["modes"]), "results": str(args.output_dir), "llm_calls": 0}
        print(json.dumps(result, indent=2))
        return
    if args.command == "impact-intent-qwen":
        metrics = run_intent_qwen(args.model, args.benchmark, args.output_dir, args.catalog, args.base_url, args.timeout, args.seed, args.max_tokens, args.profile)
        print(json.dumps({
            "status": "complete", "scenario_count": metrics["scenario_count"], "model": args.model,
            "mode": "qwen_intent_graph", "profile": args.profile, "results": str(args.output_dir),
            "llm_calls": metrics["modes"]["qwen_intent_graph"]["llm_call_count"],
        }, indent=2))
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
