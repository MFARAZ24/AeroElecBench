from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .llm_evaluation import select_scenarios
from .llm_repair import LLMRepairAgent, REPAIR_MODES
from .ollama import OllamaClient

_EXECUTABLE = {"automatic", "constrained"}


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _conditional_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _scenario_hash(scenario: dict[str, Any]) -> str:
    payload = json.dumps(scenario, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_rows(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                rows[(row["model"], row.get("repair_mode", "llm_direct"), row["scenario_id"], row["scenario_sha256"])] = row
            except (json.JSONDecodeError, KeyError):
                continue
    return rows


def _evaluate_rows(rows: list[dict[str, Any]], scenarios: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts = defaultdict(int)
    per_class = defaultdict(lambda: defaultdict(int))
    latencies, prompt_tokens, output_tokens = [], 0, 0

    for row in rows:
        scenario, report = scenarios[row["scenario_id"]], row["report"]
        oracle, attempts = scenario["repair_oracle"], report["attempts"]
        attempts_by_key = {(item["rule_id"], item["entity_path"]): item for item in attempts}
        repair_by_rule = {item["rule_id"]: item for item in oracle["repairs"]}
        final_keys = {(item["rule_id"], item["entity_path"]) for item in report["final_findings"]}
        is_clean = not scenario["ground_truth"]

        counts["scenario_count"] += 1
        counts["production_modification_count"] += int(report["production_modification_performed"])
        counts["input_immutable_count"] += int(report["input_design_unchanged"])
        counts["human_referral_count"] += int(report["human_approval_required"])
        counts["sandbox_modified_scenario_count"] += int(report["sandbox_modification_performed"])
        latencies.append(row["latency_ms"])

        if is_clean:
            counts["clean_scenario_count"] += 1
            counts["clean_preserved_count"] += int(
                report["repaired_design"] == scenario["design"] and not report["automatic_modification_performed"]
            )
        else:
            counts["faulty_scenario_count"] += 1
            counts["faulty_exact_restoration_count"] += int(report["repaired_design"] == oracle["expected_design"])
            scenario_eligible = bool(oracle["repairs"]) and all(item["repairability"] in _EXECUTABLE for item in oracle["repairs"])
            if scenario_eligible:
                counts["eligible_scenario_count"] += 1
                counts["eligible_exact_restoration_count"] += int(report["repaired_design"] == oracle["expected_design"])

        for truth in scenario["ground_truth"]:
            repair = repair_by_rule[truth["rule_id"]]
            classification = repair["repairability"]
            attempt = attempts_by_key.get((truth["rule_id"], truth["entity_path"]), {})
            status = attempt.get("status", "missing")
            per_class[classification]["count"] += 1
            per_class[classification][status] += 1
            per_class[classification]["llm_calls"] += int(attempt.get("llm_call_performed", False))

            if classification in _EXECUTABLE:
                counts["eligible_repair_count"] += 1
                counts["eligible_accepted_count"] += int(status == "accepted")
                proposal = attempt.get("proposal") or {}
                counts["oracle_patch_exact_match_count"] += int(proposal.get("operations") == repair["operations"])
            else:
                counts["noneligible_repair_count"] += 1
                correct_abstention = (
                    status == "abstained"
                    and not attempt.get("llm_call_performed", False)
                    and attempt.get("proposal") is None
                )
                counts["correct_abstention_count"] += int(correct_abstention)

        for attempt in attempts:
            counts["repair_attempt_count"] += 1
            counts["llm_call_count"] += int(attempt["llm_call_performed"])
            counts["accepted_attempt_count"] += int(attempt["status"] == "accepted")
            counts["rejected_attempt_count"] += int(attempt["status"] == "rejected")
            counts["abstained_attempt_count"] += int(attempt["status"] == "abstained")
            counts["invalid_proposal_count"] += int(attempt["llm_call_performed"] and not attempt["diagnostics"]["parse_success"])
            prompt_tokens += int(attempt["ollama_metadata"].get("prompt_eval_count") or 0)
            output_tokens += int(attempt["ollama_metadata"].get("eval_count") or 0)

            if attempt["introduced_findings"]:
                counts["regression_attempt_count"] += 1
                introduced_keys = {(item["rule_id"], item["entity_path"]) for item in attempt["introduced_findings"]}
                counts["regression_rollback_success_count"] += int(not introduced_keys & final_keys)

    result = {
        "scenario_count": counts["scenario_count"],
        "clean_scenario_count": counts["clean_scenario_count"],
        "faulty_scenario_count": counts["faulty_scenario_count"],
        "eligible_scenario_count": counts["eligible_scenario_count"],
        "eligible_repair_count": counts["eligible_repair_count"],
        "noneligible_repair_count": counts["noneligible_repair_count"],
        "repair_attempt_count": counts["repair_attempt_count"],
        "llm_call_count": counts["llm_call_count"],
        "accepted_attempt_count": counts["accepted_attempt_count"],
        "rejected_attempt_count": counts["rejected_attempt_count"],
        "abstained_attempt_count": counts["abstained_attempt_count"],
        "verified_repair_success_rate": _safe_divide(counts["eligible_accepted_count"], counts["eligible_repair_count"]),
        "oracle_patch_exact_match_rate": _safe_divide(counts["oracle_patch_exact_match_count"], counts["eligible_repair_count"]),
        "eligible_exact_restoration_rate": _safe_divide(counts["eligible_exact_restoration_count"], counts["eligible_scenario_count"]),
        "faulty_exact_restoration_rate": _safe_divide(counts["faulty_exact_restoration_count"], counts["faulty_scenario_count"]),
        "correct_abstention_rate": _safe_divide(counts["correct_abstention_count"], counts["noneligible_repair_count"]),
        "clean_preservation_rate": _safe_divide(counts["clean_preserved_count"], counts["clean_scenario_count"]),
        "invalid_proposal_rate": _safe_divide(counts["invalid_proposal_count"], counts["llm_call_count"]),
        "regression_attempt_count": counts["regression_attempt_count"],
        "regression_rollback_success_rate": _conditional_rate(counts["regression_rollback_success_count"], counts["regression_attempt_count"]),
        "sandbox_modified_scenario_count": counts["sandbox_modified_scenario_count"],
        "production_modification_count": counts["production_modification_count"],
        "input_immutability_rate": _safe_divide(counts["input_immutable_count"], counts["scenario_count"]),
        "human_referral_rate": _safe_divide(counts["human_referral_count"], counts["scenario_count"]),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "median": statistics.median(latencies) if latencies else 0.0,
        },
        "per_repairability": {
            name: {
                "count": values["count"], "accepted": values["accepted"], "rejected": values["rejected"],
                "abstained": values["abstained"], "missing": values["missing"], "llm_calls": values["llm_calls"],
            }
            for name, values in sorted(per_class.items())
        },
    }
    return result


def _save_outputs(summary: dict[str, Any], manifest: dict[str, Any], output: Path) -> None:
    (output / "repair_benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "repair_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    fields = (
        "model", "repair_mode", "scenario_count", "eligible_repair_count", "verified_repair_success_rate",
        "oracle_patch_exact_match_rate", "eligible_exact_restoration_rate",
        "correct_abstention_rate", "clean_preservation_rate", "invalid_proposal_rate",
        "regression_attempt_count", "regression_rollback_success_rate",
        "production_modification_count", "median_latency_ms",
    )
    with (output / "repair_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, metrics in summary["models"].items():
            writer.writerow({
                "model": model, "repair_mode": summary["repair_mode"],
                **{field: metrics[field] for field in fields[2:-1]},
                "median_latency_ms": metrics["latency_ms"]["median"],
            })

    rows = []
    for model, metrics in summary["models"].items():
        rollback = metrics["regression_rollback_success_rate"]
        rollback_text = "N/A" if rollback is None else f"{rollback:.3f}"
        rows.append(
            f"| {model} | {metrics['verified_repair_success_rate']:.3f} | "
            f"{metrics['eligible_exact_restoration_rate']:.3f} | "
            f"{metrics['correct_abstention_rate']:.3f} | "
            f"{metrics['clean_preservation_rate']:.3f} | "
            f"{metrics['invalid_proposal_rate']:.3f} | {rollback_text} | "
            f"{metrics['production_modification_count']} |"
        )

    note = f"""# AeroElecBench Verified Repair Results

Repair mode: **{summary['repair_mode']}**; profile: **{summary['profile']}**; scenarios per model: **{summary['scenario_count']}**.

| Model | Verified repair | Eligible exact restoration | Correct abstention | Clean preservation | Invalid proposal | Regression rollback | Production modifications |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Only automatic or constrained findings are sent to the LLM repair proposer. Ambiguous and intent-dependent findings are referred directly for human review. Candidate patches are applied only to a sandbox copy and accepted only after complete deterministic revalidation.

`faulty_exact_restoration_rate` includes intentionally unmodified ambiguous cases and should not be interpreted as the primary repair-success metric. The primary metrics are verified repair success on eligible findings, eligible exact restoration, correct abstention, clean preservation, and regression rollback.

These results use fictional synthetic ECAD artifacts and are not certification evidence.
"""
    (output / "repair_results.md").write_text(note, encoding="utf-8")


def run_repair_experiment(
    scenarios: list[dict[str, Any]],
    catalog: dict[str, Any],
    models: list[str],
    profile: str,
    output_dir: str | Path,
    base_url: str = "http://localhost:11434",
    timeout: float = 300.0,
    seed: int = 2027,
    max_tokens: int = 400,
    repair_mode: str = "llm_direct",
    benchmark_path: str | Path | None = None,
) -> dict[str, Any]:
    if repair_mode not in REPAIR_MODES:
        raise ValueError(f"repair_mode must be one of: {', '.join(REPAIR_MODES)}")
    selected, output = select_scenarios(scenarios, profile), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    client = OllamaClient(base_url, timeout)
    client.ensure_models(models)
    agent = LLMRepairAgent(catalog, client)
    response_path = output / "repair_responses.jsonl"
    existing = _load_rows(response_path)
    scenario_index = {item["scenario_id"]: item for item in selected}
    scenario_hashes = {item["scenario_id"]: _scenario_hash(item) for item in selected}
    total = len(models) * len(selected)
    completed_now = 0

    with response_path.open("a", encoding="utf-8") as handle:
        for model in models:
            for scenario in selected:
                scenario_hash = scenario_hashes[scenario["scenario_id"]]
                key = model, repair_mode, scenario["scenario_id"], scenario_hash
                if key in existing:
                    continue
                print(f"[{completed_now + 1}/{total}] {model} | {repair_mode} | {scenario['scenario_id']}", flush=True)
                started = time.perf_counter()
                report = agent.repair(scenario["design"], model, mode=repair_mode, seed=seed, max_tokens=max_tokens)
                row = {
                    "model": model, "repair_mode": repair_mode, "scenario_id": scenario["scenario_id"],
                    "scenario_sha256": scenario_hash, "category": scenario["category"],
                    "latency_ms": (time.perf_counter() - started) * 1000, "report": report,
                }
                handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                handle.flush()
                existing[key] = row
                completed_now += 1

    grouped = {}
    for model in models:
        model_rows = [
            existing[(model, repair_mode, item["scenario_id"], scenario_hashes[item["scenario_id"]])]
            for item in selected
        ]
        grouped[model] = _evaluate_rows(model_rows, scenario_index)

    summary = {
        "prototype": "AeroElecBench Verified Repair Agent",
        "version": "0.4.0",
        "profile": profile, "repair_mode": repair_mode,
        "scenario_count": len(selected),
        "models": grouped,
        "interpretation": "Controlled synthetic verified-repair study; not certification evidence.",
    }
    benchmark_hash = (
        hashlib.sha256(Path(benchmark_path).read_bytes()).hexdigest()
        if benchmark_path and Path(benchmark_path).exists() else None
    )
    manifest = {
        "seed": seed, "profile": profile, "repair_mode": repair_mode,
        "scenario_ids": [item["scenario_id"] for item in selected],
        "models": models, "temperature": 0, "max_tokens": max_tokens,
        "base_url": base_url, "benchmark_sha256": benchmark_hash,
        "oracle_exposed_to_model": False, "production_modifications_allowed": False,
    }
    _save_outputs(summary, manifest, output)
    return summary
