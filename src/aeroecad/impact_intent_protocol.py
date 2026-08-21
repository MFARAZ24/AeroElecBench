from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from .impact_graph import build_version_graph, traverse_impact
from .impact_intent import BENCHMARK_ID as DEVELOPMENT_BENCHMARK_ID
from .impact_intent import DEFAULT_BENCHMARK, _lexical_selection, evaluate_intent_predictions, load_frozen_intent_benchmark, resolve_candidate_roots
from .impact_intent_llm import MODE, PIPELINE_VERSION as PROMPT_VERSION, RESPONSE_SCHEMA, SYSTEM_PROMPT, build_intent_prompt, parse_intent_response
from .ollama import OllamaClient

PROTOCOL_VERSION = "1.0.0"
PIPELINE_VERSION = "1.4.0"
MODEL_PROVENANCE_VERSION = "1.8.0"
DEFAULT_PACKAGE = Path("benchmark/v14/intent_development_separated")
DEFAULT_PREDICTIONS = Path("results/impact_v14_intent_predictions")
DEFAULT_SCORES = Path("results/impact_v14_intent_scores")
DEFAULT_BASELINE_PREDICTIONS = Path("results/impact_v16_intent_baseline_predictions")
DEFAULT_BASELINE_SCORES = Path("results/impact_v16_intent_baseline_scores")
INPUT_FILE = "model_inputs.jsonl"
ORACLE_FILE = "oracle_reference.jsonl"
PACKAGE_MANIFEST = "package_manifest.json"
PREDICTION_FILE = "intent_predictions.jsonl"
PREDICTION_MANIFEST = "prediction_manifest.json"
BASELINE_PREDICTION_FILE = "baseline_predictions.jsonl"
BASELINE_PREDICTION_MANIFEST = "baseline_prediction_manifest.json"
ORACLE_FREE_BASELINE_MODES = ("all_diff_graph", "lexical_intent_graph")
FORBIDDEN_INPUT_KEYS = frozenset({"intent_case_type", "intent_oracle", "impact_oracle", "expected_action", "intended_candidate_ids", "root_node_ids", "affected_node_ids", "impact_paths", "split", "request_id", "source_scenario_id"})
REQUIRED_INPUT_KEYS = frozenset({"scenario_id", "before_design", "after_design", "engineering_change_request", "change_inventory"})


def intent_mode_for_model(model: str) -> str:
    """Return a stable result mode without changing the frozen Qwen identity."""
    if model == "qwen2.5:7b":
        return MODE
    slug = re.sub(r"[^a-z0-9]+", "_", model.strip().lower()).strip("_")
    if not slug:
        raise ValueError("Model name must contain at least one letter or digit")
    return f"{slug}_intent_graph"


def _model_token(model: str) -> str:
    return intent_mode_for_model(model).removesuffix("_intent_graph").upper()


def _prediction_id(package_id: str, model: str) -> str:
    if model == "qwen2.5:7b":
        return f"{package_id}-QWEN-PREDICTIONS-1.4"
    return f"{package_id}-{_model_token(model)}-PREDICTIONS-1.8"


def _evaluation_id(package_id: str, model: str) -> str:
    if model == "qwen2.5:7b":
        return f"{package_id}-OFFLINE-SCORE-1.4"
    return f"{package_id}-{_model_token(model)}-OFFLINE-SCORE-1.8"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSONL file not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Invalid or empty JSONL file: {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8"))


def _forbidden_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return (set(value) & FORBIDDEN_INPUT_KEYS) | set().union(*(_forbidden_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_forbidden_keys(item) for item in value), set())
    return set()


def _validate_inputs(rows: list[dict[str, Any]]) -> None:
    ids = [row.get("scenario_id") for row in rows]
    if len(ids) != len(set(ids)) or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("Oracle-free scenario identifiers must be unique non-empty strings")
    for row in rows:
        leaked = _forbidden_keys(row)
        if leaked:
            raise ValueError(f"Oracle fields found in model input {row.get('scenario_id')}: {', '.join(sorted(leaked))}")
        if set(row) != REQUIRED_INPUT_KEYS:
            raise ValueError(f"Unexpected model-input schema in {row['scenario_id']}")
        candidates = row["change_inventory"]
        candidate_ids = [item.get("candidate_id") for item in candidates]
        if not candidates or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"Invalid candidate inventory in {row['scenario_id']}")


def prepare_oracle_separated_package(
    benchmark_path: str | Path = DEFAULT_BENCHMARK, package_dir: str | Path = DEFAULT_PACKAGE,
    catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(benchmark_path).with_name("manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")
    benchmark_id = json.loads(manifest_path.read_text(encoding="utf-8")).get("benchmark_id")
    if benchmark_id == DEVELOPMENT_BENCHMARK_ID:
        scenarios, benchmark_manifest = load_frozen_intent_benchmark(benchmark_path, catalog_path)
    else:
        from .impact_intent_heldout_v15 import BENCHMARK_ID as HELDOUT_BENCHMARK_ID
        from .impact_intent_heldout_v15 import load_frozen_intent_heldout
        if benchmark_id != HELDOUT_BENCHMARK_ID:
            raise ValueError(f"Unsupported intent benchmark id: {benchmark_id}")
        scenarios, benchmark_manifest = load_frozen_intent_heldout(benchmark_path, catalog_path)
    source_hash = benchmark_manifest["benchmark_sha256"]
    scenarios = sorted(scenarios, key=lambda item: hashlib.sha256(f"{source_hash}:{item['scenario_id']}".encode()).hexdigest())
    inputs, oracles = [], []
    for index, scenario in enumerate(scenarios, start=1):
        opaque_id = f"INTENT-CASE-{index:04d}"
        inputs.append({
            "scenario_id": opaque_id, "before_design": scenario["before_design"], "after_design": scenario["after_design"],
            "engineering_change_request": {key: scenario["engineering_change_request"][key] for key in ("text", "max_depth")},
            "change_inventory": scenario["change_inventory"],
        })
        oracles.append({
            "scenario_id": opaque_id, "source_scenario_id": scenario["scenario_id"], "intent_case_type": scenario["intent_case_type"],
            "intent_oracle": scenario["intent_oracle"], "impact_oracle": scenario["impact_oracle"],
        })
    _validate_inputs(inputs)
    package = Path(package_dir)
    input_path, oracle_path = package / INPUT_FILE, package / ORACLE_FILE
    _write_jsonl(input_path, inputs); _write_jsonl(oracle_path, oracles)
    split = "heldout" if benchmark_manifest.get("heldout") else "development"
    manifest = {
        "package_id": f"AEROELECBENCH-IMPACT-INTENT-ORACLE-SEPARATED-{split.upper()}-1.0",
        "protocol_version": PROTOCOL_VERSION, "source_benchmark_id": benchmark_manifest["benchmark_id"],
        "source_benchmark_sha256": benchmark_manifest["benchmark_sha256"], "split": split,
        "scenario_count": len(inputs), "model_input_file": INPUT_FILE, "model_input_sha256": _sha256(input_path),
        "oracle_file": ORACLE_FILE, "oracle_sha256": _sha256(oracle_path),
        "model_input_top_level_fields": sorted(REQUIRED_INPUT_KEYS), "forbidden_model_input_fields": sorted(FORBIDDEN_INPUT_KEYS),
        "oracle_fields_present_in_model_input": False, "case_type_present_in_model_input": False,
        "opaque_scenario_ids": True, "source_scenario_ids_present_in_model_input": False, "case_order_blinded": True,
        "frozen_prompt_version": PROMPT_VERSION, "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "posthoc_tuning_allowed": benchmark_manifest.get("posthoc_tuning_allowed", False),
        "prediction_and_scoring_commands_separate": True, "production_modifications_allowed": False,
    }
    (package / PACKAGE_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_oracle_free_package(package_dir: str | Path = DEFAULT_PACKAGE) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    package = Path(package_dir)
    manifest_path = package / PACKAGE_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"Package manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported oracle-separation protocol version")
    input_path = package / manifest["model_input_file"]
    if _sha256(input_path) != manifest.get("model_input_sha256"):
        raise ValueError("Oracle-free model-input hash mismatch")
    rows = _read_jsonl(input_path); _validate_inputs(rows)
    if len(rows) != manifest.get("scenario_count"):
        raise ValueError("Oracle-free model-input scenario count mismatch")
    return rows, manifest


def _graph_result(scenario: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    selected = selection["selected_candidate_ids"]
    if selection["action"] != "report" or not selected:
        return {"status": "rejected" if selection["action"] == "rejected" else "abstained", "root_node_ids": [], "affected_node_ids": [], "impact_paths": []}
    roots = resolve_candidate_roots(scenario["before_design"], scenario["after_design"], scenario["change_inventory"], selected)
    graph = build_version_graph(scenario["before_design"], scenario["after_design"])
    affected, paths = traverse_impact(graph, roots, scenario["engineering_change_request"]["max_depth"])
    return {"status": "completed", "root_node_ids": roots, "affected_node_ids": affected, "impact_paths": paths}


def _read_predictions(path: Path, model: str, manifest: dict[str, Any], allowed_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = _read_jsonl(path)
    if len(rows) != len({row.get("scenario_id") for row in rows}):
        raise ValueError(f"Duplicate prediction records in {path}")
    expected_mode = intent_mode_for_model(model)
    for row in rows:
        if row.get("pipeline_version") != PIPELINE_VERSION or row.get("prompt_version") != PROMPT_VERSION or row.get("model") != model or row.get("mode") != expected_mode or row.get("package_id") != manifest["package_id"] or row.get("model_input_sha256") != manifest["model_input_sha256"] or row.get("scenario_id") not in allowed_ids:
            raise ValueError(f"Existing prediction provenance mismatch in {path}")
    return {row["scenario_id"]: row for row in rows}


def run_oracle_free_predictions(
    model: str = "qwen2.5:7b", package_dir: str | Path = DEFAULT_PACKAGE,
    output_dir: str | Path = DEFAULT_PREDICTIONS, base_url: str = "http://localhost:11434",
    timeout: float = 900.0, seed: int = 11107, max_tokens: int = 300,
    client: OllamaClient | None = None,
) -> dict[str, Any]:
    scenarios, package_manifest = load_oracle_free_package(package_dir)
    mode = intent_mode_for_model(model)
    output, record_path = Path(output_dir), Path(output_dir) / PREDICTION_FILE
    runtime = client or OllamaClient(base_url, timeout); runtime.ensure_models([model])
    existing = _read_predictions(record_path, model, package_manifest, {item["scenario_id"] for item in scenarios})
    completed = sum(int(row["llm_call_count"]) for row in existing.values())
    print(f"[intent-predict] oracle-free scenarios={len(scenarios)} records={len(existing)}/{len(scenarios)} model={model} calls={completed}/{len(scenarios)}; remaining={len(scenarios) - completed}", flush=True)
    for index, scenario in enumerate(scenarios, start=1):
        if scenario["scenario_id"] in existing:
            continue
        print(f"[intent-predict] scenario {index}/{len(scenarios)} {scenario['scenario_id']} - starting {model} call {completed + 1}/{len(scenarios)}; remaining={len(scenarios) - completed}", flush=True)
        started, prompt = time.perf_counter(), build_intent_prompt(scenario)
        response = runtime.chat(model, SYSTEM_PROMPT, prompt, seed=seed, max_tokens=max_tokens, response_schema=RESPONSE_SCHEMA)
        selection, diagnostics = parse_intent_response(response.content, scenario)
        record = {
            "scenario_id": scenario["scenario_id"], "model": model, "mode": mode,
            "pipeline_version": PIPELINE_VERSION, "prompt_version": PROMPT_VERSION,
            "package_id": package_manifest["package_id"], "model_input_sha256": package_manifest["model_input_sha256"],
            "llm_call_count": 1, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "selection": selection, "graph_result": _graph_result(scenario, selection), "diagnostics": diagnostics,
            "raw_response": response.content, "ollama_metadata": response.metadata,
            "oracle_file_read": False, "oracle_fields_available_to_predictor": False,
            "production_modification_performed": False, "input_designs_unchanged": True,
        }
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        existing[scenario["scenario_id"]] = record; completed += 1
        print(f"[intent-predict] completed {len(existing)}/{len(scenarios)} action={selection['action']} candidates={len(selection['selected_candidate_ids'])} elapsed={time.perf_counter() - started:.1f}s; remaining={len(scenarios) - completed}", flush=True)
    manifest = {
        "prediction_id": _prediction_id(package_manifest["package_id"], model),
        "pipeline_version": PIPELINE_VERSION, "prompt_version": PROMPT_VERSION, "package_id": package_manifest["package_id"],
        "model_input_sha256": package_manifest["model_input_sha256"], "prediction_file": PREDICTION_FILE,
        "prediction_sha256": _sha256(record_path), "scenario_count": len(scenarios), "model": model, "mode": mode,
        "model_provenance_version": MODEL_PROVENANCE_VERSION,
        "seed": seed, "temperature": 0, "max_tokens": max_tokens, "llm_call_count": completed,
        "oracle_file_read": False, "oracle_fields_available_to_predictor": False,
        "intent_oracle_exposed_to_model": False, "impact_oracle_exposed_to_model": False,
        "metrics_generated": False, "expected_labels_written": False, "oracle_correction_performed": False,
        "prediction_complete": True, "production_modifications_allowed": False,
    }
    (output / PREDICTION_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def score_frozen_predictions(
    package_dir: str | Path = DEFAULT_PACKAGE, prediction_dir: str | Path = DEFAULT_PREDICTIONS,
    output_dir: str | Path = DEFAULT_SCORES,
) -> dict[str, Any]:
    inputs, package_manifest = load_oracle_free_package(package_dir)
    package, predictions = Path(package_dir), Path(prediction_dir)
    oracle_path, record_path = package / package_manifest["oracle_file"], predictions / PREDICTION_FILE
    if _sha256(oracle_path) != package_manifest.get("oracle_sha256"):
        raise ValueError("Oracle-reference hash mismatch")
    prediction_manifest_path = predictions / PREDICTION_MANIFEST
    if not prediction_manifest_path.exists():
        raise FileNotFoundError(f"Completed prediction manifest not found: {prediction_manifest_path}")
    prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
    prediction_hash = _sha256(record_path)
    if prediction_manifest.get("prediction_sha256") != prediction_hash or prediction_manifest.get("package_id") != package_manifest["package_id"] or prediction_manifest.get("model_input_sha256") != package_manifest["model_input_sha256"] or not prediction_manifest.get("prediction_complete"):
        raise ValueError("Frozen prediction identity or hash mismatch")
    oracle_rows = _read_jsonl(oracle_path)
    oracle_by_id = {row["scenario_id"]: row for row in oracle_rows}
    if len(oracle_by_id) != len(oracle_rows) or set(oracle_by_id) != {row["scenario_id"] for row in inputs}:
        raise ValueError("Oracle-reference scenario identity mismatch")
    model = prediction_manifest.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Frozen prediction model is missing")
    mode = intent_mode_for_model(model)
    if prediction_manifest.get("mode", mode) != mode:
        raise ValueError("Frozen prediction mode does not match model identity")
    records_by_id = _read_predictions(record_path, model, package_manifest, set(oracle_by_id))
    if len(records_by_id) != len(inputs) or set(records_by_id) != set(oracle_by_id):
        raise ValueError("Prediction scenario identity mismatch")
    records = list(records_by_id.values())
    scenarios = [{**row, **oracle_by_id[row["scenario_id"]]} for row in inputs]
    selections = {row["scenario_id"]: row["selection"] for row in records}
    rejected = sum(row["selection"]["action"] == "rejected" for row in records)
    metrics, rows = evaluate_intent_predictions(scenarios, selections, mode, prediction_manifest["llm_call_count"], rejected)
    aggregate = {
        "evaluation_id": _evaluation_id(package_manifest["package_id"], model), "pipeline_version": PIPELINE_VERSION,
        "model_provenance_version": prediction_manifest.get("model_provenance_version", PIPELINE_VERSION),
        "model": model, "mode": mode, "scenario_count": len(scenarios), "modes": {mode: metrics},
    }
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    with (output / "evaluation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics)); writer.writeheader(); writer.writerow(metrics)
    with (output / "scenario_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    score_manifest = {
        "evaluation_id": aggregate["evaluation_id"], "pipeline_version": PIPELINE_VERSION,
        "model_provenance_version": aggregate["model_provenance_version"], "model": model, "mode": mode,
        "package_id": package_manifest["package_id"], "model_input_sha256": package_manifest["model_input_sha256"],
        "oracle_sha256": package_manifest["oracle_sha256"], "prediction_sha256": prediction_hash,
        "prediction_file_unchanged_during_scoring": _sha256(record_path) == prediction_hash,
        "inference_and_scoring_processes_separate": True, "oracle_loaded_only_by_scoring_command": True,
        "oracle_correction_performed": False, "metric_generation": "deterministic_offline",
        "production_modifications_allowed": False,
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(score_manifest, indent=2) + "\n", encoding="utf-8")
    return aggregate


def _baseline_selection(mode: str, scenario: dict[str, Any]) -> dict[str, Any]:
    selected = [item["candidate_id"] for item in scenario["change_inventory"]] if mode == "all_diff_graph" else _lexical_selection(scenario)
    return {"action": "report" if selected else "abstain", "selected_candidate_ids": selected}


def run_oracle_free_baselines(
    package_dir: str | Path = DEFAULT_PACKAGE, output_dir: str | Path = DEFAULT_BASELINE_PREDICTIONS,
) -> dict[str, Any]:
    scenarios, package_manifest = load_oracle_free_package(package_dir)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    record_path = output / BASELINE_PREDICTION_FILE
    records = [
        {
            "scenario_id": scenario["scenario_id"], "mode": mode, "selection": _baseline_selection(mode, scenario),
            "package_id": package_manifest["package_id"], "model_input_sha256": package_manifest["model_input_sha256"],
            "oracle_file_read": False, "oracle_fields_available_to_predictor": False, "llm_call_count": 0,
        }
        for mode in ORACLE_FREE_BASELINE_MODES for scenario in scenarios
    ]
    _write_jsonl(record_path, records)
    manifest = {
        "prediction_id": f"{package_manifest['package_id']}-DETERMINISTIC-BASELINES-1.0",
        "protocol_version": PROTOCOL_VERSION, "package_id": package_manifest["package_id"],
        "model_input_sha256": package_manifest["model_input_sha256"], "prediction_file": BASELINE_PREDICTION_FILE,
        "prediction_sha256": _sha256(record_path), "scenario_count": len(scenarios), "record_count": len(records),
        "modes": list(ORACLE_FREE_BASELINE_MODES), "llm_calls_performed": 0, "oracle_file_read": False,
        "oracle_fields_available_to_predictor": False, "metrics_generated": False,
        "prediction_complete": True, "production_modifications_allowed": False,
    }
    (output / BASELINE_PREDICTION_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def score_frozen_baselines(
    package_dir: str | Path = DEFAULT_PACKAGE, prediction_dir: str | Path = DEFAULT_BASELINE_PREDICTIONS,
    output_dir: str | Path = DEFAULT_BASELINE_SCORES,
) -> dict[str, Any]:
    inputs, package_manifest = load_oracle_free_package(package_dir)
    package, predictions = Path(package_dir), Path(prediction_dir)
    oracle_path = package / package_manifest["oracle_file"]
    if _sha256(oracle_path) != package_manifest.get("oracle_sha256"):
        raise ValueError("Oracle-reference hash mismatch")
    manifest_path = predictions / BASELINE_PREDICTION_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"Completed baseline prediction manifest not found: {manifest_path}")
    prediction_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_path = predictions / BASELINE_PREDICTION_FILE
    prediction_hash = _sha256(record_path)
    if (
        prediction_manifest.get("prediction_sha256") != prediction_hash
        or prediction_manifest.get("package_id") != package_manifest["package_id"]
        or prediction_manifest.get("model_input_sha256") != package_manifest["model_input_sha256"]
        or prediction_manifest.get("modes") != list(ORACLE_FREE_BASELINE_MODES)
        or not prediction_manifest.get("prediction_complete")
    ):
        raise ValueError("Frozen baseline prediction identity or hash mismatch")
    oracle_rows = _read_jsonl(oracle_path); oracle_by_id = {row["scenario_id"]: row for row in oracle_rows}
    input_ids = {row["scenario_id"] for row in inputs}
    if len(oracle_by_id) != len(oracle_rows) or set(oracle_by_id) != input_ids:
        raise ValueError("Oracle-reference scenario identity mismatch")
    records = _read_jsonl(record_path)
    expected_pairs = {(scenario_id, mode) for scenario_id in input_ids for mode in ORACLE_FREE_BASELINE_MODES}
    record_pairs = [(row.get("scenario_id"), row.get("mode")) for row in records]
    if len(record_pairs) != len(set(record_pairs)) or set(record_pairs) != expected_pairs:
        raise ValueError("Baseline prediction scenario or mode identity mismatch")
    scenarios = [{**row, **oracle_by_id[row["scenario_id"]]} for row in inputs]
    metrics_by_mode, rows = {}, []
    for mode in ORACLE_FREE_BASELINE_MODES:
        selections = {row["scenario_id"]: row["selection"] for row in records if row["mode"] == mode}
        metrics_by_mode[mode], mode_rows = evaluate_intent_predictions(scenarios, selections, mode)
        rows.extend(mode_rows)
    oracle_selections = {
        row["scenario_id"]: {
            "action": row["intent_oracle"]["expected_action"],
            "selected_candidate_ids": row["intent_oracle"]["intended_candidate_ids"],
        }
        for row in scenarios
    }
    metrics_by_mode["oracle_root_graph"], oracle_rows = evaluate_intent_predictions(scenarios, oracle_selections, "oracle_root_graph")
    rows.extend(oracle_rows)
    aggregate = {
        "evaluation_id": f"{package_manifest['package_id']}-DETERMINISTIC-BASELINE-SCORE-1.0",
        "pipeline_version": PIPELINE_VERSION, "scenario_count": len(scenarios), "modes": metrics_by_mode,
    }
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    fields = list(next(iter(metrics_by_mode.values())))
    with (output / "evaluation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(metrics_by_mode.values())
    with (output / "scenario_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    score_manifest = {
        "evaluation_id": aggregate["evaluation_id"], "pipeline_version": PIPELINE_VERSION,
        "package_id": package_manifest["package_id"], "model_input_sha256": package_manifest["model_input_sha256"],
        "oracle_sha256": package_manifest["oracle_sha256"], "baseline_prediction_sha256": prediction_hash,
        "baseline_prediction_file_unchanged_during_scoring": _sha256(record_path) == prediction_hash,
        "prediction_and_scoring_processes_separate": True, "oracle_loaded_only_by_scoring_command": True,
        "intent_oracle_exposed_to_non_oracle_baselines": False, "oracle_root_graph_role": "upper_bound",
        "all_diff_graph_role": "intent_agnostic_control", "lexical_intent_graph_role": "non_llm_intent_baseline",
        "llm_calls_performed": 0, "metric_generation": "deterministic_offline", "production_modifications_allowed": False,
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(score_manifest, indent=2) + "\n", encoding="utf-8")
    return aggregate
