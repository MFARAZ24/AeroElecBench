from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .catalog import load_catalog
from .impact_benchmark import read_impact_benchmark, write_impact_benchmark
from .impact_graph import build_version_graph, traverse_impact
from .impact_intent import DEFAULT_SEED as DEVELOPMENT_SEED
from .impact_intent import INTENT_CASE_TYPES, generate_intent_benchmark, resolve_candidate_roots, validate_intent_benchmark
from .impact_intent_llm import PIPELINE_VERSION as PROMPT_VERSION
from .impact_intent_llm import SYSTEM_PROMPT

DEFAULT_BENCHMARK = Path("benchmark/v15/impact_intent_heldout_30.jsonl")
DEFAULT_PACKAGE = Path("benchmark/v15/intent_heldout_separated")
DEFAULT_SEED = 15107
CASES_PER_TYPE = 5
BENCHMARK_ID = "AEROELECBENCH-IMPACT-INTENT-HELDOUT-1.5"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_values(inventory: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {item["candidate_id"]: item for item in inventory}
    component, gauge, requirement, pin = (by_id[f"CAND-0{index}"] for index in range(1, 5))
    pin_replace = next(item for item in pin["operations"] if item["path"].endswith("target.pin_id"))
    return {
        "component_id": component["entity_ids"][0], "old_part": component["operations"][0]["before"], "new_part": component["operations"][0]["after"],
        "gauge_wire_id": gauge["entity_ids"][0], "old_gauge": gauge["operations"][0]["before"], "new_gauge": gauge["operations"][0]["after"],
        "requirement_id": requirement["entity_ids"][0],
        "pin_wire_id": next(value for value in pin["entity_ids"] if value.startswith("W-")),
        "old_pin": pin_replace["before"], "new_pin": pin_replace["after"],
    }


def _heldout_request(case_type: str, variant: int, inventory: list[dict[str, Any]], global_index: int) -> tuple[str, list[str], str]:
    v = _candidate_values(inventory)
    if case_type == "single_explicit":
        cases = [
            (f"Limit the assessment to installing part {v['new_part']} on {v['component_id']}; disregard all other recorded edits.", ["CAND-01"]),
            (f"Analyze only the AWG transition on conductor {v['gauge_wire_id']} from {v['old_gauge']} to {v['new_gauge']}; every other revision item is out of scope.", ["CAND-02"]),
            (f"Determine consequences solely for the text revision to requirement {v['requirement_id']}; exclude all physical-design edits.", ["CAND-03"]),
            (f"Trace only rerouting {v['pin_wire_id']} from destination pin {v['old_pin']} to {v['new_pin']}; ignore the remaining changes.", ["CAND-04"]),
            (f"Scope this review exclusively to the wording revision recorded for {v['requirement_id']}; reject every hardware and interconnect edit.", ["CAND-03"]),
        ]
    elif case_type == "single_paraphrased":
        cases = [
            (f"Focus on the swapped equipment at {v['component_id']}; cable sizing, terminal routing, and requirement wording are unrelated.", ["CAND-01"]),
            (f"Review the conductor resizing on {v['gauge_wire_id']} for electrical margin; device, endpoint, and document edits are excluded.", ["CAND-02"]),
            (f"Follow only the updated compliance statement {v['requirement_id']}; hardware and interconnect work is outside this review.", ["CAND-03"]),
            (f"Evaluate the endpoint migration of {v['pin_wire_id']} to {v['new_pin']}; part-number, gauge, and requirement housekeeping is excluded.", ["CAND-04"]),
            ("Assess only the cable whose conductor size changed; equipment, requirement, and terminal revisions are not authorized scope.", ["CAND-02"]),
        ]
    elif case_type == "multi_intent":
        cases = [
            (f"Assess both the equipment substitution at {v['component_id']} and conductor-size change on {v['gauge_wire_id']}; exclude requirement and terminal edits.", ["CAND-01", "CAND-02"]),
            (f"Review the installed-part transition for {v['component_id']} together with modified requirement {v['requirement_id']}; ignore cable and endpoint edits.", ["CAND-01", "CAND-03"]),
            (f"Trace both the gauge adjustment on {v['gauge_wire_id']} and destination-pin move for {v['pin_wire_id']} to {v['new_pin']}; omit component and requirement changes.", ["CAND-02", "CAND-04"]),
            (f"Evaluate the revised {v['requirement_id']} plus the rerouting of {v['pin_wire_id']} from {v['old_pin']} to {v['new_pin']}; no other edit is authorized.", ["CAND-03", "CAND-04"]),
            (f"Analyze the conductor-size revision on {v['gauge_wire_id']} together with updated requirement {v['requirement_id']}; exclude hardware and endpoint edits.", ["CAND-02", "CAND-03"]),
        ]
    elif case_type == "same_entity_distractor":
        cases = [
            (f"For {v['component_id']}, assess only the part-number change from {v['old_part']} to {v['new_part']}; its simultaneous pin reroute is outside scope.", ["CAND-01"]),
            (f"At {v['component_id']}, scope only the {v['pin_wire_id']} target migration from {v['old_pin']} to {v['new_pin']}; do not include the installed-part edit.", ["CAND-04"]),
            (f"Review only the replacement part {v['new_part']} installed on {v['component_id']}, not the connector-end change recorded on that same unit.", ["CAND-01"]),
            (f"Follow only the destination reassignment of {v['pin_wire_id']} to {v['new_pin']} on {v['component_id']}; the co-located part substitution is unrelated.", ["CAND-04"]),
            (f"Limit impact analysis at {v['component_id']} to moving {v['pin_wire_id']} from {v['old_pin']} to {v['new_pin']} and reject the same-unit hardware substitution.", ["CAND-04"]),
        ]
    elif case_type == "ambiguous_request":
        cases = [
            ("Review the accepted revision and identify what it affects.", []),
            ("Determine the downstream consequences of the authorized ECAD work.", []),
            ("Assess the approved modification package.", []),
            ("Trace the sanctioned electrical-design update.", []),
            ("Report impacts associated with the latest authorized changes.", []),
        ]
    else:
        suffix = f"{global_index:03d}"
        cases = [
            (f"Assess only installation of part HELDOUT-NONPART-{suffix} on LRU-ABSENT-{suffix}; no observed alternative is authorized.", []),
            (f"Evaluate only a gauge change on missing conductor W-ABSENT-{suffix} from 30 to 28 AWG; reject every recorded edit.", []),
            (f"Trace only the revision to nonexistent requirement REQ-ABSENT-{suffix}; physical changes are outside scope.", []),
            (f"Analyze only rerouting W-ABSENT-{suffix} from pin OLD-X to NEW-X; no other endpoint edit is requested.", []),
            (f"Review exclusively the unobserved replacement of LRU-MISSING-{suffix} with HELDOUT-UNKNOWN-{suffix}.", []),
        ]
    text, selected = cases[variant]
    return text, selected, "report" if selected else "abstain"


def generate_intent_heldout(seed: int = DEFAULT_SEED, cases_per_type: int = CASES_PER_TYPE) -> list[dict[str, Any]]:
    if cases_per_type != CASES_PER_TYPE:
        raise ValueError(f"The frozen held-out configuration requires exactly {CASES_PER_TYPE} cases per type")
    scenarios = generate_intent_benchmark(seed, cases_per_type, "heldout")
    for global_index, scenario in enumerate(scenarios):
        case_type = scenario["intent_case_type"]
        variant = global_index % cases_per_type
        request, intended_ids, expected_action = _heldout_request(case_type, variant, scenario["change_inventory"], global_index)
        roots = resolve_candidate_roots(scenario["before_design"], scenario["after_design"], scenario["change_inventory"], intended_ids)
        if intended_ids:
            graph = build_version_graph(scenario["before_design"], scenario["after_design"])
            affected, paths = traverse_impact(graph, roots, scenario["engineering_change_request"]["max_depth"])
        else:
            affected, paths = [], []
        scenario["scenario_id"] = f"v15-heldout-{case_type}-{variant:03d}"
        scenario["engineering_change_request"] = {"request_id": f"HECR-{global_index + 1:04d}", "text": request, "max_depth": 3}
        scenario["intent_oracle"] = {"version": "1.5", "expected_action": expected_action, "intended_candidate_ids": intended_ids, "root_node_ids": roots}
        scenario["impact_oracle"] = {"version": "1.5", "expected_action": expected_action, "affected_node_ids": affected, "impact_paths": paths}
        scenario["split"] = "heldout"
    return scenarios


def _development_request_texts() -> set[str]:
    return {item["engineering_change_request"]["text"] for item in generate_intent_benchmark(DEVELOPMENT_SEED, 4, "development")}


def prepare_intent_heldout(
    benchmark_path: str | Path = DEFAULT_BENCHMARK, catalog_path: str | Path | None = None,
    source_registry_path: str | Path = "data/source_registry.json", seed: int = DEFAULT_SEED,
    cases_per_type: int = CASES_PER_TYPE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark, manifest_path, registry_path = Path(benchmark_path), Path(benchmark_path).with_name("manifest.json"), Path(source_registry_path)
    if benchmark.exists() or manifest_path.exists():
        if not benchmark.exists() or not manifest_path.exists():
            raise ValueError("Incomplete v1.5 held-out freeze; benchmark and manifest are both required")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("benchmark_id") != BENCHMARK_ID or manifest.get("benchmark_sha256") != _sha256(benchmark) or manifest.get("seed") != seed or manifest.get("cases_per_type") != cases_per_type:
            raise ValueError("Existing v1.5 held-out benchmark does not match the frozen configuration")
        scenarios, catalog = read_impact_benchmark(benchmark), load_catalog(catalog_path)
        validate_intent_benchmark(scenarios, catalog["rules"], "heldout")
        return scenarios, manifest
    if not registry_path.exists():
        raise FileNotFoundError(f"Source registry not found: {registry_path}")
    catalog = load_catalog(catalog_path)
    scenarios = generate_intent_heldout(seed, cases_per_type)
    validate_intent_benchmark(scenarios, catalog["rules"], "heldout")
    request_texts = {item["engineering_change_request"]["text"] for item in scenarios}
    overlap = request_texts & _development_request_texts()
    if overlap:
        raise ValueError("Held-out requests overlap development requests")
    write_impact_benchmark(scenarios, benchmark)
    target_counts = Counter(candidate_id for item in scenarios for candidate_id in item["intent_oracle"]["intended_candidate_ids"])
    manifest = {
        "benchmark_id": BENCHMARK_ID, "benchmark": benchmark.name, "benchmark_sha256": _sha256(benchmark),
        "catalog_id": catalog["catalog_id"], "catalog_version": catalog["version"], "catalog_sha256": _sha256(Path(catalog_path or "data/rules.json")),
        "source_registry_id": json.loads(registry_path.read_text(encoding="utf-8"))["registry_id"], "source_registry_sha256": _sha256(registry_path),
        "seed": seed, "cases_per_type": cases_per_type, "scenario_count": len(scenarios),
        "case_type_counts": dict(sorted(Counter(item["intent_case_type"] for item in scenarios).items())),
        "intended_candidate_counts": dict(sorted(target_counts.items())), "split_counts": {"heldout": len(scenarios)},
        "minimum_concurrent_semantic_changes": 4, "oracle_validation_rate": 1.0,
        "exact_development_request_overlap_count": 0, "heldout_request_template_count": len(request_texts),
        "prompt_version_frozen": PROMPT_VERSION, "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "prompt_frozen_before_model_run": True, "frozen_before_model_run": True,
        "intent_oracle_exposed_to_model": False, "impact_oracle_exposed_to_model": False,
        "development_only": False, "heldout": True, "posthoc_tuning_allowed": False,
        "dataset_kind": "fictional_synthetic", "production_modifications_allowed": False, "narrative_output": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return scenarios, manifest


def load_frozen_intent_heldout(benchmark_path: str | Path = DEFAULT_BENCHMARK, catalog_path: str | Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark, manifest_path = Path(benchmark_path), Path(benchmark_path).with_name("manifest.json")
    if not benchmark.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Frozen v1.5 held-out benchmark not found: {benchmark}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("benchmark_id") != BENCHMARK_ID or manifest.get("benchmark_sha256") != _sha256(benchmark):
        raise ValueError("Held-out benchmark identity or hash mismatch")
    if manifest.get("prompt_version_frozen") != PROMPT_VERSION or not manifest.get("heldout") or manifest.get("posthoc_tuning_allowed") is not False:
        raise ValueError("Held-out freeze metadata is invalid")
    scenarios, catalog = read_impact_benchmark(benchmark), load_catalog(catalog_path)
    validate_intent_benchmark(scenarios, catalog["rules"], "heldout")
    return scenarios, manifest
