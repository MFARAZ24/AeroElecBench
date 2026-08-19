from __future__ import annotations

import json
import tempfile
from pathlib import Path

from aeroecad.impact_intent import generate_intent_benchmark, prepare_intent_development
from aeroecad.impact_intent_llm import build_intent_prompt, parse_intent_response, run_intent_qwen
from aeroecad.ollama import OllamaResponse


class FakeIntentClient:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[dict[str, object]] = []

    def ensure_models(self, requested: list[str]) -> list[str]:
        return requested

    def chat(self, model: str, system: str, user: str, **_: object) -> OllamaResponse:
        self.calls += 1
        payload = json.loads(user)
        self.prompts.append(payload)
        request = payload["engineering_change_request"]
        if "authorized replacement" in request:
            action, selected = "report", ["CAND-01"]
        elif "conductor resized" in request:
            action, selected = "report", ["CAND-02"]
        elif "coordinated hardware" in request:
            action, selected = "report", ["CAND-01", "CAND-03"]
        elif "target-pin reassignment" in request:
            action, selected = "report", ["CAND-04"]
        else:
            action, selected = "abstain", []
        content = json.dumps({"action": action, "selected_candidate_ids": selected, "rationale": "Grounded only to requested observed candidates."})
        return OllamaResponse(content, {"eval_count": 12, "done_reason": "stop"})


def test_intent_prompt_uses_inventory_parity_and_hides_versions_and_oracles() -> None:
    scenario = generate_intent_benchmark(seed=107, cases_per_type=1)[0]
    prompt = build_intent_prompt(scenario)
    payload = json.loads(prompt)
    assert {"engineering_change_request", "candidate_inventory", "decision_rules", "generic_examples"} <= set(payload)
    assert "before_design" not in payload and "after_design" not in payload
    assert len(prompt) < 5000
    assert "intent_oracle" not in prompt and "impact_oracle" not in prompt and "root_node_ids" not in prompt


def test_intent_parser_rejects_unknown_candidates_and_inconsistent_actions() -> None:
    scenario = generate_intent_benchmark(seed=109, cases_per_type=1)[0]
    unknown = json.dumps({"action": "report", "selected_candidate_ids": ["CAND-99"], "rationale": "Invented."})
    inconsistent = json.dumps({"action": "abstain", "selected_candidate_ids": ["CAND-01"], "rationale": "Contradictory."})
    selection, diagnostics = parse_intent_response(unknown, scenario)
    assert selection["action"] == "rejected" and diagnostics["error"] == "unknown_candidate_id"
    selection, diagnostics = parse_intent_response(inconsistent, scenario)
    assert selection["action"] == "rejected" and diagnostics["error"] == "inconsistent_decision"


def test_qwen_intent_selection_controls_graph_and_resumes(capsys) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        benchmark, output = root / "benchmark" / "intent.jsonl", root / "results"
        prepare_intent_development(benchmark, cases_per_type=2)
        client = FakeIntentClient()
        smoke = run_intent_qwen(benchmark_path=benchmark, output_dir=output, profile="smoke", client=client)
        progress, smoke_calls = capsys.readouterr().out, client.calls
        assert smoke["scenario_count"] == 6 and smoke_calls == 6
        assert "Qwen calls=0/6; remaining=6" in progress and "Qwen remaining=0" in progress
        assert smoke["modes"]["qwen_intent_graph"]["candidate_f1"] == 1.0
        assert smoke["modes"]["qwen_intent_graph"]["impact_set_f1"] == 1.0
        run_intent_qwen(benchmark_path=benchmark, output_dir=output, profile="smoke", client=client)
        assert client.calls == smoke_calls
        development = run_intent_qwen(benchmark_path=benchmark, output_dir=output, profile="development", client=client)
        assert client.calls == 12 and development["scenario_count"] == 12
        assert development["modes"]["qwen_intent_graph"]["candidate_exact_scenario_accuracy"] == 1.0
        records = [json.loads(line) for line in (output / "intent_records.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(records) == 12 and all(row["llm_call_count"] == 1 for row in records)
        assert all(row["graph_result"]["affected_node_ids"] for row in records if row["selection"]["action"] == "report")
        manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
        assert manifest["oracle_usage"] == "offline_scoring_only" and manifest["oracle_correction_performed"] is False
        assert manifest["impact_result_source"] == "deterministic_graph_from_qwen_selected_candidates"
        assert manifest["before_after_exposed_to_model"] is False and manifest["comparison_input_parity_with_lexical_baseline"] is True
        assert not list(output.rglob("*.md"))
