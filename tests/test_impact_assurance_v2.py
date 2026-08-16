from __future__ import annotations

import json
import tempfile
from pathlib import Path

from aeroecad.impact_assurance_v2 import run_assurance_v2
from aeroecad.impact_evaluation import prepare_impact_development
from aeroecad.ollama import OllamaResponse


class FakeAssuranceV2Client:
    def __init__(self, corrupt_candidate: bool = False) -> None:
        self.calls = 0
        self.corrupt_candidate = corrupt_candidate
        self.prompts: list[str] = []

    def ensure_models(self, requested: list[str]) -> list[str]:
        return requested

    def chat(self, model: str, system: str, user: str, **_: object) -> OllamaResponse:
        self.calls += 1
        self.prompts.append(user)
        payload = json.loads(user)
        if "tool controller" in system:
            response = {"decision": payload["permitted_decisions"][0], "rationale": "Follow the bounded state policy."}
        else:
            traversal = payload["tool_observations"]["traverse_dependencies"]
            nodes = [] if self.corrupt_candidate else traversal["affected_node_ids"]
            edges = set()
            if not self.corrupt_candidate:
                for path in traversal["impact_paths"]:
                    for index, relation in enumerate(path["relations"]):
                        edges.add((path["node_ids"][index], path["node_ids"][index + 1], relation))
            response = {
                "action": "report", "affected_node_ids": nodes,
                "impact_edges": [{"source": source, "target": target, "relation": relation} for source, target, relation in sorted(edges)],
                "rationale": "Candidate synthesized from tool observations.",
            }
        return OllamaResponse(json.dumps(response), {"eval_count": 1})


def test_assurance_v2_scores_candidate_and_final_separately_and_resumes(capsys) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        benchmark, output = root / "benchmark" / "impact.jsonl", root / "results"
        prepare_impact_development(benchmark)
        client = FakeAssuranceV2Client()
        metrics = run_assurance_v2(benchmark_path=benchmark, output_dir=output, profile="smoke", client=client)
        progress, first_calls = capsys.readouterr().out, client.calls
        assert metrics["scenario_count"] == 6 and first_calls == 29
        assert metrics["stages"]["candidate"]["impact_set_f1"] == 1.0
        assert metrics["stages"]["final"]["impact_set_f1"] == 1.0
        assert metrics["validation"]["accepted_count"] == 6 and metrics["validation"]["intervention_rate"] == 0.0
        assert metrics["tools"]["analysis_tool_execution_count"] == 23
        assert metrics["tools"]["validator_execution_count"] == 6
        assert metrics["tools"]["tool_order_violation_count"] == 0
        assert "Qwen calls=0/29; remaining=29" in progress and "Qwen remaining=0" in progress
        assert len((output / "assurance_v2_records.jsonl").read_text(encoding="utf-8").splitlines()) == 6
        assert {path.name for path in output.iterdir()} == {"assurance_v2_records.jsonl", "evaluation_manifest.json", "evaluation_metrics.json", "evaluation_table.csv", "validation_table.csv"}
        manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
        assert manifest["oracle_exposed_to_model"] is False
        assert manifest["candidate_and_final_metrics_reported_separately"] is True
        assert all("impact_oracle" not in prompt and "root_node_ids" not in prompt for prompt in client.prompts)
        run_assurance_v2(benchmark_path=benchmark, output_dir=output, profile="smoke", client=client)
        assert client.calls == first_calls


def test_assurance_v2_reports_qwen_errors_before_graph_correction() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        benchmark, output = root / "benchmark" / "impact.jsonl", root / "results"
        prepare_impact_development(benchmark)
        metrics = run_assurance_v2(benchmark_path=benchmark, output_dir=output, profile="smoke", client=FakeAssuranceV2Client(corrupt_candidate=True))
        assert metrics["stages"]["candidate"]["impact_set_recall"] == 0.0
        assert metrics["stages"]["candidate"]["impact_exact_scenario_accuracy"] == 0.0
        assert metrics["stages"]["final"]["impact_set_f1"] == 1.0
        assert metrics["validation"]["corrected_count"] == 4
        assert metrics["validation"]["intervention_rate"] == 4 / 6
        assert metrics["validation"]["missing_node_count"] > 0
