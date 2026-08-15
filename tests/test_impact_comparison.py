from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from aeroecad.impact_agent import REQUIRED_TOOL_PLAN, build_impact_prompt, parse_impact_content, retrieve_text_chunks
from aeroecad.impact_benchmark import generate_impact_benchmark
from aeroecad.impact_comparison import run_impact_comparison
from aeroecad.impact_evaluation import prepare_impact_development
from aeroecad.impact_graph import analyze_change_impact
from aeroecad.ollama import OllamaResponse


class FakeImpactClient:
    def __init__(self) -> None:
        self.calls = 0

    def ensure_models(self, requested: list[str]) -> list[str]:
        return requested

    def chat(self, model: str, system: str, user: str, **_: object) -> OllamaResponse:
        self.calls += 1
        payload = json.loads(user)
        if "planning agent" in system:
            content = json.dumps({"action": "analyze", "tool_plan": REQUIRED_TOOL_PLAN, "rationale": "Use the complete bounded tool chain."})
        else:
            change = payload["change"]
            action = "no_change" if change["change_type"] == "no_change" else "abstain" if not change["evidence_complete"] else "report"
            content = json.dumps({"action": action, "affected_node_ids": [], "impact_edges": [], "rationale": "Synthetic test response."})
        return OllamaResponse(content, {"eval_count": 1})


def test_graph_baseline_infers_roots_without_oracle_root_field() -> None:
    scenarios = generate_impact_benchmark(seed=41, cases_per_type=1)
    for scenario in scenarios:
        request = copy.deepcopy(scenario["change_request"])
        request.pop("root_node_ids")
        report = analyze_change_impact(scenario["before_design"], scenario["after_design"], request)
        assert report["affected_node_ids"] == scenario["impact_oracle"]["affected_node_ids"]
        assert report["impact_paths"] == scenario["impact_oracle"]["impact_paths"]


def test_model_prompts_hide_oracle_roots_and_limit_rag_to_retrieved_text() -> None:
    scenario = next(item for item in generate_impact_benchmark(seed=43, cases_per_type=1) if item["impact_case_type"] == "component_replacement")
    direct, _ = build_impact_prompt(scenario, "llm_only")
    rag, evidence = build_impact_prompt(scenario, "text_rag", top_k=5)
    assert "impact_oracle" not in direct and "root_node_ids" not in direct
    assert "impact_oracle" not in rag and "root_node_ids" not in rag
    assert "before_design" in direct and "before_design" not in rag
    assert len(evidence) == 5 and len(retrieve_text_chunks(scenario, 5)) == 5


def test_impact_parser_rejects_hallucinated_nodes() -> None:
    scenario = next(item for item in generate_impact_benchmark(seed=47, cases_per_type=1) if item["impact_case_type"] == "component_replacement")
    content = json.dumps({"action": "report", "affected_node_ids": ["component:NOT-DECLARED"], "impact_edges": [], "rationale": "Unsupported."})
    report, diagnostics = parse_impact_content(content, scenario)
    assert report["status"] == "rejected"
    assert diagnostics == {"parse_success": False, "error": "invalid_affected_nodes"}


def test_impact_parser_reconstructs_paths_only_from_model_edges() -> None:
    scenario = next(item for item in generate_impact_benchmark(seed=49, cases_per_type=1) if item["impact_case_type"] == "component_replacement")
    oracle = scenario["impact_oracle"]
    edges = {
        (path["node_ids"][index], path["node_ids"][index + 1], relation)
        for path in oracle["impact_paths"] for index, relation in enumerate(path["relations"])
    }
    content = json.dumps({
        "action": "report", "affected_node_ids": oracle["affected_node_ids"],
        "impact_edges": [{"source": source, "target": target, "relation": relation} for source, target, relation in sorted(edges)],
        "rationale": "All affected nodes are connected by supplied evidence edges.",
    })
    report, diagnostics = parse_impact_content(content, scenario)
    assert diagnostics["parse_success"] is True
    assert report["affected_node_ids"] == oracle["affected_node_ids"]
    assert report["impact_paths"] == oracle["impact_paths"]


def test_comparison_is_resumable_and_writes_only_data_outputs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        benchmark, output = root / "benchmark" / "impact.jsonl", root / "comparison"
        prepare_impact_development(benchmark)
        client = FakeImpactClient()
        modes = ("llm_only", "text_rag", "graph_deterministic", "assurance_agent")
        metrics = run_impact_comparison(modes=modes, benchmark_path=benchmark, output_dir=output, profile="smoke", client=client)
        first_call_count = client.calls
        assert metrics["scenario_count"] == 6 and first_call_count == 16
        assert metrics["modes"]["graph_deterministic"]["oracle_action_accuracy"] == 1.0
        assert metrics["modes"]["assurance_agent"]["oracle_action_accuracy"] == 1.0
        assert metrics["modes"]["assurance_agent"]["llm_call_count"] == 4
        assert all(len((output / mode / "impact_records.jsonl").read_text(encoding="utf-8").splitlines()) == 6 for mode in modes)
        manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
        assert manifest["oracle_exposed_to_model"] is False and manifest["root_node_ids_exposed_to_model"] is False
        assert not list(output.rglob("*.md"))
        run_impact_comparison(modes=modes, benchmark_path=benchmark, output_dir=output, profile="smoke", client=client)
        assert client.calls == first_call_count
