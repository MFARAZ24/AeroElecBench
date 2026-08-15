from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROLES = {"representation_reference", "open_design_reference", "engineering_guidance", "normative_standard", "restricted_normative_standard"}

def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))

def test_source_registry_is_well_formed() -> None:
    sources = load_json("data/source_registry.json")["sources"]
    ids = [source["source_id"] for source in sources]
    assert len(ids) == len(set(ids))
    assert all(source["role"] in ALLOWED_ROLES for source in sources)
    assert all(source["url"].startswith("https://") and source["revision"] and source["usage"] for source in sources)

def test_every_rule_has_explicit_provenance_classification() -> None:
    catalog, registry = load_json("data/rules.json"), load_json("data/source_registry.json")
    source_ids = {source["source_id"] for source in registry["sources"]}
    for rule in catalog["rules"]:
        assert rule["classification"] in {"research_only", "standard_mapped"}
        assert isinstance(rule["source_refs"], list) and set(rule["source_refs"]) <= source_ids
        assert rule["basis"]
        if rule["classification"] == "standard_mapped":
            assert rule["source_refs"] and rule.get("source_clause") and rule.get("applicability")

def test_current_rules_remain_research_only() -> None:
    assert all(rule["classification"] == "research_only" for rule in load_json("data/rules.json")["rules"])
