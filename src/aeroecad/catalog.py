from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_catalog_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "rules.json"


def load_catalog(path: str | Path | None = None) -> dict[str, Any]:
    with Path(path or default_catalog_path()).open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    required = {"catalog_id", "version", "rules"}
    if missing := required - catalog.keys():
        raise ValueError(f"Rule catalog is missing fields: {sorted(missing)}")
    ids = [rule["rule_id"] for rule in catalog["rules"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Rule identifiers must be unique")
    return catalog


def rule_index(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {rule["rule_id"]: rule for rule in catalog["rules"]}
