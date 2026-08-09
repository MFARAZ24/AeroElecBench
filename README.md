# AeroECAD-Agent v0.2

AeroECAD-Agent is a research prototype for traceable, human-controlled verification of structured aerospace electrical-design data. It operates outside CATIA, uses only fictional synthetic artifacts, and is an engineer-facing review assistant—not an autonomous designer or certification authority.

## Included experiments

The prototype contains a seeded benchmark with 170 ECAD-like designs: 20 clean, 100 single-fault, and 50 mixed-fault cases. Its five fictional defect families are missing component attributes, duplicate identifiers, invalid connector-pin references, incompatible signal classes, and missing requirement traceability.

Version 0.2 provides five review modes:

| Mode | Purpose |
|---|---|
| `full` | Deterministic execution of all encoded rules |
| `retrieval_guided` | Deterministic execution of rules selected from the review request |
| `llm_only` | Local LLM reasoning with rule names and citation metadata, but no rule criteria |
| `retrieval_grounded` | Local LLM reasoning with retrieved fictional rule criteria |
| `hybrid_explainer` | Deterministic candidate detection followed by LLM verification and explanation |

All reports require human approval and perform zero automatic design modifications.

## Local setup

Python 3.10 or newer, `uv`, and Ollama are required for the LLM experiment. The deterministic baseline does not require Ollama.

```powershell
uv sync
uv run aeroecad experiment --seed 2027 --cases-per-rule 20 --clean-cases 20 --mixed-cases 50
uv run python -m unittest discover -s tests -v
```

Verify the local Ollama service and the three comparison models:

```powershell
uv run aeroecad ollama-check --models qwen2.5:7b llama3.1:8b mistral:7b
```

The program only checks existing Ollama models; it never pulls or downloads a model.

## Run the LLM comparison

Start with a 21-call smoke test using Qwen:

```powershell
uv run aeroecad llm-experiment --models qwen2.5:7b --modes llm_only retrieval_grounded hybrid_explainer --profile smoke
```

Then run the balanced 40-scenario pilot across all three models (360 total calls):

```powershell
uv run aeroecad llm-experiment --models qwen2.5:7b llama3.1:8b mistral:7b --modes llm_only retrieval_grounded hybrid_explainer --profile pilot
```

The complete 170-scenario comparison contains 1,530 calls:

```powershell
uv run aeroecad llm-experiment --models qwen2.5:7b llama3.1:8b mistral:7b --modes llm_only retrieval_grounded hybrid_explainer --profile full
```

Runs are resumable. Each completed response is appended immediately to `results\llm\ollama_responses.jsonl`; if a run is interrupted, rerun the same command and completed model/mode/scenario combinations will be skipped. Smoke-test responses are also reused by the pilot and full profiles when the benchmark is unchanged.

The structured outputs are:

- `results\llm\llm_benchmark_summary.json`
- `results\llm\llm_comparison.csv`
- `results\llm\llm_preliminary_results.md`
- `results\llm\llm_manifest.json`
- `results\llm\ollama_responses.jsonl`

The comparison reports detection precision, recall, F1, exact-scenario accuracy, clean specificity, unsupported-claim rate, parse success, invalid-finding rate, abstention, citation correctness, traceability completeness, token counts, and latency. Malformed or fabricated findings are penalized rather than silently ignored.

`qwen2.5vl:7b` is intentionally excluded because this benchmark contains structured JSON rather than images. It can be evaluated later when schematic screenshots or rendered diagrams are added.

## Interpretation boundary

Perfect deterministic performance demonstrates correct execution within the encoded synthetic rule scope. LLM results measure behavior on the same controlled fictional cases. Neither result establishes airworthiness compliance, industrial performance, proprietary-data validity, or certification readiness.
