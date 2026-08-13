# AeroECAD-Agent v0.2 Ollama Results

Profile: **smoke**; scenarios per model/mode: **7**. Models ran locally through Ollama with temperature 0 and a fixed seed.

| Model | Mode | Precision | Recall | F1 | Exact scenario | Unsupported claims | Citation correctness | Parse success | Median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5:7b | hybrid_explainer | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 91.79 s |
| llama3.1:8b | hybrid_explainer | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 110.24 s |
| mistral:7b | hybrid_explainer | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 118.43 s |

`llm_only` receives rule identifiers, titles, and citation metadata but no rule criteria. `retrieval_grounded` receives the retrieved fictional rule criteria. `hybrid_explainer` receives deterministic candidate findings and relevant rules, then verifies and explains them. All modes require human approval and make no automatic design changes.

These controlled synthetic results measure executable behavior within five fictional ECAD defect families. They are not certification evidence and do not demonstrate performance on proprietary industrial data.
