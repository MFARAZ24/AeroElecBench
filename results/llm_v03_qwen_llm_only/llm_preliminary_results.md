# AeroECAD-Agent v0.2 Ollama Results

Profile: **pilot**; scenarios per model/mode: **40**. Models ran locally through Ollama with temperature 0 and a fixed seed.

| Model | Mode | Precision | Recall | F1 | Exact scenario | Unsupported claims | Citation correctness | Parse success | Median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5:7b | llm_only | 0.057 | 0.111 | 0.075 | 0.000 | 0.943 | 1.000 | 1.000 | 115.27 s |

`llm_only` receives rule identifiers, titles, and citation metadata but no rule criteria. `retrieval_grounded` receives the retrieved fictional rule criteria. `hybrid_explainer` receives deterministic candidate findings and relevant rules, then verifies and explains them. All modes require human approval and make no automatic design changes.

These controlled synthetic results measure executable behavior within five fictional ECAD defect families. They are not certification evidence and do not demonstrate performance on proprietary industrial data.
