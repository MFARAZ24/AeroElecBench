# AeroECAD-Agent v0.2 Ollama Results

Profile: **pilot**; scenarios per model/mode: **40**. Models ran locally through Ollama with temperature 0 and a fixed seed.

| Model | Mode | Precision | Recall | F1 | Exact scenario | Unsupported claims | Citation correctness | Parse success | Median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5:7b | llm_only | 0.057 | 0.111 | 0.075 | 0.000 | 0.943 | 1.000 | 1.000 | 119.15 s |
| qwen2.5:7b | retrieval_grounded | 0.067 | 0.111 | 0.084 | 0.050 | 0.933 | 0.000 | 0.900 | 110.06 s |
| qwen2.5:7b | hybrid_explainer | 0.582 | 0.852 | 0.692 | 0.550 | 0.418 | 0.873 | 0.975 | 98.67 s |
| llama3.1:8b | llm_only | 0.085 | 0.185 | 0.117 | 0.000 | 0.915 | 1.000 | 0.950 | 127.21 s |
| llama3.1:8b | retrieval_grounded | 0.077 | 0.222 | 0.114 | 0.000 | 0.923 | 0.000 | 0.975 | 145.47 s |
| llama3.1:8b | hybrid_explainer | 0.412 | 0.870 | 0.560 | 0.425 | 0.588 | 0.781 | 0.925 | 140.67 s |
| mistral:7b | llm_only | 0.045 | 0.056 | 0.050 | 0.000 | 0.955 | 1.000 | 0.525 | 282.32 s |
| mistral:7b | retrieval_grounded | 0.065 | 0.111 | 0.082 | 0.050 | 0.935 | 0.000 | 0.725 | 214.01 s |
| mistral:7b | hybrid_explainer | 0.562 | 0.759 | 0.646 | 0.575 | 0.438 | 0.808 | 0.950 | 118.72 s |

`llm_only` receives rule identifiers, titles, and citation metadata but no rule criteria. `retrieval_grounded` receives the retrieved fictional rule criteria. `hybrid_explainer` receives deterministic candidate findings and relevant rules, then verifies and explains them. All modes require human approval and make no automatic design changes.

These controlled synthetic results measure executable behavior within five fictional ECAD defect families. They are not certification evidence and do not demonstrate performance on proprietary industrial data.
