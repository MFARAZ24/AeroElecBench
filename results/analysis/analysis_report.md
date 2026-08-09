# AeroElecBench LLM Pilot Analysis

Profile: **pilot**. Integrity check: **PASS** (360/360 expected model-mode-scenario combinations analyzed). No Ollama calls were made.

Best observed configuration by micro-F1 was **qwen2.5:7b / hybrid_explainer** with F1 **0.692** and exact-scenario accuracy **0.550**.

## Overall results

| Model | Mode | Precision | Recall | F1 | Exact | Unsupported | Clean specificity | Median latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5:7b | hybrid_explainer | 0.582 | 0.852 | 0.692 | 0.550 | 0.418 | 0.000 | 98.67 |
| mistral:7b | hybrid_explainer | 0.562 | 0.759 | 0.646 | 0.575 | 0.438 | 0.000 | 118.72 |
| llama3.1:8b | hybrid_explainer | 0.412 | 0.870 | 0.560 | 0.425 | 0.588 | 0.000 | 140.67 |
| llama3.1:8b | llm_only | 0.085 | 0.185 | 0.117 | 0.000 | 0.915 | 0.000 | 127.21 |
| llama3.1:8b | retrieval_grounded | 0.077 | 0.222 | 0.114 | 0.000 | 0.923 | 0.000 | 145.47 |
| qwen2.5:7b | retrieval_grounded | 0.067 | 0.111 | 0.084 | 0.050 | 0.933 | 0.000 | 110.06 |
| mistral:7b | retrieval_grounded | 0.065 | 0.111 | 0.082 | 0.050 | 0.935 | 0.000 | 214.01 |
| qwen2.5:7b | llm_only | 0.057 | 0.111 | 0.075 | 0.000 | 0.943 | 0.000 | 119.15 |
| mistral:7b | llm_only | 0.045 | 0.056 | 0.050 | 0.000 | 0.955 | 0.000 | 282.32 |

## Paired changes from LLM-only

Positive exact/F1 deltas favor the challenger; a negative unsupported-claim delta is better. The exact McNemar p-value uses the same pilot scenarios and is descriptive, unadjusted for multiple comparisons.

| Model | Challenger | Exact delta | F1 delta | Unsupported delta | Improved | Regressed | Exact p |
|---|---|---:|---:|---:|---:|---:|---:|
| llama3.1:8b | retrieval_grounded | +0.000 | -0.003 | +0.009 | 0 | 0 | 1.0000 |
| llama3.1:8b | hybrid_explainer | +0.425 | +0.443 | -0.327 | 17 | 0 | 0.0000 |
| mistral:7b | retrieval_grounded | +0.050 | +0.033 | -0.020 | 2 | 0 | 0.5000 |
| mistral:7b | hybrid_explainer | +0.575 | +0.596 | -0.517 | 23 | 0 | 0.0000 |
| qwen2.5:7b | retrieval_grounded | +0.050 | +0.009 | -0.011 | 2 | 0 | 0.5000 |
| qwen2.5:7b | hybrid_explainer | +0.550 | +0.617 | -0.526 | 22 | 0 | 0.0000 |

## Category breakdown

| Model | Mode | Category | N | F1 | Exact |
|---|---|---|---:|---:|---:|
| llama3.1:8b | hybrid_explainer | clean | 5 | 0.000 | 0.000 |
| llama3.1:8b | hybrid_explainer | mixed_fault | 10 | 0.893 | 0.700 |
| llama3.1:8b | hybrid_explainer | single_fault | 25 | 0.506 | 0.400 |
| llama3.1:8b | llm_only | clean | 5 | 0.000 | 0.000 |
| llama3.1:8b | llm_only | mixed_fault | 10 | 0.000 | 0.000 |
| llama3.1:8b | llm_only | single_fault | 25 | 0.222 | 0.000 |
| llama3.1:8b | retrieval_grounded | clean | 5 | 0.000 | 0.000 |
| llama3.1:8b | retrieval_grounded | mixed_fault | 10 | 0.000 | 0.000 |
| llama3.1:8b | retrieval_grounded | single_fault | 25 | 0.222 | 0.000 |
| mistral:7b | hybrid_explainer | clean | 5 | 0.000 | 0.000 |
| mistral:7b | hybrid_explainer | mixed_fault | 10 | 0.759 | 0.400 |
| mistral:7b | hybrid_explainer | single_fault | 25 | 0.691 | 0.760 |
| mistral:7b | llm_only | clean | 5 | 0.000 | 0.000 |
| mistral:7b | llm_only | mixed_fault | 10 | 0.000 | 0.000 |
| mistral:7b | llm_only | single_fault | 25 | 0.102 | 0.000 |
| mistral:7b | retrieval_grounded | clean | 5 | 0.000 | 0.000 |
| mistral:7b | retrieval_grounded | mixed_fault | 10 | 0.065 | 0.000 |
| mistral:7b | retrieval_grounded | single_fault | 25 | 0.107 | 0.080 |
| qwen2.5:7b | hybrid_explainer | clean | 5 | 0.000 | 0.000 |
| qwen2.5:7b | hybrid_explainer | mixed_fault | 10 | 0.759 | 0.500 |
| qwen2.5:7b | hybrid_explainer | single_fault | 25 | 0.738 | 0.680 |
| qwen2.5:7b | llm_only | clean | 5 | 0.000 | 0.000 |
| qwen2.5:7b | llm_only | mixed_fault | 10 | 0.000 | 0.000 |
| qwen2.5:7b | llm_only | single_fault | 25 | 0.117 | 0.000 |
| qwen2.5:7b | retrieval_grounded | clean | 5 | 0.000 | 0.000 |
| qwen2.5:7b | retrieval_grounded | mixed_fault | 10 | 0.038 | 0.000 |
| qwen2.5:7b | retrieval_grounded | single_fault | 25 | 0.122 | 0.080 |

## Error audit

There are **294** non-exact model-mode-scenario results. Inspect `failure_cases.csv` before interpreting aggregate gains. Per-rule results are in `per_rule_metrics.csv`.

## Interpretation boundary

These are controlled synthetic pilot results over five fictional defect families. They support comparison within this benchmark only; they are not certification evidence or validation on proprietary industrial ECAD data. The 40-scenario pilot is suitable for debugging and preliminary evidence, while confirmatory claims require a separately justified evaluation design.
