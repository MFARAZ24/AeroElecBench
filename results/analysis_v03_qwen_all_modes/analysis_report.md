# AeroElecBench LLM Pilot Analysis

Profile: **pilot**. Integrity check: **PASS** (120/120 expected model-mode-scenario combinations analyzed). No Ollama calls were made.

Best observed configuration by micro-F1 was **qwen2.5:7b / hybrid_explainer** with F1 **1.000** and exact-scenario accuracy **1.000**.

## Overall results

| Model | Mode | Precision | Recall | F1 | Exact | Unsupported | Clean specificity | Median latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5:7b | hybrid_explainer | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 87.76 |
| qwen2.5:7b | retrieval_grounded | 0.067 | 0.111 | 0.084 | 0.050 | 0.933 | 0.000 | 110.06 |
| qwen2.5:7b | llm_only | 0.057 | 0.111 | 0.075 | 0.000 | 0.943 | 0.000 | 115.27 |

## Paired changes from LLM-only

Positive exact/F1 deltas favor the challenger; a negative unsupported-claim delta is better. The exact McNemar p-value uses the same pilot scenarios and is descriptive, unadjusted for multiple comparisons.

| Model | Challenger | Exact delta | F1 delta | Unsupported delta | Improved | Regressed | Exact p |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen2.5:7b | retrieval_grounded | +0.050 | +0.008 | -0.010 | 2 | 0 | 0.5000 |
| qwen2.5:7b | hybrid_explainer | +1.000 | +0.925 | -0.943 | 40 | 0 | 0.0000 |

## Category breakdown

| Model | Mode | Category | N | F1 | Exact |
|---|---|---|---:|---:|---:|
| qwen2.5:7b | hybrid_explainer | clean | 5 | 0.000 | 1.000 |
| qwen2.5:7b | hybrid_explainer | mixed_fault | 10 | 1.000 | 1.000 |
| qwen2.5:7b | hybrid_explainer | single_fault | 25 | 1.000 | 1.000 |
| qwen2.5:7b | llm_only | clean | 5 | 0.000 | 0.000 |
| qwen2.5:7b | llm_only | mixed_fault | 10 | 0.000 | 0.000 |
| qwen2.5:7b | llm_only | single_fault | 25 | 0.117 | 0.000 |
| qwen2.5:7b | retrieval_grounded | clean | 5 | 0.000 | 0.000 |
| qwen2.5:7b | retrieval_grounded | mixed_fault | 10 | 0.038 | 0.000 |
| qwen2.5:7b | retrieval_grounded | single_fault | 25 | 0.122 | 0.080 |

## Error audit

There are **78** non-exact model-mode-scenario results. Inspect `failure_cases.csv` before interpreting aggregate gains. Per-rule results are in `per_rule_metrics.csv`.

## Interpretation boundary

These are controlled synthetic pilot results over five fictional defect families. They support comparison within this benchmark only; they are not certification evidence or validation on proprietary industrial ECAD data. The 40-scenario pilot is suitable for debugging and preliminary evidence, while confirmatory claims require a separately justified evaluation design.
