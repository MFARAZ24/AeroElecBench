# AeroElecBench LLM Pilot Analysis

Profile: **pilot**. Integrity check: **PASS** (40/40 expected model-mode-scenario combinations analyzed). No Ollama calls were made.

Best observed configuration by micro-F1 was **qwen2.5:7b / hybrid_explainer** with F1 **1.000** and exact-scenario accuracy **1.000**.

## Overall results

| Model | Mode | Precision | Recall | F1 | Exact | Unsupported | Clean specificity | Median latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5:7b | hybrid_explainer | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 87.76 |

## Paired changes from LLM-only

Not computed because this experiment did not include both `llm_only` and a challenger mode.

## Category breakdown

| Model | Mode | Category | N | F1 | Exact |
|---|---|---|---:|---:|---:|
| qwen2.5:7b | hybrid_explainer | clean | 5 | 0.000 | 1.000 |
| qwen2.5:7b | hybrid_explainer | mixed_fault | 10 | 1.000 | 1.000 |
| qwen2.5:7b | hybrid_explainer | single_fault | 25 | 1.000 | 1.000 |

## Error audit

There are **0** non-exact model-mode-scenario results. Inspect `failure_cases.csv` before interpreting aggregate gains. Per-rule results are in `per_rule_metrics.csv`.

## Interpretation boundary

These are controlled synthetic pilot results over five fictional defect families. They support comparison within this benchmark only; they are not certification evidence or validation on proprietary industrial ECAD data. The 40-scenario pilot is suitable for debugging and preliminary evidence, while confirmatory claims require a separately justified evaluation design.
