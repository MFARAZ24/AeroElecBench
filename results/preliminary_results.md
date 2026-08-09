# AeroECAD-Agent v0.2 Deterministic Baseline Results

The seeded synthetic benchmark contains 170 designs: 20 clean, 100 single-fault, and 50 mixed-fault scenarios. Across the benchmark, 277 violations cover five encoded fictional ECAD rule families.

| Mode | Precision | Recall | F1 | Unsupported claims | Citation correctness | Traceability completeness | Median latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full deterministic audit | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.029 ms |
| Retrieval-guided deterministic audit | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.032 ms |

The retrieval-guided mode achieved top-1 rule-retrieval accuracy of 1.000, top-3 recall of 1.000, and MRR of 1.000 on 100 targeted single-fault queries. Both modes flagged all reports for human review and performed zero automatic design modifications.

These results verify correct execution and evidence traceability within a deliberately controlled synthetic rule scope. They must not be interpreted as certification evidence or as validation on proprietary industrial ECAD artifacts. Version 0.2 adds separate LLM-only, retrieval-grounded, and hybrid explanation experiments while preserving this deterministic baseline.
