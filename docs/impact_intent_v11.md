# Qwen intent-conditioned graph evaluation v1.1

This development experiment measures whether a local Qwen model can ground a natural-language engineering change request to the intended subset of genuine edits in a concurrent before/after revision. The selected edits, rather than oracle edits, are resolved to typed graph roots and propagated deterministically.

## Observable model input

- complete before and after fictional electrical designs;
- natural-language engineering change request;
- deterministic semantic change inventory with candidate IDs and operations.

The intent oracle, impact oracle, and resolved candidate root IDs are excluded from the prompt. Oracles are used only after inference for deterministic scoring. No oracle correction is applied to the model selection or downstream graph result.

## Output contract

Qwen makes exactly one call per scenario and returns `report` with one or more supplied candidate IDs, or `abstain` with an empty selection. Unknown IDs, duplicate IDs, malformed JSON, and inconsistent action/selection pairs are rejected. Valid Qwen-selected candidates are resolved to graph roots and propagated to the configured depth.

## Profiles and recovery

- `smoke`: one scenario from each of the six intent case types, six Qwen calls total;
- `development`: all 24 scenarios, 24 Qwen calls total.

Each completed scenario is appended immediately to `intent_records.jsonl`. Repeating the same command validates and reuses existing records, so an interrupted call resumes at the first incomplete scenario. Progress logs report completed records, completed calls, and remaining calls.

## Reported measurements

Candidate precision, recall, F1, exact-set accuracy, and distractor rejection measure semantic grounding. Root, impact-set, and path metrics measure the downstream result produced from Qwen-selected candidates. Action accuracy measures report/abstain decisions. These metrics are directly comparable with the v1.0 all-diff, lexical-intent, and oracle-root controls.
