# Oracle-separated intent evaluation protocol v1.4

Version 1.4 changes evaluation architecture without changing the frozen v1.3 prompt. It addresses the distinction between hiding oracle fields from a prompt and preventing the prediction process from reading them at all.

## Three independent phases

1. `impact-intent-separate` validates the frozen benchmark and writes two hash-bound files. `model_inputs.jsonl` contains only scenario ID, before/after designs, engineering request, and deterministic change inventory. `oracle_reference.jsonl` contains case type and intent/impact references.
2. `impact-intent-predict` accepts only the separated package and produces raw, resumable predictions plus a prediction manifest. It succeeds when the oracle file is absent, writes no expected labels or metrics, and records the hash of the completed predictions.
3. `impact-intent-score` requires completed frozen predictions, verifies their hash, then loads the oracle reference and generates deterministic metrics. It never edits or corrects predictions.

The model-input validator rejects oracle, expected-label, case-type, split, request-ID, source-scenario-ID, root, impact, or path fields at any nesting level. Source scenarios are deterministically shuffled and replaced with opaque IDs so their original case-type-bearing identifiers and grouped order cannot reach prediction. Package, input, oracle, and prediction hashes link the phases while retaining a reviewable provenance trail in the separate oracle reference.

## Interpretation

The v1.3 smoke run remains a development diagnostic. The prompt was tuned using development failures and is not a zero-shot result. The v1.4 protocol is intended for a separately seeded, frozen held-out benchmark after its input and oracle files are created and the v1.3 prompt is frozen. Held-out predictions must be completed and hash-bound before offline scoring, and held-out results must not trigger further prompt changes.
