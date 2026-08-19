# Compact Qwen intent-grounding evaluation v1.2

The v1.1 full-context smoke run showed that Qwen could often name the correct candidate in its rationale but then abstained, confused wire and component identifiers, or treated unrelated after-design requirement text as the request. Version 1.2 isolates the proposed semantic role and reduces this representation confound.

## Shared pipeline

Every method starts with the same fictional before/after revision and natural-language request. A deterministic, oracle-independent version-diff stage produces the observable candidate inventory. The all-diff control, lexical baseline, and Qwen selector operate on this same inventory. Full versions remain available to deterministic root resolution and graph propagation after selection.

## Qwen input and role

Qwen receives only the request and compact candidate inventory containing candidate ID, change type, entity IDs, changed field, old value, and new value. It does not receive the intent oracle, impact oracle, resolved roots, or full designs. Two generic examples define report versus unmatched abstention without using benchmark entities or oracle labels.

Qwen selects intended candidate IDs or abstains. Valid Qwen selections—not oracle selections—are resolved against the full before/after designs and propagated through the typed graph. No oracle correction is applied. Oracles remain offline scoring references only.

## Evaluation discipline

The v1.1 six-case result is retained as a diagnostic full-context ablation. Prompt revision occurs only on the development split, where post-hoc tuning is allowed. Version 1.2 uses a new pipeline version and output directory so v1.1 records cannot be silently reused. Smoke and development profiles remain one call per scenario with progress reporting and immediate resumable record persistence.
