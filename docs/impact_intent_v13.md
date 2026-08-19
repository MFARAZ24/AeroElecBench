# Calibrated Qwen intent-grounding development evaluation v1.3

Version 1.2 substantially exceeded the all-diff and lexical controls on the 24-case development benchmark, but its errors exposed two systematic prompt defects. It reported on all four requests that used only generic approval and consequence language, and it omitted the component-replacement clause in two of four coordinated hardware-plus-requirement requests.

## Final development correction

Version 1.3 adds generic grounding rules rather than benchmark answers. Approval, update, change, and downstream-consequence language cannot authorize a candidate without a distinguishing type or synonym, entity, field, or value. A hardware, unit, or device update at a named component can identify that component's replacement but does not authorize wire or pin edits. Coordinated and conjoined requests must be decomposed so every requested clause is evaluated.

A third generic example demonstrates ambiguity abstention using fictional identifiers unrelated to the benchmark. The model still receives only the engineering request and deterministic change inventory. Full designs, intent and impact oracles, resolved roots, and expected actions remain hidden; graph propagation remains deterministic and no oracle correction is performed.

## Evaluation discipline

This revision is based only on declared development-set failure analysis and therefore remains a development result. Version 1.3 changes the provenance identifier and default output directory, preventing reuse of v1.1 or v1.2 records. After the v1.3 development diagnostic, the prompt must be frozen before generating and running a separately seeded held-out intent benchmark. No held-out result may be used for further prompt tuning.
