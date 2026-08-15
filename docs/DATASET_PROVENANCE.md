# Dataset provenance

AeroElecBench currently contains fictional synthetic electrical-connectivity artifacts. Its five encoded rules are research-only benchmark assumptions and are not claims of compliance with FAA, NASA, SAE, CPACS, VEC, Pixhawk, or OreSat requirements.

External sources are registered in `data/source_registry.json` under three uses:

- CPACS, VEC, and WireViz are representation references.
- Pixhawk and OreSat are open-design references.
- FAA AC 43.13-1B, NASA-STD-8739.4A, and SAE AS50881J are possible engineering-rule sources.

A rule may become `standard_mapped` only after recording an exact source revision, clause, applicability conditions, exceptions, and an expert-reviewed deterministic interpretation. Registering a source alone does not make a synthetic design compliant or suitable for certification.
