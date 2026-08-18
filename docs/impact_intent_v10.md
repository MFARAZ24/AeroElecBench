# AeroElecBench v1.0 intent-conditioned change-impact task

## Research question

Can a language model ground a natural-language engineering change request to the intended subset of several concurrent ECAD revision differences, while a deterministic typed graph propagates only the selected roots?

## Observable input

- Before and after electrical designs.
- A natural-language engineering change request.
- A deterministic inventory of all semantic version differences with stable candidate identifiers, structured operations, and affected entity identifiers.

The observable inventory excludes resolved graph-root identifiers. Revision metadata is excluded from semantic change candidates.

## Hidden oracle

- Intended candidate identifiers.
- Resolved intended graph roots.
- Expected report or abstain action.
- Affected nodes and exact multi-hop paths.

The model never receives either oracle. Impact paths are produced by an oracle implementation independent of the production traversal and are checked during benchmark preparation.

## Development cases

Each scenario contains four genuine semantic changes. The balanced case types are single explicit intent, paraphrased intent, multi-change intent, same-entity distractor, ambiguous request, and request with no matching observed change.

## Required baselines

- All-diff graph: propagates every observed change and measures the cost of having no intent understanding.
- Lexical intent graph: non-LLM candidate-selection baseline followed by deterministic propagation.
- LLM-grounded graph: language model selects candidate identifiers; deterministic code resolves roots and propagates.
- Oracle-root graph: hidden-oracle upper bound, never presented as an intelligent system result.

## Primary metrics

Candidate and root precision, recall, F1, exact-set accuracy, distractor rejection, impact-set precision/recall/F1, exact impact accuracy, path precision/recall/F1, action accuracy, and abstention behavior are reported separately.

## Validity controls

Development and held-out splits must be separated by topology seed, entity naming, request templates, and mutation instances. No post-hoc tuning is allowed on the held-out split. Synthetic results establish controlled task performance only and do not establish certification or proprietary-data performance.
