# Tool-Result Reliance — Diagnostic Benchmark (data card)

**Reference-free (judge-free).** Scoring is exact-match on planted, un-memorizable nonce values plus a human-validated abstention regex (kappa=0.93 development, 0.94 held-out). **No LLM is used at evaluation time.** An LLM was used only at *generation* time as a naturalness gate on realistic-prose documents; its decisions are baked into this filtered release and it is not shipped.

Terminology note: this card and the field/module names predate the paper's final vocabulary. Mapping: *confabulation* (fields) = **silent mis-grounding** (paper); *calibration* (module `M2_calibration`) = **reliance behaviour**; *curability* (modules `M3_curability`/`M3b_interface`) = **mitigation**. Module IDs are kept stable for reproducibility. All instances are in **English**.

- instances: **5097** (n=50/cell, k=5)
- generation-time QC judge (local): `mlx-community/gemma-4-26b-a4b-it-4bit` (dropped 3 awkward; structural drops 0). This judge gates document *naturalness* only; it never scores agent outputs, so its appearance in the model roster is not a circularity.
- contamination resistance: nonce values are random; regenerate with a fresh seed for an uncontaminated pull. uptake_probe.synth.make_instance(seed=20000+i, ...) reproduces every M2/M3 instance; M3b applies the named deterministic transform (debug/input_cure.py or debug/ladder_dims.py mutate) to make_instance(seed=60000+i, ...). Use a fresh seed_base for a contamination-free pull.

- domains: {'access': 700, 'devops': 1098, 'logistics': 1100, 'payments': 1099, 'support': 1100} (template=abstract control on 'access'; record=multi-field business tool outputs across 4 domains, asked field is one of several, prose drawn from a validated diverse frame pool — values stay nonce codes so scoring is unchanged).

## Coverage vs. the paper

`M2_calibration` = the paper's Table 2 grid (good + three entity variants + attr-mismatch + stale + partial across five domains, plus explicit failures error/empty/redacted). `M3b_interface` = the paper's input-ladder transforms (absence/tags/isolate on attribute; absence/isolate on entity and recency; the GOOD-side isolate control) and the label-ablation modes (neutral/bare/warn; `wrong` = plain attr_mismatch in M2). `M3_curability` = legacy signal/verify variants from an earlier intervention design (the good_verify/irrelevant_verify cells back the paper's over-abstention and verify-cue analyses). Prompt-level interventions (cite/verify/consequence), the commit scaffold, and the multi-agent verification protocol are runner-side (no document change): see `debug/mitigation_contrast.py`, `debug/commit_contrast.py`, `debug/ma_dimensions_record.py`; their per-instance raw outputs ship in `results/raw/`.

## License

- Data (this file): **CC BY 4.0**.
- Code (generator / scorer / harness): **MIT** (see `LICENSE`).

## Modules / conditions / domains

| module | condition | style | domain | n |
|---|---|---|---|---|
| M2_calibration | attr_mismatch | record | devops | 50 |
| M2_calibration | attr_mismatch | record | logistics | 50 |
| M2_calibration | attr_mismatch | record | payments | 50 |
| M2_calibration | attr_mismatch | record | support | 50 |
| M2_calibration | attr_mismatch | template | access | 50 |
| M2_calibration | empty | template | access | 50 |
| M2_calibration | error | template | access | 50 |
| M2_calibration | good | record | devops | 50 |
| M2_calibration | good | record | logistics | 50 |
| M2_calibration | good | record | payments | 50 |
| M2_calibration | good | record | support | 50 |
| M2_calibration | good | template | access | 50 |
| M2_calibration | irrelevant | record | devops | 50 |
| M2_calibration | irrelevant | record | logistics | 50 |
| M2_calibration | irrelevant | record | payments | 50 |
| M2_calibration | irrelevant | record | support | 50 |
| M2_calibration | irrelevant | template | access | 50 |
| M2_calibration | irrelevant_far | record | devops | 50 |
| M2_calibration | irrelevant_far | record | logistics | 50 |
| M2_calibration | irrelevant_far | record | payments | 50 |
| M2_calibration | irrelevant_far | record | support | 50 |
| M2_calibration | irrelevant_far | template | access | 50 |
| M2_calibration | irrelevant_near | record | devops | 50 |
| M2_calibration | irrelevant_near | record | logistics | 50 |
| M2_calibration | irrelevant_near | record | payments | 50 |
| M2_calibration | irrelevant_near | record | support | 50 |
| M2_calibration | irrelevant_near | template | access | 50 |
| M2_calibration | partial | record | devops | 50 |
| M2_calibration | partial | record | logistics | 50 |
| M2_calibration | partial | record | payments | 50 |
| M2_calibration | partial | record | support | 50 |
| M2_calibration | partial | template | access | 50 |
| M2_calibration | redacted | template | access | 50 |
| M2_calibration | stale | record | devops | 50 |
| M2_calibration | stale | record | logistics | 50 |
| M2_calibration | stale | record | payments | 50 |
| M2_calibration | stale | record | support | 50 |
| M2_calibration | stale | template | access | 50 |
| M3_curability | good_verify | record | devops | 50 |
| M3_curability | good_verify | record | logistics | 50 |
| M3_curability | good_verify | record | payments | 50 |
| M3_curability | good_verify | record | support | 50 |
| M3_curability | good_verify | template | access | 50 |
| M3_curability | irrelevant_sig_hard | record | devops | 50 |
| M3_curability | irrelevant_sig_hard | record | logistics | 50 |
| M3_curability | irrelevant_sig_hard | record | payments | 50 |
| M3_curability | irrelevant_sig_hard | record | support | 50 |
| M3_curability | irrelevant_sig_hard | template | access | 50 |
| M3_curability | irrelevant_sig_soft | record | devops | 50 |
| M3_curability | irrelevant_sig_soft | record | logistics | 50 |
| M3_curability | irrelevant_sig_soft | record | payments | 50 |
| M3_curability | irrelevant_sig_soft | record | support | 50 |
| M3_curability | irrelevant_sig_soft | template | access | 50 |
| M3_curability | irrelevant_verify | record | devops | 50 |
| M3_curability | irrelevant_verify | record | logistics | 50 |
| M3_curability | irrelevant_verify | record | payments | 50 |
| M3_curability | irrelevant_verify | record | support | 50 |
| M3_curability | irrelevant_verify | template | access | 50 |
| M3b_interface | attr_absence | record | devops | 50 |
| M3b_interface | attr_absence | record | logistics | 50 |
| M3b_interface | attr_absence | record | payments | 50 |
| M3b_interface | attr_absence | record | support | 50 |
| M3b_interface | attr_isolate | record | devops | 50 |
| M3b_interface | attr_isolate | record | logistics | 50 |
| M3b_interface | attr_isolate | record | payments | 50 |
| M3b_interface | attr_isolate | record | support | 50 |
| M3b_interface | attr_label_bare | record | devops | 50 |
| M3b_interface | attr_label_bare | record | logistics | 50 |
| M3b_interface | attr_label_bare | record | payments | 50 |
| M3b_interface | attr_label_bare | record | support | 50 |
| M3b_interface | attr_label_neutral | record | devops | 50 |
| M3b_interface | attr_label_neutral | record | logistics | 50 |
| M3b_interface | attr_label_neutral | record | payments | 50 |
| M3b_interface | attr_label_neutral | record | support | 50 |
| M3b_interface | attr_label_warn | record | devops | 48 |
| M3b_interface | attr_label_warn | record | logistics | 50 |
| M3b_interface | attr_label_warn | record | payments | 49 |
| M3b_interface | attr_label_warn | record | support | 50 |
| M3b_interface | attr_tags | record | devops | 50 |
| M3b_interface | attr_tags | record | logistics | 50 |
| M3b_interface | attr_tags | record | payments | 50 |
| M3b_interface | attr_tags | record | support | 50 |
| M3b_interface | entity_absence | record | devops | 50 |
| M3b_interface | entity_absence | record | logistics | 50 |
| M3b_interface | entity_absence | record | payments | 50 |
| M3b_interface | entity_absence | record | support | 50 |
| M3b_interface | entity_isolate | record | devops | 50 |
| M3b_interface | entity_isolate | record | logistics | 50 |
| M3b_interface | entity_isolate | record | payments | 50 |
| M3b_interface | entity_isolate | record | support | 50 |
| M3b_interface | good_isolate | record | devops | 50 |
| M3b_interface | good_isolate | record | logistics | 50 |
| M3b_interface | good_isolate | record | payments | 50 |
| M3b_interface | good_isolate | record | support | 50 |
| M3b_interface | stale_absence | record | devops | 50 |
| M3b_interface | stale_absence | record | logistics | 50 |
| M3b_interface | stale_absence | record | payments | 50 |
| M3b_interface | stale_absence | record | support | 50 |
| M3b_interface | stale_isolate | record | devops | 50 |
| M3b_interface | stale_isolate | record | logistics | 50 |
| M3b_interface | stale_isolate | record | payments | 50 |
| M3b_interface | stale_isolate | record | support | 50 |

## Scoring (consumer-side)

For each necessary fact: the agent's answer USES the result iff a `gold_values` entry appears verbatim (separator-tolerant). On unobtainable conditions (`unobtainable_entities` non-empty) any `decoys` value asserted without an abstention marker is a *silent mis-grounding* (field name: confabulation). See `uptake_probe.detect` / `uptake_probe.metrics` for the reference scorer.
