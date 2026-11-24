# Reproducing the paper

This file maps each result in *Copy First, Check the Field Later* (Findings of
EACL 2027) to the script that produced it. Run every command from the
repository root.

## Without running a model

Every reported rate can be recomputed from the logged answers in `results/raw/`.

```bash
uv run python debug/outcome_split.py        # five-way outcome split of every study -> results/outcome_split.json
uv run python debug/size_sensitivity.py     # size correlations under three sizings -> results/size_sensitivity.json
uv run python debug/tables_camera_ready.py  # LaTeX table bodies -> paper/tables/
uv run --extra viz python debug/figures_paper.py   # the three figures -> paper/figs/
```

`outcome_split.py` also checks, answer by answer, that its *asserted* outcome
reproduces the mis-grounding label each study originally recorded.

## Re-running the experiments

The runners need Apple Silicon and `uv sync --extra mlx`. All models are 4-bit
MLX conversions from the `mlx-community` namespace on the Hugging Face Hub; the
identifiers are listed at the top of each runner (for example
`debug/baseline.py`) and in the paper's reproducibility statement. The
baseline runner appends to its raw log and skips episodes that are already
present, so an interrupted run can be resumed.

## Which script produced which result

All studies use the multi-field *record* version of the instrument unless
marked otherwise. Table numbers refer to the camera-ready version; result files
are in `results/` and raw logs in `results/raw/`.

| study (paper table) | runner | aggregated result | raw log in `results/raw/` |
|---|---|---|---|
| Baseline grid and cue conditions (Tables 2, 3, 8) | `debug/baseline.py` | `baseline.json` | `baseline_raw.jsonl` |
| Label ablation (Table 6) | `debug/mechanism_record.py` | `mechanism_record.json` | `mechanism_record_raw.jsonl` |
| Logit probe (Table 12) | `debug/logprob_margin.py` | `logprob_margin.json` | `logprob_margin_raw.jsonl` |
| Prompt-level mitigation (Table 7) | `debug/mitigation_contrast.py` | `mitigation_contrast.json` | `mitigation_contrast_raw.jsonl` |
| Input ladder, attribute (Tables 4, 9) | `debug/input_cure.py` | `input_cure.json` | `input_cure_raw.jsonl` |
| Input ladder, entity / recency / good (Tables 4, 10) | `debug/ladder_dims.py` | `ladder_dims.json` | `ladder_dims_raw.jsonl` |
| Fallible detectors (Tables 5, 18) | `debug/fallible_detector.py` | `fallible_detector.json` | `fallible_detector_raw.jsonl` |
| Multi-agent verification (Table 11) | `debug/ma_dimensions_record.py`, `debug/ma_verify_aware.py` | `ma_dimensions_record.json`, `ma_verify_aware.json` | `ma_workers_record.jsonl`, `ma_record_supervision.jsonl`, `ma_aware_supervision.jsonl` |
| Commit framing (Tables 13, 14) | `debug/commit_contrast.py` | `commit_contrast.json` | `commit_contrast_raw.jsonl` |
| Outcome splits (Tables 3, 16, 17) | `debug/outcome_split.py` (offline) | `outcome_split.json` | all of the above |
| 4-bit vs 8-bit check (Limitations) | `debug/precision_check.py` | `precision_check.json` | `precision_check_raw.jsonl` |
| Detector validation (Appendix E) | `debug/validate_detector.py`, `debug/detector_heldout.py` | the two `detector_*_sheet.jsonl` files | |
| Larger roster, single-field version (Table 15) | `debug/suite_full.py` | `suite_full.json` | `suite_full_raw.jsonl` |
| Surface form, single-field version (Appendix D) | `debug/realistic_replication.py` | `realistic.json` | `realistic_raw.jsonl` |
| Scaffold, operating point, prompt wording, single-field version (Appendix D) | `debug/check_passive.py`, `debug/robustness.py`, `debug/prompt_robustness.py` | `robustness.json`, `prompt_robustness.json` | |


The remaining scripts in `debug/` and JSON files in `results/` belong to
earlier exploratory studies that the paper does not report.

## Names in the code versus the paper

The code and data fields predate the paper's final vocabulary. Identifiers are
kept stable so that the logs remain reproducible.

| in the code and data | in the paper |
|---|---|
| `confab`, confabulation | mis-grounding |
| `uptake`, `used` | utilization |
| `irrelevant` | wrong entity |
| "Tool-Trust Card" (script output) | per-model profile |
| `uptake_probe` (package name) | the instrument |
| `M2_calibration`, `M3b_interface` (dataset modules) | baseline grid, input-ladder transforms |
