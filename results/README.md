# Recorded results and evidence map

Synchronized on 2026-10-08 against the experiment outputs retained in the local
`runs_20261001.tar.gz` archive. The 24 previously published result JSON files match the corresponding
archived records semantically; none were replaced or recalculated for this update.

| Evidence | Scope / correct use |
|---|---|
| `eval_sq-4ch*_s*.json` | Main R-LiViT uniform held-out COCO results: 480 frames; development-selected checkpoints. Use `full.AP50` and `full.AP` for the main accuracy table. |
| [main_test_metrics.csv](main_test_metrics.csv) | Direct extraction from the preceding JSONs for full/day/night; values are fractions, multiply by 100 for percentage points. No independent evaluation. |
| `eval_late_*.json` | Matched seed-0/1 RGB + thermal late fusion using the same test procedure. |
| [summary_sq.json](summary_sq.json) | Training-framework evaluations and associated timing/model records. Its accuracy values differ from the separate COCO evaluation; do not use them for the main manuscript table. |
| [bootstrap_seq.json](bootstrap_seq.json) | Original sequence-resampling output (3,000 AP50 and 1,000 AP draws). Interpretation is restricted by the issue below. |
| [qual_report.json](qual_report.json), [qual_breakdown.json](qual_breakdown.json) | Nighttime seed-0 detection transitions at confidence 0.25 and IoU 0.5, with class-consistent one-to-one matching; not AP improvements or causal counts. |
| [eff_v2.json](eff_v2.json), [gsw_latency.json](gsw_latency.json) | Recorded latency and manual gate accounting. Disk I/O affects end-to-end means; custom-module THOP counts remain provisional. |
| [msq_summary.json](msq_summary.json) | Training-framework held-out metrics for the separate 782-frame projected-depth subset, not the 480-frame main test. |
| [llvip_summary.json](llvip_summary.json) | Dataset-specific retraining and full 3,463-pair test; two matched seeds. Not zero-shot transfer. |
| `legacy/*.json` | Earlier development/reporting-partition analyses, including gate profile/staticization/subsets. Not current held-out test estimates. |
| [protocol_archive.json](../paper/protocol_archive.json) | Original environment, selection records and hashes; embedded summaries span different evaluation procedures. |

## Statistical audit note

The historical `src/bootstrap_seq.py` draws sequences with replacement, then selects frames using
membership in `set(sample)`. Repeated draws of a sequence are therefore collapsed; their multiplicity
does not contribute to AP. The stored percentile bounds are consequently **not a validated ordinary
cluster-bootstrap confidence interval**. They must be recomputed with repeated sequences represented
by distinct/remapped image and annotation IDs, and a verified evaluation procedure, before supporting
statistical significance or confidence-interval claims. The bootstrap seed is 42; the evaluated model
checkpoints remain fixed at training seed 0. Even a corrected interval would not quantify variation
over training seeds.

This update preserves the historical algorithm and its numeric outputs with an explicit warning.
It does not generate replacement intervals. The copied v8 manuscript and its Fig. 4 / Fig. 6 interval
panels still reflect those recorded outputs and need corresponding revision before submission.
The main per-seed point estimates and arithmetic means are independent of this resampling issue.

## Provenance and hash conventions

New legacy evidence was copied from `runs/rlivit/{gate_profile,staticized_summary,
full_subset_summary,full_subset_summary_s1}.json` in the same archive. Calibration was copied from
the retained `datasets/RLiViT/calib_per_seq.json`. The manuscript was copied without modification,
and each of its six figures was extracted directly from the DOCX image relationship in figure order.

The manifest `md5` fields in `protocol_archive.json` hash the UTF-8 concatenation of sorted lines
after text-mode newline normalization, as implemented by `archive_protocol.manifest_md5`.
They are not raw-file MD5 values. Checkpoint `best_md5` values in that archive are abbreviated MD5
prefixes. Keep those conventions when comparing records.
