# Recorded results and evidence map

Original evidence was checked against the locally retained `runs_20261001.tar.gz` archive.
The bootstrap was corrected and recomputed on 2026-10-08; documentation was synchronized on
2026-10-09. Of the 24 previously published result JSONs, 23 remain unchanged. The earlier
`bootstrap_seq.json` is preserved exactly in `legacy/bootstrap_seq_before_20261008.json`;
the current file contains corrected results from the saved predictions.

| Evidence | Scope / correct use |
|---|---|
| `eval_sq-4ch*_s*.json` | Main R-LiViT uniform held-out COCO results: 480 frames; development-selected checkpoints. Use `full.AP50` and `full.AP` for the main accuracy table. |
| [main_test_metrics.csv](main_test_metrics.csv) | Direct extraction from the preceding JSONs for full/day/night; values are fractions, multiply by 100 for percentage points. No independent evaluation. |
| `eval_late_*.json` | Matched seed-0/1 RGB + thermal late fusion using the same test procedure. |
| [summary_sq.json](summary_sq.json) | Training-framework evaluations and associated timing/model records. Its accuracy values differ from the separate COCO evaluation; do not use them for the main manuscript table. |
| [bootstrap_seq.json](bootstrap_seq.json) | Corrected paired sequence bootstrap (3,000 AP50 and 1,000 AP draws per comparison), preserving repeated occurrences. Direct point differences and bootstrap means are separately named. |
| [qual_report.json](qual_report.json), [qual_breakdown.json](qual_breakdown.json) | Nighttime seed-0 detection transitions at confidence 0.25 and IoU 0.5, with class-consistent one-to-one matching; not AP improvements or causal counts. |
| [eff_v2.json](eff_v2.json), [gsw_latency.json](gsw_latency.json) | Recorded latency and manual gate accounting. Disk I/O affects end-to-end means; custom-module THOP counts remain provisional. |
| [msq_summary.json](msq_summary.json) | Training-framework held-out metrics for the separate 782-frame projected-depth subset, not the 480-frame main test. |
| [llvip_summary.json](llvip_summary.json) | Dataset-specific retraining and full 3,463-pair test; two matched seeds. Not zero-shot transfer. |
| `legacy/*.json` | Earlier development/reporting-partition analyses, including gate profile/staticization/subsets. The exception `bootstrap_seq_before_20261008.json` preserves the faulty earlier bootstrap on the current held-out test. |
| [protocol_archive.json](../paper/protocol_archive.json) | Original environment, selection records and hashes; embedded summaries span different evaluation procedures. |

## Bootstrap correction and validation

The retained implementation did sample sequences with replacement, but then selected frames using
membership in `set(sample)`, collapsing repeat occurrences. For example, the first seed-42 full-test
draw contains 40 sequence occurrences but only 27 distinct sequences: the earlier selection kept
324 frames, whereas the corrected bootstrap retains 480 frame occurrences. The original results
are preserved for traceability; use the current results for manuscript v9.

The corrected `src/bootstrap_seq.py` caches per-image COCO matching and accumulates every sampled
occurrence. Each model pair uses the same sampled occurrences. It uses sampling seed 42, fixed
training-seed-0 checkpoints, COCO maxDets 100, ten IoU thresholds and 101 recall thresholds. The
full-test, nighttime and depth analyses contain 40, 14 and 13 sequences, respectively. Each of the
four comparisons has 3,000 AP50 and 1,000 AP draws, totaling 16,000 paired resamples.

[bootstrap_verification.json](bootstrap_verification.json) records 20 comparisons against literal
duplication of images, annotations and predictions with fresh COCO IDs. The maximum absolute AP
error was 1.11e-16. Main full/day/night point estimates also match the archived COCO evaluations
within 1e-12. [bootstrap_draws.npz](bootstrap_draws.npz) stores all paired AP differences; these
are differences per draw, rather than sequence-ID lists. Source, input and environment provenance
are included in [bootstrap_seq.json](bootstrap_seq.json).

The `*_point` fields are direct differences on the original test set, while `*_mean` fields are
bootstrap means. Figure markers and manuscript estimates use the former. Percentile intervals
describe test-sequence composition for these fixed checkpoints and exclude training-seed
uncertainty. The nighttime GBF minus Concat AP50 lower bound is close to zero (+0.05 pp).

Depth and LLVIP tables use retained training-framework evaluations. The separate depth bootstrap
uses COCO evaluation of saved predictions: its seed-0 direct AP50 difference is +1.74 pp, with
interval [-2.68, +7.09], whereas the training-framework depth table gives +1.29 pp for seed 0.
These evaluation procedures must remain explicit when comparing the values.

## Provenance and hash conventions

New legacy evidence was copied from `runs/rlivit/{gate_profile,staticized_summary,
full_subset_summary,full_subset_summary_s1}.json` in the same archive. Calibration was copied from
the retained `datasets/RLiViT/calib_per_seq.json`. The earlier v8 manuscript remains preserved.
The current v9 manuscript updates bootstrap methods, results, figure captions and disclosure.
Fig. 4 / Fig. 6 were redrawn from corrected results; the other four figures remain unchanged.
All six current PNGs match their corresponding v9 DOCX image parts.

The manifest `md5` fields in `protocol_archive.json` hash the UTF-8 concatenation of sorted lines
after text-mode newline normalization, as implemented by `archive_protocol.manifest_md5`.
They are not raw-file MD5 values. Checkpoint `best_md5` values in that archive are abbreviated MD5
prefixes. Keep those conventions when comparing records.
