# Lightweight Shallow RGB-Thermal Fusion for Roadside Object Detection: Design and Controlled Evaluation

Code, splits, and recorded results accompanying the [latest JSC draft (v9)](paper/manuscript_v9_JSC.docx).

Repository synchronized with the locally retained data and the 2026-10-01 experiment archive. The sequence bootstrap was corrected and recomputed on **2026-10-08**; manuscript v9, figures and documentation were synchronized on **2026-10-09**. Saved predictions were reused without retraining. Author metadata remains to be completed before submission.

We compare lightweight early-fusion configurations for a YOLO11n-based roadside detector on
[R-LiViT](https://github.com/XITASO/r-livit) under a **sequence-disjoint train/dev/test protocol**
with checkpoint selection isolated from a frozen held-out test set, and evaluate every method with a
uniform COCO procedure across three training seeds.

## Configurations

| Name | Description |
|---|---|
| Concat | 4-channel RGB+thermal early-fusion baseline (single shared stem) |
| TBF | Dual-branch shallow encoders, equal fixed mixing (w = 0.5) |
| GSW | Dual-branch encoders + 64-parameter static learned channel weights |
| GBF | Dual-branch encoders + input-conditioned gate, w = σ(MLP([mean,std of both branches])) |
| Late fusion | Same-recipe RGB-only + thermal-only detectors merged by NMS |
| 5ch variants | Optional projected-LiDAR-depth third branch (exploratory, label-informed calibration) |

## Main results (frozen held-out test, COCO, three seeds)

| Model | mAP50 (s0/s1/s2) | mean mAP50 | mAP50–95 (s0/s1/s2) | mean mAP50–95 |
|---|---|---|---|---|
| Concat | 60.88 / 59.83 / 57.43 | 59.38 | 38.78 / 38.95 / 37.32 | 38.35 |
| GBF | 61.13 / 59.44 / 58.77 | 59.78 | 39.78 / 38.55 / 39.46 | 39.26 |
| GSW | 61.54 / 58.96 / 60.75 | **60.42** | 39.67 / 38.93 / 39.61 | **39.40** |

**Summary of findings**
- GSW reaches the highest three-seed mean among the single-model variants, but seed-wise ordering is
  mixed; no consistent performance benefit from input-conditioned gating (GBF) over static channel
  weighting (GSW) was observed.
- Corrected paired sequence bootstrap for fixed seed-0 checkpoints gives a direct nighttime
  GBF minus Concat mAP50 difference of +3.16 pp, with 95% percentile interval [+0.05, +9.23].
  The full-test mAP50–95 difference is +1.00 pp, with interval [+0.06, +2.33]. The nighttime
  lower bound is close to zero; these intervals exclude training-seed uncertainty. See the
  [correction and validation record](results/README.md#bootstrap-correction-and-validation).
- Nighttime GBF differences across seeds (+3.2, −5.9, +3.7 pp) are smaller than training-seed
  variability (SD ≈ 5 pp).
- Across matched seeds 0–1, two-model late fusion exceeds GSW nighttime mAP50 by ≈9.1 pp at roughly
  twice the model parameters.
- Projected LiDAR depth changes mAP50 by +1.3/−0.8/+1.0 pp across matched seeds (exploratory;
  label-informed calibration; 154-frame held-out test).
- On LLVIP (complete 3,463-pair held-out test), GBF vs. Concat differences are small and
  metric-dependent (mAP50 95.09 vs. 95.23; mAP50–95 59.78 vs. 58.79).
- Protocol sensitivity: the earlier development protocol (checkpoints selected on the reporting
  partition) attributed +6.9/+10.0 nighttime pp to GBF; under the redesigned protocol the three-seed
  mean difference is +0.3 pp. The reduction cannot be attributed exclusively to checkpoint selection
  because training-set size/composition and reporting sequences also changed.

The primary R-LiViT accuracy numbers come from `results/eval_*.json`, using the recorded COCO
evaluation procedure. [main_test_metrics.csv](results/main_test_metrics.csv) is a direct extraction
of full/day/night AP values. `summary_sq.json` contains training-framework evaluations and must
not replace the uniform held-out COCO values. The [results guide](results/README.md) maps the
evidence to the manuscript; the [updated results report](paper/final_results_report.md) records the
current values and their limits.

## Repository layout

```
configs/    dataset yamls (paths relative to repo root; put datasets/ here)
splits/     sequence-disjoint manifests + partition reports (frame stems only)
src/        training / evaluation / analysis scripts (paths relative to repo root)
results/    recorded evaluation JSONs (eval_*, bootstrap_seq, qual_*, eff_v2, gsw_latency, summaries)
paper/      manuscript_v9_JSC.docx, protocol_archive.json, final_results_report.md (older files retained)
figures/    all six exact v9 embedded figures; Fig. 4 / Fig. 6 also have SVG exports
data/       dataset links — raw data is NOT redistributed
```

## Reproduction inputs and commands

Run commands from the repository root. Install the recorded research dependencies (PyTorch,
Ultralytics, NumPy, OpenCV, Pillow, pycocotools and THOP) in a compatible environment; the custom
loader patches depend on Ultralytics internals. The historical paper/table generators also need
python-docx. This update checks source paths and recorded artifacts, not a complete retraining.

1. Obtain the datasets from their providers; see [data/README.md](data/README.md). The scripts
   expect prepared RGB/thermal images and YOLO labels under `datasets/`, not the downloads as-is.
   Use the published manifests as the authoritative partitions. R-LiViT preparation starts from
   `datasets/RLiViT/rlivit_full/manifest.csv` and paired `rgb/ir` images/labels. LLVIP expects packed
   four-channel PNGs in `datasets/LLVIP/LLVIP/yolo_fusion/images/{train,dev,val}` and matching labels;
   `val` here denotes the complete held-out test, not the development partition. The repository
   does not include a complete raw-download-to-prepared-data pipeline.
2. `python src/build_sq_split.py` creates the R-LiViT sequence partitions from the prepared full
   dataset. For depth, copy `data/calib_per_seq.json` to `datasets/RLiViT/calib_per_seq.json`, then
   run `python src/build_5ch_v2.py` and `python src/fixup_multi_sq.py` against the official inputs.
   These fitted parameters are label-informed; they are not independent calibration ground truth.
3. Provide `yolo11n.pt` at the repository root. Train with
   `python src/train_rlivit_sq.py sq-4ch 150` (or `sq-4ch-gbf`, `sq-4ch-gsw`, `sq-4ch-tbf`, `sq-rgb`,
   `sq-ir`); set `RLIVIT_SEED` to 0, 1 or 2. Depth and LLVIP use `train_multi_sq.py` and
   `train_llvip_v2.py`. Example in PowerShell: `$env:RLIVIT_SEED='0'`.
4. Evaluate frozen R-LiViT test: `python src/sq_eval.py sq-4ch 0`; late fusion:
   `python src/sq_eval.py --late sq-rgb 0 sq-ir 0`. Original `best.pt` checkpoints belong under
   `runs/rlivit_sq/rl_<experiment>_s<seed>/weights/`. They are available on request.
5. `src/qual_breakdown.py`, `src/eff_v2.py` and `src/gsw_latency.py` contain analysis procedures.
   Run `python src/bootstrap_seq.py --output results/bootstrap_seq.json` with the retained seed-0
   prediction JSONs and prepared images/labels. The corrected procedure retains repeated draws and
   checks accumulation against literal COCO duplication before computation. Each comparison uses
   3,000 AP50 and 1,000 AP draws. `python src/plot_bootstrap_figures.py` redraws Fig. 4 / Fig. 6
   from recorded results; it requires Matplotlib. Saved predictions are available on request.

Recorded environment: Ultralytics 8.4.75, Python 3.11.6, PyTorch 2.12.0.dev20260408+cu128,
NVIDIA RTX 5060 Laptop (8,151 MiB). [protocol_archive.json](paper/protocol_archive.json) contains
the original training environment, selection epochs and checkpoint hashes. The separate bootstrap
recomputation environment and input hashes are recorded in `results/bootstrap_seq.json`.
The protocol archive's embedded accuracy summaries use multiple evaluation procedures;
use `eval_*.json` for the main R-LiViT table.

The authors' original `runs/` outputs were compressed locally on 2026-10-01. This repository keeps
the small recorded result files. To reuse the original predictions/checkpoints for analysis,
restore the original archive or request the relevant files from the authors.

## Manuscript figures

| v9 figure | Exact embedded image |
|---|---|
| Fig. 1: evaluation protocol | [fig1_protocol.png](figures/fig1_protocol.png) |
| Fig. 2: architecture | [fig2_architecture.png](figures/fig2_architecture.png) |
| Fig. 3: three-seed accuracy | [fig3_three_seed_results.png](figures/fig3_three_seed_results.png) |
| Fig. 4: illumination and corrected sequence bootstrap | [fig4_illumination_bootstrap.png](figures/fig4_illumination_bootstrap.png) |
| Fig. 5: late fusion, efficiency and detection states | [fig5_late_fusion_detection_states.png](figures/fig5_late_fusion_detection_states.png) |
| Fig. 6: depth and LLVIP | [fig6_depth_llvip.png](figures/fig6_depth_llvip.png) |

Fig. 4 and Fig. 6 use the corrected intervals and original-test point differences. Fig. 6a–c
retain training-framework evaluations; Fig. 6d uses COCO evaluation of saved depth predictions.
Older `manuscript_v8_JSC.docx`, `manuscript_v4.docx`, `results_data_tables.docx`,
`fig3_seed_results.png` and manuscript-generator scripts are historical artifacts; they do not
define the current v9 bootstrap results or declarations.

## Notes / limitations (mirrored from the manuscript)

- The projected-depth study uses label-informed calibration and is an exploratory extension, not a
  claim of three-modal superiority.
- THOP operation counts over the custom gate module are provisional (manual gate accounting ≈ 9.8 M MACs).
- Per-class qualitative transitions are threshold-dependent (conf 0.25, IoU 0.5, one-to-one matching).

## License

Code is provided for research purposes. R-LiViT and LLVIP remain subject to their own terms
(see `data/README.md`). Please contact the authors for derived depth images and checkpoints.

## Submission declarations

The v9 draft states no funding and no competing interests. It describes secondary use of public
datasets, no new participant recruitment, and no required ethics approval as confirmed by the
authors. Dataset acknowledgments and official access/license references are included. Author names,
affiliations, corresponding-author details and contributions remain blank pending final confirmation.
No supplementary files are designated for submission. Raw datasets are obtained from the providers;
checkpoints and derived depth images are available on request. See [dataset terms](data/README.md).
