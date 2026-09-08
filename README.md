# Revisiting Lightweight RGB–Thermal Fusion for Roadside Object Detection under Sequence-Disjoint Evaluation

Code, splits, and recorded results accompanying the working manuscript (see `paper/manuscript_v4.docx`).

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
- For the fixed seed-0 checkpoint pair, a sequence-level bootstrap over the 40 held-out test sequences
  gives a GBF nighttime mAP50 interval excluding zero (+4.0, 95% CI [+0.2, +9.1]) and a positive
  mAP50–95 interval (+1.0, [+0.2, +2.0]).
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

All recorded numbers are in `results/`; each `eval_*.json` is the frozen-test COCO evaluation of the
development-selected checkpoint for one seed, and is the single source of truth for the manuscript.

## Repository layout

```
configs/    dataset yamls (paths relative to repo root; put datasets/ here)
splits/     sequence-disjoint manifests + partition reports (frame stems only)
src/        training / evaluation / analysis scripts (paths relative to repo root)
results/    recorded evaluation JSONs (eval_*, bootstrap_seq, qual_*, eff_v2, gsw_latency, summaries)
paper/      manuscript_v4.docx, results_data_tables.docx, final_results_report.md
figures/    figures embedded in the manuscript
data/       dataset links — raw data is NOT redistributed
```

## Reproduce

1. Download R-LiViT and LLVIP (see `data/README.md`) and lay them out under `datasets/` as the yamls
   expect (R-LiViT RGB–thermal pairs, LiDAR scans; LLVIP source pairs).
2. `python src/build_sq_split.py` reproduces the sequence-disjoint R-LiViT partitions;
   `src/build_5ch_v2.py` + `src/fixup_multi_sq.py` reproduce the projected-depth subset.
3. Train: `python src/train_rlivit_sq.py sq-4ch 150` (or `sq-4ch-gbf`, `sq-4ch-gsw`, `sq-4ch-tbf`,
   `sq-rgb`, `sq-ir`); set `RLIVIT_SEED=0/1/2`. Three-modal and LLVIP runs use
   `train_multi_sq.py` / `train_llvip_v2.py`.
4. Evaluate frozen test with uniform COCO: `python src/sq_eval.py sq-4ch 0`; late fusion:
   `python src/sq_eval.py --late sq-rgb 0 sq-ir 0`.
5. Analyses: `src/bootstrap_seq.py` (sequence-level bootstrap), `src/qual_breakdown.py` (matched
   qualitative transitions), `src/eff_v2.py` + `src/gsw_latency.py` (latency), `src/archive_protocol.py`
   (environment/checkpoint archive).

Environment used: Ultralytics 8.4.75, Python 3.11.6, PyTorch 2.12.0.dev20260408+cu128,
NVIDIA RTX 5060 Laptop (8,151 MiB). Full per-run records, selected epochs and checkpoint hashes are
archived by `src/archive_protocol.py`.

## Notes / limitations (mirrored from the manuscript)

- The projected-depth study uses label-informed calibration and is an exploratory extension, not a
  claim of three-modal superiority.
- THOP operation counts over the custom gate module are provisional (manual gate accounting ≈ 9.8 M MACs).
- Per-class qualitative transitions are threshold-dependent (conf 0.25, IoU 0.5, one-to-one matching).

## License

Code is provided for research purposes. R-LiViT and LLVIP remain subject to their own terms
(see `data/README.md`). Please contact the authors for derived depth images and checkpoints.

> Working manuscript — internal release for reproducibility review. Author list, affiliations and
> funding statements are pending.
