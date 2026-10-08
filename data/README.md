# Data sources, permissions and preparation

Raw dataset images, LiDAR scans and annotations are obtained from the original providers.
This repository publishes partition manifests, research code, fitted calibration parameters and
recorded results. It does not redistribute the original datasets or training checkpoints.

## Official sources and terms

| Dataset | Official source | Terms relevant to this study |
|---|---|---|
| R-LiViT | [Dataset DOI](https://doi.org/10.5281/zenodo.16356714), [provider repository](https://github.com/XITASO/r-livit) | Dataset release: CC BY 4.0; provide attribution and cite the dataset publication. Follow the provider's anonymization/privacy documentation. |
| LLVIP | [project page](https://bupt-ai-cz.github.io/LLVIP/), [provider repository](https://github.com/bupt-ai-cz/LLVIP) | [Provider terms](https://github.com/bupt-ai-cz/LLVIP/blob/main/Term%20of%20Use%20and%20License.md): non-commercial use, dataset citation, and no identification of individuals or invasion of privacy. |

Public availability does not waive dataset conditions. The manuscript uses these datasets for
non-commercial academic research and credits their creators. The authors confirmed that no new
participants were recruited or contacted and that ethics approval was not required for this study.

## Published partitions

- [R-LiViT RGB/thermal](../splits/rlivit_sq_manifest.csv): 2,400 pairs; 1,440 training,
  480 development and 480 held-out test, with 120/40/40 disjoint sequences. The test contains
  312 daytime and 168 nighttime frames.
- [Projected-depth subset](../splits/rlivit_multi_sq_manifest.csv): 782 frames;
  473/155/154 training/development/test. This is a separate, label-informed calibration study.
- [LLVIP development list](../splits/llvip_dev_manifest.csv): 1,000 pairs drawn from the official
  training partition with selection seed 42, leaving 11,025 training pairs. The held-out evaluation
  uses all 3,463 official test pairs, including 790 omitted from an earlier prepared copy.
- [Earlier R-LiViT partition](../splits/legacy_rlivit_full_manifest.csv) is historical development
  evidence; it must not be substituted for the current train/dev/test manifest.

Manifests contain frame stems and partition metadata. Match stems to the official downloads and
convert annotations/images into the layouts required by `configs/`; downloading alone does not
produce the prepared YOLO dataset. See the root README for expected directories.

## Derived depth and checkpoints

[calib_per_seq.json](calib_per_seq.json) contains the original fitted per-sequence calibration
parameters. Copy it to `datasets/RLiViT/calib_per_seq.json` for the depth builder. Fitting used
annotation-derived correspondences; these parameters are not independent geometric validation.
Original training checkpoints, saved predictions and derived depth images are available from the
authors on request, subject to applicable dataset terms.
