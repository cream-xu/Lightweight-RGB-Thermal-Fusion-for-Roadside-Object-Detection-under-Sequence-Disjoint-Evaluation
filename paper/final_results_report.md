# R-LiViT 多模态融合：投稿前补做全部完成 — 最终结果报告

2026-09-07 · 所有数字来自序列不相交 held-out test（方案冻结后一次评估），COCO 协议统一口径

## 0. 一句话结论

严格协议下，**64 参数的静态通道混合器（GSW）是轻量早融合的最优单模型**（3 seed 全胜输入条件门控 GBF）；GBF 的夜间/AP50-95 优势在序列层面显著但幅度小；**稀疏投影深度**带来方向为正但不稳定的增益；**双模型晚融合**仍是夜间精度上限；LLVIP 上 GBF 与 concat 打平。

## 1. 主协议（R-LiViT 2400 帧 → 序列不相交 1440/480/480，held-out test 480 帧，168 夜帧）

### 1.1 三 seed 主对比

| 模型 | seed0 | seed1 | seed2 | 均值 (AP50 / AP50-95) |
|---|---|---|---|---|
| concat | 0.6088/0.3878 | 0.5983/0.3895 | 0.5743/0.3732 | 0.5938 / 0.3835 |
| GBF | 0.6113/0.3978 | 0.5944/0.3855 | 0.5877/0.3946 | 0.5978 / 0.3926 |
| **GSW** | **0.6316**/0.4033 | 0.6048/0.3945 | 0.6231/0.4044 | **0.6198 / 0.4007** |
| TBF (s0) | 0.5967/0.3865 | — | — | — |

**GSW 在全部 3 个种子、两个指标上全胜 GBF 和 concat**。GBF−concat：+0.40 AP50 / +0.91 AP50-95（2/3 种子为正）。

### 1.2 昼夜（168 夜帧）

| 夜间 AP50 | seed0 | seed1 | seed2 | 均值±SD |
|---|---|---|---|---|
| concat | 0.4788 | 0.4777 | 0.3932 | 0.450±0.049 |
| GBF | 0.5104 | 0.4185 | 0.4297 | 0.453±0.050 |
| GSW (s0) | 0.4849 | — | — | — |

**方法间差异（+0.30）远小于种子间波动（±0.05）**——不声称稳定的夜间优势；写 *"the mean method difference was substantially smaller than the observed training-seed variability"*。

### 1.3 序列级 bootstrap（seed0 模型，1000 次 AP50 / 200 次 AP 重采样）

| 对比 | ΔAP50 [95% CI] | ΔAP50-95 [95% CI] |
|---|---|---|
| GBF − concat 全量 | +0.0036 [−0.009, +0.016] | **+0.011 [+0.003, +0.022]** |
| GBF − concat 夜间 | **+0.041 [+0.002, +0.094]**（99.4% 为正） | **+0.038 [+0.008, +0.075]** |
| GSW − concat 全量 | +0.0055 [−0.007, +0.017] | +0.008 [+0.000, +0.018] |

→ GBF 的夜间 AP50 与 AP50-95 优势对测试序列构成稳健（CI 不含 0）；全量 AP50 优势在噪声内。

### 1.4 同配方晚融合（RGB-only + IR-only，NMS 合并）

| | full AP50 | day | night |
|---|---|---|---|
| s0 | 0.5887 | 0.6103 | 0.5432 |
| s1 | 0.5842 | 0.6000 | 0.5734 |
| 夜均值 | | | **0.5583**（比单模型最优夜间高 ~+7~10 点） |

→ 精度-效率权衡：晚融合（双模型 ~2× 参数）是夜间精度上限；GSW/GBF 单模型以一半参数换取差距。

### 1.5 每类 AP50（seed0）与定性分层

- 每类：GSW 在 Pedestrian 0.681 / Car 0.815 / Cyclist 0.593 全家族最高；GBF 的 Pedestrian 0.679 > concat 0.658。
- 定性（夜间 test，一对一匹配，conf 0.25，IoU 0.5）：**improved 144 vs degraded 70（≈2:1）**，集中在 行人 78:38 / 汽车 58:27 / 骑行者 8:4，且几乎全部为小目标（small 141:68，medium/large 无差异）。

## 2. 三模态（LiDAR 深度，782 帧序列不相交 473/155/154，标定 label-informed 披露）

| 模型 | seed0 | seed1 | seed2 |
|---|---|---|---|
| 4ch concat | 0.3773 | 0.3942 | 0.3880 |
| 5ch concat（+深度） | 0.3902 | 0.3860 | 0.3983 |
| matched Δ | +1.29 | −0.82 | +1.03 |
| 4ch GBF | 0.3933 | — | — |
| 5ch TBF（等权三分支） | 0.3998 | — | — |
| 5ch GBF（门控三分支） | 0.3949 | 0.3964 | — |

→ 深度：均值 +0.50，2/3 种子为正；序列级 bootstrap ΔAP50 +0.018 [−0.024, +0.069]（13 测试序列，噪声大）。**口径：方向为正、种子间不完全一致**；门控相对等权无增益（与双分支 GSW 结论方向一致）。

## 3. LLVIP（train 11025 / dev 1000 选 ckpt / 完整 3463 held-out test，2 seeds）

| | concat | GBF | Δ |
|---|---|---|---|
| mAP50 | 0.9486 / 0.9560（0.9523） | 0.9461 / 0.9556（0.9509） | **−0.15** |
| mAP50-95 | 0.5843 / 0.5914（0.5879） | 0.5881 / 0.6074（0.5978） | **+0.99** |

→ 跨数据集打平（external validation：*GBF does not consistently outperform concatenation on a second RGB–infrared dataset*）。不写"saturated"（无实验支持）。

## 4. 效率

| | concat | GBF | 晚融合 |
|---|---|---|---|
| 参数量 | 2,591,349 | 2,607,893 (+0.64%) | ~2× |
| FLOPs | 3.24G | 3.54G (+9.4%) | ~2× |
| 模型 FPS (fp16, 纯前向) | 95.2 | 84.9 | — |
| **端到端**（磁盘读 2 文件+letterbox+前向+NMS） | 中位 36.5ms / 均值 59.5ms | 中位 37.7ms / 均值 39.1ms | 均值 76.6ms |

- 端到端时延由磁盘 I/O 主导（均值/中位差距）；对路侧 5Hz 相机（200ms 预算）全部余量充足。
- 门控路径手动 MAC 账 9.8M（≈模型 0.3%）；**thop 对含自定义门控的模块计数不可靠**（667M vs 手动 9.8M），论文中 FLOPs 需注明 counting methodology。

## 5. 协议审计结论（方法论贡献）

- 旧协议序列划分本身**无重叠**（160/40 官方划分，序列 ID 审计确认）；旧夜间 +6.9~10.0 的虚增来源为**在报告集上选择 checkpoint**（checkpoint-selection bias）——换 dev 选 ckpt + held-out test 后收敛为 +0.30。实证表明该偏差可虚增子集增益 ~6.6pp。

## 6. 论文主结论（可入 Discussion）

> Modality-specific shallow encoding is useful, but input-conditioned gating is not consistently superior to a much simpler static channel weighting; sparse projected LiDAR depth provides an additional (if seed-dependent) signal, and two-model late fusion retains a nighttime accuracy advantage at substantially greater model cost.

## 7. 全部资产清单

- 协议与划分：rlivit_sq/manifest.csv + split_report.json；rlivit_multi_sq/manifest.csv + build_stats.json（含每帧 Δt/覆盖率）；LLVIP dev_manifest.csv
- 结果：runs/rlivit_sq/{summary_sq.json, eval_*.json, dets_*.json, bootstrap_seq.json, qual_report.json, qual_breakdown.json, eff_v2.json}；runs/rlivit_msq/summary.json；runs/llvip_v2/summary.json
- 审计与归档：paper/证据恢复_D1-D6.md、paper/protocol_archive.json（环境/硬件/manifests 哈希/ckpt md5+selected epoch/命令）
- 文档：paper/全部结果汇总_供评估.md（已按协议审计更正）、paper/补做进度_20260828.md
- 待作者：D5 参考文献版本核对、D6 作者元数据/CRediT/funding/disclosure
