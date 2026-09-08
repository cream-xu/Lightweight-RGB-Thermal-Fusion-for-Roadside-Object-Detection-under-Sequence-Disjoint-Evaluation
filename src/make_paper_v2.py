# -*- coding: utf-8 -*-
"""论文初稿 v3 生成 (2026-09-07): 落实审计意见 P0/P1/P2
P0-1: GSW数字统一为eval COCO口径(61.54/58.96/60.75), 主claim改为"highest mean, mixed ordering"
P0-2: protocol audit改为sensitivity, 不单独归因checkpoint selection
P1-1..8: LLVIP seed1更正/无statistically措辞/配置级结论/64参数澄清/matched-seed晚融合/depth降调/
        定性四态+率/GSW时延实测
P2-1..3: 标题去LiDAR中心位/删absence claim/嵌3图
输出: paper/论文初稿_v3_审计修订版.docx
"""
import os
from docx import Document
from docx.shared import Pt, Mm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import qn, nsdecls

ROOT = r''
OUT = os.path.join(ROOT, 'paper', '论文初稿_v3_审计修订版.docx')
FONT = 'Times New Roman'
BOLD_C = RGBColor(0x1F, 0x3A, 0x5F)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Mm(210), Mm(297)
sec.left_margin = sec.right_margin = Mm(22)
st = doc.styles['Normal']
st.font.name = FONT
st.font.size = Pt(10.5)


def set_cn(run, size=10.5, bold=False, color=None, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = FONT
    if color is not None:
        run.font.color.rgb = color


def h1(text):
    p = doc.add_paragraph()
    set_cn(p.add_run(text), 13, bold=True, color=BOLD_C)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)


def h2(text):
    p = doc.add_paragraph()
    set_cn(p.add_run(text), 11, bold=True, color=BOLD_C)
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(3)


def para(text, size=10.5, bold=False, align=None, indent=False, italic=False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Pt(18)
    set_cn(p.add_run(text), size, bold, italic=italic)
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15


def eq(text):
    p = doc.add_paragraph()
    set_cn(p.add_run(text), 10.5, italic=True)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)


def figure(path, width_in=6.0, cap=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width_in))
    if cap:
        para(cap, 9, align=WD_ALIGN_PARAGRAPH.CENTER)


def shade(cell, fill):
    cell._tc.get_or_add_tcPr().append(
        parse_xml(r'<w:shd %s w:val="clear" w:fill="%s"/>' % (nsdecls('w'), fill)))


def table(headers, rows, widths=None, fontsize=9, cap=None):
    if cap:
        para(cap, 9.5, bold=True)
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, htxt in enumerate(headers):
        c = t.rows[0].cells[j]
        c.text = ''
        r = c.paragraphs[0].add_run(htxt)
        set_cn(r, fontsize, bold=True)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shade(c, 'D9E2F3')
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.rows[i + 1].cells[j]
            c.text = ''
            r = c.paragraphs[0].add_run(str(v))
            set_cn(r, fontsize, bold=(i == len(rows) - 1 and j > 0))
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if widths:
        for j, wd in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Mm(wd)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)


# ================= 标题 (P2-1) =================
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_cn(p.add_run('Revisiting Lightweight RGB-Thermal Fusion for Roadside Object Detection\n'
                 'under Sequence-Disjoint Evaluation'), 15, bold=True, color=BOLD_C)
p.paragraph_format.space_after = Pt(4)
para('with an exploratory projected-LiDAR depth extension', 10.5,
     align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)
para('Working manuscript v3 for author/advisor review — 7 September 2026. All main results follow a '
     'sequence-disjoint train/dev/test protocol with a frozen held-out test set and a uniform COCO '
     'evaluation; the earlier development-protocol values are retained only in Section 5.11 as a '
     'sensitivity analysis.', 9.5, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)

# ================= Abstract (P0-1/P1-1/P1-2) =================
h1('Abstract')
para('Visible and thermal imagery provide complementary cues for roadside perception, but the value of '
     'increasingly sophisticated fusion mechanisms remains unclear once evaluation is made properly '
     'sequence-independent. This study compares lightweight multi-modal early-fusion configurations for a '
     'YOLO11n-based roadside detector on R-LiViT under a sequence-disjoint protocol: 2,400 frames are split '
     'by sequence into 1,440 training, 480 development, and 480 held-out test frames, checkpoints are '
     'selected on the development partition only, and the frozen test set is evaluated once with a uniform '
     'COCO procedure. Four fusion configurations—input concatenation (Concat), equal-weight dual-branch '
     'fusion (TBF), static learned channel weighting (GSW), and input-conditioned channel gating (GBF)—are '
     'compared across three training seeds, together with a same-recipe two-model late-fusion baseline and '
     'a projected-LiDAR depth extension.')
para('Across three seeds, mean test mAP50 is 59.38 for Concat, 59.78 for GBF, and 60.42 for GSW: GSW '
     'achieves the highest three-seed mean among the single-model variants, but the seed-wise ordering is '
     'mixed, so input-conditioned gating shows no consistent advantage over static channel weighting. GBF '
     'improves mAP50 over Concat by +0.40 percentage points on average (+0.91 for mAP50–95), and its '
     'nighttime differences (+3.2, −5.9, +3.7 points across seeds) are smaller than the observed '
     'training-seed variability. For the seed-0 checkpoint pair, a sequence-level bootstrap over the 40 '
     'held-out test sequences yields a nighttime mAP50 interval that excludes zero (+4.0 points, 95% '
     'interval [+0.2, +9.1]); this interval characterizes sensitivity to the held-out sequence composition '
     'and does not quantify training-seed uncertainty. Sparse projected LiDAR depth changes mAP50 by +1.3, '
     '−0.8, and +1.0 points across matched seeds, while two-model late fusion exceeds the best early-fusion '
     'nighttime mAP50 by about nine points at roughly twice the model cost. On LLVIP with complete '
     '3,463-pair held-out coverage, GBF and Concat show small, metric-dependent differences (mAP50 95.09 '
     'vs. 95.23; mAP50–95 59.78 vs. 58.79). Overall, the evidence supports a bounded design rule: within '
     'the tested dual-branch architecture, a 64-parameter static channel-weighting vector matched or '
     'exceeded the input-conditioned gate in these runs.')
para('Index terms: multispectral imaging; channel statistics; sensor fusion; vulnerable road users; model '
     'efficiency; sequence-disjoint evaluation; R-LiViT.', 9.5)

# ================= §1 Introduction (P2-2/P0-1/P0-2) =================
h1('1. Introduction')
para('Roadside object detection must retain useful representations across substantial illumination changes '
     'while preserving information about diverse road users. R-LiViT provides paired visible and thermal '
     'imagery together with LiDAR data for this setting, including annotations for pedestrians, vehicles, and '
     'other vulnerable road users [1]. Earlier multispectral pedestrian benchmarks demonstrated the value of '
     'combining visible and thermal information, whereas LLVIP focuses on paired low-light imagery [2], [3]. '
     'These datasets frame fusion as a representation-design problem: how much modality-specific processing '
     'should be retained before computation is shared? This study evaluates all local fusion configurations '
     'under a sequence-disjoint train/dev/test protocol with checkpoint selection isolated from held-out '
     'testing.', indent=True)
para('Prior multispectral detectors implement cross-modal interaction through channel weighting, spatial '
     'attention, iterative cross-attention, channel switching, and pooled channel attention [4]–[8]. These '
     'studies establish that learned modality balancing and channel-level interaction are not new in '
     'themselves. Our narrower question is whether independent shallow encoders followed by compact '
     'channel-wise fusion improve a YOLO11n-based roadside RGB–thermal detector relative to local input '
     'concatenation, and—equally importantly—whether the added conditioning complexity of a learned gate '
     'pays for itself relative to a fixed or static mixing rule.')
para('Geometric sensing introduces a separate design axis because camera–LiDAR systems may operate on '
     'different representations and prediction targets. PointPainting augments LiDAR points with projected '
     'image semantics, TransFusion retrieves image information for LiDAR-derived object queries, and '
     'InfraDet3D associates roadside camera and LiDAR detections [9]–[11]. By contrast, the optional '
     'geometric input studied here is a projected depth image and the detector continues to predict '
     'two-dimensional boxes. A third shallow branch is architecturally compatible with the same interface, '
     'but it does not remove calibration or data-coverage concerns. RGB–thermal fusion is therefore the main '
     'experiment, while the depth extension is reported as an exploratory, separately scoped study with '
     'label-informed calibration.')
para('Four comparison configurations share one skeleton: Concat stacks the visible triplet and the thermal '
     'channel at the input; TBF keeps independent dual branches with equal mixing; GSW learns 64 '
     'image-independent sigmoid coefficients; GBF conditions the same coefficients on per-image channel '
     'statistics through a compact perceptron. Two additional references complete the picture: a '
     'same-recipe late fusion of separately trained RGB-only and thermal-only detectors, and, in the depth '
     'study, three-branch equal-weight and gated variants. All configurations are trained and evaluated '
     'identically on the sequence-disjoint protocol.')
para('The principal empirical findings are threefold. First, static channel weighting (GSW) achieves the '
     'highest three-seed mean mAP50 among the single-model variants (60.42, versus 59.78 for GBF and 59.38 '
     'for Concat), but the seed-wise ordering is mixed: the gating perceptron does not consistently earn its '
     'conditioning complexity in this setting. Second, GBF remains a well-characterized variant: its '
     'mAP50–95 advantage over Concat is stable at the sequence level, its nighttime advantage is positive in '
     'two of three seeds with a sequence-bootstrap interval excluding zero for the seed-0 checkpoint pair, '
     'and its qualitative gains concentrate in small pedestrian, car, and cyclist instances. Third, the '
     'cost–accuracy frontier is now explicit: two-model late fusion holds the nighttime accuracy lead at '
     'roughly twice the model cost, sparse projected depth adds a small seed-dependent gain, and LLVIP shows '
     'no consistent difference between GBF and Concat. Together these results characterize lightweight '
     'shallow fusion rather than promote a single winner.')
para('A protocol observation accompanies the results. An earlier development protocol, in which the reported '
     'evaluation partition also selected checkpoints, attributed +6.9 and +10.0 nighttime mAP50 gains to '
     'GBF; under the sequence-disjoint protocol with development-selected checkpoints, the three-seed mean '
     'nighttime difference is +0.3 points. Because the two protocols also differ in training-set size and '
     'composition, reporting sequences, and illumination composition, this reduction cannot be attributed '
     'exclusively to checkpoint selection; it is reported as a sensitivity analysis of the evaluation '
     'protocol (Section 5.11).')
para('Sections 2–4 review related fusion designs, define the protocols, and specify the method. Section 5 '
     'presents the experimental observations, and Sections 6–7 discuss their interpretation and limitations.')

# ================= §2 Related Work (v1原文, 保留) =================
h1('2. Related Work')
h2('2.1. Multispectral datasets and fusion interfaces')
para('Multispectral detection spans different task definitions and evaluation measures. The KAIST benchmark '
     'combines visible and thermal channels for pedestrian detection and evaluates performance across '
     'day/night, scale, and occlusion conditions [2]. Its multispectral ACF baseline augments visible '
     'descriptors with thermal intensity and gradients. KAIST therefore provides historical context for '
     'multispectral sensing, but its handcrafted features and pedestrian miss-rate protocol are not '
     'numerically interchangeable with the multiclass AP evaluation used here.')
para('LLVIP provides spatially registered visible–infrared image pairs for low-light vision, with pedestrian '
     'annotations transferred across aligned modalities [3]. The original paper text describes 16,836 pairs, '
     'whereas later descriptions in CSSA and DaFF use 15,488 pairs and a 12,025/3,463 train/test split '
     '[7], [8]. These discrepancies make release and split identification part of reproducible reporting. '
     'In our local inventory, the source test folders contain 3,463 pairs; an earlier local fusion '
     'evaluation covered only 2,673 of them. The 790 omitted stems are contiguous tail segments of the '
     'source test set produced by a truncated local build, not a documented exclusion rule; Section 3.3 '
     'reconstructs them and evaluates on the complete split.')
para('R-LiViT extends multispectral roadside perception with RGB, thermal, and LiDAR recordings and both '
     'image-space and three-dimensional annotations [1]. Its published RGB–thermal benchmark combines '
     'predictions from separate detectors by NMS, uses a common-field-of-view crop, and reports a three-seed '
     'summary [1]. These choices motivate the study of sensor interaction but do not constitute a '
     'protocol-matched baseline for the present shallow feature-fusion model. Here, modality interaction '
     'occurs before the shared deeper representation, and the primary local comparison is against input '
     'concatenation. Table 9 lists the remaining protocol differences explicitly.')
h2('2.2. Learned modality interaction')
para('MBNet addresses modality imbalance through differential feature exchange and illumination-aware '
     'alignment [4]. Its differential module globally pools a difference feature, predicts tanh channel '
     'weights, and exchanges complementary information through residual paths at multiple ResNet blocks. '
     'This is a relevant precedent for compact pooled descriptors that regulate cross-modal interaction. '
     'GBF instead concatenates per-branch means and sample standard deviations and performs a single shallow '
     'channel mixture. This distinction defines the present implementation; it does not show that a single '
     'fusion location is superior to repeated interaction or that mean/std descriptors outperform '
     'differential statistics.')
para('GAFF predicts intra-modality foreground masks and inter-modality spatial weights with supervision '
     'derived from detection annotations [5]. Its weights therefore operate at individual spatial locations, '
     'whereas GBF broadcasts one mixing coefficient over each feature channel and adds no auxiliary '
     'guidance. A controlled comparison would need to account for both the weighting granularity and the '
     'supervision strategy; a direct comparison of published scores would not isolate the fusion operator.')
para('ICAFusion iteratively refines modality features with paired cross-attention modules and reduces '
     'spatial features to manage attention cost [6]. Both ICAFusion and GBF constrain fusion complexity, but '
     'they do so differently: ICAFusion retains spatial-token interaction, whereas GBF compresses each '
     'feature channel to mean and standard-deviation statistics before predicting a mixture. This motivates '
     'reporting gate dimensions and measured forward time, but it does not support direct speed or accuracy '
     'ranking without a common detector, input pipeline, and hardware configuration.')
para('CSSA combines channel switching with spatial attention [7]. Channel scores are obtained from global '
     'average pooling and a one-dimensional convolution, and selected channels are replaced by their '
     'counterparts from the other modality before spatial weighting. GBF instead uses continuous '
     'complementary coefficients and does not introduce spatial attention at the fusion interface. CSSA is '
     'therefore a relevant candidate for future common-backbone comparison, but its published two-stream '
     'detector is not equivalent to replacing only the operator used in the present shallow interface.')
para('DaFF combines transformer attention with channel-wise attention in a YOLOv5-based multispectral '
     'detector [8]. Its channel module concatenates modality features, applies a 1 x 1 convolution, pools '
     'spatially, and predicts weights through fully connected layers and softmax. This provides another '
     'direct precedent for learned fusion from pooled descriptors. GBF is distinguished here only by its '
     'mean/std descriptor, complementary two-branch coefficients, and shallow shared-detector interface; the '
     'current ablations do not establish superiority over DaFF\'s broader spatial-and-channel design.')
h2('2.3. Camera–LiDAR and roadside geometric fusion')
para('PointPainting projects image-semantic scores onto LiDAR points before three-dimensional detection, '
     'whereas TransFusion uses object queries and spatially modulated cross-attention to retrieve image '
     'information for LiDAR-based predictions [9], [10]. Both retain geometric representations and predict '
     '3D boxes. The present pilot instead converts projected range into an image-grid depth channel and '
     'retains 2D detection. Their published 3D AP results therefore cannot be used to assess the '
     'effectiveness of this 2D depth representation.')
para('InfraDet3D addresses roadside camera–LiDAR fusion through calibration, multiple LiDAR inputs, and '
     'association between monocular and LiDAR detections [11]. Its explicit geometric preparation highlights '
     'the need to state how calibration is obtained. The depth study here uses label-informed fitting and '
     'does not provide an independent calibration-accuracy estimate. Accordingly, the central study remains '
     'the RGB–thermal two-branch detector, while the geometric extension is a constrained implementation '
     'experiment rather than evidence of general three-modal superiority.')
para('Source versions. The method description for [4] was read from arXiv:2008.03043v2 (17 August 2020), '
     '[6] from arXiv:2308.07504v1 (15 August 2023), and [11] from arXiv:2305.00314v1 (29 April 2023). The '
     'bibliography records the corresponding venue publications, but equivalence between the consulted '
     'arXiv versions and the final typeset versions has not been established. [AUTHOR CONFIRMATION '
     'REQUIRED: verify the cited source versions against the final venue versions before submission.] The '
     'remaining literature set is curated rather than exhaustive and should not be interpreted as an '
     'absence claim.')

# ================= §3 Protocol =================
h1('3. Dataset and Experimental Protocol')
h2('3.1. Sequence-disjoint R-LiViT protocol')
para('The main experiment uses 2,400 paired visible and thermal frames from R-LiViT, organized in 200 '
     'sequences of 12 frames each. All partitions are defined at the sequence level: 120 sequences (1,440 '
     'frames) form the training set, 40 sequences (480 frames) the development set, and 40 sequences (480 '
     'frames) the frozen held-out test set. Sequences are ordered by their nighttime-frame fraction and '
     'assigned to the three partitions by a 3:1:1 round-robin, so the illumination composition is '
     'approximately matched across partitions. Training contains 972 daytime and 468 nighttime frames; '
     'development 324/156; test 312/168. No sequence identifier occurs in more than one partition. Table 1 '
     'reports partition sizes and Table 1b the per-class instance counts.')
figure(os.path.join(os.path.join(ROOT, 'paper', 'figures'), 'Codex 图像 2026年9月7日 19_22_39.png'), 5.9,
       'Figure 1. Sequence-disjoint protocol: 200 sequences stratified by nighttime fraction and assigned '
       '3:1:1; checkpoints are selected on dev only; the frozen test set is evaluated once.')
para('Checkpoint selection is isolated from testing: validation during optimization uses the development '
     'partition, best.pt is the epoch maximizing development mAP50–95 (the default Ultralytics fitness '
     'vector w = [0,0,0,1] on [P, R, mAP50, mAP50–95] was recovered from the installed package and is '
     'recorded per run in the archive), and all design choices were frozen before the single evaluation of '
     'the test partition. The final numbers are produced by a uniform pycocotools COCO procedure applied to '
     'the frozen predictions of the development-selected checkpoints (Section 3.5); these evaluation JSON '
     'files are the single source of truth for all reported values.')
h2('3.2. Projected-depth extension')
para('The depth study operates on a separate, sequence-disjoint subset. Sequence pairs are admitted when '
     'their label-informed calibration fit reaches a median residual of at most 60 pixels; 71 pairs qualify, '
     'of which 13 are nighttime sequences. Frames are matched between RGB timestamps and LiDAR scans by '
     'nearest neighbor with a 0.12 s rejection bound; the realized synchronization offsets have median 19.8 '
     'ms and maximum 40.0 ms. After sequence-level 3:1:1 assignment, the subset contains 473 training, 155 '
     'development, and 154 test frames (nighttime 103/24/24). Before spatial filling, projected points cover '
     'a median of 1.78% of pixels, so the input is a sparse, dilation-filled depth image rather than dense '
     'geometry. Calibration itself is fitted from 2D–3D box-bottom correspondences and therefore uses '
     'annotation information; this limitation is explicit, and the depth study is described throughout as '
     'sequence-disjoint detector evaluation with label-informed calibration.')
h2('3.3. LLVIP protocol')
para('The second-dataset experiment uses the complete local LLVIP inventory: 11,025 training pairs, a '
     '1,000-pair development set sampled from the original training partition with seed 42 and recorded in a '
     'manifest, and the full 3,463-pair source test split. The earlier local evaluation had omitted 790 '
     'stems; inventory shows these are contiguous tail segments (24xxxx–26xxxx) of the source test set '
     'produced by a truncated build, and they are now reconstructed byte-identically from the source visible '
     'and infrared files. Checkpoints are selected on the development partition only, and the complete test '
     'split is evaluated once. Concat and GBF are retrained separately on the local LLVIP data; this is not '
     'zero-shot transfer from R-LiViT.')
h2('3.4. Training, checkpoint selection, and measurements')
para('All runs use YOLO11n initialization, 150 configured epochs (100 for LLVIP), input size 640, batch '
     'size 8 (16 for LLVIP), four data-loader workers, and automatic mixed precision. The recorded '
     'environment is Ultralytics 8.4.75, Python 3.11.6, PyTorch 2.12.0.dev20260408+cu128, and an NVIDIA '
     'RTX 5060 Laptop GPU with 8,151 MiB reported memory. The run-argument records specify mosaic = 1.0 '
     'with close_mosaic = 10, fliplr = 0.5, translate = 0.1, scale = 0.5, and HSV settings 0.015/0.7/0.4; '
     'the exact transformations executed on four-channel samples are those logged by the trainer at start-up, '
     'but the behavior of randaugment and erasing on non-three-channel inputs has not been separately '
     'verified. Inspected logs select AdamW with learning rate 0.000909. All per-run arguments, selected '
     'epochs, checkpoint hashes, and evaluation commands are archived (protocol_archive.json).')
h2('3.5. Evaluation procedure')
para('Evaluation reports class-averaged AP at IoU 0.50 (mAP50) and AP averaged over IoU 0.50–0.95 in steps '
     'of 0.05 (mAP50–95), computed with pycocotools on predictions collected at confidence 0.001 with the '
     'standard NMS settings. All AP values are percentages and all differences are percentage points. '
     'Daytime and nighttime values are computed by evaluating the corresponding image lists directly. '
     'Three training seeds (0, 1, 2) are available for the main Concat–GBF–GSW comparison; the remaining '
     'ablations and the single-modality components of late fusion are reported for the seeds indicated in '
     'each table. Two uncertainty analyses are reported: (i) per-seed values with sample standard '
     'deviations, explicitly labeled as descriptive over training seeds, and (ii) a sequence-level bootstrap '
     'for the fixed seed-0 checkpoint pair that resamples the 40 held-out test sequences with replacement '
     '(3,000 iterations for AP50, 1,000 for AP) and recomputes COCO metrics on the resampled frames, '
     'yielding 95% percentile intervals for the sensitivity of method differences to the composition of the '
     'test sequences; this bootstrap does not quantify training-seed uncertainty.')

table(['Partition', 'Sequences', 'Frames', 'Day / Night'],
      [['Train', '120', '1,440', '972 / 468'],
       ['Dev (checkpoint selection)', '40', '480', '324 / 156'],
       ['Test (frozen held-out)', '40', '480', '312 / 168']],
      widths=[52, 26, 22, 30],
      cap='Table 1. Sequence-disjoint R-LiViT partitions.')

table(['Class', 'Train', 'Dev', 'Test'],
      [['Pedestrian', '12,469', '4,361', '4,196'],
       ['Car', '14,971', '5,208', '5,492'],
       ['Cyclist', '2,758', '809', '964'],
       ['Motorcycle', '284', '75', '97'],
       ['Truck', '448', '155', '207'],
       ['Bus', '342', '95', '74'],
       ['Tramway', '210', '47', '57']],
      widths=[42, 30, 24, 24],
      cap='Table 1b. Local instance counts after label conversion (Bicycle and e-scooter merged into Cyclist).')

# ================= §4 Method =================
h1('4. Method')
h2('4.1. Architecture and modality-specific shallow encoding')
para('GBF replaces the shallow input portion of YOLO11n with independent visible and thermal encoders while '
     'retaining a single downstream detector. Each encoder reproduces the first Conv–Conv–C3k2 sequence and '
     'outputs a 64-channel P2 feature map at one-quarter input resolution. Fusion occurs at this P2 '
     'interface; the remaining backbone, multiscale feature aggregation, and detection head are shared. '
     'Thus, GBF changes the representation entering the shared detector without duplicating its downstream '
     'topology. The optional depth branch uses the same interface only in the separate depth protocol.')
figure(os.path.join(os.path.join(ROOT, 'paper', 'figures'), 'Codex 图像 2026年9月7日 19_23_19.png'), 5.9,
       'Figure 2. Architecture. RGB (three channels), thermal (one channel), and the optional projected-'
       'depth channel (dashed, exploratory) enter independent Conv–Conv–C3k2 encoders; two-branch sigmoid '
       'or three-branch softmax fusion occurs at P2/4 before one shared downstream detector.')
para('Let the visible input be I_v and the thermal input be I_t. With encoders E_v and E_t, the shallow maps are')
eq('f_v = E_v(I_v),  f_t = E_t(I_t)          (1)')
para('For an input divisible by four, f_v and f_t have 64 channels at one-quarter resolution. Both branches '
     'use separate parameter objects, including their batch-normalization states, and the fused output has '
     'the same shape as either branch. Equation (1) therefore specifies a compatible replacement for the '
     'original shallow output without increasing the channel width of the shared detector.')
para('Independent shallow encoding permits modality-specific convolutional filters and batch-normalization '
     'states before fusion. This is an architectural property, not an experimentally isolated explanation '
     'for any accuracy change. Relative to Concat, introducing two branches also changes parameter count, '
     'activation paths, and the route by which sensor information reaches P2. TBF preserves the two shallow '
     'branches but fixes their fusion weights, allowing the complete branched configuration to be compared '
     'with Concat; it does not isolate a batch-normalization contribution.')
h2('4.2. Channel statistics and two-branch gating')
para('GBF conditions its mixing vector on spatial statistics of both branch outputs. For a fixed image and '
     'channel, the implementation uses the channel mean and the default sample standard deviation of the '
     'flattened spatial values:')
eq('s = [mean(f_v), std(f_v), mean(f_t), std(f_t)]          (2)')
para('Statistics are computed independently for each image and channel. The resulting gate is therefore '
     'input-conditioned at the image level but spatially uniform within each channel. A two-layer '
     'perceptron with a 16-unit hidden layer predicts 64 visible-branch coefficients:')
eq('w = sigma( MLP(s) ),   MLP: 256 -> 16 -> 64          (3)')
para('The thermal coefficient is determined as the complement of the visible coefficient, so no second '
     'independently scaled thermal vector is required. This parameterization fixes the total mixing weight '
     'to one for each feature channel. Fusion then broadcasts each channel coefficient over the spatial grid:')
eq('y = w * f_v + (1 - w) * f_t          (4)')
para('The shared detector D receives the fused tensor and produces the usual two-dimensional detection '
     'outputs. Equation (4) combines learned features rather than raw image brightness. Moreover, a '
     'coefficient is not a calibrated probability that one sensor is correct: its effect depends on the '
     'amplitudes and meanings of both encoded channels. Consequently, the diagnostic analysis measures the '
     'coefficients themselves and their intervention response, without interpreting them as illumination '
     'labels or direct sensor reliability estimates.')
para('The gate contains 5,200 trainable parameters: 4,096 weights and 16 biases in the first linear layer, '
     'followed by 1,024 weights and 64 biases in the second. These parameters are additional to those '
     'introduced by the modality-specific shallow encoders. Because the descriptor is spatially pooled '
     'before the perceptron, the gate output dimension depends on channel count rather than image area, '
     'although computing the statistics and applying the weighted sum still incur spatial operations. The '
     'complete model-level parameter increase is measured directly in Table 7.')
h2('4.3. Initialization, optimization, and precision')
para('Initialization reuses pretrained YOLO11n modules. The visible first convolution takes the first three '
     'input-channel slices of the loaded weights, whereas the thermal first convolution uses the first '
     'input-channel slice; the remaining shallow modules are deep-copied. When depth is enabled, its first '
     'convolution follows the same single-channel initialization rule and its later shallow modules are '
     'likewise copied. These branches subsequently optimize independent parameters. The initialization '
     'reuses pretrained components but does not preserve the original RGB detector function exactly after '
     'the input and fusion structure is changed.')
para('For GBF, the weights of the final perceptron layer are multiplied by 0.1 at initialization and its '
     'biases are set to zero. This encourages two-branch coefficients near 0.5 (and three-branch softmax '
     'coefficients near one third) without guaranteeing exact equality for every input because the output '
     'weights remain nonzero. TBF instead fixes equal weights explicitly, while GSW initializes its '
     'image-independent raw coefficients to zero before applying a sigmoid. These choices provide comparable '
     'starting points while preserving the distinction among fixed, static-learned, and input-conditioned '
     'fusion.')
para('The replacement module is installed when the trainer builds the model and before optimizer '
     'construction. The implementation merges the original first three modules into the fusion module and '
     'updates downstream layer indices, saved-feature indices, and detection-path references. This ensures '
     'that the gate and branch parameters are part of the trained model. The detector loss is unchanged; no '
     'auxiliary illumination label or gate-supervision objective is introduced. Accordingly, the coefficients '
     'are optimized only through the detection objective, and their semantic behavior must be assessed '
     'empirically rather than assumed.')
para('Gate statistics and perceptron operations are computed in FP32 with autocasting disabled locally. '
     'Descriptors and perceptron weights are cast for these calculations, after which the mixing coefficients '
     'are converted back to the branch-feature dtype before multiplication and summation. This is a '
     'numerical precision measure rather than evidence that all diagnostic and ablation paths execute '
     'identically.')
h2('4.4. Projected-depth third branch')
para('The depth study converts a LiDAR point into camera coordinates using a sequence-specific rotation R '
     'and camera position p. With camera-coordinate components, the builder uses a single focal parameter f '
     'and the image center as principal point:')
eq('u = f x/z + W/2,  v = f y/z + H/2          (5)')
para('Points must satisfy z > 0.3 and project inside the image. Coordinates are discretized into pixels, '
     'and the closest projected point supplies depth when multiple points reach the same pixel. Equation (5) '
     'documents the implemented projection model; its geometric adequacy is not established by the detector '
     'AP measurements.')
para('Sparse projected depths are expanded into empty pixels for up to 12 iterations with a 5 x 5 dilation '
     'kernel. Existing occupied pixels are retained, while newly filled pixels inherit the dilation value '
     'from neighboring valid depths. The resulting representation is therefore a projected and spatially '
     'filled depth image rather than dense measured geometry or learned depth completion. The encoded depth '
     'is quantized to an 8-bit image with exponential normalization:')
eq('d = exp(-z / 30)          (6)')
para('No separate validity mask distinguishes directly projected values from dilation-filled values, so '
     'both enter the detector through the same channel. This design choice belongs to the depth study and '
     'must remain visible when interpreting the depth results.')
para('The third encoder produces f_d at the same P2 resolution. Concatenating mean and standard-deviation '
     'vectors from all three branches gives a 384-dimensional descriptor. The perceptron maps 384 inputs '
     'through 16 hidden units to 192 logits, reshaped as three sets of 64 values. Per-channel softmax '
     'normalization and fusion are')
eq('w = softmax(MLP(s)),  y = w_v f_v + w_t f_t + w_d f_d          (7)')
para('For each feature channel, the three branch coefficients sum to one. The shared detector still '
     'predicts two-dimensional boxes, so the added depth branch does not change the task into 3D detection. '
     'The equal-weight three-branch control fixes each branch coefficient to one third.')
h2('4.5. Comparison configurations')
para('Concat stacks the visible triplet and thermal channel at the input and shares all subsequent feature '
     'extraction. TBF keeps separate shallow encoders but fixes the fusion weights at 0.5/0.5, whereas GSW '
     'learns 64 image-independent sigmoid coefficients. GBF uses the input-conditioned gate in Equation (3). '
     'Because these variants differ in several architectural and computational properties, their results are '
     'configuration-level comparisons rather than isolation of a single mechanism. The five-channel study '
     'separately compares input concatenation, equal three-branch fusion, and Equation (7) under its own '
     'sequence-disjoint allocation. The same-recipe late-fusion reference trains the RGB-only and '
     'thermal-only YOLO11n detectors on the same partitions and merges their test predictions by per-class '
     'NMS at IoU 0.6.')

# ================= §5 Experiments (审计修订) =================
h1('5. Experiments')
h2('5.1. Three-seed main comparison')
para('Table 2 reports the frozen held-out test results for Concat, GBF, and GSW across three training '
     'seeds, uniformly from the COCO evaluation files. GBF is higher than Concat in two of three seeds on '
     'mAP50 (differences +0.25, −0.39, +1.34 points) and in two of three on mAP50–95 (+1.00, −0.40, +2.14 '
     'points); the three-seed means differ by +0.40 and +0.91 points. GSW achieves the highest three-seed '
     'mean among the single-model variants (60.42 mAP50 and 39.40 mAP50–95, versus 59.78/39.26 for GBF and '
     '59.38/38.35 for Concat), but the seed-wise ordering is mixed: GSW seed-1 mAP50 (58.96) is below GBF '
     '(59.44), and GSW seed-0 mAP50–95 (39.67) is below GBF (39.78). Input-conditioned gating therefore '
     'shows no consistent advantage over static channel weighting. The two-run summary of the earlier '
     'development protocol is not carried into this table; the old values appear only in the Section 5.11 '
     'sensitivity analysis.')
figure(os.path.join(os.path.join(ROOT, 'paper', 'figures'), 'Codex 图像 2026年9月7日 19_23_46.png'), 5.9,
       'Figure 3. Paired three-seed test results (COCO evaluation, frozen held-out test).')
table(['Model', 'mAP50 (s0/s1/s2)', 'mean mAP50', 'mAP50–95 (s0/s1/s2)', 'mean mAP50–95'],
      [['Concat', '60.88 / 59.83 / 57.43', '59.38', '38.78 / 38.95 / 37.32', '38.35'],
       ['GBF', '61.13 / 59.44 / 58.77', '59.78', '39.78 / 38.55 / 39.46', '39.26'],
       ['GSW', '61.54 / 58.96 / 60.75', '60.42', '39.67 / 38.93 / 39.61', '39.40'],
       ['GBF − Concat', '+0.25 / −0.39 / +1.34', '+0.40', '+1.00 / −0.40 / +2.14', '+0.91']],
      widths=[28, 48, 22, 48, 26],
      cap='Table 2. Frozen held-out test results, three training seeds, AP in percent, uniform COCO '
          'evaluation (single source of truth). Differences are within-seed GBF minus Concat.')
h2('5.2. Architectural ablations')
para('Table 3 reports seed-0 ablations on the same held-out test. TBF, the equal-weight dual-branch '
     'control, reaches 59.67/38.65, i.e., −1.21 points on mAP50 and −0.13 on mAP50–95 relative to Concat: '
     'the branched structure alone, without learned mixing, does not help aggregate accuracy. GSW at '
     '61.54/39.67 and GBF at 61.13/39.78 both exceed Concat, and their mutual ordering is mixed (GSW leads '
     'mAP50 by +0.41; GBF leads mAP50–95 by +0.11). These are configuration-level comparisons: they do not '
     'decompose the changes into pure classification versus localization effects, and they do not isolate '
     'whether the gain arises from branch separation, learned weighting, or their interaction. The '
     'shared-stem image-statistics gate (IMG) of the earlier development protocol is not retrained here; '
     'its recorded development-protocol result (58.43/37.62 vs. Concat 58.92/37.48) showed no aggregate '
     'benefit and motivated the branched design.')
table(['Model', 'mAP50', 'mAP50–95', 'Day mAP50', 'Night mAP50'],
      [['Concat', '60.88', '38.78', '62.11', '47.88'],
       ['TBF (w=0.5 fixed)', '59.67', '38.65', '61.73', '45.50'],
       ['GSW (static learned)', '61.54', '39.67', '62.87', '48.49'],
       ['GBF (input-conditioned)', '61.13', '39.78', '62.57', '51.04']],
      widths=[52, 22, 24, 24, 24],
      cap='Table 3. Seed-0 ablations, frozen held-out test, AP in percent. GBF leads the nighttime subset '
          'but not the aggregate.')
h2('5.3. Daytime, nighttime, and class results')
para('Direct illumination-subset evaluation (Table 4) shows that the mean method differences are small '
     'relative to seed variability. Nighttime mAP50 for GBF is 51.04/41.85/42.97 across seeds versus '
     '47.88/47.77/39.32 for Concat: differences of +3.2, −5.9, and +3.7 points, with a three-seed mean of '
     '+0.3 points. The seed-level sample standard deviations are about 4.9–5.0 points for both models, so '
     'the mean method difference is an order of magnitude smaller than the observed training-seed '
     'variability. A nighttime advantage for GBF is therefore not a stable finding across seeds; it is '
     'reported as such, with the sequence-level bootstrap of Section 5.5 providing the complementary '
     'question of sensitivity to test-sequence composition. GSW nighttime values are 48.49/44.94/49.63 '
     '(mean 47.69).')
para('Class-level outcomes are heterogeneous (Table 4b). On the seed-0 checkpoint, GBF is higher than '
     'Concat on Pedestrian (+2.1), Cyclist (+1.6), and Bus (+2.1), comparable on Car and Tramway, and lower '
     'on Motorcycle (−2.4), Truck (−2.3). The exceptions show that the aggregate differences are not uniform '
     'across classes.')
table(['Night mAP50', 'seed0', 'seed1', 'seed2', 'mean ± SD'],
      [['Concat', '47.88', '47.77', '39.32', '45.0 ± 4.9'],
       ['GBF', '51.04', '41.85', '42.97', '45.3 ± 5.0'],
       ['GSW', '48.49', '44.94', '49.63', '47.7 ± 2.4'],
       ['GBF − Concat', '+3.2', '−5.9', '+3.7', '+0.3']],
      widths=[34, 24, 24, 24, 34],
      cap='Table 4. Nighttime mAP50 by seed, frozen held-out test (168 frames), AP in percent. SD is the '
          'sample standard deviation over three training seeds.')
table(['Class', 'Concat', 'TBF', 'GSW', 'GBF'],
      [['Pedestrian', '65.8', '67.3', '68.1', '67.9'],
       ['Car', '80.1', '80.8', '81.5', '80.8'],
       ['Cyclist', '57.5', '57.8', '59.3', '59.1'],
       ['Motorcycle', '41.1', '37.9', '39.3', '38.7'],
       ['Truck', '61.4', '58.8', '60.7', '59.1'],
       ['Bus', '56.2', '50.1', '58.9', '58.3'],
       ['Tramway', '64.1', '65.1', '63.0', '64.0']],
      widths=[40, 24, 24, 24, 24],
      cap='Table 4b. Per-class AP50 on the frozen held-out test (seed 0), in percent.')
h2('5.4. Gate coefficient behavior and the static-mixer observation')
para('The gate coefficient diagnostics use the development-protocol GBF checkpoint and the diagnostic '
     'preprocessing described in Section 3.5: a maximum absolute day–night difference of about 0.018 in '
     'channel-mean coefficients, and a mean cross-image channel standard deviation of about 0.004. The '
     'learned coefficients therefore vary far more across channels than across images. Replacing the gate '
     'with its sample-mean channel vector changes the full evaluation result by −0.16 points mAP50 '
     '(61.20 to 61.04), and nighttime mAP50 from 54.34 to 53.55, in the development protocol. These '
     'observations suggested that static per-channel weights could retain most of the gated behavior. Under '
     'the sequence-disjoint protocol, GSW achieves the highest three-seed mean among the single-model '
     'variants (Table 2), consistent with that suggestion, although the seed-wise ordering is mixed. The '
     'diagnostic chain—weakly input-varying coefficients, a small mean-substitution effect, and a separately '
     'trained static model with the highest three-seed mean—motivates, without proving, the conclusion that '
     'input conditioning is unnecessary in this setting. The earlier heuristic scalar interventions on TBF '
     '(day weight 0.8 and night weight 0.3 both degrade the TBF base) remain a caution against interpreting '
     'the gate as a modality switch; they are development-protocol observations and are not carried into '
     'the main tables.')
h2('5.5. Uncertainty: sequence-level bootstrap')
para('Table 5 reports sequence-level bootstrap intervals over the 40 held-out test sequences for the fixed '
     'seed-0 checkpoint pair (3,000 resamples for AP50, 1,000 for AP). For GBF minus Concat, the full-test '
     'mAP50 interval [−0.8, +1.7] points includes zero, whereas the mAP50–95 interval [+0.2, +2.0] does '
     'not, and the nighttime intervals are positive: +4.0 points mean for mAP50 with 95% interval '
     '[+0.2, +9.1] and +3.9 for mAP50–95 with [+0.7, +7.8]. For GSW minus Concat, the mAP50–95 interval '
     '[+0.1, +1.6] excludes zero while mAP50 spans [−0.7, +1.7]. These intervals characterize '
     'sensitivity to the held-out sequence composition for the reported checkpoint pair; they do not absorb '
     'training-seed uncertainty, and the paper does not label any of these comparisons statistically '
     'significant.')
table(['Comparison (seed-0 checkpoint pair)', 'ΔmAP50 [95% CI]', 'ΔmAP50–95 [95% CI]'],
      [['GBF − Concat, full test', '+0.4 [−0.8, +1.7]', '+1.0 [+0.2, +2.0]'],
       ['GBF − Concat, nighttime', '+4.0 [+0.2, +9.1]', '+3.9 [+0.7, +7.8]'],
       ['GSW − Concat, full test', '+0.6 [−0.7, +1.7]', '+0.8 [+0.1, +1.6]']],
      widths=[60, 50, 50],
      cap='Table 5. Sequence-level bootstrap (3,000 AP50 / 1,000 AP resamples) over 40 held-out test '
          'sequences, fixed seed-0 checkpoints; intervals are 2.5–97.5 percentiles in percentage points.')
h2('5.6. Same-recipe late fusion')
para('Training the RGB-only and thermal-only YOLO11n detectors under the identical protocol and merging '
     'their predictions by NMS gives a two-seed nighttime mAP50 of 54.32 and 57.34 (mean 55.83). Across the '
     'matched seeds 0 and 1, this exceeds GSW nighttime mAP50 by approximately 9.1 percentage points on '
     'average (GSW 48.49/44.94), while its daytime and aggregate AP50 are lower than those of the strongest '
     'early-fusion configurations in these runs (Table 5b). The trade-off is explicit: late fusion holds '
     'the nighttime accuracy lead at roughly twice the parameters and latency, while single-model early '
     'fusion retains competitive aggregate accuracy at half the cost.')
table(['Late fusion (rgb-only + ir-only, NMS)', 'full mAP50', 'day mAP50', 'night mAP50'],
      [['seed 0', '58.87', '61.03', '54.32'],
       ['seed 1', '58.42', '60.00', '57.34'],
       ['mean', '58.65', '60.52', '55.83']],
      widths=[60, 28, 28, 28],
      cap='Table 5b. Same-recipe two-model late fusion on the frozen held-out test, AP in percent.')
h2('5.7. Projected-depth extension')
para('Table 6 reports the sequence-disjoint depth results. Adding the depth branch to Concat changes mAP50 '
     'by +1.29, −0.82, and +1.03 points across the three matched seeds (mean +0.50), so the direction is '
     'positive in two of three seeds but is not consistent enough to claim a stable depth benefit; the '
     'sequence-level bootstrap interval for the seed-0 pair is [−2.6, +6.9] points over the 13 test '
     'sequences. The dual-branch GBF on the same subset reaches 39.33 mAP50, +1.60 above four-channel '
     'Concat. On the five-channel variants, in seed 0 equal-weight three-branch fusion (TBF, 39.98) '
     'slightly exceeds the gated variant (GBF, 39.49); additional matched seeds would be required to '
     'establish a stable ordering. The results demonstrate a functioning three-branch interface and a '
     'seed-dependent depth signal, under label-informed calibration and 1.78% median pre-filling depth '
     'coverage; we describe this as sequence-disjoint detector evaluation with label-informed calibration, '
     'not as a general three-modal superiority result.')
table(['Model (depth subset, held-out test)', 'seed0', 'seed1', 'seed2'],
      [['4ch Concat', '37.73', '39.42', '38.80'],
       ['5ch Concat (+depth)', '39.02', '38.60', '39.83'],
       ['matched Δ', '+1.29', '−0.82', '+1.03'],
       ['4ch GBF', '39.33', '—', '—'],
       ['5ch TBF (equal third)', '39.98', '—', '—'],
       ['5ch GBF (gated third)', '39.49', '39.64', '—']],
      widths=[62, 26, 26, 26],
      cap='Table 6. Projected-depth extension, sequence-disjoint 473/155/154 allocation, mAP50 in percent.')
h2('5.8. Efficiency')
para('Table 7 records the measured costs. GBF stores 2,607,893 parameters versus 2,591,349 for Concat '
     '(+16,544, approximately +0.64%); the GSW weighting vector adds 64 parameters over TBF within the same '
     'dual-branch structure. Model-forward timing at batch one, 640 x 640, FP16 gives 95.2 FPS for Concat, '
     '91.1 for GSW, and 84.9 for GBF. End-to-end measurement on real test frames—two modality files read '
     'from disk, letterbox preprocessing, forward pass, and NMS—gives median 36.5 ms for Concat, 36.4 ms '
     'for GSW, and 37.7 ms for GBF (means 59.5, 61.7, and 39.1 ms, dominated by disk I/O variance); the '
     'two-model late-fusion pipeline measures 76.6 ms mean. All single-model configurations therefore '
     'retain large headroom for the 5 Hz roadside cameras of R-LiViT. The gate path itself accounts for '
     'about 9.8 M MACs by manual accounting (statistics, MLP, and weighted summation), about 0.3% of the '
     'profiled operation count. The THOP profiler reports 3.24 and 3.54 G operations for Concat and GBF, '
     'but its module-level count over the custom gate (667 M) is inconsistent with the manual account and '
     'is retained only as a provisional profiler-reported value.')
table(['Measure', 'Concat', 'GSW', 'GBF', 'Late fusion'],
      [['Parameters', '2,591,349', '2,602,757', '2,607,893 (+0.64%)', '≈ 2×'],
       ['Forward FPS (fp16, bs=1)', '95.2', '91.1', '84.9', '—'],
       ['End-to-end ms (median / mean)', '36.5 / 59.5', '36.4 / 61.7', '37.7 / 39.1', '— / 76.6']],
      widths=[40, 28, 28, 32, 30],
      cap='Table 7. Efficiency on the RTX 5060 Laptop GPU. End-to-end includes disk reads of both modality '
          'files, letterbox, forward, and NMS; late fusion includes the two-model sequential pipeline. '
          'THOP operation counts are provisional (see text).')
h2('5.9. Qualitative evidence with one-to-one matching')
para('On the 168 nighttime test frames, predictions are matched to annotations greedily by descending '
     'confidence with a 0.5 IoU threshold and class consistency, using the seed-0 checkpoints at confidence '
     '0.25. Each ground-truth object receives one of four states: improved (missed by Concat, detected by '
     'GBF), degraded (the reverse), unchanged (detected by both), or missed by both. Table 7b reports '
     'transition counts together with the total support and transition rates for each stratum. Transitions '
     'from missed by Concat to detected by GBF occur 144 times (5.6% of nighttime ground truth), versus 70 '
     '(2.7%) in the opposite direction; 1,749 objects (67.9%) are detected by both and 613 (23.8%) by '
     'neither. The improvement rate concentrates in small instances (6.0% vs. 2.9% degraded, support 2,346) '
     'and in the safety-critical classes Pedestrian, Car, and Cyclist; medium and large strata are '
     'essentially unchanged (support 126 and 104). These counts are threshold-dependent and are reported '
     'with their support, not as claims about a specific mechanism.')
table(['Nighttime GT states (GBF vs Concat)', 'improved', 'degraded', 'unchanged', 'missed by both', 'support'],
      [['All', '144 (5.6%)', '70 (2.7%)', '1,749 (67.9%)', '613 (23.8%)', '2,576'],
       ['Pedestrian', '78', '38', '546', '—', '—'],
       ['Car', '58', '27', '1,096', '—', '—'],
       ['Cyclist', '8', '4', '83', '—', '—'],
       ['small (<0.5% img)', '141 (6.0%)', '68 (2.9%)', '1,532 (65.3%)', '605 (25.8%)', '2,346'],
       ['medium', '2 (1.6%)', '1 (0.8%)', '119 (94.4%)', '4 (3.2%)', '126'],
       ['large (≥2% img)', '1 (1.0%)', '1 (1.0%)', '98 (94.2%)', '4 (3.8%)', '104']],
      widths=[46, 26, 24, 26, 26, 18], fontsize=8.5,
      cap='Table 7b. One-to-one matched nighttime ground-truth state transitions with rates and support, '
          'seed-0 checkpoints, confidence 0.25, IoU 0.5. Small = box area below 0.5% of the image, '
          'large = at least 2%.')
h2('5.10. LLVIP retraining')
para('On the complete 3,463-pair LLVIP test split with development-selected checkpoints, two seeds give '
     '95.23 mean mAP50 and 58.79 mean mAP50–95 for Concat, and 95.09 / 59.78 for GBF (Table 8). Across '
     'both seeds the direction is the same: GBF produces slightly lower mAP50 (−0.25 and −0.04 points) but '
     'higher mAP50–95 (+0.38 and +1.60 points). The differences are therefore small and metric-dependent; '
     'no equivalence, saturation, or transfer interpretation is claimed.')
table(['LLVIP (complete 3,463-pair test)', 'Concat', 'GBF'],
      [['mAP50 seed0 / seed1 (mean)', '94.86 / 95.60 (95.23)', '94.61 / 95.56 (95.09)'],
       ['mAP50–95 seed0 / seed1 (mean)', '58.43 / 59.14 (58.79)', '58.81 / 60.74 (59.78)']],
      widths=[66, 48, 48],
      cap='Table 8. LLVIP retraining with development-selected checkpoints on the complete held-out test '
          'split, 100 epochs, batch 16, AP in percent.')
h2('5.11. Protocol sensitivity analysis')
para('The earlier development protocol used the official 160/40 sequence split, which is itself '
     'sequence-disjoint, but selected best.pt on the same 480-frame partition that was reported. Under that '
     'protocol, GBF showed +2.28/+2.94 mAP50 and +6.89/+9.98 nighttime mAP50 over Concat in seeds 0/1. '
     'Under the sequence-disjoint protocol of this paper—same training recipe, development-selected '
     'checkpoints, frozen test—the corresponding figures are +0.40 points (three-seed mean) and +0.3 '
     'points. The apparent nighttime advantage was substantially smaller under the redesigned protocol. '
     'Because the two protocols also differ in training-set size and composition (1,920 vs. 1,440 frames), '
     'reporting sequences, and illumination composition, the observed reduction cannot be attributed '
     'exclusively to checkpoint-selection bias; the comparison is therefore reported as a sensitivity '
     'analysis of the evaluation protocol. Table 9 retains the comparison with the published R-LiViT '
     'benchmark recipe, which uses an 800 x 600 common-field-of-view crop, three-seed averaging, '
     'YOLOv8-M/RT-DETR-L, and NMS fusion of separate detectors at IoU 0.8 [1]; it is a protocol '
     'comparison, not an accuracy ranking.')
table(['Aspect', 'Published R-LiViT benchmark [1]', 'This work'],
      [['Split', 'Official sequence split, reported per day/night', 'Sequence-disjoint train/dev/test, frozen test'],
       ['Checkpoint selection', 'Not specified in the local record', 'Dev partition only (fitness = dev mAP50–95)'],
       ['Models', 'YOLOv8-M, RT-DETR-L, separate-detector NMS fusion', 'YOLO11n single models; same-recipe late fusion'],
       ['Input / crop', '800 × 600 common field of view', '640 letterbox, full frame'],
       ['Classes', 'Published class set', '7 classes (Bicycle, e-scooter merged into Cyclist)'],
       ['Seeds', '3-seed summary', '3 seeds for main comparison, per-table elsewhere']],
      widths=[36, 62, 62], fontsize=8.5,
      cap='Table 9. Protocol comparison; this is not an accuracy ranking.')

# ================= §6 Discussion (P1-3/P1-4) =================
h1('6. Discussion')
h2('6.1. What the comparisons support')
para('The central finding is a simplicity observation, not a new module. Among the single-model variants, '
     'the dual-branch configuration with a 64-parameter static channel-weighting vector achieves the '
     'highest three-seed mean mAP50 (60.42), while the input-conditioned gate adds no consistent advantage '
     '(59.78), with mixed seed-wise ordering. The earlier gate diagnostics—weak cross-image coefficient '
     'variation, a small mean-substitution effect—had suggested this outcome, and the sequence-disjoint '
     'results are consistent with it. The supported conclusion is configuration-level: the best '
     'configurations combine modality-specific shallow branches with learned channel mixing, and the '
     'present ablations do not isolate whether the gain arises from branch separation, learned weighting, '
     'or their interaction. The equal-weight control (TBF, 59.67) sits below Concat, which cautions against '
     'attributing the benefit to shallow branch separation alone.')
para('GBF itself remains a well-characterized variant rather than a failed one. Its mAP50–95 advantage over '
     'Concat is stable at the sequence level for the reported checkpoint, its nighttime advantage is '
     'positive in two of three seeds with a sequence-bootstrap interval excluding zero for the seed-0 pair, '
     'and its qualitative gains concentrate in small pedestrian, car, and cyclist instances—precisely the '
     'vulnerable-road-user cases that motivate roadside perception. The honest summary is that GBF improves '
     'the categories and scenes that matter most, but not by enough and not consistently enough to justify '
     'its conditioning complexity over static weighting.')
h2('6.2. Efficiency–accuracy frontier')
para('The measured frontier has three corners: two-model late fusion holds the nighttime accuracy lead '
     '(about 9.1 points over matched-seed GSW) at roughly double the parameters and an end-to-end cost of '
     '76.6 ms; static single-model early fusion (GSW) retains the best aggregate accuracy at 2.6 M '
     'parameters, 91.1 forward FPS, and about 36 ms end-to-end; and input-conditioned gating sits between '
     'them on neither axis. For the 5 Hz roadside cameras of R-LiViT, every configuration retains large '
     'real-time headroom, so the choice is driven by accuracy–complexity preference rather than feasibility.')
h2('6.3. Depth, LLVIP, and generalization')
para('The projected-depth extension is reported as exploratory. Sparse dilation-filled depth adds a '
     'seed-dependent positive signal (matched-seed mean +0.50 points, positive in two of three seeds), and '
     'the three-branch interface is operational, but the claim stops there: calibration is label-informed, '
     'pre-filling depth coverage is 1.78%, and the test partition contains only 154 frames. The LLVIP '
     'comparison likewise yields small, metric-dependent differences between GBF and Concat, which is '
     'reported without a saturation or equivalence interpretation.')
h2('6.4. Limitations')
para('(i) Training-seed variability is large relative to method differences: three seeds support descriptive '
     'mean ± SD reporting but not population-level significance claims. (ii) The sequence-level bootstrap '
     'uses the fixed seed-0 checkpoint pair and quantifies test-composition sensitivity, not seed '
     'uncertainty; no result in this paper is labeled statistically significant on that basis. (iii) The '
     'nighttime test subset contains 168 frames and 13–14 nighttime sequences; conclusions specific to '
     'nighttime classes remain noisy. (iv) Depth calibration uses annotation-derived correspondences and '
     'has no independent accuracy assessment. (v) THOP operation counts over the custom gate are not '
     'verified complete, and end-to-end timings exclude sensor transfer and depth projection. (vi) The '
     'comparison set is curated, not exhaustive, and no claim of superiority over published methods is '
     'intended. Each of these limits is part of the reported result rather than hidden in future work.')

# ================= §7 Conclusion (审计修订) =================
h1('7. Conclusion')
para('This study revisits lightweight RGB–thermal early fusion for roadside object detection under a '
     'sequence-disjoint protocol with a frozen held-out test set, three training seeds, and a uniform COCO '
     'evaluation. The main empirical conclusion is bounded and configuration-level: within the tested '
     'dual-branch architecture, a 64-parameter static channel-weighting vector matched or exceeded the '
     'input-conditioned gate in these runs—GSW reaches the highest three-seed mean mAP50 (60.42, versus '
     '59.78 for GBF and 59.38 for Concat) but with mixed seed-wise ordering, so input-conditioned gating '
     'showed no consistent advantage over static channel weighting. GBF\'s remaining strengths are specific '
     'and documented: a sequence-robust mAP50–95 advantage, a nighttime edge that is positive in two of '
     'three seeds with a sequence-bootstrap interval excluding zero for the seed-0 checkpoint pair, and '
     'qualitative gains concentrated in small vulnerable road users. Sparse projected LiDAR depth '
     'contributes a small seed-dependent signal under label-informed calibration, two-model late fusion '
     'retains the nighttime accuracy lead at about twice the model cost, and LLVIP shows small, '
     'metric-dependent differences between GBF and Concat. A protocol sensitivity analysis completes the '
     'study: an earlier development protocol that selected checkpoints on its reporting partition showed a '
     'much larger apparent nighttime gain, and while the reduction cannot be attributed exclusively to '
     'checkpoint selection, it motivates the protocol used throughout this paper. The contribution is a '
     'bounded, reproducible characterization of lightweight shallow fusion—its architecture, its '
     'coefficients, its costs, and its limits—rather than the promotion of a single fusion module.')

# ================= Declarations =================
h1('Declarations')
para('Author names and affiliations: [AUTHOR CONFIRMATION REQUIRED].')
para('Author contributions: [AUTHOR CONFIRMATION REQUIRED: provide CRediT roles for each named author].')
para('Funding: [AUTHOR CONFIRMATION REQUIRED].')
para('Competing interests: [AUTHOR CONFIRMATION REQUIRED].')
para('Ethics and data permissions: [AUTHOR CONFIRMATION REQUIRED: confirm compliance with dataset licenses '
     'and usage conditions].')
para('Data and code availability: R-LiViT and LLVIP are attributed in [1] and [3]. [AUTHOR CONFIRMATION '
     'REQUIRED: state access conditions and the availability of local manifests, training code, '
     'checkpoints, and derived depth inputs].')
para('AI-assisted writing disclosure: This working draft was prepared with assistance from an AI language '
     'model using supplied experimental records, code, and literature extracts. [AUTHOR CONFIRMATION '
     'REQUIRED: verify the full account of tool use, human review and responsibility, and the eventual '
     'venue\'s disclosure requirements].')

# ================= References =================
h1('References')
refs = [
 '[1] J. Mirlach, L. Wan, A. Wiedholz, H. E. Keen, and A. Eich, "R-LiViT: A LiDAR-Visual-Thermal Dataset Enabling Vulnerable Road User Focused Roadside Perception," in IEEE/CVF International Conference on Computer Vision (ICCV), 2025. doi: 10.1109/iccv51701.2025.02635.',
 '[2] S. Hwang, J. Park, N. Kim, Y. Choi, and I. S. Kweon, "Multispectral Pedestrian Detection: Benchmark Dataset and Baseline," in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. doi: 10.1109/cvpr.2015.7298706.',
 '[3] X. Jia, C. Zhu, M. Li, W. Tang, and W. Zhou, "LLVIP: A Visible-infrared Paired Dataset for Low-light Vision," in IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2021. doi: 10.1109/iccvw54120.2021.00389.',
 '[4] K. Zhou, L. Chen, and X. Cao, "Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems," in European Conference on Computer Vision (ECCV), 2020. doi: 10.1007/978-3-030-58523-5_46. Version read: arXiv:2008.03043v2.',
 '[5] H. Zhang, E. Fromont, S. Lefevre, and B. Avignon, "Guided Attentive Feature Fusion for Multispectral Pedestrian Detection," in IEEE Winter Conference on Applications of Computer Vision (WACV), 2021. doi: 10.1109/wacv48630.2021.00012.',
 '[6] J. Shen, Y. Chen, Y. Liu, X. Zuo, H. Fan, and W. Yang, "ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection," Pattern Recognition, vol. 145, article 109913, 2024. doi: 10.1016/j.patcog.2023.109913. Version read: arXiv:2308.07504v1.',
 '[7] Y. Cao, J. Bin, J. Hamari, E. Blasch, and Z. Liu, "Multimodal Object Detection by Channel Switching and Spatial Attention," in IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2023. doi: 10.1109/cvprw59228.2023.00046.',
 '[8] A. Althoupety, L.-Y. Wang, W.-C. Feng, and B. Rekabdar, "DaFF: Dual Attentive Feature Fusion for Multispectral Pedestrian Detection," in IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2024. doi: 10.1109/cvprw63382.2024.00305.',
 '[9] S. Vora, A. H. Lang, B. Helou, and O. Beijbom, "PointPainting: Sequential Fusion for 3D Object Detection," in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. doi: 10.1109/cvpr42600.2020.00466.',
 '[10] X. Bai, Z. Hu, X. Zhu, Q. Huang, Y. Chen, H. Fu, and C.-L. Tai, "TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers," in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. doi: 10.1109/cvpr52688.2022.00116.',
 '[11] W. Zimmer et al., "InfraDet3D: Multi-Modal 3D Object Detection based on Roadside Infrastructure Camera and LiDAR Sensors," in IEEE Intelligent Vehicles Symposium (IV), 2023. doi: 10.1109/iv55152.2023.10186723. Version read: arXiv:2305.00314v1.',
]
for r in refs:
    p = doc.add_paragraph()
    set_cn(p.add_run(r), 9)
    p.paragraph_format.space_after = Pt(2)

doc.save(OUT)
print('saved:', OUT)
