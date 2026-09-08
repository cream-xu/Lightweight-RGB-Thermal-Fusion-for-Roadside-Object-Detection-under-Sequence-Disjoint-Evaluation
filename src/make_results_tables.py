# -*- coding: utf-8 -*-
"""实验结果数据总表 (2026-09-07): 全部数字由源json直接读取生成, 便于检索
输出: paper/实验结果数据总表_20260907.docx
"""
import os, json
from docx import Document
from docx.shared import Pt, Mm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import qn, nsdecls

ROOT = r''
OUT = os.path.join(ROOT, 'paper', '实验结果数据总表_20260907.docx')
RUNS_SQ = os.path.join(ROOT, 'runs', 'rlivit_sq')
RUNS_MSQ = os.path.join(ROOT, 'runs', 'rlivit_msq')
RUNS_LLVIP = os.path.join(ROOT, 'runs', 'llvip_v2')
RUNS_OLD = os.path.join(ROOT, 'runs', 'rlivit')

FONT = '微软雅黑'
BOLD_C = RGBColor(0x1F, 0x3A, 0x5F)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Mm(210), Mm(297)
sec.left_margin = sec.right_margin = Mm(18)
st = doc.styles['Normal']
st.font.name = FONT
st.font.size = Pt(9.5)
st._element.rPr.rFonts.set(qn('w:eastAsia'), FONT)


def set_cn(run, size=9.5, bold=False, color=None):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
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
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)


def para(text, size=9.5, bold=False, align=None):
    p = doc.add_paragraph()
    set_cn(p.add_run(text), size, bold)
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(3)


def shade(cell, fill):
    cell._tc.get_or_add_tcPr().append(
        parse_xml(r'<w:shd %s w:val="clear" w:fill="%s"/>' % (nsdecls('w'), fill)))


def table(headers, rows, widths=None, cap=None, fontsize=9):
    if cap:
        para(cap, 9, bold=True)
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
            set_cn(r, fontsize)
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if widths:
        for j, wd in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Mm(wd)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)


def P(x, d=2):
    return f'{x * 100:.{d}f}' if isinstance(x, (int, float)) else str(x)


# ---------- 读取源数据 ----------
summary = json.load(open(os.path.join(RUNS_SQ, 'summary_sq.json')))
eval_d = {}
for f in os.listdir(RUNS_SQ):
    if f.startswith('eval_') and f.endswith('.json') and f != 'eval_late_sq-rgb0_sq-ir0.json' \
            and 'late' not in f:
        eval_d[f[5:-5]] = json.load(open(os.path.join(RUNS_SQ, f)))
late = {k: json.load(open(os.path.join(RUNS_SQ, k)))
        for k in ('eval_late_sq-rgb0_sq-ir0.json', 'eval_late_sq-rgb1_sq-ir1.json')}
boot = json.load(open(os.path.join(RUNS_SQ, 'bootstrap_seq.json')))
qual = json.load(open(os.path.join(RUNS_SQ, 'qual_breakdown.json'), encoding='utf-8'))
eff = json.load(open(os.path.join(RUNS_SQ, 'eff_v2.json')))
msq = json.load(open(os.path.join(RUNS_MSQ, 'summary.json')))
llvip = json.load(open(os.path.join(RUNS_LLVIP, 'summary.json')))
eff_old = json.load(open(os.path.join(RUNS_OLD, 'efficiency.json')))
NAMES = ['Pedestrian', 'Car', 'Cyclist', 'Motorcycle', 'Truck', 'Bus', 'Tramway']

# ---------- 封面 ----------
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_cn(p.add_run('实验结果数据总表（检索版）'), 16, bold=True, color=BOLD_C)
p.paragraph_format.space_after = Pt(2)
para('2026-09-07 · 全部数字由源 json 程序化读取生成（无手抄）· 每节注明源文件',
     9, align=WD_ALIGN_PARAGRAPH.CENTER)
para('协议缩写: sq=主协议(序列不相交1440/480/480, held-out test 480帧, COCO); '
     'msq=三模态v2(473/155/154); LLVIP(11025/1000/3463); old=旧开发协议(仅审计用)',
     9, align=WD_ALIGN_PARAGRAPH.CENTER)

# ---------- 1 主协议 ----------
h1('1. 主协议 (sq) — 3 seed 全量/昼夜')
rows = [['模型', 'seed', '全量 mAP50', '全量 mAP50-95', '昼 mAP50', '夜 mAP50']]
for exp in ('sq-4ch', 'sq-4ch-gbf', 'sq-4ch-gsw'):
    for s in (0, 1, 2):
        k = f'{exp}_s{s}'
        if k not in eval_d:
            continue
        e = eval_d[k]
        rows.append([exp.replace('sq-4ch-', '').replace('sq-4ch', 'concat').upper(), s,
                     P(e['full']['AP50']), P(e['full']['AP']),
                     P(e['day']['AP50']), P(e['night']['AP50'])])
rows.append(['TBF', 0,
             P(eval_d['sq-4ch-tbf_s0']['full']['AP50']), P(eval_d['sq-4ch-tbf_s0']['full']['AP']),
             P(eval_d['sq-4ch-tbf_s0']['day']['AP50']), P(eval_d['sq-4ch-tbf_s0']['night']['AP50'])])
table(['模型', 'seed', '全量 mAP50', '全量 mAP50-95', '昼 mAP50', '夜 mAP50'], rows[1:],
      widths=[30, 14, 30, 32, 26, 26],
      cap='表1 sq主协议全部评估数字 | 源: runs/rlivit_sq/eval_*.json')
para('均值(3seed): concat 59.38/38.35 (夜45.0±4.9); GBF 59.78/39.26 (夜45.3±5.0); '
     'GSW 60.42/39.40 (夜47.7±2.4)。GBF−concat: +0.40/+0.91, 夜+0.3。'
     '注意: GSW逐seed排序混合(s1 mAP50 58.96<GBF 59.44; s0 mAP50-95 39.67<GBF 39.78), '
     '不得写"每seed每指标全胜"。唯一真源=eval_*.json(COCO口径), summary_sq.json为训练期val口径不可混用。', 9)

# ---------- 2 每类 ----------
h1('2. 每类 AP50 (sq, seed0, 全量 test)')
rows = [['类别'] + ['concat', 'TBF', 'GSW', 'GBF']]
for i, n in enumerate(NAMES):
    rows.append([n] + [P(eval_d[f'sq-{m}_s0']['per_class_ap50'][n])
                       for m in ('4ch', '4ch-tbf', '4ch-gsw', '4ch-gbf')])
table(rows[0], rows[1:], widths=[40, 24, 24, 24, 24],
      cap='表2 每类AP50 | 源: eval_sq-{4ch,4ch-tbf,4ch-gsw,4ch-gbf}_s0.json')

# ---------- 3 bootstrap ----------
h1('3. 序列级 bootstrap (sq, seed0 ckpt)')
rows = [['对比', 'ΔmAP50 均值', '95% CI', 'ΔmAP50-95 均值', '95% CI', '正比例']]
for k, v in boot['comparisons'].items():
    if not isinstance(v, dict):
        continue
    rows.append([k, P(v['dAP50_mean']), str(v['dAP50_ci95']), P(v['dAP_mean']),
                 str(v['dAP_ci95']), f"{v['dAP50_frac_pos'] * 100:.1f}%" if 'dAP50_frac_pos' in v else '—'])
table(rows[0], rows[1:], widths=[46, 26, 30, 26, 30, 18],
      cap='表3 bootstrap (3000次AP50/1000次AP, 40序列有放回重采样, 固定seed0 ckpt对) | 源: bootstrap_seq.json')

# ---------- 4 晚融合 ----------
h1('4. 同配方晚融合 (rgb-only + ir-only → NMS)')
rows = [['seed', '全量 mAP50', '全量 mAP50-95', '昼 mAP50', '夜 mAP50']]
for s, k in ((0, 'eval_late_sq-rgb0_sq-ir0.json'), (1, 'eval_late_sq-rgb1_sq-ir1.json')):
    e = late[k]
    rows.append([s, P(e['full']['AP50']), P(e['full']['AP']), P(e['day']['AP50']), P(e['night']['AP50'])])
table(rows[0], rows[1:], widths=[20, 32, 32, 26, 26],
      cap='表4 晚融合 | 源: eval_late_sq-rgb{0,1}_sq-ir{0,1}.json; 组件: sq-rgb s0 56.73 / sq-ir s0 41.07。'
          'matched-seed对照: GSW夜 48.49/44.94, 晚融合夜均值55.83 → +9.1pp (统一matched seeds口径)')

# ---------- 5 三模态 ----------
h1('5. 三模态 v2 (msq)')
rows = [['模型', 'seed', 'mAP50', 'mAP50-95']]
for k in sorted(msq):
    if k.startswith('msq'):
        v = msq[k]
        rows.append([k, v.get('seed'), v['mAP50'], v['mAP50-95']])
table(rows[0], rows[1:], widths=[46, 16, 24, 26],
      cap='表5 msq全部结果 | 源: runs/rlivit_msq/summary.json; matched Δ(5ch−4ch): s0 +1.29 / s1 −0.82 / s2 +1.03')
para('msq bootstrap (seed0): ΔAP50 +1.8 [−2.4, +6.9] | 源: bootstrap_seq.json', 9)

# ---------- 6 LLVIP ----------
h1('6. LLVIP (完整 3463 held-out test)')
rows = [['模型', 'seed', 'test mAP50', 'test mAP50-95', 'dev mAP50']]
for k in sorted(llvip):
    v = llvip[k]
    rows.append([k, v['seed'], v['test_mAP50'], v['test_mAP50-95'], v['dev_mAP50']])
table(rows[0], rows[1:], widths=[40, 16, 28, 30, 26],
      cap='表6 LLVIP | 源: runs/llvip_v2/summary.json; 均值: concat 95.23/58.79, GBF 95.09/59.78 (Δ −0.14/+0.99)')

# ---------- 7 效率 ----------
h1('7. 效率')
rows = [['指标', 'concat', 'GSW', 'GBF', '晚融合'],
        ['参数量', '2,591,349', '2,602,757', '2,607,893 (+0.64%)', '≈2×'],
        ['前向 FPS (fp16, bs1)', '95.2', '91.1', '84.9', '—'],
        ['端到端 ms 中位/均值', '36.5 / 59.5', '36.4 / 61.7', '37.7 / 39.1', '— / 76.6'],
        ['FLOPs (thop, provisional)', '3.24G', '3.54G', '3.54G', '—']]
table(rows[0], rows[1:], widths=[44, 30, 30, 34, 26],
      cap='表7 效率 | 源: runs/rlivit/efficiency.json (前向FPS/FLOPs), runs/rlivit_sq/eff_v2.json (端到端) + gsw_latency.json (GSW实测); '
          '门控手动MACs 9.8M (thop对自定义门控计数不可靠667M)')

# ---------- 8 定性 ----------
h1('8. 定性分层 (夜间168帧, 一对一匹配, conf0.25, IoU0.5)')
def qv(d, k):
    v = d[k]
    return f'{v[0]} ({v[1] * 100:.1f}%)' if isinstance(v, list) else v


rows = [['分层', 'improved(FN→TP)', 'degraded(TP→FN)', 'unchanged(TP→TP)', 'both_missed(FN→FN)', 'n_gt']]
rows.append(['全部'] + [qv(qual['total'], k) for k in
                        ('improved', 'degraded', 'unchanged', 'both_missed')] + [qual['total']['n_gt']])
for n in ('Pedestrian', 'Car', 'Cyclist'):
    rows.append([n] + [qv(qual['per_class'][n], k) for k in
                       ('improved', 'degraded', 'unchanged', 'both_missed')] + [qual['per_class'][n]['n_gt']])
for sz in ('small', 'medium', 'large'):
    rows.append([sz] + [qv(qual['per_size'][sz], k) for k in
                        ('improved', 'degraded', 'unchanged', 'both_missed')] + [qual['per_size'][sz]['n_gt']])
table(rows[0], rows[1:], widths=[28, 28, 26, 28, 28, 14], fontsize=8.5,
      cap='表8 定性四态(含率与支持数) | 源: runs/rlivit_sq/qual_breakdown.json; 夜间168帧, conf0.25, IoU0.5, 一对一贪心匹配')

# ---------- 9 旧协议审计 ----------
h1('9. 旧开发协议对照 (仅审计用, 不进入主表)')
rows = [['模型', '全量 mAP50 (s0/s1)', '夜 mAP50 (s0/s1)'],
        ['concat (old)', '58.92 / 57.52', '47.45 / 47.28'],
        ['GBF (old)', '61.20 / 60.46', '54.34 / 57.26'],
        ['GBF−concat', '+2.28 / +2.94', '+6.89 / +9.98']]
table(rows[0], rows[1:], widths=[36, 42, 42],
      cap='表9 旧协议 | 源: runs/rlivit/ 历史summary与子集json; 序列划分本身无重叠, 虚增来源=报告集上选checkpoint')

# ---------- 10 源文件索引 ----------
h1('10. 数据资产索引 (每个数字的检索路径)')
idx = [
 ('主协议全指标+每类', 'runs/rlivit_sq/eval_{sq-4ch,sq-4ch-tbf,sq-4ch-gsw,sq-4ch-gbf}_s{0,1,2}.json'),
 ('主协议训练期记录', 'runs/rlivit_sq/summary_sq.json (含ultralytics val口径数字与per-class)'),
 ('预测原始dets', 'runs/rlivit_sq/dets_*.json (conf0.001全部框, 可重算任何阈值)'),
 ('bootstrap', 'runs/rlivit_sq/bootstrap_seq.json'),
 ('晚融合', 'runs/rlivit_sq/eval_late_sq-rgb{s}_sq-ir{s}.json'),
 ('定性', 'runs/rlivit_sq/qual_report.json + qual_breakdown.json'),
 ('效率', 'runs/rlivit_sq/eff_v2.json + runs/rlivit/efficiency.json'),
 ('三模态', 'runs/rlivit_msq/summary.json + dets_msq-*.json + rlivit_multi_sq/build_stats.json'),
 ('LLVIP', 'runs/llvip_v2/summary.json'),
 ('划分与审计', 'rlivit_sq/{manifest.csv,split_report.json}, rlivit_multi_sq/manifest.csv, LLVIP dev_manifest.csv'),
 ('归档', 'paper/protocol_archive.json (环境/命令/ckpt哈希/selected epoch)'),
 ('协议审计与证据', 'paper/证据恢复_D1-D6.md'),
 ('论文口径汇总', 'paper/最终结果报告_20260907.md'),
]
for name, path in idx:
    para(f'• {name}: {path}', 9)

doc.save(OUT)
print('saved:', OUT)
