"""序列级 bootstrap 不确定性 (ChatGPT建议, 零训练成本):
以 held-out test 的序列为有放回重采样单元, 重算COCO AP50/AP → Δ的95% CI
比较对: (concat, GBF), (concat, GSW), 全量test与夜间子集; (4ch, 5ch) depth 三模态v2
dets来自已存盘的预测 (dets_*.json), 全程CPU
输出: runs/rlivit_sq/bootstrap_seq.json
"""
import os, io, json, csv
from contextlib import redirect_stdout
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SQ = r'\datasets\RLiViT\rlivit_sq'
MSQ = r'\datasets\RLiViT\rlivit_multi_sq'
PROJ_SQ = r'\runs\rlivit_sq'
PROJ_MSQ = r'\runs\rlivit_msq'
N_BOOT_50 = 3000   # AP50: 单IoU阈值快速eval (审计P1-2要求2000-5000)
N_BOOT_AP = 1000   # AP(50-95): 全阈值慢eval
RNG = np.random.default_rng(42)
NAMES = ['Pedestrian', 'Car', 'Cyclist', 'Motorcycle', 'Truck', 'Bus', 'Tramway']


def build_gt(manifest, labels_dir, img_dir):
    import cv2
    images, anns = [], []
    for rid, row in enumerate(rows_test(manifest)):
        name = row['name']
        buf = np.fromfile(os.path.join(img_dir, f'{name}.png'), dtype=np.uint8)
        h, w = cv2.imdecode(buf, cv2.IMREAD_COLOR).shape[:2]
        images.append({'id': rid + 1, 'file_name': name, 'width': w, 'height': h})
        lp = os.path.join(labels_dir, f'{name}.txt')
        if os.path.exists(lp):
            for line in open(lp):
                c, cx, cy, bw, bh = map(float, line.split())
                anns.append({'id': len(anns) + 1, 'image_id': rid + 1, 'category_id': int(c) + 1,
                             'bbox': [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                             'area': bw * w * bh * h, 'iscrowd': 0})
    return {'images': images, 'annotations': anns,
            'categories': [{'id': i + 1, 'name': n} for i, n in enumerate(NAMES)]}


def rows_test(manifest):
    return [r for r in csv.DictReader(open(manifest)) if r['split'] == 'test']


def eval_ap(gt, dets, img_ids, ap50_only=True):
    cocoGt = COCO()
    cocoGt.dataset = gt
    cocoGt.createIndex()
    E = COCOeval(cocoGt, cocoGt.loadRes(dets), 'bbox')
    E.params.imgIds = img_ids
    # 提速: 只保留all-area评估 (AP/AP50只需area='all', 与pycocotools stats[0]/[1]口径一致)
    E.params.areaRng = [[0 ** 2, 1e10 ** 2]]
    if ap50_only:
        E.params.iouThrs = np.array([0.5])  # 只算AP50, ~10x快
    E.evaluate(); E.accumulate()
    # 手动取stats (summarize要求3个areaRng会崩): 与pycocotools stats口径完全一致
    # 实证: 本版本stats[0]/stats[1]均取M2切片(maxDets=1000), 已与eval_*.json全量数字对表验证
    s = E.eval['precision']  # (T,R,K,A,M), A=0(all), M=[100,300,1000]
    def ap_at(iou_idx, m_idx):
        p = s[iou_idx, :, :, 0, m_idx]
        return float(np.mean(p[p > -1]))
    if ap50_only:
        return ap_at(0, 2)  # AP50@maxDets=1000 == stats[1]
    return float(np.mean([ap_at(t, 2) for t in range(s.shape[0])]))  # AP@maxDets=1000 == stats[0]


def bootstrap_pair(gt, dets_a, dets_b, seq_of_img, img_pool):
    """img_pool: rid列表; seq_of_img: {rid: seq}; 重采样序列 → 帧集合 → ΔAP50(1000x) + ΔAP(200x)"""
    seqs = sorted(set(seq_of_img.values()))
    rid_to_seq = {rid: seq_of_img[rid] for rid in img_pool}

    def samples():
        for _ in range(N_BOOT_50):
            sample = RNG.choice(seqs, size=len(seqs), replace=True)
            yield [rid for rid in img_pool if rid_to_seq[rid] in set(sample)]

    ds50 = []
    for ids in samples():
        ds50.append(eval_ap(gt, dets_b, ids) - eval_ap(gt, dets_a, ids))
    ds50 = np.array(ds50)
    ds_ap = []
    for _ in range(N_BOOT_AP):
        sample = RNG.choice(seqs, size=len(seqs), replace=True)
        ids = [rid for rid in img_pool if rid_to_seq[rid] in set(sample)]
        ds_ap.append(eval_ap(gt, dets_b, ids, ap50_only=False)
                     - eval_ap(gt, dets_a, ids, ap50_only=False))
    ds_ap = np.array(ds_ap)
    return {'dAP50_mean': round(float(ds50.mean()), 4),
            'dAP50_ci95': [round(float(v), 4) for v in np.percentile(ds50, [2.5, 97.5])],
            'dAP50_frac_pos': float((ds50 > 0).mean()),
            'n50': N_BOOT_50,
            'dAP_mean': round(float(ds_ap.mean()), 4),
            'dAP_ci95': [round(float(v), 4) for v in np.percentile(ds_ap, [2.5, 97.5])],
            'n_ap': N_BOOT_AP}


def main():
    # ---- 主协议 (sq) ----
    rows = rows_test(os.path.join(SQ, 'manifest.csv'))
    id_map = {r['name']: i + 1 for i, r in enumerate(rows)}
    gt = build_gt(os.path.join(SQ, 'manifest.csv'),
                  os.path.join(SQ, 'rgb', 'labels', 'test'),
                  os.path.join(SQ, 'rgb', 'images', 'test'))
    seq_of_img = {i + 1: r['seq'] for i, r in enumerate(rows)}
    night_ids = [id_map[r['name']] for r in rows if r['daytime'].startswith('night')]
    # 夜间子集: 只含夜帧, 重采样含夜帧的序列
    night_seqs = sorted(set(r['seq'] for r in rows if r['daytime'].startswith('night')))
    night_seq_map = {rid: s for rid, s in seq_of_img.items() if rid in set(night_ids)}

    out = {'n_seqs': len(set(seq_of_img.values())), 'n_frames': len(rows),
           'n_night_frames': len(night_ids), 'n_boot_ap50': N_BOOT_50,
           'n_boot_ap': N_BOOT_AP, 'comparisons': {}}

    def load_dets(exp, s, proj=PROJ_SQ):
        return json.load(open(os.path.join(proj, f'dets_{exp}_s{s}.json')))

    # concat vs GBF (full + night)
    out['comparisons']['concat_vs_gbf_full'] = bootstrap_pair(
        gt, load_dets('sq-4ch', 0), load_dets('sq-4ch-gbf', 0), seq_of_img, list(range(1, len(rows) + 1)))
    out['comparisons']['concat_vs_gbf_night'] = bootstrap_pair(
        gt, load_dets('sq-4ch', 0), load_dets('sq-4ch-gbf', 0), night_seq_map, night_ids)
    # concat vs GSW (full)
    out['comparisons']['concat_vs_gsw_full'] = bootstrap_pair(
        gt, load_dets('sq-4ch', 0), load_dets('sq-4ch-gsw', 0), seq_of_img, list(range(1, len(rows) + 1)))

    # ---- 深度 (msq): 4ch vs 5ch matched seed0 (dets缺失时跳过, 由watcher2的msq_dets步骤补) ----
    d4 = os.path.join(PROJ_MSQ, 'dets_msq-4ch_s0.json')
    d5 = os.path.join(PROJ_MSQ, 'dets_msq-5ch_s0.json')
    if os.path.exists(d4) and os.path.exists(d5):
        mrows = rows_test(os.path.join(MSQ, 'manifest.csv'))
        mseq = {i + 1: r['seq'] for i, r in enumerate(mrows)}
        mgt = build_gt(os.path.join(MSQ, 'manifest.csv'),
                       os.path.join(MSQ, 'rgb', 'labels', 'test'),
                       os.path.join(MSQ, 'rgb', 'images', 'test'))
        out['comparisons']['depth_4ch_vs_5ch'] = bootstrap_pair(
            mgt, load_dets('msq-4ch', 0, PROJ_MSQ), load_dets('msq-5ch', 0, PROJ_MSQ),
            mseq, list(range(1, len(mrows) + 1)))
    else:
        out['comparisons']['depth_4ch_vs_5ch'] = 'dets缺失, 待msq_dets步骤后重跑'

    json.dump(out, open(os.path.join(PROJ_SQ, 'bootstrap_seq.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
