"""定性分层细化 (ChatGPT建议, 零GPU): 从已存dets按类/尺寸统计 improved/degraded/unchanged
一对一贪心匹配 (conf降序, IoU≥0.5), 夜间test帧
输出: runs/rlivit_sq/qual_breakdown.json
"""
import os, json, csv
from collections import defaultdict

PROJ = r'\runs\rlivit_sq'
SQ = r'\datasets\RLiViT\rlivit_sq'
IOU = 0.5
IMG_W, IMG_H = 1280, 720
SMALL = 0.005 * IMG_W * IMG_H
LARGE = 0.02 * IMG_W * IMG_H
NAMES = ['Pedestrian', 'Car', 'Cyclist', 'Motorcycle', 'Truck', 'Bus', 'Tramway']


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(1e-9, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def greedy(gt, dets, conf=0.25):
    """gt: [x1,y1,x2,y2,area,cls]; dets: 按score降序 → 每GT状态"""
    status = ['FN'] * len(gt)
    for d in sorted((x for x in dets if x['score'] >= conf), key=lambda x: -x['score']):
        p = [d['bbox'][0], d['bbox'][1], d['bbox'][0] + d['bbox'][2], d['bbox'][1] + d['bbox'][3]]
        best, bi = -1, None
        for gi, g in enumerate(gt):
            if status[gi] == 'FN' and d['category_id'] - 1 == g[5] and iou(p, g) >= IOU:
                v = iou(p, g)
                if v > best:
                    best, bi = v, gi
        if bi is not None:
            status[bi] = 'TP'
    return status


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(SQ, 'manifest.csv'))) if r['split'] == 'test']
    id_map = {r['name']: i + 1 for i, r in enumerate(rows)}
    night = {r['name'] for r in rows if r['daytime'].startswith('night')}

    dets_c = defaultdict(list)
    dets_g = defaultdict(list)
    for d in json.load(open(os.path.join(PROJ, 'dets_sq-4ch_s0.json'))):
        dets_c[d['image_id']].append(d)
    for d in json.load(open(os.path.join(PROJ, 'dets_sq-4ch-gbf_s0.json'))):
        dets_g[d['image_id']].append(d)

    K = ['improved', 'degraded', 'unchanged', 'both_missed']
    per_class = {n: {k: 0 for k in K} for n in NAMES}
    per_size = {'small': {k: 0 for k in K}, 'medium': {k: 0 for k in K},
                'large': {k: 0 for k in K}}
    tot = {k: 0 for k in K}

    for r in rows:
        name = r['name']
        if name not in night:
            continue
        rid = id_map[name]
        gt = []
        lp = os.path.join(SQ, 'rgb', 'labels', 'test', f'{name}.txt')
        if os.path.exists(lp):
            for line in open(lp):
                c, cx, cy, w, h = map(float, line.split())
                x1, y1 = (cx - w/2) * IMG_W, (cy - h/2) * IMG_H
                x2, y2 = (cx + w/2) * IMG_W, (cy + h/2) * IMG_H
                gt.append([x1, y1, x2, y2, (x2-x1)*(y2-y1), int(c)])
        sc = greedy(gt, dets_c[rid])
        sg = greedy(gt, dets_g[rid])
        for i, g in enumerate(gt):
            pair = (sc[i], sg[i])
            # 四态穷举: (FN,TP) improved; (TP,FN) degraded; (TP,TP) unchanged; (FN,FN) both_missed
            k = {'FN,TP': 'improved', 'TP,FN': 'degraded', 'TP,TP': 'unchanged',
                 'FN,FN': 'both_missed'}[f'{sc[i]},{sg[i]}']
            tot[k] += 1
            per_class[NAMES[g[5]]][k] += 1
            sz = 'small' if g[4] < SMALL else ('large' if g[4] >= LARGE else 'medium')
            per_size[sz][k] += 1

    def rates(d):
        n = sum(d.values())
        return {k: (v, round(v / n, 4) if n else 0) for k, v in d.items()} | {'n_gt': n}

    out = {'thresholds': {'iou': IOU, 'conf': 0.25, 'matching': 'conf降序贪心一对一'},
           'scope': '夜间test帧 (168帧)',
           'states': {'improved': '(FN,TP)', 'degraded': '(TP,FN)', 'unchanged': '(TP,TP)',
                      'both_missed': '(FN,FN) — 两模型均漏检, 不计入unchanged'},
           'total': rates(tot), 'per_class': {n: rates(d) for n, d in per_class.items()},
           'per_size': {sz: rates(d) for sz, d in per_size.items()},
           'size_def': {'small': f'<{SMALL:.0f}px²', 'medium': f'{SMALL:.0f}~{LARGE:.0f}px²',
                        'large': f'≥{LARGE:.0f}px²'}}
    json.dump(out, open(os.path.join(PROJ, 'qual_breakdown.json'), 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
