"""COCO协议统一评估 (序列级新协议 held-out test): 全量/昼/夜/每类 + 晚融合
所有最终评估数字统一由本脚本产出 (与训练期ultralytics val解耦, 保证口径一致可审计)

用法:
  python sq_eval.py sq-4ch s0                  # 单模型: test全量+昼+夜+每类
  python sq_eval.py --late sq-rgb s0 sq-ir s0  # 晚融合: 两模型预测→NMS合并→COCO
输出: runs/rlivit_sq/eval_{name}.json
"""
import os, sys, json, csv
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from train_rlivit_sq import load_sq_eval_model, PROJ, NAMES

SQ = 'datasets/RLiViT/rlivit_sq'
CONF = 0.001
NMS_IOU_LATE = 0.6


def read_manifest():
    rows = list(csv.DictReader(open(os.path.join(SQ, 'manifest.csv'))))
    return [r for r in rows if r['split'] == 'test']


def load_img4(name):
    rgb = np.fromfile(os.path.join(SQ, 'rgb', 'images', 'test', f'{name}.png'), dtype=np.uint8)
    ir = np.fromfile(os.path.join(SQ, 'ir', 'images', 'test', f'{name}.png'), dtype=np.uint8)
    rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
    ir = cv2.imdecode(ir, cv2.IMREAD_GRAYSCALE)
    return rgb, ir


def collect(name_rows, model, tag):
    """收集test预测 → COCO det json"""
    dets = []
    for rid, row in enumerate(name_rows):
        name = row['name']
        rgb, ir = load_img4(name)
        if tag == '4ch':
            img = np.dstack([rgb, ir])
        elif tag == 'rgb':
            img = rgb
        elif tag == 'ir':
            img = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        r = model.predict(img, conf=CONF, imgsz=640, verbose=False, half=False)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append({'image_id': rid + 1, 'category_id': int(b.cls) + 1,
                         'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(b.conf)})
    return dets


def nms(boxes, iou_thr):
    """per-class NMS (xyxy, score)"""
    import torch
    keep = []
    boxes = sorted(boxes, key=lambda b: -b['score'])
    while boxes:
        b = boxes.pop(0)
        keep.append(b)
        boxes = [o for o in boxes if o['category_id'] != b['category_id']
                 or _iou(b['bbox'], o['bbox']) < iou_thr]
    return keep


def _iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / max(1e-9, aw * ah + bw * bh - inter)


def build_gt(rows):
    """COCO GT: images + annotations from labels"""
    import cv2
    images, anns = [], []
    for rid, row in enumerate(rows):
        name = row['name']
        img = cv2.imdecode(np.fromfile(os.path.join(SQ, 'rgb', 'images', 'test', f'{name}.png'),
                                       dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        images.append({'id': rid + 1, 'file_name': name, 'width': w, 'height': h})
        lp = os.path.join(SQ, 'rgb', 'labels', 'test', f'{name}.txt')
        if os.path.exists(lp):
            for line in open(lp):
                c, cx, cy, bw, bh = map(float, line.split())
                anns.append({'id': len(anns) + 1, 'image_id': rid + 1,
                             'category_id': int(c) + 1,
                             'bbox': [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                             'area': bw * w * bh * h, 'iscrowd': 0})
    return {'images': images, 'annotations': anns,
            'categories': [{'id': i + 1, 'name': n} for i, n in enumerate(NAMES)]}


def coco_eval(gt, dets, img_ids=None, cat_ids=None):
    cocoGt = COCO()
    cocoGt.dataset = gt
    cocoGt.createIndex()
    cocoDt = cocoGt.loadRes(dets)
    E = COCOeval(cocoGt, cocoDt, 'bbox')
    if img_ids is not None:
        E.params.imgIds = img_ids
    if cat_ids is not None:
        E.params.catIds = cat_ids  # per-class: GT也限制到该类别 (否则recall被其他类GT压塌)
    E.evaluate(); E.accumulate(); E.summarize()
    return {'AP': E.stats[0], 'AP50': E.stats[1], 'AP75': E.stats[2],
            'AP_S': E.stats[3], 'AP_M': E.stats[4], 'AP_L': E.stats[5],
            'AR100': E.stats[8]}


def main():
    rows = read_manifest()
    if '--late' in sys.argv:
        i = sys.argv.index('--late')
        exp1, s1, exp2, s2 = sys.argv[i + 1], sys.argv[i + 2], sys.argv[i + 3], sys.argv[i + 4]
        m1 = load_sq_eval_model(exp1, int(s1))
        m2 = load_sq_eval_model(exp2, int(s2))
        tag1 = 'rgb' if exp1.endswith('rgb') else 'ir'
        tag2 = 'rgb' if exp2.endswith('rgb') else 'ir'
        dets = collect(rows, m1, tag1) + collect(rows, m2, tag2)
        # per-image NMS: 按image_id分组
        from collections import defaultdict
        by_img = defaultdict(list)
        for d in dets:
            by_img[d['image_id']].append(d)
        merged = []
        for iid, ds in by_img.items():
            merged.extend(nms(ds, NMS_IOU_LATE))
        name = f'late_{exp1}{s1}_{exp2}{s2}'
        out = {'mode': 'late-fusion', 'exp1': f'{exp1}_s{s1}', 'exp2': f'{exp2}_s{s2}',
               'nms_iou': NMS_IOU_LATE}
    else:
        exp, s = sys.argv[1], sys.argv[2]
        m = load_sq_eval_model(exp, int(s))
        tag = '4ch' if exp.startswith('sq-4ch') else ('rgb' if exp.endswith('rgb') else 'ir')
        merged = collect(rows, m, tag)
        name = f'{exp}_s{s}'
        out = {'mode': 'single', 'exp': f'{exp}_s{s}'}

    gt = build_gt(rows)
    day_ids = [r['name'] for r in rows if r['daytime'] == 'day']
    night_ids = [r['name'] for r in rows if r['daytime'].startswith('night')]
    id_map = {r['name']: i + 1 for i, r in enumerate(rows)}
    out['n_test'] = len(rows)
    out['full'] = coco_eval(gt, merged)
    out['day'] = coco_eval(gt, merged, [id_map[n] for n in day_ids])
    out['night'] = coco_eval(gt, merged, [id_map[n] for n in night_ids])
    out['day_n_frames'] = len(day_ids)
    out['night_n_frames'] = len(night_ids)
    out['per_class_ap50'] = {}
    for c in range(7):
        cat_dets = [d for d in merged if d['category_id'] == c + 1]
        r = coco_eval(gt, cat_dets, cat_ids=[c + 1])
        out['per_class_ap50'][NAMES[c]] = round(float(r['AP50']), 4)
    json.dump(out, open(os.path.join(PROJ, f'eval_{name}.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
