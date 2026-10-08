"""收集三模态v2协议各模型的test预测 (供bootstrap与定性分析, 推理一次存盘复用)
输出: runs/rlivit_msq/dets_{exp}_s{seed}.json
"""
import os, json, csv
import cv2
import numpy as np
from train_multi_sq import load_msq_model, PROJ

MSQ = 'datasets/RLiViT/rlivit_multi_sq'
CONF = 0.001

MODELS = [('msq-4ch', 0), ('msq-4ch', 1), ('msq-4ch', 2),
          ('msq-4ch-gbf', 0),
          ('msq-5ch', 0), ('msq-5ch', 1), ('msq-5ch', 2),
          ('msq-5ch-tbf', 0), ('msq-5ch-gbf', 0), ('msq-5ch-gbf', 1)]


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(MSQ, 'manifest.csv'))) if r['split'] == 'test']
    for exp, s in MODELS:
        outp = os.path.join(PROJ, f'dets_{exp}_s{s}.json')
        if os.path.exists(outp):
            print(exp, s, '已存在, 跳过')
            continue
        try:
            m = load_msq_model(exp, s)
        except Exception as e:
            print(exp, s, 'SKIP', e)
            continue
        ch = 5 if exp.startswith('msq-5ch') else 4
        dets = []
        for rid, row in enumerate(rows):
            name = row['name']
            def rd(mod, gray):
                p = os.path.join(MSQ, mod, 'images', 'test', f'{name}.png')
                buf = np.fromfile(p, dtype=np.uint8)
                if gray:
                    g = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                    return g[..., None]
                return cv2.imdecode(buf, cv2.IMREAD_COLOR)
            img = np.concatenate([rd('rgb', False), rd('ir', True)] + ([rd('depth', True)] if ch >= 5 else []), axis=2)
            r = m.predict(img, conf=CONF, imgsz=640, verbose=False, half=False)[0]
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                dets.append({'image_id': rid + 1, 'category_id': int(b.cls) + 1,
                             'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(b.conf)})
        json.dump(dets, open(outp, 'w'))
        print(exp, s, 'dets:', len(dets), '→', outp)


if __name__ == '__main__':
    main()
