"""LLVIP 干净协议重组 (2026-08-28):
- 从 12025 train 随机抽 1000 stems → dev (checkpoint选择专用, 移出train)
- train 剩 11025; val = 完整 3463 source test (已补全790)
- 输出: yolo_fusion/{images,labels}/{train,dev,val} + dev_manifest.csv
- 随机种子42, 抽样记录进 dev_manifest 保证可审计
"""
import os, csv, random, shutil

BASE = r'\datasets\LLVIP\LLVIP\yolo_fusion'
SEED = 42
N_DEV = 1000

random.seed(SEED)
train_stems = sorted(f.rsplit('.', 1)[0] for f in os.listdir(os.path.join(BASE, 'images', 'train')))
assert len(train_stems) == 12025
dev_stems = sorted(random.sample(train_stems, N_DEV))

for stem in dev_stems:
    for kind in ('images', 'labels'):
        ext = 'png' if kind == 'images' else 'txt'
        src = os.path.join(BASE, kind, 'train', f'{stem}.{ext}')
        dst = os.path.join(BASE, kind, 'dev', f'{stem}.{ext}')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)

with open(os.path.join(BASE, 'dev_manifest.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['name', 'split'])
    for s in dev_stems:
        w.writerow([s, 'dev'])

print('dev:', len(os.listdir(os.path.join(BASE, 'images', 'dev'))),
      '| train now:', len(os.listdir(os.path.join(BASE, 'images', 'train'))),
      '| val:', len(os.listdir(os.path.join(BASE, 'images', 'val'))))
