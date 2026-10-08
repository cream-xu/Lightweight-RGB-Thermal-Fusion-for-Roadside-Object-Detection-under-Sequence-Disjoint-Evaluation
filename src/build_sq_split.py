"""新协议: R-LiViT 2400帧 按序列划分 train/dev/test (3:1:1, 按夜帧占比分层)
- 序列为单位 → 序列不相交 (消除旧帧级划分的序列内泄漏)
- 按每序列夜帧占比排序后 3:1:1 轮转分配 → 三个split夜帧占比近似一致
- dev 用于 checkpoint 选择, test 仅在方案冻结后评估一次 (held-out)
- 物理复制 rgb/ir 的 images+labels 到 rlivit_sq/{rgb,ir}/{images,labels}/{train,dev,test}
输出: rlivit_sq/manifest.csv + rlivit_sq/split_report.json
"""
import os, csv, json, shutil
from collections import Counter

SRC = 'datasets/RLiViT/rlivit_full'
DST = 'datasets/RLiViT/rlivit_sq'

rows = list(csv.DictReader(open(os.path.join(SRC, 'manifest.csv'))))
assert len(rows) == 2400, len(rows)

# 每序列夜帧占比
seq_night = Counter()
seq_total = Counter()
for r in rows:
    s = r['seq']
    seq_total[s] += 1
    if r['daytime'].startswith('night'):
        seq_night[s] += 1
seqs = sorted(seq_total)
seq_ratio = {s: seq_night[s] / seq_total[s] for s in seqs}

# 按夜帧占比排序, 3:1:1 轮转
order = sorted(seqs, key=lambda s: (seq_ratio[s], s))
assign = {}
for i, s in enumerate(order):
    assign[s] = ['train', 'train', 'train', 'dev', 'test'][i % 5]

n_train = sum(1 for s in seqs if assign[s] == 'train')
n_dev = sum(1 for s in seqs if assign[s] == 'dev')
n_test = sum(1 for s in seqs if assign[s] == 'test')
print(f'sequences: train={n_train} dev={n_dev} test={n_test} (共{len(seqs)})')

# 复制文件
counts = Counter()
new_rows = []
for r in rows:
    s = r['seq']
    split = assign[s]
    name = r['name']
    for mod in ('rgb', 'ir'):
        for kind in ('images', 'labels'):
            ext = 'png' if kind == 'images' else 'txt'
            src_p = os.path.join(SRC, mod, kind, 'val' if r['split'] == 'val' else 'train', f'{name}.{ext}')
            dst_p = os.path.join(DST, mod, kind, split, f'{name}.{ext}')
            os.makedirs(os.path.dirname(dst_p), exist_ok=True)
            shutil.copy(src_p, dst_p)
    counts[split] += 1
    new_rows.append({'name': name, 'split': split, 'daytime': r['daytime'], 'seq': s})

print('frames:', dict(counts))
night_by_split = Counter()
for r in new_rows:
    if r['daytime'].startswith('night'):
        night_by_split[r['split']] += 1
print('night frames per split:', dict(night_by_split))

with open(os.path.join(DST, 'manifest.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['name', 'split', 'daytime', 'seq'])
    w.writeheader()
    w.writerows(new_rows)

json.dump({'split': {s: assign[s] for s in seqs},
           'frames': dict(counts), 'night': dict(night_by_split),
           'seq_ratio': {s: round(seq_ratio[s], 3) for s in seqs}},
          open(os.path.join(DST, 'split_report.json'), 'w'), indent=1)
print('saved manifest + split_report to', DST)
