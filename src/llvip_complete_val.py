"""LLVIP val 补全: 补建缺失的790张4ch融合PNG (visible BGR + infrared alpha)
已逐字节验证配方与现有val一致; 标签(3463)已完整无需补
输出: 补全后 yolo_fusion/images/val = 3463
"""
import os
import numpy as np
import cv2

BASE = 'datasets/LLVIP/LLVIP'
VAL_DIR = os.path.join(BASE, 'yolo_fusion', 'images', 'val')

src_test = set(f.rsplit('.', 1)[0] for f in os.listdir(os.path.join(BASE, 'infrared', 'test')))
have = set(f.rsplit('.', 1)[0] for f in os.listdir(VAL_DIR))
missing = sorted(src_test - have)
print('to build:', len(missing))

n = 0
for name in missing:
    vis = np.fromfile(os.path.join(BASE, 'visible', 'test', f'{name}.jpg'), dtype=np.uint8)
    vis = cv2.imdecode(vis, cv2.IMREAD_COLOR)
    ir = np.fromfile(os.path.join(BASE, 'infrared', 'test', f'{name}.jpg'), dtype=np.uint8)
    ir = cv2.imdecode(ir, cv2.IMREAD_GRAYSCALE)
    comp = np.dstack([vis, ir])
    ok, buf = cv2.imencode('.png', comp)
    assert ok, name
    buf.tofile(os.path.join(VAL_DIR, f'{name}.png'))
    n += 1

# 清理旧缓存 (ultralytics下次评估会重建)
for c in ('val.cache',):
    p = os.path.join(BASE, 'yolo_fusion', 'labels', c)
    if os.path.exists(p):
        os.remove(p)
        print('removed stale cache:', p)
p = os.path.join(BASE, 'yolo_fusion', 'labels', 'val')
for c in os.listdir(p):
    if c.endswith('.cache'):
        os.remove(os.path.join(p, c))
        print('removed stale cache:', c)

print('done, val images now:', len(os.listdir(VAL_DIR)),
      '| labels:', len(os.listdir(os.path.join(BASE, 'yolo_fusion', 'labels', 'val'))))
