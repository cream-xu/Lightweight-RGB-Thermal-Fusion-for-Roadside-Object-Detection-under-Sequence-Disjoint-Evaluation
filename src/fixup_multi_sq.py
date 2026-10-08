"""rlivit_multi_sq 布局修复 (build_5ch_v2完成后运行):
mod/{split}/*.png → mod/images/{split}/*.png; labels/{split} → rgb/labels/{split}
(yaml path=rgb根, ultralytics按 rgb/labels 解析; multi_read按 <root>/<mod>/images/<split> 跨根)
"""
import os, shutil

OUT = 'datasets/RLiViT/rlivit_multi_sq'

for mod in ('rgb', 'ir', 'depth', 'int'):
    for split in ('train', 'dev', 'test'):
        src = os.path.join(OUT, mod, split)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(OUT, mod, 'images', split)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            shutil.move(os.path.join(src, f), os.path.join(dst, f))
        os.rmdir(src)

for split in ('train', 'dev', 'test'):
    src = os.path.join(OUT, 'labels', split)
    if not os.path.isdir(src):
        continue
    dst = os.path.join(OUT, 'rgb', 'labels', split)
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))

for split in ('train', 'dev', 'test'):
    n_img = len(os.listdir(os.path.join(OUT, 'rgb', 'images', split)))
    n_lab = len(os.listdir(os.path.join(OUT, 'rgb', 'labels', split)))
    print(split, 'images:', n_img, 'labels:', n_lab)
print('fixup done')
