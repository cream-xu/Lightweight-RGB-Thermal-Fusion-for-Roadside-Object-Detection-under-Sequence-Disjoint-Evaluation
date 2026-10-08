"""
R-LiViT 三模态数据集构建: RGB+IR+Depth(5ch) / +Intensity(6ch) / RGB+IR(4ch) / RGB / IR
- 标定参数来自 calib_per_seq.json (fine, med < min_med)
- 深度图: 点云投影→按像素取最近→漫滩填充洞→exp衰减归一化
- 标签: VOCC XML → YOLO txt (CLASS_MAP 8类)
- 划分: 按序列对切分不串场景; 随机种子固定
用法: python build_5ch.py [min_med]
"""
import xml.etree.ElementTree as ET
import numpy as np
import os, sys, bisect, json, random
import cv2
from PIL import Image

BASE = 'datasets/RLiViT'
RGB_T_DIR = os.path.join(BASE, 'R-LiViT_RGB-T')
LIDAR_DIR = os.path.join(BASE, 'R-LiViT_LiDAR')
OUT = os.path.join(BASE, 'rlivit_multi')
IMG_W, IMG_H = 1280, 720

# RGB-T类名 → 统一类别ID
CLASS_IDS = ['Pedestrian', 'Car', 'Cyclist', 'Motorcycle', 'Truck', 'Bus', 'Tramway']
CLASS_MAP = {
    'person': 0, 'car': 1, 'bicycle': 2, 'escooter': 2, 'motorcycle': 3,
    'truck': 4, 'bus': 5, 'tramway': 6,
}
R0 = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)


def build_rotation(yaw, pitch):
    yaw_r, pitch_r = np.deg2rad(yaw), np.deg2rad(pitch)
    Ry = np.array([[np.cos(yaw_r), 0, np.sin(yaw_r)],
                   [0, 1, 0],
                   [-np.sin(yaw_r), 0, np.cos(yaw_r)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch_r), -np.sin(pitch_r)],
                   [0, np.sin(pitch_r), np.cos(pitch_r)]])
    return Rx @ Ry @ R0


def load_lidar_ts(ls):
    with open(os.path.join(LIDAR_DIR, 'timestamps', f'{ls}.txt')) as fh:
        next(fh)
        rows = []
        for line in fh:
            p = line.strip().split(',')
            if len(p) == 2:
                rows.append((int(p[0]), float(p[1])))
    return rows


def nearest_frame(rows, ts, max_diff=0.12):
    ts_list = [r[1] for r in rows]
    pos = bisect.bisect_left(ts_list, ts)
    best, bd = None, 1e18
    for i in (pos - 1, pos):
        if 0 <= i < len(rows):
            d = abs(ts_list[i] - ts)
            if d < bd:
                bd, best = d, rows[i][0]
    if best is None or bd > max_diff:
        return None
    return best


def project_depth_intensity(points, R, cam_pos, fx):
    """点云→深度图+强度图, 逐像素取最近点, 漫滩填充, 归一化"""
    pts_cam = (R @ (points[:, :3] - cam_pos).T).T
    z = pts_cam[:, 2]
    valid = z > 0.3
    u = fx * pts_cam[:, 0] / z + IMG_W / 2
    v = fx * pts_cam[:, 1] / z + IMG_H / 2
    in_img = valid & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    zz = z[in_img]
    uu = u[in_img].astype(np.int32)
    vv = v[in_img].astype(np.int32)
    ii = np.abs(points[in_img, 3]) if points.shape[1] >= 4 else np.ones(len(zz))

    # 逐像素最近点: 按z升序, 首次写入即该像素最近
    srt = np.argsort(zz, kind='stable')
    depth = np.full((IMG_H, IMG_W), 999.0, dtype=np.float32)
    inten = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    for s in srt:
        y, x = vv[s], uu[s]
        if depth[y, x] == 999.0:
            depth[y, x] = zz[s]
            inten[y, x] = ii[s]

    # 漫滩填充: 5x5核迭代12轮≈填60px空隙
    k = np.ones((5, 5), np.uint8)
    for _ in range(12):
        empty = depth >= 998.5
        if not empty.any():
            break
        d = cv2.dilate(np.where(depth < 998.5, depth, 0.0), k)
        fill = empty & (d > 0)
        depth[fill] = d[fill]
    for _ in range(12):
        empty = inten <= 0
        if not empty.any():
            break
        it = cv2.dilate(inten, k)
        fill = empty & (it > 0)
        inten[fill] = it[fill]

    # 归一化: 深度 exp衰减 (近→1, 远→0), 强度按98分位
    with np.errstate(invalid='ignore'):
        depth_n = np.where(depth < 998.5, np.exp(-depth / 30.0), 0.0).astype(np.float32)
    inten_max = np.percentile(inten, 98) + 1e-6
    inten_n = np.clip(inten / inten_max, 0, 1).astype(np.float32)
    return depth_n, inten_n


def make_yolo_label(xml_path, out_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        cls = CLASS_MAP.get(name)
        if cls is None:
            continue
        bb = obj.find('bndbox')
        xmin, ymin = float(bb.find('xmin').text), float(bb.find('ymin').text)
        xmax, ymax = float(bb.find('xmax').text), float(bb.find('ymax').text)
        w, h = xmax - xmin, ymax - ymin
        assert w > 0 and h > 0, f'{xml_path} 坏框'
        cx, cy = (xmin + xmax) / 2 / IMG_W, (ymin + ymax) / 2 / IMG_H
        lines.append(f'{cls} {cx:.6f} {cy:.6f} {w / IMG_W:.6f} {h / IMG_H:.6f}')
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    min_med = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    calib = json.load(open(os.path.join(BASE, 'calib_per_seq.json')))

    # 选取序列对: fine优秀种子优先
    used = {}
    for key, v in calib.items():
        fv = v.get('fine')
        med = fv['med'] if fv else v['med']
        if med > min_med:
            continue
        if fv:
            used[key] = (fv['yaw'], fv['pitch'], fv['h'], fv['fx'], med, v['daytime'])
        else:
            used[key] = (v['yaw'], v['pitch'], v['h'], v['fx'], med, v['daytime'])
    print(f'采用序列对: {len(used)} (med<= {min_med}px)')

    subdirs = ['images_rgb', 'images_ir', 'images_depth', 'images_int', 'labels']
    for s in subdirs:
        os.makedirs(os.path.join(OUT, s), exist_ok=True)

    random.seed(42)
    manifest = []
    total = {'day': 0, 'night': 0}

    for key, (yaw, pitch, h, fx, med, daytime) in sorted(used.items()):
        loc, rs, ls = key.split('|')
        R = build_rotation(yaw, pitch)
        cam_pos = np.array([0, 0, h])
        rows = load_lidar_ts(ls)
        ann_dir = os.path.join(RGB_T_DIR, 'annotations', rs)
        if not os.path.isdir(ann_dir):
            continue
        for f in sorted(os.listdir(ann_dir)):
            if not f.endswith('.xml'):
                continue
            frame = f[:-4]
            tree = ET.parse(os.path.join(ann_dir, f))
            ts_rgb = int(tree.getroot().find('timestampRGB').text) / 1e9
            lframe = nearest_frame(rows, ts_rgb)
            if lframe is None:
                continue
            pc = np.fromfile(os.path.join(LIDAR_DIR, 'point_clouds', ls, f'{lframe:06d}.bin'),
                             dtype=np.float32).reshape(-1, 4)
            depth_n, inten_n = project_depth_intensity(pc, R, cam_pos, fx)

            rgb = np.array(Image.open(os.path.join(RGB_T_DIR, 'rgb', rs, f'{frame}.png')).convert('RGB'))
            ir = np.array(Image.open(os.path.join(RGB_T_DIR, 'thermal', rs, f'{frame}.png')).convert('L'))
            if rgb.shape[0] != IMG_H or ir.shape[0] != IMG_H:
                print(f'跳过尺寸异常: {key} {frame} {rgb.shape}/{ir.shape}')
                continue

            split = 'train' if random.random() < 0.88 else 'val'
            base = f'{rs}_{frame}'
            Image.fromarray(rgb).save(os.path.join(OUT, 'images_rgb', f'{base}.png'))
            Image.fromarray(ir).save(os.path.join(OUT, 'images_ir', f'{base}.png'))
            Image.fromarray((depth_n * 255).astype(np.uint8)).save(os.path.join(OUT, 'images_depth', f'{base}.png'))
            Image.fromarray((inten_n * 255).astype(np.uint8)).save(os.path.join(OUT, 'images_int', f'{base}.png'))
            make_yolo_label(os.path.join(ann_dir, f), os.path.join(OUT, 'labels', f'{base}.txt'))

            manifest.append(f'{base},{split},{daytime},{med:.1f}')
            total[daytime] = total.get(daytime, 0) + 1
            print(f'{key} {frame}: LiDAR帧={lframe} 保存完成 (RGB+IR+Depth+Intensity+labels)')

    with open(os.path.join(OUT, 'manifest.csv'), 'w') as fh:
        fh.write('name,split,daytime,med\n' + '\n'.join(manifest))
    print(f'\n完成: 共{len(manifest)}帧 (day={total.get("day", 0)}, night={total.get("night", 0)})')
    print('目录: rlivit_multi/{images_rgb,images_ir,images_depth,images_int,labels}')
