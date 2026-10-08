"""三模态子集 v2 (2026-08-28, 投稿前协议修正):
① med<=60px 序列对全收 (71对, 13夜序列, 夜帧60→~156)
② 序列不相交划分: 按RGB-T序列 3:1:1 (夜序列前置排序→夜帧均摊到三个split)
③ 记录审计数据: 每帧时间同步Δt、深度覆盖率(填充前有效像素占比)、每split标定med分布
④ 标定披露: label-informed calibration (3D↔2D框底心锚点), 局限写入论文
用法: python build_5ch_v2.py
输出: rlivit_multi_sq/ + build_stats.json
"""
import xml.etree.ElementTree as ET
import numpy as np
import os, sys, bisect, json, csv
from PIL import Image
import cv2
import build_5ch as B

OUT = 'datasets/RLiViT/rlivit_multi_sq'
MAX_MED = 60.0
IMG_W, IMG_H = B.IMG_W, B.IMG_H


def project_with_stats(pc, R, cam_pos, fx):
    """投影 + 返回 (depth_n, inten_n, 覆盖率)"""
    pts_cam = (R @ (pc[:, :3] - cam_pos).T).T
    z = pts_cam[:, 2]
    valid = z > 0.3
    u = fx * pts_cam[:, 0] / z + IMG_W / 2
    v = fx * pts_cam[:, 1] / z + IMG_H / 2
    in_img = valid & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    zz = z[in_img]
    uu = u[in_img].astype(np.int32)
    vv = v[in_img].astype(np.int32)
    ii = np.abs(pc[in_img, 3]) if pc.shape[1] >= 4 else np.ones(len(zz))
    covered_px = len(np.unique(vv * IMG_W + uu))
    coverage = covered_px / (IMG_W * IMG_H)

    srt = np.argsort(zz, kind='stable')
    depth = np.full((IMG_H, IMG_W), 999.0, dtype=np.float32)
    inten = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    for s in srt:
        y, x = vv[s], uu[s]
        if depth[y, x] == 999.0:
            depth[y, x] = zz[s]
            inten[y, x] = ii[s]
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
    with np.errstate(invalid='ignore'):
        depth_n = np.where(depth < 998.5, np.exp(-depth / 30.0), 0.0).astype(np.float32)
    inten_max = np.percentile(inten, 98) + 1e-6
    inten_n = np.clip(inten / inten_max, 0, 1).astype(np.float32)
    return depth_n, inten_n, coverage


def main():
    calib = json.load(open(os.path.join(B.BASE, 'calib_per_seq.json')))
    used = {}
    for key, v in calib.items():
        fv = v.get('fine')
        med = fv['med'] if fv else v['med']
        if med > MAX_MED:
            continue
        f = fv or v
        used[key] = (f['yaw'], f['pitch'], f['h'], f['fx'], med, v['daytime'])
    print(f'采用序列对: {len(used)} (med<={MAX_MED}px)')

    # 序列级划分: 夜序列在前, 3:1:1 轮转
    seqs = sorted(set(k.split('|')[1] for k in used))
    seq_daytime = {}
    for k, (_, _, _, _, _, daytime) in used.items():
        seq_daytime[k.split('|')[1]] = daytime
    order = sorted(seqs, key=lambda s: (0 if seq_daytime[s] == 'night' else 1, s))
    assign = {}
    for i, s in enumerate(order):
        assign[s] = ['train', 'train', 'train', 'dev', 'test'][i % 5]
    from collections import Counter
    print('seq split:', Counter(assign.values()), '| night seqs:', sum(1 for s in order if seq_daytime[s] == 'night'))

    for split in ('train', 'dev', 'test'):
        for mod in ('rgb', 'ir', 'depth', 'int', 'labels'):
            os.makedirs(os.path.join(OUT, mod, split), exist_ok=True)

    stats = {'max_med': MAX_MED, 'dt_sync': [], 'coverage': [], 'med_by_split': {},
             'frames_by_split': Counter(), 'night_frames_by_split': Counter()}
    manifest = []

    for key, (yaw, pitch, h, fx, med, daytime) in sorted(used.items()):
        loc, rs, ls = key.split('|')
        split = assign[rs]
        R = B.build_rotation(yaw, pitch)
        cam_pos = np.array([0, 0, h])
        rows = B.load_lidar_ts(ls)
        ann_dir = os.path.join(B.RGB_T_DIR, 'annotations', rs)
        if not os.path.isdir(ann_dir):
            continue
        for f in sorted(os.listdir(ann_dir)):
            if not f.endswith('.xml'):
                continue
            frame = f[:-4]
            tree = ET.parse(os.path.join(ann_dir, f))
            ts_rgb = int(tree.getroot().find('timestampRGB').text) / 1e9
            ts_list = [r[1] for r in rows]
            pos = bisect.bisect_left(ts_list, ts_rgb)
            best_lf, best_dt = None, 1e18
            for i in (pos - 1, pos):
                if 0 <= i < len(rows):
                    d = abs(ts_list[i] - ts_rgb)
                    if d < best_dt:
                        best_dt, best_lf = d, rows[i][0]
            if best_lf is None or best_dt > 0.12:
                continue
            pc = np.fromfile(os.path.join(B.LIDAR_DIR, 'point_clouds', ls, f'{best_lf:06d}.bin'),
                             dtype=np.float32).reshape(-1, 4)
            depth_n, inten_n, coverage = project_with_stats(pc, R, cam_pos, fx)

            rgb = np.array(Image.open(os.path.join(B.RGB_T_DIR, 'rgb', rs, f'{frame}.png')).convert('RGB'))
            ir = np.array(Image.open(os.path.join(B.RGB_T_DIR, 'thermal', rs, f'{frame}.png')).convert('L'))
            if rgb.shape[0] != IMG_H or ir.shape[0] != IMG_H:
                continue

            base = f'{rs}_{frame}'
            Image.fromarray(rgb).save(os.path.join(OUT, 'rgb', split, f'{base}.png'))
            Image.fromarray(ir).save(os.path.join(OUT, 'ir', split, f'{base}.png'))
            Image.fromarray((depth_n * 255).astype(np.uint8)).save(os.path.join(OUT, 'depth', split, f'{base}.png'))
            Image.fromarray((inten_n * 255).astype(np.uint8)).save(os.path.join(OUT, 'int', split, f'{base}.png'))
            B.make_yolo_label(os.path.join(ann_dir, f), os.path.join(OUT, 'labels', split, f'{base}.txt'))

            manifest.append({'name': base, 'split': split, 'daytime': daytime,
                             'seq': rs, 'med': med, 'dt_sync': round(best_dt, 4),
                             'coverage': round(coverage, 4)})
            stats['dt_sync'].append(best_dt)
            stats['coverage'].append(coverage)
            stats['frames_by_split'][split] += 1
            if daytime == 'night':
                stats['night_frames_by_split'][split] += 1

    with open(os.path.join(OUT, 'manifest.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'split', 'daytime', 'seq', 'med', 'dt_sync', 'coverage'])
        w.writeheader()
        w.writerows(manifest)

    for split in ('train', 'dev', 'test'):
        meds = [m['med'] for m in manifest if m['split'] == split]
        stats['med_by_split'][split] = {
            'n': len(meds),
            'min': round(float(min(meds)), 1) if meds else None,
            'median': round(float(np.median(meds)), 1) if meds else None,
            'max': round(float(max(meds)), 1) if meds else None}
    stats['dt_sync'] = {'n': len(stats['dt_sync']),
                        'median_s': round(float(np.median(stats['dt_sync'])), 4),
                        'max_s': round(float(max(stats['dt_sync'])), 4)}
    stats['coverage'] = {'median': round(float(np.median(stats['coverage'])), 4),
                         'min': round(float(min(stats['coverage'])), 4)}
    stats['frames_by_split'] = dict(stats['frames_by_split'])
    stats['night_frames_by_split'] = dict(stats['night_frames_by_split'])
    stats['seq_assign'] = assign
    json.dump(stats, open(os.path.join(OUT, 'build_stats.json'), 'w'), indent=1)
    print('frames:', stats['frames_by_split'], '| night:', stats['night_frames_by_split'])
    print('dt_sync median %.4fs max %.4fs | coverage median %.4f' %
          (stats['dt_sync']['median_s'], stats['dt_sync']['max_s'], stats['coverage']['median']))
    print('saved to', OUT)


if __name__ == '__main__':
    main()
