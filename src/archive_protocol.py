"""实验归档 (2026-08-28, 对照修改清单C-实验记录与材料):
收集环境版本/硬件/数据集manifests/训练命令/ckpt哈希与selected epoch/结果导出
输出: paper/protocol_archive.json
"""
import os, sys, json, csv, hashlib, subprocess, platform
import numpy as np
import torch

ROOT = r''
ARCH = {'generated_at': __import__('datetime').datetime.now().isoformat(timespec='seconds')}

# 环境
import ultralytics, cv2
ARCH['env'] = {
    'platform': platform.platform(),
    'python': sys.version.split()[0],
    'torch': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'ultralytics': ultralytics.__version__,
    'cv2': cv2.__version__,
    'numpy': np.__version__,
}
try:
    gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                               '--format=csv,noheader'], capture_output=True, text=True,
                              timeout=15)
    ARCH['nvidia_smi'] = gpu_info.stdout.strip()
except Exception as e:
    ARCH['nvidia_smi'] = f'ERR {e}'


def md5(p, n_bytes=1024 * 256):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        while True:
            b = f.read(n_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def manifest_md5(p):
    rows = sorted(open(p).readlines())
    return hashlib.md5(''.join(rows).encode('utf-8')).hexdigest()


# 数据集manifests
ARCH['manifests'] = {}
for key, p in [
    ('rlivit_full', 'datasets/RLiViT/rlivit_full/manifest.csv'),
    ('rlivit_sq', 'datasets/RLiViT/rlivit_sq/manifest.csv'),
    ('rlivit_multi_sq', 'datasets/RLiViT/rlivit_multi_sq/manifest.csv'),
    ('llvip_dev', 'datasets/LLVIP/LLVIP/yolo_fusion/dev_manifest.csv'),
]:
    fp = os.path.join(ROOT, p)
    if os.path.exists(fp):
        ARCH['manifests'][key] = {'md5': manifest_md5(fp), 'path': p}

# 训练命令与ckpt
def scan_runs(proj, tag):
    runs = {}
    for d in sorted(os.listdir(proj)):
        rp = os.path.join(proj, d)
        best = os.path.join(rp, 'weights', 'best.pt')
        last = os.path.join(rp, 'weights', 'last.pt')
        if not os.path.exists(best):
            continue
        entry = {'best_md5': md5(best, 1024 * 1024)[:16]}
        res = os.path.join(rp, 'results.csv')
        if os.path.exists(res):
            rows = list(csv.DictReader(open(res)))
            if rows:
                b = max(rows, key=lambda r: float(r.get('metrics/mAP50-95(B)') or 0))
                entry['selected_epoch'] = int(b['epoch'])
                entry['val_mAP50-95(B)'] = float(b['metrics/mAP50-95(B)'])
                entry['val_mAP50(B)'] = float(b['metrics/mAP50(B)'])
        runs[d] = entry
    ARCH[f'runs_{tag}'] = runs
    return runs


scan_runs(os.path.join(ROOT, 'runs/rlivit'), 'old_protocol')
scan_runs(os.path.join(ROOT, 'runs/rlivit_sq'), 'sq_protocol')
scan_runs(os.path.join(ROOT, 'runs/rlivit_msq'), 'msq_protocol')
scan_runs(os.path.join(ROOT, 'runs/llvip_v2'), 'llvip_v2')

# 结果汇总
for key, p in [
    ('summary_sq', 'runs/rlivit_sq/summary_sq.json'),
    ('summary_msq', 'runs/rlivit_msq/summary.json'),
    ('summary_llvip_v2', 'runs/llvip_v2/summary.json'),
    ('efficiency', 'runs/rlivit/efficiency.json'),
]:
    fp = os.path.join(ROOT, p)
    if os.path.exists(fp):
        ARCH[f'results_{key}'] = json.load(open(fp))

# 训练命令 (脚本即记录: 固定seed/epochs/batch等)
ARCH['commands'] = {
    'sq': 'RLIVIT_SEED={0,1,2} python train_rlivit_sq.py sq-4ch|sq-4ch-gbf|sq-4ch-tbf|sq-4ch-gsw|sq-rgb|sq-ir 150',
    'sq_eval': 'python sq_eval.py sq-4ch 0 | python sq_eval.py --late sq-rgb 0 sq-ir 0',
    'msq': 'RLIVIT_SEED=0 python train_multi_sq.py msq-4ch|msq-4ch-gbf|msq-5ch|msq-5ch-tbf|msq-5ch-gbf 150',
    'llvip': 'RLIVIT_SEED={0,1} python train_llvip_v2.py concat|gbf 100',
    'hyper': 'imgsz=640 batch=8(sq/msq)/16(llvip) workers=4 amp=True epochs=150(sq/msq)/100(llvip) cache=False device=0',
}
ARCH['protocol_notes'] = {
    'sq': '序列不相交 120/40/40 (1440/480/480帧), 按夜帧占比分层; dev(156夜)选ckpt, test(168夜)held-out',
    'msq': '三模态v2: med<=60px 71序列对(13夜), 序列级3:1:1, 每帧dt_sync/coverage记录于build_stats.json; 标定为label-informed(3D↔2D框底心), 论文披露',
    'llvip': '11025 train / 1000 dev(seed42抽样) / 3463 test(补全790); best.pt按dev fitness=纯mAP50-95',
    'ckpt_criterion': 'fitness = w·[P,R,mAP50,mAP50-95], w=[0,0,0,1.0] → 纯val mAP50-95 (ultralytics 8.4.75)',
    'aug': 'hsv(0.015,0.7,0.4) translate0.1 scale0.5 fliplr0.5 mosaic1.0 close_mosaic10 mixup0 copy_paste0 erasing0.4 (训练日志回显)',
    'eval_uniform': '最终数字统一由sq_eval.py(pycocotools COCO)产出: 全量/昼/夜/每类; 训练期val仅用于选ckpt',
}

out = os.path.join(ROOT, 'paper', 'protocol_archive.json')
json.dump(ARCH, open(out, 'w'), indent=1)
print('saved', out)
print('runs sq:', len(ARCH.get('runs_sq_protocol', {})),
      '| msq:', len(ARCH.get('runs_msq_protocol', {})),
      '| llvip_v2:', len(ARCH.get('runs_llvip_v2', {})))
