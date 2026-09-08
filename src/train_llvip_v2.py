"""LLVIP 干净协议训练 v2 (2026-08-28, 投稿前协议修正)
- train(11025) / dev(1000, best.pt按dev fitness选择) / 完整test(3463, held-out, 已补全790)
- 用法: python train_llvip_v2.py concat|gbf [epochs=100]
- seed: RLIVIT_SEED env
输出: runs/llvip_v2/rl_llvip-{concat,gbf}_s{seed}/ + summary.json (dev+test双记录)
"""
import os, sys, json
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import train_rlivit as T

CFG_DEV = r'\datasets\configs\llvip_4ch_dev.yaml'
CFG_FULL = r'\datasets\configs\llvip_4ch_full.yaml'
PROJ = r'\runs\llvip_v2'


def llvip_multi_read(im_path, channels):
    buf = np.fromfile(str(im_path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


# 模块级: spawn worker以__mp_main__重执行时覆写读取器 (train_rlivit模块级补丁已用原始multi_read打过)
T.multi_read = llvip_multi_read
if os.environ.get('RLIVIT_CHANNELS', '').isdigit() and int(os.environ['RLIVIT_CHANNELS']) >= 4:
    T.patch_multi_loader()


def load_eval(exp, seed):
    best = os.path.join(PROJ, f'rl_llvip-{exp}_s{seed}', 'weights', 'best.pt')
    m = YOLO(best)
    if exp == 'gbf':
        ck = torch.load(best, map_location='cpu', weights_only=False)
        src = ck.get('ema') or ck.get('model')
        if src is not None and isinstance(src.model[0], T.GatedFusion):
            m.model = src
    m.model = m.model.float()
    return m


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else 'concat'
    assert exp in ('concat', 'gbf')
    n_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    seed = int(os.environ.get('RLIVIT_SEED', '0'))

    T.multi_read = llvip_multi_read
    os.environ['RLIVIT_CHANNELS'] = '4'
    T.patch_multi_loader()
    print('[LLVIP-v2] 4ch加载器已启用')

    model = T.make_model(4, gbf=(exp == 'gbf'))
    if exp == 'gbf':
        os.environ['RLIVIT_GBF'] = 'gbf'
        T.patch_trainer_get_model()

    # 训练: val=dev (best.pt按dev选择, 与test完全隔离)
    model.train(
        data=CFG_DEV, epochs=n_epochs, imgsz=640, batch=16, device=0, workers=4,
        project=PROJ, name=f'rl_llvip-{exp}_s{seed}', exist_ok=True, seed=seed,
        cache=False, amp=True,
    )

    # 冻结方案: 用dev选出的best.pt在完整3463 test评估一次
    m = load_eval(exp, seed)
    r_test = m.val(data=CFG_FULL, plots=False, verbose=False, workers=0)
    r_dev = m.val(data=CFG_DEV, plots=False, verbose=False, workers=0)
    summary = {
        'exp': f'llvip-{exp}', 'seed': seed, 'epochs': n_epochs,
        'test_mAP50': round(float(r_test.box.map50), 4),
        'test_mAP50-95': round(float(r_test.box.map), 4),
        'dev_mAP50': round(float(r_dev.box.map50), 4),
        'dev_mAP50-95': round(float(r_dev.box.map), 4),
    }
    out = os.path.join(PROJ, 'summary.json')
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev[f'llvip-{exp}_s{seed}'] = summary
    json.dump(prev, open(out, 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
