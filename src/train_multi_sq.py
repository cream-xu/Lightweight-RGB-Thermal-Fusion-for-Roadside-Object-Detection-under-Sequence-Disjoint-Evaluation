"""三模态子集v2训练 (2026-08-28): rlivit_multi_sq 序列不相交 train/dev/test
- 4ch双分支 (RGB+IR) 与 5ch三分支 (RGB+IR+Depth) 同协议对比
- dev选best.pt (fitness=dev mAP50-95), test仅训练后评估一次
- 用法: python train_multi_sq.py msq-4ch|msq-4ch-gbf|msq-5ch|msq-5ch-tbf|msq-5ch-gbf [epochs=150]
- seed: RLIVIT_SEED
输出: runs/rlivit_msq/rl_{exp}_s{seed}/ + summary.json
"""
import os, sys, json
import torch
from ultralytics import YOLO
import train_rlivit as T

PROJ = r'\runs\rlivit_msq'
YAML_DIR = T.YAML_DIR

EXPS = {
    'msq-4ch': ('rlivit_msq_4ch_dev.yaml', 'rlivit_msq_4ch_test.yaml', 4, ''),
    'msq-4ch-gbf': ('rlivit_msq_4ch_dev.yaml', 'rlivit_msq_4ch_test.yaml', 4, 'gbf'),
    'msq-5ch': ('rlivit_msq_5ch_dev.yaml', 'rlivit_msq_5ch_test.yaml', 5, ''),
    'msq-5ch-tbf': ('rlivit_msq_5ch_dev.yaml', 'rlivit_msq_5ch_test.yaml', 5, 'tb3'),
    'msq-5ch-gbf': ('rlivit_msq_5ch_dev.yaml', 'rlivit_msq_5ch_test.yaml', 5, 'gb3'),
}

NAMES = ['Pedestrian', 'Car', 'Cyclist', 'Motorcycle', 'Truck', 'Bus', 'Tramway']


def load_msq_model(exp, seed):
    best = os.path.join(PROJ, f'rl_{exp}_s{seed}', 'weights', 'best.pt')
    m = YOLO(best)
    if exp.endswith(('-gbf', '-tbf')):
        ck = torch.load(best, map_location='cpu', weights_only=False)
        src = ck.get('ema') or ck.get('model')
        if src is not None and isinstance(src.model[0], T.GatedFusion):
            m.model = src
    m.model = m.model.float()
    return m


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else 'msq-5ch-gbf'
    assert exp in EXPS, f'未知实验 {exp}: {list(EXPS)}'
    dev_yaml, test_yaml, channels, mode = EXPS[exp]
    n_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    seed = int(os.environ.get('RLIVIT_SEED', '0'))

    os.environ['RLIVIT_CHANNELS'] = str(channels)
    T.patch_multi_loader()
    print(f'[msq] 多通道加载器已启用 (channels={channels})')

    gbf = mode != ''
    model = T.make_model(channels, gbf=gbf)
    if gbf:
        os.environ['RLIVIT_GBF'] = mode
        T.patch_trainer_get_model()

    model.train(
        data=os.path.join(YAML_DIR, dev_yaml), epochs=n_epochs, imgsz=640, batch=8,
        device=0, workers=4, project=PROJ, name=f'rl_{exp}_s{seed}',
        exist_ok=True, seed=seed, cache=False, amp=True,
    )

    m = load_msq_model(exp, seed)
    r = m.val(data=os.path.join(YAML_DIR, test_yaml), plots=False, verbose=False, workers=0)
    summary = {
        'exp': exp, 'seed': seed, 'channels': channels, 'mode': mode,
        'mAP50': round(float(r.box.map50), 4),
        'mAP50-95': round(float(r.box.map), 4),
        'ap50': {NAMES[i]: round(float(v), 4) for i, v in enumerate(r.box.ap50)},
    }
    out = os.path.join(PROJ, 'summary.json')
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev[f'{exp}_s{seed}'] = summary
    json.dump(prev, open(out, 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
