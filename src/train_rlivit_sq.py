"""R-LiViT 序列级新协议训练 (2026-08-28, 投稿前协议修正)
- 序列不相交 train(1440帧/120seq) / dev(480帧/40seq) / test(480帧/40seq, held-out)
- 训练期val=dev (best.pt按dev fitness选择); 训练后对test仅评估一次并记录
- 用法: python train_rlivit_sq.py sq-4ch|sq-4ch-gbf|sq-4ch-tbf|sq-4ch-gsw|sq-rgb|sq-ir [epochs=150]
- seed: RLIVIT_SEED env (0/1/2)
输出: runs/rlivit_sq/rl_{exp}_s{seed}/ + summary_sq.json (test结果)
"""
import os, sys, json
import torch
from ultralytics import YOLO
import train_rlivit as T

PROJ = r'\runs\rlivit_sq'
YAML_DIR = T.YAML_DIR

EXPS = {
    'sq-4ch': ('rlivit_sq_4ch_dev.yaml', 'rlivit_sq_4ch_test.yaml', 4, ''),
    'sq-4ch-gbf': ('rlivit_sq_4ch_dev.yaml', 'rlivit_sq_4ch_test.yaml', 4, 'gbf'),
    'sq-4ch-tbf': ('rlivit_sq_4ch_dev.yaml', 'rlivit_sq_4ch_test.yaml', 4, 'tbf'),
    'sq-4ch-gsw': ('rlivit_sq_4ch_dev.yaml', 'rlivit_sq_4ch_test.yaml', 4, 'gsw'),
    'sq-rgb': ('rlivit_sq_rgb_dev.yaml', 'rlivit_sq_rgb_dev.yaml', 3, ''),
    'sq-ir': ('rlivit_sq_ir_dev.yaml', 'rlivit_sq_ir_dev.yaml', 3, ''),
}
# sq-rgb/sq-ir 无独立test yaml: test评估由sq_eval.py统一用COCO协议做(含子集)

NAMES = ['Pedestrian', 'Car', 'Cyclist', 'Motorcycle', 'Truck', 'Bus', 'Tramway']


def load_sq_eval_model(exp, seed):
    """加载 sq 协议的 best.pt (dev选择), 恢复GatedFusion结构"""
    best = os.path.join(PROJ, f'rl_{exp}_s{seed}', 'weights', 'best.pt')
    m = YOLO(best)
    if exp.endswith(('-gbf', '-tbf', '-gsw')):
        ck = torch.load(best, map_location='cpu', weights_only=False)
        src = ck.get('ema') or ck.get('model')
        if src is not None and isinstance(src.model[0], T.GatedFusion):
            m.model = src
    m.model = m.model.float()
    return m


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else 'sq-4ch'
    assert exp in EXPS, f'未知实验 {exp}: {list(EXPS)}'
    dev_yaml, test_yaml, channels, mode = EXPS[exp]
    n_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    seed = int(os.environ.get('RLIVIT_SEED', '0'))

    # 多通道加载器 (spawn-safe: 模块级patch在worker中生效)
    os.environ['RLIVIT_CHANNELS'] = str(channels)
    if channels >= 4:
        T.patch_multi_loader()
        print(f'[sq] 多通道加载器已启用 (channels={channels})')

    gbf = mode != ''
    model = T.make_model(channels, gbf=gbf)
    if gbf:
        os.environ['RLIVIT_GBF'] = mode
        T.patch_trainer_get_model()

    cfg_dev = os.path.join(YAML_DIR, dev_yaml)
    model.train(
        data=cfg_dev, epochs=n_epochs, imgsz=640, batch=8, device=0, workers=4,
        project=PROJ, name=f'rl_{exp}_s{seed}', exist_ok=True, seed=seed,
        cache=False, amp=True,
    )

    # 冻结方案后: 用dev选出的best.pt在test上评估一次 (4ch用test yaml; rgb/ir用dev yaml路径但sq_eval统一重评)
    m = load_sq_eval_model(exp, seed)
    r = m.val(data=os.path.join(YAML_DIR, test_yaml), plots=False, verbose=False, workers=0)
    summary = {
        'exp': exp, 'seed': seed, 'channels': channels, 'mode': mode,
        'mAP50': round(float(r.box.map50), 4),
        'mAP50-95': round(float(r.box.map), 4),
        'ap50': {NAMES[i]: round(float(v), 4) for i, v in enumerate(r.box.ap50)},
    }
    out = os.path.join(PROJ, 'summary_sq.json')
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev[f'{exp}_s{seed}'] = summary
    json.dump(prev, open(out, 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
