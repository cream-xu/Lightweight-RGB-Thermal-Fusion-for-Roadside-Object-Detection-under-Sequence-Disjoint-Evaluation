"""GSW 效率实测 (P1-8): 前向FPS (95.2式基准) + 端到端 (磁盘+letterbox+前向+NMS)
输出: runs/rlivit_sq/gsw_latency.json
"""
import os, sys, json, time
import numpy as np
import torch
from train_rlivit_sq import load_sq_eval_model, PROJ
from eff_v2 import measure_e2e

SQ_IMG = r'\datasets\RLiViT\rlivit_sq\rgb\images\test'
SQ_IR = r'\datasets\RLiViT\rlivit_sq\ir\images\test'
N_IMG = 40


def forward_fps(model):
    model = model.model.half().cuda().eval()
    with torch.no_grad():
        x = torch.randn(1, 4, 640, 640, device='cuda', dtype=torch.float16)
        for _ in range(20):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            model(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 100
    return {'ms': round(dt * 1e3, 2), 'fps': round(1 / dt, 1)}


def main():
    m = load_sq_eval_model('sq-4ch-gsw', 0)
    imgs = sorted([os.path.join(SQ_IMG, f) for f in os.listdir(SQ_IMG)])[:N_IMG]
    out = {'forward': forward_fps(m),
           'e2e': measure_e2e(m, imgs, 'gsw', ir_dir=SQ_IR)}
    json.dump(out, open(os.path.join(PROJ, 'gsw_latency.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
