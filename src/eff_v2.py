"""效率复核 v2 (2026-08-28, 对照修改清单C-效率项):
① 门控ops是否被thop覆盖: GatedFusion子模块 thop数 vs 手动MAC账 (conv支+MLP+统计ops)
② 端到端时延: 磁盘读取4ch PNG → letterbox → fp16前向 → NMS (concat vs gbf)
   (深度投影为离线预计算, 单独报告每帧投影成本 — 从build_5ch_v2计时)
③ 输出: runs/rlivit_sq/eff_v2.json
用法: python eff_v2.py  (GPU空闲时)
"""
import os, sys, json, time
import numpy as np
import cv2
import torch
import train_rlivit as T
from train_rlivit_sq import load_sq_eval_model, PROJ

N_IMG = 40   # 端到端用N_IMG张真实test图
IMGSZ = 640


def gate_manual_macs(gf):
    """GatedFusion门控路径手动MAC账: MLP(256→16→64) + 统计(mean/std) + 加权"""
    c = gf.c_out
    mlp_macs = 256 * 16 + 16 * c  # 两Linear
    # mean/std: H/4*W/4 = 160*160 = 25600 px per 64ch branch ×2 (mean+std) ×2分支
    px = (IMGSZ // 4) * (IMGSZ // 4)
    stat_macs = 2 * 2 * c * px
    # 加权: 2*c*px (两个分支各乘加)
    mix_macs = 2 * c * px
    return {'mlp_macs': mlp_macs, 'stat_macs': stat_macs, 'mix_macs': mix_macs,
            'total_gate_macs': mlp_macs + stat_macs + mix_macs}


def measure_e2e(model, imgs, name, ir_dir=None):
    """端到端: 磁盘→letterbox→前向→NMS (超时测速); ir_dir给定时叠IR成4ch (sq协议分模态存储)"""
    from ultralytics.utils.nms import non_max_suppression
    model = model.model.half().cuda().eval()
    ts = []
    with torch.no_grad():
        for _ in range(5):  # warmup
            x = torch.randn(1, 4, IMGSZ, IMGSZ, device='cuda', dtype=torch.float16)
            model(x)
        for img in imgs:
            t0 = time.perf_counter()
            # 磁盘读取 (np.fromfile+imdecode, 与训练/推理实际路径一致)
            buf = np.fromfile(img, dtype=np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if ir_dir is not None:
                bufi = np.fromfile(os.path.join(ir_dir, os.path.basename(img)), dtype=np.uint8)
                imi = cv2.imdecode(bufi, cv2.IMREAD_GRAYSCALE)
                im = np.dstack([im, imi])
            # letterbox 640 (与ultralytics preprocess一致)
            h0, w0 = im.shape[:2]
            r = IMGSZ / max(h0, w0)
            w, h = min(round(w0 * r), IMGSZ), min(round(h0 * r), IMGSZ)
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            im = cv2.copyMakeBorder(im, 0, IMGSZ - h, 0, IMGSZ - w,
                                    cv2.BORDER_CONSTANT, value=114)
            x = torch.from_numpy(np.transpose(im, (2, 0, 1))[None]).cuda().half() / 255.0
            pred = model(x)  # (1,4,8400)
            if isinstance(pred, tuple):
                pred = pred[0]
            non_max_suppression(pred, conf_thres=0.25, iou_thres=0.7,
                                max_det=300, nc=7)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return {'ms_mean': round(float(ts.mean() * 1e3), 2),
            'ms_p50': round(float(np.median(ts) * 1e3), 2),
            'fps_mean': round(float(1 / ts.mean()), 1),
            'n': len(ts)}


def main():
    imgs = sorted([os.path.join(T.SQ if hasattr(T, 'SQ') else
                                'datasets/RLiViT/rlivit_sq',
                                'rgb', 'images', 'test', f)
                   for f in os.listdir('datasets/RLiViT/rlivit_sq/rgb/images/test')])[:N_IMG]

    out = {}
    # ① thop覆盖审计: 对GatedFusion单独profile + 手动账
    m_gbf = load_sq_eval_model('sq-4ch-gbf', 0) if os.path.exists(
        os.path.join(PROJ, 'rl_sq-4ch-gbf_s0', 'weights', 'best.pt')) else None
    if m_gbf is not None:
        gf = m_gbf.model.model[0]
        try:
            from thop import profile as thop_profile
            x4 = torch.randn(1, 4, IMGSZ, IMGSZ)
            fl, _ = thop_profile(gf, inputs=(x4,), verbose=False)
            manual = gate_manual_macs(gf)
            out['gate_thop_MACs'] = int(fl)
            out['gate_manual'] = manual
            out['gate_thop_ratio'] = round(fl / sum(manual.values()), 3)
            print('[审计] thop计数 vs 手动账:', int(fl), manual,
                  'ratio', out['gate_thop_ratio'])
        except Exception as e:
            out['gate_audit_error'] = str(e)

    # ② 端到端时延 (concat vs gbf, 4ch = rgb+ir双文件)
    SQ_IR_IMG = 'datasets/RLiViT/rlivit_sq/ir/images/test'
    for exp in ('sq-4ch', 'sq-4ch-gbf'):
        try:
            m = load_sq_eval_model(exp, 0)
            out[f'e2e_{exp}'] = measure_e2e(m, imgs, exp, ir_dir=SQ_IR_IMG)
            print(exp, out[f'e2e_{exp}'])
        except Exception as e:
            out[f'e2e_{exp}_error'] = str(e)
            print(exp, 'ERR', e)

    # ③ 晚融合端到端: rgb模型 + ir模型 串行前向 + 合并NMS
    try:
        from ultralytics.utils.nms import non_max_suppression
        m_rgb = load_sq_eval_model('sq-rgb', 0).model.half().cuda().eval()
        m_ir = load_sq_eval_model('sq-ir', 0).model.half().cuda().eval()
        SQ_IMG = 'datasets/RLiViT/rlivit_sq/rgb/images/test'
        SQ_IR = 'datasets/RLiViT/rlivit_sq/ir/images/test'
        ts = []
        with torch.no_grad():
            for _ in range(5):
                m_rgb(torch.randn(1, 3, IMGSZ, IMGSZ, device='cuda', dtype=torch.float16))
                m_ir(torch.randn(1, 3, IMGSZ, IMGSZ, device='cuda', dtype=torch.float16))
            for f in imgs[:N_IMG]:
                t0 = time.perf_counter()
                buf = np.fromfile(f, dtype=np.uint8)
                im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)[..., :3]
                r = IMGSZ / max(im.shape[:2])
                w, h = min(round(im.shape[1] * r), IMGSZ), min(round(im.shape[0] * r), IMGSZ)
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
                im = cv2.copyMakeBorder(im, 0, IMGSZ - h, 0, IMGSZ - w, cv2.BORDER_CONSTANT, value=114)
                x = torch.from_numpy(np.transpose(im, (2, 0, 1))[None]).cuda().half() / 255.0
                pr = m_rgb(x)
                if isinstance(pr, tuple):
                    pr = pr[0]
                name = os.path.basename(f)
                bufi = np.fromfile(os.path.join(SQ_IR, name), dtype=np.uint8)
                imi = cv2.imdecode(bufi, cv2.IMREAD_GRAYSCALE)
                imi = cv2.resize(imi, (w, h), interpolation=cv2.INTER_LINEAR)
                imi = cv2.copyMakeBorder(imi, 0, IMGSZ - h, 0, IMGSZ - w, cv2.BORDER_CONSTANT, value=114)
                xi = torch.from_numpy(imi[None, None]).repeat(1, 3, 1, 1).cuda().half() / 255.0
                pi = m_ir(xi)
                if isinstance(pi, tuple):
                    pi = pi[0]
                merged = torch.cat([pr, pi], dim=1)
                non_max_suppression(merged, conf_thres=0.25, iou_thres=0.6, max_det=300, nc=7)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        ts = np.array(ts)
        out['e2e_late'] = {'ms_mean': round(float(ts.mean() * 1e3), 2),
                           'fps_mean': round(float(1 / ts.mean()), 1), 'n': len(ts),
                           'note': 'rgb模型+ir模型串行前向+合并NMS(iou0.6), 双模型≈2×参数'}
        print('late', out['e2e_late'])
    except Exception as e:
        out['e2e_late_error'] = str(e)
        print('late ERR', e)

    json.dump(out, open(os.path.join(PROJ, 'eff_v2.json'), 'w'), indent=1)
    print('saved eff_v2.json')


if __name__ == '__main__':
    main()
