"""
R-LiViT 三模态对比实验: rgb / ir / 4ch(RGB+IR) / 5ch(+Depth) / 6ch(+Intensity)
- 融合加载: monkeypatch BaseDataset.load_image, 按stem跨模态根读取拼接 (真多通道, 非丢alpha伪4ch)
- rgb/ir 基线走标准加载器 (ir用cv2 IMREAD_COLOR自动灰度→3ch)
- 训练后评估 best.pt 输出 per-class AP
- IMG(-img): 单stem输出逐通道门控 (已定论: 全量打平, 门控未学会光照分化)
- GBF(-gbf): 双分支门控融合 — RGB/IR各自独立 stem→P2 (模态专属BN/激活),
  光照感知门控在融合前逐通道加权 y = w·f_rgb + (1-w)·f_ir, w=σ(MLP([两分支mean+std]))
- TBF(-tbf): 消融 — 同样双分支但w恒0.5, 分离"双分支结构"与"门控"的贡献
用法: python train_rlivit.py rgb|ir|4ch|5ch|6ch|...-img|...-gbf|...-tbf
"""
import os, sys, math, json
from copy import deepcopy
import numpy as np
import torch, torch.nn as nn
import cv2
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.base import BaseDataset

MULTI_BASE = 'datasets/RLiViT/rlivit_multi'
YAML_DIR = 'configs'
PRETRAINED = 'yolo11n.pt'
PROJ = 'runs/rlivit'

EXPS = {
    'rgb': ('rlivit_rgb.yaml', 3),
    'ir':  ('rlivit_ir.yaml', 3),
    '4ch': ('rlivit_4ch.yaml', 4),
    '5ch': ('rlivit_5ch.yaml', 5),
    '6ch': ('rlivit_6ch.yaml', 6),
    # IMG门控融合变体 (复用对应yaml, 通道数相同)
    '4ch-img': ('rlivit_4ch.yaml', 4),
    '5ch-img': ('rlivit_5ch.yaml', 5),
    '6ch-img': ('rlivit_6ch.yaml', 6),
    # 归因实验: concat结构+fp32协议 (验证门控vs协议差异)
    '5ch-fp32': ('rlivit_5ch.yaml', 5),
    # 全量2400帧 RGB-T (官方序列级划分)
    'full-rgb': ('rlivit_full_rgb.yaml', 3),
    'full-ir': ('rlivit_full_ir.yaml', 3),
    'full-4ch': ('rlivit_full_4ch.yaml', 4),
    'full-4ch-img': ('rlivit_full_4ch.yaml', 4),
    # GBF双分支门控融合 / TBF静态双分支消融 (全量)
    'full-4ch-gbf': ('rlivit_full_4ch.yaml', 4),
    'full-4ch-tbf': ('rlivit_full_4ch.yaml', 4),
    # GSW: 静态可学习逐通道权重融合 (验证"门控学到的是静态权重向量"叙述)
    'full-4ch-gsw': ('rlivit_full_4ch.yaml', 4),
    # 三分支门控融合 (530帧三模态协议): RGB/IR/Depth各自独立分支, 门控softmax三支混合
    '5ch-gbf': ('rlivit_5ch.yaml', 5),
    '5ch-tbf': ('rlivit_5ch.yaml', 5),
    # 双分支GBF@530帧对照: 隔离"小数据集"与"第三分支"两个变量
    # (5ch-gbf 0.7437 < 5ch concat 0.7554, 需判断是数据量还是depth分支导致)
    '4ch-gbf': ('rlivit_4ch.yaml', 4),
}


class GatedStem(nn.Module):
    """光照感知模态门控 (IMG): 输入全局模态统计(各通道均值) → MLP → sigmoid → 逐通道调制stem输出
    动机: IR夜间增益大/深度白天增益大 → 模态有用性随场景条件变化, 固定拼接无法自适应
    可解释性: 门控权重w是输入光照/模态能量的可微函数, 可画"门控-光照"曲线
    注意: 保持 conv/bn/act 属性名与ultralytics Conv完全一致, state_dict键名不变,
          预训练权重才能按 model.0.conv.weight 正常加载
    __module__固定为train_rlivit: 否则直接执行时类注册在__main__, 外部脚本加载ckpt会pickle失败"""

    def __init__(self, conv_mod, in_ch):
        super().__init__()
        self.conv = conv_mod.conv   # Conv2d
        self.bn = conv_mod.bn
        self.act = conv_mod.act
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, 16), nn.ReLU(inplace=True),
            nn.Linear(16, self.conv.out_channels))
        # 接近恒等初始化: 末层权重×0.1, bias=+2.2 → 初始w≈sigmoid(2.2)=0.90,
        # 门控从"近似无影响"开始学, 不干扰预训练特征
        self.mlp[2].weight.data.mul_(0.1)
        nn.init.constant_(self.mlp[2].bias, 2.2)

    def forward(self, x):
        # 门控分支全程fp32: ①AMP下fp16门控梯度NaN ②val时AutoBackend(fp16=True)会half化模型参数
        # 所有权重显式.float()副本计算 (非inplace, 不改参数dtype)
        import torch.nn.functional as F
        with torch.autocast(device_type='cuda', enabled=False):
            x32 = x.float()
            s = x32.mean(dim=(2, 3))               # (B, in_ch) 全局模态统计
            h = torch.relu(s @ self.mlp[0].weight.float().t() + self.mlp[0].bias.float())
            logit = h @ self.mlp[2].weight.float().t() + self.mlp[2].bias.float()
            w = torch.sigmoid(logit)               # (B, C) 逐通道门控权重
            c = self.conv
            f = F.conv2d(x32, c.weight.float(),
                         c.bias.float() if c.bias is not None else None,
                         c.stride, c.padding, c.dilation, c.groups)
            bn = self.bn
            f = F.batch_norm(f, bn.running_mean.float(), bn.running_var.float(),
                             bn.weight.float() if bn.weight is not None else None,
                             bn.bias.float() if bn.bias is not None else None,
                             bn.training, bn.momentum, bn.eps)
            f = self.act(f)
            return f * w.view(-1, f.shape[1], 1, 1)


class GatedFusion(nn.Module):
    """双分支门控融合 (GBF): RGB支与IR支各自独立走 stem→P1→P2 (stride4),
    光照感知门控在融合前逐通道加权求和: y = w·f_rgb + (1-w)·f_ir,
    w = σ(MLP([两分支特征mean+std])) — 4×c_out维输入
    与IMG(单stem输出门控, 已定论无效)的三个区别:
    ① 模态专属BN/激活: 共享stem时IR/RGB分布互相污染BN统计, 分开支各自归一化
    ② 门控条件从"原始像素均值"升级为"两分支语义特征的mean+std", 光照信号强得多
    ③ 门控作用于融合前(逐模态加权)而非融合后(整体缩放, 易被后续层吸收)
    tbf(gated=False): w恒0.5 → 纯双分支加性融合, 分离结构与门控贡献
    gsw(static=True): w=σ(w_raw) 静态可学习逐通道权重, 起点σ(0)=0.5与TBF同 —
    gbf_diag实证门控近似静态权重向量(跨图std 0.0043)后, 此消融验证"去掉MLP/逐图条件
    是否等权" → 若gsw≈gbf, 方法可简化为64参数静态融合权重
    注意: 分支模块(m0/m1/m2)用深拷贝承接预训练权重; __module__固定train_rlivit
    保证ckpt pickle; 门控数学全程fp32(AMP NaN教训, 同GatedStem)"""

    def __init__(self, m0, m1, m2, gated=True, static=False, tri=False):
        super().__init__()
        old = m0.conv
        rgb_conv = nn.Conv2d(3, old.out_channels, kernel_size=old.kernel_size,
                             stride=old.stride, padding=old.padding, bias=old.bias is not None)
        ir_conv = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=old.bias is not None)
        with torch.no_grad():
            rgb_conv.weight[:] = old.weight[:, :3]     # 预训练RGB三通道
            ir_conv.weight[:] = old.weight[:, 0:1]     # IR用R通道复制 (同4ch策略)
        self.rgb = nn.ModuleList([deepcopy(m0), deepcopy(m1), deepcopy(m2)])
        self.ir = nn.ModuleList([deepcopy(m0), deepcopy(m1), deepcopy(m2)])
        self.rgb[0].conv = rgb_conv
        self.ir[0].conv = ir_conv
        self.tri = tri
        if tri:
            depth_conv = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                                   stride=old.stride, padding=old.padding, bias=old.bias is not None)
            with torch.no_grad():
                depth_conv.weight[:] = old.weight[:, 0:1]   # 深度与IR同为R通道初始化
            self.depth = nn.ModuleList([deepcopy(m0), deepcopy(m1), deepcopy(m2)])
            self.depth[0].conv = depth_conv
        m2_last = m2[-1] if isinstance(m2, nn.Sequential) else m2
        self.c_out = m2_last.cv2.conv.out_channels   # 分支P2输出通道数
        self.gated = gated
        self.static = static
        if static:
            self.w_raw = nn.Parameter(torch.zeros(self.c_out))  # σ(0)=0.5 与TBF同起点
        elif gated:
            n_branch = 3 if tri else 2
            # 门控条件 = 各分支语义特征的mean+std (2*n_branch*c_out维)
            # 输出: 双分支sigmoid用c_out (w与1-w), 三分支softmax用3*c_out
            self.mlp = nn.Sequential(
                nn.Linear(2 * n_branch * self.c_out, 16), nn.ReLU(inplace=True),
                nn.Linear(16, self.c_out if not tri else 3 * self.c_out))
            self.mlp[2].weight.data.mul_(0.1)
            nn.init.constant_(self.mlp[2].bias, 0.0)   # 初始w=σ(0)=0.5 / softmax(0)=1/3 等权融合

    def forward(self, x):
        xr, xi = x[:, :3], x[:, 3:4]
        for m in self.rgb:
            xr = m(xr)
        for m in self.ir:
            xi = m(xi)
        # getattr容错: 重构前训练的旧ckpt无tri属性(双分支), 新代码直接复评不炸
        tri = getattr(self, 'tri', False)
        if tri:
            xd = x[:, 4:5]
            for m in self.depth:
                xd = m(xd)
        if self.gated:
            # 门控分支全程fp32: AMP下fp16门控梯度NaN (与GatedStem同款教训)
            with torch.autocast(device_type='cuda', enabled=False):
                if tri:
                    fr, fi, fd = xr.float(), xi.float(), xd.float()
                    s = torch.cat([fr.mean(dim=(2, 3)), fr.std(dim=(2, 3)),
                                   fi.mean(dim=(2, 3)), fi.std(dim=(2, 3)),
                                   fd.mean(dim=(2, 3)), fd.std(dim=(2, 3))], dim=1)
                    h = torch.relu(s @ self.mlp[0].weight.float().t() + self.mlp[0].bias.float())
                    logit = h @ self.mlp[2].weight.float().t() + self.mlp[2].bias.float()
                    w = torch.softmax(logit.view(-1, 3, self.c_out), dim=1)  # (B,3,C)
                    # 必须cast回分支dtype: 否则fp32的w泄漏进half模型(最终val/推理), 下一层half conv崩溃
                    w = w.to(xr.dtype).view(-1, 3 * self.c_out, 1, 1)
                    return (w[:, :self.c_out] * xr + w[:, self.c_out:2 * self.c_out] * xi
                            + w[:, 2 * self.c_out:] * xd)
                fr, fi = xr.float(), xi.float()
                s = torch.cat([fr.mean(dim=(2, 3)), fr.std(dim=(2, 3)),
                               fi.mean(dim=(2, 3)), fi.std(dim=(2, 3))], dim=1)
                h = torch.relu(s @ self.mlp[0].weight.float().t() + self.mlp[0].bias.float())
                logit = h @ self.mlp[2].weight.float().t() + self.mlp[2].bias.float()
                w = torch.sigmoid(logit)               # (B, c_out)
        elif getattr(self, 'static', False):
            w = torch.sigmoid(self.w_raw.float()).expand(xr.shape[0], -1)
        else:
            w = xr.new_full((xr.shape[0], xr.shape[1]), 0.5 if not tri else 1.0 / 3)
        # 必须cast回分支dtype: 否则fp32的w泄漏进half模型(最终val/推理), 下一层half conv崩溃
        w = w.to(xr.dtype).view(-1, self.c_out, 1, 1)
        if tri:
            return w * xr + w * xi + w * xd
        return w * xr + (1 - w) * xi


def multi_read(im_path, channels):
    """RGB路径 → 跨根加载拼接 (BGR,3)+(IR,1)[+(depth,1)][+(int,1)] → (H,W,C) uint8
    支持任意根目录(rlivit_multi/rlivit_full): 从绝对路径解析 <root>/<mod>/images/<split>/<stem>.png"""
    parts = im_path.replace('\\', '/').split('/')
    root = '/'.join(parts[:-4])          # .../rlivit_full 或 .../rlivit_multi
    split, stem = parts[-2], parts[-1].rsplit('.', 1)[0]  # <root>/<mod>/images/<split>/<stem>.png

    def mod(mod, gray):
        p = os.path.join(root, mod, 'images', split, f'{stem}.png')
        if gray:
            # 注意: ultralytics会patch cv2.imread, 灰度可能返回(H,W)或(H,W,1)
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            return g[..., None] if g.ndim == 2 else g
        return cv2.imread(p)

    ims = []
    if channels >= 3:
        ims.append(mod('rgb', False))
    if channels >= 4:
        ims.append(mod('ir', True))
    if channels >= 5:
        ims.append(mod('depth', True))
    if channels >= 6:
        ims.append(mod('int', True))
    return ims[0] if len(ims) == 1 else np.concatenate(ims, axis=2)


def patch_multi_loader():
    """替换 BaseDataset.load_image: 多通道时用跨根读取, 其余逻辑等同原版"""
    orig = BaseDataset.load_image

    def _patched(self, i, rect_mode=True, resize_short=False):
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:
            if fn.exists():
                try:
                    im = np.load(fn)
                    if im.shape[-1] != self.channels:
                        Path(fn).unlink(missing_ok=True)
                        im = multi_read(f, self.channels)
                except Exception:
                    Path(fn).unlink(missing_ok=True)
                    im = multi_read(f, self.channels)
            else:
                im = multi_read(f, self.channels)
            if im is None:
                raise FileNotFoundError(f'Image Not Found {f}')
            h0, w0 = im.shape[:2]
            if rect_mode:
                if resize_short:
                    r = self.imgsz / min(h0, w0)
                    if r != 1:
                        w, h = (math.ceil(w0 * r), self.imgsz) if h0 < w0 else (self.imgsz, math.ceil(h0 * r))
                        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    r = self.imgsz / max(h0, w0)
                    if r != 1:
                        w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != 'ram':
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            return im, (h0, w0), im.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    BaseDataset.load_image = _patched


def make_model(channels, img=False, gbf=False):
    """首层卷积 3ch→channels; 额外通道用R通道权重初始化; img/gbf=True时结构由
    DetectionTrainer.get_model patch在优化器构建前装入(否则mlp不被训练)"""
    model = YOLO(PRETRAINED)
    if channels == 3:
        return model
    if img or gbf:
        print(f'[{"IMG" if img else "GBF"}] 结构将由 get_model patch 装入 (in={channels})')
        return model
    stem = model.model.model[0]
    old = stem.conv
    new = nn.Conv2d(channels, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = old.weight.clone()
        new.weight[:, 3:] = old.weight[:, 0:1].clone()
    stem.conv = new
    return model


def _wrap_gate(stem, channels):
    """GatedStem + 复制ultralytics模块属性(f/i/np等, _predict_once依赖)"""
    gate = GatedStem(stem, channels)
    for a in ('f', 'i', 'np', 'type'):
        setattr(gate, a, getattr(stem, a, None))
    return gate


def _build_gbf(m0, m1, m2, gated=True, static=False, tri=False):
    """用预训练好的前3层(m0/m1/m2)构建GatedFusion, 复制f/i/np属性"""
    gf = GatedFusion(m0, m1, m2, gated=gated, static=static, tri=tri)
    for a in ('f', 'i', 'np', 'type'):
        setattr(gf, a, getattr(m0, a, None))
    return gf


def _merge_p2(model, gf):
    """把backbone前3层(0,1,2)合并为单层GatedFusion并修正连接索引:
    - 层3之后全部前移2位: 旧层k → 新层k-2
    - 所有f引用(跨层特征拼接/Detect头)>2的索引减2, m.i同步更新
    - self.save(需保存的层索引)同步变换 (parse_model在__init__里算好, 不会自动更新)
    注意: yolo11n中backbone C3k2的f=-1(无跨层skip), 跨层引用只来自head的Concat/Detect,
    但本函数按实际f值数据驱动处理, 不依赖该假设"""
    old_save = model.save
    layers = [gf] + list(model.model[3:])
    for i, m in enumerate(layers):
        m.i = i
        if isinstance(m.f, list):
            m.f = [-1 if v == -1 else v - 2 for v in m.f]
    model.save = sorted(set((v - 2) if v > 2 else v for v in old_save if v not in (1, 2)))
    model.model = nn.Sequential(*layers)


def _img_attach(trainer, channels):
    """trainer重建模型后把门控装回trainer.model与EMA (每个trainer实例只装一次)"""
    if getattr(trainer, '_img_applied', False):
        return
    stem = trainer.model.model[0]
    if hasattr(stem, 'mlp'):
        return
    gate = _wrap_gate(stem, channels).to(next(trainer.model.parameters()).device)
    trainer.model.model[0] = gate
    trainer.ema.ema.model[0] = deepcopy(gate)
    trainer._img_applied = True
    print(f'[IMG] 门控已装入 trainer.model 与 EMA (in={channels})')


def patch_val_fp32():
    """img实验的val强制fp32: AutoBackend(fp16=True)会half化模型,
    门控手动fp32输出与后续fp16层dtype不匹配报错"""
    from ultralytics.engine.validator import BaseValidator
    if getattr(patch_val_fp32, '_patched', False):
        return
    _orig = BaseValidator.__call__

    def _patched(self, trainer=None, model=None):
        if os.environ.get('RLIVIT_IMG') == '1':
            self.args.half = False
        return _orig(self, trainer, model)

    BaseValidator.__call__ = _patched
    patch_val_fp32._patched = True


def patch_trainer_get_model():
    """在get_model构建模型后立即装门控 → 优化器构建时mlp参数已在模型内,
    否则mlp不在优化器里, 门控永远不被训练 (w恒0.5, 已实测定位)"""
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    if getattr(patch_trainer_get_model, '_patched', False):
        return
    _orig = DetectionTrainer.get_model

    def _patched(self, cfg=None, weights=None, verbose=True):
        model = _orig(self, cfg, weights, verbose)
        ch = self.data.get('channels', 3)
        gbf_mode = os.environ.get('RLIVIT_GBF', '')
        if gbf_mode in ('gbf', 'tbf', 'gsw', 'gb3', 'tb3') and ch >= 4:
            gf = _build_gbf(model.model[0], model.model[1], model.model[2],
                            gated=(gbf_mode in ('gbf', 'gb3')), static=(gbf_mode == 'gsw'),
                            tri=(gbf_mode in ('gb3', 'tb3')))
            _merge_p2(model, gf)
            print(f'[GBF] {"三" if gf.tri else "双"}分支门控融合已装入 (mode={gbf_mode}, c_out={gf.c_out})')
        elif ch >= 4 and os.environ.get('RLIVIT_IMG') == '1':
            gate = _wrap_gate(model.model[0], ch)
            gate = gate.to(next(model.parameters()).device)
            model.model[0] = gate
            print(f'[IMG] get_model后装入门控 (in={ch})')
        return model

    DetectionTrainer.get_model = _patched
    patch_trainer_get_model._patched = True


def load_eval_model(exp):
    """加载best.pt用于评估/预测; img变体需重建门控结构并恢复训练好的mlp权重
    (ultralytics从ckpt的yaml重建模型会丢弃GatedStem)"""
    best = os.path.join(PROJ, f'rl_{exp}{os.environ.get("RLIVIT_TAG", "")}', 'weights', 'best.pt')
    m = YOLO(best)
    if exp.endswith('-img'):
        channels = EXPS[exp][1]
        ck = torch.load(best, map_location='cpu', weights_only=False)
        ema = ck.get('ema') or ck.get('model')
        gate = _wrap_gate(m.model.model[0], channels)
        if ema is not None and hasattr(ema.model[0], 'mlp'):
            gate.mlp.load_state_dict(ema.model[0].mlp.state_dict())
        m.model.model[0] = gate
        print(f'[IMG-eval] 门控重建+权重恢复 (channels={channels})')
    elif exp.endswith(('-gbf', '-tbf', '-gsw')):
        ck = torch.load(best, map_location='cpu', weights_only=False)
        src = ck.get('ema') or ck.get('model')
        if src is not None and isinstance(src.model[0], GatedFusion):
            m.model = src   # ckpt内模型含GatedFusion, 直接用 (pickle需train_rlivit命名空间, 已注册)
            print(f'[GBF-eval] 双分支模型直接取自ckpt (gated={src.model[0].gated})')
        else:
            print('[GBF-eval] ckpt无GatedFusion, 使用yaml重建模型(结构未恢复, 仅用于诊断)')
    m.model = m.model.float()  # ckpt保存为fp16, 评估统一fp32
    return m


# pickle兼容: 类始终以 train_rlivit.GatedStem 序列化
# 直接执行时主模块叫__main__/__mp_main__, 注册别名使pickle能找到同一个类对象
GatedStem.__module__ = 'train_rlivit'
GatedFusion.__module__ = 'train_rlivit'
if __name__ in ('__main__', '__mp_main__'):
    sys.modules.setdefault('train_rlivit', sys.modules[__name__])


# 模块级补丁: Windows spawn的DataLoader会重执行脚本顶层(作为__mp_main__),
# 若只在__main__里打补丁, worker进程的BaseDataset仍是未打补丁的版本
if os.environ.get('RLIVIT_CHANNELS', '').isdigit() and int(os.environ['RLIVIT_CHANNELS']) >= 4:
    patch_multi_loader()
    print(f'[spawn-safe] 多通道加载器已启用 (channels={os.environ["RLIVIT_CHANNELS"]})')


if __name__ == '__main__':
    exp = sys.argv[1] if len(sys.argv) > 1 else '5ch'
    assert exp in EXPS, f'未知实验 {exp}, 可选: {list(EXPS)}'
    yaml_name, channels = EXPS[exp]
    n_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    resume = len(sys.argv) > 3 and sys.argv[3] == 'resume'
    cfg = os.path.join(YAML_DIR, yaml_name)

    # 设置环境变量使模块级补丁在spawn子进程中同样生效
    os.environ['RLIVIT_CHANNELS'] = str(channels)
    if channels >= 4:
        patch_multi_loader()
        print(f'[patch] 多通道加载器已启用 (channels={channels})')

    img = exp.endswith('-img')
    gbf = exp.endswith(('-gbf', '-tbf', '-gsw'))
    model = make_model(channels, img=img, gbf=gbf)
    if img:
        os.environ['RLIVIT_IMG'] = '1'
        patch_trainer_get_model()
        patch_val_fp32()
    if gbf:
        # GBF输出为fp16原生(门控w cast到分支dtype), val可走half=True与基线同协议
        if channels >= 5:
            # 5ch: RGB/IR/Depth三分支 (LiDAR深度投影作为独立分支输入)
            os.environ['RLIVIT_GBF'] = 'gb3' if exp.endswith('-gbf') else 'tb3'
        else:
            os.environ['RLIVIT_GBF'] = ('gbf' if exp.endswith('-gbf')
                                        else 'tbf' if exp.endswith('-tbf') else 'gsw')
        patch_trainer_get_model()
    # 门控已实现AMP兼容(手动fp32计算+val强制fp32), img实验默认amp=True与基线同协议;
    # 5ch-fp32为归因对照; RLIVIT_FORCE_FP32=1 可强制img也fp32
    fp32 = (exp.endswith('-fp32')) or (img and os.environ.get('RLIVIT_FORCE_FP32') == '1')
    seed = int(os.environ.get('RLIVIT_SEED', '0'))
    tag = os.environ.get('RLIVIT_TAG', '')
    model.train(
        data=cfg, epochs=n_epochs, imgsz=640, batch=8, device=0, workers=4,
        project=PROJ, name=f'rl_{exp}{tag}', exist_ok=True, seed=seed, cache=False,
        resume=resume, amp=(not fp32),
    )

    r = load_eval_model(exp).val(data=cfg, plots=False, verbose=False)
    names = list(r.names.values())
    summary = {
        'exp': exp, 'channels': channels,
        'mAP50': round(float(r.box.map50), 4),
        'mAP50-95': round(float(r.box.map), 4),
        'ap50': {names[i]: round(float(v), 4) for i, v in enumerate(r.box.ap50)},
    }
    out = os.path.join(PROJ, 'summary.json')
    if os.path.exists(out):
        prev = json.load(open(out))
    else:
        prev = {}
    prev[exp + tag] = summary
    json.dump(prev, open(out, 'w'), indent=1)
    print(json.dumps(summary, indent=1))
