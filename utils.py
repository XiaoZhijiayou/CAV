#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Refactored Utility Module
=========================
重构要点概览（英文描述已使用过去式）：
1. Removed duplicated imports & unified命名。
2. Added 类型注解、文档字符串、错误检查。
3. 将功能分区：
   - 量化 quantization
   - 剪枝 pruning
   - 损失 losses (Top-k)
   - 数据 data loading
   - 模型 model 构建
   - 指纹样本选择 fingerprint selection
4. 量化：加入 per-tensor / per-channel 选项；防止 max==min；返回 scale/zero_point 便于复现；避免重复遍历。
5. Top-k 损失：vector 化；加入输入检查；熵使用稳定 log(p+eps)；hook 改写为对负 logits 的惩罚；可选“margin”模式。
6. sensitivity_loss_topk：支持两种模式（"loop" 原实现 / "approx" 近似一次反向），默认 loop；添加开关 second_order 控制是否构建二阶梯度图。
7. get_model：补全对 ResNet18 / VGG16_CIFAR10 / VGG16_GTSRB / EfficientNet-B0 的支持；增加 strict 加载与信息提示。
8. 数据加载：统一接口，train/test 变换分离；GTSRB 支持下载开关；CIFAR100 可选是否升采样到 224；提供 get_dataset_stats 接口便于外部复用。
9. 指纹样本选择：
   - 支持三种模型结构自动定位“最后线性层”参数。
   - 增加 normalize/resize（可配置）。
   - 覆盖统计返回详细结构（总数 / 已覆盖 / 新增）。
   - 保存时可选择软链接或复制。
   - 可配置 loss 类型：{'xe_pred','neg_entropy','margin'}。
10. 通用：添加 seed_everything、分布式无依赖防护（仅单机场景）、设备自动探测。
"""
from __future__ import annotations
import os
import math
import copy
import random
import shutil
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Iterable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import prune as prune_api
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torchvision.models as tv_models
from PIL import Image
from torchvision.transforms.functional import to_tensor

# =========================
# 1. Reproducibility
# =========================

def seed_everything(seed: int = 42):    # 随机数种子
    """设置随机种子。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 2. Quantization (Simulation)
# =========================
@dataclass
class QuantResult:
    tensor: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    max_val: torch.Tensor
    min_val: torch.Tensor


def _safe_range(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return t.max(), t.min()


def quantize_fp32_to_int8_sim(  #   模拟将浮点张量权重做 int8 量化再反量化
    t: torch.Tensor,
    per_channel: bool = False,
    ch_axis: int = 0,
    keep_fp32: bool = True,
    rounding: int = 4,
    eps: float = 1e-8
) -> QuantResult:
    """模拟对浮点权重进行对称 / 非对称 int8 量化并反量化。
    当前实现采用 (min,max) 动态范围 (非对称)；zero_point 会保持在张量 / 通道域。
    
    Args:
        t: 原始浮点张量
        per_channel: 是否按通道量化
        ch_axis: 通道维（用于 per_channel）
        keep_fp32: 返回的张量是否保持 float32（模拟量化）
        rounding: 反量化后保留的小数位
        eps: 防止除零
    Returns:
        QuantResult 数据类
    """
    if per_channel:
        # 计算各通道 min/max
        dim = ch_axis
        max_v = t.amax(dim=tuple(i for i in range(t.dim()) if i != dim), keepdim=True)
        min_v = t.amin(dim=tuple(i for i in range(t.dim()) if i != dim), keepdim=True)
    else:
        max_scalar, min_scalar = t.max(), t.min()
        max_v = max_scalar
        min_v = min_scalar

    # scale / zero_point
    scale = (max_v - min_v) / 255.0
    scale = torch.where(scale.abs() < eps, torch.full_like(scale, eps), scale)
    zero_point = (-min_v / scale).round()

    int_val = (t / scale + zero_point).round().clamp(0, 255)
    dequant = (int_val - zero_point) * scale

    if rounding >= 0:
        factor = 10 ** rounding
        dequant = (dequant * factor).round() / factor

    out = dequant.to(t.dtype) if keep_fp32 else int_val.to(torch.int8)
    return QuantResult(out, scale.squeeze(), zero_point.squeeze(), max_v.squeeze(), min_v.squeeze())


def quantize_state_dict(    #    对整个 state_dict 的参数进行量化
    state: Dict[str, torch.Tensor],
    per_channel: bool = False,
    per_channel_layers: Optional[Iterable[str]] = None,
    exclude_keys: Optional[Iterable[str]] = None,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """对 state_dict 中的参数进行模拟量化。
    per_channel_layers 仅在 per_channel=True 时有效，用于指定需要 per-channel 的键前缀。
    """
    new_state = {}
    exclude = set(exclude_keys or [])
    per_channel_layers = set(per_channel_layers or [])
    for k, v in state.items():
        if k in exclude or not torch.is_floating_point(v):
            new_state[k] = v.clone()
            continue
        pc = per_channel and any(k.startswith(pref) for pref in per_channel_layers)
        qr = quantize_fp32_to_int8_sim(v, per_channel=pc)
        new_state[k] = qr.tensor
        if verbose:
            rng = (qr.min_val.item(), qr.max_val.item()) if not pc else (float(qr.min_val.min()), float(qr.max_val.max()))
            print(f"[Quant] {k}: per_channel={pc} range={rng}")
    return new_state

# =========================
# 3. Pruning Helpers
# =========================
PRUNE_METHODS = {       #   剪枝的方式
    "random": prune_api.random_unstructured,
    "l1": prune_api.l1_unstructured,
}

def get_submodule(model: nn.Module, path: str) -> nn.Module:
    current = model
    for attr in path.split('.'):
        if isinstance(current, nn.Sequential) and attr.isdigit():
            current = current[int(attr)]
        else:
            if not hasattr(current, attr):
                raise AttributeError(f"Path '{path}' invalid at '{attr}'")
            current = getattr(current, attr)
    return current


def prune_layer(model: nn.Module, layer_path: str, amount: float = 0.01, method: str = 'random'):   # 对单层做非结构化剪枝
    if method not in PRUNE_METHODS:
        raise ValueError(f"Invalid prune method: {method}")
    module = get_submodule(model, layer_path)   # 按照路径选取子模块
    if not hasattr(module, 'weight'):
        raise AttributeError(f"Layer '{layer_path}' has no weight")
    PRUNE_METHODS[method](module, name='weight', amount=amount)
    return module

# =========================
# 4. Top-k Losses
# =========================

def _prepare_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if logits.dim() != 2:
        raise ValueError("logits shape must be (N,C) or (C,)")
    return logits


def entropy_loss_topk(logits: torch.Tensor, k: int, eps: float = 1e-8) -> torch.Tensor:         # 熵值损失
    logits = _prepare_logits(logits)
    k = min(k, logits.size(1))
    probs = F.softmax(logits, dim=-1)
    topk_p, _ = probs.topk(k, dim=-1, largest=True)
    topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp_min(eps)  # 归一化 Top-k 概率
    entropy = -(topk_p * (topk_p.clamp_min(eps)).log()).sum(dim=-1)  # sum over k
    return entropy.mean()


def variance_loss_topk(logits: torch.Tensor, k: int) -> torch.Tensor:           # 方差损失  Top-k logits 方差
    logits = _prepare_logits(logits)
    k = min(k, logits.size(1))
    topk_vals, _ = logits.topk(k, dim=-1, largest=True)
    return topk_vals.var(dim=-1, unbiased=False).mean()


def hook_loss_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    对 Top-k logits 的绝对值取负号作为损失。
    损失值越小，对应 Top-k logits 的绝对值越大。
    """
    logits = _prepare_logits(logits)
    k = min(k, logits.size(1))
    topk_vals, _ = logits.topk(k, dim=-1, largest=True)
    loss = -topk_vals.abs().sum(dim=-1)
    return loss.mean()



def sensitivity_loss_topk(
    logits: torch.Tensor,
    biases: Optional[Iterable[torch.Tensor]],
    k: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """估计 Top-k logits 对所有 bias 参量的平均梯度幅值。"""
    logits = _prepare_logits(logits)
    bias_list = [b for b in (biases or []) if b is not None]
    if not bias_list:
        return torch.tensor(0.0, device=logits.device, requires_grad=False)

    k = min(k, logits.size(1))
    _, idx = logits.topk(k, dim=-1, largest=True)
    batch_scores: List[torch.Tensor] = []

    for b in range(logits.size(0)):
        sens_each = []
        for cls in idx[b]:
            logit_val = logits[b, cls]
            grads = torch.autograd.grad(
                logit_val,
                bias_list,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            mags = [
                g.pow(2).sum() if g is not None else torch.tensor(0.0, device=logits.device)
                for g in grads
            ]
            if mags:
                sens_each.append(torch.stack(mags).sum())
        if sens_each:
            batch_scores.append(torch.stack(sens_each).mean())

    if not batch_scores:
        return torch.tensor(0.0, device=logits.device, requires_grad=False)
    return torch.stack(batch_scores).mean()


def logit_input_grad_norm(              ### 对应输出概率与其对应的输入的偏导的l2范数
    logits: torch.Tensor,
    inputs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute ||∂logits/∂inputs||_2 for monitoring purposes.
    """
    grad = torch.autograd.grad(
        outputs=logits.sum(),
        inputs=inputs,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    if grad is None:
        return torch.tensor(0.0, device=logits.device)
    return grad.pow(2).sum().sqrt().clamp_min(eps)


# =========================
# 6a. Gradient-Spectral Losses for Fragile Watermark
# =========================

def grad_energy_and_logdet(
    logits: torch.Tensor,
    params: Iterable[torch.Tensor],
    labels: torch.Tensor,
    delta: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    基于 log p(y_i | x_i) 构造梯度矩阵 G，返回：
      - L_mag:  平均 ||g_i||^2
      - L_logdet: log det(G G^T + delta I)
      - trace_S: tr(S) = sum ||g_i||^2 （可做监控用）
    logits: [N,C]
    params: 想要观察的参数列表（例如 [last_weight, last_bias]）
    labels: [N]，每个样本对应的“秘密标签” y_i
    """
    logits = _prepare_logits(logits)
    device = logits.device
    N = logits.size(0)

    # 1) 取 secret label 的 log prob 作为 z_i（log_softmax 更稳）
    log_probs = F.log_softmax(logits, dim=-1)
    logp = log_probs[torch.arange(N, device=device), labels]  # [N]

    # 2) 针对每个样本求 g_i = ∇_θ log p(y_i | x_i)
    g_list: List[torch.Tensor] = []
    param_list: List[torch.Tensor] = [p for p in params if p is not None]

    if len(param_list) == 0:
        return (torch.tensor(0.0, device=device, requires_grad=False),
                torch.tensor(0.0, device=device, requires_grad=False),
                torch.tensor(0.0, device=device, requires_grad=False))

    total_dim = sum(p.numel() for p in param_list)

    for i in range(N):
        grads = torch.autograd.grad(
            logp[i],
            param_list,
            retain_graph=True,   # 后面还要对 x 反向
            create_graph=True,   # 允许二阶（对 g_i 再对 x 求梯度）
            allow_unused=True,
        )
        flat_grads = []
        for p, g in zip(param_list, grads):
            if g is None:
                flat_grads.append(torch.zeros_like(p, device=device).reshape(-1))
            else:
                flat_grads.append(g.reshape(-1))

        g_flat = torch.cat(flat_grads, dim=0)
        if g_flat.numel() != total_dim:
            padded = torch.zeros(total_dim, device=device)
            n = min(total_dim, g_flat.numel())
            padded[:n] = g_flat[:n]
            g_flat = padded

        g_list.append(g_flat)

    # G: [N, P_eff]
    G = torch.stack(g_list, dim=0)

    # 3) 梯度能量: 平均 ||g_i||^2
    grad_norm_sq = (G ** 2).sum(dim=1)   # [N]
    L_mag = grad_norm_sq.mean()
    trace_S = grad_norm_sq.sum()

    # 4) 方向覆盖：对行归一化后的 Gram 矩阵做 logdet，避免纯放大带来的假“体积”
    G_unit = F.normalize(G, p=2, dim=1)  # 每行范数≈1，仅保留方向
    H = G_unit @ G_unit.t()              # [N, N]
    H_reg = H + delta * torch.eye(N, device=device)
    eigvals = torch.linalg.eigvalsh(H_reg)
    eigvals_clamped = eigvals.clamp(min=1e-8)
    L_logdet = eigvals_clamped.log().sum()

    return L_mag, L_logdet, trace_S


# =========================
# 5. Data Loading
# =========================
"""数据集统计与加载工具。
改动：
- 统一使用 DATASET_STATS（你原先提供的 _DATASET_STATS）作为唯一来源。
- get_dataset_stats 直接复用该映射，支持大小写不敏感；新增别名支持（如 'CIFAR-10'）。
- build_transforms 与 get_dataloader 均调用同一接口，避免重复硬编码。
"""

# 统一数据集统计表（可在外部 from utils import get_dataset_stats 调用）
DATASET_INFO = {
    "cifar10":  {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010], "channels": 3, "input_size": (32, 32)},
    "cifar100": {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761], "channels": 3, "input_size": (32, 32)},
    "gtsrb":    {"mean": [0.3403, 0.3121, 0.3214], "std": [0.2724, 0.2608, 0.2669], "channels": 3, "input_size": (32, 32)},
    "mnist":    {"mean": [0.1307], "std": [0.3081], "channels": 1, "input_size": (28, 28)},
}

# 可选别名（大小写、连字符等）
_DATASET_ALIASES = {
    'cifar-10': 'cifar10',
    'cifar_10': 'cifar10',
    'cifar-100': 'cifar100',
    'cifar_100': 'cifar100',
    'gtsrb': 'gtsrb',
    'mnist': 'mnist',
}

def _canonical_name(name: str) -> str:
    key = name.lower()
    return _DATASET_ALIASES.get(key, key)


def get_dataset_stats(name: str, target_channels: Optional[int] = None):           #   获取数据部分
    """返回 (mean, std)，可选将通道统计广播到 target_channels。"""
    key = _canonical_name(name)
    if key not in DATASET_INFO:
        raise ValueError(f"Unsupported dataset: {name}")
    info = DATASET_INFO[key]
    mean, std = info["mean"], info["std"]
    if target_channels is not None and len(mean) != target_channels:
        if len(mean) == 1:
            mean = mean * target_channels
            std = std * target_channels
        else:
            # 如果已有多通道而目标通道为 1，采用通道均值
            m = sum(mean) / len(mean)
            s = sum(std) / len(std)
            mean, std = [m] * target_channels, [s] * target_channels
    return mean, std


def get_dataset_channels(name: str) -> int:
    key = _canonical_name(name)
    if key not in DATASET_INFO:
        raise ValueError(f"Unsupported dataset: {name}")
    return DATASET_INFO[key]["channels"]


def get_dataset_input_size(name: str) -> Tuple[int, int]:
    key = _canonical_name(name)
    if key not in DATASET_INFO:
        raise ValueError(f"Unsupported dataset: {name}")
    return DATASET_INFO[key]["input_size"]


# =========================
# 5a. IO spec helper
# =========================

def get_io_spec(model_name: str, dataset_name: str) -> Dict[str, object]:
    """
    Return a dict describing how to adapt dataset to the model:
    - channels: input channels expected by model
    - target_size: optional resize size (square) for data pipeline
    - model_input_size: (H,W) fed to model (for dynamic layers like LeNet flatten)
    """
    ds_ch = get_dataset_channels(dataset_name)
    ds_hw = get_dataset_input_size(dataset_name)

    channels = ds_ch if model_name == 'lenet' else 3
    target_size = 224 if model_name == 'efficientnet_b0' else None
    model_input_size = (target_size or ds_hw[0], target_size or ds_hw[1])

    return {
        "channels": channels,
        "target_size": target_size,
        "model_input_size": model_input_size,
    }


def build_transforms(
    name: str,
    train: bool,
    target_channels: Optional[int] = None,
    target_size: Optional[int] = None,
):
    """构建数据增强 / 预处理流水线。
    target_channels: 若与数据集原始通道不同，会用 Grayscale/expand 调整到目标通道。
    target_size: 若指定，则统一 Resize/RandomCrop 到该尺寸（方形）。
    """
    ds_ch = get_dataset_channels(name)
    mean, std = get_dataset_stats(name, target_channels=target_channels or ds_ch)
    name_l = name.lower()

    channel_adapt = []
    if target_channels is not None and target_channels != ds_ch:
        channel_adapt.append(transforms.Grayscale(num_output_channels=target_channels))

    def _resize_block(train_flag: bool):
        if target_size is None:
            return []
        if train_flag:
            if target_size >= 128:
                return [transforms.Resize(target_size + 32), transforms.RandomCrop(target_size)]
            return [transforms.Resize(target_size), transforms.RandomCrop(target_size)]
        return [transforms.Resize(target_size)]

    if name_l.startswith('cifar'):
        base = []
        if target_size:
            base.extend(_resize_block(train))
        else:
            if train:
                base.extend([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])
        base.append(transforms.ToTensor())
        base.append(transforms.Normalize(mean, std))
        return transforms.Compose(channel_adapt + base)
    elif name_l == 'gtsrb':
        base = [
            transforms.Resize((target_size, target_size)) if target_size else transforms.Resize((32,32))
        ]
        if train:
            base.append(transforms.RandomRotation(15))
        base.append(transforms.ToTensor())
        base.append(transforms.Normalize(mean, std))
        return transforms.Compose(channel_adapt + base)
    elif name_l == 'mnist':
        base = []
        if target_size:
            base.append(transforms.Resize((target_size, target_size)))
        base.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return transforms.Compose(channel_adapt + base)
    else:
        raise ValueError(name)


def get_dataloader(
    dataset_name: str,
    batch_size: int = 128,
    data_root: str = './data',
    train: bool = True,
    num_workers: int = 2,
    resize_224: bool = False,
    download: bool = True,
    target_channels: Optional[int] = None,
    target_size: Optional[int] = None,
):
    """统一 DataLoader 构建。
    resize_224: 兼容旧参数；若为 True 则 target_size=224。
    target_channels/target_size: 显式指定输出通道和尺寸。
    返回: (loader, mean, std, num_classes)
    """
    name_l = dataset_name.lower()
    if name_l == 'cifar10':
        Dataset = datasets.CIFAR10; num_classes = 10; ds_kwargs = dict(root=data_root, train=train, download=download)
    elif name_l == 'cifar100':
        Dataset = datasets.CIFAR100; num_classes = 100; ds_kwargs = dict(root=data_root, train=train, download=download)
    elif name_l == 'gtsrb':
        Dataset = datasets.GTSRB; split = 'train' if train else 'test'; num_classes = 43; ds_kwargs = dict(root=data_root, split=split, download=download)
    elif name_l == 'mnist':
        Dataset = datasets.MNIST; num_classes = 10; ds_kwargs = dict(root=data_root, train=train, download=download)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if resize_224:
        target_size = 224

    transform = build_transforms(
        name_l,
        train=train,
        target_channels=target_channels,
        target_size=target_size,
    )
    dataset = Dataset(transform=transform, **ds_kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
    mean, std = get_dataset_stats(name_l, target_channels=target_channels or get_dataset_channels(name_l))
    return loader, mean, std, num_classes

# =========================
# 6. Model Construction
# =========================

def get_model(
    model_name: str,
    num_classes: int,
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
    lenet_in_channels: Optional[int] = None,
    lenet_input_size: Optional[Tuple[int, int]] = None,
):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bias_extractors = {
        'resnet18': lambda m: m.linear.bias,
        'resnet34': lambda m: m.linear.bias,
        'resnet50': lambda m: m.linear.bias,
        'vgg16': lambda m: m.classifier[-1].bias,
        'vgg16_gtsrb': lambda m: m.classifier[-1].bias,
        'efficientnet_b0': lambda m: m.classifier[1].bias,
        'lenet': lambda m: m.fc3.bias,
    }

    if model_name == 'resnet18':
        from model import ResNet18
        model = ResNet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        from model import ResNet34
        model = ResNet34(num_classes=num_classes)
    elif model_name == 'resnet50':
        from model import ResNet50
        model = ResNet50(num_classes=num_classes)
    elif model_name == 'vgg16':
        from model import VGG16_CIFAR10
        model = VGG16_CIFAR10(num_classes=num_classes)
    elif model_name == 'vgg16_gtsrb':
        from model import VGG16_GTSRB
        model = VGG16_GTSRB(num_classes=num_classes)
    elif model_name == 'efficientnet_b0':
        if model_path:
            model = efficientnet_b0(weights=None)
        else:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            model = efficientnet_b0(weights=weights)
        in_feats = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feats, num_classes)
    elif model_name == 'lenet':
        from model import lenet
        in_ch = 1 if lenet_in_channels is None else lenet_in_channels
        inp_hw = (28, 28) if lenet_input_size is None else lenet_input_size
        model = lenet(in_channels=in_ch, num_classes=num_classes, input_size=inp_hw)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_path and os.path.isfile(model_path):
        state = torch.load(model_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if not strict:
            print(f"[Load] missing={missing}, unexpected={unexpected}")

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = True
    bias_extractor = bias_extractors.get(model_name)
    bias = bias_extractor(model) if bias_extractor else None
    if bias is not None:
        bias.requires_grad_(True)
    return model, bias

# =========================
# 7. Normalization Helpers
# =========================

def denormalize(img: torch.Tensor, mean: List[float], std: List[float]):
    mean_t = torch.tensor(mean, device=img.device).view(-1,1,1)
    std_t = torch.tensor(std, device=img.device).view(-1,1,1)
    return img * std_t + mean_t


def normalize_img(img: torch.Tensor, mean: List[float], std: List[float]):
    mean_t = torch.tensor(mean, device=img.device).view(-1,1,1)
    std_t = torch.tensor(std, device=img.device).view(-1,1,1)
    return (img - mean_t) / std_t

# =========================
# 8. Coverage / Fingerprint Sample Selection
# =========================
@dataclass
class CoverageStats:
    total_params: int
    covered_params: int
    new_covered: int
    file_used: str

SUPPORTED_FINGERPRINT_MODELS = (tv_models.EfficientNet,)


def _get_last_fc_weight(model: nn.Module) -> torch.Tensor:
    # 尝试多种常见结构
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features <= 2000:
            # 取最后一个线性层（简单启发式）
            pass
    # 明确匹配
    params = dict(model.named_parameters())
    candidate_keys = [
        'classifier.1.weight', 'classifier[-1].weight', 'linear.weight', 'fc.weight', 'classifier.weight'
    ]
    for k in candidate_keys:
        if k in params:
            return params[k]
    # 回退：寻找最后出现的线性层 weight
    last_key = None
    for k in params:
        if k.endswith('.weight') and params[k].dim() == 2:
            last_key = k
    if last_key is None:
        raise ValueError("Could not locate a final FC weight tensor")
    return params[last_key]


def select_fingerprint_samples(
    model: nn.Module,
    image_dir: str,
    threshold_cov: int = 100,
    threshold_sen: float = 0.1,
    max_keep: Optional[int] = None,
    loss_type: Literal['xe_pred','neg_entropy','margin'] = 'xe_pred',
    out_dir: Optional[str] = None,
    resize: Optional[Tuple[int,int]] = None,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    link: bool = False,
) -> List[CoverageStats]:
    device = next(model.parameters()).device
    model.eval()
    w = _get_last_fc_weight(model)
    c, h = w.shape
    cover = torch.zeros((c,h), dtype=torch.bool, device=device)

    if out_dir is None:
        out_dir = os.path.join(image_dir, f"cov{threshold_cov}_sen{threshold_sen}")
    os.makedirs(out_dir, exist_ok=True)

    # 收集 png / jpg
    files = []
    for root,_,fs in os.walk(image_dir):
        for f in fs:
            if f.lower().endswith(('.png','.jpg','.jpeg')):
                files.append(os.path.join(root,f))
    random.shuffle(files)

    stats: List[CoverageStats] = []

    for fp in files:
        if max_keep is not None and len(stats) >= max_keep:
            break
        img = Image.open(fp).convert('RGB')
        if resize:
            img = img.resize(resize, Image.BILINEAR)
        t = to_tensor(img).unsqueeze(0).to(device)
        if mean is not None and std is not None:
            mean_t = torch.tensor(mean, device=device).view(1,-1,1,1)
            std_t = torch.tensor(std, device=device).view(1,-1,1,1)
            t = (t - mean_t)/std_t

        out = model(t)
        if isinstance(out, tuple):
            out = out[0]
        pred = out.argmax(dim=1)
        if loss_type == 'xe_pred':
            loss = F.cross_entropy(out, pred)
        elif loss_type == 'neg_entropy':
            p = F.softmax(out, dim=1)
            loss = - (p * (p.clamp_min(1e-8)).log()).sum(dim=1).mean()
        else:  # margin
            top2 = torch.topk(out, 2, dim=1).values
            loss = -(top2[:,0] - top2[:,1]).mean()

        model.zero_grad()
        grad_w = torch.autograd.grad(loss, w, retain_graph=False, create_graph=False)[0]
        binary = grad_w.abs() > threshold_sen
        new_cov = ((~cover) & binary).sum().item()
        if new_cov >= threshold_cov:
            # 保存
            dst = os.path.join(out_dir, os.path.basename(fp))
            if link:
                try:
                    os.symlink(os.path.abspath(fp), dst)
                except OSError:
                    shutil.copy(fp, dst)
            else:
                shutil.copy(fp, dst)
            cover |= binary
            stat = CoverageStats(total_params=c*h, covered_params=int(cover.sum().item()), new_covered=int(new_cov), file_used=fp)
            stats.append(stat)
            print(f"[FP] saved {fp}, new={new_cov}, covered={stat.covered_params}/{stat.total_params}")
        else:
            print(f"[FP] skip {fp}, new_cov={new_cov}")
        if cover.all():
            print("[FP] full coverage reached.")
            break
    return stats

# =========================
# 9. Misc Helpers
# =========================

def count_nonzero(t: torch.Tensor) -> int:
    return int((t != 0).sum().item())


def sparsity(t: torch.Tensor) -> float:
    return 1.0 - count_nonzero(t)/t.numel()

__all__ = [
    'seed_everything',
    'quantize_state_dict','quantize_fp32_to_int8_sim','QuantResult',
    'prune_layer','get_submodule','PRUNE_METHODS',
    'entropy_loss_topk','variance_loss_topk','hook_loss_topk','sensitivity_loss_topk',
    'get_dataloader','build_transforms','get_dataset_stats','get_dataset_channels','get_dataset_input_size','get_io_spec',
    'get_model','denormalize','normalize_img',
    'select_fingerprint_samples','CoverageStats',
    'count_nonzero','sparsity'
]


class CriticalityMonitor:
    """
    创新点1支持组件：拓扑临界监视器
    用于捕获 ReLU 层的输入，并计算其是否处于"临界状态" (接近0)
    """
    def __init__(self, model):
        self.hooks = []
        self.layer_inputs = []
        self.model = model

    def register_hooks(self):
        # 自动遍历模型，找到所有的 ReLU 层
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                # 注册 forward_hook，我们需要获取 input (激活前的值)
                hook = module.register_forward_hook(self.hook_fn)
                self.hooks.append(hook)

    def hook_fn(self, module, input, output):
        # input[0] 是 ReLU 之前的张量 (Batch, Channel, H, W)
        # 我们只关心它们的绝对值
        self.layer_inputs.append(input[0])

    def get_criticality_loss(self):
        """
        计算拓扑临界损失：
        迫使神经元的值靠近 0 (处于开关的临界点)
        """
        if not self.layer_inputs:
            return torch.tensor(0.0).cuda()
        
        loss = 0
        for inp in self.layer_inputs:
            # 策略：只惩罚那些绝对值本身就很小的神经元 (比如小于 0.1 的)
            # 让它们变得更小，卡在 0 附近
            abs_val = torch.abs(inp)
            
            # 创建一个 mask，只关注本来就接近边界的值 (focus on critical neurons)
            # 这里阈值选 0.5，可视情况调整
            mask = (abs_val < 0.5).float()
            
            # 计算这些临界神经元的 L2 范数，让它们趋向于 0
            loss += torch.sum((abs_val * mask) ** 2) / (torch.sum(mask) + 1e-8)
            
        # 清空列表，为下一次 Forward 做准备
        self.layer_inputs = [] 
        return loss

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []