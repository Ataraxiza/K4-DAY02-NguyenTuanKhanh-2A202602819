"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Giao diện bạn phải giữ:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except Exception:  # pragma: no cover
    timm = None

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp."""
    if timm is None:
        raise ImportError("timm chưa được cài đặt; cần pip install timm")

    model_name = SUGGESTED_BACKBONES.get(name, name)
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    if init == "frozen":
        freeze_backbone(model)
    if hasattr(model, "pretrained_cfg"):
        cfg = getattr(model, "pretrained_cfg")
        if isinstance(cfg, dict) and "url" in cfg:
            model.pretrained_cfg = cfg
    return model


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head."""
    head = None
    for attr in ("get_classifier", "fc", "head", "classifier"):
        if hasattr(model, attr):
            candidate = getattr(model, attr)
            if callable(candidate):
                head = candidate()
            else:
                head = candidate
            if head is not None:
                break

    head_params = set(id(p) for p in getattr(head, "parameters", lambda: [])()) if head is not None else set()
    for name, param in model.named_parameters():
        if id(param) in head_params:
            continue
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52."""
    groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if any(key in lname for key in ("fc", "head", "classifier")):
            key = (lr_head, weight_decay)
        elif param.ndim <= 1:
            key = (lr_backbone, 0.0)
        else:
            key = (lr_backbone, weight_decay)
        groups.setdefault(key, []).append(param)

    out = [{"params": params, "lr": lr, "weight_decay": wd} for (lr, wd), params in groups.items()]
    return out


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size, dùng hook tự tính."""
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(
        1,
        3,
        img_size,
        img_size,
        device=device,
    )
    total_macs = 0.0

    def hook(module, inputs, output):
        nonlocal total_macs
        if isinstance(module, nn.Conv2d):
            inp = inputs[0]
            c_out = module.out_channels
            kernel_prod = module.kernel_size[0] * module.kernel_size[1]
            total_macs += inp.shape[0] * inp.shape[2] * inp.shape[3] * c_out * kernel_prod * (inp.shape[1] / module.groups)
        elif isinstance(module, nn.Linear):
            inp = inputs[0]
            total_macs += inp.numel() * module.out_features

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    try:
        with torch.no_grad():
            _ = model(dummy)
    finally:
        for h in handles:
            h.remove()
    return total_macs / 1e9
