"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm phải chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện bạn nên giữ:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.eval()
    filenames = []
    labels = []
    logits_list = []

    autocast = nullcontext()
    if device.type == "cuda":
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16)

    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                if len(batch) == 3:
                    x, y, name = batch[:3]
                else:
                    x, y = batch[:2]
                    name = [str(i) for i in range(x.shape[0])]
            else:
                raise TypeError(f"Batch không đúng định dạng: {type(batch)}")

            if view is not None:
                x = view(x)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast:
                logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            logits = logits.detach().cpu().float().numpy()

            filenames.extend([str(v) for v in name])
            labels.append(y.detach().cpu().numpy())
            logits_list.append(logits)

    if not logits_list:
        return [], np.asarray([], dtype=np.int64), np.empty((0, 0), dtype=np.float32)

    return filenames, np.concatenate(labels, axis=0).astype(np.int64), np.concatenate(logits_list, axis=0)


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop`, và tuỳ chọn thêm bản lật."""
    if x.ndim != 4:
        raise ValueError(f"views_multicrop cần tensor 4D [N,C,H,W], nhận {tuple(x.shape)}")
    _, _, h, w = x.shape
    crop = int(min(crop, h, w))
    if crop <= 0:
        return [x]
    ys = [0, 0, h - crop, h - crop, max(0, (h - crop) // 2)]
    xs = [0, w - crop, 0, w - crop, max(0, (w - crop) // 2)]
    views = []
    for y0, x0 in zip(ys, xs):
        patch = x[:, :, y0:y0 + crop, x0:x0 + crop]
        views.append(patch)
        views.append(torch.flip(patch, dims=[-1]))
    return views


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch."""
    out = []
    for size in sizes:
        if size <= 0:
            continue
        out.append(F.interpolate(x, size=(int(size), int(size)), mode="bilinear", align_corners=False))
    return out


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán."""
    if not logits_per_view:
        raise ValueError("logits_per_view rỗng")
    arr = np.asarray(logits_per_view, dtype=np.float64)
    if space == "prob":
        return np.stack([_softmax(z) for z in arr], axis=0).mean(axis=0)
    if space == "logit":
        return _softmax(arr.mean(axis=0))
    raise ValueError(f"space={space!r} không hợp lệ; chọn 'prob' hoặc 'logit'")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình."""
    if not list_of_probs:
        raise ValueError("list_of_probs rỗng")
    arr = np.asarray(list_of_probs, dtype=np.float64)
    out = arr.mean(axis=0)
    return out / out.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL."""
    val_logits = np.asarray(val_logits, dtype=np.float64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    if val_logits.ndim != 2:
        raise ValueError(f"val_logits phải có dạng (N, K), nhận {val_logits.shape}")

    best_T = 1.0
    best_nll = float("inf")
    for T in np.exp(np.linspace(np.log(0.1), np.log(10.0), 500)):
        probs = _softmax(val_logits / T)
        p = probs[np.arange(len(val_labels)), val_labels]
        nll = -np.log(np.clip(p, 1e-12, 1.0)).mean()
        if nll < best_nll:
            best_nll = float(nll)
            best_T = float(T)
    return best_T


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T)."""
    logits = np.asarray(logits, dtype=np.float64)
    if float(T) <= 0:
        raise ValueError(f"T phải > 0, nhận {T}")
    return _softmax(logits / float(T))


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước."""
    model.eval()

    def _recurse(module):
        children = list(module.named_children())
        for idx, (name, child) in enumerate(children):
            if isinstance(child, nn.Conv2d) and idx + 1 < len(children):
                next_name, next_child = children[idx + 1]
                if isinstance(next_child, nn.BatchNorm2d):
                    bn = next_child
                    conv = child
                    gamma = bn.weight.detach().clone() if bn.weight is not None else torch.ones_like(bn.running_mean)
                    beta = bn.bias.detach().clone() if bn.bias is not None else torch.zeros_like(bn.running_mean)
                    mean = bn.running_mean.detach().clone()
                    var = bn.running_var.detach().clone()
                    denom = torch.sqrt(var + bn.eps)
                    scale = (gamma / denom).reshape(-1, *([1] * (conv.weight.dim() - 1)))
                    new_weight = conv.weight.detach() * scale
                    if conv.bias is not None:
                        new_bias = beta + (conv.bias.detach() - mean) * (gamma / denom)
                    else:
                        new_bias = beta - mean * (gamma / denom)
                    fused = nn.Conv2d(
                        conv.in_channels,
                        conv.out_channels,
                        conv.kernel_size,
                        stride=conv.stride,
                        padding=conv.padding,
                        dilation=conv.dilation,
                        groups=conv.groups,
                        bias=True,
                        padding_mode=conv.padding_mode,
                    )
                    fused.weight.data.copy_(new_weight)
                    fused.bias.data.copy_(new_bias)
                    setattr(module, name, fused)
                    setattr(module, next_name, nn.Identity())
                    continue
            _recurse(child)

    _recurse(model)
    return model
