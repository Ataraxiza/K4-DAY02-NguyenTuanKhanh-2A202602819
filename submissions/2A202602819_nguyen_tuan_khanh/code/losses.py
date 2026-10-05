"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm/lớp có `raise NotImplementedError`.
Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện bạn phải giữ:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted"."""
    kind = (kind or "ce").lower()
    if kind in {"ce", "cross_entropy"}:
        weight = kw.get("weight")
        return nn.CrossEntropyLoss(weight=weight)
    if kind == "ls":
        return LabelSmoothingCE(smoothing=float(kw.get("smoothing", 0.1)))
    if kind == "focal":
        return FocalLoss(gamma=float(kw.get("gamma", 2.0)), alpha=kw.get("alpha"))
    if kind in {"ce_weighted", "weighted_ce"}:
        return nn.CrossEntropyLoss(weight=kw.get("weight"))
    raise ValueError(f"criterion={kind!r} không hỗ trợ")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = float(smoothing)
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError(f"smoothing phải trong [0,1), nhận {self.smoothing}")

    def forward(self, logits, target):
        if self.smoothing == 0.0:
            return F.cross_entropy(logits, target)

        log_probs = F.log_softmax(logits, dim=1)
        n_classes = logits.size(1)
        with torch.no_grad():
            target_onehot = F.one_hot(target, num_classes=n_classes).float()
            smooth_target = target_onehot * (1.0 - self.smoothing) + self.smoothing / n_classes
        loss = -(smooth_target * log_probs).sum(dim=1)
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL = -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        target = target.long()
        p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        log_p_t = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        alpha_t = torch.ones_like(p_t)
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[target]
        loss = -alpha_t * ((1.0 - p_t).pow(self.gamma)) * log_p_t
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN."""
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.clip(counts, 1e-9, None)
    if beta <= 0:
        w = 1.0 / counts
        w = w / w.mean()
    else:
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
        w = w * (len(w) / w.sum())
    return torch.as_tensor(w, dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn."""
    if x.ndim != 4:
        raise ValueError(f"x phải tensor 4D [B,C,H,W], nhận {tuple(x.shape)}")
    if y.ndim != 1:
        raise ValueError(f"y phải tensor 1D [B], nhận {tuple(y.shape)}")

    b, _, h, w = x.shape
    perm = torch.randperm(b, device=x.device)
    y_a = y
    y_b = y[perm]
    if alpha <= 0.0:
        lam = 1.0
    else:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()

    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
        return x_mix, (y_a, y_b, float(lam))

    if mode == "cutmix":
        cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        cx = torch.randint(0, w - cut_w + 1, (1,), device=x.device).item()
        cy = torch.randint(0, h - cut_h + 1, (1,), device=x.device).item()
        x_mix = x.clone()
        x_mix[:, :, cy:cy + cut_h, cx:cx + cut_w] = x[perm, :, cy:cy + cut_h, cx:cx + cut_w]
        lam = 1.0 - (cut_w * cut_h) / (w * h)
        return x_mix, (y_a, y_b, float(lam))

    raise ValueError(f"mode={mode!r} không hợp lệ; chọn 'mixup' hoặc 'cutmix'")


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn bằng trọng số lam."""
    y_a, y_b, lam = targets
    loss_a = criterion(logits, y_a)
    loss_b = criterion(logits, y_b)
    return lam * loss_a + (1.0 - lam) * loss_b
