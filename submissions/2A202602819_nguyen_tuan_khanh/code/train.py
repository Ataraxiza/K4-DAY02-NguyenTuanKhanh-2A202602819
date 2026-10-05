"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

PSEUDO-CODE: chỉ có khung (cấu hình và quy ước đặt tên file); bạn tự hoàn thiện mọi hàm có
`raise NotImplementedError` và các bước TODO trong `run()`. Dùng MỘT hàm `run(cfg)` cho mọi cấu hình
(RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) phải tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

# Ghi file dự đoán đúng định dạng bằng hàm có sẵn trong eval.py (repo gốc):
#     from eval import save_predictions, compute_metrics
# Log theo epoch (history.csv) và config.json bạn tự ghi bằng pandas/json.


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số."""
    try:
        params = model.param_groups(
            cfg.lr_backbone,
            cfg.lr_head,
            cfg.weight_decay,
        )
    except (AttributeError, TypeError):
        from model import param_groups
        params = param_groups(
            model,
            cfg.lr_backbone,
            cfg.lr_head,
            cfg.weight_decay,
        )
    return torch.optim.AdamW(params, weight_decay=0.0)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0."""
    warmup_steps = max(1, int(round(cfg.warmup_epochs * steps_per_epoch)))
    total_steps = max(1, int(cfg.epochs * steps_per_epoch))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps + 1)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class EMA:
    """Trung bình động trọng số."""

    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model) -> None:
        with torch.no_grad():
            current = model.state_dict()
            for key, value in self.shadow.items():
                if key in current:
                    self.shadow[key].mul_(self.decay).add_(current[key].detach(), alpha=1.0 - self.decay)

    def copy_to(self, model) -> None:
        model.load_state_dict(self.shadow, strict=False)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện."""
    model.train()
    if cfg.init == "frozen":
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.eval()
    total_loss = 0.0
    total_batches = 0
    lr_values = []

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise TypeError(f"Batch không hợp lệ trong train_one_epoch: {type(batch)}")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if device.type == 'cuda' and cfg.amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if cfg.mix:
                    from losses import mix_batch, mixed_loss
                    x, (y_a, y_b, lam) = mix_batch(x, y, alpha=cfg.mix_alpha, mode=cfg.mix)
                    logits = model(x)
                    loss = mixed_loss(criterion, logits, (y_a, y_b, lam))
                else:
                    logits = model(x)
                    loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if cfg.mix:
                from losses import mix_batch, mixed_loss
                x, (y_a, y_b, lam) = mix_batch(x, y, alpha=cfg.mix_alpha, mode=cfg.mix)
                logits = model(x)
                loss = mixed_loss(criterion, logits, (y_a, y_b, lam))
            else:
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * x.size(0)
        total_batches += x.size(0)
        lr_values.append(optimizer.param_groups[0]['lr'])

    return {"train_loss": total_loss / max(1, total_batches), "lr": float(np.mean(lr_values))}


def evaluate(model, loader, criterion, device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient."""
    model.eval()
    filenames = []
    y_true = []
    logits_all = []
    loss_sum = 0.0
    n = 0

    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 3:
                x, y, name = batch
            elif len(batch) == 2:
                x, y = batch
                name = [str(i) for i in range(x.shape[0])]
            else:
                raise TypeError(f"Batch không hợp lệ trong evaluate: {type(batch)}")

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            if criterion is not None:
                loss_sum += criterion(logits, y).item() * x.size(0)
            filenames.extend([str(v) for v in name])
            y_true.append(y.detach().cpu().numpy())
            logits_all.append(logits.detach().cpu().numpy())
            n += x.size(0)

    if not logits_all:
        return [], np.asarray([], dtype=np.int64), np.empty((0, 0), dtype=np.float32), 0.0

    logits_all = np.concatenate(logits_all, axis=0)
    y_true = np.concatenate(y_true, axis=0).astype(np.int64)
    return filenames, y_true, logits_all, loss_sum / max(1, n)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    if history:
        ax1.plot(epochs, [h.get("train_loss", np.nan) for h in history], label="train loss", marker="o")
        ax1.plot(epochs, [h.get("val_loss", np.nan) for h in history], label="val loss", marker="s")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        ax2.plot(epochs, [h.get("macro_f1", np.nan) for h in history], label="macro-F1 val", marker="o", color="tab:green")
        ax2.set_title("Validation macro-F1")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Macro-F1")
        ax2.legend()

        if any('lr' in h for h in history):
            ax3 = ax2.twinx()
            ax3.plot(epochs, [h.get("lr", np.nan) for h in history], label="lr", linestyle="--", color="tab:orange")
            ax3.set_ylabel("LR")
            ax3.legend(loc="lower right")
    fig.suptitle(title)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết."""
    set_seed(cfg.seed)
    import sys
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import dataset
    import losses
    import model as model_mod
    import eval as eval_lib

    run_root = run_dir(cfg)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)

    train_tf = dataset.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = dataset.build_transforms(train=False, img_size=cfg.img_size, aug=cfg.aug)
    train_loader = dataset.make_loader(train_df, cfg.images_dir, train_tf, cfg.batch_size, train=True,
                                      sampler=cfg.sampler, num_workers=cfg.num_workers)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, val_tf, cfg.batch_size, train=False,
                                    sampler=None, num_workers=cfg.num_workers)
    test_loader = None
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(test_df, cfg.images_dir, val_tf, cfg.batch_size, train=False,
                                         sampler=None, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_mod.build_model(cfg.backbone, pretrained=(cfg.init != "scratch"), num_classes=9,
                                  drop_rate=cfg.drop_rate, init=cfg.init)
    model.to(device)

    counts = train_df["Label"].value_counts().reindex(range(9), fill_value=0).to_numpy(dtype=np.float64)
    weight = None
    if cfg.loss in {"ce_weighted", "weighted_ce"} or cfg.class_weight_beta is not None:
        beta = cfg.class_weight_beta if cfg.class_weight_beta is not None else 0.0
        weight = losses.class_weights(counts, beta=beta)
    criterion = losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma,
                                       alpha=None, weight=weight)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, max(1, len(train_loader)))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == 'cuda')
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None

    history = []
    best_state = None
    best_ema_state = None
    best_f1 = -1.0
    best_epoch = -1
    start = time()

    for epoch in range(cfg.epochs):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        val_names, y_val, val_logits, val_loss = evaluate(model, val_loader, criterion, device)
        val_probs = np.exp(val_logits - val_logits.max(axis=1, keepdims=True))
        val_probs /= val_probs.sum(axis=1, keepdims=True)
        metrics = eval_lib.compute_metrics(y_val, val_probs.argmax(1), val_probs)
        val_macro = float(metrics["macro_f1"])
        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": float(train_stats["train_loss"]),
            "val_loss": float(val_loss),
            "macro_f1": val_macro,
            "lr": float(train_stats["lr"]),
        }
        history.append(epoch_data)

        if val_macro > best_f1 or (abs(val_macro - best_f1) < 1e-12 and best_epoch < 0):
            best_f1 = val_macro
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ema is not None:
                best_ema_state = {k: v.detach().clone() for k, v in ema.shadow.items()}

    if ema is not None and best_ema_state is not None:
        model.load_state_dict(best_ema_state, strict=False)
    elif best_state is not None:
        model.load_state_dict(best_state)

    val_names, y_val, val_logits, val_loss = evaluate(model, val_loader, criterion, device)
    val_probs = np.exp(val_logits - val_logits.max(axis=1, keepdims=True))
    val_probs /= val_probs.sum(axis=1, keepdims=True)
    val_pred = val_probs.argmax(1)

    final_metrics = eval_lib.compute_metrics(y_val, val_pred, val_probs)
    val_macro_f1 = float(final_metrics["macro_f1"])
    val_top1 = float((val_pred == y_val).mean())

    pred_path_val = pred_path(cfg, "val")
    pred_path_val.parent.mkdir(parents=True, exist_ok=True)
    eval_lib.save_predictions(pred_path_val, val_names, y_val, val_probs)

    if cfg.save_test_predictions and test_loader is not None:
        test_names, y_test, test_logits, _ = evaluate(model, test_loader, criterion, device)
        test_probs = np.exp(test_logits - test_logits.max(axis=1, keepdims=True))
        test_probs /= test_probs.sum(axis=1, keepdims=True)
        eval_lib.save_predictions(pred_path(cfg, "test"), test_names, y_test, test_probs)

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_root / "history.csv", index=False)
    plot_curves(history, run_root / "curves.png", f"{cfg.exp_id} | seed {cfg.seed}")

    summary = {
        "best_epoch": int(best_epoch),
        "val_macro_f1": val_macro_f1,
        "val_top1": val_top1,
        "train_time_per_epoch": float((time() - start) / max(1, len(history))),
        "params_m": float(model_mod.count_params(model)),
        "gmacs": float(model_mod.count_gmacs(model, cfg.img_size)),
    }
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict."""
    if pairs is None:
        return {}
    values = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override không hợp lệ: {pair!r}; cần dạng KEY=VALUE")
        key, value = pair.split("=", 1)
        key = key.strip()
        if key not in Config.__dataclass_fields__:
            raise ValueError(f"Key {key!r} không có trong Config")
        field = Config.__dataclass_fields__[key]
        default = field.default
        if value.strip().lower() in {"none", "null", "nan"}:
            parsed = None
        elif isinstance(default, bool):
            parsed = value.strip().lower() in {"1", "true", "yes", "y"}
        elif isinstance(default, int):
            parsed = int(value)
        elif isinstance(default, float):
            parsed = float(value)
        else:
            parsed = value
        values[key] = parsed
    return values


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="*", default=[], help="Các override dạng KEY=VALUE")
    args = parser.parse_args()

    cfg = Config()
    overrides = parse_overrides(args.set)
    for key, value in overrides.items():
        setattr(cfg, key, value)

    result = run(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
