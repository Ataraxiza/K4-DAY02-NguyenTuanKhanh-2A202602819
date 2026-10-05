"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Giao diện:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from torchvision import transforms


NUM_CLASSES = 9

CLASS_NAMES = [
    "Chinee Apple",
    "Lantana",
    "Parkinsonia",
    "Parthenium",
    "Prickly Acacia",
    "Rubber Vine",
    "Siam Weed",
    "Snake Weed",
    "Negatives",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EXPECTED_TOTAL = 17509


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train/val/test CSV của fold được chọn."""

    labels_dir = Path(labels_dir)

    train_path = labels_dir / f"train_subset{fold}.csv"
    val_path = labels_dir / f"val_subset{fold}.csv"
    test_path = labels_dir / f"test_subset{fold}.csv"

    # Fail early nếu file không tồn tại.
    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Các CSV split thực tế chỉ có Filename và Label.
    required_columns = {"Filename", "Label"}

    for name, df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name}_df thiếu cột: {sorted(missing)}"
            )

    return train_df, val_df, test_df


def check_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    images_dir: str | Path,
) -> dict:
    """Kiểm tra split trước khi train."""

    images_dir = Path(images_dir)

    dfs = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    # ---------------------------------------------------------
    # 1. Kiểm tra số ảnh và số ảnh theo lớp
    # ---------------------------------------------------------
    n = {
        name: int(len(df))
        for name, df in dfs.items()
    }

    per_class = {}

    for name, df in dfs.items():
        counts = (
            df["Label"]
            .value_counts()
            .reindex(range(NUM_CLASSES), fill_value=0)
            .astype(int)
        )

        per_class[name] = {
            int(label): int(count)
            for label, count in counts.items()
        }

    # Mỗi filename chỉ nên xuất hiện một lần trong một split.
    for name, df in dfs.items():
        duplicated = df["Filename"][
            df["Filename"].duplicated()
        ].tolist()

        assert not duplicated, (
            f"{name} có Filename bị trùng. "
            f"Ví dụ: {duplicated[:5]}"
        )

    # ---------------------------------------------------------
    # 2. Kiểm tra overlap giữa các split
    # ---------------------------------------------------------
    train_files = set(train_df["Filename"])
    val_files = set(val_df["Filename"])
    test_files = set(test_df["Filename"])

    overlap = {
        "train_val": len(train_files & val_files),
        "train_test": len(train_files & test_files),
        "val_test": len(val_files & test_files),
    }

    assert overlap["train_val"] == 0, (
        f"train ∩ val không rỗng: {overlap['train_val']} ảnh"
    )

    assert overlap["train_test"] == 0, (
        f"train ∩ test không rỗng: {overlap['train_test']} ảnh"
    )

    assert overlap["val_test"] == 0, (
        f"val ∩ test không rỗng: {overlap['val_test']} ảnh"
    )

    # ---------------------------------------------------------
    # 3. Kiểm tra tổng số ảnh
    # ---------------------------------------------------------
    all_files = train_files | val_files | test_files

    assert len(all_files) == EXPECTED_TOTAL, (
        f"Tổng số Filename duy nhất = {len(all_files)}, "
        f"kỳ vọng {EXPECTED_TOTAL}"
    )

    assert (
        len(train_df) + len(val_df) + len(test_df)
        == EXPECTED_TOTAL
    ), (
        "Tổng số dòng của train/val/test không bằng "
        f"{EXPECTED_TOTAL}"
    )

    # ---------------------------------------------------------
    # 4. Kiểm tra mọi file ảnh tồn tại
    # ---------------------------------------------------------
    missing_files = []

    for filename in all_files:
        image_path = images_dir / str(filename)

        if not image_path.is_file():
            missing_files.append(str(filename))

    assert not missing_files, (
        f"Không tìm thấy {len(missing_files)} ảnh trong "
        f"{images_dir}. Ví dụ: {missing_files[:10]}"
    )

    # ---------------------------------------------------------
    # In báo cáo
    # ---------------------------------------------------------
    print("=== Split check ===")

    print("\nSố ảnh:")
    for name, count in n.items():
        print(f"  {name:5s}: {count}")

    print("\nSố ảnh theo lớp:")
    for name, counts in per_class.items():
        print(f"\n{name}:")

        for label, count in counts.items():
            class_name = CLASS_NAMES[label]

            print(
                f"  {label}: {class_name:16s} -> {count}"
            )

    print("\nOverlap:")

    for pair, count in overlap.items():
        print(f"  {pair:12s}: {count}")

    print(f"\nTổng ảnh duy nhất: {len(all_files)}")
    print(f"Ảnh thiếu: {len(missing_files)}")

    return {
        "n": n,
        "per_class": per_class,
        "overlap": overlap,
        "total_unique": len(all_files),
        "missing": len(missing_files),
    }


def build_transforms(
    train: bool,
    img_size: int = 224,
    aug: str = "basic",
):
    """Tạo torchvision transform cho train hoặc val/test."""

    if img_size <= 0:
        raise ValueError("img_size phải > 0")

    valid_aug = {
        "basic",
        "color",
        "trivial",
        "randaug",
    }

    if aug not in valid_aug:
        raise ValueError(
            f"aug={aug!r} không hợp lệ. "
            f"Chọn một trong {sorted(valid_aug)}"
        )

    if not train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ])

    # ---------------------------------------------------------
    # Train augmentation
    # ---------------------------------------------------------
    ops = [
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.8, 1.0),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    if aug == "basic":
        pass

    elif aug == "color":
        ops.extend([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
        ])

    elif aug == "trivial":
        ops.extend([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomGrayscale(p=0.05),
        ])

    elif aug == "randaug":
        ops.extend([
            transforms.RandAugment(
                num_ops=2,
                magnitude=9,
            ),
        ])

    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    return transforms.Compose(ops)


class DeepWeedsDataset(Dataset):
    """Dataset đọc DeepWeeds từ DataFrame."""

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str | Path,
        transform=None,
    ):
        required_columns = {"Filename", "Label"}

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"DataFrame thiếu cột: {sorted(missing)}"
            )

        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]

        filename = str(row["Filename"])
        label = int(row["Label"])

        image_path = self.images_dir / filename

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy ảnh: {image_path}"
            )

        # RGB đảm bảo mọi ảnh đều có 3 channels.
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, filename


def make_loader(
    df: pd.DataFrame,
    images_dir: str | Path,
    transform,
    batch_size: int,
    train: bool,
    sampler: str | None = None,
    num_workers: int = 2,
):
    """Tạo DataLoader cho DeepWeeds."""

    if batch_size <= 0:
        raise ValueError("batch_size phải > 0")

    if num_workers < 0:
        raise ValueError("num_workers phải >= 0")

    if sampler not in (None, "balanced"):
        raise ValueError(
            f"sampler={sampler!r} không hợp lệ. "
            "Chọn None hoặc 'balanced'."
        )

    dataset = DeepWeedsDataset(
        df=df,
        images_dir=images_dir,
        transform=transform,
    )

    weighted_sampler = None
    shuffle = False

    # ---------------------------------------------------------
    # Balanced sampler
    # ---------------------------------------------------------
    if sampler == "balanced":
        if not train:
            raise ValueError(
                "sampler='balanced' chỉ nên dùng cho train=True."
            )

        labels = df["Label"].astype(int).to_numpy()

        class_counts = (
            pd.Series(labels)
            .value_counts()
            .to_dict()
        )

        weights = [
            1.0 / class_counts[int(label)]
            for label in labels
        ]

        weighted_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(
                weights,
                dtype=torch.double,
            ),
            num_samples=len(weights),
            replacement=True,
        )

    elif train:
        shuffle = True

    # ---------------------------------------------------------
    # Worker seeding
    # ---------------------------------------------------------
    def seed_worker(worker_id: int):
        # torch.initial_seed() được DataLoader tự tạo
        # theo seed của generator/worker.
        worker_seed = torch.initial_seed() % (2**32)

        import numpy as np
        import random

        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(42)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if weighted_sampler is None else False,
        sampler=weighted_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return loader
