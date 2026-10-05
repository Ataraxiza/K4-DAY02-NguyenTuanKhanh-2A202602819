"""benchmark.py - đo độ trễ suy luận đúng cách.

Quy tắc:
  - warmup >= 10 lần
  - synchronize GPU trước và sau đoạn đo
  - >= 50 lần đo
  - báo cáo p50, p95, p99
  - ghi rõ GPU, dtype, batch, độ phân giải, torch version
"""
from __future__ import annotations

import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian chạy fn() và trả về latency theo milliseconds.

    Args:
        fn: callable không có argument.
        warmup: số lần chạy đầu bỏ qua.
        iters: số lần đo thật.
        sync: hàm đồng bộ, ví dụ torch.cuda.synchronize.

    Returns:
        {
            "p50": ...,
            "p95": ...,
            "p99": ...,
            "mean": ...,
            "n": iters,
        }
    """

    if warmup < 0:
        raise ValueError("warmup phải >= 0")

    if iters < 50:
        raise ValueError(
            "iters phải >= 50 theo yêu cầu benchmark."
        )

    if sync is None:
        sync_fn = lambda: None
    else:
        sync_fn = sync

    # ---------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------
    for _ in range(warmup):
        fn()

    # Đảm bảo tất cả warmup đã hoàn thành trước khi đo.
    sync_fn()

    # ---------------------------------------------------------
    # Timed iterations
    # ---------------------------------------------------------
    times_ms = []

    for _ in range(iters):
        # GPU execution là asynchronous nên phải sync trước
        # khi bắt đầu timestamp.
        sync_fn()

        t0 = time.perf_counter()

        fn()

        # Đảm bảo đoạn fn() thực sự hoàn thành trước khi
        # đọc thời gian.
        sync_fn()

        t1 = time.perf_counter()

        times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.asarray(times_ms, dtype=np.float64)

    return {
        "p50": float(np.percentile(times_ms, 50)),
        "p95": float(np.percentile(times_ms, 95)),
        "p99": float(np.percentile(times_ms, 99)),
        "mean": float(np.mean(times_ms)),
        "n": int(iters),
    }


def latency_report(
    model,
    batch_size: int,
    img_size: int,
    dtype: str = "fp32",
    device: str = "cuda",
    warmup: int = 10,
    iters: int = 100,
) -> dict:
    """Đo latency forward của model.

    dtype:
        "fp32" -> FP32
        "amp"  -> autocast AMP
        "fp16" -> model.half() + FP16 input

    Tiền xử lý KHÔNG được tính.
    Chỉ đo model forward.

    Returns:
        Dict phù hợp để ghi vào sheet Latency.
    """

    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError(
            f"dtype={dtype!r} không hợp lệ. "
            "Chọn 'fp32', 'amp' hoặc 'fp16'."
        )

    if batch_size <= 0:
        raise ValueError("batch_size phải > 0")

    if img_size <= 0:
        raise ValueError("img_size phải > 0")

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' nhưng CUDA không khả dụng."
        )

    dev = torch.device(device)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = model.to(dev)
    model.eval()

    # Không mutate model dtype nếu dùng AMP.
    if dtype == "fp16":
        model = model.half()

    # ---------------------------------------------------------
    # Input
    # ---------------------------------------------------------
    x = torch.randn(
        batch_size,
        3,
        img_size,
        img_size,
        device=dev,
        dtype=torch.float32,
    )

    if dtype == "fp16":
        x = x.half()

    # ---------------------------------------------------------
    # AMP context
    # ---------------------------------------------------------
    use_amp = dtype == "amp"

    if use_amp and dev.type == "cuda":
        amp_context = torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
    else:
        # AMP CPU không phải mục tiêu benchmark của lab.
        # Nếu chạy CPU, giữ FP32.
        amp_context = torch.autocast(
            device_type=dev.type,
            enabled=False,
        )

    def forward():
        with torch.inference_mode():
            with amp_context:
                _ = model(x)

    # ---------------------------------------------------------
    # Synchronization
    # ---------------------------------------------------------
    sync = torch.cuda.synchronize if dev.type == "cuda" else None

    result = bench(
        forward,
        warmup=max(warmup, 10),
        iters=max(iters, 50),
        sync=sync,
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------
    if dev.type == "cuda":
        gpu_name = torch.cuda.get_device_name(dev)
    else:
        gpu_name = "CPU"

    p50 = result["p50"]

    images_per_s = (
        float(batch_size / (p50 / 1000.0))
        if p50 > 0
        else float("inf")
    )

    report = {
        "gpu": gpu_name,
        "dtype": dtype.upper(),
        "batch": int(batch_size),
        "img_size": int(img_size),
        "p50": result["p50"],
        "p95": result["p95"],
        "p99": result["p99"],
        "mean": result["mean"],
        "images_per_s": images_per_s,
        "torch": torch.__version__,
        "preprocessing": "excluded",
        "warmup": max(warmup, 10),
        "iters": max(iters, 50),
    }

    return report


def tta_latency(model, k_views: int, **kw) -> dict:
    """Đo latency của TTA K-view.

    Vì hàm interface chỉ nhận model và k_views, không nhận danh sách
    augmentation cụ thể, benchmark này mô phỏng TTA bằng K forward
    passes trên K input views.

    Các input views được tạo trước và KHÔNG tính vào latency.

    kw được truyền cho latency_report(), ví dụ:
        batch_size=1
        img_size=224
        dtype="fp32"
        device="cuda"
        warmup=10
        iters=100

    Returns thêm:
        k_views
        single_view_p50
        expected_k_times_p50
        measured_tta_p50
        relative_to_single
    """

    if k_views < 1:
        raise ValueError("k_views phải >= 1")

    # ---------------------------------------------------------
    # Kiểm tra / chuẩn hoá benchmark parameters
    # ---------------------------------------------------------
    batch_size = kw.get("batch_size", 1)
    img_size = kw.get("img_size", 224)
    dtype = kw.get("dtype", "fp32")
    device = kw.get("device", "cuda")
    warmup = kw.get("warmup", 10)
    iters = kw.get("iters", 100)

    if batch_size <= 0:
        raise ValueError("batch_size phải > 0")

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' nhưng CUDA không khả dụng."
        )

    dev = torch.device(device)

    # ---------------------------------------------------------
    # Prepare model
    # ---------------------------------------------------------
    model = model.to(dev)
    model.eval()

    if dtype == "fp16":
        model = model.half()

    elif dtype not in {"fp32", "amp"}:
        raise ValueError(
            f"dtype={dtype!r} không hợp lệ."
        )

    # ---------------------------------------------------------
    # Prepare K views.
    #
    # Các tensor được tạo trước benchmark nên preprocessing /
    # tạo input không nằm trong latency.
    # ---------------------------------------------------------
    input_dtype = (
        torch.float16
        if dtype == "fp16"
        else torch.float32
    )

    views = [
        torch.randn(
            batch_size,
            3,
            img_size,
            img_size,
            device=dev,
            dtype=input_dtype,
        )
        for _ in range(k_views)
    ]

    if dtype == "amp" and dev.type == "cuda":
        amp_context = torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
    else:
        amp_context = torch.autocast(
            device_type=dev.type,
            enabled=False,
        )

    def single_view_forward():
        with torch.inference_mode():
            with amp_context:
                _ = model(views[0])

    def tta_forward():
        with torch.inference_mode():
            with amp_context:
                for x in views:
                    _ = model(x)

    sync = (
        torch.cuda.synchronize
        if dev.type == "cuda"
        else None
    )

    # ---------------------------------------------------------
    # Measure single-view baseline
    # ---------------------------------------------------------
    single = bench(
        single_view_forward,
        warmup=max(warmup, 10),
        iters=max(iters, 50),
        sync=sync,
    )

    # ---------------------------------------------------------
    # Measure actual K-view TTA
    # ---------------------------------------------------------
    tta = bench(
        tta_forward,
        warmup=max(warmup, 10),
        iters=max(iters, 50),
        sync=sync,
    )

    single_p50 = single["p50"]
    tta_p50 = tta["p50"]

    expected = k_views * single_p50

    relative = (
        tta_p50 / single_p50
        if single_p50 > 0
        else float("inf")
    )

    if dev.type == "cuda":
        gpu_name = torch.cuda.get_device_name(dev)
    else:
        gpu_name = "CPU"

    result = {
        "gpu": gpu_name,
        "dtype": dtype.upper(),
        "batch": int(batch_size),
        "img_size": int(img_size),
        "k_views": int(k_views),

        # Single-view measurements
        "single_view_p50": single["p50"],
        "single_view_p95": single["p95"],
        "single_view_p99": single["p99"],

        # Actual TTA measurements
        "p50": tta["p50"],
        "p95": tta["p95"],
        "p99": tta["p99"],

        # Comparison requested by GUIDE 4.1
        "expected_k_times_p50": float(expected),
        "relative_to_single": float(relative),

        "torch": torch.__version__,
        "preprocessing": "excluded",
        "warmup": max(warmup, 10),
        "iters": max(iters, 50),
    }

    return result
