from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None


@dataclass(frozen=True)
class RuntimeInfo:
    device: torch.device
    gpu_available: bool
    gpu_count: int
    faiss_gpu_available: bool
    faiss_gpu_count: int
    cpu_threads: int
    faiss_resources: Any | None = None


def _detect_faiss_gpus() -> int:
    if faiss is None:
        return 0
    get_num = getattr(faiss, "get_num_gpus", None)
    if callable(get_num):
        try:
            return int(get_num())
        except Exception:
            return 0
    return 0


def _configure_threads(cpu_threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))
    try:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
    except Exception:
        pass


def configure_runtime(
    logger: logging.Logger | None = None,
    device: str | None = None,
    cpu_threads: int = 16,
) -> RuntimeInfo:
    gpu_available = torch.cuda.is_available()
    if device is None:
        use_gpu = gpu_available
        device_str = "cuda" if use_gpu else "cpu"
    else:
        device_str = device
        use_gpu = device_str.startswith("cuda")
        if use_gpu and not gpu_available:
            device_str = "cpu"
            use_gpu = False
    torch_device = torch.device(device_str)
    gpu_count = torch.cuda.device_count() if gpu_available else 0

    _configure_threads(cpu_threads)

    if use_gpu:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    faiss_gpu_count = _detect_faiss_gpus()
    faiss_gpu_available = faiss_gpu_count > 0
    faiss_resources = None
    if faiss_gpu_available and faiss is not None:
        try:
            faiss_resources = faiss.StandardGpuResources()
        except Exception:
            faiss_resources = None

    if logger is not None:
        if use_gpu:
            logger.info(
                "runtime gpu_enabled=true gpu_count=%d device=%s",
                gpu_count,
                torch_device,
            )
        else:
            logger.info(
                "runtime gpu_enabled=false device=cpu cpu_threads=%d",
                cpu_threads,
            )
        logger.info(
            "runtime faiss_gpu_available=%s faiss_gpu_count=%d",
            str(faiss_gpu_available).lower(),
            faiss_gpu_count,
        )

    return RuntimeInfo(
        device=torch_device,
        gpu_available=use_gpu,
        gpu_count=gpu_count,
        faiss_gpu_available=faiss_gpu_available,
        faiss_gpu_count=faiss_gpu_count,
        cpu_threads=cpu_threads,
        faiss_resources=faiss_resources,
    )
