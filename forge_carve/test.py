import sys

import torch

try:
    import faiss  # type: ignore
except Exception as exc:
    print(f"faiss import failed: {exc}")
    sys.exit(1)

gpu_available = torch.cuda.is_available()
gpu_count = torch.cuda.device_count() if gpu_available else 0
faiss_gpu_count = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0

print(f"torch_cuda_available={gpu_available}")
print(f"torch_cuda_device_count={gpu_count}")
print(f"faiss_gpu_available={faiss_gpu_count > 0}")
print(f"faiss_gpu_count={faiss_gpu_count}")
