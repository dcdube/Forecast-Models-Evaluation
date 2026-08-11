"""Shared command-line GPU selection helpers."""

SUPPORTED_GPU_IDS = (0, 1)
GPU_HELP = (
    "GPU index for supported local models: 0 uses GPU 0, 1 uses GPU 1; "
    "any other value uses CPU."
)

def uses_gpu(gpu):
    """Return whether ``gpu`` selects one of the supported CUDA devices."""
    return gpu in SUPPORTED_GPU_IDS


def torch_device(gpu):
    """Return the PyTorch device selected by the common ``--gpu`` option."""
    return f"cuda:{gpu}" if uses_gpu(gpu) else "cpu"


def lightning_device_config(gpu):
    """Return PyTorch Lightning trainer arguments for the selected device."""
    if uses_gpu(gpu):
        return {"accelerator": "gpu", "devices": [gpu]}
    return {"accelerator": "cpu", "devices": 1}


def execution_target(gpu):
    """Return a human-readable name for logs."""
    return f"GPU {gpu}" if uses_gpu(gpu) else "CPU"
