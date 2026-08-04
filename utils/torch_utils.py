# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import logging

try:
    import torch
except Exception:  # Torch is required for execution, but discovery must still be safe.
    torch = None

from modiff.hardware import get_normalized_devices, legacy_device_list


logger = logging.getLogger('modiff')

_CPU_FALLBACK = {
    'cpu:0': {
        'arch': 'cpu',
        'name': 'CPU (0)',
        'label': ['cpu:0'],
        'total_memory': 0,
        'index': 0,
    }
}

def device_list():
    try:
        normalized = get_normalized_devices(torch_module=torch)
        devices = legacy_device_list({'devices': normalized})
        return devices or {key: dict(value) for key, value in _CPU_FALLBACK.items()}
    except Exception as exc:
        logger.warning(f'Hardware device discovery failed; using the CPU fallback: {exc}')
        return {key: dict(value) for key, value in _CPU_FALLBACK.items()}

DEVICE_LIST = device_list()
DEFAULT_DEVICE = list(DEVICE_LIST.keys())[0]
CPU_DEVICE = 'cpu:0' if 'cpu:0' in DEVICE_LIST else DEFAULT_DEVICE
IS_CUDA = any('cuda' in device for device in DEVICE_LIST.keys())

def str_to_dtype(dtype, *args):
    return {
        'auto': None,
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float8_e4m3fn': torch.float8_e4m3fn,
    }[dtype]

def compile(model):
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_check_all_directions = True
    model.to(memory_format=torch.channels_last)

    return torch.compile(model, mode='max-autotune', fullgraph=True)

def TensorToImage(tensor):
    from torchvision.transforms import v2 as tt

    if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
        tensor = list(tensor.unbind(0))

    tensor = tensor if isinstance(tensor, list) else [tensor]
    output = []
    for t in tensor:
        if t.ndim == 4:
            t = t.squeeze(0)
        output.append(tt.ToPILImage()(t.clamp(0, 1).float()))

    return output

def ImageToTensor(image):
    from torchvision.transforms import v2 as tt
    #return tt.ToTensor()(image)
    transform = tt.Compose([
        tt.ToImage(),
        tt.ToDtype(torch.float32, scale=True)
    ])
    if isinstance(image, list):
        return [transform(img) for img in image]
    return transform(image)

def get_memory_stats():
    try:
        if torch is None or not torch.cuda.is_available():
            return {}
        stats = torch.cuda.memory_stats()
        if not hasattr(stats, 'get'):
            return {}
    except Exception:
        return {}

    return {
        'current': stats.get('allocated_bytes.all.current', 0),
        'peak': stats.get('allocated_bytes.all.peak', 0),
        'allocated': stats.get('allocated_bytes.all.allocated', 0),
        'freed': stats.get('allocated_bytes.all.freed', 0),
    }

def reset_memory_stats():
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return
