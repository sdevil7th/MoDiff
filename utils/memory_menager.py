# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import logging
logger = logging.getLogger('modiff')
import torch
import gc
import time
import nanoid
import os
from utils.torch_utils import DEFAULT_DEVICE

GIB = 1024 ** 3


def _tensor_bytes(tensor):
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def _model_size_bytes(model):
    """Count unique parameter/buffer storage across modules or pipelines."""
    seen_tensors = set()
    seen_modules = set()
    total = 0
    candidates = [model]
    components = getattr(model, 'components', None)
    if isinstance(components, dict):
        candidates.extend(components.values())
    for candidate in candidates:
        if not isinstance(candidate, torch.nn.Module) or id(candidate) in seen_modules:
            continue
        seen_modules.add(id(candidate))
        for tensor in [*candidate.parameters(recurse=True), *candidate.buffers(recurse=True)]:
            identity = id(tensor)
            if identity not in seen_tensors:
                seen_tensors.add(identity)
                total += _tensor_bytes(tensor)
    return total


def _model_device(model):
    direct = getattr(model, 'device', None)
    if direct is not None:
        return str(direct)
    candidates = [model]
    components = getattr(model, 'components', None)
    if isinstance(components, dict):
        candidates.extend(components.values())
    for candidate in candidates:
        if isinstance(candidate, torch.nn.Module):
            try:
                return str(next(candidate.parameters()).device)
            except (StopIteration, AttributeError):
                continue
    return 'cpu'


def _accelerator_reserve_bytes(total_bytes):
    os_reserve = (600 if os.name == 'nt' else 400) * 1024 ** 2
    if os.name == 'nt' and total_bytes > 15 * GIB:
        os_reserve += 100 * 1024 ** 2
    return int(0.8 * GIB) + os_reserve

def memory_flush():
    gc.collect()

    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        logger.debug("Failed to query CUDA availability during memory flush", exc_info=True)
        cuda_available = False

    if cuda_available:
        # CUDA can keep an error pending after OOM. Cleanup should never make
        # the caller fail just because a cache flush observes that stale state.
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.debug("Failed to empty CUDA cache", exc_info=True)
        try:
            torch.cuda.ipc_collect()
        except Exception:
            logger.debug("Failed to collect CUDA IPC cache", exc_info=True)

    mps = getattr(torch, 'mps', None)
    try:
        mps_available = bool(mps and mps.is_available())
    except Exception:
        logger.debug("Failed to query MPS availability during memory flush", exc_info=True)
        mps_available = False

    if mps_available:
        try:
            torch.mps.empty_cache()
        except Exception:
            logger.debug("Failed to empty MPS cache", exc_info=True)

class MemoryManager:
    def __init__(self):
        self.cache = {}
        self.policy = str(os.environ.get('MODIFF_MODEL_CACHE_POLICY') or 'lru').strip().lower()
        if self.policy not in {'lru', 'no_cache', 'high_ram'}:
            self.policy = 'lru'

    def add(self, model, priority=1) -> str:
        if hasattr(model, '_mm_id'):
            model_id = model._mm_id
            if model_id in self.cache:
                self.update(model_id, model=model, priority=priority)
                return model_id

        model_id = nanoid.generate()
        model._mm_id = model_id
        self.cache[model_id] = {
            'model': model,
            'priority': priority,
            'last_used': time.time(),
            'size': _model_size_bytes(model),
        }
        self._evict_system_ram_pressure(exclude_ids=[model_id])
        return model_id

    def remove(self, model):
        model_id = model if isinstance(model, str) else model._mm_id if hasattr(model, '_mm_id') else None
        if model_id is None or model_id not in self.cache:
            return None

        try:
            self.cache[model_id]['model'] = self.cache[model_id]['model'].to('cpu')
        except Exception:
            # should prevent errors with quantized models
            pass

        self.cache[model_id]['model'] = None
        del self.cache[model_id]
        memory_flush()
        return model_id

    def update(self, model_id, model=None, priority=None):
        model_id = model_id if isinstance(model_id, str) else model_id._mm_id if hasattr(model_id, '_mm_id') else None
        if model_id is None or model_id not in self.cache:
            return None

        if model is not None:
            self.cache[model_id]['model'] = model
            self.cache[model_id]['size'] = _model_size_bytes(model)
        if priority is not None:
            self.cache[model_id]['priority'] = priority

        self.cache[model_id]['last_used'] = time.time()
        memory_flush()
        return model_id

    def get_model(self, model_id):
        if model_id not in self.cache:
            return None

        return self.cache[model_id]['model']

    def load_model(self, model, device, exclude=None):
        model_id = model if isinstance(model, str) else model._mm_id if hasattr(model, '_mm_id') else None
        if model_id is None or model_id not in self.cache:
            return None

        exclude_ids = []
        for v in (exclude or []):
            k = v if isinstance(v, str) else v._mm_id if hasattr(v, '_mm_id') else None
            if k and k in self.cache:
                exclude_ids.append(k)
        exclude_ids.append(model_id)

        self.cache[model_id]['last_used'] = time.time()
        x = self.cache[model_id]['model']

        if _model_device(x) == str(device):
            return x

        cache_priority = self._get_unload_candidates(device, exclude_ids)

        memory_flush()

        # Proactively evict by actual parameter bytes before relying on an OOM.
        if str(device).startswith('cuda'):
            try:
                index = torch.device(device).index or 0
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                required = int(self.cache[model_id].get('size') or 0) + _accelerator_reserve_bytes(total_bytes)
                while required > free_bytes and cache_priority:
                    candidate_id = cache_priority.pop(0)[2]
                    logger.debug(
                        "Evicting model %s before load: %d bytes free, %d bytes required",
                        candidate_id,
                        free_bytes,
                        required,
                    )
                    self.unload_model(candidate_id)
                    free_bytes, _ = torch.cuda.mem_get_info(index)
            except Exception:
                logger.debug("Could not apply proactive accelerator eviction", exc_info=True)

        while True:
            try:
                #memory_current = torch.cuda.mem_get_info()[0] if 'cuda' in device else 0

                x = x.to(device)
                #self.cache[model_id]['model'] = x
                #self.cache[model_id]['device'] = device
                #if self.cache[model_id]['size'] == 0 and 'cuda' in device:
                #    self.cache[model_id]['size'] = memory_current - torch.cuda.mem_get_info()[0]
                return x
            except torch.OutOfMemoryError as e:
                if not cache_priority:
                    logger.warning(f"Cannot free enough memory to load model {model_id}")
                    raise e

                k = cache_priority.pop(0)[2]
                logger.debug(f"OOM. Trying to unload lower priority model: {k}")
                self.unload_model(k)
            except Exception as e:
                logger.error(f"Error loading model {model_id}")
                raise e

    def unload_model(self, model):
        model_id = model if isinstance(model, str) else model._mm_id if hasattr(model, '_mm_id') else None
        if model_id is None or model_id not in self.cache:
            return None

        unloaded = self.cache[model_id]['model'].to('cpu')
        self.cache[model_id]['model'] = unloaded
        memory_flush()

        return self.cache[model_id]['model']

    def unload_all(self, device=None):
        device = device if device else DEFAULT_DEVICE
        for k, v in self.cache.items():
            if _model_device(v['model']) == str(device):
                self.unload_model(k)

    def clear(self):
        cleared = len(self.cache)
        # Cleanup is discarding these objects, not keeping CPU-resident copies.
        # Calling pipeline.to('cpu') here can transiently materialize an entire
        # Accelerate-offloaded model in system RAM while its GPU allocation is
        # still live. On unified-memory ROCm hosts that peak can OOM the server
        # during a model-family switch. Drop every manager-owned reference
        # atomically, let object destruction release hooks/storage, then flush
        # allocator caches once.
        records = list(self.cache.values())
        self.cache.clear()
        for record in records:
            record['model'] = None
        del records
        memory_flush()
        return cleared

    def exec(self, func, device, models=None, exclude=None, args=None, kwargs=None, inference_mode=True):
        exclude_ids = []
        for v in (exclude or []):
            k = v if isinstance(v, str) else v._mm_id if hasattr(v, '_mm_id') else None
            if k and k in self.cache:
                exclude_ids.append(k)

        # auto load the models, add them to the exclude list
        active_models = list(models or [])
        for v in active_models:
            k = v if isinstance(v, str) else v._mm_id if hasattr(v, '_mm_id') else None
            if k and k in self.cache:
                exclude_ids.append(k)
                self.load_model(v, device)

        # Get a list of all models on the target device that can be unloaded.
        cache_priority = self._get_unload_candidates(device, exclude_ids=exclude_ids)

        args = args or []
        kwargs = kwargs or {}

        try:
            while True:
                try:
                    if inference_mode:
                        with torch.inference_mode():
                            return func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except torch.cuda.OutOfMemoryError as e:
                    # If we're out of memory, we need to unload a model.
                    if not cache_priority:
                        # If there are no more models to unload, we have failed.
                        logger.error("OOM during exec. No models left to unload to free memory.")
                        raise e

                    # Unload the lowest-priority model.
                    k = cache_priority.pop(0)[2]
                    logger.debug(f"OOM during exec. Unloading model '{k}' to free VRAM.")
                    self.unload_model(k)
                except Exception as e:
                    logger.error(f"An unexpected error occurred during exec: {e}")
                    raise e
        finally:
            if self.policy == 'no_cache':
                for active in active_models:
                    self.unload_model(active)
            self._evict_system_ram_pressure(exclude_ids=exclude_ids)

    def _get_unload_candidates(self, device, exclude_ids=None):
        """
        Gets a sorted list of models that are candidates for unloading from a device.
        The list is sorted by priority and last-used time, so the first element
        is the best candidate for unloading.
        """
        cache_priority = []
        for k, v in self.cache.items():
            # Check if the model is on the target device and not in the exclude list
            if _model_device(v['model']) == str(device) and k not in (exclude_ids or []):
                cache_priority.append((v['priority'], v['last_used'], k))

        # Sort by priority then last_used to find the best unload candidate
        cache_priority.sort(key=lambda x: (x[0], x[1]))
        return cache_priority

    def _evict_system_ram_pressure(self, exclude_ids=None):
        if self.policy == 'high_ram':
            return []
        try:
            from modiff.hardware import system_memory_snapshot

            memory = system_memory_snapshot()
            available = memory.get('available_bytes')
            total = memory.get('total_bytes')
            floor = max(4 * GIB, int(total * 0.1)) if isinstance(total, int) else 4 * GIB
            if not isinstance(available, int) or available >= floor:
                return []
        except Exception:
            return []
        excluded = set(exclude_ids or [])
        candidates = sorted(
            (
                (record['priority'], record['last_used'], model_id)
                for model_id, record in self.cache.items()
                if model_id not in excluded and _model_device(record['model']).startswith('cpu')
            ),
            key=lambda item: (item[0], item[1]),
        )
        evicted = []
        while candidates and available < floor:
            model_id = candidates.pop(0)[2]
            self.remove(model_id)
            evicted.append(model_id)
            memory = system_memory_snapshot()
            available = int(memory.get('available_bytes') or 0)
        return evicted

memory_manager = MemoryManager()
