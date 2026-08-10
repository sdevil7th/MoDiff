# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import logging
logger = logging.getLogger('modiff')
from contextlib import contextmanager
from contextvars import ContextVar
from modiff.modelstore import modelstore
from utils.memory_menager import memory_manager
import numpy as np
import torch
import sys
import threading
import time

_DIFFUSERS_PROGRESS_PATCH_LOCK = threading.RLock()
_DIFFUSERS_PROGRESS_STACK = threading.local()
_NODE_MESSAGE_IDENTITY = ContextVar("modiff_node_message_identity", default=None)


@contextmanager
def node_message_context(identity=None):
    """Bind workflow ownership to dynamic node messages in this invocation.

    Field actions may run concurrently with the serialized graph worker. A
    context variable keeps their browser/workflow identity local to the
    executor invocation instead of reading an unrelated global current task.
    """

    normalized = dict(identity) if isinstance(identity, dict) else {}
    token = _NODE_MESSAGE_IDENTITY.set(normalized)
    try:
        yield
    finally:
        _NODE_MESSAGE_IDENTITY.reset(token)


def _node_message_identity():
    explicit = _NODE_MESSAGE_IDENTITY.get()
    if explicit is not None:
        return dict(explicit)

    current_server = _server()
    describe = getattr(current_server, "_current_dynamic_message_identity_payload", None)
    return dict(describe()) if callable(describe) else {}


def _loading_progress_stack():
    stack = getattr(_DIFFUSERS_PROGRESS_STACK, "value", None)
    if stack is None:
        stack = []
        _DIFFUSERS_PROGRESS_STACK.value = stack
    return stack


def _loading_item_label(item):
    if isinstance(item, tuple) and item and isinstance(item[0], str):
        return item[0]
    if isinstance(item, str):
        return item
    return None


class _StructuredLoadingProgress:
    """Proxy a Hugging Face tqdm bar into MoDiff's structured node progress."""

    def __init__(self, bar, report, description=None, total=None):
        self._bar = bar
        self._report = report
        self._description = str(description or getattr(bar, "desc", "") or "").strip().rstrip(".")
        self._total = total if isinstance(total, (int, float)) and total > 0 else getattr(bar, "total", None)
        self._manual_current = int(getattr(bar, "n", 0) or 0)
        self._last_report_at = 0.0
        self._last_report_progress = None

    def __getattr__(self, name):
        return getattr(self._bar, name)

    def _emit(self, current, *, item=None, starting=False):
        total = self._total
        if not isinstance(total, (int, float)) or total <= 0:
            return
        current = max(0, min(float(current), float(total)))
        stack = _loading_progress_stack()
        parent = stack[-1] if stack else None
        if parent and parent.get("total"):
            parent_total = float(parent["total"])
            parent_index = float(parent.get("index") or 1)
            ratio = ((parent_index - 1) + current / float(total)) / parent_total
        else:
            ratio = current / float(total)
        progress = min(99, max(0, int(round(ratio * 100))))
        step = max(0, min(int(current), int(total)))
        item_label = _loading_item_label(item)
        if starting and item_label and "component" in self._description.lower():
            component_scope = "pipeline" if "pipeline component" in self._description.lower() else "model"
            message = f"Loading {component_scope} component {int(current) + 1}/{int(total)}: {item_label}"
            step = min(int(total), int(current) + 1)
        else:
            label = self._description or "Loading"
            message = f"{label} {step}/{int(total)}"
        now = time.monotonic()
        if (
            not starting
            and step not in (0, int(total))
            and progress == self._last_report_progress
            and now - self._last_report_at < 0.25
        ):
            return
        description = self._description.lower()
        component = item_label if starting and item_label and "component" in description else None
        shard_current = step if "shard" in description else None
        shard_total = int(total) if "shard" in description else None
        try:
            try:
                self._report(
                    progress,
                    message,
                    step,
                    int(total),
                    component=component,
                    shard_current=shard_current,
                    shard_total=shard_total,
                )
            except TypeError as error:
                # Preserve the established four-argument extension callback
                # while MoDiff's reporter consumes the richer metadata.
                if "unexpected keyword argument" not in str(error):
                    raise
                self._report(progress, message, step, int(total))
            self._last_report_at = now
            self._last_report_progress = progress
        except Exception as error:
            logger.debug("Could not publish structured loader progress: %s", error)

    def __iter__(self):
        index = 0
        for item in self._bar:
            index += 1
            context = {"index": index, "total": self._total, "description": self._description}
            self._emit(index - 1, item=item, starting=True)
            stack = _loading_progress_stack()
            stack.append(context)
            completed = False
            try:
                yield item
                completed = True
            finally:
                if stack and stack[-1] is context:
                    stack.pop()
                elif context in stack:
                    stack.remove(context)
            if completed:
                self._emit(index, item=item)

    def __enter__(self):
        self._bar.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        result = self._bar.__exit__(exc_type, exc_value, traceback)
        if exc_type is None:
            bar_current = int(getattr(self._bar, "n", 0) or 0)
            self._manual_current = max(self._manual_current, bar_current)
            self._emit(self._manual_current)
        return result

    def close(self):
        result = self._bar.close()
        # Some Hugging Face loaders advance the underlying tqdm counter
        # directly and only finalize it while closing the bar. Publish that
        # terminal value instead of leaving the queue/node snapshot at N-1/N
        # throughout the following silent model-placement work.
        bar_current = int(getattr(self._bar, "n", 0) or 0)
        self._manual_current = max(self._manual_current, bar_current)
        self._emit(self._manual_current)
        return result

    def update(self, amount=1):
        previous = self._manual_current
        result = self._bar.update(amount)
        bar_current = int(getattr(self._bar, "n", 0) or 0)
        # Disabled tqdm bars intentionally leave ``n`` unchanged. Structured
        # progress must still advance when terminal rendering is disabled.
        self._manual_current = max(previous + int(amount or 0), bar_current)
        self._emit(self._manual_current)
        return result


def _server():
    from modiff.server import server
    return server

def _module_map():
    # Import lazily so importing NodeBase directly cannot leave the modules
    # package partially initialized through a circular dependency.
    from modules import MODULE_MAP
    return MODULE_MAP

def get_module_output(module_name, class_name):
    module_map = _module_map()
    params = module_map[module_name][class_name]['params'] if module_name in module_map and class_name in module_map[module_name] else {}
    return { p: None for p in params if 'display' in params[p] and params[p]['display'] == 'output' }

def get_default_params(module_name, class_name):
    module_map = _module_map()
    params = module_map[module_name][class_name]['params'] if module_name in module_map and class_name in module_map[module_name] else {}
    return { k: v for k, v in params.items() if not 'display' in v or v['display'] not in ['output', 'ui'] }

def deep_equal(a, b):
    # 1. Identity check
    if a is b:
        return True

    # 2. Type check
    if type(a) is not type(b):
        return False

    # 3. Specific type checks
    if isinstance(a, torch.Tensor):
        if a.device != b.device or a.dtype != b.dtype or a.shape != b.shape:
            return False
        return torch.equal(a, b)

    # For numpy arrays
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        return np.array_equal(a, b)

    # For PIL-like images. NumPy arrays also expose ``tobytes`` and ``size``,
    # so keep this check after the ndarray branch and require image metadata.
    if hasattr(a, 'mode') and hasattr(a, 'size') and hasattr(a, 'tobytes') and callable(a.tobytes):
        if a.size != b.size or a.mode != b.mode:
            return False
        return a.tobytes() == b.tobytes()

    # For lists and tuples, check length and then each element recursively
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y) for x, y in zip(a, b))

    # For sets, the standard equality check should be sufficient
    if isinstance(a, set):
        return a == b

    # For dictionaries, check keys and then each value recursively
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(deep_equal(a[key], b[key]) for key in a)

    # 4. Generic object checks
    if hasattr(a, 'to_dict') and callable(a.to_dict):
        return deep_equal(a.to_dict(), b.to_dict())

    # As a fallback, compare the objects' __dict__ attributes recursively
    if hasattr(a, '__dict__'):
        return deep_equal(a.__dict__, b.__dict__)

    # 5. Default case: use standard equality for primitives (int, str, etc.)
    return a == b

def recursive_type_cast(value, ttype, key):
    if value is None:
        return None

    if isinstance(value, list):
        return [recursive_type_cast(v, ttype, key) for v in value]
    if isinstance(value, dict):
        return {k: recursive_type_cast(v, ttype, f"{key}.{k}") for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(recursive_type_cast(list(value), ttype, key))
        #return tuple(recursive_type_cast(v, type, f"{key}.{i}") for i, v in enumerate(value))
    if isinstance(value, np.ndarray):
        return np.array(recursive_type_cast(value.tolist(), ttype, key), dtype=value.dtype)

    try:
        if ttype.startswith('int'):
            if isinstance(value, str) and value.strip() == '':
                return 0
            return int(float(value))  # Convert via float first to handle "1.0" -> 1
        if ttype.startswith('float'):
            if isinstance(value, str) and value.strip() == '':
                return 0.0
            return float(value)
        if ttype.startswith('str') or ttype.startswith('text'):
            return str(value or '')
        if ttype.startswith('bool'):
            if isinstance(value, str):
                # Handle string representations of booleans
                value_lower = value.lower().strip()
                if value_lower in ('true', '1', 'yes', 'on', 'y'):
                    return True
                if value_lower in ('false', '0', 'no', 'off', 'n'):
                    return False
                try:
                    return bool(float(value_lower))
                except ValueError:
                    return value
            return bool(value)
        return value
    except Exception:
        return value

class NodeBase:
    CALLBACK = 'execute'
    # Subclasses may list validated inputs that affect how a resident object is
    # used but not how it is constructed. Changes to these values should update
    # the node's current parameters without discarding expensive cached output,
    # while still invalidating results produced by connected descendants.
    cache_ignored_params = frozenset()

    def __init__(self, node_id=None):
        self.node_id = node_id
        # take everything before the last dot
        self.module_name = ".".join(self.__class__.__module__.split(".")[:-1])
        self.class_name = self.__class__.__name__

        self.params = {}
        self.default_params = get_default_params(self.module_name, self.class_name)
        self.output = get_module_output(self.module_name, self.class_name)

        self._sid = None
        self._has_changed = False
        self._cache_invalidated = False
        self._execution_time = { 'last': None, 'min': None, 'max': None }
        self._memory_usage = { 'last': None, 'min': None, 'max': None }
        self._mm_models = []
        self._interrupt = False
        self._progress_started_at = None
        self._progress_last_at = None
        self._skip_params_check = _module_map()[self.module_name][self.class_name].get('skipParamsCheck', False)

    def invalidate_cache(self):
        """Force the next graph invocation to execute this node.

        Connected nodes can receive the same mutable pipeline object across
        graph runs even when an upstream adapter node changed that object in
        place.  Object equality cannot represent that provenance, so the graph
        executor uses this one-shot invalidation signal when a source node
        actually re-executed.
        """
        self._cache_invalidated = True

    def _cache_params_equal(self, previous, current):
        """Compare cached inputs, allowing security-sensitive nodes to tighten equality."""

        return deep_equal(previous, current)

    def __call__(self, **kwargs):
        self._interrupt = False
        self._progress_started_at = None
        self._progress_last_at = None

        if self._skip_params_check:
            params = kwargs
        else:
            # filter out params that are not in the default_params
            params = { key: kwargs[key] for key in kwargs if key in self.default_params }

        # if node_id is None, the class was called directly, so we execute it without further processing
        if self.node_id is None:
            return getattr(self, self.CALLBACK)(**params)

        # params normalization and validation
        if not self._skip_params_check:
            for key, value in params.items():
                if value is None:
                    value = self.default_params[key]['default'] if 'default' in self.default_params[key] else None
                    params[key] = value

                if 'type' in self.default_params[key]:
                    type = self.default_params[key]['type']
                    if isinstance(type, list):
                        type = type[0]
                    params[key] = recursive_type_cast(value, type, key)

                if 'options' in self.default_params[key] and not self.default_params[key].get('fieldOptions', {}).get('noValidation', False):
                    options = self.default_params[key]['options']
                    value_list = [value] if not isinstance(value, list) else value
                    option_type = self.default_params[key].get('type')
                    if isinstance(option_type, list):
                        option_type = option_type[0] if option_type else None

                    def matches_option(candidate, option):
                        if deep_equal(candidate, option):
                            return True
                        if not isinstance(option_type, str):
                            return False
                        return deep_equal(candidate, recursive_type_cast(option, option_type, key))

                    if isinstance(options, list):
                        if any(not any(matches_option(v, option) for option in options) for v in value_list):
                            params[key] = []
                            #raise ValueError(f"Module {self.module_name}.{self.class_name}: Invalid value for {key}: {value} (options: {options})")
                    elif isinstance(options, dict):
                        if any(not any(matches_option(v, option) for option in options) for v in value_list):
                            params[key] = {}
                            #raise ValueError(f"Module {self.module_name}.{self.class_name}: Invalid value for {key}: {value} (options: {options})")
                    else:
                        raise ValueError(f"Module {self.module_name}.{self.class_name}: Invalid options format for {key}: {options}")

        # if we added a model not in the huggingface cache, we set a flag that
        # later will be used to tell the client to update its local cache
        update_hf_cache = False
        update_local_cache = False
        for key in self.default_params:
            is_modelselect = self.default_params[key].get('display') == 'modelselect'
            if is_modelselect:
                if isinstance(params[key], str):
                    sources = self.default_params[key].get('fieldOptions', { 'sources': ['hub'] }).get('sources', ['hub'])
                    params[key] = { 'source': sources[0], 'value': params[key] }

                if params[key].get('source') == 'hub':
                    if not modelstore.is_hf_cached(params[key].get('value')):
                        update_hf_cache = True
                elif params[key].get('source') == 'local':
                    if not modelstore.is_local_cached(params[key].get('value')):
                        update_local_cache = True

        # post processing
        for key in self.default_params:
            if 'postProcess' in self.default_params[key]:
                # we pass the current value and the dict of all the values for cross parameter validation
                params[key] = self.default_params[key]['postProcess'](params[key], params)

        self._has_changed = False # flag to know if the node has changed since the last execution

        ignored_cache_params = set(getattr(self, 'cache_ignored_params', ()) or ())
        previous_cache_params = {
            key: value for key, value in self.params.items() if key not in ignored_cache_params
        }
        current_cache_params = {
            key: value for key, value in params.items() if key not in ignored_cache_params
        }
        previous_ignored_params = {
            key: value for key, value in self.params.items() if key in ignored_cache_params
        }
        current_ignored_params = {
            key: value for key, value in params.items() if key in ignored_cache_params
        }
        ignored_params_changed = not deep_equal(previous_ignored_params, current_ignored_params)

        # If any load-relevant value changed, or output is empty, execute the
        # node. Validated passthrough inputs are still recorded below so
        # diagnostics reflect the current graph invocation.
        if (
            self._cache_invalidated
            or (not self._cache_params_equal(previous_cache_params, current_cache_params))
            or any(v is None for v in self.output.values())
        ):
            self._cache_invalidated = False
            self._has_changed = True
            self.params = params
            self.output = {k: None for k in self.output}
            del params
            if self._mm_models:
                for model_id in self._mm_models:
                    memory_manager.remove(model_id)
                self._mm_models = []

            try:
                output = getattr(self, self.CALLBACK)(**self.params)
            except Exception as e:
                self.params = {}
                #self.output = {k: None for k in self.output}
                raise RuntimeError(f"Error executing {self.module_name}.{self.class_name}: {e}") from e

            if output and isinstance(output, dict):
                # output and self.output keys must be the same
                if set(output.keys()) != set(self.output.keys()) and not self._skip_params_check:
                    raise ValueError(f"Module {self.module_name}.{self.class_name}: Output keys do not match: {output.keys()} != {self.output.keys()}")

                self.output = output
            # elif output is not None:
            #     if len(self.output) > 1:
            #         raise ValueError(f"Module {self.module_name}.{self.class_name}: Only one output returned, but multiple are expected ({self.output.keys()})")
            #     # if only one output is returned, assign it to the first output
            #     self.output[next(iter(self.output))] = output

            # inform the client that the huggingface cache needs to be updated
            if update_hf_cache:
                modelstore.update_hf()
                _server().queue_message({
                    "type": "hf_cache_update",
                    "node": self.node_id,
                }, self._sid)
            if update_local_cache:
                modelstore.update_local()
                _server().queue_message({
                    "type": "local_cache_update",
                    "node": self.node_id,
                }, self._sid)
        else:
            # A cache-ignored value can reconfigure the same resident output
            # without repeating its expensive construction.  Preserve that
            # cache hit, but publish the semantic change so connected nodes do
            # not reuse results computed under the previous contract.
            self._has_changed = ignored_params_changed
            self.params = params

        return self.output

    def __del__(self):
        try:
            if sys.meta_path is None:
                return  # Python is shutting down, skip cleanup

            for model_id in self._mm_models:
                memory_manager.remove(model_id)
        except (ImportError, AttributeError, RuntimeError):
            # Python is shutting down or import system is unavailable
            pass

        # Partially constructed nodes can reach ``__del__`` when their
        # constructor raises (for example, a guarded optional model loader).
        # Cleanup must never emit a secondary exception that hides the useful
        # construction error.
        self.__dict__.pop("params", None)
        self.__dict__.pop("output", None)

    def pipe_callback(self, pipe, step_index, timestep, callback_kwargs):
        if not self.node_id:
            return callback_kwargs

        if self._interrupt:
            pipe._interrupt = True
            # Some Diffusers loops inspect `_interrupt` before invoking the
            # callback, which can start another multi-minute step. Raising at
            # this completed-step boundary gives the worker an immediate,
            # cleanly classified interruption and preserves normal cleanup.
            raise InterruptedError("Execution interrupted by the user after the current model step.")

        current_task = _server().current_task or {}
        runtime_limit = (current_task.get('runtimeHints') or {}).get('maxRuntimeSeconds')
        started_at = current_task.get('started_at')
        if runtime_limit and started_at and time.time() - float(started_at) >= float(runtime_limit):
            pipe._interrupt = True
            raise TimeoutError(
                f"Execution reached the configured {int(runtime_limit)} second runtime limit after the current model step."
            )

        if hasattr(pipe, '_cfg_cutoff_step') and pipe._cfg_cutoff_step is not None:
            cutoff_step = int(pipe._num_timesteps * pipe._cfg_cutoff_step)
            if step_index == cutoff_step:
                pipe._guidance_scale = 0.0
                if 'prompt_embeds' in callback_kwargs:
                    callback_kwargs['prompt_embeds'] = callback_kwargs['prompt_embeds'][-1:]
                if 'pooled_prompt_embeds' in callback_kwargs:
                    callback_kwargs['pooled_prompt_embeds'] = callback_kwargs['pooled_prompt_embeds'][-1:]

        now = time.time()
        if self._progress_started_at is None:
            self._progress_started_at = now
        elapsed = max(0.0, now - self._progress_started_at)
        completed_steps = step_index + 1
        total_steps = int(pipe._num_timesteps)
        # The timer starts at the first completed-step callback, so at step 0
        # there is not yet a measured interval. From step 1 onward, divide by
        # the number of intervals since that boundary (step_index), not by the
        # total completed-step count. Dividing by completed_steps made the
        # first useful long-video ETA exactly half of the observed runtime.
        measured_intervals = step_index
        average_step_seconds = elapsed / measured_intervals if measured_intervals > 0 else None
        eta_seconds = average_step_seconds * max(0, total_steps - completed_steps) if average_step_seconds is not None else None
        self._progress_last_at = now
        progress = int(completed_steps / total_steps * 100)
        self.progress(
            progress,
            phase="denoising",
            message=f"Denoising {completed_steps}/{total_steps}",
            current_step=completed_steps,
            total_steps=total_steps,
            elapsed_seconds=elapsed,
            average_step_seconds=average_step_seconds,
            eta_seconds=eta_seconds,
        )

        return callback_kwargs

    def trigger_output(self, output, value=None):
        if not self.node_id or output not in self.output:
            return

        if value is not None:
            self.output[output] = value

        _server().trigger_node(self.node_id, output, self._sid)


    """
    ╭───────────╮
      WebSocket
    ╰───────────╯
    """

    def ws_message(self, message):
        if not self._sid:
            return

        _server().queue_message(message, self._sid)

    @contextmanager
    def diffusers_loading_progress(self):
        """Publish Diffusers/Transformers loading as normal node progress.

        Pipeline loading exposes these stages only through the libraries' tqdm
        facades. Patch both facades for one loader call, preserving the terminal
        bars while forwarding nested component, checkpoint-shard, and weight
        counts to the queue, websocket, graph node, and activity notification.
        """

        if not self.node_id:
            yield
            return
        try:
            from diffusers.utils import logging as diffusers_logging
        except Exception:
            yield
            return
        progress_facades = [diffusers_logging]
        try:
            from transformers.utils import logging as transformers_logging

            if transformers_logging is not diffusers_logging:
                progress_facades.append(transformers_logging)
            # Transformers 5 copies the tqdm function into this module at
            # import time, so patching only the logging facade does not reach
            # its per-weight loader bar.
            from transformers import core_model_loading

            progress_facades.append(core_model_loading)
        except Exception:
            pass
        progress_facades = [
            facade
            for index, facade in enumerate(progress_facades)
            if callable(getattr(facade, "tqdm", None)) and facade not in progress_facades[:index]
        ]

        with _DIFFUSERS_PROGRESS_PATCH_LOCK:
            owner_thread_id = threading.get_ident()
            originals = [(facade, facade.tqdm) for facade in progress_facades]
            last_reported_progress = -1

            def report(
                progress,
                message,
                current,
                total,
                *,
                component=None,
                shard_current=None,
                shard_total=None,
            ):
                nonlocal last_reported_progress
                # A component can expose more than one sequential nested bar
                # (for example checkpoint shards followed by Transformers
                # weights). Never make the node/notification bar move backward.
                progress = max(last_reported_progress, progress)
                last_reported_progress = progress
                normalized_message = str(message or "").lower()
                if shard_total is not None or "shard" in normalized_message or "weight" in normalized_message:
                    phase = "shard_loading"
                elif component is not None or "component" in normalized_message:
                    phase = "component_loading"
                else:
                    phase = "loading"
                self.progress(
                    progress,
                    phase=phase,
                    message=message,
                    current_step=current,
                    total_steps=total,
                    component=component,
                    shard_current=shard_current,
                    shard_total=shard_total,
                )

            def structured_tqdm_factory(original_tqdm):
                def structured_tqdm(*args, **kwargs):
                    bar = original_tqdm(*args, **kwargs)
                    # These facades are module-global. A concurrent model
                    # download on another thread must retain its own progress
                    # channel rather than being attributed to this graph node.
                    if threading.get_ident() != owner_thread_id:
                        return bar
                    return _StructuredLoadingProgress(
                        bar,
                        report,
                        description=kwargs.get("desc"),
                        total=kwargs.get("total"),
                    )

                return structured_tqdm

            for facade, original_tqdm in originals:
                facade.tqdm = structured_tqdm_factory(original_tqdm)
            try:
                yield
            finally:
                for facade, original_tqdm in reversed(originals):
                    facade.tqdm = original_tqdm

    def progress(
        self,
        progress: int,
        phase: str = "unknown",
        message: str | None = None,
        status: str = "running",
        current_step: int | None = None,
        total_steps: int | None = None,
        elapsed_seconds: float | None = None,
        average_step_seconds: float | None = None,
        eta_seconds: float | None = None,
        component: str | None = None,
        shard_current: int | None = None,
        shard_total: int | None = None,
    ):
        if not self._sid or not self.node_id:
            return

        current_task = getattr(_server(), "current_task", None)
        task_id = current_task.get("task_id") if current_task else None
        attempt_index = current_task.get("attempt_index") if current_task else None
        runtime_hints = current_task.get("runtimeHints") if current_task else None
        payload = {
            "type": "progress",
            "node": self.node_id,
            "progress": progress,
            "task_id": task_id,
            "attempt_index": attempt_index,
            "status": status,
            "phase": phase,
        }
        if isinstance(runtime_hints, dict):
            if runtime_hints.get("clientRunId"):
                payload["client_run_id"] = runtime_hints.get("clientRunId")
            if runtime_hints.get("runInputHash"):
                payload["run_input_hash"] = runtime_hints.get("runInputHash")
        if message:
            payload["message"] = message
        if current_step is not None:
            payload["current_step"] = current_step
        if total_steps is not None:
            payload["total_steps"] = total_steps
        if elapsed_seconds is not None:
            payload["elapsed_seconds"] = elapsed_seconds
        if average_step_seconds is not None:
            payload["average_step_seconds"] = average_step_seconds
        if eta_seconds is not None:
            payload["eta_seconds"] = eta_seconds
        if component:
            payload["component"] = component
        if shard_current is not None:
            payload["shard_current"] = shard_current
        if shard_total is not None:
            payload["shard_total"] = shard_total
        payload["last_heartbeat_at"] = time.time()

        payload = _server().record_node_progress(payload)
        _server().queue_message(payload)

    def _queue_dynamic_node_message(self, message):
        if not self._sid or not self.node_id:
            return

        identity = _node_message_identity()
        target_sid = identity.get("sid") or self._sid
        payload = {
            **message,
            **identity,
            "sid": target_sid,
        }
        _server().queue_message(payload, target_sid)

    def send_node_definition(self, params):
        if not self._sid or not self.node_id:
            return

        current_server = _server()
        describe = getattr(current_server, "describe_node_params", None)
        public_params = describe(params) if callable(describe) else params
        self._queue_dynamic_node_message({
            "type": "node_definition",
            "node": self.node_id,
            "params": public_params,
        })

    def set_field_visibility(self, fields: dict):
        if not self._sid or not self.node_id:
            return

        self._queue_dynamic_node_message({
            "type": "set_field_visibility",
            "node": self.node_id,
            "fields": fields,
        })

    def set_field_value(self, field: dict):
        if not self._sid or not self.node_id:
            return

        self._queue_dynamic_node_message({
            "type": "set_field_value",
            "node": self.node_id,
            "fields": field,
        })

    def set_field_params(self, field: str, params: dict):
        if not self._sid or not self.node_id:
            return

        self._queue_dynamic_node_message({
            "type": "set_field_params",
            "node": self.node_id,
            "field": field,
            "params": params,
        })

    def get_signal_value(self, field: str, timeout: int = 5):
        if not self._sid or not self.node_id:
            return None

        server = _server()
        if not self._sid in server.ws_sessions:
            raise ValueError("WebSocket session not available for this node.")

        result = server.get_signal_value(self.node_id, field, self._sid, timeout=timeout)
        if isinstance(result, dict) and '__MODIFF_ERROR' in result:
            raise ValueError(result['__MODIFF_ERROR'])

        return result

    def notify(self, message: str, variant: str = 'default', persist: bool = False, autoHideDuration: int = 0):
        if not self._sid or not self.node_id:
            return

        _server().queue_message({
            "type": "notification",
            "node": self.node_id,
            "message": message,
            "variant": variant,
            "persist": persist,
            "autoHideDuration": autoHideDuration,
        })


    """
    ╭────────────────╮
      Memory Manager
    ╰────────────────╯
    """

    def mm_add(self, model, priority=1):
        if self.node_id is None:
            return model

        model_id = memory_manager.add(model, priority)

        if model_id not in self._mm_models:
            self._mm_models.append(model_id)

        return model_id

    def mm_remove(self, model):
        if self.node_id is None:
            return model.to('cpu')

        model_id = memory_manager.remove(model)
        if model_id is not None:
            self._mm_models.remove(model_id)

        return model_id

    def mm_get(self, model):
        if self.node_id is None:
            return model

        return memory_manager.get_model(model)

    def mm_update(self, model, **kwargs):
        if self.node_id is None:
            return

        return memory_manager.update(model, **kwargs)

    def mm_load(self, model, device=None):
        if self.node_id is None:
            return model.to(device)

        return memory_manager.load_model(model, device)

    def mm_exec(self, func, device, models=None, exclude=None, args=None, kwargs=None):
        if self.node_id is None:
            return func(*(args or ()), **(kwargs or {}))

        return memory_manager.exec(func, device, models, exclude, args, kwargs)

    def mm_unload_all(self, device=None):
        if self.node_id is None:
            return

        memory_manager.unload_all(device)
