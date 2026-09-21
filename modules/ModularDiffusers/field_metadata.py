"""Presentation-only contexts for the reviewed generic Modular field actions.

Reuse the existing callback implementations and NodeBase message protocol without
constructing an executable node. In particular, ModelsLoader's destructor removes
components by node ID even when that particular instance never loaded weights.
This context has no model cache, component collection, executor or destructor.
Keep its explicitly borrowed methods under review when changing those callbacks.
"""

import threading
from types import MethodType

from modiff.NodeBase import NodeBase
from .denoise import Denoise
from .embeddings import EncodePrompt, ImageEmbeddings
from .latents import DecodeLatents, ImageEncode
from .loaders import ModelsLoader


class _FieldContext:
    _queue_dynamic_node_message = NodeBase._queue_dynamic_node_message
    send_node_definition = NodeBase.send_node_definition
    set_field_visibility = NodeBase.set_field_visibility
    set_field_value = NodeBase.set_field_value
    set_field_params = NodeBase.set_field_params
    get_signal_value = NodeBase.get_signal_value

    # These helpers read declarations and publish UI messages only. Do not bind
    # the loader constructor, execution, cache or component-release methods here.
    _begin_pipeline_identity_refresh = ModelsLoader._begin_pipeline_identity_refresh
    _publish_pipeline_identity = ModelsLoader._publish_pipeline_identity
    _selected_repository = staticmethod(ModelsLoader._selected_repository)
    _reviewed_workflow_variants = staticmethod(ModelsLoader._reviewed_workflow_variants)
    refresh_pipeline_identity = ModelsLoader.refresh_pipeline_identity

    def __init__(self, *, node_id, sid, node_type):
        self.node_id = node_id
        self._sid = sid
        self.node_type = node_type
        self._model_type = ""
        self._pipeline_class = None
        self.model_types_loaded = False
        self._pipeline_identity_generation = 0
        self._pipeline_identity_lock = threading.Lock()


def metadata_field_callback(action, method, *, node_id, sid):
    node_class = {
        "ModelsLoader": ModelsLoader,
        "EncodePrompt": EncodePrompt,
        "Denoise": Denoise,
        "DecodeLatents": DecodeLatents,
        "ImageEncode": ImageEncode,
        "ImageEmbeddings": ImageEmbeddings,
    }[action]
    context = _FieldContext(node_id=node_id, sid=sid, node_type=getattr(node_class, "node_type", None))
    return MethodType(getattr(node_class, method), context)
