"""Node message transport without executable node construction or ownership."""

from copy import deepcopy
from types import MethodType

from modiff.NodeBase import NodeBase


class FieldMessageContext:
    _queue_dynamic_node_message = NodeBase._queue_dynamic_node_message
    send_node_definition = NodeBase.send_node_definition
    set_field_visibility = NodeBase.set_field_visibility
    set_field_value = NodeBase.set_field_value
    set_field_params = NodeBase.set_field_params
    get_signal_value = NodeBase.get_signal_value

    def __init__(self, *, node_id, sid, class_name):
        self.node_id = node_id
        self._sid = sid
        self.class_name = class_name


def ordinary_field_callback(node_class, method, *, node_id, sid):
    # Some reviewed callbacks inspect their static field names through
    # self.__class__.params. Copy just that declaration, never the executable MRO.
    presentation = type("FieldMetadata", (FieldMessageContext,), {"params": deepcopy(node_class.params)})
    context = presentation(node_id=node_id, sid=sid, class_name=node_class.__name__)
    return MethodType(getattr(node_class, method), context)
