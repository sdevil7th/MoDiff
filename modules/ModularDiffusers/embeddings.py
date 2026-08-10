# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import logging

from diffusers import ComponentSpec

from modiff.NodeBase import NodeBase

from . import MESSAGE_DURATION, components
from .modular_utils import (
    normalize_modular_runtime_params,
    pipeline_class_from_model_type,
    pipeline_class_from_runtime_inputs,
    require_modiff_node_contract,
    reject_undeclared_modular_generator,
)
from .route_state import (
    ROUTE_STATE_OUTPUT,
    issue_wan_image_encoder_route_state,
    preflight_wan_image_encoder_inputs,
    reject_route_reserved_inputs_before_identity_resolution,
    resolve_managed_component_by_id,
    require_cataloged_wan_action_source,
    require_component_binding,
    require_route_state_shape_before_identity_resolution,
    route_contract_for_model_type,
    snapshot_wan_source_media,
    validate_route_field_contract,
    validate_wan_image_encoder_route_state,
    wan_image_encoder_contract_from_component,
    wan_image_processor_config_seal,
)
from .utils import collect_model_ids


logger = logging.getLogger("modiff")

PROMPT_EMBEDDING_FIELDS = [
    "prompt_embeds",
    "prompt_embeds_mask",
    "negative_prompt_embeds",
    "negative_prompt_embeds_mask",
]

MISSING_EMBEDDINGS_MESSAGE = (
    "Prompt embeddings are missing from Encode Prompt. "
    "Update or recreate the Studio graph after the model fields finish refreshing."
)


def _state_get(state, key, default=None):
    if hasattr(state, "get"):
        return state.get(key, default)
    values = getattr(state, "values", None)
    if isinstance(values, dict):
        return values.get(key, default)
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def extract_prompt_embeddings(state):
    embeddings = {}
    if hasattr(state, "get_by_kwargs"):
        try:
            embeddings = state.get_by_kwargs("denoiser_input_fields") or {}
        except KeyError:
            embeddings = {}

    if embeddings:
        return embeddings

    fallback = {field: _state_get(state, field) for field in PROMPT_EMBEDDING_FIELDS}
    fallback = {field: value for field, value in fallback.items() if value is not None}
    if fallback:
        return fallback

    raise KeyError("embeddings")


class EncodePrompt(NodeBase):
    label = "Encode Prompt"
    category = "embedding"
    resizable = True
    skipParamsCheck = True
    node_type = "text_encoder"
    params = {
        "text_encoders": {
            "label": "Text Encoders *",
            "type": "diffusers_auto_models",
            "display": "input",
            "onSignal": "update_node",
        },
    }

    def update_node(self, values, ref):
        node_params = {}
        model_type = self.get_signal_value("text_encoders")

        if self._model_type == model_type:
            return None

        if model_type is None or model_type == "":
            self._model_type = ""
            self._pipeline_class = None
            self.send_node_definition(node_params)
            return None
        try:
            self._pipeline_class = pipeline_class_from_model_type(model_type)
            _, node_config = require_modiff_node_contract(
                self._pipeline_class,
                self.node_type,
                resolve_blocks=False,
            )
        except ValueError:
            self._model_type = ""
            self._pipeline_class = None
            self.send_node_definition(node_params)
            raise
        self._model_type = model_type

        node_params_to_update = node_config["params"]
        node_params_to_update.pop("text_encoders", None)

        node_params.update(**node_params_to_update)
        # The client merges this refreshed definition with the current field values.
        self.send_node_definition(node_params)

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._model_type = ""
        self._pipeline_class = None

    def execute(self, **kwargs):
        kwargs = dict(kwargs)
        self._pipeline_class = pipeline_class_from_runtime_inputs(self._pipeline_class, kwargs)
        # 1. Get node config
        blocks, node_config = require_modiff_node_contract(self._pipeline_class, self.node_type)

        # 2. create pipeline
        repo_id = None
        if (te := kwargs.get("text_encoders")) and "repo_id" in te:
            repo_id = te["repo_id"]

        if repo_id is None:
            self.notify(
                "You have to connect the text encoder(s)",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            return None

        # Enforce the backend-issued action schema before initializing blocks.
        kwargs = normalize_modular_runtime_params(kwargs, node_config)

        # Components came from the reviewed ModelsLoader contract. Re-reading
        # repository config here would let a later cache mutation choose fresh
        # component type hints outside that validation boundary.
        self._pipeline = blocks.init_pipeline(components_manager=components)

        # 3. update components
        expected_component_names = blocks.component_names
        model_input_names = node_config["model_input_names"]
        model_ids = collect_model_ids(
            kwargs, target_key_names=model_input_names, target_model_names=expected_component_names
        )

        if model_ids:
            components_to_update = components.get_components_by_ids(ids=model_ids, return_dict_with_names=True)
            if components_to_update:
                self._pipeline.update_components(**components_to_update)

        # 4. compile a dict of runtime inputs from kwargs based on node_config["input_names"]
        node_kwargs = {}
        input_names = node_config["input_names"]
        for name in input_names:
            if name not in kwargs:
                continue
            value = kwargs.get(name)

            # if a dict is passed and is not an pipeline input, we unpack and process its contents
            # e.g. `embeddings` from text_encoder node
            if isinstance(value, dict) and name not in blocks.input_names:
                for k, v in value.items():
                    if k in blocks.input_names:
                        node_kwargs[k] = v
                    else:
                        expected_inputs = "\n  - ".join(blocks.input_names)
                        logger.warning(
                            f"Input '{name}:{k}' is not expected by {self.node_type} blocks.\n"
                            f"Expected inputs:\n  - {expected_inputs} \n"
                            f"Blocks: {blocks}"
                        )
            # pass the value as it is to the pipeline
            elif name in blocks.input_names:
                node_kwargs[name] = value
            else:
                expected_inputs = "\n  - ".join(blocks.input_names)
                logger.warning(
                    f"Input '{name}' is not expected by {self.node_type} blocks.\n"
                    f"Expected inputs:\n  - {expected_inputs} \n"
                    f"Blocks: {blocks}"
                )

        # 5. run the pipeline,
        try:
            node_output_state = self._pipeline(**node_kwargs)
        except ValueError as e:
            self.notify(str(e), variant="error", persist=False, autoHideDuration=MESSAGE_DURATION)
            return None

        # 6. prepare the outputs dict based on node_config["output_names"]
        output_names = node_config["output_names"].copy()
        outputs = {}
        for name in output_names:
            if name == "doc":
                outputs["doc"] = self._pipeline.blocks.doc
            elif name == "embeddings":
                try:
                    outputs["embeddings"] = extract_prompt_embeddings(node_output_state)
                except KeyError:
                    self.notify(
                        MISSING_EMBEDDINGS_MESSAGE,
                        variant="error",
                        persist=False,
                        autoHideDuration=MESSAGE_DURATION,
                    )
                    return None
            else:
                outputs[name] = node_output_state.get(name)
        return outputs


class ImageEmbeddings(NodeBase):
    label = "Image Embeddings"
    category = "embedding"
    resizable = True
    skipParamsCheck = True
    node_type = "image_encoder"
    params = {
        "image_encoder": {
            "label": "Image Encoder *",
            "display": "input",
            "type": "diffusers_auto_model",
            "onSignal": "update_node",
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._model_type = ""
        self._pipeline_class = None

    def _cache_params_equal(self, previous, current):
        equal = super()._cache_params_equal(previous, current)
        if not equal or not isinstance(current, dict) or self._pipeline_class is None:
            return equal
        model_type = getattr(self._pipeline_class, "__name__", "")
        if route_contract_for_model_type(model_type) != "wan_i2v":
            return True
        require_cataloged_wan_action_source(
            image=current.get("image"),
            last_image=current.get("last_image"),
        )
        _blocks, node_config = require_modiff_node_contract(
            self._pipeline_class,
            self.node_type,
            resolve_blocks=False,
        )
        current = normalize_modular_runtime_params(dict(current), node_config)
        route_state = self.output.get(ROUTE_STATE_OUTPUT)
        if route_state is None or getattr(self, "_pipeline", None) is None:
            return False
        binding = require_component_binding(
            current.get("image_encoder"),
            label="image encoder",
            expected_model_type=model_type,
            expected_role="image_encoder",
        )
        resident_image_encoder = resolve_managed_component_by_id(
            components,
            current.get("image_encoder"),
            label="Image Embeddings image encoder",
        )
        if getattr(self._pipeline, "image_encoder", None) is not resident_image_encoder:
            raise ValueError("The resident Wan image pipeline does not hold the exact connected image encoder.")
        validate_wan_image_encoder_route_state(
            route_state,
            binding=binding,
            model_type=model_type,
            image=current.get("image"),
            last_image=current.get("last_image"),
            height=current.get("height"),
            width=current.get("width"),
            image_embeds=self.output.get("image_embeds"),
            image_encoder=resident_image_encoder,
            image_processor=getattr(self._pipeline, "image_processor", None),
            execution_device=getattr(self._pipeline, "_execution_device", None),
        )
        return True

    def update_node(self, values, ref):
        node_params = {}
        model_type = self.get_signal_value("image_encoder")

        if self._model_type == model_type:
            if not model_type or self._pipeline_class is None:
                return None
            _, node_config = require_modiff_node_contract(
                self._pipeline_class,
                self.node_type,
                resolve_blocks=False,
            )
            node_params_to_update = dict(node_config["params"])
            node_params_to_update.pop("image_encoder", None)
            self.send_node_definition(node_params_to_update)
            return None

        if model_type is None or model_type == "":
            self._model_type = ""
            self._pipeline_class = None
            self.send_node_definition(node_params)
            return None
        try:
            self._pipeline_class = pipeline_class_from_model_type(model_type)
            _, node_config = require_modiff_node_contract(
                self._pipeline_class,
                self.node_type,
                resolve_blocks=False,
            )
        except ValueError:
            self._model_type = ""
            self._pipeline_class = None
            self.send_node_definition(node_params)
            raise
        self._model_type = model_type

        node_params_to_update = node_config["params"]
        node_params_to_update.pop("image_encoder", None)
        node_params.update(**node_params_to_update)
        self.send_node_definition(node_params)

    def execute(self, **kwargs):
        kwargs = dict(kwargs)
        require_route_state_shape_before_identity_resolution(kwargs)
        reject_undeclared_modular_generator(kwargs)
        reject_route_reserved_inputs_before_identity_resolution(kwargs)
        self._pipeline_class = pipeline_class_from_runtime_inputs(self._pipeline_class, kwargs)
        model_type = getattr(self._pipeline_class, "__name__", "")
        wan_route = route_contract_for_model_type(model_type) == "wan_i2v"
        if wan_route:
            require_cataloged_wan_action_source(
                image=kwargs.get("image"),
                last_image=kwargs.get("last_image"),
            )

        # 1. Get node config
        blocks, node_config = require_modiff_node_contract(self._pipeline_class, self.node_type)
        validate_route_field_contract(kwargs, node_config)
        kwargs = normalize_modular_runtime_params(kwargs, node_config)
        source_snapshot = None
        image_preflight = None
        route_binding = None
        if wan_route:
            source_snapshot = snapshot_wan_source_media(kwargs.get("image"), kwargs.get("last_image"))
            image_preflight = preflight_wan_image_encoder_inputs(
                image=kwargs.get("image"),
                last_image=kwargs.get("last_image"),
                height=kwargs.get("height"),
                width=kwargs.get("width"),
            )
            route_binding = require_component_binding(
                kwargs.get("image_encoder"),
                label="image encoder",
                expected_model_type=model_type,
                expected_role="image_encoder",
            )
            preinit_image_encoder = resolve_managed_component_by_id(
                components,
                kwargs.get("image_encoder"),
                label="Image Embeddings image encoder",
            )
            preinit_image_encoder_config_seal = wan_image_encoder_contract_from_component(preinit_image_encoder)

        # 2. Create pipeline
        repo_id = None
        revision = None
        if (image_encoder := kwargs.get("image_encoder")) and "repo_id" in image_encoder:
            repo_id = image_encoder["repo_id"]
            revision = image_encoder.get("revision")

        if repo_id is None:
            self.notify(
                "You have to connect the image encoder",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            return None

        self._pipeline = blocks.init_pipeline(components_manager=components)

        # 3. Update components
        expected_component_names = blocks.component_names
        model_input_names = node_config["model_input_names"]
        model_ids = collect_model_ids(
            kwargs,
            target_key_names=model_input_names,
            target_model_names=expected_component_names,
        )

        # The image encoder contract does not currently expose its processor as
        # a model input, so load the matching Diffusers component explicitly.
        # Network writes remain owned by the app's download flow.
        spec = ComponentSpec(
            name="image_processor",
            repo=repo_id,
            subfolder="image_processor",
            variant="",
            revision=revision,
        )
        comp = spec.load(local_files_only=True)
        comp_id = components.add("image_processor", comp, collection=self.node_id)
        model_ids.append(comp_id)

        components_to_update = {}
        if model_ids:
            components_to_update = components.get_components_by_ids(ids=model_ids, return_dict_with_names=True)
            if components_to_update:
                self._pipeline.update_components(**components_to_update)

        installed_image_encoder = getattr(self._pipeline, "image_encoder", None)
        installed_image_processor = getattr(self._pipeline, "image_processor", None)
        image_processor_config_seal = None
        if wan_route:
            require_component_binding(
                kwargs.get("image_encoder"),
                label="image encoder",
                expected_model_type=model_type,
                expected_token=route_binding,
                expected_role="image_encoder",
            )
            if components_to_update.get("image_encoder") is not installed_image_encoder:
                raise ValueError("The Wan image pipeline did not install the exact connected image encoder.")
            if components_to_update.get("image_processor") is not installed_image_processor:
                raise ValueError("The Wan image pipeline did not install the exact locally resolved image processor.")
            if installed_image_encoder is not preinit_image_encoder:
                raise ValueError("The connected Wan image encoder changed during pipeline initialization.")
            if wan_image_encoder_contract_from_component(installed_image_encoder) != preinit_image_encoder_config_seal:
                raise ValueError("The connected Wan image encoder contract changed during pipeline initialization.")
            image_processor_config_seal = wan_image_processor_config_seal(
                installed_image_processor,
                workflow=image_preflight[0],
            )

        # 5. Compile runtime inputs from kwargs based on node_config["input_names"]
        node_kwargs = {}
        input_names = node_config["input_names"]

        for name in input_names:
            if name not in kwargs:
                continue
            value = kwargs.get(name)

            if isinstance(value, dict) and name not in blocks.input_names:
                for k, v in value.items():
                    if k in blocks.input_names:
                        node_kwargs[k] = v
            elif name in blocks.input_names:
                node_kwargs[name] = value

        # 6. Run the pipeline
        if wan_route:
            live_image_encoder = resolve_managed_component_by_id(
                components,
                kwargs.get("image_encoder"),
                label="Image Embeddings image encoder",
            )
            if live_image_encoder is not installed_image_encoder:
                raise ValueError("The connected Wan image encoder changed before upstream execution.")
            if (
                getattr(self._pipeline, "image_encoder", None) is not installed_image_encoder
                or getattr(self._pipeline, "image_processor", None) is not installed_image_processor
            ):
                raise ValueError("Wan image encoder components changed before upstream execution.")
            if (
                wan_image_processor_config_seal(installed_image_processor, workflow=image_preflight[0])
                != image_processor_config_seal
            ):
                raise ValueError("The Wan image processor configuration changed before upstream execution.")
            if wan_image_encoder_contract_from_component(installed_image_encoder) != preinit_image_encoder_config_seal:
                raise ValueError("The connected Wan image encoder contract changed before upstream execution.")
        try:
            node_output_state = self._pipeline(**node_kwargs)
        except ValueError as e:
            self.notify(str(e), variant="error", persist=False, autoHideDuration=MESSAGE_DURATION)
            return None

        if wan_route:
            require_component_binding(
                kwargs.get("image_encoder"),
                label="image encoder",
                expected_model_type=model_type,
                expected_token=route_binding,
                expected_role="image_encoder",
            )
            if (
                getattr(self._pipeline, "image_encoder", None) is not installed_image_encoder
                or getattr(self._pipeline, "image_processor", None) is not installed_image_processor
            ):
                raise ValueError("Wan image encoder components changed during upstream execution.")
            if resolve_managed_component_by_id(
                components,
                kwargs.get("image_encoder"),
                label="Image Embeddings image encoder",
            ) is not installed_image_encoder:
                raise ValueError("The connected Wan image encoder changed during upstream execution.")
            if (
                wan_image_processor_config_seal(installed_image_processor, workflow=image_preflight[0])
                != image_processor_config_seal
            ):
                raise ValueError("The Wan image processor configuration changed during upstream execution.")
            if wan_image_encoder_contract_from_component(installed_image_encoder) != preinit_image_encoder_config_seal:
                raise ValueError("The connected Wan image encoder contract changed during upstream execution.")

        # 7. Prepare outputs based on node_config["output_names"]
        output_names = node_config["output_names"].copy()
        outputs = {}
        for name in output_names:
            if name == "doc":
                outputs["doc"] = self._pipeline.blocks.doc
            elif name == ROUTE_STATE_OUTPUT:
                if not wan_route or route_binding is None or source_snapshot is None or image_preflight is None:
                    raise ValueError("The Wan Image Embeddings route is missing its backend binding.")
                outputs[name] = issue_wan_image_encoder_route_state(
                    binding=route_binding,
                    image=kwargs.get("image"),
                    last_image=kwargs.get("last_image"),
                    height=kwargs.get("height"),
                    width=kwargs.get("width"),
                    image_embeds=node_output_state.get("image_embeds"),
                    image_encoder=installed_image_encoder,
                    image_processor=installed_image_processor,
                    resized_image=node_output_state.get("resized_image"),
                    resized_last_image=node_output_state.get("resized_last_image"),
                    execution_device=self._pipeline._execution_device,
                    source_snapshot=source_snapshot,
                    preflight_geometry=image_preflight,
                )
            else:
                outputs[name] = node_output_state.get(name)

        return outputs
