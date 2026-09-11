"""Generic, exact SDXL IP-Adapter encoding action."""

import torch
from diffusers import BaseGuidance, ComponentSpec

from modiff.NodeBase import NodeBase
from modiff.auxiliary_ip_adapter import resolve_reviewed_sdxl_ip_adapter

from . import components
from .modular_utils import (
    normalize_modular_runtime_params,
    pipeline_class_from_model_type,
    pipeline_class_from_runtime_inputs,
    require_modiff_node_contract,
)
from .route_state import (
    clear_sdxl_ip_adapter_state,
    issue_sdxl_ip_adapter_bundle,
    prepare_sdxl_ip_adapter_unet,
    reject_route_reserved_inputs_before_identity_resolution,
    require_component_binding,
    require_route_state_shape_before_identity_resolution,
    require_sdxl_ip_adapter_bundle,
    resolve_managed_component_by_id,
    sdxl_ip_adapter_feature_extractor_contract,
    sdxl_ip_adapter_image_encoder_contract,
    sdxl_ip_adapter_unet_contract,
)


class IPAdapter(NodeBase):
    label = "IP-Adapter Embeddings"
    category = "embedding"
    resizable = True
    skipParamsCheck = True
    node_type = "ip_adapter"
    params = {
        "unet": {
            "label": "Denoise Model *",
            "display": "input",
            "type": "diffusers_auto_model",
            "required": True,
            "onSignal": "update_node",
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._model_type = ""
        self._pipeline_class = None
        self._pipeline = None
        self._image_encoder = None
        self._image_encoder_identity = None

    def update_node(self, values, ref):
        model_type = self.get_signal_value("unet")
        if not model_type:
            self._model_type = ""
            self._pipeline_class = None
            self.send_node_definition({})
            return None
        try:
            pipeline_class = pipeline_class_from_model_type(model_type)
            _blocks, node_config = require_modiff_node_contract(
                pipeline_class,
                self.node_type,
                resolve_blocks=False,
            )
        except ValueError:
            self._model_type = ""
            self._pipeline_class = None
            self.send_node_definition({})
            raise
        self._model_type = model_type
        self._pipeline_class = pipeline_class
        node_params = dict(node_config["params"])
        node_params.pop("unet", None)
        self.send_node_definition(node_params)
        return None

    def _cache_params_equal(self, previous, current):
        equal = super()._cache_params_equal(previous, current)
        if not equal or not isinstance(current, dict):
            return equal
        self._pipeline_class = pipeline_class_from_runtime_inputs(self._pipeline_class, current)
        model_type = getattr(self._pipeline_class, "__name__", "")
        binding = require_component_binding(
            current.get("unet"),
            label="IP-Adapter UNet",
            expected_model_type=model_type,
            expected_role="denoiser",
        )
        unet = resolve_managed_component_by_id(components, current.get("unet"), label="IP-Adapter UNet")
        guider = current.get("guider")
        if not isinstance(guider, BaseGuidance):
            raise TypeError("SDXL IP-Adapter requires a connected Diffusers Guider.")
        require_sdxl_ip_adapter_bundle(
            self.output.get("ip_adapter"),
            binding=binding,
            unet=unet,
            guider=guider,
        )
        return True

    def _load_image_encoder(self, artifact, *, dtype, device):
        identity = (
            artifact.repository,
            artifact.revision,
            artifact.image_encoder_subfolder,
            artifact.image_encoder_class,
            str(dtype),
            str(device),
        )
        if self._image_encoder is not None and self._image_encoder_identity == identity:
            sdxl_ip_adapter_image_encoder_contract(self._image_encoder)
            return self._image_encoder

        from transformers import CLIPVisionModelWithProjection

        if artifact.image_encoder_class != CLIPVisionModelWithProjection.__name__:
            raise ValueError("The installed Transformers runtime does not match the reviewed image-encoder class.")
        spec = ComponentSpec(
            name="image_encoder",
            type_hint=CLIPVisionModelWithProjection,
            repo=artifact.repository,
            subfolder=artifact.image_encoder_subfolder,
            revision=artifact.revision,
        )
        reusable = components.get_components_by_ids(
            ids=components.get_ids(names="image_encoder"),
            return_dict_with_names=False,
        )
        matching = [
            component
            for component in reusable.values()
            if getattr(component, "_diffusers_load_id", None) == spec.load_id
            and isinstance(component, CLIPVisionModelWithProjection)
            and getattr(component, "dtype", None) == dtype
        ]
        if len(matching) > 1:
            raise ValueError(
                "The shared ComponentsManager contains multiple compatible SDXL IP-Adapter image encoders. "
                "Switch model families once to clear the ambiguous resident state."
            )
        if matching:
            image_encoder = matching[0]
            sdxl_ip_adapter_image_encoder_contract(image_encoder)
            self._image_encoder = image_encoder
            self._image_encoder_identity = identity
            return image_encoder
        if any(getattr(component, "_diffusers_load_id", None) == spec.load_id for component in reusable.values()):
            raise ValueError(
                "The resident SDXL IP-Adapter image encoder has an incompatible class or dtype. "
                "Switch model families before changing the image-encoder runtime dtype."
            )

        image_encoder = spec.load(local_files_only=True, torch_dtype=dtype)
        image_encoder.to(device=device)
        sdxl_ip_adapter_image_encoder_contract(image_encoder)
        self._image_encoder = image_encoder
        self._image_encoder_identity = identity
        return image_encoder

    def execute(self, **kwargs):
        kwargs = dict(kwargs)
        require_route_state_shape_before_identity_resolution(kwargs)
        reject_route_reserved_inputs_before_identity_resolution(kwargs)
        self._pipeline_class = pipeline_class_from_runtime_inputs(self._pipeline_class, kwargs)
        self._model_type = getattr(self._pipeline_class, "__name__", "")
        blocks, node_config = require_modiff_node_contract(self._pipeline_class, self.node_type)
        if self._model_type != "StableDiffusionXLModularPipeline":
            raise ValueError("The generic IP-Adapter action currently supports only its reviewed SDXL contract.")
        kwargs = normalize_modular_runtime_params(kwargs, node_config)
        binding = require_component_binding(
            kwargs.get("unet"),
            label="IP-Adapter UNet",
            expected_model_type=self._model_type,
            expected_role="denoiser",
        )
        unet = resolve_managed_component_by_id(components, kwargs.get("unet"), label="IP-Adapter UNet")
        guider = kwargs.get("guider")
        if not isinstance(guider, BaseGuidance):
            raise TypeError("SDXL IP-Adapter requires a connected Diffusers Guider.")
        artifact = resolve_reviewed_sdxl_ip_adapter(
            selection=kwargs.get("adapter_model"),
            revision=kwargs.get("adapter_revision"),
            weight_name=kwargs.get("adapter_weight_name"),
        )
        previous_state = prepare_sdxl_ip_adapter_unet(
            unet,
            binding=binding,
        )
        pipeline = blocks.init_pipeline(components_manager=components)
        image_encoder = self._load_image_encoder(
            artifact,
            dtype=getattr(unet, "dtype", torch.float32),
            device=pipeline._execution_device,
        )
        pipeline.update_components(unet=unet, image_encoder=image_encoder, guider=guider)
        if (
            getattr(pipeline, "unet", None) is not unet
            or getattr(pipeline, "image_encoder", None) is not image_encoder
            or getattr(pipeline, "guider", None) is not guider
        ):
            raise ValueError("The SDXL IP-Adapter pipeline did not install its exact connected components.")
        feature_extractor = getattr(pipeline, "feature_extractor", None)
        sdxl_ip_adapter_feature_extractor_contract(feature_extractor)

        mutated = False
        try:
            if previous_state is not None and previous_state._artifact_identity != artifact.identity:
                pipeline.unload_ip_adapter()
                clear_sdxl_ip_adapter_state(unet, expected_state=previous_state)
                previous_state = None
            if previous_state is None:
                mutated = True
                pipeline.load_ip_adapter(
                    str(artifact.load_directory),
                    subfolder="",
                    weight_name=artifact.weight_name,
                    local_files_only=True,
                )
            mutated = True
            pipeline.set_ip_adapter_scale(kwargs["adapter_scale"])
            sdxl_ip_adapter_unet_contract(unet, scale=kwargs["adapter_scale"])
            output_state = pipeline(ip_adapter_image=kwargs["ip_adapter_image"])
            if (
                getattr(pipeline, "unet", None) is not unet
                or getattr(pipeline, "image_encoder", None) is not image_encoder
                or getattr(pipeline, "guider", None) is not guider
            ):
                raise ValueError("SDXL IP-Adapter components changed during upstream encoding.")
            if resolve_managed_component_by_id(
                components,
                kwargs.get("unet"),
                label="IP-Adapter UNet",
            ) is not unet:
                raise ValueError("The connected SDXL UNet changed during IP-Adapter encoding.")
            sdxl_ip_adapter_image_encoder_contract(image_encoder)
            sdxl_ip_adapter_feature_extractor_contract(feature_extractor)
            sdxl_ip_adapter_unet_contract(unet, scale=kwargs["adapter_scale"])
            bundle = issue_sdxl_ip_adapter_bundle(
                binding=binding,
                unet=unet,
                artifact_identity=artifact.identity,
                image_encoder=image_encoder,
                feature_extractor=feature_extractor,
                guider=guider,
                scale=kwargs["adapter_scale"],
                image=kwargs["ip_adapter_image"],
                ip_adapter_embeds=output_state.get("ip_adapter_embeds"),
                negative_ip_adapter_embeds=output_state.get("negative_ip_adapter_embeds"),
            )
            self._pipeline = pipeline
        except Exception:
            if mutated:
                try:
                    pipeline.unload_ip_adapter()
                finally:
                    if previous_state is not None:
                        clear_sdxl_ip_adapter_state(unet, expected_state=previous_state)
            raise
        return {"ip_adapter": bundle, "doc": pipeline.blocks.doc}
