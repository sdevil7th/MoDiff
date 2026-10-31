"""Shared image-stage runtime for loader-validated owners and sealed material.

Only actual loader-validated selections may install an owner. Strategies are
declared backend code, never graph-supplied callables or pipeline-name fallback.
Tensor work is lazy and restricted to executed, bounded material outputs.
"""
from dataclasses import dataclass, field
from types import MappingProxyType
import hashlib
import json
import math
import re
import weakref

_SEAL = object()
_OWNER_KEY = '_modiff_image_stage_owner'
_MAX_MATERIAL_BYTES = 1024**3
_MAX_MATERIAL_ITEMS = 256


def _plain(value, depth=0):
    if depth > 6:
        raise ValueError('Image-stage metadata exceeds its depth bound.')
    if value is None or type(value) is bool:
        return value
    if type(value) is str and len(value) <= 4096:
        return value
    if type(value) in (int, float) and math.isfinite(value) and abs(value) <= 2**53 - 1:
        return value
    if type(value) in (tuple, list) and len(value) <= 64:
        return [_plain(item, depth + 1) for item in value]
    if type(value) is dict and len(value) <= 64 and all(type(key) is str and len(key) <= 128 for key in value):
        return {key: _plain(item, depth + 1) for key, item in value.items()}
    raise ValueError('Image-stage metadata must be finite plain backend values.')


def _plain_key(value):
    encoded = json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False)
    if len(encoded.encode('utf-8')) > 16_384:
        raise ValueError('Image-stage metadata exceeds its byte bound.')
    return encoded


def _immutable_plain(value):
    if type(value) is dict:
        return MappingProxyType({key: _immutable_plain(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_immutable_plain(item) for item in value)
    return value


@dataclass(frozen=True)
class OwnerSelection:
    profile_id: str
    task: str
    pipeline_class: str
    repository: str
    revision: str
    artifacts: tuple

    def __post_init__(self):
        for value in (self.profile_id, self.task, self.pipeline_class, self.repository, self.revision):
            if type(value) is not str or not value or len(value) > 256 or value != value.strip():
                raise ValueError('Exact bounded loader selection required.')
        if (type(self.artifacts) is not tuple or not 1 <= len(self.artifacts) <= 8
                or any(type(pair) is not tuple or len(pair) != 2
                       or type(pair[0]) is not str or not pair[0] or len(pair[0]) > 256
                       or type(pair[1]) is not str or re.fullmatch('[0-9a-f]{40}', pair[1]) is None
                       for pair in self.artifacts)
                or (self.repository, self.revision) != self.artifacts[0]):
            raise ValueError('Exact immutable actually loaded artifact census required.')


@dataclass(frozen=True, eq=False)
class _OwnerRecord:
    owner: object
    selection: OwnerSelection
    strategy_key: str
    targets: tuple
    components: tuple
    epoch: object
    seal: object


@dataclass(frozen=True, eq=False)
class RawEncoding:
    material: dict
    consumed: dict
    constraints: dict
    output_context: dict
    omitted: dict = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class EncodedInputs:
    record: object
    configuration: str
    material: object
    consumed: object
    constraints: object
    output_context: object
    omitted: object
    stamps: tuple
    logical_material_bytes: int
    normalization_copy_bytes: int
    seal: object


@dataclass(frozen=True, eq=False)
class LatentBinding:
    encoding: EncodedInputs
    latent: object
    stamp: tuple
    width: int
    height: int
    seal: object


def configuration_key(owner):
    return _plain_key({
        'recipe': vars(owner).get('_modiff_runtime_config'),
        'adapters': vars(owner).get('_modiff_adapter_scales', {}),
    })


def _consumed_metadata(consumed, omitted):
    from modiff.execution_input_provenance import _bounded_value, MAX_RECORD_BYTES
    if type(consumed) is not dict or len(consumed) > 64 or type(omitted) is not dict or len(omitted) > 64:
        raise ValueError('Image-stage consumed metadata must be a finite plain object.')
    safe, exclusions, remaining = {}, dict(omitted), MAX_RECORD_BYTES
    for key, value in consumed.items():
        if type(key) is not str or not key or len(key) > 128:
            raise ValueError('Image-stage consumed metadata keys must be bounded plain strings.')
        if key in exclusions:
            raise ValueError('A consumed field cannot also be explicitly omitted.')
        bounded, reason = _bounded_value(value)
        size = len(json.dumps(bounded, ensure_ascii=True).encode('ascii')) if not reason else 0
        if not reason and size > remaining:
            reason = 'record-size-limit'
        if reason:
            exclusions[key] = reason
        else:
            safe[key] = bounded
            remaining -= size
    if any(type(key) is not str or len(key) > 128 or type(reason) is not str or len(reason) > 128
           for key, reason in exclusions.items()):
        raise ValueError('Image-stage metadata omissions must be bounded plain strings.')
    return _immutable_plain(safe), _immutable_plain(exclusions)


def install_owner(owner, selection, strategy):
    """Called only after the existing loader's selection/load/hooks/MM succeed."""
    if type(selection) is not OwnerSelection:
        raise ValueError('A validated current loader selection is required.')
    if vars(owner).get('_modiff_stage_mutation_failed') is True:
        raise ValueError('A failed image-stage mutation requires complete reset or a new loader.')
    strategy.validate_selection(selection)
    strategy.validate_current_owner(owner, selection)
    target_map = strategy.targets(owner)
    if type(target_map) is not dict or set(target_map) != {'encoder', 'denoiser', 'decoder'}:
        raise ValueError('The declared image-stage strategy lacks actual runtime targets.')
    targets = tuple((name, weakref.ref(target_map[name])) for name in ('encoder', 'denoiser', 'decoder'))
    components = strategy.components(owner)
    if (type(components) is not tuple or not 1 <= len(components) <= 32
            or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not str for row in components)
            or len({name for name, _ in components}) != len(components)):
        raise ValueError('Finite exact component ownership declaration required.')
    references = tuple((name, None if component is None else weakref.ref(component))
                       for name, component in components)
    record = _OwnerRecord(weakref.ref(owner), selection, strategy.key, targets, references, object(), _SEAL)
    setattr(owner, _OWNER_KEY, record)
    return record


def require_owner(owner, strategy, record=None):
    current = vars(owner).get(_OWNER_KEY)
    if (type(current) is not _OwnerRecord or current.seal is not _SEAL
            or current.owner() is not owner or current.strategy_key != strategy.key
            or record is not None and current is not record):
        raise ValueError('Image-stage owner is incomplete, replaced or stale.')
    strategy.validate_selection(current.selection)
    strategy.validate_current_owner(owner, current.selection)
    targets = strategy.targets(owner)
    if any(targets.get(name) is not reference() or reference() is None for name, reference in current.targets):
        raise ValueError('The actual sampler or decoder owner changed.')
    components = strategy.components(owner)
    if len(components) != len(current.components):
        raise ValueError('Image-stage component census changed.')
    for (name, component), (expected_name, reference) in zip(components, current.components, strict=True):
        if name != expected_name or (component is not None if reference is None else reference() is not component):
            raise ValueError('Image-stage component ownership changed.')
    return current


def begin_mutation(owner):
    """Suspend stage reuse before the FIRST official component mutation."""
    previous = vars(owner).get(_OWNER_KEY)
    if type(previous) is _OwnerRecord and previous.seal is _SEAL:
        setattr(owner, _OWNER_KEY, None)
        owner._modiff_stage_mutation_failed = True
        owner._modiff_stage_mutation_record = previous
        return previous
    return None


def finish_mutation(owner, previous, strategy):
    """Only a completely successful reviewed mutation may restore the owner."""
    if previous is not None:
        owner._modiff_stage_mutation_failed = False
        try:
            install_owner(owner, previous.selection, strategy)
        except Exception:
            owner._modiff_stage_mutation_failed = True
            setattr(owner, _OWNER_KEY, None)
            raise
        owner._modiff_stage_mutation_record = None


def begin_adapter_transaction(owner):
    """Invalidate input material before the first ordinary official mutation."""
    if vars(owner).get('_modiff_stage_mutation_failed') is True:
        raise ValueError('A failed stage adapter mutation requires a new successful loader or complete reviewed reset.')
    record = vars(owner).get(_OWNER_KEY)
    if record is None:
        return None
    if type(record) is not _OwnerRecord or record.seal is not _SEAL:
        raise ValueError('An exact current image-stage owner is required for adapter mutation.')
    strategy = strategy_for_key(record.strategy_key)
    require_owner(owner, strategy, record)
    # The split-owner bridge already brackets each actual owner mutation, and
    # needs its complete owner during official target preflight.
    if getattr(strategy, 'manages_adapter_mutations', False):
        return None
    return begin_mutation(owner), strategy


def finish_adapter_transaction(owner, transaction):
    previous, strategy = transaction
    finish_mutation(owner, previous, strategy)


def reset_adapter_owner(owner, unload):
    """Ordinary loader adoption removes actual adapters before a new epoch."""
    current = vars(owner).get(_OWNER_KEY)
    record = current if current is not None else vars(owner).get('_modiff_stage_mutation_record')
    if type(record) is not _OwnerRecord or record.seal is not _SEAL:
        raise ValueError('A reviewed complete image-stage owner is required for adapter reset.')
    strategy = strategy_for_key(record.strategy_key)
    if getattr(strategy, 'manages_adapter_mutations', False):
        # The split bridge validates both actual target inventories and can
        # recover its own partial failure only by complete official unload.
        return unload()
    require_owner(owner, strategy, record)
    inventories = []
    for _name, component in strategy.components(owner):
        if component is None:
            continue
        inventory = getattr(component, 'peft_config', {})
        if type(inventory) is not dict or len(inventory) > 32:
            raise ValueError('Actual adapter reset inventory is malformed or exceeds its bound.')
        inventories.append((component, inventory))
    scales = vars(owner).get('_modiff_adapter_scales', {})
    if type(scales) is not dict or len(scales) > 32:
        raise ValueError('Actual adapter scale metadata is malformed or exceeds its bound.')
    if not scales and not any(inventory for _component, inventory in inventories):
        # Ordinary no-adapter adoption retains valid encoded material.
        return unload()
    previous = begin_mutation(owner)
    unload()
    if any(getattr(component, 'peft_config', {}) for component, _inventory in inventories):
        raise ValueError('Official unload did not clear the complete owned adapter inventory.')
    owner._modiff_adapter_scales = {}
    finish_mutation(owner, previous, strategy)


def _tensor_stamp(value):
    return (tuple(value.shape), value.dtype, value.device, value._version)


def _freeze_material(value):
    import numpy as np
    import torch
    from PIL import Image
    memo, stamps = {}, []
    count = logical = copied = 0

    def freeze(item, depth=0):
        nonlocal count, logical, copied
        count += 1
        if count > _MAX_MATERIAL_ITEMS or depth > 8:
            raise ValueError('Image-stage material exceeds its finite structure bound.')
        if isinstance(item, torch.Tensor):
            if id(item) in memo:
                return memo[id(item)]
            size = item.numel() * item.element_size()
            logical += size
            if logical > _MAX_MATERIAL_BYTES or item.device.type == 'meta':
                raise ValueError('Image-stage material exceeds its byte bound or is not materialized.')
            if (torch.is_inference(item) or item.grad_fn is not None
                    or item.requires_grad or item._base is not None):
                # Own only executed bounded data, not an autograd graph or an
                # unrelated backing allocation. The copy is explicitly counted.
                # Inference output also needs a real ordinary version counter.
                with torch.inference_mode(False), torch.no_grad():
                    normalized = item.detach().clone()
                copied += size
            else:
                normalized = item
            stamp = _tensor_stamp(normalized)
            memo[id(item)] = normalized
            stamps.append(('tensor', weakref.ref(normalized), stamp))
            return normalized
        if isinstance(item, Image.Image):
            if id(item) in memo:
                return memo[id(item)]
            if (item.width <= 0 or item.height <= 0 or item.width * item.height > 16 * 1024**2
                    or len(item.getbands()) > 4):
                raise ValueError('Image-stage media exceeds its finite pixel bound.')
            size = len(item.tobytes())
            logical += size
            if logical > _MAX_MATERIAL_BYTES:
                raise ValueError('Image-stage media exceeds its byte bound.')
            normalized = item.copy()
            memo[id(item)] = normalized
            stamps.append(('pil', weakref.ref(normalized),
                (normalized.mode, normalized.size, hashlib.sha256(normalized.tobytes()).digest())))
            copied += size
            return normalized
        if isinstance(item, np.ndarray):
            if id(item) in memo:
                return memo[id(item)]
            logical += item.nbytes
            if logical > _MAX_MATERIAL_BYTES or item.dtype.hasobject:
                raise ValueError('Image-stage arrays exceed their bound or contain objects.')
            normalized = item.copy()
            normalized.setflags(write=False)
            memo[id(item)] = normalized
            stamps.append(('numpy', weakref.ref(normalized),
                (normalized.shape, normalized.dtype.str, hashlib.sha256(normalized.tobytes()).digest())))
            copied += item.nbytes
            return normalized
        if type(item) is dict and len(item) <= 64 and all(type(key) is str and len(key) <= 128 for key in item):
            return MappingProxyType({key: freeze(child, depth + 1) for key, child in item.items()})
        if type(item) in (tuple, list) and len(item) <= 64:
            return tuple(freeze(child, depth + 1) for child in item)
        return _plain(item)

    frozen = freeze(value)
    return frozen, tuple(stamps), logical, copied


def _check_stamps(stamps):
    for kind, reference, stamp in stamps:
        value = reference()
        if value is None:
            raise ValueError('Image-stage material was released.')
        actual = (_tensor_stamp(value) if kind == 'tensor' else
                  (value.mode, value.size, hashlib.sha256(value.tobytes()).digest()) if kind == 'pil' else
                  (value.shape, value.dtype.str, hashlib.sha256(value.tobytes()).digest()))
        if actual != stamp:
            raise ValueError('Image-stage material was modified.')


def require_encoding(owner, strategy, value, normalized_values=None):
    if type(value) is not EncodedInputs or value.seal is not _SEAL:
        raise ValueError('Denoise requires executed matching image conditioning.')
    require_owner(owner, strategy, value.record)
    if value.configuration != configuration_key(owner):
        raise ValueError('Image-stage recipe or adapter configuration changed.')
    _check_stamps(value.stamps)
    if normalized_values is not None:
        strategy.validate_denoise_values(value.constraints, normalized_values)
        decoder_guard = getattr(strategy, 'validate_decoder_envelope', None)
        if callable(decoder_guard):
            decoder_guard(value.material, normalized_values)
    return value


def encode(owner, strategy, normalized_values):
    import torch
    record = require_owner(owner, strategy)
    configuration = configuration_key(owner)
    with torch.no_grad():
        result = strategy.encode(owner, normalized_values)
    if type(result) is not RawEncoding:
        raise ValueError('A declared image encoder must return its actual material result.')
    constraints = _plain(result.constraints)
    consumed, omitted = _consumed_metadata(result.consumed, result.omitted)
    payload, stamps, logical, copied = _freeze_material({
        'material': result.material, 'output_context': result.output_context,
    })
    require_owner(owner, strategy, record)
    if configuration != configuration_key(owner):
        raise ValueError('Image owner changed during encoding.')
    return EncodedInputs(record, configuration, payload['material'],
        consumed, _immutable_plain(constraints), payload['output_context'], omitted,
        stamps, logical, copied, _SEAL)


def denoise(owner, strategy, encoding, normalized_values, callback_kwargs):
    import torch
    encoding = require_encoding(owner, strategy, encoding, normalized_values)
    with torch.no_grad():
        latent = strategy.denoise(owner, encoding, normalized_values, callback_kwargs)
    require_encoding(owner, strategy, encoding, normalized_values)
    frozen, _stamps, _logical, _copied = _freeze_material({'latent': latent})
    latent = frozen['latent']
    binding = LatentBinding(encoding, weakref.ref(latent), _tensor_stamp(latent),
        normalized_values['width'], normalized_values['height'], _SEAL)
    # Ordinary full-pipeline latent outputs retain their historical raw decode
    # contract. Only an actual new stage result requires its executed binding.
    latent._modiff_image_stage_latent_binding = binding
    return latent, binding


def require_latent(owner, strategy, latent, binding, width, height):
    if type(binding) is not LatentBinding or binding.seal is not _SEAL:
        raise ValueError('Decode requires the executed matching latent binding.')
    require_encoding(owner, strategy, binding.encoding)
    if (binding.latent() is not latent or binding.stamp != _tensor_stamp(latent)
            or (width, height) != (binding.width, binding.height)):
        raise ValueError('Image latents, geometry or owner do not match the sampler output.')
    return binding.encoding


def decoder_context(owner, strategy, latent, binding, width, height):
    encoding = require_latent(owner, strategy, latent, binding, width, height)
    return strategy.targets(owner)['decoder'], encoding


def postprocess(owner, strategy, encoding, decoded, output_type):
    require_encoding(owner, strategy, encoding)
    result = strategy.postprocess(decoded, output_type, encoding.output_context)
    require_encoding(owner, strategy, encoding)
    return result




def require_selected_stage_profile(strategy, loader_values, primary_pair):
    from modiff.diffusers_profiles import resolve_execution_profiles_for_loader
    from modiff.model_artifact_catalog import catalog_repository_pin

    if type(loader_values) is not dict or type(primary_pair) is not tuple or len(primary_pair) != 2:
        raise ValueError("A successful exact image loader context is required.")
    requested_id = loader_values.get("execution_profile_id")
    requested_mode = loader_values.get("mode")
    if (type(requested_id) is not str or requested_id not in strategy.profile_ids
            or type(requested_mode) is not str or requested_mode not in strategy.tasks):
        raise ValueError("Image stage profile/task is missing or outside its declaration.")
    profiles, reason = resolve_execution_profiles_for_loader(
        "modules.DiffusersImage", "LoadPipeline", loader_values,
    )
    if reason is not None or len(profiles) != 1:
        raise ValueError("One current authoritative image execution profile is required.")
    profile = profiles[0]
    if (not profile.public or profile.execution_path != "direct-diffusers-image"
            or profile.id != requested_id or requested_mode not in profile.modes
            or loader_values.get("pipeline_class") != profile.pipeline_class):
        raise ValueError("The current image stage profile/class/task does not agree.")
    selection = loader_values.get("model_id")
    if (type(selection) is not dict or set(selection) != {"source", "value"}
            or selection.get("source") != "hub"):
        raise ValueError("Separate image stages require their exact reviewed Hub primary selection.")
    repo, revision = primary_pair
    if (type(repo) is not str or type(revision) is not str
            or selection.get("value") != repo or loader_values.get("revision") != revision
            or repo != profile.default_repo):
        # In particular, Canny uses its OWN current profile/repo. It does not
        # borrow Depth's adapter.default_repo or compatible/repair artifacts.
        raise ValueError("Image stages do not match the actual selected primary profile pair.")
    pin = catalog_repository_pin(repo, model_type=profile.pipeline_class)
    if pin is None or pin.get("repo") != repo or pin.get("revision") != revision:
        raise ValueError("Image stage artifact does not match the current immutable catalog pair.")
    return profile




def _validate_image_decoder_envelope(material, values):
    # These declared stages use the existing real Image Decode layout envelope.
    # Refuse before cached return or official sampling, rather than widening it.
    import torch
    prompt = material.get('prompt_embeds')
    width, height = values.get('width'), values.get('height')
    if (not isinstance(prompt, torch.Tensor) or prompt.ndim != 3
            or type(width) is not int or type(height) is not int
            or not 1 <= prompt.shape[0] <= 8 or width <= 0 or height <= 0
            or prompt.shape[0] * width * height > 16 * 1024 * 1024):
        raise ValueError('Encoded batch and geometry exceed the existing image decoder envelope.')


class _FluxInputStrategy:
    validate_decoder_envelope = staticmethod(_validate_image_decoder_envelope)
    def __init__(self, key, *, control, profile_ids, tasks):
        self.key = key
        self.control = control
        # Scalar controls actually used by encoding/repetition/material seals.
        self.encode_fields = frozenset({'width', 'height', 'max_sequence_length',
            'num_images_per_prompt', *(('seed',) if control else ())})
        self.profile_ids = frozenset(profile_ids)
        self.tasks = frozenset(tasks)

    def validate_selection(self, selection):
        # The helper is merged alongside these strategies in the shared module.
        require_selected_stage_profile(self, {
            "execution_profile_id": selection.profile_id,
            "pipeline_class": selection.pipeline_class,
            "mode": selection.task,
            "model_id": {"source": "hub", "value": selection.repository},
            "revision": selection.revision,
        }, (selection.repository, selection.revision))
        if selection.artifacts != ((selection.repository, selection.revision),):
            raise ValueError("The selected image stage has an unexpected loaded artifact bundle.")

    def validate_current_owner(self, owner, selection):
        from modules.DiffusersImage.main import _image_pipeline_adapter

        adapter = _image_pipeline_adapter(owner)
        current = vars(owner)
        loader_values = current.get("_modiff_stage_loader_values")
        if type(loader_values) is not dict:
            raise ValueError("The actual successful image loader context is missing.")
        loader_values = _plain(loader_values)
        profile = require_selected_stage_profile(
            self, loader_values, (selection.repository, selection.revision),
        )
        if (adapter.pipeline_class != selection.pipeline_class
                or profile.id != selection.profile_id
                or loader_values.get("mode") != selection.task
                or current.get("_modiff_loaded_stage_artifacts") != selection.artifacts
                or current.get("_modiff_image_pipeline_class") != selection.pipeline_class
                or current.get("_modiff_image_mode") != selection.task
                or current.get("_modiff_image_source") != "hub"
                or current.get("_modiff_image_repo") != selection.repository
                or current.get("_modiff_image_revision") != selection.revision):
            raise ValueError("The actual current image owner differs from its successful loader selection.")

    def targets(self, owner):
        return {"encoder": owner, "denoiser": owner, "decoder": owner}

    def components(self, owner):
        names = (
            "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2",
            "transformer", "vae", "scheduler", "image_processor",
            *(("mask_processor",) if not self.control else ()),
        )
        return tuple((name, getattr(owner, name)) for name in names)

    def encode(self, owner, values):
        import torch

        # Connected Generator support needs its own initial/final-state and
        # graph-cache proof. Never replace a supplied generator with a seed.
        if values.get("generator") is not None:
            raise ValueError("Separate image stages do not yet accept an external Generator.")
        if values.get("latents") is not None or values.get("masked_image_latents") is not None:
            raise ValueError("Separate image stages require their reviewed source/seed encoding path.")
        count = values.get("num_images_per_prompt", 1)
        if values.get("prompt_embeds") is not None and count != 1:
            raise ValueError("Precomputed conditioning repetition needs separate stage proof.")
        width, height = values["width"], values["height"]
        joint = values.get("joint_attention_kwargs")
        scale = joint.get("scale") if joint is not None else None
        prompt, pooled, _text_ids = owner.encode_prompt(
            prompt=values.get("prompt"), prompt_2=values.get("prompt_2"),
            device=owner._execution_device, num_images_per_prompt=count,
            max_sequence_length=values["max_sequence_length"],
            prompt_embeds=values.get("prompt_embeds"),
            pooled_prompt_embeds=values.get("pooled_prompt_embeds"),
            lora_scale=scale,
        )
        batch = prompt.shape[0]
        material = {"prompt_embeds": prompt, "pooled_prompt_embeds": pooled}
        constraints = {
            "width": width, "height": height,
            "maxSequenceLength": values["max_sequence_length"], "loraScale": scale,
        }
        output_context = {}
        if self.control:
            generator = _stage_seed_generator(owner, values["seed"])
            control = owner.prepare_image(
                image=values["control_image"], width=width, height=height,
                batch_size=batch, num_images_per_prompt=count,
                device=owner._execution_device, dtype=owner.vae.dtype,
            )
            if control.ndim == 4:
                control = owner.vae.encode(control).latent_dist.sample(generator=generator)
                control = (control - owner.vae.config.shift_factor) * owner.vae.config.scaling_factor
                latent_height, latent_width = control.shape[2:]
                control = owner._pack_latents(
                    control, batch, owner.transformer.config.in_channels // 8,
                    latent_height, latent_width,
                )
            state = generator.get_state()
            if state.dtype != torch.uint8 or state.device.type != "cpu" or state.numel() > 65536:
                raise ValueError("Encoded Generator state exceeds its finite CPU byte bound.")
            material.update(control_image=control, post_vae_rng_state=state)
            constraints.update(seed=values["seed"], generatorDevice=str(generator.device))
        else:
            # Source/mask are copied and stamped by the common material seal.
            # Their real VAE preprocessing stays inside the official full call.
            output_context.update(source=values["image"], mask=values["mask_image"])
        return RawEncoding(material, {
            "width": width, "height": height,
            **({"seed": values["seed"]} if self.control else {}),
            "prompt": None if values.get("prompt_embeds") is not None else values.get("prompt"),
            "prompt_2": None if values.get("prompt_embeds") is not None else values.get("prompt_2"),
            "max_sequence_length": values["max_sequence_length"],
            "num_images_per_prompt": count,
            "requestedImagesPerPrompt": count,
            "effectiveBatch": batch,
            "consumedImagesPerPrompt": 1,
            "inputEncoding": "text_and_control" if self.control else "text_only",
        }, constraints, output_context)

    def validate_denoise_values(self, constraints, values):
        joint = values.get("joint_attention_kwargs")
        scale = joint.get("scale") if joint is not None else None
        if ((values["width"], values["height"]) != (constraints["width"], constraints["height"])
                or values["max_sequence_length"] != constraints["maxSequenceLength"]
                or scale != constraints["loraScale"]):
            raise ValueError("Changed image geometry or text encoding controls require current encoded inputs.")
        if self.control and values["seed"] != constraints["seed"]:
            raise ValueError("Changed Control seed requires current encoded conditioning.")
        if values.get("generator") is not None:
            raise ValueError("Separate image stages do not yet accept an external Generator.")

    def denoise(self, owner, encoding, values, callback_kwargs):
        import torch

        material, constraints = encoding.material, encoding.constraints
        if self.control:
            generator = torch.Generator(device=constraints["generatorDevice"])
            # Every actual denoise restores the immutable post-VAE snapshot;
            # neither this call nor a cache hit advances the retained snapshot.
            generator.set_state(material["post_vae_rng_state"])
        else:
            generator = _stage_seed_generator(owner, values["seed"])
        call = {
            "prompt": None, "prompt_2": None,
            "prompt_embeds": material["prompt_embeds"],
            "pooled_prompt_embeds": material["pooled_prompt_embeds"],
            "num_images_per_prompt": 1,
            "width": constraints["width"], "height": constraints["height"],
            "num_inference_steps": values["num_inference_steps"],
            "guidance_scale": values["guidance_scale"],
            "generator": generator, "output_type": "latent", "return_dict": True,
            "max_sequence_length": values["max_sequence_length"],
            **callback_kwargs,
        }
        if values.get("sigmas") is not None:
            call["sigmas"] = values["sigmas"]
        if values.get("joint_attention_kwargs") is not None:
            call["joint_attention_kwargs"] = values["joint_attention_kwargs"]
        if self.control:
            call["control_image"] = material["control_image"]
        else:
            call.update(image=encoding.output_context["source"],
                        mask_image=encoding.output_context["mask"], strength=values["strength"])
        return owner(**call).images

    def postprocess(self, decoded, output_type, output_context):
        if not self.control and output_type == "pil":
            from modules.DiffusersImage.main import composite_masked_pil_outputs

            return composite_masked_pil_outputs(decoded, output_context["source"], output_context["mask"])
        return decoded


def _stage_seed_generator(owner, seed):
    import torch

    device = owner._execution_device or owner.device or "cpu"
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except Exception:
        return torch.Generator(device="cpu").manual_seed(seed)


FILL_TEXT_STAGE = _FluxInputStrategy(
    "flux-fill-text:v1", control=False,
    profile_ids=("flux-fill:direct",), tasks=("inpaint", "outpaint"),
)
CONTROL_STAGE = _FluxInputStrategy(
    "flux-control-text-image:v1", control=True,
    profile_ids=("flux-canny:direct", "flux-depth:direct"), tasks=("control_image",),
)




class _PriorImageStrategy:
    validate_decoder_envelope = staticmethod(_validate_image_decoder_envelope)
    manages_adapter_mutations = True
    key = 'flux-redux-prior:v1'
    encode_fields = frozenset()  # Prior has fixed length/count; base owns output geometry/RNG.
    profile_ids = frozenset({'flux-redux:direct'})
    tasks = frozenset({'edit_image', 'multi_image_reference_edit'})

    def validate_selection(self, selection):
        from modiff.studio_execution_specs import studio_model_requirements_for_pair
        require_selected_stage_profile(self, {
            'execution_profile_id': selection.profile_id, 'mode': selection.task,
            'pipeline_class': selection.pipeline_class,
            'model_id': {'source': 'hub', 'value': selection.repository},
            'revision': selection.revision,
        }, (selection.repository, selection.revision))
        requirements = studio_model_requirements_for_pair(selection.pipeline_class, selection.task)
        expected = ((selection.repository, selection.revision),
            *((item['repo'], item['revision']) for item in requirements))
        if len(expected) != 2 or selection.artifacts != expected:
            raise ValueError('Image prior stages require both exact actually loaded artifacts.')

    def validate_current_owner(self, owner, selection):
        from modules.DiffusersImage.main import _image_pipeline_adapter
        adapter = _image_pipeline_adapter(owner)
        current = vars(owner)
        values = current.get('_modiff_stage_loader_values')
        if type(values) is not dict:
            raise ValueError('The actual successful prior/base loader context is missing.')
        profile = require_selected_stage_profile(self, _plain(values),
            (selection.repository, selection.revision))
        if (adapter.pipeline_class != selection.pipeline_class or profile.id != selection.profile_id
                or values.get('mode') != selection.task
                or current.get('_modiff_loaded_stage_artifacts') != selection.artifacts
                or current.get('_modiff_image_pipeline_class') != selection.pipeline_class
                or current.get('_modiff_image_mode') != selection.task
                or current.get('_modiff_image_source') != 'hub'
                or current.get('_modiff_image_repo') != selection.repository
                or current.get('_modiff_image_revision') != selection.revision
                or owner.base.text_encoder is not None or owner.base.text_encoder_2 is not None
                or owner.base.tokenizer is not None or owner.base.tokenizer_2 is not None
                or owner.prior.text_encoder is None or owner.prior.text_encoder_2 is None):
            raise ValueError('Image prior/base stages no longer match the successful actual owner.')

    def targets(self, owner):
        return {'encoder': owner.prior, 'denoiser': owner.base, 'decoder': owner.base}

    def components(self, owner):
        prior = ('image_encoder', 'image_embedder', 'feature_extractor',
                 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2')
        base = ('transformer', 'vae', 'scheduler', 'image_processor',
                'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2')
        return tuple(('prior.' + name, getattr(owner.prior, name)) for name in prior) + tuple(
            ('base.' + name, getattr(owner.base, name)) for name in base)

    def encode(self, owner, values):
        from modules.DiffusersImage.call_inputs import normalize_call_inputs, apply_call_inputs
        if values.get('generator') is not None or values.get('latents') is not None:
            raise ValueError('The selected prior stages do not accept external RNG or latent inputs.')
        image = values['image']
        call = {'image': image, 'prompt': values.get('prompt') or None, 'return_dict': True}
        count = len(image) if isinstance(image, (list, tuple)) else (
            int(image.shape[0]) if hasattr(image, 'shape') and len(image.shape) == 4 else 1)
        if count > 1:
            scales = [1., *([values['reference_strength']] * (count - 1))]
            call.update(prompt_embeds_scale=scales, pooled_prompt_embeds_scale=scales)
        extras = normalize_call_inputs('FluxReduxPipeline', {
            key: values.get(key) for key in ('prompt_embeds_scale', 'pooled_prompt_embeds_scale',
                'prompt_2', 'prompt_embeds', 'pooled_prompt_embeds')})
        for key in ('prompt_embeds_scale', 'pooled_prompt_embeds_scale'):
            if isinstance(extras.get(key), list) and len(extras[key]) != count:
                raise ValueError('A prior scale list requires one literal scale per actual reference.')
        # The official prior scales supplied pooled embeddings in-place. The
        # stage may not mutate a connected upstream material/cache output.
        input_bytes = sum(extras[key].numel() * extras[key].element_size()
            for key in ('prompt_embeds', 'pooled_prompt_embeds') if key in extras)
        if input_bytes > _MAX_MATERIAL_BYTES:
            raise ValueError('Supplied prior conditioning exceeds the bounded stage material reserve.')
        for key in ('prompt_embeds', 'pooled_prompt_embeds'):
            if key in extras:
                extras[key] = extras[key].clone()
        apply_call_inputs(extras, call)
        # This is the actual official prior, only in Encode Inputs. Its own
        # public call performs the upstream weighted sum and hook finalizer.
        result = owner.prior(**call)
        consumed = {key: call.get(key) for key in ('prompt', 'prompt_2',
            'prompt_embeds_scale', 'pooled_prompt_embeds_scale')}
        consumed.update(negative_prompt=None, reference_strength=values['reference_strength'],
            num_images_per_prompt=1, input_copy_bytes=input_bytes,
            reference_count=count, prior_max_sequence_length=512)
        return RawEncoding({'prompt_embeds': result.prompt_embeds,
            'pooled_prompt_embeds': result.pooled_prompt_embeds}, consumed, {}, {})

    def validate_denoise_values(self, constraints, values):
        if values.get('generator') is not None or values.get('latents') is not None:
            raise ValueError('The selected prior stages require the ordinary validated seed path.')

    def denoise(self, owner, encoding, values, callback_kwargs):
        return owner.base(prompt_embeds=encoding.material['prompt_embeds'],
            pooled_prompt_embeds=encoding.material['pooled_prompt_embeds'],
            width=values['width'], height=values['height'],
            num_inference_steps=values['num_inference_steps'], guidance_scale=values['guidance_scale'],
            max_sequence_length=values['max_sequence_length'],
            generator=_stage_seed_generator(owner.base, values['seed']),
            output_type='latent', return_dict=True, **callback_kwargs).images

    def postprocess(self, decoded, output_type, output_context):
        return decoded


PRIOR_IMAGE_STAGE = _PriorImageStrategy()


def strategy_for_key(key):
    strategies = {strategy.key: strategy for strategy in (PRIOR_IMAGE_STAGE, FILL_TEXT_STAGE, CONTROL_STAGE)}
    if type(key) is not str or key not in strategies:
        raise ValueError('A declared backend image-stage implementation is required.')
    return strategies[key]




def _require_split_lora_owner(owner):
    require_owner(owner, PRIOR_IMAGE_STAGE)
    if vars(owner).get('_modiff_loaded_stage_offload_mode') != 'none':
        raise ValueError('Split-owner LoRA mutation currently requires NONE placement; fresh hooked target staging is not proved.')
    contract = vars(owner).get('_modiff_stage_loading_contract')
    if (type(contract) is not tuple or len(contract) != 3
            or contract[0] != 'none' or type(contract[1]) is not str
            or contract[1] not in ('none', 'cuda') or contract[2] is not False):
        raise ValueError('Split-owner LoRA requires an actual dense NONE/single-resident loading contract.')
    _require_unhooked_split_components(owner)
    return owner.base.transformer, owner.prior.text_encoder


def _require_unhooked_split_components(owner):
    import torch
    from diffusers.hooks.hooks import HookRegistry
    # Official LoRA methods inspect and reapply hooks on each supplied owner.
    # Inspect its complete owned component trees, including VAE/T5/image encoder,
    # rather than only the two selected injection targets.
    for pipeline in (owner.base, owner.prior):
        hooks = vars(pipeline).get('_all_hooks')
        if hooks is not None and (type(hooks) not in (tuple, list) or hooks):
            raise ValueError('Split-owner LoRA refuses installed pipeline hooks.')
    seen = set()
    for _name, component in PRIOR_IMAGE_STAGE.components(owner):
        if not isinstance(component, torch.nn.Module):
            continue
        for module in torch.nn.Module.modules(component):
            if id(module) in seen:
                continue
            seen.add(id(module))
            if len(seen) > 65536:
                raise ValueError('Split-owner hook census exceeds its finite bound.')
            values = vars(module)
            if values.get('_hf_hook') is not None:
                raise ValueError('Split-owner LoRA refuses installed Accelerate hooks.')
            registry = values.get('_diffusers_hook')
            if registry is not None:
                hooks = vars(registry).get('hooks') if type(registry) is HookRegistry else None
                if type(hooks) is not dict or hooks:
                    raise ValueError('Split-owner LoRA refuses installed Diffusers hooks.')


def _split_lora_inventory(owner):
    transformer, text_encoder = _require_split_lora_owner(owner)
    inventory = vars(owner).get('_modiff_split_lora_inventory', {})
    if (type(inventory) is not dict or len(inventory) > 32
            or any(type(name) is not str or not name or len(name) > 128
                   or type(parts) is not tuple or not parts or set(parts) - {'transformer', 'text_encoder'}
                   or len(parts) != len(set(parts)) for name, parts in inventory.items())):
        raise ValueError('Split-owner adapter inventory is not bounded or complete.')
    for role, model in (('transformer', transformer), ('text_encoder', text_encoder)):
        actual = getattr(model, 'peft_config', {})
        expected = {name for name, roles in inventory.items() if role in roles}
        if type(actual) is not dict or set(actual) != expected:
            raise ValueError('Actual split-component adapter inventory changed outside the reviewed owner.')
    return inventory


def _validate_split_lora_state(owner, state):
    import torch
    from peft.tuners.tuners_utils import BaseTunerLayer
    transformer, text_encoder = _require_split_lora_owner(owner)
    if type(state) is not dict or not state or len(state) > 16384:
        raise ValueError('Split-owner LoRA requires a finite nonempty canonical tensor inventory.')
    paired = {}
    for key, value in state.items():
        if type(key) is not str or len(key) > 1024:
            raise ValueError('LoRA component keys must be bounded plain strings.')
        match = re.fullmatch(r'(transformer|text_encoder)\.(.+)\.lora_([AB])\.weight', key)
        if match is None:
            raise ValueError('Unreviewed LoRA prefix, normalization, DoRA or noncanonical component key.')
        role, path, side = match.groups()
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or value.layout != torch.strided or not value.is_floating_point()
                or value.ndim != 2 or any(size <= 0 for size in value.shape)):
            raise ValueError('LoRA weights must be materialized canonical CPU floating-point matrices.')
        if side in paired.setdefault((role, path), {}):
            raise ValueError('Duplicate LoRA matrix role.')
        paired[(role, path)][side] = value
    for (role, path), matrices in paired.items():
        if set(matrices) != {'A', 'B'}:
            raise ValueError('Every LoRA target requires its complete A/B pair.')
        component = transformer if role == 'transformer' else text_encoder
        try:
            layer = component.get_submodule(path)
        except (AttributeError, KeyError) as error:
            raise ValueError('LoRA target is absent from its actual owned component.') from error
        if isinstance(layer, BaseTunerLayer):
            layer = layer.get_base_layer()
        if type(layer) is not torch.nn.Linear:
            raise ValueError('This split-owner LoRA target is not a reviewed dense Linear layer.')
        a, b = matrices['A'], matrices['B']
        if a.shape[0] != b.shape[1] or a.shape[1] != layer.in_features or b.shape[0] != layer.out_features:
            raise ValueError('LoRA target shapes do not match; parameter expansion is not permitted.')
    return tuple(role for role in ('transformer', 'text_encoder') if any(key[0] == role for key in paired))


def prepare_split_lora(owner, pretrained_model_name_or_path_or_dict, **kwargs):
    """Called by LoadAdapter BEFORE replace/unload or any component mutation."""
    from diffusers.loaders.lora_pipeline import FluxLoraLoaderMixin
    _split_lora_inventory(owner)
    if type(pretrained_model_name_or_path_or_dict) is dict:
        if kwargs:
            raise ValueError('A preflighted LoRA tensor inventory cannot carry file or loader overrides.')
        state = dict(pretrained_model_name_or_path_or_dict)
    else:
        allowed = {'weight_name', 'use_safetensors', 'adapter_name'}
        if set(kwargs) - allowed or kwargs.get('use_safetensors') is not True:
            raise ValueError('Split-owner LoRA accepts only the reviewed local Safetensors loader options.')
        # Existing LoadAdapter has already resolved containment and SHA256.
        # Only the original official parser is used; never fetch a Hub file.
        state, alphas, metadata = FluxLoraLoaderMixin.lora_state_dict(
            pretrained_model_name_or_path_or_dict, weight_name=kwargs.get('weight_name'),
            use_safetensors=True, local_files_only=True, return_alphas=True,
            return_lora_metadata=True)
        if alphas or metadata:
            raise ValueError('Split-owner LoRA alpha/metadata variants need a separately reviewed contract.')
    _validate_split_lora_state(owner, state)
    return state


def load_split_lora(owner, state, *, adapter_name):
    from diffusers.loaders.lora_pipeline import FluxLoraLoaderMixin
    inventory = _split_lora_inventory(owner)
    roles = _validate_split_lora_state(owner, state)
    if (type(adapter_name) is not str or not adapter_name or len(adapter_name) > 128
            or adapter_name in inventory or len(inventory) >= 32):
        raise ValueError('LoRA adapter name already exists, is malformed or exceeds the owned capacity.')
    previous = begin_mutation(owner)
    # Partial official failure remains poisoned until complete known unload or
    # a new successful loader. No same-pointer cache rebind repairs the token.
    if 'transformer' in roles:
        FluxLoraLoaderMixin.load_lora_into_transformer(state, network_alphas=None,
            transformer=owner.base.transformer, adapter_name=adapter_name,
            metadata=None, _pipeline=owner.base, low_cpu_mem_usage=False)
    if 'text_encoder' in roles:
        FluxLoraLoaderMixin.load_lora_into_text_encoder(state, network_alphas=None,
            text_encoder=owner.prior.text_encoder, prefix='text_encoder', lora_scale=1.,
            adapter_name=adapter_name, metadata=None, _pipeline=owner.prior, low_cpu_mem_usage=False)
    owner._modiff_split_lora_inventory = {**inventory, adapter_name: roles}
    finish_mutation(owner, previous, PRIOR_IMAGE_STAGE)
    _split_lora_inventory(owner)


def set_split_adapters(owner, adapter_names, adapter_weights):
    from diffusers.loaders.lora_base import set_adapters_for_text_encoder
    inventory = _split_lora_inventory(owner)
    if (type(adapter_names) is not list or type(adapter_weights) is not list
            or len(adapter_names) != len(adapter_weights) or not 1 <= len(adapter_names) <= 32
            or any(type(name) is not str for name in adapter_names)
            or len(adapter_names) != len(set(adapter_names)) or set(adapter_names) != set(inventory)
            or any(type(weight) not in (int, float) or not math.isfinite(weight) or not -2 <= weight <= 2
                   for weight in adapter_weights)):
        raise ValueError('Ordered split-owner adapters require the complete actual inventory and bounded scalar weights.')
    previous = begin_mutation(owner)
    for role, component in (('transformer', owner.base.transformer), ('text_encoder', owner.prior.text_encoder)):
        pairs = [(name, scale) for name, scale in zip(adapter_names, adapter_weights, strict=True)
                 if role in inventory[name]]
        if not pairs:
            continue
        names, scales = map(list, zip(*pairs, strict=True))
        if role == 'transformer':
            component.set_adapters(names, scales)
        else:
            set_adapters_for_text_encoder(names, component, scales)
    owner._modiff_adapter_scales = dict(zip(adapter_names, adapter_weights, strict=True))
    finish_mutation(owner, previous, PRIOR_IMAGE_STAGE)


def unload_split_lora(owner):
    from diffusers.loaders.lora_base import _remove_text_encoder_monkey_patch
    # A failed partial mutation can be recovered only by actual complete unload,
    # never by ignoring the incomplete target inventory or changing graph hints.
    current = vars(owner).get('_modiff_image_stage_owner')
    previous = current if current is not None else vars(owner).get('_modiff_stage_mutation_record')
    if previous is None:
        # Historically loaded whole Redux bundles exposed no adapter API.
        # Their ordinary no-adapter reuse must remain a no-op, without a grant.
        if (getattr(owner.base.transformer, 'peft_config', {})
                or getattr(owner.prior.text_encoder, 'peft_config', {})):
            raise ValueError('Split-owner adapter reset lacks its successful loader record.')
        return
    if current is not None and vars(owner).get('_modiff_stage_mutation_failed') is not True:
        require_owner(owner, PRIOR_IMAGE_STAGE, current)
        actual = (getattr(owner.base.transformer, 'peft_config', {}),
                  getattr(owner.prior.text_encoder, 'peft_config', {}),
                  vars(owner).get('_modiff_split_lora_inventory', {}),
                  vars(owner).get('_modiff_adapter_scales', {}))
        if all(type(value) is dict and not value for value in actual):
            # No physical adapter mutation is needed. Retain the same complete
            # epoch and hooks at every already-supported loader placement.
            return
    PRIOR_IMAGE_STAGE.validate_current_owner(owner, previous.selection)
    if vars(owner).get('_modiff_loaded_stage_offload_mode') != 'none':
        raise ValueError('Split-owner LoRA reset requires NONE placement.')
    contract = vars(owner).get('_modiff_stage_loading_contract')
    if (type(contract) is not tuple or len(contract) != 3
            or contract[0] != 'none' or type(contract[1]) is not str
            or contract[1] not in ('none', 'cuda') or contract[2] is not False):
        raise ValueError('Split-owner LoRA reset requires an actual dense NONE/single-resident loading contract.')
    _require_unhooked_split_components(owner)
    begin_mutation(owner)
    owner.base.unload_lora_weights()
    _remove_text_encoder_monkey_patch(owner.prior.text_encoder)
    if getattr(owner.base.transformer, 'peft_config', {}) or getattr(owner.prior.text_encoder, 'peft_config', {}):
        raise ValueError('Split-owner LoRA reset did not clear both actual component inventories.')
    owner._modiff_split_lora_inventory = {}
    owner._modiff_adapter_scales = {}
    finish_mutation(owner, previous, PRIOR_IMAGE_STAGE)
