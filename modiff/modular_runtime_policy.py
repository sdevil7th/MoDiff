"""Pure native loader policy shared by inspection and component ownership.

Keep this module free of model-library imports: Auto inspects controls before
the executor is allowed to load weights or execute data suppliers.
"""

RUNTIME_POLICY_FIELDS = ('attention_backend', 'vae_slicing', 'vae_tiling', 'inpaint_compatibility', 'weight_loading_mode')


def normalize_modular_runtime_policy(
    attention_backend='inherit', vae_slicing=None, vae_tiling=None, inpaint_compatibility='native',
    weight_loading_mode='cpu',
):
    """Missing native controls inherit; explicit False remains a distinct policy."""
    if attention_backend is None or attention_backend == '':
        attention_backend = 'inherit'
    if not isinstance(attention_backend, str):
        raise TypeError('Native Modular attention must be a declared backend string.')
    if attention_backend not in {'inherit', 'auto', 'native', '_native_math'}:
        raise ValueError('Native Modular attention must be inherit, auto, native, or _native_math.')
    for name, value in (('vae_slicing', vae_slicing), ('vae_tiling', vae_tiling)):
        if value is not None and type(value) is not bool:
            raise TypeError(f'{name} must be an explicit boolean or unset.')
    if type(inpaint_compatibility) is not str or inpaint_compatibility not in {'native', 'whole_v1'}:
        raise ValueError("Qwen inpaint compatibility must be 'native' or 'whole_v1'.")
    policy = {'attention_backend': attention_backend, 'vae_slicing': vae_slicing, 'vae_tiling': vae_tiling}
    if inpaint_compatibility == 'whole_v1':
        policy['inpaint_compatibility'] = 'whole_v1'
    from modiff.native_weight_loading import normalize_weight_loading_mode

    # Missing/default CPU loading keeps historical ownership identities.
    if normalize_weight_loading_mode(weight_loading_mode) == 'cuda':
        policy['weight_loading_mode'] = 'cuda'
    return policy


def runtime_policy_from_settings(settings):
    """Project a plan/diagnostic onto the same policy the native loader uses."""
    return normalize_modular_runtime_policy(**{
        field: settings[field] for field in RUNTIME_POLICY_FIELDS if field in settings
    })
