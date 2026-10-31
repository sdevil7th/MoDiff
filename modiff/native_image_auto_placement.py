"""Declared native image scope for the existing Auto planners."""
from __future__ import annotations

from modiff.studio_execution_specs import studio_capability_definition, studio_execution_spec_for_pair


def declared_image_owner(profile, repository, mode, *, capability=None):
    """Inspect reviewed media metadata, never class names or executable suppliers."""
    if capability is None:
        capability = studio_capability_definition(profile.model_type, repository=repository)
    if capability:
        return (capability.get('modeOutputKinds') or {}).get(
            mode, capability.get('outputKind') or 'image',
        ) == 'image'
    spec = studio_execution_spec_for_pair(profile.model_type, mode, execution_profile_id=profile.id)
    repositories = {getattr(profile, 'default_repo', None), getattr(profile, 'fallback_repo', None),
                    *getattr(profile, 'compatible_repos', ())}
    if (not spec or spec['executionProfileId'] != profile.id or repository not in repositories
            or spec['loaderModule'] != profile.loader_module or spec['loaderAction'] != profile.loader_action
            or spec['executionPath'] != profile.execution_path):
        return False
    consumed = {edge[0] for edge in spec['edges']}
    terminals = {role[1] for role in spec['roles'] if role[0] not in consumed}
    return terminals == {'modules.Image.Preview'}
