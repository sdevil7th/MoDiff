"""Campaign-quality prompts and params for review generation.

Public Studio templates already use campaign briefs (camera, lighting, identity
lock, materials, negatives). Hidden-candidate authoring drafts are bounded to
700 characters. This module overlays richer original MoDiff briefs at submit
time so generated review assets match that public-template bar.

The Comfy research catalog is titles and tags only. This module compares
complexity against that catalog and against public TEMPLATE_PROMPTS. It never
imports or executes Comfy graphs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from modiff.local_review_receipts import canonical_content_hash


SCHEMA_VERSION = 1
COMFY_CATALOG_PATH = Path("data/research/comfy-workflow-catalog.v1.json")
PUBLIC_TEMPLATES_PATH = Path("src/studio/templates.ts")
AUTHORING_SPEC_PATH = Path("data/template-authoring-specs.v1.json")
_BOUNDARY = {
    "importsComfyGraphs": False,
    "executesComfyNodes": False,
    "galleryRegistered": False,
    "datasetPublished": False,
    "publicTemplateMutated": False,
    "authoringSpecMutated": False,
}

_NEGATIVE_IMAGE = (
    "low resolution, distorted geometry, duplicate subjects, malformed hands, inconsistent reflections, "
    "watermark, logo, illegible text, oversharpening, compression artifacts, plastic materials, "
    "toy-like proportions, neon glow, floating objects, extra limbs, identity drift"
)
_NEGATIVE_VIDEO = (
    "flicker, frame jitter, temporal inconsistency, warped motion, duplicate subjects, identity drift, "
    "camera jumps, watermark, illegible text, compression artifacts, sudden scene changes, melted geometry"
)
_NEGATIVE_AUDIO = (
    "clipping, abrupt cuts, distorted vocals, unintelligible lyrics, harsh digital aliasing, "
    "unresolved ending, noisy mix, competing lead instruments"
)

# Public-template class briefs: camera, lighting, materials, identity, finish.
_TEXT_TO_IMAGE = (
    (
        "Campaign objective: create a finished editorial product photograph of one unbranded translucent "
        "cobalt glass table radio on pale limestone. Product identity: compact mid-century shell with "
        "rounded corners, one large analog tuning dial, a woven speaker grille, four small rubber feet, "
        "and no invented controls, branding, or lettering. Composition and camera: natural eye-level 50 mm "
        "three-quarter view, complete object in frame, sharp grille and dial microdetail, correct contact "
        "shadow, generous editorial negative space. Lighting and finish: soft north-window key, weak warm "
        "bounce, glass wall thickness, restrained blue-and-amber palette, limestone pores, fine film grain, "
        "physically plausible reflections, and no cinematic fantasy glow."
    ),
    (
        "Campaign objective: photograph a quiet rain-soaked tram stop at blue hour as a documentary still. "
        "Show one red umbrella, wet pavement reflections, a timber shelter, a single waiting figure in a "
        "navy coat with no readable logos, and layered urban depth. Camera: 35 mm eye-level, straight "
        "verticals, complete shelter in frame, rain beads on metal and glass. Light the scene only with "
        "cool overcast sky, one warm practical under the shelter, and restrained sodium bounce from the "
        "street. Preserve believable wet asphalt, fabric weight, window reflections, atmospheric haze, and "
        "fine grain. No brand marks, extra people, neon nightclub color, or illustration."
    ),
    (
        "Campaign objective: create a museum-catalog botanical field-study plate of six imaginary alpine "
        "flowers arranged on warm archival paper. Draw delicate ink contours with subtle watercolor washes, "
        "clear specimen spacing, and a quiet scientific hierarchy without any printed labels, numbers, or "
        "captions. Camera: overhead 85 mm copy-stand view, paper filling the frame with generous margins. "
        "Use north-window daylight, gentle warm bounce from the table, visible paper fibers, believable "
        "pigment granulation, and restrained umber-sage-cream palette. Keep every stem physically attached, "
        "shadows consistent with a single window, and the composition original. No watermarks, logos, "
        "synthetic neon, floating petals, or crowded collage edges."
    ),
    (
        "Campaign objective: photograph a sunlit modern reading room carved into warm sandstone with linen "
        "seating, one mature olive tree, long geometric shadows, and calm human scale. Camera: 24 mm "
        "architectural three-quarter view, straight verticals, complete seating group, tactile material "
        "detail in stone, linen, and leaves. Light only with late-morning sun from camera left and soft "
        "sky fill. Preserve sandstone pores, fabric weave, olive-leaf translucency, accurate contact "
        "shadows, and quiet editorial negative space. No people, signage, extra furniture, floating lamps, "
        "warped architecture, or fantasy glow."
    ),
    (
        "Campaign objective: create a macro photograph of a handcrafted miniature harbor town inside an "
        "open walnut music box. Show tiny wooden boats, a lighthouse, painted quays, and precise physical "
        "joinery. Camera: 100 mm macro, shallow depth of field, complete music-box interior, sharp "
        "foreground boats with gentle falloff. Light with warm practicals inside the box and a soft cool "
        "rim from outside. Preserve walnut grain, paint chips, dust in the air, grounded shadows, and "
        "intricate handmade detail. No branding, giant figures, melted miniatures, missing lid, or CGI gloss."
    ),
    (
        "Campaign objective: make a documentary portrait of an elderly bicycle mechanic in a compact "
        "workshop. Identity lock: weathered hands, honest expression, grey stubble, faded chambray shirt, "
        "leather apron, no logos. Environment: crowded timber bench, worn tools, a half-rebuilt wheel, "
        "window light from camera left. Camera: 85 mm, waist-up, natural skin texture, shallow but "
        "readable background. Preserve tool wear, oil stains, fabric tooth, and unobtrusive composition. "
        "No beauty-retouch plastic skin, extra people, readable posters, extra limbs, or cinematic smoke."
    ),
)
_TEXT_TO_VIDEO = (
    (
        "A slow cinematic dolly through a quiet greenhouse just after rain. Droplets slide from broad "
        "leaves, morning mist catches the light, and condensation beads remain physically coherent on glass. "
        "Camera: smooth 35 mm dolly-in at chest height, stable horizon, no cuts. Light with soft overcast "
        "skylight and faint warm bounce from brick. Keep plant identity, leaf scale, and water motion "
        "continuous for the full shot. No flicker, jump cuts, extra people, or melting geometry."
    ),
    (
        "Wide coastal grassland at dusk as wind moves in visible waves across the grass. A lone cyclist "
        "crosses the frame once, clouds drift naturally, and exposure stays stable. Camera: locked-off 24 mm "
        "establishing view with a very slow pan, continuous motion, no teleporting. Preserve believable "
        "cloth, wheel rotation, horizon, and gradual light change. No flicker, duplicate cyclists, or "
        "sudden weather resets."
    ),
    (
        "Close-up of a ceramic artist shaping a bowl on a spinning wheel. Hands move deliberately, wet clay "
        "retains its form, water catches the rim, and the camera makes a subtle arc. Keep identity, "
        "workshop background, and wheel speed temporally consistent. Soft side light, practical warmth, "
        "no jump cuts, no extra limbs, no clay that ignores gravity."
    ),
    (
        "A small research boat moving through calm arctic water beneath low fog. Ice fragments drift slowly, "
        "soft light changes gradually, and the horizon remains stable. Camera: gentle 50 mm lateral drift, "
        "continuous wake, no cuts. Preserve hull identity, fog density, and physically plausible water. No "
        "flicker, teleporting ice, or sudden sunbursts."
    ),
)
_TEXT_TO_AUDIO = (
    (
        "Instrumental downtempo electronic piece with brushed drums, warm analog bass, soft granular "
        "textures, and a restrained melodic arc. Mix should stay clean, develop gradually over a verse-like "
        "middle, and resolve without an abrupt cut. No vocals, no sirens, no unintelligible speech."
    ),
    (
        "Intimate chamber-folk cue led by fingerpicked acoustic guitar, muted cello, light hand percussion, "
        "and room ambience. Natural dynamics, a memorable four-bar motif, and a gentle ending. No electric "
        "distortion, no sudden loudness jumps, no crowd noise."
    ),
    (
        "Atmospheric science-documentary score with glassy mallets, low strings, subtle pulses, and spacious "
        "reverberation. Evolving structure, no abrupt cuts, controlled loudness, and a resolved final chord. "
        "No vocals, no trailer brass hits, no clipping."
    ),
)

_WORKFLOW_BRIEFS = {
    "CogView4Pipeline:text_to_image": {
        "prompt": (
            "Natural product photograph of one small dark-green enamel kettle on a worn oak table beside a window. "
            "Complete kettle, curved handle, short spout, fitted lid, subtle chips in the enamel, soft morning side light, "
            "realistic contact shadow, muted brown and green colors. No people, cups, food, text, labels, logos, extra "
            "kettles, reflections of people, neon, or illustration."
        ),
        "negativePrompt": (
            "person, people, hand, cup, food, text, letters, label, logo, watermark, extra kettle, duplicate object, "
            "cropped kettle, detached handle, warped spout, floating object, neon, oversaturation, illustration"
        ),
        "provenance": "original_modiff_campaign_brief_cogview4_kettle_v1",
    },
    "DreamLiteMobilePipeline:text_to_image": {
        "prompt": (
            "Architectural photograph of a quiet sandstone courtyard with one linen lounge chair beneath one mature olive "
            "tree. Empty scene, warm late-morning sunlight, straight walls, clean geometry, realistic stone texture, grounded "
            "contact shadows, muted sand and sage colors. No people, additional chairs, text, signs, warped furniture, sand "
            "dunes, or fantasy elements."
        ),
        "negativePrompt": (
            "person, people, crowd, extra chair, multiple chairs, text, letters, sign, logo, watermark, warped furniture, "
            "crooked walls, floating objects, sand dunes, desert, overexposure, neon, illustration, fantasy"
        ),
        "provenance": "original_modiff_campaign_brief_dreamlite_mobile_courtyard_v1",
    },
    "FluxSchnellPipeline:text_to_image": {
        "prompt": (
            "Natural photograph of one weathered stone lantern beside a shallow forest stream after rain. Complete lantern, "
            "moss texture, wet river stones, fern leaves, soft fog, gentle overcast light, shallow focus, muted green and gray "
            "colors. No people, buildings, text, symbols, extra lanterns, frames, neon, or fantasy glow."
        ),
        "negativePrompt": (
            "person, people, building, temple, text, letters, symbols, inscription, logo, watermark, extra lantern, duplicate "
            "object, cropped lantern, warped stone, floating object, frame, border, neon, fantasy, blur, overexposure"
        ),
        "provenance": "original_modiff_campaign_brief_flux_schnell_lantern_v1",
    },
    "GlmImagePipeline:text_to_image": {
        "prompt": (
            "Natural photograph of one open red umbrella resting upright on rain-wet cobblestones beneath a simple iron "
            "streetlamp at blue hour. Complete umbrella, curved wooden handle, dark stone wall, soft warm lamp light, "
            "realistic rain beads and reflections, muted red, blue, and gray colors, quiet empty scene. No people, vehicles, "
            "shops, signs, text, logos, extra umbrellas, neon, or illustration."
        ),
        "negativePrompt": (
            "person, people, hand, vehicle, shop, sign, text, letters, logo, watermark, extra umbrella, duplicate object, "
            "cropped umbrella, broken handle, warped canopy, floating object, neon, oversaturation, overexposure, illustration"
        ),
        "provenance": "original_modiff_campaign_brief_glm_umbrella_v1",
    },
    "FluxDevPipeline:text_to_image": {
        "prompt": (
            "Natural photograph of one small weathered red wooden shed in a green meadow after rain. Complete shed, one "
            "closed door, simple pitched roof, peeling red paint, wet grass, a few gray stones, distant pine trees, soft "
            "overcast light, realistic contact shadows, muted natural colors. No people, vehicles, signs, text, logos, "
            "extra buildings, neon, or fantasy elements."
        ),
        "negativePrompt": (
            "person, people, vehicle, sign, text, letters, logo, watermark, extra building, duplicate shed, cropped shed, "
            "open doorway, warped roof, floating object, neon, oversaturation, overexposure, illustration, fantasy"
        ),
        "provenance": "original_modiff_campaign_brief_flux_dev_shed_v1",
    },
    "LatentConsistencyModelPipeline:edit_image": {
        "prompt": (
            "Photorealistic blue tabletop radio matching the source: one rounded body, horizontal grille, one plain amber "
            "dial, two short feet, centered on beige. Blank front except grille and dial. Warm window light, matte enamel. "
            "No markings, text, logos, numbers, extra controls, second dial, people, frames, or extra objects."
        ),
        "negativePrompt": (
            "person, human, portrait, figure, character, text, letters, words, label, caption, signature, logo, watermark, "
            "screen, picture frame, card, poster, extra radio, extra dial, detached parts, cropped object, neon, distorted geometry"
        ),
        "provenance": "original_modiff_campaign_brief_lcm_edit_v3",
    },
    "LatentConsistencyModelPipeline:text_to_image": {
        "prompt": (
            "Close photograph of two cream alpine flowers growing together from dark rocky soil. Show short green stems "
            "and attached leaves. Soft dawn light, shallow focus, crisp petals, muted sage and umber, natural soil contact, "
            "quiet mountain background. No vase, frame, paper, card, text, labels, logo, detached petals, or extra "
            "flowers."
        ),
        "negativePrompt": (
            "text, letters, words, labels, numbers, signature, logo, watermark, paper, card, floral border, wreath, "
            "picture frame, vase, floating petals, detached leaves, duplicate blooms, extra flowers, cropped bloom, "
            "neon, distorted geometry"
        ),
        "provenance": "original_modiff_campaign_brief_v6",
    },
    "JoyImageEditPipeline:text_to_image": {
        "prompt": (
            "Natural product photograph of one cream ceramic table lamp on a dark walnut side table against a plain warm "
            "gray wall. Complete lamp, rounded base, simple linen shade, switched on, soft amber glow, realistic contact "
            "shadow, visible ceramic and fabric texture, quiet uncluttered scene. No people, books, flowers, text, labels, "
            "logos, extra lamps, neon, or illustration."
        ),
        "negativePrompt": (
            "person, people, hand, book, flower, text, letters, label, logo, watermark, extra lamp, duplicate object, "
            "cropped lamp, detached shade, warped table, floating object, neon, oversaturation, overexposure, illustration"
        ),
        "provenance": "original_modiff_campaign_brief_joyimage_lamp_v1",
    },
    "LuminaPipeline:text_to_image": {
        "prompt": (
            "Natural product photograph of one shallow cobalt-blue ceramic bowl on folded beige linen beside a window. "
            "Complete bowl, smooth round rim, subtle handmade glaze, empty interior, soft morning side light, realistic "
            "contact shadow, muted blue and warm neutral colors, quiet background. No people, hands, food, flowers, text, "
            "labels, logos, extra bowls, neon, or illustration."
        ),
        "negativePrompt": (
            "person, people, hand, food, flower, text, letters, label, logo, watermark, extra bowl, duplicate object, "
            "cropped bowl, warped rim, floating object, neon, oversaturation, overexposure, illustration"
        ),
        "provenance": "original_modiff_campaign_brief_lumina_bowl_v1",
    },
    "OmniGenPipeline:text_to_image": {
        "prompt": (
            "Natural photograph of one mustard-yellow canvas backpack resting on a simple gray wooden bench beside a "
            "plaster wall. Complete backpack, closed flap, two shoulder straps, visible canvas weave, soft afternoon side "
            "light, realistic contact shadow, muted yellow and gray colors, quiet uncluttered scene. No people, hands, "
            "clothing, text, labels, logos, extra bags, neon, or illustration."
        ),
        "negativePrompt": (
            "person, people, hand, clothing, text, letters, label, logo, watermark, extra bag, duplicate backpack, cropped "
            "backpack, detached strap, warped bench, floating object, neon, oversaturation, overexposure, illustration"
        ),
        "provenance": "original_modiff_campaign_brief_omnigen_backpack_v1",
    },
    "SanaPAGPipeline:text_to_image": {
        "prompt": (
            "Natural photograph of one small red canvas tent in a quiet green meadow at sunrise. Complete tent, taut "
            "fabric, short grass with dew, a few gray stones, distant pine trees, pale mist, soft golden side light, "
            "realistic shadows, muted natural colors. No people, campfire, vehicles, buildings, signs, text, logos, "
            "extra tents, neon, or fantasy elements."
        ),
        "negativePrompt": (
            "person, people, campfire, vehicle, building, cabin, text, letters, sign, logo, watermark, extra tent, "
            "duplicate tent, cropped tent, warped fabric, floating objects, neon, oversaturation, illustration, fantasy"
        ),
        "provenance": "original_modiff_campaign_brief_sana_pag_tent_v1",
    },
    "SanaSprintPipeline:text_to_image": {
        "prompt": (
            "Natural photograph of one mustard-yellow raincoat hanging from a single wooden peg in a quiet mudroom. "
            "Complete raincoat, closed front, visible hood and sleeves, waxed fabric texture, worn oak wall, slate floor, "
            "soft overcast window light, realistic folds and shadow, muted yellow and brown colors. No people, hands, "
            "boots, umbrellas, text, labels, logos, extra coats, neon, or illustration."
        ),
        "negativePrompt": (
            "person, people, hand, boots, umbrella, text, letters, label, logo, watermark, extra coat, duplicate raincoat, "
            "cropped garment, missing sleeve, detached hood, floating clothing, neon, oversaturation, overexposure, illustration"
        ),
        "provenance": "original_modiff_campaign_brief_sana_sprint_raincoat_v1",
    },
    "StableDiffusionPipeline:text_to_image": {
        "prompt": (
            "Documentary photograph of one red wooden canoe on a pebble shore at dawn. Complete canoe, wet stones, lake "
            "mist, dark pine forest, distant mountains, natural reflections, soft overcast light, subdued blue and rust "
            "colors. No people, buildings, signs, text, logos, frames, cards, or extra boats."
        ),
        "negativePrompt": (
            "text, letters, words, numbers, sign, signature, logo, watermark, frame, border, card, poster, people, person, "
            "extra canoe, duplicate boat, cropped canoe, floating boat, distorted hull, neon, illustration, oversaturated"
        ),
        "provenance": "original_modiff_campaign_brief_sd15_canoe_v1",
    },
    "ZImageModularPipeline:text_to_image": {
        "prompt": (
            "Natural photograph of one weathered terracotta watering can resting on a worn greenhouse potting bench. "
            "Complete can, single curved handle, long narrow spout, matte clay texture, a few water droplets, old timber, "
            "soft leafy background, diffused morning light, realistic contact shadow, muted earth and green colors. No "
            "people, hands, flowers, text, labels, logos, extra cans, neon, or illustration."
        ),
        "negativePrompt": (
            "person, people, hand, flower, text, letters, label, logo, watermark, extra watering can, duplicate object, "
            "cropped can, missing handle, broken spout, warped vessel, floating object, neon, oversaturation, illustration"
        ),
        "provenance": "original_modiff_campaign_brief_z_image_watering_can_v1",
    },
}

_WORKFLOW_FORM_OVERRIDES = {
    "CogView4Pipeline:text_to_image": {"seed": 137},
    "DreamLiteMobilePipeline:text_to_image": {"seed": 137},
    "FluxSchnellPipeline:text_to_image": {"seed": 137},
    "GlmImagePipeline:text_to_image": {"steps": 50, "seed": 137},
    "FluxDevPipeline:text_to_image": {
        "steps": 20,
        "width": 768,
        "height": 768,
        "guidanceScale": 3.5,
        "seed": 137,
    },
    "LatentConsistencyModelPipeline:edit_image": {"strength": 0.4},
    "LatentConsistencyModelPipeline:text_to_image": {"seed": 137},
    "JoyImageEditPipeline:text_to_image": {"seed": 137},
    "LuminaPipeline:text_to_image": {"seed": 137},
    "OmniGenPipeline:text_to_image": {"seed": 137},
    "SanaPAGPipeline:text_to_image": {"guidanceScale": 4.5, "seed": 137},
    "SanaSprintPipeline:text_to_image": {"seed": 137},
    "ZImageModularPipeline:text_to_image": {"steps": 8, "guidanceScale": 1, "seed": 211},
}

_MODE_BRIEFS = {
    "character_animate": (
        "Animate the supplied character using the pose and face performances while preserving identity, "
        "clothing, proportions, background continuity, and natural temporal motion. Keep seams stable, "
        "avoid identity drift, and match the source timing exactly."
    ),
    "character_replace": (
        "Replace the performer with the supplied character while preserving the source staging, camera "
        "movement, background, timing, face performance, and clean mask boundaries. Do not restage the shot."
    ),
    "control_edit_image": (
        "Restyle the source as a refined editorial photograph while preserving subject identity and "
        "composition and following the supplied control geometry exactly. Match lighting direction, keep "
        "edges clean, and add no extra subjects or text."
    ),
    "control_image": (
        "Create a detailed cinematic scene that follows the supplied control structure precisely, with "
        "coherent lighting, realistic materials, clean edges, and no unwanted text. Preserve the control "
        "silhouette, camera height, and major spatial masses."
    ),
    "control_inpaint": (
        "Replace only the masked region, follow the supplied control geometry, and match the source "
        "perspective, lighting, texture, scale, and edge transitions. Leave every unmasked pixel conceptually "
        "unchanged."
    ),
    "control_to_video": (
        "Generate a temporally coherent cinematic shot that follows the supplied control motion and "
        "structure, with stable subjects, smooth movement, and consistent lighting. No flicker or geometry drift."
    ),
    "control_video_to_video": (
        "Transform the source video while following the control video frame by frame; preserve timing and "
        "camera motion, keep subjects stable, and avoid flicker or geometry drift."
    ),
    "edit_image": (
        "Transform the supplied image into a polished editorial scene with warmer natural light and refined "
        "materials while preserving the main subject, pose, perspective, and recognizable composition. Do not "
        "invent a second hero object or change identity."
    ),
    "image_to_text": (
        "Describe the supplied image accurately and concisely. Identify the main subject, setting, visible "
        "actions, composition, lighting, and any clearly legible text without guessing hidden details."
    ),
    "image_to_video": (
        "Animate the supplied image into a short cinematic shot with subtle camera movement, physically "
        "plausible subject motion, stable identity, coherent depth, and no sudden scene changes."
    ),
    "inpaint": (
        "Replace only the masked area with a believable matching element; preserve all unmasked pixels "
        "conceptually and match perspective, illumination, material texture, scale, and depth of field."
    ),
    "layer_decomposition": (
        "Separate the supplied image into an ordered, useful layer stack: foreground subject, distinct "
        "occluding elements, midground, and background, with clean alpha boundaries and complete visual "
        "reconstruction."
    ),
    "multi_image_reference_edit": (
        "Combine the primary subject and composition from the first reference with the material palette and "
        "supporting details from the remaining references; keep identities distinct and spatial relationships "
        "coherent."
    ),
    "outpaint": (
        "Extend the supplied image beyond its original frame with a seamless continuation of perspective, "
        "lighting, architecture or landscape, texture, and depth; do not duplicate the central subject."
    ),
    "reference_to_video": (
        "Create a coherent short cinematic shot guided by the supplied reference, preserving its subject "
        "identity, visual language, palette, and spatial relationships while adding natural motion."
    ),
    "text_generation": (
        "Write a concise production brief for a 20-second documentary shot about a community repairing a "
        "storm-damaged footbridge. Include setting, subject action, camera plan, sound cues, and a clear "
        "ending in five short bullet points."
    ),
    "text_to_3d": (
        "A compact mid-century table radio with rounded corners, a large tuning dial, woven speaker grille, "
        "four rubber feet, and clean watertight geometry suitable for a neutral turntable preview. No text, "
        "no extra parts, no non-manifold surfaces."
    ),
    "video_to_video": (
        "Restyle the supplied video as a restrained cinematic documentary while preserving timing, camera "
        "motion, subject identity, scene layout, and temporal continuity."
    ),
}

_BUILTIN_PREFIXES = ("Builtin",)
_TEXT_MODES = {"text_to_image", "text_to_video", "text_to_audio"}


class GenerationCampaignBriefError(ValueError):
    """Raised when campaign briefs or complexity evidence cannot be built fail-closed."""


def _word_count(value: str) -> int:
    return len(re.findall(r"[A-Za-z0-9’'-]+", value))


def _stable_pick(key: str, values: tuple[str, ...]) -> str:
    import hashlib

    index = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % len(values)
    return values[index]


def brief_for(*, workflow_id: str, mode: str) -> dict[str, Any] | None:
    """Return an overlay brief for a prompt-bearing mode, or None when not applicable."""

    workflow_brief = _WORKFLOW_BRIEFS.get(workflow_id)
    if workflow_brief is not None:
        prompt = workflow_brief["prompt"]
        negative = workflow_brief["negativePrompt"]
        provenance = workflow_brief["provenance"]
    elif mode == "text_to_image":
        prompt = _stable_pick(workflow_id, _TEXT_TO_IMAGE)
        negative = _NEGATIVE_IMAGE
    elif mode == "text_to_video":
        prompt = _stable_pick(workflow_id, _TEXT_TO_VIDEO)
        negative = _NEGATIVE_VIDEO
    elif mode == "text_to_audio":
        prompt = _stable_pick(workflow_id, _TEXT_TO_AUDIO)
        negative = _NEGATIVE_AUDIO
    elif mode in _MODE_BRIEFS:
        prompt = _MODE_BRIEFS[mode]
        negative = _NEGATIVE_IMAGE if "image" in mode or mode.endswith("_edit") else ""
        if "video" in mode:
            negative = _NEGATIVE_VIDEO
    else:
        return None
    return {
        "prompt": prompt,
        "negativePrompt": negative,
        "provenance": provenance if workflow_brief is not None else "original_modiff_campaign_brief_v1",
        "wordCount": _word_count(prompt),
        "characterCount": len(prompt),
    }


def _authoring_defaults() -> dict[tuple[str, str], dict[str, int | float]]:
    path = Path(__file__).resolve().parents[1] / AUTHORING_SPEC_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    defaults: dict[tuple[str, str], dict[str, int | float]] = {}
    for spec in payload.get("specifications") or []:
        if not isinstance(spec, dict):
            continue
        model_type = spec.get("modelType")
        mode = spec.get("mode")
        if not isinstance(model_type, str) or not isinstance(mode, str):
            continue
        fields = (spec.get("defaultsPlan") or {}).get("fields") or []
        captured: dict[str, int | float] = {}
        for field in fields:
            if not isinstance(field, dict):
                continue
            parameter = field.get("parameter")
            value = field.get("value")
            if parameter == "num_inference_steps" and isinstance(value, int) and value > 0:
                captured["steps"] = value
            elif parameter in {"width", "height"} and isinstance(value, int) and value > 0:
                captured[parameter] = value
            elif (
                parameter == "guidance_scale"
                and not isinstance(value, bool)
                and isinstance(value, (int, float))
                and value >= 0
            ):
                captured["guidanceScale"] = value
        if captured:
            defaults[(model_type, mode)] = captured
    return defaults


_SHARED_MEMORY_T2I_CAP_FAMILIES = (
    "joyimage",
    "qwenimage",
    "lumina",
    "omnigen",
    "hunyuanimage",
    "ovisimage",
    "sana",
    "pixart",
)


def param_overlay_for(mode: str, model_type: str) -> dict[str, Any]:
    """Stronger generation settings that still respect native few-step recipes."""

    family = model_type.lower()
    overlay: dict[str, Any] = {"resourceMode": "auto", "randomSeed": False, "seed": 42}
    if "animatelcm" in family:
        # AnimateLCMPipeline matches the generic "lcm" token but rejects guidance > 2.
        overlay.update({"steps": 6, "guidanceScale": 1.5})
    elif "ltx" in family and "ltx2" not in family:
        # Cached LTX 0.9.8 13B distilled checkpoint requires exactly 8 steps and CFG 1.
        overlay.update({"steps": 8, "guidanceScale": 1.0})
    elif "sanavideo" in family or "sanaimagetovideo" in family:
        overlay.update({"steps": 30, "width": 832, "height": 480})
    elif "wanti2v" in family:
        # 5B TI2V is ~3 min/step on this APU; 30 steps exceeds the 40-minute job ceiling.
        overlay.update({"steps": 8})
    elif "latentconsistency" in family or "lcm" in family:
        overlay.update({"steps": 4, "width": 768, "height": 768, "guidanceScale": 8.5})
    elif "fluxschnell" in family:
        # FLUX.1 Schnell is distilled for one-to-four steps and guidance 0.
        overlay.update({"steps": 4, "width": 1024, "height": 1024, "guidanceScale": 0.0})
    elif "turbo" in family or "sprint" in family or "dreamlitemobile" in family or "ernieimage" in family:
        overlay.update({"steps": 4, "width": 768, "height": 768})
    elif mode == "text_to_image":
        overlay.update({"steps": 28, "width": 1024, "height": 1024, "guidanceScale": 7.5})
    elif mode == "text_to_video":
        overlay.update({"steps": 30})
    elif mode == "text_to_audio":
        overlay.update({"steps": 50})
    elif mode == "audio_repaint":
        # Campaign fixture tone_a.wav is 2s; Studio graph defaults repaintingEnd to 10s.
        overlay.update({"repaintingStart": 0, "repaintingEnd": 1.5})
    elif mode in {
        "edit_image",
        "control_image",
        "control_edit_image",
        "control_inpaint",
        "inpaint",
        "outpaint",
        "multi_image_reference_edit",
    }:
        # Flux Canny/Depth/Fill authoring graphs use guidance 30; DiffusersImage caps at 20.
        overlay.update({"steps": 28, "guidanceScale": 7.0})
    if "acestep" in family or "longcataudio" in family:
        # Native flash SDPA rejects ACE-Step / LongCat AudioDiT attention masks.
        overlay["attentionBackend"] = "_native_math"
    native = _authoring_defaults().get((model_type, mode), {})
    if "steps" in native and isinstance(overlay.get("steps"), int):
        overlay["steps"] = min(overlay["steps"], native["steps"])
    if "width" in native:
        overlay["width"] = native["width"]
    if "height" in native:
        overlay["height"] = native["height"]
    if "guidanceScale" in native:
        overlay["guidanceScale"] = native["guidanceScale"]
    # Multi-shard transformers complete denoise then SIGKILL on VAE decode at 1024
    # on shared-memory APUs. Cap after native defaults so authoring 1024 cannot win.
    # Do not apply to video families: SanaVideoPipeline matches "sana" but requires 832x480.
    if mode in {
        "text_to_image",
        "edit_image",
        "control_image",
        "control_edit_image",
        "control_inpaint",
        "inpaint",
        "outpaint",
        "multi_image_reference_edit",
    } and any(token in family for token in _SHARED_MEMORY_T2I_CAP_FAMILIES):
        overlay["steps"] = min(int(overlay.get("steps") or 16), 16)
        overlay["width"] = min(int(overlay.get("width") or 512), 512)
        overlay["height"] = min(int(overlay.get("height") or 512), 512)
    if mode in {
        "edit_image",
        "control_image",
        "control_edit_image",
        "control_inpaint",
        "inpaint",
        "outpaint",
        "multi_image_reference_edit",
    }:
        # DiffusersImage.Inpaint/Edit reject sides above 1024 (Chroma authoring uses 1344).
        overlay["width"] = min(int(overlay.get("width") or 1024), 1024)
        overlay["height"] = min(int(overlay.get("height") or 1024), 1024)
        if "flux" in family:
            # Flux control + 1024 VAE decode SIGKILLs this shared-memory APU after denoise.
            overlay["steps"] = min(int(overlay.get("steps") or 16), 16)
            overlay["width"] = min(int(overlay.get("width") or 512), 512)
            overlay["height"] = min(int(overlay.get("height") or 512), 512)
    if isinstance(overlay.get("guidanceScale"), (int, float)) and overlay["guidanceScale"] > 20:
        overlay["guidanceScale"] = 20.0
    return overlay


def campaign_document() -> dict[str, Any]:
    text_to_image = [
        {"id": f"text_to_image_{index}", "prompt": prompt, "negativePrompt": _NEGATIVE_IMAGE}
        for index, prompt in enumerate(_TEXT_TO_IMAGE)
    ]
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "generation_campaign_briefs",
        "boundary": dict(_BOUNDARY),
        "textToImage": text_to_image,
        "textToVideo": [{"prompt": prompt, "negativePrompt": _NEGATIVE_VIDEO} for prompt in _TEXT_TO_VIDEO],
        "textToAudio": [{"prompt": prompt, "negativePrompt": _NEGATIVE_AUDIO} for prompt in _TEXT_TO_AUDIO],
        "modeBriefs": {mode: brief for mode, brief in _MODE_BRIEFS.items()},
    }


def _public_template_prompt_stats(client_root: Path) -> dict[str, Any]:
    path = client_root / PUBLIC_TEMPLATES_PATH
    if not path.is_file():
        raise GenerationCampaignBriefError(f"Public templates source is missing: {path}")
    text = path.read_text(encoding="utf-8")
    start = text.find("const TEMPLATE_PROMPTS")
    if start < 0:
        raise GenerationCampaignBriefError("TEMPLATE_PROMPTS was not found in public templates.")
    chunk = text[start : start + 120_000]
    prompts = re.findall(r"'((?:\\'|[^']){120,})'", chunk)
    if len(prompts) < 8:
        raise GenerationCampaignBriefError("Could not parse enough public template prompts for comparison.")
    word_counts = sorted(_word_count(item.replace("\\'", "'")) for item in prompts)
    char_counts = sorted(len(item) for item in prompts)
    return {
        "promptCount": len(prompts),
        "wordMin": word_counts[0],
        "wordMedian": word_counts[len(word_counts) // 2],
        "wordMax": word_counts[-1],
        "characterMin": char_counts[0],
        "characterMedian": char_counts[len(char_counts) // 2],
        "characterMax": char_counts[-1],
    }


def _comfy_catalog_stats(root: Path) -> dict[str, Any]:
    path = root / COMFY_CATALOG_PATH
    if not path.is_file():
        raise GenerationCampaignBriefError(f"Comfy research catalog is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GenerationCampaignBriefError("Comfy research catalog must be an object.")
    boundary = payload.get("boundary") if isinstance(payload.get("boundary"), dict) else {}
    templates = payload.get("templates")
    if not isinstance(templates, list):
        raise GenerationCampaignBriefError("Comfy research catalog templates must be a list.")
    prompt_fields = 0
    title_words: list[int] = []
    tag_counts: list[int] = []
    for item in templates:
        if not isinstance(item, dict):
            continue
        if any(key.lower() == "prompt" or key.lower().endswith("prompt") for key in item):
            prompt_fields += 1
        title = item.get("title")
        if isinstance(title, str) and title.strip():
            title_words.append(_word_count(title))
        tags = item.get("tags")
        if isinstance(tags, list):
            tag_counts.append(len(tags))
    title_words.sort()
    return {
        "templateCount": len(templates),
        "storedPromptFieldCount": prompt_fields,
        "importsComfyGraphs": bool(boundary.get("importsComfyGraphs")),
        "titleWordMedian": title_words[len(title_words) // 2] if title_words else 0,
        "titleWordMax": title_words[-1] if title_words else 0,
        "tagCountMedian": sorted(tag_counts)[len(tag_counts) // 2] if tag_counts else 0,
        "source": payload.get("source"),
        "mappingMeaning": boundary.get("mappingMeaning"),
    }


def _authoring_prompt_stats(root: Path) -> dict[str, Any]:
    path = root / AUTHORING_SPEC_PATH
    if not path.is_file():
        return {"available": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = []
    for specification in payload.get("specifications") or []:
        prompt = (specification.get("promptPlan") or {}).get("prompt")
        if isinstance(prompt, str) and prompt:
            prompts.append(prompt)
    if not prompts:
        return {"available": True, "promptCount": 0}
    word_counts = sorted(_word_count(item) for item in prompts)
    char_counts = sorted(len(item) for item in prompts)
    return {
        "available": True,
        "promptCount": len(prompts),
        "wordMedian": word_counts[len(word_counts) // 2],
        "characterMedian": char_counts[len(char_counts) // 2],
        "characterMax": char_counts[-1],
    }


def compare_prompt_complexity(*, root: Path, client_root: Path) -> dict[str, Any]:
    """Compare campaign briefs with public templates and the Comfy catalog."""

    public = _public_template_prompt_stats(client_root)
    comfy = _comfy_catalog_stats(root)
    authoring = _authoring_prompt_stats(root)
    campaign_prompts = list(_TEXT_TO_IMAGE) + list(_TEXT_TO_VIDEO) + list(_TEXT_TO_AUDIO)
    campaign_words = sorted(_word_count(item) for item in campaign_prompts)
    campaign_chars = sorted(len(item) for item in campaign_prompts)
    campaign = {
        "promptCount": len(campaign_prompts),
        "wordMin": campaign_words[0],
        "wordMedian": campaign_words[len(campaign_words) // 2],
        "wordMax": campaign_words[-1],
        "characterMin": campaign_chars[0],
        "characterMedian": campaign_chars[len(campaign_chars) // 2],
        "characterMax": campaign_chars[-1],
    }
    meets_public_bar = campaign["wordMedian"] >= public["wordMedian"]
    exceeds_comfy_titles = campaign["wordMedian"] >= max(comfy["titleWordMedian"] * 8, 40)
    document = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "generation_prompt_complexity_comparison",
        "boundary": dict(_BOUNDARY),
        "publicTemplates": public,
        "comfyCatalog": comfy,
        "authoringDrafts": authoring,
        "campaignBriefs": campaign,
        "verdict": {
            "meetsPublicTemplateMedian": meets_public_bar,
            "exceedsComfyCatalogTitleComplexity": exceeds_comfy_titles,
            "comfyGraphsImported": False,
            "note": (
                "Comfy catalog entries store titles and tags, not prompts. Campaign briefs are compared "
                "against MoDiff public TEMPLATE_PROMPTS, which is the product complexity bar."
            ),
        },
    }
    document["contentHash"] = canonical_content_hash(document)
    return document


def form_overrides_for(workflow_id: str, mode: str, model_type: str) -> dict[str, Any]:
    overlay = param_overlay_for(mode, model_type)
    overlay.update(_WORKFLOW_FORM_OVERRIDES.get(workflow_id, {}))
    brief = brief_for(workflow_id=workflow_id, mode=mode)
    if brief:
        overlay["prompt"] = brief["prompt"]
        if brief["negativePrompt"]:
            overlay["negativePrompt"] = brief["negativePrompt"]
    return overlay


def is_builtin_model(model_type: str) -> bool:
    return model_type.startswith(_BUILTIN_PREFIXES)
