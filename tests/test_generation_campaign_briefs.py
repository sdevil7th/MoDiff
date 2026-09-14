import unittest
from pathlib import Path

from modiff.generation_campaign_briefs import (
    GenerationCampaignBriefError,
    brief_for,
    compare_prompt_complexity,
    form_overrides_for,
    is_builtin_model,
    param_overlay_for,
)


ROOT = Path(__file__).resolve().parents[1]
CLIENT_ROOT = ROOT.parent / "MoDiff-client"


class GenerationCampaignBriefTests(unittest.TestCase):
    def test_ace_step_uses_a_structured_long_form_song_recipe(self):
        brief = brief_for(workflow_id="AceStepAudioPipeline:text_to_audio", mode="text_to_audio")
        overlay = form_overrides_for(
            "AceStepAudioPipeline:text_to_audio",
            "text_to_audio",
            "AceStepAudioPipeline",
        )

        self.assertEqual(brief["provenance"], "official_ace_step_prompt_contract_adapted_modiff_song_v1")
        self.assertIn("complete progressive alternative-metal song", brief["prompt"])
        self.assertIn("[verse]", overlay["lyrics"])
        self.assertIn("[chorus]", overlay["lyrics"])
        self.assertEqual(overlay["audioDuration"], 60)
        self.assertEqual(overlay["steps"], 8)
        self.assertEqual(overlay["guidanceScale"], 1)
        self.assertEqual(overlay["shift"], 3)
        self.assertEqual(overlay["bpm"], 126)
        self.assertEqual(overlay["keyscale"], "D minor")
        self.assertEqual(overlay["timesignature"], "4")

    def test_ace_step_source_modes_never_target_a_two_second_showcase(self):
        continuation = form_overrides_for(
            "AceStepAudioPipeline:audio_continuation",
            "audio_continuation",
            "AceStepAudioPipeline",
        )
        variation = form_overrides_for(
            "AceStepAudioPipeline:audio_variation",
            "audio_variation",
            "AceStepAudioPipeline",
        )
        repaint = form_overrides_for(
            "AceStepAudioPipeline:audio_repaint",
            "audio_repaint",
            "AceStepAudioPipeline",
        )

        self.assertEqual(continuation["audioDuration"], 75)
        self.assertEqual(continuation["extensionDuration"], 15)
        self.assertEqual(variation["audioDuration"], 60)
        self.assertEqual(variation["audioCoverStrength"], 0.65)
        self.assertEqual(repaint["audioDuration"], 75)
        self.assertEqual((repaint["repaintingStart"], repaint["repaintingEnd"]), (28, 43))
        self.assertTrue(all(item["steps"] == 8 for item in (continuation, variation, repaint)))

    def test_audioldm2_restores_the_official_quality_recipe_and_identifiable_sound(self):
        brief = brief_for(workflow_id="AudioLDM2Pipeline:text_to_audio", mode="text_to_audio")
        overlay = form_overrides_for(
            "AudioLDM2Pipeline:text_to_audio",
            "text_to_audio",
            "AudioLDM2Pipeline",
        )

        self.assertIn("three distinct low thunder rolls", brief["prompt"])
        self.assertIn("Low quality", brief["negativePrompt"])
        self.assertEqual(overlay["audioDuration"], 30)
        self.assertEqual(overlay["numWaveforms"], 3)
        self.assertEqual(overlay["steps"], 200)
        self.assertEqual(overlay["guidanceScale"], 3.5)

    def test_pixart_uses_the_native_showcase_recipe_and_distinct_model_specific_briefs(self):
        standard = form_overrides_for(
            "PixArtSigmaPipeline:text_to_image",
            "text_to_image",
            "PixArtSigmaPipeline",
        )
        pag = form_overrides_for(
            "PixArtSigmaPAGPipeline:text_to_image",
            "text_to_image",
            "PixArtSigmaPAGPipeline",
        )

        for overlay in (standard, pag):
            self.assertEqual(overlay["dtype"], "float32")
            self.assertEqual((overlay["width"], overlay["height"]), (1024, 1024))
            self.assertEqual(overlay["steps"], 20)
            self.assertEqual(overlay["maxSequenceLength"], 300)
            self.assertNotIn("Campaign objective", overlay["prompt"])
        self.assertEqual(standard["guidanceScale"], 4.5)
        self.assertEqual(pag["guidanceScale"], 1.0)
        self.assertIn("kinetic sculpture", standard["prompt"])
        self.assertIn("orrery", pag["prompt"])
        self.assertNotEqual(standard["prompt"], pag["prompt"])
        self.assertEqual(pag["pagScale"], 4.0)
        self.assertEqual(pag["pagAdaptiveScale"], 0.0)

    def test_longcat_edit_uses_the_official_precise_edit_recipe(self):
        overlay = form_overrides_for(
            "LongCatImageEditPipeline:edit_image",
            "edit_image",
            "LongCatImageEditPipeline",
        )

        self.assertIn("Change only the open red umbrella canopy", overlay["prompt"])
        self.assertIn("every other object unchanged", overlay["prompt"])
        self.assertEqual(overlay.get("negativePrompt", ""), "")
        self.assertEqual(overlay["dtype"], "bfloat16")
        self.assertEqual(overlay["steps"], 50)
        self.assertEqual(overlay["guidanceScale"], 4.5)
        self.assertEqual(overlay["seed"], 43)
        self.assertFalse(overlay["randomSeed"])
        self.assertEqual(overlay["offloadMode"], "model_cpu")
        self.assertEqual(overlay["maxSequenceLength"], 512)

    def test_sdxl_instruct_pix2pix_uses_the_official_768_edit_recipe(self):
        overlay = form_overrides_for(
            "StableDiffusionXLInstructPix2PixPipeline:edit_image",
            "edit_image",
            "StableDiffusionXLInstructPix2PixPipeline",
        )

        self.assertIn("traditional Prussian-blue cyanotype", overlay["prompt"])
        self.assertIn("exact number and arrangement", overlay["prompt"])
        self.assertEqual(overlay.get("negativePrompt", ""), "")
        self.assertEqual(overlay["dtype"], "float16")
        self.assertEqual((overlay["width"], overlay["height"]), (768, 768))
        self.assertEqual(overlay["steps"], 30)
        self.assertEqual(overlay["guidanceScale"], 3.0)
        self.assertEqual(overlay["conditioningScale"], 1.5)
        self.assertEqual(overlay["seed"], 43)
        self.assertFalse(overlay["randomSeed"])
        self.assertEqual(overlay["offloadMode"], "model_cpu")

    def test_longcat_short_native_proof_has_an_identifiable_sound_contract(self):
        brief = brief_for(workflow_id="LongCatAudioDiTPipeline:text_to_audio", mode="text_to_audio")
        overlay = form_overrides_for(
            "LongCatAudioDiTPipeline:text_to_audio",
            "text_to_audio",
            "LongCatAudioDiTPipeline",
        )

        self.assertIn("small ocean waves washing over rounded pebbles", brief["prompt"])
        self.assertEqual(overlay["audioDuration"], 5)
        self.assertEqual(overlay["steps"], 20)
        self.assertEqual(overlay["guidanceScale"], 4)

    def test_cogvideox_uses_the_official_representative_frame_and_step_recipe(self):
        for workflow_id, model_type, mode in (
            ("CogVideoXPipeline:text_to_video", "CogVideoXPipeline", "text_to_video"),
            ("CogVideoXVideoToVideoPipeline:video_to_video", "CogVideoXVideoToVideoPipeline", "video_to_video"),
        ):
            with self.subTest(workflow_id=workflow_id):
                overlay = form_overrides_for(workflow_id, mode, model_type)
                self.assertEqual(
                    {key: overlay[key] for key in ("width", "height", "numFrames", "steps", "guidanceScale")},
                    {"width": 720, "height": 480, "numFrames": 49, "steps": 50, "guidanceScale": 6},
                )

    def test_text_to_image_briefs_are_campaign_grade_and_original(self):
        brief = brief_for(workflow_id="StableDiffusionXLPipeline:text_to_image", mode="text_to_image")
        self.assertIsNotNone(brief)
        self.assertGreaterEqual(brief["wordCount"], 80)
        self.assertGreaterEqual(brief["characterCount"], 550)
        self.assertNotIn("Comfy", brief["prompt"])
        self.assertIn("Campaign objective", brief["prompt"])
        self.assertTrue(brief["negativePrompt"])

    def test_lcm_quality_retry_avoids_card_layout_and_fits_prompt_budget(self):
        brief = brief_for(workflow_id="LatentConsistencyModelPipeline:text_to_image", mode="text_to_image")
        self.assertEqual(brief["provenance"], "original_modiff_campaign_brief_v6")
        self.assertIn("two cream alpine flowers growing together", brief["prompt"])
        self.assertIn("short green stems and attached leaves", brief["prompt"])
        self.assertIn("No vase, frame, paper, card", brief["prompt"])
        self.assertIn("paper, card, floral border", brief["negativePrompt"])
        self.assertIn("floating petals", brief["negativePrompt"])
        self.assertLessEqual(brief["wordCount"], 60)
        self.assertLessEqual(brief["characterCount"], 500)

        unrelated = brief_for(workflow_id="StableDiffusionXLPipeline:text_to_image", mode="text_to_image")
        self.assertEqual(unrelated["provenance"], "original_modiff_campaign_brief_v1")

    def test_sd15_quality_retry_uses_a_text_free_single_subject_contract(self):
        brief = brief_for(
            workflow_id="StableDiffusionPipeline:text_to_image",
            mode="text_to_image",
        )

        self.assertEqual(brief["provenance"], "original_modiff_campaign_brief_sd15_canoe_v1")
        self.assertIn("one red wooden canoe", brief["prompt"])
        self.assertIn("No people, buildings, signs", brief["prompt"])
        self.assertIn("extra canoe, duplicate boat", brief["negativePrompt"])
        self.assertLessEqual(brief["wordCount"], 70)
        self.assertLessEqual(brief["characterCount"], 500)

    def test_dreamlite_mobile_retry_binds_one_chair_and_a_new_seed(self):
        brief = brief_for(
            workflow_id="DreamLiteMobilePipeline:text_to_image",
            mode="text_to_image",
        )
        overlay = form_overrides_for(
            "DreamLiteMobilePipeline:text_to_image",
            "text_to_image",
            "DreamLiteMobilePipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "original_modiff_campaign_brief_dreamlite_mobile_courtyard_v1",
        )
        self.assertIn("one linen lounge chair", brief["prompt"])
        self.assertIn("No people, additional chairs", brief["prompt"])
        self.assertLessEqual(brief["wordCount"], 60)
        self.assertEqual(overlay["seed"], 137)

    def test_cogview4_corrective_recipe_uses_official_quality_settings(self):
        brief = brief_for(
            workflow_id="CogView4Pipeline:text_to_image",
            mode="text_to_image",
        )
        overlay = form_overrides_for(
            "CogView4Pipeline:text_to_image",
            "text_to_image",
            "CogView4Pipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "official_cogview4_recipe_adapted_modiff_observatory_v2",
        )
        self.assertIn("cliffside observatory library", brief["prompt"])
        self.assertIn("brass telescope", brief["prompt"])
        self.assertLessEqual(brief["wordCount"], 130)
        self.assertEqual(overlay["steps"], 50)
        self.assertEqual((overlay["width"], overlay["height"]), (1280, 768))
        self.assertEqual(overlay["guidanceScale"], 3.5)
        self.assertEqual(overlay["seed"], 314159)

    def test_janus_image_recipe_is_native_simple_and_does_not_invent_negative_prompt_support(self):
        brief = brief_for(
            workflow_id="HuggingFaceAnyToAnyModel:text_to_image",
            mode="text_to_image",
        )

        self.assertEqual(
            brief["provenance"],
            "official_janus_sampled_image_contract_adapted_modiff_sailboat_v1",
        )
        self.assertIn("small red sailboat", brief["prompt"])
        self.assertIn("crisp readable silhouette", brief["prompt"])
        self.assertEqual(brief["negativePrompt"], "")
        self.assertLessEqual(brief["wordCount"], 70)

    def test_smolvlm_recipe_is_officially_grounded_and_constrains_factual_output(self):
        brief = brief_for(
            workflow_id="HuggingFaceImageTextToTextModel:image_to_text",
            mode="image_to_text",
        )
        overlay = form_overrides_for(
            "HuggingFaceImageTextToTextModel:image_to_text",
            "image_to_text",
            "HuggingFaceImageTextToTextModel",
        )

        self.assertEqual(
            brief["provenance"],
            "official_smolvlm_single_image_recipe_adapted_modiff_rain_shelter_v1",
        )
        self.assertIn("one factual sentence of 25 to 45 words", brief["prompt"])
        self.assertIn("single person", brief["prompt"])
        self.assertIn("Do not infer", brief["prompt"])
        self.assertEqual(brief["negativePrompt"], "")
        self.assertEqual(overlay["dtype"], "bfloat16")

    def test_flux_schnell_retry_binds_one_lantern_and_a_new_seed(self):
        brief = brief_for(
            workflow_id="FluxSchnellPipeline:text_to_image",
            mode="text_to_image",
        )
        overlay = form_overrides_for(
            "FluxSchnellPipeline:text_to_image",
            "text_to_image",
            "FluxSchnellPipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "original_modiff_campaign_brief_flux_schnell_lantern_v1",
        )
        self.assertIn("one weathered stone lantern", brief["prompt"])
        self.assertIn("No people, buildings, text", brief["prompt"])
        self.assertLessEqual(brief["wordCount"], 55)
        self.assertEqual(overlay["seed"], 137)

    def test_hunyuandit_pag_uses_the_distilled_checkpoint_and_official_pag_recipe(self):
        brief = brief_for(
            workflow_id="HunyuanDiTPAGPipeline:text_to_image",
            mode="text_to_image",
        )
        overlay = form_overrides_for(
            "HunyuanDiTPAGPipeline:text_to_image",
            "text_to_image",
            "HunyuanDiTPAGPipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "official_hunyuandit_pag_recipe_adapted_modiff_suzhou_corridor_v1",
        )
        self.assertIn("one complete circular moon gate", brief["prompt"])
        self.assertIn("Empty scene. No people, text", brief["prompt"])
        self.assertLessEqual(brief["wordCount"], 70)
        self.assertEqual(
            {key: overlay[key] for key in ("steps", "width", "height", "guidanceScale", "pagScale", "seed")},
            {
                "steps": 25,
                "width": 1024,
                "height": 1024,
                "guidanceScale": 4.0,
                "pagScale": 3.0,
                "seed": 314159,
            },
        )

    def test_joyimage_text_to_image_uses_the_official_native_recipe(self):
        brief = brief_for(
            workflow_id="JoyImageEditPipeline:text_to_image",
            mode="text_to_image",
        )
        overlay = form_overrides_for(
            "JoyImageEditPipeline:text_to_image",
            "text_to_image",
            "JoyImageEditPipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "official_joyimage_edit_t2i_recipe_adapted_modiff_violin_workshop_v1",
        )
        self.assertIn("One unfinished violin", brief["prompt"])
        self.assertIn("No people, extra instruments, text", brief["prompt"])
        self.assertEqual(
            {key: overlay[key] for key in ("steps", "width", "height", "guidanceScale", "maxSequenceLength", "seed")},
            {
                "steps": 40,
                "width": 1024,
                "height": 1024,
                "guidanceScale": 4.0,
                "maxSequenceLength": 4096,
                "seed": 42,
            },
        )

    def test_joyimage_edit_uses_the_official_camera_control_contract(self):
        brief = brief_for(
            workflow_id="JoyImageEditPipeline:edit_image",
            mode="edit_image",
        )
        overlay = form_overrides_for(
            "JoyImageEditPipeline:edit_image",
            "edit_image",
            "JoyImageEditPipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "official_joyimage_edit_camera_control_template_violin_workshop_v1",
        )
        self.assertIn("Camera rotation: Yaw 20.0°, Pitch -12.0°", brief["prompt"])
        self.assertIn("Keep the 3D scene static; only change the viewpoint", brief["prompt"])
        self.assertEqual(brief["negativePrompt"], "")
        self.assertEqual(
            {key: overlay[key] for key in ("steps", "width", "height", "guidanceScale", "maxSequenceLength", "seed")},
            {
                "steps": 40,
                "width": 1024,
                "height": 1024,
                "guidanceScale": 4.0,
                "maxSequenceLength": 4096,
                "seed": 43,
            },
        )
    def test_flux_dev_retry_uses_reviewed_recipe_and_one_building(self):
        brief = brief_for(
            workflow_id="FluxDevPipeline:text_to_image",
            mode="text_to_image",
        )
        overlay = form_overrides_for(
            "FluxDevPipeline:text_to_image",
            "text_to_image",
            "FluxDevPipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "original_modiff_campaign_brief_flux_dev_shed_v1",
        )
        self.assertIn("one small weathered red wooden shed", brief["prompt"])
        self.assertIn("No people, vehicles, signs, text", brief["prompt"])
        self.assertLessEqual(brief["wordCount"], 65)
        self.assertEqual(
            {key: overlay[key] for key in ("steps", "width", "height", "guidanceScale", "seed")},
            {"steps": 20, "width": 768, "height": 768, "guidanceScale": 3.5, "seed": 137},
        )

    def test_glm_image_retry_uses_native_guidance_and_one_object(self):
        brief = brief_for(
            workflow_id="GlmImagePipeline:text_to_image",
            mode="text_to_image",
        )
        overlay = form_overrides_for(
            "GlmImagePipeline:text_to_image",
            "text_to_image",
            "GlmImagePipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "original_modiff_campaign_brief_glm_umbrella_v1",
        )
        self.assertIn("one open red umbrella", brief["prompt"])
        self.assertIn("No people, vehicles, shops, signs, text", brief["prompt"])
        self.assertLessEqual(brief["wordCount"], 70)
        self.assertEqual(overlay["guidanceScale"], 1.5)
        self.assertEqual(overlay["steps"], 50)
        self.assertEqual(overlay["seed"], 137)

    def test_sana_pag_retry_uses_reviewed_guidance_and_a_simple_scene(self):
        brief = brief_for(
            workflow_id="SanaPAGPipeline:text_to_image",
            mode="text_to_image",
        )
        overlay = form_overrides_for(
            "SanaPAGPipeline:text_to_image",
            "text_to_image",
            "SanaPAGPipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "original_modiff_campaign_brief_sana_pag_tent_v1",
        )
        self.assertIn("one small red canvas tent", brief["prompt"])
        self.assertIn("No people, campfire", brief["prompt"])
        self.assertLessEqual(brief["wordCount"], 60)
        self.assertEqual(overlay["guidanceScale"], 4.5)
        self.assertEqual(overlay["seed"], 137)

    def test_sana_retry_uses_the_official_native_recipe_and_a_distinct_scene(self):
        brief = brief_for(
            workflow_id="SanaPipeline:text_to_image",
            mode="text_to_image",
        )
        overlay = form_overrides_for(
            "SanaPipeline:text_to_image",
            "text_to_image",
            "SanaPipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "official_sana_diffusers_recipe_adapted_modiff_orthophoto_v3",
        )
        self.assertIn("90-degree nadir angle", brief["prompt"])
        self.assertIn("one long rust-red almond-shaped gravel bar centered", brief["prompt"])
        self.assertIn("no sky or horizon is visible", brief["prompt"])
        self.assertIn("oblique view, angled camera", brief["negativePrompt"])
        self.assertLessEqual(brief["wordCount"], 120)
        self.assertEqual(
            {key: overlay[key] for key in ("steps", "width", "height", "guidanceScale", "maxSequenceLength", "seed")},
            {
                "steps": 20,
                "width": 1024,
                "height": 1024,
                "guidanceScale": 4.5,
                "maxSequenceLength": 300,
                "seed": 314159,
            },
        )

    def test_sana_sprint_uses_a_distinct_single_subject_brief_and_locked_seed(self):
        brief = brief_for(
            workflow_id="SanaSprintPipeline:text_to_image",
            mode="text_to_image",
        )
        overlay = form_overrides_for(
            "SanaSprintPipeline:text_to_image",
            "text_to_image",
            "SanaSprintPipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "original_modiff_campaign_brief_sana_sprint_raincoat_v1",
        )
        self.assertIn("one mustard-yellow raincoat", brief["prompt"])
        self.assertIn("No people, hands, boots", brief["prompt"])
        self.assertLessEqual(brief["wordCount"], 65)
        self.assertEqual(overlay["steps"], 2)
        self.assertEqual(overlay["guidanceScale"], 4.5)
        self.assertEqual(overlay["seed"], 137)

    def test_sana_sprint_edit_uses_official_img2img_defaults_and_native_card_size(self):
        brief = brief_for(
            workflow_id="SanaSprintPipeline:edit_image",
            mode="edit_image",
        )
        overlay = form_overrides_for(
            "SanaSprintPipeline:edit_image",
            "edit_image",
            "SanaSprintPipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "official_sana_sprint_img2img_recipe_adapted_modiff_snowy_shelter_v2",
        )
        self.assertIn("Preserve the exact tram-shelter architecture", brief["prompt"])
        self.assertIn("Remove the person, red umbrella", brief["prompt"])
        self.assertIn("changed camera angle", brief["negativePrompt"])
        self.assertEqual(
            {
                key: overlay[key]
                for key in (
                    "steps",
                    "width",
                    "height",
                    "guidanceScale",
                    "maxSequenceLength",
                    "strength",
                    "seed",
                )
            },
            {
                "steps": 2,
                "width": 1024,
                "height": 512,
                "guidanceScale": 4.5,
                "maxSequenceLength": 300,
                "strength": 0.6,
                "seed": 271828,
            },
        )
    def test_z_image_uses_native_turbo_settings_and_one_object(self):
        brief = brief_for(
            workflow_id="ZImageModularPipeline:text_to_image",
            mode="text_to_image",
        )
        overlay = form_overrides_for(
            "ZImageModularPipeline:text_to_image",
            "text_to_image",
            "ZImageModularPipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "original_modiff_campaign_brief_z_image_watering_can_v1",
        )
        self.assertIn("one weathered terracotta watering can", brief["prompt"])
        self.assertIn("No people, hands, flowers, text", brief["prompt"])
        self.assertLessEqual(brief["wordCount"], 65)
        self.assertEqual(overlay["steps"], 8)
        self.assertEqual(overlay["guidanceScale"], 1)
        self.assertEqual(overlay["seed"], 211)

    def test_lumina_retry_uses_native_guidance_and_one_object(self):
        brief = brief_for(
            workflow_id="LuminaPipeline:text_to_image",
            mode="text_to_image",
        )
        overlay = form_overrides_for(
            "LuminaPipeline:text_to_image",
            "text_to_image",
            "LuminaPipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "original_modiff_campaign_brief_lumina_bowl_v1",
        )
        self.assertIn("one shallow cobalt-blue ceramic bowl", brief["prompt"])
        self.assertIn("No people, hands, food", brief["prompt"])
        self.assertLessEqual(brief["wordCount"], 65)
        self.assertEqual(overlay["guidanceScale"], 4.0)
        self.assertEqual(overlay["seed"], 137)

    def test_omnigen_uses_official_native_recipe_without_shared_memory_campaign_cap(self):
        brief = brief_for(
            workflow_id="OmniGenPipeline:text_to_image",
            mode="text_to_image",
        )
        overlay = form_overrides_for(
            "OmniGenPipeline:text_to_image",
            "text_to_image",
            "OmniGenPipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "official_omnigen_diffusers_recipe_adapted_modiff_glasswing_v3",
        )
        self.assertIn("one Greta oto glasswing butterfly", brief["prompt"])
        self.assertIn("exactly six coherent legs", brief["prompt"])
        self.assertIn("No opaque white wing panels", brief["prompt"])
        self.assertEqual(brief["negativePrompt"], "")
        self.assertGreaterEqual(brief["wordCount"], 60)
        self.assertEqual(overlay["steps"], 50)
        self.assertEqual(overlay["width"], 1024)
        self.assertEqual(overlay["height"], 1024)
        self.assertEqual(overlay["guidanceScale"], 2.5)
        self.assertEqual(overlay["seed"], 271828)

    def test_omnigen_edit_uses_reference_first_native_recipe_without_shared_memory_cap(self):
        brief = brief_for(
            workflow_id="OmniGenPipeline:edit_image",
            mode="edit_image",
        )
        overlay = form_overrides_for(
            "OmniGenPipeline:edit_image",
            "edit_image",
            "OmniGenPipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "official_omnigen_localized_edit_recipe_adapted_modiff_tram_panel_v2",
        )
        self.assertIn("two localized edits", brief["prompt"])
        self.assertIn("plain matte cobalt-blue enamel panel", brief["prompt"])
        self.assertIn("Preserve the person's exact identity", brief["prompt"])
        self.assertEqual(brief["negativePrompt"], "")
        self.assertEqual(overlay["steps"], 50)
        self.assertEqual(overlay["width"], 1024)
        self.assertEqual(overlay["height"], 1024)
        self.assertEqual(overlay["guidanceScale"], 2.0)
        self.assertEqual(overlay["conditioningScale"], 1.6)
        self.assertEqual(overlay["seed"], 222)

    def test_omnigen_multi_reference_uses_ordered_native_recipe_without_shared_memory_cap(self):
        brief = brief_for(
            workflow_id="OmniGenPipeline:multi_image_reference_edit",
            mode="multi_image_reference_edit",
        )
        overlay = form_overrides_for(
            "OmniGenPipeline:multi_image_reference_edit",
            "multi_image_reference_edit",
            "OmniGenPipeline",
        )

        self.assertEqual(
            brief["provenance"],
            "official_omnigen_ordered_multi_reference_recipe_adapted_modiff_harbor_lantern_v2",
        )
        self.assertIn("second reference as the complete outer object", brief["prompt"])
        self.assertIn("simplified miniature harbor from the first", brief["prompt"])
        self.assertIn("exactly two clearly separated wooden sailboats", brief["prompt"])
        self.assertEqual(brief["negativePrompt"], "")
        self.assertEqual(overlay["steps"], 50)
        self.assertEqual(overlay["width"], 1024)
        self.assertEqual(overlay["height"], 1024)
        self.assertEqual(overlay["guidanceScale"], 2.5)
        self.assertEqual(overlay["conditioningScale"], 1.6)
        self.assertEqual(overlay["seed"], 667)

    def test_few_step_models_keep_native_step_counts(self):
        lcm = param_overlay_for("text_to_image", "LatentConsistencyModelPipeline")
        flux_schnell = param_overlay_for("text_to_image", "FluxSchnellPipeline")
        sdxl = param_overlay_for("text_to_image", "StableDiffusionXLPipeline")
        ernie = param_overlay_for("text_to_image", "ErnieImagePipeline")
        self.assertEqual(lcm["resourceMode"], "auto")
        self.assertEqual(lcm["steps"], 4)
        self.assertEqual(flux_schnell["steps"], 4)
        self.assertEqual(flux_schnell["guidanceScale"], 0.0)
        self.assertGreaterEqual(sdxl["steps"], 20)
        self.assertGreaterEqual(sdxl["width"], 1024)
        self.assertLessEqual(ernie["steps"], 8)
        self.assertEqual(ernie["width"], 1024)
        self.assertEqual(ernie["height"], 1024)
        joy = param_overlay_for("text_to_image", "JoyImageEditPipeline")
        self.assertLessEqual(joy["steps"], 16)
        self.assertEqual(joy["width"], 512)
        self.assertEqual(joy["height"], 512)
        lumina = param_overlay_for("text_to_image", "LuminaPipeline")
        self.assertLessEqual(lumina["steps"], 16)
        self.assertEqual(lumina["width"], 512)
        self.assertEqual(lumina["height"], 512)

    def test_campaign_uses_reviewed_native_guidance_when_authoring_captures_it(self):
        self.assertEqual(param_overlay_for("text_to_image", "CogView4Pipeline")["guidanceScale"], 3.5)
        self.assertEqual(param_overlay_for("text_to_image", "LuminaPipeline")["guidanceScale"], 4.0)
        self.assertEqual(param_overlay_for("text_to_image", "OmniGenPipeline")["guidanceScale"], 2.5)
        animatelcm = param_overlay_for("text_to_video", "AnimateLCMPipeline")
        self.assertLessEqual(animatelcm["steps"], 8)
        self.assertGreaterEqual(animatelcm["guidanceScale"], 0)
        self.assertLessEqual(animatelcm["guidanceScale"], 2)
        ltx = param_overlay_for("text_to_video", "LTXVideoPipeline")
        self.assertEqual(ltx["steps"], 8)
        self.assertEqual(ltx["guidanceScale"], 1.0)
        ltx_condition = param_overlay_for("text_to_video", "LTXConditionPipeline")
        self.assertEqual(ltx_condition["steps"], 8)
        self.assertEqual(ltx_condition["guidanceScale"], 1.0)
        sana_t2i = param_overlay_for("text_to_image", "SanaPipeline")
        self.assertEqual(sana_t2i["width"], 1024)
        self.assertEqual(sana_t2i["height"], 1024)
        self.assertEqual(sana_t2i["steps"], 20)
        sana_video = param_overlay_for("text_to_video", "SanaVideoPipeline")
        self.assertEqual(sana_video["width"], 832)
        self.assertEqual(sana_video["height"], 480)
        wan_t2v = param_overlay_for("text_to_video", "WanVideoPipeline")
        self.assertEqual(wan_t2v["steps"], 30)
        wan_ti2v = param_overlay_for("text_to_video", "WanTI2VPipeline")
        self.assertLessEqual(wan_ti2v["steps"], 8)
        ace = param_overlay_for("text_to_audio", "AceStepAudioPipeline")
        self.assertEqual(ace["attentionBackend"], "_native_math")
        longcat_audio = param_overlay_for("text_to_audio", "LongCatAudioDiTPipeline")
        self.assertEqual(longcat_audio["attentionBackend"], "_native_math")
        chroma_inpaint = param_overlay_for("inpaint", "ChromaInpaintPipeline")
        self.assertLessEqual(chroma_inpaint["width"], 1024)
        self.assertLessEqual(chroma_inpaint["height"], 1024)
        flux_canny_edit = param_overlay_for("control_edit_image", "FluxCannyPipeline")
        self.assertLessEqual(flux_canny_edit["guidanceScale"], 20)
        self.assertLessEqual(flux_canny_edit["width"], 512)
        self.assertLessEqual(flux_canny_edit["height"], 512)
        self.assertLessEqual(flux_canny_edit["steps"], 16)
        flux_canny_inpaint = param_overlay_for("control_inpaint", "FluxCannyPipeline")
        self.assertLessEqual(flux_canny_inpaint["guidanceScale"], 20)
        self.assertLessEqual(flux_canny_inpaint["width"], 512)

    def test_generic_audio_repaint_overlay_does_not_adapt_to_a_synthetic_fixture(self):
        overlay = param_overlay_for("audio_repaint", "AceStepAudioPipeline")
        self.assertNotIn("repaintingStart", overlay)
        self.assertNotIn("repaintingEnd", overlay)

    def test_complexity_meets_public_templates_and_exceeds_comfy_titles(self):
        if not CLIENT_ROOT.is_dir():
            self.skipTest("Sibling MoDiff-client checkout is required for public-template comparison.")
        report = compare_prompt_complexity(root=ROOT, client_root=CLIENT_ROOT)
        comfy = report["comfyCatalog"]
        self.assertEqual(comfy["storedPromptFieldCount"], 0)
        self.assertFalse(comfy["importsComfyGraphs"])
        self.assertGreaterEqual(comfy["templateCount"], 100)
        self.assertTrue(report["verdict"]["meetsPublicTemplateMedian"])
        self.assertTrue(report["verdict"]["exceedsComfyCatalogTitleComplexity"])
        self.assertGreaterEqual(report["campaignBriefs"]["wordMedian"], report["publicTemplates"]["wordMedian"])
        self.assertGreater(report["campaignBriefs"]["wordMedian"], comfy["titleWordMedian"] * 8)

    def test_form_overlay_includes_auto_and_prompt(self):
        overlay = form_overrides_for(
            "StableDiffusionXLPipeline:text_to_image",
            "text_to_image",
            "StableDiffusionXLPipeline",
        )
        self.assertEqual(overlay["resourceMode"], "auto")
        self.assertIn("Campaign objective", overlay["prompt"])
        self.assertFalse(is_builtin_model("StableDiffusionXLPipeline"))
        self.assertTrue(is_builtin_model("BuiltinImageOperation"))

    def test_lcm_quality_retry_seed_is_workflow_scoped(self):
        lcm = form_overrides_for(
            "LatentConsistencyModelPipeline:text_to_image",
            "text_to_image",
            "LatentConsistencyModelPipeline",
        )
        unrelated = form_overrides_for(
            "StableDiffusionPipeline:text_to_image",
            "text_to_image",
            "StableDiffusionPipeline",
        )
        self.assertFalse(lcm["randomSeed"])
        self.assertEqual(lcm["seed"], 137)
        self.assertEqual(unrelated["seed"], 42)

    def test_lcm_edit_retry_binds_source_specific_brief_and_strength(self):
        brief = brief_for(
            workflow_id="LatentConsistencyModelPipeline:edit_image",
            mode="edit_image",
        )
        overlay = form_overrides_for(
            "LatentConsistencyModelPipeline:edit_image",
            "edit_image",
            "LatentConsistencyModelPipeline",
        )
        unrelated = form_overrides_for(
            "StableDiffusionXLImg2ImgPipeline:edit_image",
            "edit_image",
            "StableDiffusionXLImg2ImgPipeline",
        )

        self.assertEqual(brief["provenance"], "original_modiff_campaign_brief_lcm_edit_v3")
        self.assertIn("one rounded body", brief["prompt"])
        self.assertIn("Blank front except grille and dial", brief["prompt"])
        self.assertLessEqual(brief["wordCount"], 55)
        self.assertLessEqual(brief["characterCount"], 450)
        self.assertIn("person, human, portrait", brief["negativePrompt"])
        self.assertEqual(overlay["strength"], 0.4)
        self.assertNotIn("strength", unrelated)

    def test_missing_comfy_catalog_fails_closed(self):
        with self.assertRaises(GenerationCampaignBriefError):
            compare_prompt_complexity(root=Path("/tmp/missing-modiff-root"), client_root=CLIENT_ROOT)
