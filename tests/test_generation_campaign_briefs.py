from pathlib import Path
import unittest

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

    def test_cogview4_retry_uses_native_guidance_and_one_object(self):
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
            "original_modiff_campaign_brief_cogview4_kettle_v1",
        )
        self.assertIn("one small dark-green enamel kettle", brief["prompt"])
        self.assertIn("No people, cups, food, text", brief["prompt"])
        self.assertLessEqual(brief["wordCount"], 65)
        self.assertEqual(overlay["guidanceScale"], 3.5)
        self.assertEqual(overlay["seed"], 137)

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

    def test_joyimage_retry_uses_native_guidance_and_one_object(self):
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
            "original_modiff_campaign_brief_joyimage_lamp_v1",
        )
        self.assertIn("one cream ceramic table lamp", brief["prompt"])
        self.assertIn("No people, books, flowers, text", brief["prompt"])
        self.assertLessEqual(brief["wordCount"], 65)
        self.assertEqual(overlay["guidanceScale"], 4.0)
        self.assertEqual(overlay["seed"], 137)

    def test_omnigen_retry_uses_native_guidance_and_one_object(self):
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
            "original_modiff_campaign_brief_omnigen_backpack_v1",
        )
        self.assertIn("one mustard-yellow canvas backpack", brief["prompt"])
        self.assertIn("No people, hands, clothing, text", brief["prompt"])
        self.assertLessEqual(brief["wordCount"], 65)
        self.assertEqual(overlay["guidanceScale"], 2.5)
        self.assertEqual(overlay["seed"], 137)

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
        self.assertEqual(sana_t2i["width"], 512)
        self.assertEqual(sana_t2i["height"], 512)
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

    def test_audio_repaint_overlay_respects_short_campaign_fixture(self):
        overlay = param_overlay_for("audio_repaint", "AceStepAudioPipeline")
        self.assertEqual(overlay["repaintingStart"], 0)
        self.assertLessEqual(overlay["repaintingEnd"], 2)
        self.assertGreater(overlay["repaintingEnd"], overlay["repaintingStart"])

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
