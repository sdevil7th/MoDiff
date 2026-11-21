"""Owned Cosmos safety components using the reviewed package's check methods.

Only constructors select local, immutable artifacts. Prompt moderation, face
processing and SDK output validation retain their original implementations.
No encoder, package global, Hub resolver or process environment is patched.
"""

# The LocalBlocklist decision sequence is adapted from cosmos-guardrail 0.3.1,
# Copyright 2024 The NVIDIA Team and The HuggingFace Team. All rights reserved.
# Licensed under Apache-2.0; copied/modified from that package's
# cosmos_guardrail/cosmos_guardrail.py. See THIRD_PARTY_NOTICES.md and LICENSE.
# Copied/modified upstream origin: https://github.com/NVIDIA/Cosmos.

from __future__ import annotations

from importlib import metadata
from zipfile import ZipFile

import torch

from modiff.cosmos_safety_contract import (
    COSMOS_SAFETY_BUNDLE_MEMBER,
    COSMOS_SAFETY_MODEL_TYPES,
    COSMOS_SAFETY_RUNTIME_PROFILE_ID,
    cosmos_safety_artifacts,
    verify_cosmos_safety_snapshot,
)
from modiff.optional_runtime_execution import (
    _NodeOptionalRuntimeProfile, assert_optional_runtime_ready, optional_runtime_requirement_for_profiles,
)
from utils.huggingface import exact_cached_snapshot_path
from utils.memory_menager import memory_manager


def require_cosmos_safety_runtime():
    from modiff.optional_runtimes import (
        OPTIONAL_RUNTIME_PROFILES, assert_optional_runtime_distribution_compatibility,
    )
    assert_optional_runtime_distribution_compatibility(OPTIONAL_RUNTIME_PROFILES[COSMOS_SAFETY_RUNTIME_PROFILE_ID])
    profile = _NodeOptionalRuntimeProfile(
        id="cosmos3:mandatory-safety", optional_runtime_profiles=(COSMOS_SAFETY_RUNTIME_PROFILE_ID,),
    )
    assert_optional_runtime_ready(optional_runtime_requirement_for_profiles((profile,)))
    if metadata.version("cosmos-guardrail") != "0.3.1":
        raise ValueError("Cosmos safety requires the exact reviewed cosmos-guardrail 0.3.1 runtime.")


def build_cosmos_safety_checker(*, owner_node_id, model_type):
    if model_type not in COSMOS_SAFETY_MODEL_TYPES or not isinstance(owner_node_id, str) or not owner_node_id:
        raise ValueError("Cosmos safety requires a reviewed Models Loader owner.")
    require_cosmos_safety_runtime()
    artifacts = cosmos_safety_artifacts()
    snapshots = [
        verify_cosmos_safety_snapshot(
            exact_cached_snapshot_path(item["repository"], item["revision"], marker_file=item["marker"]), item
        )
        for item in artifacts
    ]
    # Import only after consent/runtime and all exact artifact bytes validate.
    from cosmos_guardrail.cosmos_guardrail import (
        Blocklist, CosmosSafetyChecker, GuardrailRunner, Qwen3Guard, RetinaFaceFilter,
        cfg_re50, load_model, RetinaFace, to_ascii,
    )
    from better_profanity import Profanity
    from nltk.corpus.reader.wordnet import WordNetCorpusReader
    from nltk.data import ZipFilePathPointer
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize.destructive import NLTKWordTokenizer
    from nltk.tokenize.punkt import PunktTokenizer, load_punkt_params
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class LocalWordNetReader(WordNetCorpusReader):
        def map_wn(self, version="wordnet"):
            # The checker only lemmatizes English words. Initialization's
            # multilingual mapping must not consult a second global corpus.
            if version != "wordnet":
                raise ValueError("Cosmos WordNet does not support external corpus mappings.")
            return None

        def open(self, file):
            # The generic corpus reader uses the process-wide NLTK sandbox.
            # This explicit reader permits only the official finite corpus
            # members within an already-digest-verified owned ZIP. The NLTK
            # ZIP pointer retains its decompression checks and text reader.
            if file not in self._FILES:
                raise ValueError("Cosmos WordNet requested an unreviewed corpus member.")
            return self._root.join(file).open(encoding=self.encoding(file))

    class LocalWordNetLemmatizer(WordNetLemmatizer):
        def __init__(self):
            archive = ZipFile(snapshots[0] / "blocklist/nltk_data/corpora/wordnet.zip", "r")
            self._owned_wordnet = LocalWordNetReader(
                ZipFilePathPointer(archive, "wordnet/"),
                omw_reader=None,
            )

        def _morphy(self, form, pos, check_exceptions=True):
            return self._owned_wordnet._morphy(form, pos, check_exceptions=check_exceptions)

    class LocalPunktTokenizer(PunktTokenizer):
        def load_lang(self, lang="english"):
            if lang != "english":
                raise ValueError("The reviewed Cosmos blocklist uses the English tokenizer.")
            with ZipFile(snapshots[0] / "blocklist/nltk_data/tokenizers/punkt_tab.zip", "r") as archive:
                self._params = load_punkt_params(ZipFilePathPointer(archive, "punkt_tab/english/"))
            self._lang = lang

    def read_owned_keywords(folder):
        # The official reader suppresses file read failures. A required
        # safety artifact must remain readable and errors must fail the task.
        prefix = "blocklist/" + folder + "/"
        words = []
        for entry in artifacts[0]["files"]:
            if entry["path"].startswith(prefix):
                with (snapshots[0] / entry["path"]).open(encoding="utf-8") as handle:
                    words.extend(line.strip() for line in handle)
        return words

    class LocalBlocklist(Blocklist):
        def __init__(self):
            self.checkpoint_dir = str(snapshots[0] / "blocklist")
            self.guardrail_partial_match_min_chars = 4
            self.guardrail_partial_match_letter_count = 0.5
            # Exact owned readers preserve the upstream lemmatization and
            # censorship methods without changing process-wide corpus paths
            # or another owner's profanity configuration.
            self.lemmatizer = LocalWordNetLemmatizer()
            self._sentences = LocalPunktTokenizer()
            self._words = NLTKWordTokenizer()
            self.profanity = Profanity()
            self.blocklist_words = read_owned_keywords("custom")
            self.whitelist_words = read_owned_keywords("whitelist")
            self.exact_match_words = read_owned_keywords("exact_match")
            self.profanity.load_censor_words(custom_words=self.blocklist_words, whitelist_words=self.whitelist_words)

        def is_safe(self, input_prompt=""):
            # Cosmos Guardrail 0.3.1's decision sequence, with only its global
            # word_tokenize call replaced by the same owned NLTK tokenizers.
            if not input_prompt:
                return False, "Input is empty"
            input_prompt = to_ascii(input_prompt)
            censored, message = self.censor_prompt(input_prompt)
            if censored:
                return False, message
            tokens = [token for sentence in self._sentences.tokenize(input_prompt)
                      for token in self._words.tokenize(sentence)]
            lemmas = [self.lemmatizer.lemmatize(token) for token in tokens]
            censored, message = self.censor_prompt(" ".join(lemmas))
            if censored:
                return False, message
            censored, message = self.check_against_whole_word_blocklist(
                input_prompt, self.exact_match_words, self.guardrail_partial_match_min_chars,
                self.guardrail_partial_match_letter_count,
            )
            if censored:
                return False, message
            return True, "Input is safe"

    class LocalQwen3Guard(Qwen3Guard):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.dtype = torch.bfloat16
            self.model = AutoModelForCausalLM.from_pretrained(
                str(snapshots[1]), local_files_only=True, trust_remote_code=False, use_safetensors=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(snapshots[1]), local_files_only=True, trust_remote_code=False,
            )

        def is_safe(self, prompt):
            # Upstream catches inference failures and allows the prompt. A
            # mandatory checker must fail the task when moderation cannot run.
            return self.extract_label_and_categories(prompt)

    class LocalRetinaFaceFilter(RetinaFaceFilter):
        def __init__(self):
            torch.nn.Module.__init__(self)
            # Never modify the package's shared cfg_re50 dictionary.
            self.cfg = dict(cfg_re50)
            self.cfg["pretrain"] = False
            self.batch_size = 1
            self.confidence_threshold = 0.7
            self.net = RetinaFace(cfg=self.cfg, phase="test")
            self.net = load_model(self.net, snapshots[0] / "face_blur_filter/Resnet50_Final.pth")
            self.eval()

    class OwnedCosmosSafetyChecker(CosmosSafetyChecker):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.text_guardrail = GuardrailRunner(safety_models=[LocalBlocklist(), LocalQwen3Guard()])
            self.video_guardrail = GuardrailRunner(safety_models=[], postprocessors=[LocalRetinaFaceFilter()])
            # Official runners use plain lists. Register those same models so
            # the existing MemoryManager counts real parameters and buffers.
            self._owned_models = torch.nn.ModuleList(self.nn_models)
            self._modiff_cosmos_owner = owner_node_id
            self._modiff_cosmos_model_type = model_type
            self._modiff_cosmos_artifacts = tuple((item["repository"], item["revision"]) for item in artifacts)

        def to(self, *args, **kwargs):
            torch.nn.Module.to(self, *args, **kwargs)
            return self

        @property
        def device(self):
            return next(self.parameters()).device

        @property
        def dtype(self):
            return next(self.parameters()).dtype

    checker = OwnedCosmosSafetyChecker()
    checker._modiff_cosmos_owned_checker = True
    return checker


def attach_cosmos_safety_checker(pipeline, pipeline_components, *, model_type):
    """Resolve an already-sealed bundle's managed checker, never load another."""
    if model_type not in COSMOS_SAFETY_MODEL_TYPES:
        return
    require_cosmos_safety_runtime()
    from cosmos_guardrail.cosmos_guardrail import CosmosSafetyChecker
    from .route_state import require_component_binding
    require_component_binding(
        pipeline_components, label="Cosmos pipeline component bundle",
        expected_model_type=model_type, expected_role="pipeline_components",
    )
    value = pipeline_components.get(COSMOS_SAFETY_BUNDLE_MEMBER)
    if not isinstance(value, dict) or set(value) != {"model_id", "owner_node_id"}:
        raise ValueError("The connected Cosmos bundle is missing its owned safety checker.")
    checker = memory_manager.get_model(value["model_id"])
    if (
        not isinstance(checker, CosmosSafetyChecker)
        or getattr(checker, "_modiff_cosmos_owned_checker", False) is not True
        or checker._modiff_cosmos_owner != value["owner_node_id"]
        or checker._modiff_cosmos_model_type != model_type
        or checker._modiff_cosmos_artifacts != tuple(
            (item["repository"], item["revision"]) for item in cosmos_safety_artifacts()
        )
    ):
        raise ValueError("The connected Cosmos safety checker is stale or belongs to another owner.")
    # Partial SDK stages omit this config default; the SDK property reads it
    # even after enable_safety_checker sets its runtime flag. Safety is mandatory.
    pipeline.register_to_config(enable_safety_checker=True)
    pipeline.enable_safety_checker(checker)


def reassign_cosmos_safety_owner(bundle, previous_owner, next_owner):
    """Follow the existing loader cache reassignment without creating another owner."""
    if not isinstance(bundle, dict) or bundle.get("model_type") not in COSMOS_SAFETY_MODEL_TYPES:
        return
    from .route_state import require_component_binding
    require_component_binding(bundle, label="Cosmos pipeline component bundle", expected_role="pipeline_components")
    value = bundle.get(COSMOS_SAFETY_BUNDLE_MEMBER)
    if not isinstance(value, dict) or value.get("owner_node_id") != previous_owner:
        raise ValueError("Cannot transfer a Cosmos checker from a different loader owner.")
    checker = memory_manager.get_model(value.get("model_id"))
    if checker is not None:
        if getattr(checker, "_modiff_cosmos_owner", None) != previous_owner:
            raise ValueError("Cannot transfer an unowned Cosmos checker.")
        checker._modiff_cosmos_owner = next_owner
    # An evicted model stays absent; the existing NodeBase cache eviction check
    # forces a fresh loader execution before it can supply another stage.
    value["owner_node_id"] = next_owner
