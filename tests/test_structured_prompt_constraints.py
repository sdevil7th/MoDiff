"""Pure no-model grammar tests; not adapter/runtime/native qualification."""

import codecs
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from modiff import structured_prompt_constraints as constraints
from modiff.structured_prompt_constraints import (
    AuthoredLayoutPrefixConstraint, ByteVocabulary, ConstraintCancelled,
    ConstraintError, ConstraintLimits,
)


EOS, OTHER_SPECIAL = 1000, 1001
PROMPT = (9000, 9001)
T2I = [{"id": "fox", "bbox": [200, 200, 800, 800], "desc": "Authored fox."}]
I2I = [
    {"id": "fox", "from": "ref_image_0", "src_bbox": [10, 20, 300, 400],
     "tgt_bbox": [200, 200, 800, 800], "desc": "Authored fox."},
    {"id": "remove", "from": "ref_image_1", "src_bbox": [10, 10, 20, 20],
     "tgt_bbox": None, "desc": "Remove the object."},
    {"id": "scarf", "from": "target", "src_bbox": None,
     "tgt_bbox": [300, 300, 600, 600], "desc": "Add a scarf."},
]


def vocabulary(*, extra=None, limits=None):
    tokens = {i: bytes((i,)) for i in range(256)}
    tokens.update(extra or {})
    return ByteVocabulary(tokens, special_token_ids={EOS, OTHER_SPECIAL},
                          eos_token_ids={EOS}, limits=limits)


def request(*, rows=None, images=0, vocab=None, **kwargs):
    return AuthoredLayoutPrefixConstraint(
        vocab or vocabulary(), rows if rows is not None else T2I,
        mode="i2i_bbox" if images else "t2i_bbox", image_count=images,
        prompt_token_ids=PROMPT, deadline_monotonic=time.monotonic() + 30, **kwargs)


def text(caption="A red fox <fox>.", rows=None):
    return caption + " " + json.dumps(rows if rows is not None else T2I,
                                      ensure_ascii=False, separators=(",", ":"))


def emit(grammar, raw):
    """Actual callback-style token selection, including incomplete UTF-8 tokens."""
    emitted = []
    for identity in raw:
        assert identity in grammar(0, [*PROMPT, *emitted])
        emitted.append(identity)
    assert EOS in grammar(0, [*PROMPT, *emitted])
    return grammar.finalize([*emitted, EOS])


def independent_completed_oracle(value, expected, images):
    """A separate JSON/tag oracle; production's stricter validator stays separate."""
    caption, payload = value.split("[", 1)
    actual = json.loads("[" + payload)
    assert len(actual) == len(expected)
    for before, after in zip(expected, actual):
        assert set(before) == set(after)
        assert {key: v for key, v in before.items() if key != "desc"} == {
            key: v for key, v in after.items() if key != "desc"}
        assert isinstance(after["desc"], str) and after["desc"].strip()
    tags = re.findall(r"<([^<>]*)>", caption)
    for row in expected:
        assert tags.count(row["id"]) == 1
    assert all(tag in {row["id"] for row in expected}
               or tag in {f"ref_image_{i}" for i in range(images)} for tag in tags)


def test_model_emitted_t2i_table_is_exact_and_raw_wording_is_not_composed():
    proposed = [{**T2I[0], "desc": "A vivid fox with a blue scarf."}]
    output = text("A vivid fox <fox> wears a blue scarf.", proposed)
    assert emit(request(), output.encode()) == output
    independent_completed_oracle(output, T2I, 0)
    assert output != text()  # The grammar did not substitute authored prose.


def test_multiple_reference_source_removal_placement_and_equal_geometry_keep_ids():
    rows = [*I2I, {**I2I[0], "id": "other"}]
    output = text("Move <fox> and <other> from <ref_image_0>, remove <remove> from "
                  "<ref_image_1>, and add <scarf>.", rows)
    assert emit(request(rows=rows, images=2), output.encode()) == output
    independent_completed_oracle(output, rows, 2)


@pytest.mark.parametrize("field,value", [
    ("id", "renamed"), ("bbox", [200, 200, 801, 800]),
    ("bbox", [200.0, 200, 800, 800]),
])
def test_model_cannot_change_authoritative_t2i_literals(field, value):
    altered = [{**T2I[0], field: value}]
    with pytest.raises(ConstraintError, match="violates"):
        request().finalize([*text(rows=altered).encode(), EOS])


@pytest.mark.parametrize("index,field,value", [
    (0, "from", "ref_image_1"), (0, "src_bbox", [10, 20, 301, 400]),
    (0, "tgt_bbox", None), (1, "tgt_bbox", [1, 1, 2, 2]),
    (2, "from", "ref_image_0"),
])
def test_model_cannot_change_authoritative_source_action_or_geometry(index, field, value):
    rows = [{**row} for row in I2I]
    rows[index][field] = value
    with pytest.raises(ConstraintError, match="violates"):
        request(rows=I2I, images=2).finalize([
            *text("<fox> <remove> <scarf>", rows).encode(), EOS])


@pytest.mark.parametrize("caption", [
    "A fox.", "A <fox> and another <fox>.", "A <invented> and <fox>.",
    "A <ref_image_0> and <fox>.", "A <<fox>>.", "A <fox> with >.",
    "A ```fox``` <fox>.", "A [scene] <fox>.",
])
def test_caption_requires_exact_tags_without_invented_authority_or_competing_table(caption):
    with pytest.raises(ConstraintError, match="violates"):
        request().finalize([*text(caption).encode(), EOS])


def test_no_target_tag_can_begin_after_all_are_seen_without_actual_references():
    grammar = request()
    allowed = grammar(0, [*PROMPT, *b"<fox>"])
    assert ord("<") not in allowed and ord("[") in allowed


def test_real_tokens_can_cross_caption_table_description_and_final_boundaries():
    output = text()
    vocab = vocabulary(extra={300: output.encode(), 301: b'<fox>[{"id":"fox",'})
    grammar = request(vocab=vocab)
    assert 300 in grammar(0, PROMPT)
    assert 301 in grammar(0, PROMPT)
    assert grammar(0, [*PROMPT, 300]) == [EOS]
    assert grammar.finalize([300, EOS]) == output


def test_alias_tokens_keep_both_ids_and_caller_mapping_mutation_cannot_change_trie():
    raw = {1: b"A", 2: b"A", 3: b"<", 4: b"f"}
    vocab = ByteVocabulary(raw, special_token_ids={EOS}, eos_token_ids={EOS})
    raw[1] = b"["
    assert vocab.token_bytes[1] == b"A"
    assert {1, 2} <= set(request(vocab=vocab)(0, PROMPT))
    with pytest.raises(TypeError):
        vocab.token_bytes[1] = b"["


def test_utf8_raw_and_escaped_json_preserve_unicode_and_string_boundaries():
    rows = [{**T2I[0], "desc": "狐狸 🦊 é \"quoted\" \\ path\nnext"}]
    output = text("狐狸 🦊 <fox>.", rows)
    result = emit(request(), output.encode("utf-8"))
    assert json.loads("[" + result.split("[", 1)[1])[0]["desc"] == rows[0]["desc"]
    escaped = text().replace('"Authored fox."', '"\\uD83E\\uDD8A \\u00e9 \\tfox"')
    assert emit(request(), escaped.encode()).endswith('"\\uD83E\\uDD8A \\u00e9 \\tfox"}]')


@pytest.mark.parametrize("bad", [
    b"\xc0\xaf", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\xff", b"\x7f",
    b"\\uD800", b"\\uDC00", b"\\uD800\\u0041", b"\\x41", b"\\u0001", b"\\b",
    b"raw\nnewline", b"```", b"", b"   ",
])
def test_invalid_utf8_escapes_surrogates_controls_and_empty_descriptions_refuse(bad):
    output = text().encode().replace(b"Authored fox.", bad)
    with pytest.raises(ConstraintError, match="violates"):
        request().finalize([*output, EOS])


def test_initial_allowed_byte_mask_matches_independent_utf8_prefix_oracle():
    expected = set()
    for byte in range(256):
        if byte in (9, 10, 13) or 32 <= byte <= 126 and byte not in (62, 91, 93, 96):
            expected.add(byte)
        if 0xC2 <= byte <= 0xF4:
            expected.add(byte)  # Incomplete but potentially valid UTF-8 prefix.
    assert set(request()(0, PROMPT)) == expected


@pytest.mark.parametrize("prefix", [
    b"", b"\xc2", b"\xe0", b"\xed", b"\xef\xbf", b"\xf0",
    b"\xf4", b"\xf0\x9f", b"\xf0\x9f\xa6",
])
def test_every_next_caption_byte_agrees_with_independent_python_utf8_decoder(prefix):
    expected = set()
    for byte in range(256):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        try:
            decoded = decoder.decode(prefix + bytes((byte,)), final=False)
        except UnicodeDecodeError:
            continue
        if decoder.getstate()[0]:
            # Python buffers ED A0 until the third byte before refusing its
            # surrogate. A prefix is eligible only if some strict completion
            # exists. UTF-8 continuation intervals include one endpoint here.
            completions = []
            for suffix in (b"\x80", b"\xbf", b"\x80\x80", b"\xbf\xbf",
                           b"\x80\x80\x80", b"\xbf\xbf\xbf"):
                try:
                    completions.append((prefix + bytes((byte,)) + suffix).decode("utf-8"))
                except UnicodeDecodeError:
                    pass
            if not completions:
                continue
        if (any(ord(character) < 32 and character not in "\t\n\r" or 0x7F <= ord(character) <= 0x9F
                for character in decoded) or any(character in ">[]`" for character in decoded)):
            continue
        expected.add(byte)
    grammar = request()
    assert set(grammar(0, [*PROMPT, *b"A ", *prefix])) == expected


def test_cached_prose_lookup_skips_vocabulary_but_exact_length_bound_is_not_cached_away():
    grammar = request()
    first = grammar(0, [*PROMPT, ord("A")])
    visits = grammar.lookup_node_visits
    second = grammar(0, [*PROMPT, ord("A"), ord("B")])
    assert second == first and grammar.lookup_node_visits == visits
    limits = ConstraintLimits(max_description_chars=2)
    grammar = request(vocab=vocabulary(limits=limits), rows=[{**T2I[0], "desc": "ok"}])
    prefix = b'<fox>[{"id":"fox","bbox":[200,200,800,800],"desc":"AB'
    allowed = grammar(0, [*PROMPT, *prefix])
    assert ord('"') in allowed and ord("C") not in allowed


@pytest.mark.parametrize("kind", ["missing_eos", "early_eos", "trailing_text", "special", "incomplete"])
def test_truncation_specials_and_eos_never_repair_incomplete_output(kind):
    output = list(text().encode())
    sequence = {
        "missing_eos": output, "early_eos": [*b"<fox>", EOS],
        "trailing_text": [*output, *b"commentary", EOS],
        "special": [*output, OTHER_SPECIAL, EOS], "incomplete": [*output[:-1], EOS],
    }[kind]
    with pytest.raises(ConstraintError):
        request().finalize(sequence)


def test_changed_prompt_forked_sequence_batch_beam_and_sampling_refuse():
    grammar = request()
    grammar(0, [*PROMPT, ord("A")])
    with pytest.raises(ConstraintError, match="Forked"):
        grammar(0, [*PROMPT, ord("B")])
    with pytest.raises(ConstraintError, match="cannot be reused"):
        grammar(0, PROMPT)
    with pytest.raises(ConstraintError, match="prompt prefix"):
        request()(0, [9999, PROMPT[1]])
    with pytest.raises(ConstraintError, match="single batch"):
        request()(1, PROMPT)
    for options in ({"num_beams": 2}, {"batch_size": 2}, {"do_sample": True}):
        with pytest.raises(ConstraintError, match="greedy"):
            request(**options)


def test_cancellation_deadline_build_trie_and_cache_limits_are_finite(monkeypatch):
    stop = [False]
    grammar = request(should_stop=lambda: stop[0])
    stop[0] = True
    with pytest.raises(ConstraintCancelled):
        grammar(0, PROMPT)
    stop[0] = False
    with pytest.raises(ConstraintError, match="cannot be reused"):
        grammar(0, PROMPT)
    with pytest.raises(ConstraintCancelled):
        ByteVocabulary({0: b"A"}, special_token_ids={EOS}, eos_token_ids={EOS}, should_stop=lambda: True)
    with pytest.raises(ConstraintError, match="node bound"):
        vocabulary(limits=ConstraintLimits(max_trie_nodes=2))
    tiny = request(vocab=vocabulary(limits=ConstraintLimits(max_cache_bytes=1024)))
    tiny(0, PROMPT)
    assert tiny._cache_bytes <= 1024 and not tiny._cache
    grammar = request()
    monkeypatch.setattr(constraints, "time", SimpleNamespace(monotonic=lambda: grammar._deadline + 1))
    with pytest.raises(ConstraintError, match="time bound"):
        grammar(0, PROMPT)


@pytest.mark.parametrize("rows,images", [
    ([], 0), ([T2I[0], T2I[0]], 0), ([{**T2I[0], "id": "ref_image_0"}], 0),
    ([{**T2I[0], "bbox": [1.0, 2, 3, 4]}], 0),
    ([{**I2I[0], "from": "ref_image_1"}], 1),
    ([{**I2I[2], "tgt_bbox": None}], 1),
])
def test_invalid_authoritative_input_refuses_before_generation(rows, images):
    with pytest.raises(ConstraintError):
        request(rows=rows, images=images)


def test_tokenizer_or_resource_bound_misdeclarations_refuse():
    for tokens, special, eos in (({1: b""}, {EOS}, {EOS}), ({EOS: b"A"}, {EOS}, {EOS}),
                                 ({True: b"A"}, {EOS}, {EOS}), ({1: b"A"}, {EOS}, set())):
        with pytest.raises(ConstraintError):
            ByteVocabulary(tokens, special_token_ids=special, eos_token_ids=eos)
    for limits in ({"max_regions": 33}, {"max_new_tokens": 0}, {"lookup_seconds": float("inf")},
                   {"max_cache_bytes": True}):
        with pytest.raises(ConstraintError):
            ConstraintLimits(**limits)


def test_vocabulary_fingerprint_binds_special_eos_and_version_as_well_as_bytes():
    first = ByteVocabulary({1: b"A"}, special_token_ids={EOS, OTHER_SPECIAL}, eos_token_ids={EOS})
    second = ByteVocabulary({1: b"A"}, special_token_ids={EOS, OTHER_SPECIAL}, eos_token_ids={OTHER_SPECIAL})
    assert first.fingerprint != second.fingerprint


def test_module_does_not_import_model_runtime_or_application_services():
    script = r"""
import importlib.util
import sys

name = "isolated_structured_prompt_constraints"
spec = importlib.util.spec_from_file_location(name, sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
assert not any(name == root or name.startswith(root + ".")
               for root in ("torch", "diffusers", "transformers", "modiff.server", "modiff.main")
               for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(Path(constraints.__file__).resolve())],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr
