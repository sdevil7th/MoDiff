"""Pure, bounded prefix constraints for model-emitted authored-layout output.

This is an Expert-only building block, not an enabled recipe or an executor.
Every caption/table byte still has to be emitted by the model. There is no
post-generation repair or deterministic composition of a missing table. The
existing independent ``validate_target_layout_output`` must validate the raw
result before a proposal can be applied. Format validity says nothing about
whether the generated wording is useful or faithful to the requested image.

The adapter must supply authoritative projected rows, a verified *ordinary*
token-ID to raw-byte mapping, all special IDs, and its existing task deadline.
No tokenizer/model imports, filesystem access, global patches or network occur
here. Tokenizer provenance and model/runtime ownership remain adapter duties.
"""

from array import array
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import re
import time
from types import MappingProxyType
from typing import NamedTuple
import unicodedata


CONSTRAINT_VERSION = "authored-layout-byte-prefix-v1"


class ConstraintError(ValueError):
    """A malformed request, invalid continuation or finite bound refuses."""


class ConstraintCancelled(ConstraintError):
    """The owning task cancelled; this constraint cannot be resumed."""


@dataclass(frozen=True)
class ConstraintLimits:
    max_regions: int = 32
    max_references: int = 10
    max_caption_chars: int = 131072
    max_description_chars: int = 4096
    max_output_bytes: int = 256 * 1024
    max_new_tokens: int = 4096
    max_vocabulary_tokens: int = 300000
    max_token_bytes: int = 512
    max_vocabulary_bytes: int = 16 * 1024 * 1024
    max_trie_nodes: int = 2000000
    max_cache_bytes: int = 16 * 1024 * 1024
    build_seconds: float = 10.0
    lookup_seconds: float = 2.0
    max_request_seconds: float = 3600.0

    def __post_init__(self):
        ceilings = {
            "max_regions": 32, "max_references": 10,
            "max_caption_chars": 131072, "max_description_chars": 4096,
            "max_output_bytes": 256 * 1024, "max_new_tokens": 4096,
            "max_vocabulary_tokens": 300000, "max_token_bytes": 512,
            "max_vocabulary_bytes": 16 * 1024 * 1024,
            "max_trie_nodes": 2000000, "max_cache_bytes": 16 * 1024 * 1024,
        }
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= ceiling:
                raise ConstraintError(f"Invalid finite constraint bound: {name}.")
        for name, ceiling in (("build_seconds", 10), ("lookup_seconds", 2),
                              ("max_request_seconds", 3600)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= ceiling:
                raise ConstraintError(f"Invalid finite constraint bound: {name}.")


def _check_stop(should_stop, deadline):
    if should_stop is not None and should_stop():
        raise ConstraintCancelled("The owning task cancelled constrained generation.")
    if time.monotonic() >= deadline:
        raise ConstraintError("The constrained-generation time bound expired.")


class ByteVocabulary:
    """Immutable ordinary-token bytes and a compact trie; no decoding guesses.

    Aliases with identical bytes retain every token ID. Specials are never
    represented as ordinary text. Construction and storage have explicit caps;
    they do not replace the owning application's whole-process memory checks.
    """

    def __init__(self, token_bytes, *, special_token_ids, eos_token_ids,
                 limits=None, should_stop=None):
        self.limits = limits or ConstraintLimits()
        special, eos = frozenset(special_token_ids), frozenset(eos_token_ids)
        if (not eos or not eos <= special or any(type(i) is not int or not 0 <= i < 2**31
                                                for i in special)):
            raise ConstraintError("Declare exact nonnegative special and EOS token IDs.")
        if not hasattr(token_bytes, "items") or not 0 < len(token_bytes) <= self.limits.max_vocabulary_tokens:
            raise ConstraintError("The ordinary vocabulary size is outside its finite bound.")
        deadline = time.monotonic() + self.limits.build_seconds
        rows, payload, digest = [], 0, hashlib.sha256()
        for index, (identity, raw) in enumerate(token_bytes.items()):
            if index % 128 == 0:
                _check_stop(should_stop, deadline)
            if (type(identity) is not int or not 0 <= identity < 2**31 or identity in special
                    or type(raw) is not bytes or not 0 < len(raw) <= self.limits.max_token_bytes):
                raise ConstraintError("Ordinary token IDs require bounded nonempty exact bytes, excluding specials.")
            payload += len(raw)
            if payload > self.limits.max_vocabulary_bytes:
                raise ConstraintError("The ordinary vocabulary payload exceeds its finite bound.")
            rows.append((raw, identity))
        rows.sort()
        self._first, self._next, self._head = array("i", [-1]), array("i", [-1]), array("i", [-1])
        self._byte = array("B", [0])
        self._ids, self._terminal_next = array("i"), array("i")
        previous, stack = b"", [0]
        for index, (raw, identity) in enumerate(rows):
            if index % 128 == 0:
                _check_stop(should_stop, deadline)
            common = 0
            while common < min(len(previous), len(raw)) and previous[common] == raw[common]:
                common += 1
            del stack[common + 1:]
            for byte in raw[common:]:
                if len(self._byte) >= self.limits.max_trie_nodes:
                    raise ConstraintError("The token trie exceeds its finite node bound.")
                parent, node = stack[-1], len(self._byte)
                self._first.append(-1)
                self._next.append(self._first[parent])
                self._head.append(-1)
                self._byte.append(byte)
                self._first[parent] = node
                stack.append(node)
            node = stack[-1]
            self._terminal_next.append(self._head[node])
            self._head[node] = len(self._ids)
            self._ids.append(identity)
            digest.update(identity.to_bytes(4, "big") + len(raw).to_bytes(2, "big") + raw)
            previous = raw
        _check_stop(should_stop, deadline)
        digest.update(json.dumps({"version": CONSTRAINT_VERSION, "special": sorted(special),
                                  "eos": sorted(eos)}, separators=(",", ":")).encode("ascii"))
        self.token_bytes = MappingProxyType({identity: raw for raw, identity in rows})
        self.special_token_ids, self.eos_token_ids = special, eos
        self.max_token_bytes = max(map(len, self.token_bytes.values()))
        self.fingerprint = digest.hexdigest()
        self.trie_nodes = len(self._byte)
        self.trie_array_bytes = sum(len(a) * a.itemsize for a in
                                   (self._first, self._next, self._head, self._byte,
                                    self._ids, self._terminal_next))


# UTF-8 state: remaining continuation bytes, accumulated scalar, next min/max.
_UTF8_EMPTY = (0, 0, 0, 0)


def _unicode_byte(state, byte):
    remaining, scalar, low, high = state
    if remaining:
        if not low <= byte <= high:
            return None
        scalar = (scalar << 6) | (byte & 63)
        remaining -= 1
        return ((_UTF8_EMPTY, scalar) if not remaining else
                ((remaining, scalar, 0x80, 0xBF), None))
    if byte < 0x80:
        return _UTF8_EMPTY, byte
    if 0xC2 <= byte <= 0xDF:
        return (1, byte & 31, 0x80, 0xBF), None
    if 0xE0 <= byte <= 0xEF:
        return (2, byte & 15, 0xA0 if byte == 0xE0 else 0x80,
                0x9F if byte == 0xED else 0xBF), None
    if 0xF0 <= byte <= 0xF4:
        return (3, byte & 7, 0x90 if byte == 0xF0 else 0x80,
                0x8F if byte == 0xF4 else 0xBF), None
    return None


def _text_scalar(scalar):
    return (0 <= scalar <= 0x10FFFF and not 0xD800 <= scalar <= 0xDFFF
            and (unicodedata.category(chr(scalar)) != "Cc" or scalar in (9, 10, 13)))


def _box(value, *, nullable=False):
    if nullable and value is None:
        return
    if (type(value) is not list or len(value) != 4
            or any(type(x) is not int or not 0 <= x <= 1000 for x in value)
            or value[0] >= value[2] or value[1] >= value[3]):
        raise ConstraintError("Authoritative boxes must be exact positive integer 0..1000 coordinates.")


def _project_rows(rows, mode, image_count, limits):
    if (mode not in ("t2i_bbox", "i2i_bbox") or type(image_count) is not int
            or not 0 <= image_count <= limits.max_references
            or (mode == "t2i_bbox") != (image_count == 0)
            or type(rows) is not list or not 0 < len(rows) <= limits.max_regions):
        raise ConstraintError("An exact nonempty authored table and matching actual reference count are required.")
    required = {"id", "bbox", "desc"} if mode == "t2i_bbox" else {"id", "from", "src_bbox", "tgt_bbox", "desc"}
    identities, headers = [], []
    for row in rows:
        if type(row) is not dict or set(row) != required:
            raise ConstraintError("Authoritative rows require the exact declared keys.")
        identity = row["id"]
        if (type(identity) is not str or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", identity) is None
                or re.fullmatch(r"ref_image_[0-9]+", identity) or identity in identities):
            raise ConstraintError("Authored IDs must be unique and cannot impersonate reference IDs.")
        description = row["desc"]
        if (type(description) is not str or not description.strip()
                or len(description) > limits.max_description_chars
                or any(not _text_scalar(ord(char)) for char in description)):
            raise ConstraintError("The authored description is invalid or outside its finite bound.")
        if mode == "t2i_bbox":
            _box(row["bbox"])
            fixed = {"id": identity, "bbox": row["bbox"]}
        else:
            _box(row["tgt_bbox"], nullable=True)
            origin = row["from"]
            reference = re.fullmatch(r"ref_image_(0|[1-9][0-9]?)", origin) if type(origin) is str else None
            if origin == "target" and row["src_bbox"] is None and row["tgt_bbox"] is not None:
                pass
            elif reference is not None and int(reference[1]) < image_count:
                _box(row["src_bbox"])
            else:
                raise ConstraintError("Authoritative source mapping must name an actual supplied image or target placement.")
            fixed = {"id": identity, "from": origin, "src_bbox": row["src_bbox"], "tgt_bbox": row["tgt_bbox"]}
        # This is grammar data, never appended to or substituted for model output.
        header = json.dumps(fixed, ensure_ascii=True, allow_nan=False, separators=(",", ":"))[:-1] + ',"desc":"'
        headers.append(header.encode("ascii"))
        identities.append(identity)
    if sum(map(len, headers)) > 64 * 1024:
        raise ConstraintError("The authored fixed table exceeds its finite input bound.")
    return tuple(identities), tuple(headers)


class _State(NamedTuple):
    phase: str = "caption"
    seen: int = 0
    tag: bytes = b""
    segment: int = 0
    offset: int = 0
    utf8: tuple = _UTF8_EMPTY
    escape: int = 0
    digits: int = 0
    value: int = 0
    high_surrogate: int = 0
    nonwhite: bool = False
    caption_chars: int = 0
    description_chars: int = 0
    total_bytes: int = 0


class AuthoredLayoutPrefixConstraint:
    """One task-owned batch-one/beam-one official prefix callback.

    ``__call__(batch_id, input_ids)`` returns the allowed next token IDs. Input
    IDs may be a plain sequence or expose ``tolist``; no tensor library is
    imported. A changed prompt, forked/replayed continuation, cancellation or
    bound failure permanently refuses this instance. Use a fresh instance for
    each Preview/Regenerate. ``finalize`` requires a genuine terminal EOS and
    returns the exact emitted text, without table repair or normalization.
    """

    def __init__(self, vocabulary, rows, *, mode, image_count, prompt_token_ids,
                 deadline_monotonic, batch_size=1, num_beams=1, do_sample=False,
                 should_stop=None):
        if (not isinstance(vocabulary, ByteVocabulary) or type(batch_size) is not int
                or batch_size != 1 or type(num_beams) is not int or num_beams != 1
                or do_sample is not False):
            raise ConstraintError("Authored prefix constraints require explicit batch one, beam one and greedy decoding.")
        if (type(deadline_monotonic) not in (int, float) or not math.isfinite(deadline_monotonic)
                or deadline_monotonic <= time.monotonic()):
            raise ConstraintError("The existing owning-task deadline is required.")
        prompt = tuple(prompt_token_ids)
        if not 0 < len(prompt) <= 131072 or any(type(i) is not int or i < 0 for i in prompt):
            raise ConstraintError("The exact bounded prompt token prefix is required.")
        self.vocabulary, self.limits = vocabulary, vocabulary.limits
        self._deadline = min(deadline_monotonic, time.monotonic() + self.limits.max_request_seconds)
        self._should_stop, self._prompt = should_stop, prompt
        identities, headers = _project_rows(rows, mode, image_count, self.limits)
        self._all_seen = (1 << len(identities)) - 1
        self._tags = {b"<" + identity.encode("ascii") + b">": 1 << index
                      for index, identity in enumerate(identities)}
        self._tags.update({f"<ref_image_{index}>".encode("ascii"): 0 for index in range(image_count)})
        self._segments = ((b"[" + headers[0]), *tuple(b'"},' + header for header in headers[1:]), b'"}]')
        self._state, self._generated, self._failed = _State(), (), False
        self._cache, self._cache_bytes = OrderedDict(), 0
        self.lookup_node_visits = 0
        _check_stop(self._should_stop, self._deadline)

    def _step(self, state, byte):
        # One compact immutable state is constructed per accepted byte. In
        # particular, ordinary prose no longer copies fourteen dataclass fields
        # separately for byte count, UTF-8 progress and completed character.
        (phase, seen, tag, segment, offset, utf8, escape, digits, value,
         high_surrogate, nonwhite, caption_chars, description_chars, total_bytes) = state
        if total_bytes >= self.limits.max_output_bytes or phase == "done":
            return None
        total_bytes += 1
        scalar, description_character = None, False
        if phase == "fixed":
            literal = self._segments[segment]
            if byte != literal[offset]:
                return None
            offset += 1
            if offset == len(literal):
                offset = 0
                if segment == len(self._segments) - 1:
                    phase = "done"
                else:
                    phase, description_chars, nonwhite = "description", 0, False
                    utf8, escape, digits, value, high_surrogate = _UTF8_EMPTY, 0, 0, 0, 0
        elif phase == "tag":
            candidate = tag + bytes((byte,))
            available = [(name, bit) for name, bit in self._tags.items() if not bit or not seen & bit]
            if not any(name.startswith(candidate) for name, _bit in available):
                return None
            scalar, tag = byte, candidate
            for name, bit in available:
                if candidate == name:
                    phase, tag, seen = "caption", b"", seen | bit
                    break
        elif phase == "caption":
            structural = False
            if utf8 == _UTF8_EMPTY:
                if byte == 91:
                    if seen != self._all_seen or not nonwhite:
                        return None
                    phase, segment, offset, structural = "fixed", 0, 1, True
                elif byte == 60:
                    if not any(not bit or not seen & bit for bit in self._tags.values()):
                        return None
                    phase, tag, scalar, structural = "tag", b"<", byte, True
                elif byte in (62, 93):
                    return None
            if not structural:
                decoded = _unicode_byte(utf8, byte)
                if decoded is None:
                    return None
                utf8, scalar = decoded
        else:
            # Description JSON strings: exact UTF-8 or well-formed escapes.
            description_character = True
            if escape == 3:
                if byte != 92:
                    return None
                escape = 4
            elif escape == 4:
                if byte != 117:
                    return None
                escape, digits, value = 2, 0, 0
            elif escape == 2:
                if byte not in b"0123456789abcdefABCDEF":
                    return None
                value, digits = value * 16 + int(chr(byte), 16), digits + 1
                if digits == 4:
                    if high_surrogate:
                        if not 0xDC00 <= value <= 0xDFFF:
                            return None
                        scalar = 0x10000 + ((high_surrogate - 0xD800) << 10) + value - 0xDC00
                    elif 0xD800 <= value <= 0xDBFF:
                        escape, high_surrogate, digits, value = 3, value, 0, 0
                    elif 0xDC00 <= value <= 0xDFFF:
                        return None
                    else:
                        scalar = value
                    if scalar is not None:
                        escape, digits, value, high_surrogate = 0, 0, 0, 0
            elif escape == 1:
                if byte == 117:
                    escape, digits, value = 2, 0, 0
                else:
                    scalar = {34: 34, 92: 92, 47: 47, 110: 10, 114: 13, 116: 9}.get(byte)
                    if scalar is None:
                        return None
                    escape = 0
            else:
                structural = False
                if utf8 == _UTF8_EMPTY:
                    if byte == 34:
                        if not nonwhite:
                            return None
                        phase, segment, offset, structural = "fixed", segment + 1, 1, True
                    elif byte == 92:
                        escape, structural = 1, True
                    elif byte < 32:
                        return None
                if not structural:
                    decoded = _unicode_byte(utf8, byte)
                    if decoded is None:
                        return None
                    utf8, scalar = decoded
        if scalar is not None:
            if not _text_scalar(scalar) or scalar == 96:
                return None
            nonwhite = nonwhite or not chr(scalar).isspace()
            if description_character:
                description_chars += 1
                if description_chars > self.limits.max_description_chars:
                    return None
            else:
                caption_chars += 1
                if caption_chars > self.limits.max_caption_chars:
                    return None
        return _State(phase, seen, tag, segment, offset, utf8, escape, digits, value,
                      high_surrogate, nonwhite, caption_chars, description_chars, total_bytes)

    def _consume(self, state, raw):
        for index, byte in enumerate(raw):
            if index % 128 == 0:
                _check_stop(self._should_stop, self._deadline)
            state = self._step(state, byte)
            if state is None:
                return None
        return state

    def _cache_key(self, state):
        # Far from a length boundary, every possible next token has the same
        # eligibility regardless of prose length. Near the boundary retain the
        # exact counters. This avoids scanning the vocabulary per prose token.
        margin = self.vocabulary.max_token_bytes
        return state._replace(
                       total_bytes=state.total_bytes if self.limits.max_output_bytes - state.total_bytes <= margin else 0,
                       caption_chars=state.caption_chars if self.limits.max_caption_chars - state.caption_chars <= margin else 0,
                       description_chars=state.description_chars if self.limits.max_description_chars - state.description_chars <= margin else 0)

    def _allowed(self, state):
        if state.phase == "done":
            return tuple(sorted(self.vocabulary.eos_token_ids))
        key = self._cache_key(state)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][0]
        deadline = min(self._deadline, time.monotonic() + self.limits.lookup_seconds)
        result, stack, visits = [], [(0, state)], 0
        vocab = self.vocabulary
        while stack:
            node, parent_state = stack.pop()
            child = vocab._first[node]
            while child != -1:
                visits += 1
                if visits % 128 == 0:
                    _check_stop(self._should_stop, deadline)
                candidate = self._step(parent_state, vocab._byte[child])
                if candidate is not None:
                    terminal = vocab._head[child]
                    while terminal != -1:
                        result.append(vocab._ids[terminal])
                        terminal = vocab._terminal_next[terminal]
                    if vocab._first[child] != -1:
                        stack.append((child, candidate))
                child = vocab._next[child]
        _check_stop(self._should_stop, deadline)
        self.lookup_node_visits += visits
        allowed = tuple(sorted(result))
        if not allowed:
            raise ConstraintError("No ordinary token can continue this exact authored grammar.")
        # Conservative Python tuple/int/key overhead accounting, not just payload.
        cost = 1024 + len(allowed) * 48
        if cost <= self.limits.max_cache_bytes:
            while self._cache and self._cache_bytes + cost > self.limits.max_cache_bytes:
                _old_key, (_old_value, old_cost) = self._cache.popitem(last=False)
                self._cache_bytes -= old_cost
            self._cache[key] = allowed, cost
            self._cache_bytes += cost
        return allowed

    def __call__(self, batch_id, input_ids):
        try:
            if self._failed:
                raise ConstraintError("A refused constrained-generation instance cannot be reused.")
            _check_stop(self._should_stop, self._deadline)
            if type(batch_id) is not int or batch_id != 0:
                raise ConstraintError("Only the owning single batch is supported.")
            ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
            if (len(ids) > len(self._prompt) + self.limits.max_new_tokens
                    or any(type(identity) is not int for identity in ids)
                    or tuple(ids[:len(self._prompt)]) != self._prompt):
                raise ConstraintError("The exact owning prompt prefix changed or exceeded its bound.")
            generated = tuple(ids[len(self._prompt):])
            if generated[:len(self._generated)] != self._generated or len(generated) < len(self._generated):
                raise ConstraintError("Forked or replayed continuations require a fresh constraint instance.")
            for identity in generated[len(self._generated):]:
                raw = self.vocabulary.token_bytes.get(identity)
                if raw is None:
                    raise ConstraintError("Special/unknown tokens cannot enter a live text continuation.")
                self._state = self._consume(self._state, raw)
                if self._state is None:
                    raise ConstraintError("The emitted continuation violates the exact authored grammar.")
            self._generated = generated
            if len(generated) >= self.limits.max_new_tokens:
                raise ConstraintError("The generation token bound was reached before terminal EOS.")
            return list(self._allowed(self._state))
        except Exception:
            self._failed = True
            raise

    def finalize(self, generated_token_ids):
        """Verify genuine terminal EOS; return raw emitted UTF-8, never repaired."""
        try:
            if self._failed:
                raise ConstraintError("A refused constrained-generation instance cannot be finalized.")
            _check_stop(self._should_stop, self._deadline)
            ids = tuple(generated_token_ids)
            if (not ids or len(ids) > self.limits.max_new_tokens
                    or any(type(identity) is not int for identity in ids)
                    or ids[-1] not in self.vocabulary.eos_token_ids):
                raise ConstraintError("Incomplete/truncated output without genuine terminal EOS is refused.")
            state, chunks = _State(), []
            for identity in ids[:-1]:
                raw = self.vocabulary.token_bytes.get(identity)
                if raw is None:
                    raise ConstraintError("Unexpected special/unknown token in generated output.")
                state = self._consume(state, raw)
                if state is None:
                    raise ConstraintError("Generated output violates the exact authored grammar.")
                chunks.append(raw)
            if state.phase != "done" or tuple(ids[:-1])[:len(self._generated)] != self._generated:
                raise ConstraintError("Only the same complete model-emitted table can be finalized.")
            return b"".join(chunks).decode("utf-8", errors="strict")
        except Exception:
            self._failed = True
            raise
