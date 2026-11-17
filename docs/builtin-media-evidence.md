# Deterministic media execution evidence

Ordinary image, audio and video utilities use the existing graph executor.
They require no model checkpoint and do not acquire model, Auto, hardware or
public-asset approval from a successful processing result.

For an isolated qualification process, explicitly set
`MODIFF_CAPTURE_BUILTIN_MEDIA=1` before starting the backend. It is off by
default. The supported ordinary loaders/processors/exporters then attach an
optional, node-local `mediaEvidence` section to their existing resolved-input
receipt. The legacy generation summary and field accounting remain separate.

The capture records normalized node arguments, ordered input/output identities,
and root file identities. Image identities bind dimensions, mode and raw pixels.
Audio identities bind frames-first little-endian float32 samples, shape and
sample rate. File identities bind encoded bytes and their size. These are
observations, not a claim that every exposed control is consumed by every task.

Image Load decodes the same snapshot whose original bytes were hashed. Captured
WAV loading likewise hashes and decodes one byte snapshot. Compressed-audio
conversion remains marked as converted; its decoded WAV does not masquerade as
an execution-bound original-container receipt. Qualification must reject such
an unresolved root binding.

For captured literal video paths, Process Video streams each source once into
an owned temporary snapshot while hashing it; FFmpeg consumes only that snapshot.
Snapshots are deleted after the call. No original input is changed. Existing
structured retained-asset behavior is preserved and is not silently relabelled
as captured file-input evidence.

Capture is bounded to 16 members per port, 256 MiB per file, 64 MiB of raw pixel
or sample bytes per member, and 32 KiB of metadata per node. Unsupported or
oversized evidence is explicit rather than a fabricated complete receipt.
These capture limits do not constrain ordinary user workflows. Optional metadata
is budgeted after existing receipt sections; omission sets
`mediaEvidenceStatus.truncated` without evicting legacy node records or summaries.

The paired client's canonical recorder supports a separate
`modiff.canonical-builtin-media-evidence.v1` descriptor. It retains baseline and
explicit exercise runs, root bytes, original accepted graphs, native task and
completion messages, source/runtime identities and every output. It rechecks
all files, normalized controls, ordered wire joins and actual consumed root
identities. Independent small-fixture checks cover exact image pixels, PCM16
sample tolerances, video frame order/count/rate and codec tolerances. Retaining a
registry row alone is insufficient; release projection repeats validation.

Image preview encodings can differ from raw pixels. Both the exact raw PNG and
the native durable preview must be checked. Video export can normalize dimensions
to its codec's macroblock boundary, and its explicit output FPS can retime an
input clip. Record the actual delivered geometry, frame count and rate; do not
infer them from the source. Silent video fixtures make no audio-preservation claim.

All records and temporary snapshots remain local plaintext under the configured
storage boundary. Keep private qualification workspaces and generated media out
of Git. Public publication and subjective quality/rights review retain their
separate existing approval requirements.
