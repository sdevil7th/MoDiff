# Explicit constrained wording for authored layouts

This Expert-only source contract is separate from raw wording, strict generated
tables and append-details. It is not enabled by a default, Auto, template or
Gallery grant. The four versioned parent recipes use the current official
provider profiles. Existing private recipes, adapter declarations, system
prompts and their receipts are unchanged.

The operation `enhance_prompt_constrained_authored_v1` requires a nonempty
authored layout and the full original caption and ordered references. An
official Transformers `prefix_allowed_tokens_fn` limits the same model's next
token choices to a caption and table preserving every authored ID, source,
action, box, null and row order. Only wording and descriptions are generated.
The model must emit every output byte. No missing table is appended, geometry
repaired, or output silently replaced. The independent existing strict parser
must still accept the complete raw result before Apply can be offered.

## Runtime and resource boundary

The explicit recipe fixes FP32, `cuda:0`, no model quantization, greedy decoding,
one batch and one beam, zero minimum new tokens and at most the existing 3584
new-token limit. It does not change the diffusion model's independent storage
format. It is not a fit claim for a 32 GiB Windows host, NF4 or every device.

The request validates the selected official processor/tokenizer against the
immutable adapter revision and three bounded local metadata files. Ordinary
BPE token bytes, ByteLevel decoding, all added controls, real special flags,
runtime versions and native EOS configuration must match. No download,
installation, arbitrary tokenizer, remote code or custom generation function
is introduced. The native EOS list remains `[248046,248044]`; the constraint
permits only terminal `248046`, after the table is complete. All 33 added
controls remain excluded from prose, including 12 not marked special by the
tokenizer. Full ordinary decoding must agree byte-for-byte with the grammar.

The exact Qwen3.5 tokenizer class in Transformers 5.18 reconstructs the source
preprocessing regex to include Unicode combining marks. The contract therefore
pins the immutable source preprocessing object and the effective runtime object
separately. The model, decoder, normalizer, postprocessor and added-token objects
must remain exactly equal, and the complete effective semantic hash must match.
This accepts only that reviewed pair; it neither changes the tokenizer nor
normalizes or repairs generated output.

Each explicit preallocation starts one source-owned 1800 second cooperative
window, carried through CPU preparation, model loading, generation and final
validation. The task's normal Stop and exact ownership checks remain active.
Cooperative checks cannot preempt a single stuck device kernel. A fresh request
owns its vocabulary, prefix state and cache; they are released in `finally`.
There is no global vocabulary cache or mutation of the retained model.

The new preparation reserves an additional 256 MiB of RAM. Independent hard
module ceilings retain a 16 MiB vocabulary payload and mask cache, two million
trie nodes, ten second compilation and two second allowed-token lookup limits,
32 authored regions, ten references and 256 KiB output. These are refusal
bounds, not a guarantee that all 32 rows can complete within 3584 tokens.

## Proposal and proof boundary

Only this operation adds the mandatory `authoredLayoutConstraint` contract,
`consumedInput.fingerprints.authored_layout_constraint` and
`proposal.constraintCompletion`. They bind independently computed source and
tokenizer authority, exact projected rows and actual prompt IDs to the current
producer's consumed-input receipt. Completion records the actual final EOS,
full untrimmed raw-output hash and generated count. A verified final EOS at
the token bound is a stop; a missing EOS or incomplete table is a refusal.
Old operations retain their exact interfaces and finish conventions.

Format constraints do not prove useful or faithful wording, generated geometry,
model image quality, hardware fit, Auto readiness or native acceptance. The
normal source/current-input checks, explicit Apply, fresh Regenerate, atomic
Undo, Save/cold reload and Release remain required. Pure tokenizer/grammar and
synthetic API tests are reported separately from any later real model output.
