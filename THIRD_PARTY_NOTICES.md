# Third-party notices

## Mellon and Mellon Client

Substantial portions of the backend source are adapted from
[`cubiq/Mellon`](https://github.com/cubiq/Mellon) at comparison baseline
[`5fd242921d13bff9fb03f4de405fdd39c2335e1f`](https://github.com/cubiq/Mellon/tree/5fd242921d13bff9fb03f4de405fdd39c2335e1f),
Copyright 2024 Matteo Spinelli. Historical generated client files under `web/`
are adapted from
[`cubiq/Mellon-client`](https://github.com/cubiq/Mellon-client) at comparison
baseline
[`af0c5801f843453a1700733596e99fe6589b2e86`](https://github.com/cubiq/Mellon-client/tree/af0c5801f843453a1700733596e99fe6589b2e86),
Copyright 2024 Matteo Spinelli. Both upstream repositories are licensed under
the Apache License 2.0; the project `LICENSE` contains that license text.

These revisions are evidence-based pre-import comparison baselines selected
from repository history and file similarity. They are not asserted to be
proven Git ancestors or necessarily the exact snapshots used for MoDiff's
original import. MoDiff has modified the adapted files. See the
[source-provenance map](docs/source-provenance.md) for the affected path
families and the treatment of formats that cannot carry comments.

## Hugging Face Diffusers

Portions of `modules/ModularDiffusers/pipeline_schema.py` are derived from
Hugging Face Diffusers' `src/diffusers/modular_pipelines/mellon_node_utils.py`
at commit `bb56997d4b7e87f0743f26a612f49ec4e7ce7213`. Diffusers is licensed under
the Apache License 2.0; the project `LICENSE` contains that license text.

Source: <https://github.com/huggingface/diffusers/blob/bb56997d4b7e87f0743f26a612f49ec4e7ce7213/src/diffusers/modular_pipelines/mellon_node_utils.py>

That commit remains the provenance of the adapted source. The separately
reviewed executable Diffusers dependency is currently pinned to
`90b4e34e79a86ec5e7f2437634fe95ecd2108796`.

## Comfy workflow template research metadata

`data/research/comfy-workflow-catalog.v1.json` contains classified catalog
metadata derived from
[`Comfy-Org/workflow_templates`](https://github.com/Comfy-Org/workflow_templates)
at commit `d9e66019b85da231b7c936ad9cb7ff08cec16557`, Copyright (c) 2023-present
Comfy Org. The upstream catalog is licensed under the MIT License. MoDiff does
not copy or execute the catalog's workflow graphs; the ledger records only
source provenance, classifications, and conservative semantic candidates.

MIT License

Copyright (c) 2023-present Comfy Org

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Janus-Pro-1B model weights

MoDiff's reviewed artifact catalog can acquire
[`deepseek-community/Janus-Pro-1B`](https://huggingface.co/deepseek-community/Janus-Pro-1B)
at immutable revision `1655280bb75959cc1cb85529a2a8b26e7016072e` through the
app. The repository distinguishes its MIT-licensed code from its model
weights. The weights are governed by the retained **DEEPSEEK LICENSE
AGREEMENT, Version 1.0** (identified in MoDiff policy metadata as the DeepSeek
Model License Agreement v1.0), not the MIT License. That agreement includes use
restrictions and
notice, distribution, hosted-use, and downstream-compliance obligations.

The exact reviewed license text is retained at
`licenses/DeepSeek-Model-License-1.0.txt`. It is sourced from
[`deepseek-ai/DeepSeek-LLM`](https://github.com/deepseek-ai/DeepSeek-LLM/blob/6712a86bfb7dd25c73383c5ad2eb7a8db540258b/LICENSE-MODEL)
at revision `6712a86bfb7dd25c73383c5ad2eb7a8db540258b` and has SHA-256
`09b2b4b4614509ff8baccd3c220e9d9b99e55152925b501f0d5f8b5e36eea982`.
Catalog availability and a user's acknowledgement do not constitute product or
legal approval; product and user compliance review remains required.

## Font software

MoDiff redistributes the following font software in source builds and/or the
bundled web client. These fonts are not licensed under MoDiff's Apache-2.0
license.

- **IBM Plex Mono**, supplied by `@fontsource/ibm-plex-mono` 5.2.7.
  The package notice says: Copyright 2017 IBM Corp. All rights reserved.
  The authoritative IBM Plex notice is: Copyright © 2017 IBM Corp. with
  Reserved Font Name "Plex".
- **Source Sans Pro**, supplied by `@fontsource/source-sans-pro` 5.2.5.
  The package notice says: Google Inc. The authoritative Source Sans notice
  is: Copyright 2010-2024 Adobe (<http://www.adobe.com/>), with Reserved Font
  Name "Source". All Rights Reserved. Source is a trademark of Adobe in the
  United States and/or other countries.

Both published packages declare the SIL Open Font License, Version 1.1.

## SIL Open Font License, Version 1.1

Version 1.1 - 26 February 2007

### Preamble

The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font creation
efforts of academic and linguistic communities, and to provide a free and
open framework in which fonts may be shared and improved in partnership
with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The fonts,
including any derivative works, can be bundled, embedded, redistributed
and/or sold with any software provided that any reserved names are not used
by derivative works. The fonts and derivatives, however, cannot be released
under any other type of license. The requirement for fonts to remain under
this license does not apply to any document created using the fonts or their
derivatives.

### Definitions

"Font Software" refers to the set of files released by the Copyright
Holder(s) under this license and clearly marked as such. This may include
source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the
copyright statement(s).

"Original Version" refers to the collection of Font Software components as
distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting, or
substituting -- in part or in whole -- any of the components of the Original
Version, by changing formats or by porting the Font Software to a new
environment.

"Author" refers to any designer, engineer, programmer, technical writer or
other person who contributed to the Font Software.

### Permission and conditions

Permission is hereby granted, free of charge, to any person obtaining a copy
of the Font Software, to use, study, copy, merge, embed, modify, redistribute,
and sell modified and unmodified copies of the Font Software, subject to the
following conditions:

1. Neither the Font Software nor any of its individual components, in
   Original or Modified Versions, may be sold by itself.
2. Original or Modified Versions of the Font Software may be bundled,
   redistributed and/or sold with any software, provided that each copy
   contains the above copyright notice and this license. These can be
   included either as stand-alone text files, human-readable headers or in
   the appropriate machine-readable metadata fields within text or binary
   files as long as those fields can be easily viewed by the user.
3. No Modified Version of the Font Software may use the Reserved Font Name(s)
   unless explicit written permission is granted by the corresponding
   Copyright Holder. This restriction only applies to the primary font name
   as presented to the users.
4. The name(s) of the Copyright Holder(s) or the Author(s) of the Font
   Software shall not be used to promote, endorse or advertise any Modified
   Version, except to acknowledge the contribution(s) of the Copyright
   Holder(s) and the Author(s) or with their explicit written permission.
5. The Font Software, modified or unmodified, in part or in whole, must be
   distributed entirely under this license, and must not be distributed
   under any other license. The requirement for fonts to remain under this
   license does not apply to any document created using the Font Software.

### Termination

This license becomes null and void if any of the above conditions are not
met.

### Disclaimer

THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF COPYRIGHT, PATENT,
TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, INCLUDING ANY GENERAL, SPECIAL,
INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF THE USE OR INABILITY TO USE
THE FONT SOFTWARE OR FROM OTHER DEALINGS IN THE FONT SOFTWARE.
