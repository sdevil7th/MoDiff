# Source Provenance Map

This map supplements [the third-party notices](../THIRD_PARTY_NOTICES.md). It
records inherited path families and the engineering treatment used to make
MoDiff modifications visible; it does not replace the repository license or
legal review.

## Comparison baselines

- Backend: [`cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f`](https://github.com/cubiq/Mellon/tree/5fd242921d13bff9fb03f4de405fdd39c2335e1f), Copyright 2024 Matteo Spinelli.
- Historical generated client bundle: [`cubiq/Mellon-client@af0c5801f843453a1700733596e99fe6589b2e86`](https://github.com/cubiq/Mellon-client/tree/af0c5801f843453a1700733596e99fe6589b2e86), Copyright 2024 Matteo Spinelli.

The commits above are evidence-based pre-import comparison baselines selected
from repository history and file similarity. They are not proven Git ancestors
and are not claimed to be the exact revisions used for the original import.

## Adapted paths

| Treatment | Paths |
| --- | --- |
| Prominent in-file modification comment | `README.md`, `run.sh`, `main.py`, `modiff/{NodeBase,client,config,modelstore,server}.py`, `config.example.ini`, `pyproject.toml`, `modules/__init__.py`, retained adapted sources under `modules/{Color,Image,ImageFilters,ModularDiffusers,Primitive,Spandrel,Tensor,Text,Video}/`, and `utils/{huggingface,memory_menager,paths,torch_utils}.py` |
| Project-level notice because JSON has no comments | `data/graphs/modular_diffusers/{dynamic_node,image_to_image,multiple_image_edit,quantization,text_to_image}.json` |
| Project-level notice because files are generated or binary | historical files under `web/`; these must be regenerated from the compatible client rather than hand-edited |
| No MoDiff-modification header because the current file matched the comparison baseline byte-for-byte during the audit | `.python-version`, `LICENSE`, `custom/.gitkeep`, `utils/image.py`, and `modules/ModularDiffusers/main.py` |

New MoDiff-only files are outside the inherited path list. The separately
adapted Hugging Face Diffusers helper is documented in
[the third-party notices](../THIRD_PARTY_NOTICES.md) and in its own source
header.

## Commentless formats

Adding comments to JSON or generated bundle files would invalidate their
format, schema, integrity, or reproducibility. The repository-level notice and
this map make those modifications visible without changing the artifacts.
Whether that project-level treatment alone satisfies Apache-2.0 section 4(b)
for every commentless modified file remains a legal-review question; it is not
resolved by this engineering documentation.
