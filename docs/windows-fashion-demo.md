# Windows fashion demo setup

The backend ships the matching built frontend. Pulling MoDiff is sufficient to
test the application; the separate client checkout is only needed for development
and recording/rehearsal scripts. Use the paired revision identifiers supplied
with the handoff, not a mixture of old and new contracts.

## Update and launch

In your existing MoDiff checkout, preserve uncommitted local work before switching:

```powershell
git fetch origin
git switch feat/generic-diffusers-workbench
git pull --ff-only
.\install.ps1 -Accelerator nvidia
.\run.ps1
```

The installer provisions the reviewed Windows environment; do not copy a Linux
virtual environment or optional-runtime directory. Open the local address shown
by the launcher. Use Setup's explicit optional-runtime installation/activation
when a selected workflow requires it, then follow any restart instruction.
Opening a workflow or enabling Developer mode does not grant code consent.

## Transfer the curated chapters

The separately supplied `fashion-demo-windows.zip` contains `manifest.json`,
thirteen `workflows/*.workflow.json` files and `data/fashion-demo/` images. Git does
not contain these generated images, model weights or machine-specific approvals.
Extract the archive into a new folder. Compare the archive SHA-256 with the
handoff receipt before using it:

```powershell
Get-FileHash .\fashion-demo-windows.zip -Algorithm SHA256
```

1. Copy the extracted `data/fashion-demo` directory into your MoDiff checkout's
   `data` directory. Keep the `fashion-demo` directory name: workflow paths use
   `@data/fashion-demo/<image-hash>.<extension>`. Files are content-addressed;
   don't replace an existing different file under the same name.
2. In **Nodes → Custom nodes**, stage
   `examples/custom_nodes/EditorialRegions` with module name
   **FashionEditorialRegions**. Inspect the source/dependencies, then explicitly
   enable it. This exact name matches the saved graphs. Do not copy another
   machine's approval database.
3. Enter Developer mode. Drag one numbered workflow JSON onto an empty canvas.
   Use **Save As** with the corresponding title from `manifest.json`. Repeat for
   the chapters you want to present. Import does not download or load models.
4. Open each saved chapter, inspect its image full-size, refresh and reopen.
   Check the custom fields, prompt, edges and source checkpoint. A missing image
   means the media directory is misplaced, not that it should be regenerated.
5. Before Run, use Model Manager/Setup to resolve the exact declared model and
   runtime requirements. Review model license/access requirements with your own
   account. The package contains no access token.

## RTX 4080, 16 GB VRAM, 32 GB system RAM

Start with chapter 11's CPU-only custom processing to verify the transfer and
extension, then chapter 01 (Z-Image Turbo) and a Klein 4B editing chapter. The
curated generation recipes use batch one, BF16 and model-CPU offload. Monitor
both system RAM and VRAM; CPU offload is not unlimited memory.

The saved images are 1024×1024. For an initial hardware smoke test, make a
**Save As** copy and lower resolution if necessary. Model alignment requirements
still apply. Review region placement against the new image; the curated polygons
were authored for the retained composition, not arbitrary regenerated people.

Kontext and Qwen Image 2.1 chapters are optional live chapters on this machine:
their successful Linux runs do not establish that these full-weight recipes fit
within 16 GB VRAM / 32 GB RAM on Windows. It is valid to inspect their retained
results and graph transitions without claiming live Windows qualification.
Do not silently substitute a different quantization or model revision.

For presentation, follow the [13-chapter guide](fashion-editorial-demo.md). Label
retained results and omitted generation waits explicitly. Exact pixel protection
applies outside the compositor mask; it is not an identity guarantee inside a
generative edit.
