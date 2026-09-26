# Light & Palette Director

Stage this directory using **Nodes → Custom nodes**, review the source and enable
its exact hash. It is ordinary operator-approved Python, not a model or a prompt
wrapper. It uses the existing Pillow/NumPy dependencies and normal graph executor.

Connect any generator's **decoded images** to Image. Directed image is RGB;
Soft mask is L (white = editable region). Connect these to an image/mask-capable
refinement operation, and optionally to separate Preview nodes. The elliptical
region, feather, three-color luminance palette and directional gradient are
deterministic CPU operations. This is geometric art direction, not semantic
segmentation or physically accurate relighting. Transparent inputs are composited
on white explicitly. Each image is limited to 4,194,304 pixels and batches to four.
Coordinates/radii are normalized to image dimensions. Strength zero preserves the
RGB source exactly but does not disable the mask; refinement strength is separate.

For a code-reload demonstration, edit the **installed** module's `main.py`, replace
the smoothstep line with `mask = mask ** 2`, then Review reload and approve the new
hash. Ports and saved settings stay unchanged. Do not edit a staged source copy
and expect the already-installed module to change. Changing code never grants
approval automatically; importing a saved workflow does not grant it either.
