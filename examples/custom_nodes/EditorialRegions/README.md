# Editorial Regions

A model-independent custom image node with editable polygon mask algebra (union,
subtraction, intersection), feathering, linear-light exposure, opposed color gels
and saturation. It outputs an RGB image, L mask and diagnostics. Coordinates are
normalized image coordinates; this is explicit art direction, **not segmentation**.

Stage this directory in Nodes → Custom nodes, inspect and consent to Enable code.
Connect decoded images to Source image. Draw a polygon by editing the JSON; add a
subtract polygon to protect a face or another region. White mask pixels are editable.
Connect Directed image and Edit mask to a compatible inpaint operation.

Use the second node, Protected Editorial Composite, after diffusion: connect the
original image, generated edit and the same mask. Pixels with zero mask remain
byte-identical to the RGB original. A protected region is not identity conditioning
inside the mask. Inspect feathered seams and the generated result before presenting.

Bounds: one image (or batch of one), 4,194,304 pixels, 16 polygons, 32 points per
polygon, 16KiB JSON. No filesystem/network/model loading and no new dependencies;
uses existing Pillow and NumPy. The source image is not modified in place.
