"""Immutable app-managed defaults for generic upscaler nodes."""

REAL_ESRGAN_X2_REPO = "nateraw/real-esrgan"
REAL_ESRGAN_X2_FILENAME = "RealESRGAN_x2plus.pth"
REAL_ESRGAN_X2_REVISION = "42efb9c3eeed1f5c0c8a626cf5f7f4481dfbb094"
REAL_ESRGAN_X2_SHA256 = "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
REAL_ESRGAN_X2_BYTE_SIZE = 67_061_725
REAL_ESRGAN_X2_LICENSE = "bsd-3-clause"


def real_esrgan_x2_model_selection() -> dict[str, object]:
    """Return a fresh model-selector value for the reviewed native x2 weight."""

    return {
        "source": "hub",
        "value": f"{REAL_ESRGAN_X2_REPO}/{REAL_ESRGAN_X2_FILENAME}",
        "revision": REAL_ESRGAN_X2_REVISION,
        "sha256": REAL_ESRGAN_X2_SHA256,
        "byteSize": REAL_ESRGAN_X2_BYTE_SIZE,
        "license": REAL_ESRGAN_X2_LICENSE,
    }
