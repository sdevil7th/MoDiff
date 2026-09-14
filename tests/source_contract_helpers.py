"""Source-code receipts identify Git text, independently of checkout newlines.

This normalization is only for upstream Python source comparisons. Wheel,
archive, model-weight and sealed-overlay integrity checks remain byte-exact.
"""

import hashlib


def source_sha256(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
