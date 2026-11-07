"""Canonical bytes for upstream Python source-code contracts.

Git may check out the same immutable source revision with LF or CRLF. Only
Python source comparisons normalize those newlines; artifact, wheel, model and
sealed-overlay hashes continue to identify exact stored bytes.
"""


def canonical_python_source(body: bytes) -> bytes:
    body.decode("utf-8")
    return body.replace(b"\r\n", b"\n")
