"""SHA-256 checksum helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path, chunk_size: int = _CHUNK_SIZE) -> str:
    """Return the hex SHA-256 digest of a file without loading it entirely into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
