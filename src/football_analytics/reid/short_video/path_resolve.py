"""Resolve Windows / WSL paths for short-video product inputs (read-only)."""

from __future__ import annotations

import os
from pathlib import Path


class ShortVideoInputError(RuntimeError):
    """Raised when the short video cannot be resolved."""


def windows_to_wsl_candidates(windows_path: str) -> list[Path]:
    """Expand a Windows path into candidate WSL absolute paths."""
    text = windows_path.strip().strip('"').strip("'")
    # Normalize backslashes
    norm = text.replace("/", "\\")
    candidates: list[Path] = []

    # wslpath if available
    try:
        import subprocess

        out = subprocess.check_output(
            ["wslpath", "-u", text], text=True, stderr=subprocess.DEVNULL
        ).strip()
        if out:
            candidates.append(Path(out))
    except (OSError, subprocess.CalledProcessError):
        pass

    if len(norm) >= 3 and norm[1] == ":" and norm[2] == "\\":
        drive = norm[0].lower()
        rest = norm[3:].replace("\\", "/")
        candidates.append(Path(f"/mnt/{drive}/{rest}"))

    # Common username mismatch: stated Users/<name> vs actual Windows profile folder
    filename = Path(norm.replace("\\", "/")).name
    downloads_glob = list(Path("/mnt/c/Users").glob(f"*/Downloads/{filename}"))
    candidates.extend(downloads_glob)

    # Deduplicate preserving order
    seen: set[str] = set()
    out_paths: list[Path] = []
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out_paths.append(p)
    return out_paths


def resolve_short_video_path(windows_or_wsl: str) -> Path:
    """Resolve user-provided path to an existing file; never invent another clip."""
    text = windows_or_wsl.strip().strip('"').strip("'")
    direct = Path(text).expanduser()
    if direct.is_file():
        return direct.resolve()

    for cand in windows_to_wsl_candidates(text):
        if cand.is_file():
            return cand.resolve()

    # Filename-only search under Downloads (still require exact basename)
    basename = Path(text.replace("\\", "/")).name
    if basename and basename.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
        hits = list(Path("/mnt/c/Users").glob(f"*/Downloads/{basename}"))
        hits = [h for h in hits if h.is_file()]
        if len(hits) == 1:
            return hits[0].resolve()
        if len(hits) > 1:
            raise ShortVideoInputError(
                f"multiple Downloads hits for {basename}: {[str(h) for h in hits]}"
            )

    raise ShortVideoInputError(
        f"BLOCKED_SHORT_VIDEO_INPUT_NOT_PROVIDED: unresolved path {windows_or_wsl!r}"
    )


def copy_to_product_input(source: Path, dest_dir: Path) -> Path:
    """Copy source into isolated product input (does not modify original)."""
    import shutil

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.name
    if dest.exists():
        # Reuse identical copy only
        from football_analytics.ingest.checksum import sha256_file

        if sha256_file(dest) == sha256_file(source) and dest.stat().st_size == source.stat().st_size:
            return dest.resolve()
        raise ShortVideoInputError(f"product input already exists with different content: {dest}")
    shutil.copy2(source, dest)
    # Ensure we never write back to Windows source
    os.chmod(dest, 0o444)
    return dest.resolve()
