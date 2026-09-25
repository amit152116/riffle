"""Audio identity.

Identity is the hash of the encoded audio stream, not of the file. Stream
copy means no decode, so this is I/O-bound, and tag edits do not change it.
A container that cannot be stream-copied falls back to a whole-file hash,
which is a weaker guarantee and is recorded as such.
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


class HashError(Exception):
    """The file could not be hashed."""


@dataclass(frozen=True)
class HashResult:
    audio_hash: str
    hash_method: str


def _streamhash(path: Path) -> str | None:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", str(path), "-map", "0:a", "-c:a", "copy",
         "-f", "streamhash", "-hash", "sha256", "-"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    # Lines look like: 0,a,SHA256=<hex>
    for line in proc.stdout.splitlines():
        if "=" in line:
            digest = line.rsplit("=", 1)[1].strip()
            if len(digest) == 64:
                return digest
    return None


def whole_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audio_identity(path: Path) -> HashResult:
    path = Path(path)
    if not path.exists():
        raise HashError(f"no such file: {path}")

    digest = _streamhash(path)
    if digest is not None:
        return HashResult(digest, "streamhash")

    # No decodable audio stream at all is an error, not a fallback case.
    # codec_name alone is not enough: ffprobe guesses a demuxer from the file
    # extension and will report e.g. codec_name=flac for garbage bytes named
    # *.flac, with sample_rate=0. Requiring a positive sample_rate is what
    # actually distinguishes real audio from an extension-based guess.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_name,sample_rate",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    lines = probe.stdout.split()
    has_codec = len(lines) >= 1 and lines[0].strip() not in ("", "N/A")
    has_rate = len(lines) >= 2 and lines[1].strip().isdigit() and int(lines[1]) > 0
    if probe.returncode != 0 or not (has_codec and has_rate):
        raise HashError(f"no audio stream in {path}")

    return HashResult(whole_file_sha256(path), "whole_file")
