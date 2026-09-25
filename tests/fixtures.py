"""ffmpeg-backed audio fixtures.

A constant tone fingerprints degenerately, so the generated signal sweeps in
frequency and pulses in amplitude to give Chromaprint something to work with.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import mutagen


def _run(args: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
    )


def make_tone(path: Path, seconds: float, freq: int = 440,
              codec: str = "flac", bitrate: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    expr = (
        f"sin(2*PI*t*({freq}+{freq // 2}*sin(2*PI*t/7)))"
        f"*(0.4+0.3*sin(2*PI*t*1.7))"
    )
    args = [
        "-f", "lavfi",
        "-i", f"aevalsrc={expr}:s=44100:d={seconds}",
        "-ac", "1", "-c:a", codec,
    ]
    if bitrate:
        args += ["-b:a", bitrate]
    _run(args + [str(path)])
    return path


def make_silence(path: Path, seconds: float, codec: str = "flac") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono:d={seconds}",
        "-c:a", codec, str(path),
    ])
    return path


def transcode(src: Path, dst: Path, codec: str,
              bitrate: str | None = None) -> Path:
    args = ["-i", str(src), "-ac", "1", "-c:a", codec]
    if bitrate:
        args += ["-b:a", bitrate]
    _run(args + [str(dst)])
    return dst


def trim(src: Path, dst: Path, start: float,
         duration: float | None = None) -> Path:
    args = ["-i", str(src), "-ss", str(start)]
    if duration is not None:
        args += ["-t", str(duration)]
    _run(args + ["-c:a", "flac", str(dst)])
    return dst


def concat(dst: Path, *srcs: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    listing = dst.with_suffix(".txt")
    listing.write_text("".join(f"file '{s.resolve()}'\n" for s in srcs))
    _run(["-f", "concat", "-safe", "0", "-i", str(listing),
          "-c:a", "flac", str(dst)])
    listing.unlink()
    return dst


def retag(path: Path, **tags: str) -> Path:
    f = mutagen.File(path, easy=True)
    for k, v in tags.items():
        f[k] = v
    f.save()
    return path
