import shutil
import subprocess

import audiolib


def test_package_imports():
    assert audiolib.__version__ == "0.1.0"


def test_ffmpeg_available():
    assert shutil.which("ffmpeg") is not None


def test_fpcalc_available():
    assert shutil.which("fpcalc") is not None


def test_streamhash_muxer_available():
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-muxers"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "streamhash" in out
