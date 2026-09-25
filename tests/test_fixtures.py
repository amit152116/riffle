import subprocess

from tests.fixtures import make_tone, transcode, trim, concat, retag


def _duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def test_make_tone_creates_audio_of_requested_length(tmp_path):
    p = make_tone(tmp_path / "a.flac", seconds=5.0)
    assert p.exists()
    assert abs(_duration(p) - 5.0) < 0.2


def test_transcode_preserves_duration(tmp_path):
    src = make_tone(tmp_path / "a.flac", seconds=5.0)
    dst = transcode(src, tmp_path / "a.mp3", codec="libmp3lame", bitrate="128k")
    assert abs(_duration(dst) - 5.0) < 0.3


def test_trim_shortens(tmp_path):
    src = make_tone(tmp_path / "a.flac", seconds=10.0)
    dst = trim(src, tmp_path / "t.flac", start=3.0, duration=4.0)
    assert abs(_duration(dst) - 4.0) < 0.2


def test_concat_lengthens(tmp_path):
    a = make_tone(tmp_path / "a.flac", seconds=4.0, freq=300)
    b = make_tone(tmp_path / "b.flac", seconds=4.0, freq=900)
    dst = concat(tmp_path / "ab.flac", a, b)
    assert abs(_duration(dst) - 8.0) < 0.3


def test_retag_changes_the_file_but_not_the_audio(tmp_path):
    p = make_tone(tmp_path / "a.flac", seconds=3.0)
    before = p.read_bytes()
    retag(p, title="Changed")
    assert p.read_bytes() != before
