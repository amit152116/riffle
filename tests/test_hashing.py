import pytest

from riffle import hashing
from tests.fixtures import make_tone, retag, transcode


def test_streamhash_is_stable_across_retagging(tmp_path):
    p = make_tone(tmp_path / "a.flac", seconds=3.0)
    before = hashing.audio_identity(p)
    retag(p, title="Something Else", artist="Someone")
    after = hashing.audio_identity(p)
    assert before.hash_method == "streamhash"
    assert after.audio_hash == before.audio_hash


def test_whole_file_hash_changes_on_retagging(tmp_path):
    p = make_tone(tmp_path / "a.flac", seconds=3.0)
    before = hashing.whole_file_sha256(p)
    retag(p, title="Something Else")
    assert hashing.whole_file_sha256(p) != before


def test_identical_audio_in_different_files_hashes_the_same(tmp_path):
    a = make_tone(tmp_path / "a.flac", seconds=3.0)
    b = tmp_path / "b.flac"
    b.write_bytes(a.read_bytes())
    retag(b, title="Different Tags")
    assert hashing.audio_identity(a).audio_hash == \
           hashing.audio_identity(b).audio_hash


def test_different_audio_hashes_differently(tmp_path):
    a = make_tone(tmp_path / "a.flac", seconds=3.0, freq=440)
    b = make_tone(tmp_path / "b.flac", seconds=3.0, freq=1200)
    assert hashing.audio_identity(a).audio_hash != \
           hashing.audio_identity(b).audio_hash


def test_transcoding_changes_the_stream_hash(tmp_path):
    a = make_tone(tmp_path / "a.flac", seconds=3.0)
    b = transcode(a, tmp_path / "a.mp3", codec="libmp3lame", bitrate="128k")
    assert hashing.audio_identity(a).audio_hash != \
           hashing.audio_identity(b).audio_hash


def test_path_with_newline_and_quotes(tmp_path):
    # Review Focus 2: these must never be pasted into a shell string.
    weird = tmp_path / "we'ird\nname \"x\".flac"
    make_tone(weird, seconds=2.0)
    result = hashing.audio_identity(weird)
    assert result.hash_method == "streamhash"
    assert len(result.audio_hash) == 64


def test_non_audio_file_raises(tmp_path):
    p = tmp_path / "notaudio.flac"
    p.write_bytes(b"this is not audio")
    with pytest.raises(hashing.HashError):
        hashing.audio_identity(p)
