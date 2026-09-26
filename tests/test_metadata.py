"""Tests for AcoustID response parsing and metadata extraction."""
import json

from riffle import store


def test_migration_4_creates_musicbrainz_match(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    assert "musicbrainz_match" in tables
    cols = [r[1] for r in conn.execute("PRAGMA table_info(musicbrainz_match)")]
    assert "artists_json" in cols
    assert "recording_mbid" in cols


def test_migration_4_adds_audio_content_id_to_cache(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(acoustid_cache)")]
    assert "audio_content_id" in cols


SAMPLE_RESPONSE = {
    "status": "ok",
    "results": [{
        "id": "abc-123",
        "score": 0.95,
        "recordings": [{
            "id": "rec-001",
            "title": "Tere Bina",
            "artists": [
                {"id": "art-001", "name": "A.R. Rahman"},
                {"id": "art-002", "name": "Chinmayi"},
            ],
            "releasegroups": [{
                "id": "rg-001",
                "title": "Guru",
                "type": "Album",
            }],
        }],
    }],
}


def test_parse_acoustid_response_valid():
    from riffle import metadata
    result = metadata.parse_acoustid_response(json.dumps(SAMPLE_RESPONSE))
    assert result is not None
    assert result["acoustid_id"] == "abc-123"
    assert result["recording_mbid"] == "rec-001"
    assert result["recording_title"] == "Tere Bina"
    artists = json.loads(result["artists_json"])
    assert len(artists) == 2
    assert artists[0]["name"] == "A.R. Rahman"
    assert artists[0]["mbid"] == "art-001"
    assert result["release_title"] == "Guru"
    assert result["score"] == 0.95


def test_parse_acoustid_response_no_recordings():
    from riffle import metadata
    response = {"status": "ok", "results": [{"id": "abc", "score": 0.5}]}
    result = metadata.parse_acoustid_response(json.dumps(response))
    assert result is None


def test_parse_acoustid_response_empty_results():
    from riffle import metadata
    response = {"status": "ok", "results": []}
    result = metadata.parse_acoustid_response(json.dumps(response))
    assert result is None


def test_parse_acoustid_response_multiple_results_picks_highest():
    from riffle import metadata
    response = {
        "status": "ok",
        "results": [
            {"id": "low", "score": 0.3, "recordings": [{"id": "r1", "title": "Low",
             "artists": [{"id": "a1", "name": "X"}], "releasegroups": []}]},
            {"id": "high", "score": 0.9, "recordings": [{"id": "r2", "title": "High",
             "artists": [{"id": "a2", "name": "Y"}], "releasegroups": []}]},
        ],
    }
    result = metadata.parse_acoustid_response(json.dumps(response))
    assert result["acoustid_id"] == "high"
    assert result["recording_title"] == "High"


def test_parse_acoustid_response_single_artist():
    from riffle import metadata
    response = {
        "status": "ok",
        "results": [{"id": "x", "score": 0.8, "recordings": [{
            "id": "r1", "title": "Song",
            "artists": [{"id": "a1", "name": "Solo Artist"}],
            "releasegroups": [],
        }]}],
    }
    result = metadata.parse_acoustid_response(json.dumps(response))
    artists = json.loads(result["artists_json"])
    assert len(artists) == 1
    assert artists[0]["name"] == "Solo Artist"


def _insert_cache_row(conn, cid, response_dict, lookup_key=None):
    """Insert a track, audio_content, and a cached AcoustID response."""
    conn.execute(
        "INSERT OR IGNORE INTO audio_content (id, audio_hash, hash_method) "
        "VALUES (?, ?, 'streamhash')", (cid, f"h{cid}"))
    conn.execute(
        "INSERT OR IGNORE INTO track (id, path, size, mtime, audio_content_id, present) "
        "VALUES (?, ?, 100, 1.0, ?, 1)", (cid, f"/music/track{cid}.mp3", cid))
    key = lookup_key or f"key{cid}"
    conn.execute(
        "INSERT INTO acoustid_cache (lookup_key, response_json, fetched_at, audio_content_id) "
        "VALUES (?, ?, '2026-01-01T00:00:00', ?)",
        (key, json.dumps(response_dict), cid))


def test_parse_all_stores_results(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _insert_cache_row(conn, 1, SAMPLE_RESPONSE)
    from riffle import metadata
    result = metadata.parse_all(conn)
    assert result["parsed"] == 1
    row = conn.execute(
        "SELECT * FROM musicbrainz_match WHERE audio_content_id = 1"
    ).fetchone()
    assert row is not None
    assert row["recording_title"] == "Tere Bina"


def test_parse_all_idempotent(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _insert_cache_row(conn, 1, SAMPLE_RESPONSE)
    from riffle import metadata
    metadata.parse_all(conn)
    result = metadata.parse_all(conn)
    assert result["parsed"] == 0
    assert result["cached"] == 1


def test_parse_all_no_match(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    empty_response = {"status": "ok", "results": []}
    _insert_cache_row(conn, 1, empty_response)
    from riffle import metadata
    result = metadata.parse_all(conn)
    assert result["parsed"] == 0
    assert result["no_match"] == 1


def test_parse_all_does_not_reparse_no_match_rows(tmp_path):
    """M7: a no_match outcome is a stable fact (AcoustID's own response says
    no recording matched) and must be remembered, not re-parsed forever."""
    conn = store.connect(tmp_path / "db.sqlite")
    empty_response = {"status": "ok", "results": []}
    _insert_cache_row(conn, 1, empty_response)
    from riffle import metadata
    first = metadata.parse_all(conn)
    assert first["no_match"] == 1

    second = metadata.parse_all(conn)
    assert second["no_match"] == 0
    assert second["parsed"] == 0


def test_render_metadata_matched_count_excludes_no_match_sentinels(tmp_path):
    """M7 consumer fix: once no_match rows are cached as sentinel rows in
    musicbrainz_match, 'Matched:' must still count only real matches, not
    every attempted row."""
    conn = store.connect(tmp_path / "db.sqlite")
    _insert_cache_row(conn, 1, SAMPLE_RESPONSE)
    _insert_cache_row(conn, 2, {"status": "ok", "results": []})
    from riffle import metadata
    metadata.parse_all(conn)
    text = metadata.render_metadata(conn)
    assert "Matched:             1" in text


def test_render_metadata_top_artists_skips_no_match_sentinels(tmp_path):
    """M7 consumer fix: a sentinel row has artists_json=NULL. The top-artists
    loop must not crash trying to json.loads(None)."""
    conn = store.connect(tmp_path / "db.sqlite")
    _insert_cache_row(conn, 1, SAMPLE_RESPONSE)
    _insert_cache_row(conn, 2, {"status": "ok", "results": []})
    from riffle import metadata
    metadata.parse_all(conn)
    text = metadata.render_metadata(conn)  # must not raise
    assert "A.R. Rahman" in text


def test_parse_all_empty_cache(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    from riffle import metadata
    result = metadata.parse_all(conn)
    assert result["parsed"] == 0
    assert result["cached"] == 0
    assert result["no_match"] == 0


def test_render_metadata(tmp_path):
    conn = store.connect(tmp_path / "db.sqlite")
    _insert_cache_row(conn, 1, SAMPLE_RESPONSE)
    from riffle import metadata
    metadata.parse_all(conn)
    text = metadata.render_metadata(conn)
    assert "Matched" in text or "matched" in text


def test_cli_metadata_as_json_emits_valid_json(tmp_path):
    """M6: --as-json must emit machine-readable JSON, not just suppress the
    human-readable report."""
    from typer.testing import CliRunner
    from riffle.cli import app

    db = str(tmp_path / "db.sqlite")
    conn = store.connect(tmp_path / "db.sqlite")
    _insert_cache_row(conn, 1, SAMPLE_RESPONSE)
    conn.close()

    runner = CliRunner()
    result = runner.invoke(app, ["--db", db, "metadata", "--as-json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["parsed"] == 1
