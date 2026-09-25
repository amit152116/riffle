from riffle import rank, store


def _setup(tmp_path, tier=1, formed_by_chain=0):
    conn = store.connect(tmp_path / "db.sqlite")
    conn.execute("INSERT INTO match_run (id, status) VALUES (1,'complete')")
    conn.execute("INSERT INTO dup_group (id, run_id, tier, formed_by_chain) "
                 "VALUES (1,1,?,?)", (tier, formed_by_chain))
    conn.execute("INSERT INTO audio_content (id, audio_hash, hash_method) "
                 "VALUES (1,'h','streamhash')")
    return conn


def _add(conn, tid, path, bitrate, completeness, mtime, dev=1, inode=None,
        duration=None):
    # A distinct audio_content row per call, only when duration matters to
    # the test: real tier-1 members can have different encodings/lengths,
    # each with their own audio_content.duration, unlike the single shared
    # content row every other test in this file uses.
    if duration is not None:
        content_id = tid + 1000  # _setup already created audio_content id=1
        conn.execute(
            "INSERT INTO audio_content (id, audio_hash, hash_method, "
            " duration) VALUES (?,?, 'streamhash', ?)",
            (content_id, f"h{tid}", duration))
    else:
        content_id = 1
    conn.execute(
        "INSERT INTO track (id, path, bitrate, tag_completeness, mtime, "
        " dev, inode, audio_content_id, present) VALUES (?,?,?,?,?,?,?,?,1)",
        (tid, path, bitrate, completeness, mtime, dev, inode or tid,
         content_id))
    conn.execute("INSERT INTO group_member (group_id, track_id, "
                 " audio_content_id) VALUES (1,?,?)", (tid, content_id))


def test_lossless_wins_over_higher_bitrate_lossy(tmp_path):
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.mp3", 320000, 4, 100.0)
    _add(conn, 2, "/m/a.flac", 900000, 1, 200.0)
    assert rank.rank_group(conn, 1) == 2


def test_bitrate_breaks_ties_among_lossy(tmp_path):
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.mp3", 128000, 4, 100.0)
    _add(conn, 2, "/m/b.mp3", 320000, 4, 100.0)
    assert rank.rank_group(conn, 1) == 2


def test_tag_completeness_breaks_bitrate_ties(tmp_path):
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.mp3", 320000, 1, 100.0)
    _add(conn, 2, "/m/b.mp3", 320000, 4, 100.0)
    assert rank.rank_group(conn, 1) == 2


def test_oldest_mtime_wins_last(tmp_path):
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.mp3", 320000, 4, 500.0)
    _add(conn, 2, "/m/b.mp3", 320000, 4, 100.0)
    assert rank.rank_group(conn, 1) == 2


def test_hardlinked_copy_is_never_a_loser(tmp_path):
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.flac", 900000, 4, 100.0, dev=1, inode=42)
    _add(conn, 2, "/m/b.flac", 900000, 4, 100.0, dev=1, inode=42)
    keeper = rank.rank_group(conn, 1)
    losers = [r["track_id"] for r in conn.execute(
        "SELECT track_id FROM group_member "
        "WHERE group_id = 1 AND is_keeper = 0")]
    assert keeper is not None
    assert losers == []


def test_tier_2_group_gets_no_keeper(tmp_path):
    conn = _setup(tmp_path, tier=2)
    _add(conn, 1, "/m/a.flac", 900000, 4, 100.0)
    _add(conn, 2, "/m/b.mp3", 128000, 1, 200.0)
    assert rank.rank_group(conn, 1) is None
    assert conn.execute(
        "SELECT count(*) c FROM group_member WHERE is_keeper = 1"
    ).fetchone()["c"] == 0


def test_chain_group_gets_no_keeper(tmp_path):
    conn = _setup(tmp_path, tier=1, formed_by_chain=1)
    _add(conn, 1, "/m/a.flac", 900000, 4, 100.0)
    _add(conn, 2, "/m/b.mp3", 128000, 1, 200.0)
    assert rank.rank_group(conn, 1) is None


def test_ranking_is_reproducible_under_full_ties(tmp_path):
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/zzz.mp3", 320000, 4, 100.0)
    _add(conn, 2, "/m/aaa.mp3", 320000, 4, 100.0)
    assert rank.rank_group(conn, 1) == 2  # path sort breaks the tie


def test_longest_duration_wins_before_mtime(tmp_path):
    # Review finding I11: the spec's order is lossless > bitrate > tags >
    # longest duration > oldest mtime, but duration was missing entirely,
    # so a full-length track tied on lossless/bitrate/tags could lose to a
    # trimmed copy purely because of mtime -- the opposite of what a
    # reasonable person keeping their music would want.
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/full.flac", 900000, 4, 500.0, duration=240.0)
    _add(conn, 2, "/m/trimmed.flac", 900000, 4, 100.0, duration=180.0)
    assert rank.rank_group(conn, 1) == 1


def test_null_mtime_does_not_outrank_a_real_mtime(tmp_path):
    # Related to I11: None was mapped to 0.0, which is *greater* than every
    # negative -mtime for a real, positive Unix timestamp, so a track with
    # no recorded mtime would always incorrectly win that tie-break level
    # over one with a real, older mtime.
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.mp3", 320000, 4, mtime=None)
    _add(conn, 2, "/m/b.mp3", 320000, 4, mtime=50.0)
    assert rank.rank_group(conn, 1) == 2
