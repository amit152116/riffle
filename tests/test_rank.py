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
    # tag_completeness is a generated column (Migration 7) -- `completeness`
    # here selects how many of the four tag_* columns to populate, so the
    # generated value comes out to exactly the number the test asked for.
    tag_title = "T" if completeness >= 1 else None
    tag_artist = "A" if completeness >= 2 else None
    tag_album = "B" if completeness >= 3 else None
    tag_genre = "G" if completeness >= 4 else None
    conn.execute(
        "INSERT INTO track (id, path, bitrate, tag_title, tag_artist, "
        " tag_album, tag_genre, mtime, dev, inode, audio_content_id, "
        " present) VALUES (?,?,?,?,?,?,?,?,?,?,?,1)",
        (tid, path, bitrate, tag_title, tag_artist, tag_album, tag_genre,
         mtime, dev, inode or tid, content_id))
    conn.execute("INSERT INTO group_member (group_id, track_id) "
                 "VALUES (1,?)", (tid,))


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


# --- effective bandwidth ---------------------------------------------------

def _bw(conn, tid, cutoff_hz, cliff_db=40.0):
    """Record a measurement for the content `_add(..., duration=...)` made.

    cutoff_hz=None with a cliff_db means "measured, full band"; both None
    means "unmeasurable".
    """
    conn.execute(
        "INSERT INTO audio_bandwidth (audio_content_id, cutoff_hz, cliff_db, "
        " measured_at) VALUES (?,?,?, 'now')",
        (tid + 1000, cutoff_hz, cliff_db))


def test_higher_real_bandwidth_beats_higher_bitrate(tmp_path):
    # A "320 kbps" file cut off at 16 kHz is a re-encode of a worse source;
    # a ~250 kbps file reaching 20 kHz has more of the original.
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/up.mp3", 320000, 4, 100.0, duration=200.0)
    _add(conn, 2, "/m/real.mp3", 250000, 4, 100.0, duration=200.0)
    _bw(conn, 1, 16000)
    _bw(conn, 2, 20000)
    assert rank.rank_group(conn, 1) == 2


def test_similar_bandwidth_falls_back_to_bitrate(tmp_path):
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.mp3", 320000, 4, 100.0, duration=200.0)
    _add(conn, 2, "/m/b.mp3", 250000, 4, 100.0, duration=200.0)
    _bw(conn, 1, 19500)
    _bw(conn, 2, 20000)  # within the 1.5 kHz tolerance
    assert rank.rank_group(conn, 1) == 1


def test_a_full_band_file_counts_as_the_widest(tmp_path):
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.mp3", 320000, 4, 100.0, duration=200.0)
    _add(conn, 2, "/m/b.mp3", 192000, 4, 100.0, duration=200.0)
    _bw(conn, 1, 16000)
    _bw(conn, 2, None, cliff_db=5.0)  # measured, no cliff
    assert rank.rank_group(conn, 1) == 2


def test_a_missing_measurement_disables_the_bandwidth_rule(tmp_path):
    # With one copy unmeasured there is nothing fair to compare, so the old
    # bitrate ordering decides rather than guessing.
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.mp3", 320000, 4, 100.0, duration=200.0)
    _add(conn, 2, "/m/b.mp3", 250000, 4, 100.0, duration=200.0)
    _bw(conn, 1, 16000)
    assert rank.rank_group(conn, 1) == 1


def test_an_unmeasurable_file_counts_as_missing(tmp_path):
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.mp3", 320000, 4, 100.0, duration=200.0)
    _add(conn, 2, "/m/b.mp3", 250000, 4, 100.0, duration=200.0)
    _bw(conn, 1, 16000)
    _bw(conn, 2, None, cliff_db=None)  # unmeasurable, not "full band"
    assert rank.rank_group(conn, 1) == 1


def test_bandwidth_rule_is_transitive_and_order_independent(tmp_path):
    # 16k/320, 17k/256, 18k/192: a pairwise "beats by 1.5 kHz else bitrate"
    # rule would cycle. Keeping only files within 1.5 kHz of the widest and
    # ranking those by the normal key gives one answer for any input order.
    specs = [("a", 320000, 16000), ("b", 256000, 17000), ("c", 192000, 18000)]
    winners = set()
    for order in ([0, 1, 2], [2, 1, 0], [1, 2, 0]):
        conn = _setup(tmp_path / f"o{''.join(map(str, order))}")
        for n, idx in enumerate(order, start=1):
            name, br, cut = specs[idx]
            _add(conn, n, f"/m/{name}.mp3", br, 4, 100.0, duration=200.0)
            _bw(conn, n, cut)
        winner = rank.rank_group(conn, 1)
        winners.add(conn.execute("SELECT path FROM track WHERE id = ?",
                                 (winner,)).fetchone()["path"])
    assert winners == {"/m/b.mp3"}


def test_lossless_still_wins_whatever_its_measured_cutoff(tmp_path):
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.flac", 900000, 1, 200.0, duration=200.0)
    _add(conn, 2, "/m/b.mp3", 320000, 4, 100.0, duration=200.0)
    _bw(conn, 1, 16000)
    _bw(conn, 2, 20000)
    assert rank.rank_group(conn, 1) == 1


def test_a_20_khz_file_is_not_beaten_by_a_full_band_one(tmp_path):
    # 20 vs 22 kHz is inaudible; "no cliff" is treated as reaching 21 kHz,
    # within tolerance of 20, so bitrate decides.
    conn = _setup(tmp_path)
    _add(conn, 1, "/m/a.mp3", 320000, 4, 100.0, duration=200.0)
    _add(conn, 2, "/m/b.mp3", 192000, 4, 100.0, duration=200.0)
    _bw(conn, 1, 20000)
    _bw(conn, 2, None, cliff_db=5.0)
    assert rank.rank_group(conn, 1) == 1
