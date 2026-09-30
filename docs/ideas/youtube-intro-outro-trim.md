# Idea: trim YouTube intros/outros

Status: parked (2026-09-29). Not urgent. Prototype scripts saved in
`docs/ideas/trim-prototype/`.

## Problem

YouTube rips in `Amit's Music/youtube music/` often carry extra audio the
studio version lacks: spoken intros, channel tags, film dialogue, silent or
"thanks for watching" tails. `rank.py` prefers the *longer* duration, so after
dedup the YouTube version (with the intro) can beat the clean studio copy.

## Approach A: align against a clean twin (prototyped, works)

When a dup group holds a YouTube file and a non-YouTube file:

1. `match.compare(fp_yt, fp_ref, cfg, item_s)` gives `best_offset` (items).
   Both fingerprints share Chromaprint's fixed delay, so it cancels:
   `intro_secs = offset * item_s` (sign flipped if YT is the `b` side).
2. `outro_secs = yt_duration - intro - ref_duration`.
3. Cut with `ffmpeg -ss <keep_from> -t <ref_duration> -c copy` (no re-encode).
4. Refine the start with a 10 ms loudness-envelope cross-correlation against
   the reference's first ~20 s (`refine.py`). Shift was 0.01-0.06 s on the
   3 files tried, so fingerprint timing alone is already close.

Prototype result on run 3: 89 pairs -> 40 `trim`, 32 `no-trim`,
16 `yt-shorter` (YT is missing audio; do not trim), 1 `ref-partial`.
Previews of g100, g35, g84 sounded right. g46 has a ~0.2 s quiet pre-beat that
the reference lacks (real YouTube audio, not misalignment); fix candidate:
snap the start to the first strong onset (>15 dB jump within 0.5 s).

Gotchas:
- Quarantined files are `present = 0`, so `matchrun.load_fingerprints`
  skips them. Read `fingerprint` directly (canonical purpose).
- The reference may have its own intro (e.g. spoken `DJJOhAL.Com` tags).
- `yt-shorter` and `ref-partial` rows must never be trimmed automatically.
- Write trimmed files to a new folder; keep originals; rerun `match` after.

## Approach B: files with no reference (not built)

~1000 YouTube files have no clean twin. No ground truth, so layer signals and
only auto-trim when several agree; otherwise emit a review list.

1. **Expected length**: MusicBrainz/AcoustID recording length via
   `riffle enrich`. If the file is >= 15 s longer, a trim is needed and the
   implied cut point bounds where to look. First thing to build. Check how
   many tracks `enrich` has already resolved.
2. **Speech vs music** segmenter (`inaSpeechSegmenter`, or an Essentia model)
   for spoken intros/outros. Misses instrumental intros and can mistake a
   song's own spoken opening for an intro.
3. **Silence/energy boundaries** (`ffmpeg silencedetect`): gaps >= ~1 s near the
   start/end, followed by a sharp loudness change.
4. **Self-repetition** (weak): non-song material at the start does not recur
   later. Tiebreaker only.

Proposed rule: auto-trim only when expected length says trim is needed AND a
speech/silence boundary lands within a few seconds of the implied cut point.

## Possible ranking change

Longest-duration-wins in `rank_key` can pick the intro version. Options:
prefer the file closest to the group's median duration, or de-prioritise
files under `youtube music/`. Not changed.

## Next steps when picked up

1. Fold `refine.py` + onset snapping into a `riffle trim-report` command.
2. Generate all `trim` rows into a separate folder; spot-check by ear.
3. Count MusicBrainz lengths available; build Approach B step 1.
4. Rerun `match` so trimmed copies group with their references.
