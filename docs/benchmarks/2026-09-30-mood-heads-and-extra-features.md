# Mood classifier heads and extra DSP features (2026-09-30)

Question: does a suggested list of extra audio features (spectral rolloff/flux/
contrast, bass/mid/treble shares, RMS spread, chroma entropy) and Essentia's
mood classifier heads add anything to what riffle already stores?

Subset: the 416-track benchmark subset (`~/riffle-bench`), read-only. There is
no mood ground truth in it (its labels are hindi/punjabi), so the only mood
ground truth is the hand-marked Y/N in `mood-heads-judging.md` and
`mood-heads-judging-middle.md` (Y = the track really has the named property).

Scripts: `bench/mood_heads.py`, `bench/extra_features.py`,
`bench/extra_eval.py`, `bench/extra_sim_eval.py`.

## 1. Classifier heads on the Discogs-EffNet embeddings

Ten heads (happy, sad, aggressive, relaxed, party, danceability,
voice_instrumental, tonal_atonal, acoustic, electronic) run on the stored
embeddings, no audio decoding. Approachability and engagement 404 at the URL
tried and were not tested.

Extremes (6 highest + 6 lowest per head, judged by ear):

| Head | High correct | Low correct |
|---|---|---|
| happy | 6/6 | 5/6 |
| sad, relaxed, party, danceability, electronic | 6/6 | 6/6 |
| aggressive | 5/6 | 6/6 |
| voice_instrumental | 6/6 | 2/6 |
| acoustic, tonal_atonal | not judged | not judged |

- `voice_instrumental` scores 0.95 on average (std 0.10): nearly every track is
  "voice". Useless on this library. `aggressive` is almost constant (p90 0.12).
- The mood heads are strongly correlated with each other (sad~party -0.92,
  party~danceability 0.92, sad~relaxed 0.90): mostly one calm-to-energetic axis.
- Middle scores (0.25-0.75, 10 random tracks each, n is tiny):
  happy 7 Y / 3 N with no trend in score; sad 9 Y / 1 ?, sad even at 0.33;
  relaxed 3 Y / 6 N, Y only at 0.55+; party 1 Y / 9 N.
  The middle of the range is not a smooth scale. Any use needs a per-head
  cutoff, and 10 tracks per head cannot set one.
- `tonal_atonal`: unverifiable. Synthetic chord / cluster / noise all score
  0.00. On 36 real tracks, noise-vocoding took the top group 0.85 -> 0.00 and
  pitch scrambling 0.85 -> 0.11, but the bottom group was already 0.00 and
  includes melodic songs (Kishore Kumar, Lata Mangeshkar). It correlates with
  dissonance (-0.69) and zcr (-0.60) more than with `key_strength` (+0.35).
  It looks like a production-style signal, not tonality. Not usable as such.

## 2. Extra DSP features (`bench/extra_features.py`, 60 s middle excerpt)

- Stability across 32 YouTube-vs-library copy pairs (median copy difference
  over median random-pair difference): all 0.04-0.25. Treble share 0.11, so no
  sign that it tracks transcode quality on these pairs.
- Redundancy, R^2 from production features: rms_mean 0.93, rolloff95 0.92,
  rolloff85 0.90, treble 0.89, flux_mean 0.86 (redundant); dyn_range 0.20,
  mid 0.24, bass 0.38, lowmid 0.39, chroma_entropy 0.42, contrast_mean 0.46.
- Against the judged labels (n = 11-22 per head): production features already
  reach CV AUC 0.98-1.00 for sad/relaxed/party/danceability/electronic, so
  extras cannot add anything measurable. Only happy has room: 0.70 -> 0.77 on
  20 tracks, within noise. The labelled tracks were picked from the heads'
  score extremes, which makes them easy to separate.

## 3. Extras in the similarity benchmark (`bench/extra_sim_eval.py`)

Six low-redundancy extras (dyn_range, bass, lowmid, mid, chroma_entropy,
contrast_mean) added as one percentile-ranked component, with a shuffled-column
control at each weight.

| Representation | dup hit@1 | label@10 |
|---|---|---|
| effnet | 0.99 | 0.807 |
| effnet + extras (w=0.1) | 0.94 | 0.795 |
| hybrid (EffNet + BPM/key/energy) | 0.90 | 0.762 |
| hybrid + extras (w=0.1) | 0.90 | 0.767 |
| hybrid + shuffled (w=0.1) | 0.84 | 0.763 |
| MFCC-based production score | 0.80 | 0.678 |
| production + extras (w=0.2) | 0.80 | 0.685 |
| extras alone | 0.71 | 0.626 |

The extras carry real information (alone: dup 0.71 vs 0.01 shuffled) but make
the best representation worse and only nudge the weaker ones.

## Decision

No new feature is added to the schema. Nothing here beat the existing
features or the EffNet embedding on a metric we can measure.

The heads remain a possible source of high/low flags (sad, relaxed, party,
danceability, electronic, happy at the extremes) but need a larger labelled set
to fit per-head cutoffs. `tonal_atonal`, `voice_instrumental` and `aggressive`
are not useful on this library.

Limits: 416 tracks, judged labels n = 11-22 per head, 32 duplicate pairs, the
mood labels come from one listener, and the middle-score sheet covers only four
heads.
