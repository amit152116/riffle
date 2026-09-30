"""Read-only: propose intro/outro trims for YouTube files using a clean twin.

For each dup group, pair every file under 'youtube music/' with every other
file (the reference). Alignment offset (items) between the two fingerprints
gives how much extra audio the YouTube file has at the start; the fixed
Chromaprint delay is identical in both files and cancels.
"""
import csv, itertools, sqlite3, sys
from riffle import fingerprint, match, store

DB = "/home/amit_152116/riffle-mymusic.sqlite"
OUT = "/home/amit_152116/riffle-trim-report.csv"
RUN = int(sys.argv[1]) if len(sys.argv) > 1 else 3
YT = "/youtube music/"

conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
# Quarantined files are no longer 'present', so matchrun.load_fingerprints
# skips them; read every canonical fingerprint directly.
fps = {r["cid"]: store.unpack_fingerprint(r["fp_raw"], r["fp_length"])
       for r in conn.execute(
           "SELECT audio_content_id cid, fp_raw, fp_length FROM fingerprint "
           "WHERE purpose = 'canonical'")}
item_s = fingerprint.item_duration_seconds()
cfg = match.DEFAULT_MATCH_CONFIG

rows = conn.execute("""
    SELECT gm.group_id, gm.is_keeper, t.id track_id, t.audio_content_id cid,
           ac.duration dur,
           COALESCE((SELECT q.dst_path FROM quarantine_log q
                     WHERE q.track_id = t.id AND q.state = 'moved'
                     ORDER BY q.id DESC LIMIT 1), t.path) path,
           EXISTS(SELECT 1 FROM quarantine_log q
                  WHERE q.track_id = t.id AND q.state = 'moved') quarantined
    FROM group_member gm
    JOIN dup_group g ON g.id = gm.group_id
    JOIN track t ON t.id = gm.track_id
    JOIN audio_content ac ON ac.id = t.audio_content_id
    WHERE g.run_id = ? AND g.tier = 1
    ORDER BY gm.group_id""", (RUN,)).fetchall()

groups = {}
for r in rows:
    groups.setdefault(r["group_id"], []).append(r)

out = []
for gid, members in groups.items():
    yts = [m for m in members if YT in m["path"] or YT in m["path"].replace("\\", "/")]
    refs = [m for m in members if m not in yts]
    for y, ref in itertools.product(yts, refs):
        if y["cid"] == ref["cid"] or y["cid"] not in fps or ref["cid"] not in fps:
            continue
        a, b = sorted((y["cid"], ref["cid"]))
        ev = match.compare(fps[a], fps[b], cfg, item_s)
        lead_items = ev.best_offset if y["cid"] == a else -ev.best_offset
        intro = lead_items * item_s
        ref_cov = ev.coverage_a if ref["cid"] == a else ev.coverage_b
        yt_dur, ref_dur = y["dur"] or 0.0, ref["dur"] or 0.0
        outro = yt_dur - intro - ref_dur
        if ev.tier != 1:
            flag = "weak-match"
        elif ref_cov < 0.9:
            flag = "ref-partial"        # reference itself not a full song
        elif intro < -1.5 or outro < -1.5:
            flag = "yt-shorter"         # YT missing audio the reference has
        elif intro < 1.5 and outro < 1.5:
            flag = "no-trim"
        elif intro > 60 or outro > 60:
            flag = "check-large"
        else:
            flag = "trim"
        out.append({
            "group": gid, "flag": flag,
            "yt_file": y["path"], "yt_in_quarantine": y["quarantined"],
            "reference": ref["path"],
            "yt_secs": round(yt_dur, 1), "ref_secs": round(ref_dur, 1),
            "trim_start": round(max(intro, 0), 1),
            "trim_end": round(max(outro, 0), 1),
            "keep_from": round(max(intro, 0), 1),
            "keep_to": round(max(intro, 0) + ref_dur, 1),
            "bit_error": round(ev.mean_bit_error, 2),
            "ref_cov": round(ref_cov, 2),
        })

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])) if out else None
    if w:
        w.writeheader(); w.writerows(out)

from collections import Counter
print("pairs:", len(out), dict(Counter(o["flag"] for o in out)))
for o in sorted((o for o in out if o["flag"] == "trim"),
                key=lambda o: -(o["trim_start"] + o["trim_end"]))[:12]:
    print(f'g{o["group"]:<4} start {o["trim_start"]:>6}s end {o["trim_end"]:>6}s  '
          f'{o["yt_file"].split("/")[-1][:60]}')
print("wrote", OUT)
