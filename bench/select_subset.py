"""Pick a small, purpose-built benchmark subset from the real library.

Groups (each answers one benchmark question):
  dup      members of fingerprint-confirmed tier-1 groups: known same recording
  edit     both files of "probable" pairs (same song, intro/outro edit)
  punjabi / hindi   weakly labelled by folder, balanced across source
                    (YouTube rips vs older library files), excluding anything
                    in a duplicate group so labels and duplicates don't mix
Read-only on the library DB; writes ~/riffle-bench/subset.json.
"""
import collections, json, os, random, sqlite3, sys

from riffle import report

DB = os.path.expanduser("~/riffle-mymusic.sqlite")
OUT = os.path.expanduser("~/riffle-bench/subset.json")
RUN = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SEED = 7
ROOT = "/home/amit_152116/myDisk/Media/Amit's Music/"
rng = random.Random(SEED)

conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row


def source(path):
    return "yt" if "/youtube music/" in path else "lib"


def label(path):
    rel = path[len(ROOT):] if path.startswith(ROOT) else path
    top = rel.split("/")
    if top[0] == "youtube music":
        return {"punjabi": "punjabi", "Hindi": "hindi"}.get(top[1])
    if top[0] in ("Punjabi", "punjabi song"):
        return "punjabi"
    if top[0] in ("Heartfalling 1", "Heartfalling 2", "Retro"):
        return "hindi"
    return None


tracks = {}


def add(row, role, group=None):
    if row["id"] in tracks:
        tracks[row["id"]]["roles"].add(role)
        return
    tracks[row["id"]] = {"id": row["id"], "cid": row["audio_content_id"],
                         "path": row["path"], "source": source(row["path"]),
                         "label": label(row["path"]), "group": group,
                         "roles": {role}}


# --- dup: tier-1 groups, prefer cross-source (mp3 + YouTube) ---------------
groups = collections.defaultdict(list)
for r in conn.execute(
        "SELECT gm.group_id, t.* FROM group_member gm "
        "JOIN dup_group g ON g.id = gm.group_id "
        "JOIN track t ON t.id = gm.track_id "
        "WHERE g.run_id = ? AND g.tier = 1 AND t.present = 1", (RUN,)):
    groups[r["group_id"]].append(r)
cross = [g for g, ms in groups.items() if len({source(m["path"]) for m in ms}) > 1]
same = [g for g in groups if g not in cross]
rng.shuffle(cross); rng.shuffle(same)
chosen = cross[:30] + same[:10]
dup_track_ids = {m["id"] for ms in groups.values() for m in ms}   # ALL dup members
for g in chosen:
    for m in groups[g][:3]:
        add(m, "dup", group=g)

# --- edit: probable pairs (60-85% coverage, intro/outro edits) -------------
pairs = report.probable_pairs(conn, RUN)
rng.shuffle(pairs)
for n, p in enumerate(pairs[:20]):
    for key in ("a_content_id", "b_content_id"):
        for r in conn.execute("SELECT * FROM track WHERE audio_content_id = ? "
                              "AND present = 1 ORDER BY id LIMIT 1", (p[key],)):
            add(r, "edit", group=f"edit{n}")
            dup_track_ids.add(r["id"])

# --- labelled: balanced by label x source, excluding all dup members -------
buckets = collections.defaultdict(list)
for r in conn.execute("SELECT * FROM track WHERE present = 1"):
    lab = label(r["path"])
    if lab and r["id"] not in dup_track_ids:
        buckets[(lab, source(r["path"]))].append(r)
for key, rows in sorted(buckets.items()):
    rng.shuffle(rows)
    for r in rows[:75]:
        add(r, key[0])

out = []
for t in tracks.values():
    t = dict(t); t["roles"] = sorted(t["roles"]); out.append(t)
json.dump(out, open(OUT, "w"), indent=1)
by = collections.Counter(r for t in out for r in t["roles"])
print("tracks:", len(out), dict(by))
print("label x source pool sizes (before sampling):",
      {f"{k[0]}/{k[1]}": len(v) for k, v in sorted(buckets.items())})
print("dup groups chosen:", len(chosen), "(cross-source", len([g for g in chosen if g in cross]), ")")
print("wrote", OUT)
