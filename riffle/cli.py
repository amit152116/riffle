"""Command-line interface."""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import typer

from riffle import (approve as approve_mod, fingerprint, group, matchrun,
                      quarantine, rank, report as report_mod, scan as scan_mod,
                      store)

app = typer.Typer(add_completion=False, help="Music library deduplicator.")
_state: dict = {}


@app.callback()
def main(db: str = typer.Option("riffle.sqlite", help="Database path")):
    _state["db"] = Path(db)


@contextlib.contextmanager
def _session():
    """Acquire the exclusive lock and connection for one command, and always
    release both -- a single OS process naturally releases the lock on exit,
    but a persistent process (the test suite's CliRunner, or any in-process
    caller) does not, and would otherwise wrongly refuse every later call in
    the same process as "another riffle process holds" its own prior lock.
    """
    db = _state["db"]
    try:
        with store.exclusive_lock(db):
            conn = store.connect(db)
            try:
                yield conn
            finally:
                conn.close()
    except store.LockError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)


@app.command()
def scan(dirs: list[str], verify_hashes: bool = False,
         retry_errors: bool = False):
    with _session() as conn:
        scan_mod.scan(conn, [Path(d) for d in dirs],
                      verify_hashes=verify_hashes, retry_errors=retry_errors)
        written = fingerprint.fingerprint_pending(conn)
    typer.echo(f"scanned; {written} new fingerprints")


@app.command()
def match():
    with _session() as conn:
        run_id = matchrun.run_match(conn)
        n = group.build_groups(conn, run_id)
        for row in conn.execute("SELECT id FROM dup_group WHERE run_id = ?",
                                (run_id,)):
            rank.rank_group(conn, row["id"])
    typer.echo(f"run {run_id}: {n} groups")


@app.command()
def report(run: int = typer.Option(..., "--run"),
           tier: int | None = None, as_json: bool = False):
    with _session() as conn:
        data = report_mod.report_data(conn, run, tier)
    typer.echo(json.dumps(data, indent=2) if as_json
               else report_mod.render_text(data))


@app.command()
def approve(run: int = typer.Option(..., "--run"),
            group_id: int | None = typer.Option(None, "--group"),
            tier: int | None = None, all: bool = False, yes: bool = False):
    with _session() as conn:
        if group_id is not None:
            approve_mod.approve_group(conn, run, group_id)
            typer.echo(f"group {group_id} approved")
            return
        if not (all and tier is not None):
            typer.echo("error: pass --group, or --tier N --all", err=True)
            raise typer.Exit(code=2)

        ids = approve_mod.approve_tier(conn, run, tier)
        typer.echo(f"{len(ids)} group(s) would be approved at tier {tier}")
        if not ids:
            return
        if not yes and not typer.confirm("Approve them all?"):
            typer.echo("nothing approved")
            return
        typer.echo(f"{approve_mod.commit_tier(conn, run, tier)} approved")


@app.command()
def reject(run: int = typer.Option(..., "--run"),
           group_id: int = typer.Option(..., "--group")):
    with _session() as conn:
        approve_mod.reject_group(conn, run, group_id)
    typer.echo(f"group {group_id} rejected")


@app.command()
def apply(run: int = typer.Option(..., "--run")):
    with _session() as conn:
        # Quarantine directories belong at the scan roots, one per
        # filesystem, not beside every file. Recover them from the last
        # completed scan.
        row = conn.execute(
            "SELECT roots FROM scan_run WHERE status = 'complete' "
            "ORDER BY id DESC LIMIT 1").fetchone()
        roots = [Path(r) for r in json.loads(row["roots"])] if row else []
        result = quarantine.apply_run(conn, run, roots)
    typer.echo(f"moved {result['moved']}, failed {result['failed']}, "
               f"skipped groups {result['skipped_groups']}, "
               f"modified since run {result['modified']}")


@app.command()
def undo(run: int = typer.Option(..., "--run")):
    with _session() as conn:
        result = quarantine.undo_run(conn, run)
    typer.echo(f"restored {result['restored']}, refused {result['refused']}")


@app.command()
def enrich(api_key: str = typer.Option(..., envvar="ACOUSTID_API_KEY")):
    from riffle import enrich as enrich_mod

    with _session() as conn:
        result = enrich_mod.enrich(conn, api_key)
    typer.echo(f"looked up {result['looked_up']}, cached {result['cached']}, "
               f"failed {result['failed']}")


@app.command()
def health(as_json: bool = False):
    from riffle import health as health_mod

    with _session() as conn:
        data = health_mod.health_report(conn)
    typer.echo(json.dumps(data, indent=2) if as_json
               else health_mod.render_health(data))


@app.command()
def quality(as_json: bool = False,
            limit: int | None = typer.Option(None, help="Max tracks to analyze")):
    from riffle import quality as quality_mod

    with _session() as conn:
        result = quality_mod.quality_scan(conn, limit=limit)
        typer.echo(f"analyzed {result['analyzed']}, cached {result['cached']}, "
                   f"failed {result['failed']}, issues {result['issues']}")
        if not as_json:
            typer.echo("")
            typer.echo(quality_mod.render_quality(conn))


@app.command()
def features(as_json: bool = False,
             limit: int | None = typer.Option(None, help="Max tracks to analyze")):
    from riffle import features as features_mod

    with _session() as conn:
        try:
            result = features_mod.feature_scan(conn, limit=limit)
        except ImportError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1)
        if as_json:
            typer.echo(json.dumps(result))
        else:
            typer.echo(f"analyzed {result['analyzed']}, cached {result['cached']}, "
                       f"failed {result['failed']}")
            typer.echo("")
            typer.echo(features_mod.render_features(conn))


@app.command()
def metadata(as_json: bool = False):
    from riffle import metadata as metadata_mod

    with _session() as conn:
        result = metadata_mod.parse_all(conn)
        if as_json:
            typer.echo(json.dumps(result))
        else:
            typer.echo(f"parsed {result['parsed']}, cached {result['cached']}, "
                       f"no_match {result['no_match']}, failed {result['failed']}")
            typer.echo("")
            typer.echo(metadata_mod.render_metadata(conn))


@app.command()
def browse(sort_by: str = "artist", genre: str | None = None,
           bpm_min: float | None = None, bpm_max: float | None = None,
           key: str | None = None, cluster: int | None = None,
           limit: int = 50, as_json: bool = False):
    from riffle import collection

    bpm_range = (bpm_min, bpm_max) if bpm_min is not None and bpm_max is not None else None
    with _session() as conn:
        rows = collection.browse(conn, sort_by=sort_by, genre=genre,
                                 bpm_range=bpm_range, key=key,
                                 cluster=cluster, limit=limit)
        if as_json:
            typer.echo(json.dumps(rows, indent=2))
        else:
            typer.echo(collection.render_browse(rows))


@app.command()
def stats(as_json: bool = False):
    from riffle import collection

    with _session() as conn:
        data = collection.stats(conn)
        if as_json:
            typer.echo(json.dumps(data, indent=2))
        else:
            typer.echo(collection.render_stats(data))


@app.command()
def cluster(n: int | None = typer.Option(None, help="Number of clusters"),
            rebuild: bool = False, as_json: bool = False):
    from riffle import cluster as cluster_mod

    with _session() as conn:
        try:
            if rebuild:
                result = cluster_mod.rebuild_clusters(conn, n_clusters=n)
            else:
                result = cluster_mod.cluster_tracks(conn, n_clusters=n)
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1)
        if as_json:
            typer.echo(json.dumps(result))
        else:
            typer.echo(f"{result['n_clusters']} clusters, sizes: {result['sizes']}")
            typer.echo("")
            typer.echo(cluster_mod.render_clusters(conn))


@app.command(name="build-index")
def build_index(top_k: int = 20, rebuild: bool = False, as_json: bool = False):
    from riffle import similarity

    with _session() as conn:
        if rebuild:
            result = similarity.full_rebuild(conn, top_k)
        else:
            result = similarity.build_similarity(conn, top_k)
        if as_json:
            typer.echo(json.dumps(result))
            return
        typer.echo(f"new {result['new_tracks']}, pairs {result['pairs_stored']}, "
                   f"updated {result['existing_updated']}")


@app.command()
def similar(track: str, n: int = 10, as_json: bool = False):
    from riffle import shuffle as shuffle_mod

    with _session() as conn:
        try:
            track_id = shuffle_mod.resolve_track(conn, track)
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1)
        rows = shuffle_mod.find_similar(conn, track_id, n=n)
        if as_json:
            typer.echo(json.dumps(rows, indent=2))
        else:
            typer.echo(shuffle_mod.render_similar(rows))


@app.command()
def shuffle(seed: str | None = None, n: int = 20,
            genre: str | None = None,
            bpm_min: float | None = None, bpm_max: float | None = None,
            as_json: bool = False):
    from riffle import shuffle as shuffle_mod

    bpm_range = (bpm_min, bpm_max) if bpm_min is not None and bpm_max is not None else None
    with _session() as conn:
        try:
            seed_id = shuffle_mod.resolve_track(conn, seed) if seed else None
            playlist = shuffle_mod.smart_shuffle(
                conn, seed_track_id=seed_id, n=n, genre=genre, bpm_range=bpm_range)
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1)
        if as_json:
            typer.echo(json.dumps(playlist, indent=2))
        else:
            typer.echo(shuffle_mod.render_playlist(playlist))


@app.command()
def calibrate(pairs_file: str):
    from riffle import calibrate as cal_mod
    from riffle import match as match_mod

    entries = cal_mod.load_pairs(Path(pairs_file))
    best, scores = cal_mod.search(entries, match_mod.DEFAULT_MATCH_CONFIG)
    typer.echo(json.dumps({"config": best, "scores": scores}, indent=2,
                          default=str))
