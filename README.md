# audiolib

Local music library scanner and duplicate detector.

Design: `docs/superpowers/specs/2026-09-25-audiolib-dedup-design.md`
Plan: `docs/superpowers/plans/2026-09-25-audiolib-dedup.md`

## Requirements

- Python 3.12
- ffmpeg
- fpcalc, from the libchromaprint-tools package

## Usage

    uv run audiolib scan ~/Music
    uv run audiolib match
    uv run audiolib report --run 1
    uv run audiolib approve --run 1 --tier 1 --all
    uv run audiolib apply --run 1

Nothing is ever deleted. `apply` moves losers into a quarantine directory
on the same filesystem, and `undo` puts them back.
