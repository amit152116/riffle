# riffle

Local music library scanner and duplicate detector.

Design: `docs/superpowers/specs/2026-09-25-riffle-dedup-design.md`
Plan: `docs/superpowers/plans/2026-09-25-riffle-dedup.md`

## Requirements

- Python 3.12
- ffmpeg
- fpcalc, from the libchromaprint-tools package

## Usage

    uv run riffle scan ~/Music
    uv run riffle match
    uv run riffle report --run 1
    uv run riffle approve --run 1 --tier 1 --all
    uv run riffle apply --run 1

Nothing is ever deleted. `apply` moves losers into a quarantine directory
on the same filesystem, and `undo` puts them back.
