#!/usr/bin/env bash
# Execute as a script, never source into an interactive login shell.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python -u -B scripts/consolidate_p5.py --prepare \
  plan/p5/storage/primary-20261123T094822803672Z/stage-20261126T121550709575Z.json
python -u -B scripts/consolidate_p5.py --verify
python -u -B scripts/consolidate_p5.py --finalize
du -B1 -s plan/p5/runs
df -h .
