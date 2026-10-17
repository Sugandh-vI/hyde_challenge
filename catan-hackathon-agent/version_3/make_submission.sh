#!/usr/bin/env bash
# Build a submission zip for a version_3 agent folder.
#
#   bash version_3/make_submission.sh version_3/v3          -> version_3/v3.zip
#   bash version_3/make_submission.sh version_3/v3_presets/p1 -> version_3/p1.zip
#
# The zip has agent.py at the TOP LEVEL and only the files that agent needs: the core modules
# (v3agent.py + v3_*.py) and one weights file.  Scripts, tools, logs and __pycache__ are excluded.
# If v3/weights_best.json exists it is shipped as weights.json (that is the file the sweep tuned);
# otherwise v3/weights.json is shipped unchanged.
set -euo pipefail

AGENT_DIR="${1:-}"
if [ -z "$AGENT_DIR" ]; then
  echo "usage: bash version_3/make_submission.sh <agent_folder>" >&2
  exit 2
fi
AGENT_DIR="$(cd "$AGENT_DIR" && pwd)"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V3="$HERE/v3"
OUT="$HERE/submission"
NAME="$(basename "$AGENT_DIR")"
ZIP="$HERE/$NAME.zip"

if [ ! -f "$AGENT_DIR/agent.py" ]; then
  echo "ERROR: $AGENT_DIR/agent.py not found" >&2
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"

echo "agent folder: $AGENT_DIR"
cp "$AGENT_DIR/agent.py" "$OUT/agent.py"
echo "  + agent.py"

# Any other .py sitting in the agent's own folder (preset wrappers have none).
for f in "$AGENT_DIR"/*.py; do
  [ -e "$f" ] || continue
  b="$(basename "$f")"
  [ "$b" = "agent.py" ] && continue
  cp "$f" "$OUT/$b"
  echo "  + $b"
done

# Core modules: needed by both v3/agent.py and the preset wrappers.
if [ ! -f "$OUT/v3agent.py" ]; then
  for f in "$V3"/v3*.py; do
    [ -e "$f" ] || continue
    cp "$f" "$OUT/"
    echo "  + $(basename "$f")"
  done
  if [ -f "$V3/weights_best.json" ]; then
    cp "$V3/weights_best.json" "$OUT/weights.json"
    echo "  + weights.json (from weights_best.json)"
  else
    cp "$V3/weights.json" "$OUT/weights.json"
    echo "  + weights.json"
  fi
else
  for f in "$AGENT_DIR"/*.json; do
    [ -e "$f" ] || continue
    b="$(basename "$f")"
    case "$b" in
      weights_smoke.json|sweep_state*.json) continue ;;
    esac
    cp "$f" "$OUT/$b"
    echo "  + $b"
  done
  [ -f "$OUT/weights.json" ] || cp "$V3/weights.json" "$OUT/weights.json"
fi

rm -rf "$OUT/__pycache__"
find "$OUT" -name "*.pyc" -delete 2>/dev/null || true

rm -f "$ZIP"
( cd "$OUT" && zip -qr "$ZIP" . -x "*.pyc" "__pycache__/*" )
echo
echo "created $ZIP"
unzip -l "$ZIP" | tail -n +2
