#!/usr/bin/env bash
set -euo pipefail

# Copies my_team/agent.py plus any helper .py and data files (weights.json etc.)
# into a clean folder submission/ and zips it as my_team.zip with agent.py at TOP LEVEL of zip.
# Exclude results.csv, scripts, __pycache__, versions.
# Run from catan-hackathon-agent/ : ./my_team/make_submission.sh
# Or from any dir: bash my_team/make_submission.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MY_TEAM_DIR="$SCRIPT_DIR"
SUBMISSION_DIR="$ROOT_DIR/submission"
ZIP_PATH="$ROOT_DIR/my_team.zip"

echo "Script dir: $SCRIPT_DIR"
echo "Root dir: $ROOT_DIR"

# Clean previous
rm -rf "$SUBMISSION_DIR"
mkdir -p "$SUBMISSION_DIR"

# Copy agent.py (must exist)
if [ ! -f "$MY_TEAM_DIR/agent.py" ]; then
  echo "ERROR: $MY_TEAM_DIR/agent.py not found" >&2
  exit 1
fi

echo "Copying agent files..."

# Copy agent.py
cp "$MY_TEAM_DIR/agent.py" "$SUBMISSION_DIR/"

# Copy helper .py files (search.py, etc.) but exclude run_eval.py, tune_weights.py and other scripts
# We include any .py that agent.py might import, but exclude the evaluation/tuning scripts to keep zip clean.
# Policy: copy all .py in my_team except run_eval.py, tune_weights.py, agent_v1.py, agent_v2.py
# However spec says copies my_team/agent.py plus any helper .py and data files (weights.json etc.)
# into clean folder submission/. Exclude results.csv, scripts, __pycache__, versions.
# To be safe, we copy all .py except known scripts and results; but we include search.py if present.

for f in "$MY_TEAM_DIR"/*.py; do
  [ -e "$f" ] || continue
  base=$(basename "$f")
  case "$base" in
    run_eval.py|tune_weights.py)
      echo "Skipping $base (eval/tune script)"
      ;;
    agent_v1.py|agent_v2.py)
      echo "Skipping $base (version history)"
      ;;
    *)
      echo "Copying $base"
      cp "$f" "$SUBMISSION_DIR/"
      ;;
  esac
done

# Copy data files: weights.json, weights_tuned.json, *.json, *.pt, *.pth, etc. if present
for ext in json pt pth weights txt dat csv; do
  for f in "$MY_TEAM_DIR"/*."$ext"; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    # Exclude results.csv and tuning_log.csv per spec
    if [ "$base" = "results.csv" ] || [ "$base" = "tuning_log.csv" ] || [ "$base" = "EXPERIMENTS.md" ]; then
      echo "Skipping $base (log/results)"
      continue
    fi
    echo "Copying data $base"
    cp "$f" "$SUBMISSION_DIR/"
  done
done

# Also check for weights.json explicitly even if extension handling missed it (already handled)
# Ensure weights.json is copied if exists (should be)
if [ -f "$MY_TEAM_DIR/weights.json" ] && [ ! -f "$SUBMISSION_DIR/weights.json" ]; then
  cp "$MY_TEAM_DIR/weights.json" "$SUBMISSION_DIR/"
fi

# Remove __pycache__ if any got copied (shouldn't but guard)
rm -rf "$SUBMISSION_DIR/__pycache__"
find "$SUBMISSION_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$SUBMISSION_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true

# List what will be zipped
echo "Submission contents:"
ls -lh "$SUBMISSION_DIR"

# Create zip with agent.py at top level
rm -f "$ZIP_PATH"
# Zip from inside submission dir so files are at top level
(
  cd "$SUBMISSION_DIR"
  # Use -r to recurse, but we only have top-level files; still safe
  zip -r "$ZIP_PATH" . -x "*.pyc" -x "__pycache__/*"
)

echo "Created $ZIP_PATH"
unzip -l "$ZIP_PATH" | head -n 50
echo "Done. Upload with ~/upload.sh my_team.zip on team machine, or evaluate locally."

