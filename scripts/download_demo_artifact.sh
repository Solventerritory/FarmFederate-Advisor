#!/usr/bin/env bash
set -euo pipefail

# Download and inspect GitHub Actions artifact (bash/curl version)
# Usage:
#   ./scripts/download_demo_artifact.sh [ARTIFACT_ID] [LINES]
# Example:
#   ./scripts/download_demo_artifact.sh 5211661711 20
# Notes:
# - Creates artifact.zip and extracts to artifact_zip/
# - Prompts for a short-lived GitHub PAT (Actions read / repo read).
# - Revoke the PAT immediately after use.

ARTIFACT_ID=${1:-5211661711}
LINES=${2:-20}
OUTZIP=${3:-artifact.zip}
OUTDIR=${4:-artifact_zip}

read -s -p "Enter GitHub PAT (Actions/artifact read scope): " GHTOKEN
echo
if [[ -z "$GHTOKEN" ]]; then
  echo "No token provided. Aborting." >&2
  exit 1
fi

API_URL="https://api.github.com/repos/Solventerritory/FarmFederate-Advisor/actions/artifacts/$ARTIFACT_ID/zip"

echo "Downloading artifact $ARTIFACT_ID..."
curl -fSL -H "Authorization: token $GHTOKEN" -H "User-Agent: GH-Artifact-Downloader" "$API_URL" -o "$OUTZIP"

echo "Extracting to $OUTDIR..."
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

if command -v unzip >/dev/null 2>&1; then
  unzip -q "$OUTZIP" -d "$OUTDIR"
else
  # fallback to Python extraction
  python - <<PY
import zipfile
zipfile.ZipFile("$OUTZIP").extractall("$OUTDIR")
PY
fi

# find CSV(s)
CSV=$(find "$OUTDIR" -type f \( -iname "*demo_farm*history*.csv" -o -iname "*demo_farm*history.csv" -o -iname "*.csv" \) | head -n1 || true)
if [[ -z "$CSV" ]]; then
  echo "No CSV files found in artifact. Listing extracted files:"
  find "$OUTDIR" -maxdepth 4 -type f -print
  exit 0
fi

echo "Found CSV: $CSV"
echo "--- First $LINES lines ---"
head -n "$LINES" "$CSV"
echo "--- End ---"

echo "Tip: Revoke the PAT you used immediately after this operation (GitHub -> Settings -> Developer settings -> Personal access tokens)."

# zero sensitive variables from env
GHTOKEN=''

exit 0
