#!/usr/bin/env bash
set -euo pipefail

# Local helper to verify demo CSV uploaded to Google Drive using a base64-encoded service account
# Usage:
#   export GDRIVE_SERVICE_ACCOUNT=<base64-json>
#   export GDRRIVE_FOLDER=FarmFederate-demo-artifacts   # optional
#   ./scripts/check_gdrive_upload.sh

if [[ -z "${GDRIVE_SERVICE_ACCOUNT:-}" ]]; then
  echo "Set GDRIVE_SERVICE_ACCOUNT (base64-encoded service account JSON) in environment" >&2
  exit 2
fi

echo "Decoding service account..."
echo "$GDRIVE_SERVICE_ACCOUNT" | base64 --decode > /tmp/gdrive_sa.json

if ! command -v rclone >/dev/null 2>&1; then
  echo "Installing rclone..."
  curl https://rclone.org/install.sh | bash
fi

cat > /tmp/rclone.conf <<'EOF'
[gdrive]
# Uses service account JSON to authenticate
type = drive
scope = drive.file
service_account_file = /tmp/gdrive_sa.json
EOF

FOLDER="${GDRIVE_FOLDER:-FarmFederate-demo-artifacts}"

echo "Listing files in Drive folder: $FOLDER"
rclone --config /tmp/rclone.conf lsf gdrive:"$FOLDER" -R || { echo "rclone list failed" >&2; exit 3; }

if rclone --config /tmp/rclone.conf lsf gdrive:"$FOLDER" | grep -q "demo_farm_1_history.csv"; then
  echo "OK: demo_farm_1_history.csv found in $FOLDER"
  exit 0
else
  echo "MISSING: demo_farm_1_history.csv not found in $FOLDER" >&2
  exit 4
fi