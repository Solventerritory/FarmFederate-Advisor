#!/usr/bin/env bash
set -euo pipefail

# Local helper to verify demo CSV uploaded to Azure Blob
# Usage:
#   export AZURE_STORAGE_CONNECTION_STRING="..."
#   export AZURE_CONTAINER=mycontainer
#   ./scripts/check_azure_upload.sh

if [[ -z "${AZURE_STORAGE_CONNECTION_STRING:-}" || -z "${AZURE_CONTAINER:-}" ]]; then
  echo "Set AZURE_STORAGE_CONNECTION_STRING and AZURE_CONTAINER environment vars" >&2
  exit 2
fi

if ! command -v az >/dev/null 2>&1; then
  echo "Installing Azure CLI..."
  curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
fi

echo "Checking Azure container: $AZURE_CONTAINER for demo_farm_1_history.csv"
if az storage blob exists --connection-string "$AZURE_STORAGE_CONNECTION_STRING" --container-name "$AZURE_CONTAINER" --name demo_farm_1_history.csv --query exists | grep -q true; then
  echo "OK: demo_farm_1_history.csv found in container $AZURE_CONTAINER"
  exit 0
else
  echo "MISSING: demo_farm_1_history.csv not found in container $AZURE_CONTAINER" >&2
  exit 4
fi