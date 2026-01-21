#!/usr/bin/env bash
set -euo pipefail

# Local helper to verify demo CSV uploaded to AWS S3
# Usage:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_REGION=us-west-2   # optional
#   export S3_BUCKET=my-bucket
#   ./scripts/check_s3_upload.sh

if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" || -z "${S3_BUCKET:-}" ]]; then
  echo "Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and S3_BUCKET environment vars" >&2
  exit 2
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "Installing AWS CLI..."
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp
  sudo /tmp/aws/install
fi

aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
if [[ -n "${AWS_REGION:-}" ]]; then aws configure set region "$AWS_REGION"; fi

S3_PREFIX=${S3_PREFIX:-demo_artifacts}
OBJECT="s3://${S3_BUCKET}/${S3_PREFIX}/demo_farm_1_history.csv"

echo "Checking for $OBJECT"
if aws s3 ls "$OBJECT" >/dev/null 2>&1; then
  echo "OK: demo_farm_1_history.csv found in $OBJECT"
  exit 0
else
  echo "MISSING: demo_farm_1_history.csv not found in $OBJECT" >&2
  exit 4
fi