# Downloading CI Artifact (demo-farm-history)

This document explains how to securely download the `demo-farm-history` artifact produced by the Qdrant memory smoke CI run and inspect the demo CSV (`results/demo_farm_1_history.csv`).

## Options

### PowerShell (Windows)
- A helper script is available: `scripts/download_demo_artifact.ps1`.
- Run from repo root:
  - pwsh scripts\download_demo_artifact.ps1 -ArtifactId 5211661711 -Lines 20
- The script will prompt for a short-lived GitHub PAT (enter securely). It downloads, extracts, and prints the first N lines.

### Bash / curl (Linux / macOS / Windows w/ WSL)
- A helper script is available: `scripts/download_demo_artifact.sh`.
- Make it executable and run from repo root:
  - chmod +x scripts/download_demo_artifact.sh
  - ./scripts/download_demo_artifact.sh 5211661711 20
- The script prompts for a short-lived GitHub PAT and prints the top N lines of the CSV.

## Security Notes
- Create a short-lived PAT with minimal scopes (Actions read / repo read) only for this task.
- Revoke the PAT immediately after you've inspected the artifact:
  - GitHub → Settings → Developer settings → Personal access tokens → Revoke
- Do NOT paste PATs into chat or public places.

## Troubleshooting
- If you receive an HTTP 401: ensure the PAT has the correct scopes and is not expired.
- If no CSV is found in the extracted archive: the artifact may not contain CSVs or the naming differs; the scripts will list extracted files for inspection.

## Publish directly from CI to Google Drive (optional)
You can configure the Qdrant CI workflow to upload `results/demo_farm_1_history.csv` to Google Drive automatically when the demo runs.

How it works (already implemented in workflow):
- The workflow looks for the repository secret `GDRIVE_SERVICE_ACCOUNT` (a **base64-encoded** Google service account JSON). If present, it installs `rclone`, writes the decoded JSON to a temp file, configures an `rclone` drive remote using the service account, and uploads `results/demo_farm_1_history.csv` into the drive folder `FarmFederate-demo-artifacts` (default). The step is non-fatal if upload fails.

Setup steps:
1. Create a Google Cloud service account with the Drive API enabled and add a JSON key. See: https://cloud.google.com/iam/docs/creating-managing-service-account-keys
2. Base64-encode the JSON file (to make passing it via a secret safe):
   - cat key.json | base64 -w0  # Linux/macOS
   - for Windows PowerShell: [Convert]::ToBase64String([IO.File]::ReadAllBytes('key.json'))
3. In GitHub repo Settings → Secrets → Actions, add a new secret named `GDRIVE_SERVICE_ACCOUNT` with the base64 value from step 2.
4. Optional: change the target folder by editing the workflow `GDRIVE_FOLDER` env value (default `FarmFederate-demo-artifacts`).

Notes:
- The step is gated by presence of `GDRIVE_SERVICE_ACCOUNT`, so it will do nothing unless you configure the secret.
- Use least privilege for the service account: prefer `drive.file` scope and restrict to a single folder if possible.

CI Verification (optional, recommended)
- When `GDRIVE_SERVICE_ACCOUNT` is set, the workflow now runs a verification step that lists the target Drive folder and fails the job if `demo_farm_1_history.csv` is not found. This gives earlier feedback that the export+upload succeeded.
- You can override the default Drive folder by setting a secret `GDRIVE_FOLDER` to the desired folder name; otherwise the workflow uses `FarmFederate-demo-artifacts`.

Local verification helper
- A helper script is available: `scripts/check_gdrive_upload.sh`.
- Usage:
  - export GDRIVE_SERVICE_ACCOUNT=<base64-json>
  - export GDRIVE_FOLDER=FarmFederate-demo-artifacts   # optional
  - ./scripts/check_gdrive_upload.sh

If you'd like, I can also add an additional GitHub Actions job that copies the CSV into a cloud object storage (S3 / Azure Blob) instead of Drive; tell me which provider you'd like and I will draft that workflow change.

## AWS S3 (optional)
You can configure the CI workflow to upload the CSV to an S3 bucket. Steps:
1. Create an S3 bucket (e.g., `my-bucket`) and ensure the account used has PutObject permissions to the bucket.
2. In GitHub repo Settings → Secrets → Actions, add the following secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION` (optional)
   - `S3_BUCKET` (bucket name)
3. The workflow will upload `results/demo_farm_1_history.csv` to `s3://${S3_BUCKET}/demo_artifacts/` and verify its existence.

Local verification helper: `scripts/check_s3_upload.sh` — set the environment variables above and run the script.

## Azure Blob (optional)
You can configure the CI workflow to upload the CSV to an Azure Storage container. Steps:
1. Create a storage account and a container (e.g., `demo-artifacts`).
2. Generate a connection string and add it to GitHub repo Settings → Secrets → Actions as `AZURE_STORAGE_CONNECTION_STRING`.
3. Add the container name as secret `AZURE_CONTAINER`.
4. The workflow will upload `results/demo_farm_1_history.csv` to the specified container and verify it exists.

Local verification helper: `scripts/check_azure_upload.sh` — set `AZURE_STORAGE_CONNECTION_STRING` and `AZURE_CONTAINER` env vars and run the script.
