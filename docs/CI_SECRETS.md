# CI Secrets Reference

This file lists optional repository secrets used by the Qdrant Memory Smoke workflow for publishing demo artifacts.

- GDRIVE_SERVICE_ACCOUNT (base64-encoded Google service account JSON)
  - Purpose: Upload demo CSV to Google Drive using rclone and a service account.
  - Notes: Base64-encode the JSON key to avoid newlines in GitHub secrets.

- GDRIVE_FOLDER (optional)
  - Purpose: Target Google Drive folder name where the CSV is uploaded (default: `FarmFederate-demo-artifacts`).

- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION (optional)
- S3_BUCKET
  - Purpose: When set, the workflow uploads the demo CSV to `s3://${S3_BUCKET}/demo_artifacts/` and verifies presence.
  - Notes: Ensure the IAM key has permissions to PutObject and List on the target bucket.

- AZURE_STORAGE_CONNECTION_STRING
- AZURE_CONTAINER
  - Purpose: When set, the workflow uploads the demo CSV to the specified Azure Blob container and verifies presence.

Security recommendations
- Use least privilege credentials for all provider secrets.
- Rotate/revoke keys routinely and restrict access to the minimum set of repos and actions.
