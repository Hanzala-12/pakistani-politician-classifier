#!/usr/bin/env bash
set -euo pipefail

# Lightweight script to initialise DVC remote placeholders for this project.
# Run locally to configure a remote: set REMOTE_NAME and REMOTE_URL then run.

REMOTE_NAME=${1:-default}
REMOTE_URL=${2:-}

if [ -z "$REMOTE_URL" ]; then
  echo "Usage: $0 REMOTE_NAME REMOTE_URL"
  echo "Example: $0 s3://my-bucket/dvc-cache"
  exit 1
fi

if ! command -v dvc >/dev/null 2>&1; then
  echo "dvc not installed. Install with: pip install dvc[s3]"
  exit 1
fi

echo "Adding DVC remote '$REMOTE_NAME' -> $REMOTE_URL"
dvc remote add -f $REMOTE_NAME "$REMOTE_URL"
echo "Run 'dvc push' after configuring credentials to push tracked data." 
