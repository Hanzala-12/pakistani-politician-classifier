#!/usr/bin/env bash
set -euo pipefail

# Minimal EC2 deployment helper. Assumes Docker is installed on EC2 instance and
# you have SSH access via key. This script mirrors the GitHub Actions approach
# but is intended for local use by maintainers.

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <ec2-user>@<host> <path-to-key.pem> <docker-image>"
  exit 1
fi

REMOTE=$1
KEY=$2
IMAGE=$3

ssh -i "$KEY" -o StrictHostKeyChecking=no "$REMOTE" <<EOF
  docker pull $IMAGE
  docker stop politician-api || true
  docker rm politician-api || true
  docker run -d --name politician-api -p 8000:8000 \
    -v /home/ubuntu/models:/app/models \
    -v /home/ubuntu/results:/app/results \
    --restart unless-stopped $IMAGE
EOF

echo "Deployment triggered on $REMOTE for image $IMAGE"
