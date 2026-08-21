#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="${DOSERAD_IMAGE_TAG:-doserad-photon-ct:latest}"
docker build --platform linux/amd64 --tag "$IMAGE_TAG" "$REPO_DIR"
docker image inspect "$IMAGE_TAG" \
    --format '{{ index .Config.Labels "org.grand-challenge.api-method" }}' \
    | grep -Fx invoke

