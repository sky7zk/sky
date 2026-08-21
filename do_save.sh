#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="${DOSERAD_IMAGE_TAG:-doserad-photon-ct:latest}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

"$REPO_DIR/do_build.sh"
"$REPO_DIR/prepare_model.sh"
docker save "$IMAGE_TAG" | gzip -c > "$REPO_DIR/doserad_photon_ct_${STAMP}.tar.gz"
echo "Created $REPO_DIR/doserad_photon_ct_${STAMP}.tar.gz"

