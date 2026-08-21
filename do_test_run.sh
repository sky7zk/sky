#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${1:-$REPO_DIR/test/input}"
OUTPUT_DIR="${2:-$REPO_DIR/test/output}"
IMAGE_TAG="${DOSERAD_IMAGE_TAG:-doserad-photon-ct:latest}"
CONTAINER_NAME="doserad-photon-ct-test-$$"

test -d "$INPUT_DIR"
"$REPO_DIR/prepare_model.sh"
"$REPO_DIR/do_build.sh"
mkdir -p "$OUTPUT_DIR"

cleanup() {
    docker rm --force "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run --detach --rm --network=none --gpus all \
    --name "$CONTAINER_NAME" \
    --volume "$REPO_DIR/model:/opt/ml/model:ro" \
    --volume "$INPUT_DIR:/input:ro" \
    --volume "$OUTPUT_DIR:/output" \
    "$IMAGE_TAG" >/dev/null

healthy=0
for _ in $(seq 1 30); do
    if docker exec "$CONTAINER_NAME" python -c \
        'import urllib.request; urllib.request.urlopen("http://127.0.0.1:4743/health").read()' \
        >/dev/null 2>&1; then
        healthy=1
        break
    fi
    sleep 5
done
if [[ "$healthy" != 1 ]]; then
    docker logs "$CONTAINER_NAME"
    echo "Container did not become healthy" >&2
    exit 1
fi

docker exec "$CONTAINER_NAME" python -c \
    'import urllib.request; r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:4743/invoke", method="POST")); assert r.status == 201, r.status'

docker logs "$CONTAINER_NAME"
test "$(find "$OUTPUT_DIR/images" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 10
echo "Verified Grand Challenge outputs in $OUTPUT_DIR"

