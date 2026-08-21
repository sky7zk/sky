#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${DOSERAD_PROJECT_DIR:-/home/a123456/zk/DoseRAD_v2_transfer_20260804/project/doserad_v2}"
RUN_DIR="$PROJECT_DIR/outputs/mae_dose_exactmask_352x240x240_seed2026_lr3e4_bs8_metal_stratified"
CHECKPOINT="${DOSERAD_CHECKPOINT:-$RUN_DIR/best_full_ct_beam_mae.pth}"
NORMALIZATION="${DOSERAD_NORMALIZATION:-$PROJECT_DIR/configs/normalization_seed2026_metal_stratified.json}"

test -f "$CHECKPOINT"
test -f "$NORMALIZATION"
mkdir -p "$REPO_DIR/model"
cp "$CHECKPOINT" "$REPO_DIR/model/best_full_ct_beam_mae.pth"
cp "$NORMALIZATION" "$REPO_DIR/model/normalization.json"
tar -C "$REPO_DIR/model" -czf "$REPO_DIR/model.tar.gz" \
    best_full_ct_beam_mae.pth normalization.json

echo "Created $REPO_DIR/model.tar.gz"

