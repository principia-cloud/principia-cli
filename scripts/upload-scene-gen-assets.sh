#!/usr/bin/env bash
set -euo pipefail

# Upload scene-gen assets to S3.
# Maintainer-only — requires AWS CLI with write access to the bucket.
#
# Usage:
#   ./scripts/upload-scene-gen-assets.sh [--matfuse PATH] [--objathor PATH]
#
# Defaults assume the assets live in the standard local locations.

S3_BUCKET="principia-scene-gen-assets"
S3_REGION="us-east-1"

MATFUSE_CKPT="${HOME}/.principia/data/scene-gen/matfuse/matfuse-full.ckpt"
OBJATHOR_BASE="${HOME}/.objathor-assets/2023_09_23"

info()  { printf '\033[1;34m[info]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[ok]\033[0m    %s\n' "$*"; }
error() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --matfuse)  shift; MATFUSE_CKPT="$1" ;;
        --objathor) shift; OBJATHOR_BASE="$1" ;;
        *) error "Unknown flag: $1" ;;
    esac
    shift
done

command -v aws >/dev/null 2>&1 || error "AWS CLI is required. Install: https://aws.amazon.com/cli/"

# ---------------------------------------------------------------------------
# MatFuse checkpoint
# ---------------------------------------------------------------------------
if [ -f "$MATFUSE_CKPT" ]; then
    info "Uploading MatFuse checkpoint ($(du -h "$MATFUSE_CKPT" | cut -f1))..."
    aws s3 cp "$MATFUSE_CKPT" \
        "s3://${S3_BUCKET}/matfuse/matfuse-full.ckpt" \
        --region "$S3_REGION"
    ok "MatFuse checkpoint uploaded"
else
    error "MatFuse checkpoint not found at ${MATFUSE_CKPT}"
fi

# ---------------------------------------------------------------------------
# ObjaThor annotations + features
# ---------------------------------------------------------------------------
if [ -d "$OBJATHOR_BASE" ]; then
    info "Uploading ObjaThor annotations..."
    aws s3 cp "${OBJATHOR_BASE}/annotations.json.gz" \
        "s3://${S3_BUCKET}/objathor/2023_09_23/annotations.json.gz" \
        --region "$S3_REGION"
    ok "Annotations uploaded"

    info "Uploading ObjaThor features..."
    aws s3 cp "${OBJATHOR_BASE}/features/clip_features.pkl" \
        "s3://${S3_BUCKET}/objathor/2023_09_23/features/clip_features.pkl" \
        --region "$S3_REGION"
    aws s3 cp "${OBJATHOR_BASE}/features/sbert_features.pkl" \
        "s3://${S3_BUCKET}/objathor/2023_09_23/features/sbert_features.pkl" \
        --region "$S3_REGION"
    ok "Features uploaded"

    info "Syncing ObjaThor assets directory (this may take a while)..."
    aws s3 sync "${OBJATHOR_BASE}/assets/" \
        "s3://${S3_BUCKET}/objathor/2023_09_23/assets/" \
        --region "$S3_REGION"
    ok "Assets synced"
else
    error "ObjaThor assets directory not found at ${OBJATHOR_BASE}"
fi

ok "All scene-gen assets uploaded to s3://${S3_BUCKET}/"
