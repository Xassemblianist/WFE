#!/usr/bin/env bash
# Bir spot GPU VM baslatarak WFE tahmini kosar (vm_startup.sh ile).
# Kullanim: bash deploy/launch_forecast.sh <bolge> <saat>
set -euo pipefail
cd "$(dirname "$0")"
[ -f config.env ] || { echo "once config.env olustur (config.env.example'dan)"; exit 1; }
source config.env

REGION="${1:-turkey}"
HOURS="${2:-24}"
NAME="wfe-run-${REGION}-$(date -u +%H%M%S)"

gcloud compute instances create "$NAME" \
  --project "$PROJECT_ID" --zone "$ZONE" \
  --machine-type "$MACHINE" \
  --accelerator "type=${GPU_TYPE},count=1" \
  --provisioning-model=SPOT --instance-termination-action=DELETE \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu124-ubuntu-2204 --image-project=deeplearning-platform-release \
  --boot-disk-size=60GB \
  --scopes=cloud-platform \
  --metadata="wfe_region=${REGION},wfe_hours=${HOURS},wfe_bucket=${BUCKET},wfe_repo=${REPO},wfe_cuda_arch=${CUDA_ARCH},install-nvidia-driver=True" \
  --metadata-from-file=startup-script=vm_startup.sh

echo "spot VM '$NAME' baslatildi. Loglar:"
echo "  gcloud compute instances get-serial-port-output $NAME --zone $ZONE"
