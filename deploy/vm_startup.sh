#!/usr/bin/env bash
# WFE spot GPU VM baslangic scripti — GCP metadata olarak gecirilir.
# Repoyu ceker, derler, tahmini kosar, urunleri GCS'e yukler, VM'i siler.
# Deep Learning VM imaji (CUDA + surucu onceden kurulu) varsayilir.
set -euo pipefail

# metadata'dan parametreler (launch_forecast.sh gecirir)
META="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
HDR="Metadata-Flavor: Google"
REGION=$(curl -s -H "$HDR" "$META/wfe_region")
HOURS=$(curl -s -H "$HDR" "$META/wfe_hours")
BUCKET=$(curl -s -H "$HDR" "$META/wfe_bucket")
REPO=$(curl -s -H "$HDR" "$META/wfe_repo")
CUDA_ARCH=$(curl -s -H "$HDR" "$META/wfe_cuda_arch")

cd /opt
[ -d WFE ] || git clone --depth 1 "$REPO" WFE
cd WFE

# Python bagimliliklari + eccodes
pip3 install -q -r requirements.txt || true
apt-get install -y libeccodes-tools >/dev/null 2>&1 || true

# derle (CUDA arch VM GPU'suna gore)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
cmake --build build

# operasyonel tahmin (en guncel GFS dongusu, prep+kosu+harita+netcdf+dogrulama)
python3 tools/run_forecast.py "cases/${REGION}.ini" --hours "${HOURS}"

# urunleri GCS'e yukle (harita png + netcdf + meta)
OUT="out/${REGION}"
STAMP=$(date -u +%Y%m%d_%H%M)
gsutil -m cp "${OUT}"/map_*.png "gs://${BUCKET}/${REGION}/${STAMP}/" || true
gsutil -m cp "${OUT}"/meta.json "${OUT}"/wfe_out.nc \
    "gs://${BUCKET}/${REGION}/${STAMP}/" || true
# "latest" isaretcisi (servis bunu okur)
echo "${STAMP}" | gsutil cp - "gs://${BUCKET}/${REGION}/latest.txt" || true

# spot VM'i sil (kendini kapat -> maliyet durur)
NAME=$(curl -s -H "$HDR" "http://metadata.google.internal/computeMetadata/v1/instance/name")
ZONE=$(curl -s -H "$HDR" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $NF}')
gcloud --quiet compute instances delete "$NAME" --zone "$ZONE"
