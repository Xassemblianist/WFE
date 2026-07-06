# WFE — GPU-yerlisi bölgesel hava tahmin modeli. Derleme + koşu + API tek imaj.
# GCP spot GPU (T4/L4/A100) icin: sm_75 varsayilan, CUDA_ARCH build-arg ile degistir.
#
#   docker build --build-arg CUDA_ARCH=75 -t wfe .
#   docker run --gpus all wfe python3 tools/run_forecast.py cases/turkey.ini --hours 24
#   docker run --gpus all -p 8000:8000 wfe uvicorn app:app --host 0.0.0.0 --port 8000 --app-dir server

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG CUDA_ARCH=75
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build git \
      python3 python3-pip \
      libeccodes-dev libeccodes-tools \
      libgeos-dev libproj-dev proj-data proj-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# C++/CUDA modelini derle
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
    && cmake --build build

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Varsayilan: API servisi. Kosu icin komutu gecersiz kil (yukaridaki ornek).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "server"]
