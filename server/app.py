"""WFE Tahmin API'si (FastAPI).

Model çıktılarını (harita ürünleri, nokta tahmini) servis eder ve isteğe bağlı
koşu tetikler. Serving GPU gerektirmez (üretilmiş ürünleri okur); koşu tetikleme
build/wfe.exe + prep gerektirir (GPU'lu makinede).

Çalıştırma (yerel):  uvicorn app:app --host 0.0.0.0 --port 8000  (server/ dizininden)
Dokümantasyon:       http://localhost:8000/docs
"""

import subprocess
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import products
from regions import ROOT, REGIONS, read_ini

app = FastAPI(title="WFE Forecast API",
              description="GPU-yerlisi bölgesel hava tahmin modeli — ürün servisi",
              version="1.0.0")

_jobs = {}  # basit iş durumu takibi (id -> durum)


@app.get("/health")
def health():
    return {"status": "ok", "model": "WFE", "regions": list(REGIONS)}


@app.get("/regions")
def regions():
    return [{"id": k, **{kk: vv for kk, vv in v.items() if kk != "case"}}
            for k, v in REGIONS.items()]


@app.get("/products/{region}")
def region_manifest(region: str):
    if region not in REGIONS:
        raise HTTPException(404, "bilinmeyen bölge")
    return products.manifest(region)


@app.get("/products/{region}/map/{name}")
def region_map(region: str, name: str):
    if region not in REGIONS or "/" in name or "\\" in name:
        raise HTTPException(404)
    cfg = read_ini(ROOT / REGIONS[region]["case"])
    p = ROOT / cfg["out_dir"] / name
    if not p.exists() or p.suffix != ".png":
        raise HTTPException(404, "harita yok")
    return FileResponse(p, media_type="image/png")


@app.get("/point/{region}")
def point(region: str, lat: float = Query(..., ge=-90, le=90),
          lon: float = Query(..., ge=-180, le=180)):
    if region not in REGIONS:
        raise HTTPException(404, "bilinmeyen bölge")
    try:
        return products.point_forecast(region, lat, lon)
    except FileNotFoundError:
        raise HTTPException(404, "bu bölge için henüz koşu yok")


def _run_job(job_id, region, hours):
    _jobs[job_id] = {"state": "running", "region": region}
    try:
        r = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "run_forecast.py"),
             str(ROOT / REGIONS[region]["case"]), "--hours", str(hours)],
            cwd=ROOT)
        _jobs[job_id] = {"state": "done" if r.returncode == 0 else "failed",
                         "region": region}
    except Exception as e:  # noqa
        _jobs[job_id] = {"state": "error", "detail": str(e)}


@app.post("/run/{region}")
def trigger_run(region: str, hours: int = 24):
    """Bir bölge için tahmin koşusu tetikle (arka planda; GPU'lu makinede)."""
    if region not in REGIONS:
        raise HTTPException(404, "bilinmeyen bölge")
    job_id = f"{region}-{len(_jobs)}"
    _jobs[job_id] = {"state": "queued", "region": region}
    threading.Thread(target=_run_job, args=(job_id, region, hours),
                     daemon=True).start()
    return {"job_id": job_id}


@app.get("/run/{job_id}")
def job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "iş bulunamadı")
    return {"job_id": job_id, **_jobs[job_id]}


# Web arayüzü (statik) — API rotalarindan sonra, kok dizinde
_web = ROOT / "web"
if _web.exists():
    app.mount("/", StaticFiles(directory=str(_web), html=True), name="web")
