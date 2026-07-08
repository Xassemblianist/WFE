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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

import overlay
import products
import terrain
from regions import ROOT, REGIONS, read_ini

app = FastAPI(title="WFE Forecast API",
              description="GPU-yerlisi bölgesel hava tahmin modeli — ürün servisi",
              version="1.0.0")

# Ayrı dev sunucudan (ör. Vite :5173) erişim için CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs = {}  # basit iş durumu takibi (id -> durum)


@app.get("/health")
def health():
    return {"status": "ok", "model": "WFE", "regions": list(REGIONS)}


@app.get("/regions")
def regions():
    return [{"id": k, **{kk: vv for kk, vv in v.items() if kk != "case"}}
            for k, v in REGIONS.items()]


def _check_run(run):
    if run is not None and not (len(run) == 10 and run.isdigit()):
        raise HTTPException(400, "geçersiz koşu (YYYYMMDDHH bekleniyor)")
    return run


@app.get("/runs/{region}")
def region_runs(region: str):
    """Gezinebilir koşular: güncel + arşiv (geçmiş tahminler)."""
    if region not in REGIONS:
        raise HTTPException(404, "bilinmeyen bölge")
    return products.runs_list(region)


@app.get("/products/{region}")
def region_manifest(region: str, run: str | None = None):
    if region not in REGIONS:
        raise HTTPException(404, "bilinmeyen bölge")
    return products.manifest(region, _check_run(run))


@app.get("/products/{region}/map/{name}")
def region_map(region: str, name: str):
    if region not in REGIONS or "/" in name or "\\" in name:
        raise HTTPException(404)
    cfg = read_ini(ROOT / REGIONS[region]["case"])
    p = ROOT / cfg["out_dir"] / name
    if not p.exists() or p.suffix != ".png":
        raise HTTPException(404, "harita yok")
    return FileResponse(p, media_type="image/png")


@app.get("/overlay/{region}/{field}/{step}.png")
def overlay_png(region: str, field: str, step: int):
    """Tek alan, şeffaf arka planlı, renk-eşlemeli overlay PNG (harita bindirmesi)."""
    if region not in REGIONS:
        raise HTTPException(404, "bilinmeyen bölge")
    if field not in overlay.FIELDS:
        raise HTTPException(404, "bilinmeyen alan")
    png = overlay.render_png(region, field, step)
    if png is None:
        raise HTTPException(404, "bu adım/alan için veri yok")
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/data/{region}/{field}/{step}.png")
def data_png(region: str, field: str, step: int, run: str | None = None):
    """Ham alan verisi — 16-bit R/G paketli RGB PNG (istemci-tarafı render).

    run=YYYYMMDDHH → arşivlenmiş (geçmiş) koşudan."""
    if region not in REGIONS:
        raise HTTPException(404, "bilinmeyen bölge")
    if field not in overlay.FIELDS:
        raise HTTPException(404, "bilinmeyen alan")
    run = _check_run(run)
    png = overlay.render_data_png(region, field, step, run)
    if png is None:
        raise HTTPException(404, "bu adım/alan için veri yok")
    # ETag=mtime + no-cache: aynı döngü yeniden koşulursa (dosya değişir)
    # tarayıcı 304 yeniden-doğrulamasıyla bayat kareyi atar
    mt = overlay._src_mtime(region, field, step, run)
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "no-cache",
                             "ETag": f'"{mt}"' if mt else '"0"'})


@app.get("/uv/{region}/{step}.png")
def uv_png(region: str, step: int):
    """Yüzeye-yakın rüzgâr bileşenleri (u,v) — 8-bit RGB PNG (partiküller)."""
    if region not in REGIONS:
        raise HTTPException(404, "bilinmeyen bölge")
    png = overlay.render_uv_png(region, step)
    if png is None:
        raise HTTPException(404, "bu adım için rüzgâr verisi yok")
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/terrain/{region}/{kind}.png")
def terrain_png(region: str, kind: str):
    """Yükseklik alanları (model / hires) — istemci-tarafı lapse-rate detaylandırma."""
    if region not in REGIONS:
        raise HTTPException(404, "bilinmeyen bölge")
    png = terrain.terrain_png(region, kind)
    if png is None:
        raise HTTPException(404, "arazi verisi yok")
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=604800"})


@app.get("/field/{region}/{field}/{step}.json")
def field_values(region: str, field: str, step: int):
    """Ham ızgara değerleri (istemci-tarafı renklendirme/kontur için)."""
    if region not in REGIONS:
        raise HTTPException(404, "bilinmeyen bölge")
    if field not in overlay.FIELDS:
        raise HTTPException(404, "bilinmeyen alan")
    j = overlay.field_json(region, field, step)
    if j is None:
        raise HTTPException(404, "bu adım/alan için veri yok")
    return j


@app.get("/colormap")
def colormaps():
    """Tüm alanların renk skalası meta verisi (efsane çizimi için)."""
    return overlay.colormap_meta()


@app.get("/colormap/{field}")
def colormap_one(field: str):
    m = overlay.colormap_meta(field)
    if m is None:
        raise HTTPException(404, "bilinmeyen alan")
    return m


@app.get("/point/{region}")
def point(region: str, lat: float = Query(..., ge=-90, le=90),
          lon: float = Query(..., ge=-180, le=180), run: str | None = None):
    if region not in REGIONS:
        raise HTTPException(404, "bilinmeyen bölge")
    try:
        return products.point_forecast(region, lat, lon, _check_run(run))
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
