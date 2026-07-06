"""WFE çıktı ürünleri: koşu manifesti + nokta (lat/lon) tahmin zaman serisi."""

import datetime as dtm
import glob
import json
from pathlib import Path

import numpy as np

from regions import ROOT, REGIONS, read_ini


def _domain_latlon(prep_dir, nx, ny, npf):
    """wfe_init.bin'den model lat/lon + arazi yüksekliği (interior)."""
    raw = np.fromfile(prep_dir / "wfe_init.bin", dtype=np.float32)
    n2 = nx * ny
    hgt = raw[4 * npf:4 * npf + n2].reshape(ny, nx)
    o = 4 * npf + 2 * n2 + 2 * n2   # z,th,qv,u + h,fcor + tsk,land
    plat = raw[o:o + n2].reshape(ny, nx)
    plon = raw[o + n2:o + 2 * n2].reshape(ny, nx)
    return plat, plon, hgt


def _steps(outdir):
    return sorted(int(Path(p).stem.split("_")[1])
                  for p in glob.glob(str(outdir / "thp_*.bin")))


def manifest(region):
    """Bir bölgenin en güncel koşusunun ürün manifesti (JSON-uyumlu dict)."""
    r = REGIONS[region]
    cfg = read_ini(ROOT / r["case"])
    outdir = ROOT / cfg["out_dir"]
    meta_p = outdir / "meta.json"
    if not meta_p.exists():
        return {"region": region, "available": False}
    meta = json.loads(meta_p.read_text())
    prep = ROOT / cfg["input_dir"]
    start = read_ini(prep / "wfe_input.ini").get("start", "?") if \
        (prep / "wfe_input.ini").exists() else "?"
    init = None
    if start != "?":
        init = dtm.datetime.strptime(start, "%Y%m%d%H").replace(
            tzinfo=dtm.timezone.utc).isoformat()
    steps = _steps(outdir)
    dt = meta["dt"]
    maps = sorted(Path(p).name for p in glob.glob(str(outdir / "map_*.png")))
    return {
        "region": region, "title": r["title"], "available": bool(steps),
        "init": init, "dx_m": meta["dx"], "nx": meta["nx"], "ny": meta["ny"],
        "steps": [{"step": s, "fhour": round(s * dt / 3600, 1)} for s in steps],
        "maps": maps,
    }


def point_forecast(region, lat, lon):
    """En yakın grid hücresinde 2m sıcaklık / 10m rüzgâr / yağış zaman serisi."""
    r = REGIONS[region]
    cfg = read_ini(ROOT / r["case"])
    outdir = ROOT / cfg["out_dir"]
    prep = ROOT / cfg["input_dir"]
    meta = json.loads((outdir / "meta.json").read_text())
    nx, ny, dt = meta["nx"], meta["ny"], meta["dt"]
    npf = int(read_ini(prep / "wfe_input.ini")["np_prof"])
    plat, plon, hgt = _domain_latlon(prep, nx, ny, npf)
    d2 = (plat - lat) ** 2 + (plon - lon) ** 2
    idx = int(d2.argmin())
    jj, ii = divmod(idx, nx)
    if d2.flat[idx] > 1.0:
        return {"error": "nokta alan dışında"}
    start = read_ini(prep / "wfe_input.ini")["start"]
    init = dtm.datetime.strptime(start, "%Y%m%d%H").replace(tzinfo=dtm.timezone.utc)

    def rd2(var, s):
        p = outdir / f"{var}_{s:06d}.bin"
        if not p.exists():
            return None
        return float(np.fromfile(p, dtype=np.float32).reshape(ny, nx)[jj, ii])

    series, prev_rain = [], None
    for s in _steps(outdir):
        t2 = rd2("t2m", s)
        u10 = rd2("u10", s)
        rain = rd2("rain", s)
        dr = 0.0 if (rain is None or prev_rain is None) else max(0.0, rain - prev_rain)
        if rain is not None:
            prev_rain = rain
        valid = (init + dtm.timedelta(seconds=s * dt)).isoformat()
        series.append({
            "valid": valid, "fhour": round(s * dt / 3600, 1),
            "t2m_C": None if t2 is None else round(t2 - 273.15, 1),
            "wind10_ms": None if u10 is None else round(u10, 1),
            "precip_mm": round(dr, 1),
        })
    return {
        "region": region, "lat": lat, "lon": lon,
        "grid": {"i": ii, "j": jj, "elev_m": round(float(hgt[jj, ii]), 0),
                 "grid_lat": round(float(plat[jj, ii]), 3),
                 "grid_lon": round(float(plon[jj, ii]), 3)},
        "init": init.isoformat(), "series": series,
    }
