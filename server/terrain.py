"""Model alanı üzerinde yüksek çözünürlüklü arazi servisi (detay-artırma).

Amaç: 6 km'lik model sıcaklığını görüntülerken vadi/sırt detayını geri
kazanmak. İstemci, T_görüntü = T_model(bilineer) + Γ·(z_model(bilineer) − z_hires)
lapse-rate düzeltmesi uygular (Γ ≈ 6.5 K/km — meteoblue tarzı fiziksel
downscaling). Bunun için iki yükseklik alanı servis edilir:

  /terrain/{region}/model.png  — modelin gördüğü arazi (nx×ny, wfe_init.bin)
  /terrain/{region}/hires.png  — gerçek arazi, model quad'ı üzerinde K× örneklenmiş
                                 (prep'teki AWS Terrain Tiles mozaiğinden)

İkisi de 16-bit R/G paketli PNG (ELEV_RANGE ile; satır 0 = kuzey).
Hires ızgara bir kez hesaplanıp .npy olarak prep dizinine önbelleklenir.
"""

import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from regions import ROOT, REGIONS, read_ini

ELEV_RANGE = (-500.0, 5500.0)   # m — istemci src/lib/ranges.ts ile birebir
K_UPSAMPLE = 6                  # hires ızgara = model çözünürlüğü × K (yakınlaştırma detayı)
TILE = 256

_PNG_CACHE = {}                 # (region, kind) -> (mtime, bytes)


def _cfg(region):
    r = REGIONS[region]
    cfg = read_ini(ROOT / r["case"])
    return cfg, ROOT / cfg["out_dir"], ROOT / cfg["input_dir"]


def _grid_latlon_hgt(region):
    """wfe_init.bin'den model lat/lon + model arazisi (ny,nx)."""
    cfg, outdir, prep = _cfg(region)
    meta = json.loads((outdir / "meta.json").read_text())
    nx, ny = meta["nx"], meta["ny"]
    npf = int(read_ini(prep / "wfe_input.ini")["np_prof"])
    raw = np.fromfile(prep / "wfe_init.bin", dtype=np.float32)
    n2 = nx * ny
    hgt = raw[4 * npf:4 * npf + n2].reshape(ny, nx)
    o = 4 * npf + 2 * n2 + 2 * n2
    plat = raw[o:o + n2].reshape(ny, nx)
    plon = raw[o + n2:o + 2 * n2].reshape(ny, nx)
    return plat, plon, hgt, prep


def _bilinear(arr, yi, xi):
    """2B dizide vektörize bilineer örnekleme (kenar-kelepçeli)."""
    h, w = arr.shape[:2]
    x = np.clip(xi, 0.0, w - 1.001)
    y = np.clip(yi, 0.0, h - 1.001)
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    fx = x - x0
    fy = y - y0
    a = arr[y0, x0]
    b = arr[y0, x0 + 1]
    c = arr[y0 + 1, x0]
    d = arr[y0 + 1, x0 + 1]
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy


def _hires_on_quad(region):
    """Gerçek arazi (terrain.bin mozaiği), model quad'ı üzerinde K× ızgara.

    Sonuç prep dizinine önbelleklenir (terrain.bin mtime'ına bağlı)."""
    plat, plon, hgt, prep = _grid_latlon_hgt(region)
    ny, nx = hgt.shape
    tb = prep / "terrain.bin"
    if not tb.exists():
        return None
    cache = prep / f"hires_quad_k{K_UPSAMPLE}.npy"
    if cache.exists() and cache.stat().st_mtime >= tb.stat().st_mtime:
        return np.load(cache)

    meta = json.loads((prep / "terrain.json").read_text())
    z, xt0, yt0 = meta["zoom"], meta["xtile0"], meta["ytile0"]
    mosaic = np.memmap(tb, dtype=np.float32, mode="r",
                       shape=(meta["ny_px"], meta["nx_px"]))

    # K× ızgaranın lat/lon'u: plat/plon'un kesirli grid koordinatında bilineeri
    W, H = nx * K_UPSAMPLE, ny * K_UPSAMPLE
    gx = (np.arange(W, dtype=np.float64) + 0.5) / K_UPSAMPLE - 0.5
    gy = (np.arange(H, dtype=np.float64) + 0.5) / K_UPSAMPLE - 0.5
    GX, GY = np.meshgrid(gx, gy)
    lat = _bilinear(plat.astype(np.float64), GY, GX)
    lon = _bilinear(plon.astype(np.float64), GY, GX)

    # lat/lon -> mozaik pikseli (Web Mercator, get_terrain.py ile aynı)
    n = TILE * (2 ** z)
    mx = (lon + 180.0) / 360.0 * n - xt0 * TILE
    latr = np.radians(lat)
    my = (1 - np.log(np.tan(latr) + 1 / np.cos(latr)) / math.pi) / 2 * n - yt0 * TILE
    hi = _bilinear(mosaic, my, mx).astype(np.float32)
    # terrarium denizde batimetri içerir (negatif derinlik) — deniz yüzeyi 0'a
    # kelepçelenir, aksi halde lapse düzeltmesi denizde sahte ısınma üretir
    np.clip(hi, 0.0, ELEV_RANGE[1], out=hi)
    np.save(cache, hi)
    return hi


def _encode16(g):
    lo, hi = ELEV_RANGE
    q = np.clip((g.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)
    v16 = np.round(q * 65535.0).astype(np.uint32)
    rgb = np.zeros((*g.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (v16 >> 8).astype(np.uint8)
    rgb[..., 1] = (v16 & 255).astype(np.uint8)
    img = Image.fromarray(np.flipud(rgb), "RGB")   # satır 0 = kuzey
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=6)
    return buf.getvalue()


def terrain_png(region, kind):
    """kind ∈ {model, hires} — 16-bit paketli yükseklik PNG'si (bytes) / None."""
    if kind not in ("model", "hires"):
        return None
    _, _, prep = _cfg(region)
    src = prep / ("terrain.bin" if kind == "hires" else "wfe_init.bin")
    if not src.exists():
        return None
    mtime = src.stat().st_mtime
    hit = _PNG_CACHE.get((region, kind))
    if hit and hit[0] == mtime:
        return hit[1]
    if kind == "model":
        _, _, hgt, _ = _grid_latlon_hgt(region)
        data = _encode16(hgt)
    else:
        hi = _hires_on_quad(region)
        if hi is None:
            return None
        data = _encode16(hi)
    _PNG_CACHE[(region, kind)] = (mtime, data)
    return data
