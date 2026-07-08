"""Tek-alan georeferanslı overlay üretimi (harita bindirmesi için).

Model çıktı ızgarasını (float32 binary) meteoroloji standardına yakın renk
eşlemeleriyle şeffaf-arkaplanlı RGBA PNG'ye çevirir. Ayrıca alanın dört köşe
lat/lon'unu (maplibre `image` source'un dört-köşe quad'ı) ve renk skalası
meta verisini sunar. Alan Lambert konformal projeksiyonlu olduğundan (hafif
eğik) düz bbox yerine dört köşe kullanılır — daha doğru yerleşim.
"""

import glob
import io
from pathlib import Path

import numpy as np
from PIL import Image

from regions import ROOT, REGIONS, read_ini

# ---------------------------------------------------------------------------
# Renk eşlemeleri:  (değer, R, G, B, alfa)  — kanal-başına lineer interpolasyon.
# Değerler görüntü biriminde (t2m °C, wind m/s, precip mm, cloud g/kg-kolon).
# ---------------------------------------------------------------------------
COLORMAPS = {
    "t2m": {
        # canlı, ayrışık tonlar (soluk orta-ton çamurlaşmasına karşı):
        # moru derin soğuğa, camgöbeğini 0°C'ye, yeşil-sarı-turuncu-kırmızıyı
        # 8-35°C bandına yay; 6.5K/km lapse detayı bantlar arasında okunur.
        "label": "2 m Sıcaklık", "unit": "°C", "opacity": 0.94,
        "stops": [
            (-45, 20, 8, 48, 1), (-34, 48, 22, 120, 1), (-26, 78, 42, 182, 1),
            (-19, 70, 78, 224, 1), (-13, 56, 120, 238, 1), (-8, 52, 162, 240, 1),
            (-3, 62, 200, 234, 1), (0, 84, 224, 214, 1), (3, 96, 224, 160, 1),
            (7, 106, 216, 102, 1), (11, 160, 224, 76, 1), (15, 220, 226, 62, 1),
            (19, 248, 200, 48, 1), (23, 250, 160, 40, 1), (27, 246, 116, 38, 1),
            (31, 234, 72, 40, 1), (35, 206, 38, 46, 1), (40, 160, 18, 60, 1),
            (45, 118, 10, 70, 1), (50, 86, 8, 62, 1),
        ],
    },
    "wind": {
        "label": "10 m Rüzgâr", "unit": "m/s", "opacity": 0.78,
        "stops": [
            (0, 180, 235, 200, 0.0), (1, 175, 230, 185, 0.35),
            (3, 190, 230, 150, 0.60), (6, 235, 225, 130, 0.72),
            (9, 245, 195, 95, 0.78), (12, 240, 150, 75, 0.82),
            (16, 225, 100, 70, 0.85), (20, 205, 60, 90, 0.88),
            (26, 170, 45, 140, 0.90), (34, 120, 35, 160, 0.92),
        ],
    },
    "precip": {
        "label": "Yağış (saatlik)", "unit": "mm", "opacity": 0.88,
        "stops": [
            (0.0, 255, 255, 255, 0.0), (0.1, 190, 225, 250, 0.35),
            (0.5, 140, 200, 245, 0.55), (1, 95, 175, 240, 0.68),
            (2, 65, 140, 235, 0.76), (4, 55, 105, 225, 0.80),
            (8, 70, 80, 215, 0.83), (16, 120, 60, 205, 0.86),
            (32, 175, 55, 175, 0.88), (64, 210, 70, 130, 0.90),
        ],
    },
    "cloud": {
        "label": "Bulutluluk (kolon)", "unit": "g/kg", "opacity": 0.85,
        "stops": [
            (0.0, 255, 255, 255, 0.0), (0.03, 248, 248, 250, 0.20),
            (0.15, 238, 240, 244, 0.42), (0.5, 224, 227, 234, 0.62),
            (1.2, 205, 210, 222, 0.78), (3.0, 180, 188, 205, 0.90),
        ],
    },
    "mslp": {
        "label": "Deniz Sv. Basıncı", "unit": "hPa", "opacity": 0.72,
        "stops": [
            (980, 33, 102, 172, 0.85), (995, 67, 147, 195, 0.85),
            (1005, 146, 197, 222, 0.82), (1010, 209, 229, 240, 0.78),
            (1013, 247, 247, 247, 0.70), (1016, 253, 219, 199, 0.78),
            (1022, 244, 165, 130, 0.82), (1030, 214, 96, 77, 0.85),
            (1045, 178, 24, 43, 0.88),
        ],
    },
}

# Manifestte reklamı yapılan alanlar = veri yolu olan (field_grid destekli) alanlar.
FIELDS = ["t2m", "wind", "precip", "cloud", "mslp"]

# Veri-PNG kodlama aralıkları — istemci ile BİREBİR aynı olmalı
# (web-app/src/lib/ranges.ts). Skaler alanlar 16-bit R/G paketli, rüzgâr
# bileşenleri 8-bit (R=u, G=v). Görüntü satır-0 = kuzey (flipud).
DATA_RANGES = {
    "t2m": (-45.0, 50.0),      # °C
    "wind": (0.0, 45.0),       # m/s
    "precip": (0.0, 80.0),     # mm / çıktı aralığı
    "cloud": (0.0, 8.0),       # g/kg kolon
    "mslp": (960.0, 1050.0),   # hPa
}
UV_RANGE = 45.0                # m/s, her iki bileşen için ±


def _cfg(region):
    r = REGIONS[region]
    cfg = read_ini(ROOT / r["case"])
    return cfg, ROOT / cfg["out_dir"], ROOT / cfg["input_dir"]


def outdir_for(region, run=None):
    """Ürün dizini: güncel koşu (out/<bölge>) veya arşivlenmiş koşu.

    run = 'YYYYMMDDHH' → out/archive/<case-adı>/<run> (run_operational arşivi;
    yüzey 2B alanları içerir — 3B alanlar arşivlenmediğinden bulut/basınç/uv
    arşiv koşularında yoktur). Arşiv anahtarı case adıdır (turkey → turkey6km)."""
    _, outdir, _ = _cfg(region)
    if run:
        case_name = Path(REGIONS[region]["case"]).stem
        return ROOT / "out" / "archive" / case_name / run
    return outdir


def _dims(region, run=None):
    import json
    meta = json.loads((outdir_for(region, run) / "meta.json").read_text())
    return meta["nx"], meta["ny"], meta["nz"], meta["dt"]


def _steps(outdir):
    return sorted(int(Path(p).stem.split("_")[1])
                  for p in glob.glob(str(outdir / "thp_*.bin")))


def domain_corners(region):
    """Alanın dört köşesi (lat/lon) + bbox. wfe_init.bin'den model ızgarası."""
    nx, ny, nz, _ = _dims(region)
    _, _, prep = _cfg(region)
    npf = int(read_ini(prep / "wfe_input.ini")["np_prof"])
    raw = np.fromfile(prep / "wfe_init.bin", dtype=np.float32)
    n2 = nx * ny
    o = 4 * npf + 2 * n2 + 2 * n2
    plat = raw[o:o + n2].reshape(ny, nx)
    plon = raw[o + n2:o + 2 * n2].reshape(ny, nx)
    # maplibre image coords sırası: [TL, TR, BR, BL] (görüntü kuzey-üst)
    corners = [
        [float(plon[ny - 1, 0]), float(plat[ny - 1, 0])],       # NW (TL)
        [float(plon[ny - 1, nx - 1]), float(plat[ny - 1, nx - 1])],  # NE (TR)
        [float(plon[0, nx - 1]), float(plat[0, nx - 1])],       # SE (BR)
        [float(plon[0, 0]), float(plat[0, 0])],                 # SW (BL)
    ]
    bounds = [float(plon.min()), float(plat.min()),
              float(plon.max()), float(plat.max())]
    center = [float(plon.mean()), float(plat.mean())]
    return {"corners": corners, "bounds": bounds, "center": center}


def _read_2d(outdir, var, step, ny, nx):
    p = outdir / f"{var}_{step:06d}.bin"
    if not p.exists():
        return None
    return np.fromfile(p, dtype=np.float32).reshape(ny, nx)


def _prev_step(steps, step):
    prev = [s for s in steps if s < step]
    return prev[-1] if prev else None


# fiziksel sabitler (src/core/constants.hpp ile birebir) — mslp indirgeme için
_GRAV, _CP, _RD, _P00, _EPS61 = 9.81, 1004.5, 287.04, 1.0e5, 0.61


def _mslp_grid(region, step):
    """Deniz seviyesine indirgenmiş basınç [hPa].

    Model taban Exner'i (π=1 @ z=0, hidrostatik — base_state.cpp ile birebir)
    prep profil tablolarından (z, θ, qv) yeniden kurulur; yüzey basıncı
    π'+π_taban'dan hesaplanıp hipsometrik denklemle deniz seviyesine indirgenir.
    """
    nx, ny, nz, _ = _dims(region)
    _, outdir, prep = _cfg(region)
    pip_p = outdir / f"pip_{step:06d}.bin"
    if not pip_p.exists():
        return None
    npf = int(read_ini(prep / "wfe_input.ini")["np_prof"])
    raw = np.fromfile(prep / "wfe_init.bin", dtype=np.float32)
    n2 = nx * ny
    ztab = raw[0:npf].astype(np.float64)
    th_tab = raw[npf:2 * npf].astype(np.float64)
    qv_tab = raw[2 * npf:3 * npf].astype(np.float64)
    h = raw[4 * npf:4 * npf + n2].reshape(ny, nx).astype(np.float64)

    # taban Exner profili (hidrostatik integrasyon, π=1 @ z=0)
    dz = float(ztab[1] - ztab[0])
    thv = th_tab * (1.0 + _EPS61 * np.maximum(qv_tab, 0.0))
    pib = np.empty_like(ztab)
    pib[0] = 1.0
    for k in range(1, len(ztab)):
        pib[k] = pib[k - 1] - _GRAV * dz / (_CP * 0.5 * (thv[k - 1] + thv[k]))

    def _lev0(path):
        return np.fromfile(path, dtype=np.float32).reshape(nz, ny, nx)[0].astype(np.float64)

    pip0 = _lev0(pip_p)
    thp_p, qv_p = outdir / f"thp_{step:06d}.bin", outdir / f"qv_{step:06d}.bin"
    thp0 = _lev0(thp_p) if thp_p.exists() else np.zeros((ny, nx))
    qv0 = np.maximum(_lev0(qv_p), 0.0) if qv_p.exists() else np.zeros((ny, nx))

    hc = np.clip(h, ztab[0], ztab[-1])
    pib_s = np.interp(hc, ztab, pib)
    thb_s = np.interp(hc, ztab, th_tab)
    pi_s = np.clip(pib_s + pip0, 1e-3, None)
    p_sfc = _P00 * pi_s ** (_CP / _RD)
    Tv = (thb_s + thp0) * pi_s * (1.0 + _EPS61 * qv0)
    Tv_bar = Tv + 0.0065 * hc * 0.5          # 6.5 K/km ile katman-ortalama sanal T
    p_msl = p_sfc * np.exp(_GRAV * hc / (_RD * Tv_bar))
    return (p_msl / 100.0).astype(np.float32)  # Pa -> hPa


def field_grid(region, field, step, run=None):
    """Bir alan+adım için (ny,nx) ızgara — görüntü biriminde. Yoksa None.

    run verilirse arşivlenmiş koşudan okur (yalnız yüzey alanları)."""
    nx, ny, nz, _ = _dims(region, run)
    outdir = outdir_for(region, run)
    if field == "t2m":
        g = _read_2d(outdir, "t2m", step, ny, nx)
        return None if g is None else g - 273.15
    if field == "wind":
        return _read_2d(outdir, "u10", step, ny, nx)
    if field == "precip":
        # önceki adım: rain dosyalarının kendisinden (arşivde thp yok)
        steps = sorted(int(Path(p).stem.split("_")[1])
                       for p in glob.glob(str(outdir / "rain_*.bin")))
        cur = _read_2d(outdir, "rain", step, ny, nx)
        if cur is None:
            return None
        ps = _prev_step(steps, step)
        prev = _read_2d(outdir, "rain", ps, ny, nx) if ps is not None else None
        g = cur if prev is None else np.maximum(0.0, cur - prev)
        return g
    if run:
        return None  # 3B alanlar (cloud/mslp) arşivlenmez
    if field == "cloud":
        p = outdir / f"qc_{step:06d}.bin"
        if not p.exists():
            return None
        qc = np.fromfile(p, dtype=np.float32).reshape(nz, ny, nx)
        # kolon toplamı (g/kg) — görsel bulutluluk vekili
        return qc.sum(axis=0) * 1000.0
    if field == "mslp":
        return _mslp_grid(region, step)
    return None


def _apply_colormap(g, field):
    stops = COLORMAPS[field]["stops"]
    vs = np.array([s[0] for s in stops], dtype=np.float64)
    rgba = np.zeros((*g.shape, 4), dtype=np.float64)
    gc = np.clip(g, vs[0], vs[-1])
    for c in range(4):
        rgba[..., c] = np.interp(gc, vs, [s[c + 1] for s in stops])
    out = np.empty((*g.shape, 4), dtype=np.uint8)
    out[..., :3] = np.clip(rgba[..., :3], 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(rgba[..., 3] * 255.0, 0, 255).astype(np.uint8)
    return out


# Alan başına birincil kaynak dosya (önbellek geçersizleştirme için mtime kaynağı)
_FIELD_FILE = {"t2m": "t2m", "wind": "u10", "precip": "rain",
               "cloud": "qc", "mslp": "pip"}

_PNG_CACHE = {}       # (region, field, step, upscale) -> (mtime, bytes)
_CACHE_MAX = 256


def _src_mtime(region, field, step, run=None):
    p = outdir_for(region, run) / f"{_FIELD_FILE.get(field, field)}_{step:06d}.bin"
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def render_png(region, field, step, upscale=4):
    """Şeffaf-arkaplanlı, renk-eşlemeli overlay PNG (bytes) veya None.

    maplibre raster'ı GPU'da bilineer örneklediğinden ağır ×6 büyütme/optimize
    gereksiz — hafif büyütme + hızlı PNG (optimize kapalı) kullanılır ve sonuç
    (kaynak mtime'a göre) önbelleğe alınır (zaman kaydırıcıda anında dönüş)."""
    if field not in COLORMAPS:
        return None
    key = (region, field, step, upscale)
    mtime = _src_mtime(region, field, step)
    hit = _PNG_CACHE.get(key)
    if hit and mtime is not None and hit[0] == mtime:
        return hit[1]

    g = field_grid(region, field, step)
    if g is None:
        return None
    rgba = _apply_colormap(g, field)          # (ny,nx,4)
    img = Image.fromarray(np.flipud(rgba), "RGBA")  # kuzey-üst
    if upscale and upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=6)    # dengeli sıkıştırma (sonuç önbelleklenir)
    data = buf.getvalue()

    if mtime is not None:
        if len(_PNG_CACHE) >= _CACHE_MAX:
            _PNG_CACHE.pop(next(iter(_PNG_CACHE)))
        _PNG_CACHE[key] = (mtime, data)
    return data


def render_data_png(region, field, step, run=None):
    """Ham alan verisi, 16-bit R/G paketli RGB PNG (istemci-tarafı render için).

    v16 = (v - lo)/(hi - lo) * 65535;  R = v16>>8, G = v16&255, B = 0.
    Alfa kanalı YOK (RGB) — canvas geri-okumasında premultiply kaybı olmaz.
    Satır 0 = kuzey (overlay ile aynı yön). run: arşivlenmiş koşu."""
    if field not in DATA_RANGES:
        return None
    key = (region, "data", field, step, run)
    mtime = _src_mtime(region, field, step, run)
    hit = _PNG_CACHE.get(key)
    if hit and mtime is not None and hit[0] == mtime:
        return hit[1]

    try:
        g = field_grid(region, field, step, run)
    except (FileNotFoundError, OSError):
        return None  # arşiv dizini/meta yok → 404
    if g is None:
        return None
    lo, hi = DATA_RANGES[field]
    q = np.clip((g.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)
    v16 = np.round(q * 65535.0).astype(np.uint32)
    rgb = np.zeros((*g.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (v16 >> 8).astype(np.uint8)
    rgb[..., 1] = (v16 & 255).astype(np.uint8)
    img = Image.fromarray(np.flipud(rgb), "RGB")
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=6)
    data = buf.getvalue()
    if mtime is not None:
        if len(_PNG_CACHE) >= _CACHE_MAX:
            _PNG_CACHE.pop(next(iter(_PNG_CACHE)))
        _PNG_CACHE[key] = (mtime, data)
    return data


def render_uv_png(region, step):
    """Yüzeye-yakın rüzgâr bileşenleri (model seviye-0 u,v), 8-bit RGB PNG.

    R = u, G = v; her ikisi [-UV_RANGE, +UV_RANGE] → [0,255]. Partikül
    animasyonu için 8-bit hassasiyet yeterli. Satır 0 = kuzey."""
    nx, ny, nz, _ = _dims(region)
    _, outdir, _ = _cfg(region)
    up, vp = outdir / f"u_{step:06d}.bin", outdir / f"v_{step:06d}.bin"
    if not up.exists() or not vp.exists():
        return None
    key = (region, "uv", step)
    try:
        mtime = up.stat().st_mtime
    except OSError:
        mtime = None
    hit = _PNG_CACHE.get(key)
    if hit and mtime is not None and hit[0] == mtime:
        return hit[1]

    u0 = np.fromfile(up, dtype=np.float32).reshape(nz, ny, nx)[0]
    v0 = np.fromfile(vp, dtype=np.float32).reshape(nz, ny, nx)[0]

    def enc(a):
        q = np.clip((a + UV_RANGE) / (2.0 * UV_RANGE), 0.0, 1.0)
        return np.round(q * 255.0).astype(np.uint8)

    rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    rgb[..., 0] = enc(u0)
    rgb[..., 1] = enc(v0)
    img = Image.fromarray(np.flipud(rgb), "RGB")
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=6)
    data = buf.getvalue()
    if mtime is not None:
        if len(_PNG_CACHE) >= _CACHE_MAX:
            _PNG_CACHE.pop(next(iter(_PNG_CACHE)))
        _PNG_CACHE[key] = (mtime, data)
    return data


def field_json(region, field, step):
    """Ham ızgara değerleri (istemci-tarafı okuma/kontur için)."""
    g = field_grid(region, field, step)
    if g is None:
        return None
    ny, nx = g.shape
    return {
        "region": region, "field": field, "step": step,
        "nx": nx, "ny": ny, "unit": COLORMAPS[field]["unit"],
        "min": float(np.nanmin(g)), "max": float(np.nanmax(g)),
        # satır 0 = güney (ızgara yönü); istemci gerekiyorsa çevirir
        "values": np.round(g, 2).astype(float).tolist(),
    }


def colormap_meta(field=None):
    """Renk skalası meta verisi (efsane/legend çizimi için)."""
    def one(f):
        c = COLORMAPS[f]
        return {"field": f, "label": c["label"], "unit": c["unit"],
                "opacity": c["opacity"],
                "range": list(DATA_RANGES.get(f, (None, None))),
                "stops": [{"value": s[0], "rgba": [s[1], s[2], s[3], s[4]]}
                          for s in c["stops"]]}
    if field:
        return one(field) if field in COLORMAPS else None
    return [one(f) for f in FIELDS]
