"""Yuksek cozunurluklu arazi indirici (AWS Terrain Tiles — anahtarsiz acik veri).

Bir model alani (case ini'nin Lambert projeksiyonu + gridi) icin AWS
elevation-tiles-prod terrarium PNG karolarini (~30-90m) indirir, RGB->yukseklik
decode eder, Web Mercator mozaigi + meta yazar. prep_gfs.py bunu GFS'in kaba
orografisi yerine kullanir -> yuksek cozunurlukte gercek dag/kiyi arazisi.

Kullanim: python tools/get_terrain.py cases/antalya.ini [--zoom 10]
"""

import argparse
import io
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from prep_gfs import Lambert, read_ini

TILE = 256


def lonlat_to_gpix(lon, lat, z):
    """lat/lon -> Web Mercator global piksel (zoom z)."""
    n = TILE * 2 ** z
    gx = (lon + 180.0) / 360.0 * n
    latr = math.radians(lat)
    gy = (1 - math.log(math.tan(latr) + 1 / math.cos(latr)) / math.pi) / 2 * n
    return gx, gy


def fetch_tile(z, x, y):
    url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
    for attempt in range(3):
        try:
            d = urllib.request.urlopen(url, timeout=30).read()
            im = np.asarray(Image.open(io.BytesIO(d)).convert("RGB")).astype(np.float32)
            return im[:, :, 0] * 256 + im[:, :, 1] + im[:, :, 2] / 256 - 32768
        except Exception as e:
            if attempt == 2:
                print(f"  karo {z}/{x}/{y} basarisiz: {e}")
                return np.zeros((TILE, TILE), np.float32)
            time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--zoom", type=int, default=10)
    ap.add_argument("--margin", type=float, default=0.4, help="alan kenar payi [derece]")
    args = ap.parse_args()

    cfg = read_ini(args.case)
    nx, ny = int(cfg["nx"]), int(cfg["ny"])
    dx, dy = float(cfg["dx"]), float(cfg["dy"])
    lat0, lon0 = float(cfg["proj_lat0"]), float(cfg["proj_lon0"])
    lat1 = float(cfg.get("proj_lat1", lat0 - 5))
    lat2 = float(cfg.get("proj_lat2", lat0 + 5))
    proj = Lambert(lat0, lon0, lat1, lat2)

    # alan kose+kenar noktalarindan lat/lon bbox
    xr = (np.arange(nx) + 0.5) * dx - nx * dx / 2
    yr = (np.arange(ny) + 0.5) * dy - ny * dy / 2
    X, Y = np.meshgrid(xr, yr)
    plat, plon = proj.to_latlon(X, Y)
    m = args.margin
    latmin, latmax = plat.min() - m, plat.max() + m
    lonmin, lonmax = plon.min() - m, plon.max() + m
    z = args.zoom
    print(f"alan: lat {latmin:.2f}..{latmax:.2f}, lon {lonmin:.2f}..{lonmax:.2f}, zoom {z}")

    gx0, gy0 = lonlat_to_gpix(lonmin, latmax, z)   # sol-ust
    gx1, gy1 = lonlat_to_gpix(lonmax, latmin, z)   # sag-alt
    xt0, yt0 = int(gx0 // TILE), int(gy0 // TILE)
    xt1, yt1 = int(gx1 // TILE), int(gy1 // TILE)
    ntx, nty = xt1 - xt0 + 1, yt1 - yt0 + 1
    print(f"karolar: {ntx}x{nty} = {ntx*nty} adet indiriliyor ...")

    mosaic = np.zeros((nty * TILE, ntx * TILE), np.float32)
    for j in range(nty):
        for i in range(ntx):
            mosaic[j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE] = \
                fetch_tile(z, xt0 + i, yt0 + j)
        print(f"  satir {j+1}/{nty}", flush=True)

    outdir = Path(cfg["input_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    mosaic.tofile(outdir / "terrain.bin")
    meta = {"zoom": z, "xtile0": xt0, "ytile0": yt0,
            "ny_px": mosaic.shape[0], "nx_px": mosaic.shape[1]}
    (outdir / "terrain.json").write_text(json.dumps(meta))
    print(f"yazildi: {outdir}/terrain.bin ({mosaic.shape}, "
          f"{mosaic.min():.0f}..{mosaic.max():.0f} m)")


if __name__ == "__main__":
    main()
