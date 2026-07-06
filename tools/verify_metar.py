"""WFE tahminini GERCEK istasyon gozlemleriyle dogrula (METAR).

GFS-vs-GFS yerine gercek yer istasyonu gozlemleri: gecerli saat icin alandaki
METAR'lari aviationweather.gov API'sinden ceker, model 2m sicaklik ve 10m
ruzgarini istasyon konumlarina en-yakin-komsu ile esler, bias/RMSE raporlar.

Kullanim: python tools/verify_metar.py cases/turkey.ini --fhour 24
Gereksinim: model ciktisi t2m/u10 iceren (physics=simple) kosudan olmali.
"""

import argparse
import datetime as dtm
import json
import urllib.request
from pathlib import Path

import numpy as np


def read_ini(path):
    kv = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].split(";")[0]
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def fetch_metars(bbox):
    """aviationweather.gov METAR JSON API — alandaki guncel gozlemler.
    bbox = (lat_min, lon_min, lat_max, lon_max). API tarihsel sorgu desteklemez;
    yalniz guncel/son gozlemler doner (gecerli zaman ~simdi olmali)."""
    b = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    url = f"https://aviationweather.gov/api/data/metar?bbox={b}&format=json&hours=3"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "wfe-verify"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"METAR indirilemedi: {e}")
        return []
    return data if isinstance(data, list) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--fhour", type=int, default=24)
    args = ap.parse_args()

    cfg = read_ini(args.case)
    outdir = Path(cfg["out_dir"])
    prep = Path(cfg["input_dir"])
    meta = json.loads((outdir / "meta.json").read_text())
    nx, ny, nz, dt = meta["nx"], meta["ny"], meta["nz"], meta["dt"]
    if "t2m" not in meta["vars"]:
        print("model ciktisi 2m/10m tani icermiyor (physics=simple ile kos)")
        return
    imeta = read_ini(prep / "wfe_input.ini")
    start = imeta["start"]
    valid = dtm.datetime.strptime(start, "%Y%m%d%H").replace(
        tzinfo=dtm.timezone.utc) + dtm.timedelta(hours=args.fhour)

    # model istasyon eslemesi icin lat/lon + arazi yuksekligi (prep wfe_init.bin'den)
    npf = int(imeta["np_prof"])
    raw = np.fromfile(prep / "wfe_init.bin", dtype=np.float32)
    n2 = nx * ny
    hgt = raw[4 * npf:4 * npf + n2].reshape(ny, nx)          # model arazi yuksekligi [m]
    o = 4 * npf + 2 * n2 + 2 * n2  # z,th,qv,u + h,fcor + tsk,land
    plat = raw[o:o + n2].reshape(ny, nx); o += n2
    plon = raw[o:o + n2].reshape(ny, nx); o += n2

    step = int(round(args.fhour * 3600 / dt))
    t2m = np.fromfile(outdir / f"t2m_{step:06d}.bin", dtype=np.float32).reshape(ny, nx)
    u10 = np.fromfile(outdir / f"u10_{step:06d}.bin", dtype=np.float32).reshape(ny, nx)

    bbox = (float(plat.min()), float(plon.min()), float(plat.max()), float(plon.max()))
    print(f"gecerli zaman: {valid:%Y-%m-%d %H}Z | alan lat {bbox[0]:.1f}..{bbox[2]:.1f}")
    obs = fetch_metars(bbox)
    print(f"{len(obs)} METAR alindi")
    if not obs:
        print("gozlem yok — API/ag erisimini kontrol edin")
        return
    # gecerli zamana ±90 dk penceredeki gozlemleri sec
    vts = valid.timestamp()
    obs = [o for o in obs if abs(o.get("obsTime", 0) - vts) <= 5400]
    if not obs:
        print(f"UYARI: gecerli zamana (±90dk) yakin gozlem yok — API yalniz guncel "
              f"veri dondurur; gecerli zaman ~simdi olan bir tahmin kosun "
              f"(run_forecast.py). Atlaniyor.")
        return
    print(f"gecerli zamana yakin {len(obs)} gozlem eslendi")

    latf, lonf = plat.ravel(), plon.ravel()
    hgtf = hgt.ravel()
    LAPSE = 0.0065  # K/m — istasyon-model yukseklik farki icin duzeltme
    dT, dTr, dW, nT, nW = [], [], [], 0, 0
    for o in obs:
        try:
            slat, slon = float(o["lat"]), float(o["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        d2 = (latf - slat) ** 2 + (lonf - slon) ** 2
        idx = int(d2.argmin())
        if d2[idx] > 0.5:  # ~0.7 derece: alan disi
            continue
        jj, ii = divmod(idx, nx)
        temp = o.get("temp")
        if temp is not None:
            raw_d = t2m[jj, ii] - 273.15 - float(temp)
            dT.append(raw_d)
            # yukseklik duzeltmesi: model hucresini istasyon yuksekligine indir
            selev = o.get("elev")
            if selev is not None:
                corr = t2m[jj, ii] + LAPSE * (hgtf[idx] - float(selev))
                dTr.append(corr - 273.15 - float(temp))
            nT += 1
        ws = o.get("wspd")  # knot
        if ws is not None:
            dW.append(u10[jj, ii] - float(ws) * 0.514444)
            nW += 1

    print(f"\n{'alan':22s} {'bias':>8s} {'RMSE':>8s} {'N':>5s}")
    if nT:
        dT = np.array(dT)
        print(f"{'2m sicaklik (ham)':22s} {dT.mean():+8.2f} {np.sqrt((dT**2).mean()):8.2f} "
              f"{nT:5d}   C")
    if dTr:
        dTr = np.array(dTr)
        print(f"{'2m sicaklik (yuks.duz.)':22s} {dTr.mean():+8.2f} "
              f"{np.sqrt((dTr**2).mean()):8.2f} {len(dTr):5d}   C")
    if nW:
        dW = np.array(dW)
        print(f"{'10m ruzgar':22s} {dW.mean():+8.2f} {np.sqrt((dW**2).mean()):8.2f} "
              f"{nW:5d}   m/s")


if __name__ == "__main__":
    main()
