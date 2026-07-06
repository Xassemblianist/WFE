"""WFE gercek veri on islemcisi (WPS muadili).

NOAA GFS 0.25 GRIB2 verisini NOMADS'tan indirir, Lambert konformal model
gridine yatay/dikey interpole eder ve C++ cekirdegin okuyacagi baslangic +
sinir kosulu dosyalarini yazar.

Kullanim:
  python tools/prep_gfs.py cases/turkey.ini --date 20260704 --cycle 00 --hours 24

Cikti (ini'deki input_dir altina):
  wfe_input.ini    meta (grid, zamanlar)
  wfe_init.bin     taban profilleri + arazi + f + baslangic 3B alanlari
  wfe_bdy_FFF.bin  her sinir saati icin 5 alan (u, v, th, pi, qv)

Gereksinim: numpy, eccodes (pip install eccodes)
"""

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import eccodes as ec

RD, CP, P00, GRAV = 287.04, 1004.5, 1.0e5, 9.81
OMEGA, REARTH = 7.292e-5, 6370000.0

LEVELS = [1000, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500,
          450, 400, 350, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]


def read_ini(path):
    kv = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].split(";")[0]
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


class Lambert:
    """Kuresel Lambert konformal konik (NWP standardi, R=6370 km)."""

    def __init__(self, lat0, lon0, lat1, lat2):
        f1, f2, f0 = map(math.radians, (lat1, lat2, lat0))
        self.lon0 = math.radians(lon0)
        if abs(lat1 - lat2) < 1e-6:
            self.n = math.sin(f1)
        else:
            self.n = (math.log(math.cos(f1) / math.cos(f2)) /
                      math.log(math.tan(math.pi / 4 + f2 / 2) /
                               math.tan(math.pi / 4 + f1 / 2)))
        self.F = math.cos(f1) * math.tan(math.pi / 4 + f1 / 2) ** self.n / self.n
        self.rho0 = REARTH * self.F / math.tan(math.pi / 4 + f0 / 2) ** self.n

    def to_latlon(self, x, y):
        """Grid metre koordinatlarindan (merkez=0,0) lat/lon [derece]."""
        rho = np.sign(self.n) * np.sqrt(x * x + (self.rho0 - y) ** 2)
        th = np.arctan2(x, self.rho0 - y)
        lon = np.degrees(self.lon0 + th / self.n)
        lat = np.degrees(2 * np.arctan((REARTH * self.F / rho) ** (1.0 / self.n)) -
                         math.pi / 2)
        return lat, lon

    def wind_alpha(self, lon_deg):
        """Dunya->grid ruzgar rotasyon acisi [rad]."""
        return self.n * (np.radians(lon_deg) - self.lon0)


def build_zeta(cfg, nz):
    """C++ Metric ile birebir ayni dikey seviye insasi."""
    dz = float(cfg.get("dz", 500))
    zw = np.zeros(nz + 1)
    if cfg.get("stretch", "none") == "geometric":
        d = float(cfg.get("dz0", dz))
        ratio = float(cfg.get("dz_ratio", 1.05))
        dzmax = float(cfg.get("dz_max", dz * 4))
        for k in range(1, nz + 1):
            zw[k] = zw[k - 1] + d
            d = min(d * ratio, dzmax)
    else:
        zw = np.arange(nz + 1) * dz
    zc = 0.5 * (zw[:-1] + zw[1:])
    return zw, zc


def download(date, cyc, fh, cache: Path, bbox):
    fname = f"gfs.t{cyc}z.pgrb2.0p25.f{fh:03d}"
    out = cache / f"{date}_{cyc}_{fh:03d}.grib2"
    if out.exists() and out.stat().st_size > 10000:
        return out
    lvars = "".join(f"&var_{v}=on" for v in
                    ["UGRD", "VGRD", "TMP", "RH", "HGT", "PRES", "LAND", "SOILW"])
    levs = ("".join(f"&lev_{p}_mb=on" for p in LEVELS) + "&lev_surface=on"
            + "&lev_0-0.1_m_below_ground=on")
    url = ("https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
           f"?dir=%2Fgfs.{date}%2F{cyc}%2Fatmos&file={fname}{lvars}{levs}"
           f"&subregion=&leftlon={bbox[0]}&rightlon={bbox[1]}"
           f"&toplat={bbox[3]}&bottomlat={bbox[2]}")
    print(f"  indiriliyor: f{fh:03d} ...", flush=True)
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, out)
            if out.stat().st_size >= 10000 and out.read_bytes()[:4] == b"GRIB":
                return out
        except Exception as e:
            print(f"    deneme {attempt+1} basarisiz: {e}")
        out.unlink(missing_ok=True)
        if attempt < 2:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"indirme 3 denemede basarisiz: f{fh:03d}")


def read_grib(path):
    """(shortName, level) -> 2D alan [lat artan, lon artan] + lat/lon eksenleri."""
    fields, lats, lons = {}, None, None
    with open(path, "rb") as f:
        while True:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                break
            name = ec.codes_get(gid, "shortName")
            ltype = ec.codes_get(gid, "typeOfLevel")
            lev = ec.codes_get(gid, "level")
            ni, nj = ec.codes_get(gid, "Ni"), ec.codes_get(gid, "Nj")
            vals = np.asarray(ec.codes_get_values(gid)).reshape(nj, ni)
            lat1 = ec.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
            lat2 = ec.codes_get(gid, "latitudeOfLastGridPointInDegrees")
            lon1 = ec.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
            lon2 = ec.codes_get(gid, "longitudeOfLastGridPointInDegrees")
            if lat1 > lat2:  # kuzeyden guneye taranmis -> cevir
                vals = vals[::-1, :]
                lat1, lat2 = lat2, lat1
            if lats is None:
                lats = np.linspace(lat1, lat2, nj)
                lons = np.linspace(lon1, lon2, ni)
            key = (name, lev if ltype == "isobaricInhPa" else ltype)
            fields[key] = vals
            ec.codes_release(gid)
    return fields, lats, lons


def bilinear(field, lats, lons, plat, plon):
    """Duzgun lat-lon gridinden noktalara bilinear interpolasyon."""
    fi = np.clip((plon - lons[0]) / (lons[1] - lons[0]), 0, len(lons) - 1.001)
    fj = np.clip((plat - lats[0]) / (lats[1] - lats[0]), 0, len(lats) - 1.001)
    i0 = fi.astype(int)
    j0 = fj.astype(int)
    wx = fi - i0
    wy = fj - j0
    return ((1 - wy) * ((1 - wx) * field[j0, i0] + wx * field[j0, i0 + 1]) +
            wy * ((1 - wx) * field[j0 + 1, i0] + wx * field[j0 + 1, i0 + 1]))


def vinterp(prof, zsrc, ztgt):
    """Kolon bazli dikey dogrusal interpolasyon (vektorize).
    prof, zsrc: (nl, ny, nx) — zsrc artan; ztgt: (nz, ny, nx)."""
    nl = zsrc.shape[0]
    idx = (zsrc[None, :, :, :] < ztgt[:, None, :, :]).sum(axis=1)  # (nz,ny,nx)
    idx = np.clip(idx, 1, nl - 1)
    zlo = np.take_along_axis(zsrc[None].repeat(ztgt.shape[0], 0), idx[:, None] - 1, 1)[:, 0]
    zhi = np.take_along_axis(zsrc[None].repeat(ztgt.shape[0], 0), idx[:, None], 1)[:, 0]
    flo = np.take_along_axis(prof[None].repeat(ztgt.shape[0], 0), idx[:, None] - 1, 1)[:, 0]
    fhi = np.take_along_axis(prof[None].repeat(ztgt.shape[0], 0), idx[:, None], 1)[:, 0]
    w = np.clip((ztgt - zlo) / np.maximum(zhi - zlo, 1e-6), 0.0, 1.0)
    return flo * (1 - w) + fhi * w


def qsat_of(p, T):
    es = 610.78 * np.exp(17.269 * (T - 273.16) / (T - 35.86))
    return 0.622 * es / np.maximum(p - es, 1.0)


def sample_hires_terrain(prep_dir, plat, plon, cell_m):
    """Yuksek coz. DEM'i (AWS terrarium mozaik) model gridine alan-ortalamali
    ornekle. Web Mercator piksel -> kutu-yumusatma (hucre boyutu) -> bilinear."""
    meta = json.loads((Path(prep_dir) / "terrain.json").read_text())
    z = meta["zoom"]
    dem = np.fromfile(Path(prep_dir) / "terrain.bin",
                      dtype=np.float32).reshape(meta["ny_px"], meta["nx_px"])
    TILE = 256
    n = TILE * 2 ** z
    latm = np.radians(plat.mean())
    pix_m = 40075016.0 * math.cos(latm) / n            # piksel boyutu [m]
    w = max(1, int(round(cell_m / pix_m)))             # hucre = kac piksel
    # ayrilabilir kutu-yumusatma (integral goruntu ile hizli)
    k = np.ones(w) / w
    sm = dem
    for ax in (0, 1):
        sm = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), ax, sm)
    # model noktalari -> global piksel -> yerel piksel
    gx = (plon + 180.0) / 360.0 * n - meta["xtile0"] * TILE
    latr = np.radians(plat)
    gy = (1 - np.log(np.tan(latr) + 1 / np.cos(latr)) / np.pi) / 2 * n \
        - meta["ytile0"] * TILE
    gx = np.clip(gx, 0, sm.shape[1] - 1.001)
    gy = np.clip(gy, 0, sm.shape[0] - 1.001)
    i0 = gx.astype(int); j0 = gy.astype(int)
    fx = gx - i0; fy = gy - j0
    h = ((1 - fy) * ((1 - fx) * sm[j0, i0] + fx * sm[j0, i0 + 1]) +
         fy * ((1 - fx) * sm[j0 + 1, i0] + fx * sm[j0 + 1, i0 + 1]))
    return np.maximum(h, 0.0)


def smooth121(a, npass=2):
    for _ in range(npass):
        b = a.copy()
        b[1:-1, :] = 0.25 * a[:-2, :] + 0.5 * a[1:-1, :] + 0.25 * a[2:, :]
        a = b.copy()
        a[:, 1:-1] = 0.25 * b[:, :-2] + 0.5 * b[:, 1:-1] + 0.25 * b[:, 2:]
    return a


def process_hour(path, proj, plat, plon, z3):
    """Bir GRIB dosyasindan model gridine 5 alan: u, v, th, pi, qv."""
    fields, lats, lons = read_grib(path)
    nl = len(LEVELS)
    ny, nx = plat.shape
    gh = np.zeros((nl, ny, nx))
    uu = np.zeros_like(gh)
    vv = np.zeros_like(gh)
    tt = np.zeros_like(gh)
    rh = np.zeros_like(gh)
    lev_sorted = LEVELS  # 1000 -> 10 mb sirali: gh ARTAN (vinterp bunu bekler)
    for l, p in enumerate(lev_sorted):
        gh[l] = bilinear(fields[("gh", p)], lats, lons, plat, plon)
        uu[l] = bilinear(fields[("u", p)], lats, lons, plat, plon)
        vv[l] = bilinear(fields[("v", p)], lats, lons, plat, plon)
        tt[l] = bilinear(fields[("t", p)], lats, lons, plat, plon)
        rh[l] = bilinear(fields[("r", p)], lats, lons, plat, plon)

    lnp = np.log(np.array(lev_sorted, dtype=np.float64) * 100.0)[:, None, None]
    lnp3 = np.broadcast_to(lnp, gh.shape)

    u3 = vinterp(uu, gh, z3)
    v3 = vinterp(vv, gh, z3)
    t3 = vinterp(tt, gh, z3)
    r3 = vinterp(rh, gh, z3)
    p3 = np.exp(vinterp(lnp3, gh, z3))

    th3 = t3 * (P00 / p3) ** (RD / CP)
    pi3 = (p3 / P00) ** (RD / CP)
    qv3 = np.clip(r3, 0, 100) / 100.0 * qsat_of(p3, t3)

    # dunya -> grid ruzgar rotasyonu
    alpha = proj.wind_alpha(plon)
    ca, sa = np.cos(alpha), np.sin(alpha)
    ug = v3 * sa + u3 * ca
    vg = v3 * ca - u3 * sa
    return (ug.astype(np.float32), vg.astype(np.float32), th3.astype(np.float32),
            pi3.astype(np.float32), qv3.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", help="model case ini dosyasi (grid + projeksiyon)")
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--cycle", default="00")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--bdy-interval", type=int, default=3, help="[saat]")
    args = ap.parse_args()

    cfg = read_ini(args.case)
    nx, ny, nz = int(cfg["nx"]), int(cfg["ny"]), int(cfg["nz"])
    dx, dy = float(cfg["dx"]), float(cfg["dy"])
    lat0, lon0 = float(cfg["proj_lat0"]), float(cfg["proj_lon0"])
    lat1 = float(cfg.get("proj_lat1", lat0 - 5))
    lat2 = float(cfg.get("proj_lat2", lat0 + 5))
    outdir = Path(cfg["input_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    cache = Path("out/gfs_cache")
    cache.mkdir(parents=True, exist_ok=True)

    proj = Lambert(lat0, lon0, lat1, lat2)
    xr = (np.arange(nx) + 0.5) * dx - nx * dx / 2
    yr = (np.arange(ny) + 0.5) * dy - ny * dy / 2
    X, Y = np.meshgrid(xr, yr)
    plat, plon = proj.to_latlon(X, Y)
    print(f"alan: lat {plat.min():.1f}..{plat.max():.1f}, lon {plon.min():.1f}..{plon.max():.1f}")
    bbox = (math.floor(plon.min()) - 3, math.ceil(plon.max()) + 3,
            math.floor(plat.min()) - 3, math.ceil(plat.max()) + 3)

    zw, zc = build_zeta(cfg, nz)
    zt = zw[-1]
    print(f"dikey: nz={nz}, zt={zt/1000:.1f} km")

    # arazi + yuzey alanlari: f000'den
    f0 = download(args.date, args.cycle, 0, cache, bbox)
    fields0, lats, lons = read_grib(f0)
    if cfg.get("terrain_source", "gfs") == "tiles":
        h = sample_hires_terrain(outdir, plat, plon, dx)   # yuksek coz. DEM
        h = smooth121(h, npass=1)                          # hafif: metrik stabilite
        print(f"arazi (yuksek coz. DEM): {h.min():.0f}..{h.max():.0f} m")
    else:
        h = bilinear(fields0[("orog", "surface")], lats, lons, plat, plon)
        h = smooth121(h, npass=2)
        h = np.maximum(h, 0.0)
        print(f"arazi (GFS orografisi): {h.min():.0f}..{h.max():.0f} m")

    fcor = 2 * OMEGA * np.sin(np.radians(plat))
    tsk = bilinear(fields0[("t", "surface")], lats, lons, plat, plon)
    lkey = ("lsm", "surface") if ("lsm", "surface") in fields0 else ("land", "surface")
    land = (bilinear(fields0[lkey], lats, lons, plat, plon) > 0.5).astype(np.float64)
    skey = next((k for k in fields0 if k[0] == "soilw"), None)
    if skey is not None:
        soilw = bilinear(fields0[skey], lats, lons, plat, plon)
    else:
        soilw = np.full_like(tsk, 0.25)  # SOILW yoksa ilkim ortalamasi
    soilw = np.clip(soilw, 0.0, 0.6)
    print(f"yuzey: TSK {tsk.min():.0f}..{tsk.max():.0f} K, kara %{land.mean()*100:.0f}, "
          f"toprak nemi {soilw[land>0.5].mean():.2f} m3/m3")

    # model fiziksel yukseklikleri (Gal-Chen, C++ ile birebir)
    z3 = h[None, :, :] + zc[:, None, None] * (zt - h[None, :, :]) / zt

    hours = list(range(0, args.hours + 1, args.bdy_interval))
    init_fields = None
    for fh in hours:
        path = download(args.date, args.cycle, fh, cache, bbox)
        u, v, th, pi, qv = process_hour(path, proj, plat, plon, z3)
        arr = np.concatenate([a.ravel() for a in (u, v, th, pi, qv)]).astype(np.float32)
        arr.tofile(outdir / f"wfe_bdy_{fh:03d}.bin")
        print(f"  f{fh:03d}: th {th.min():.0f}..{th.max():.0f} K, "
              f"|u|max {max(abs(u.min()), abs(u.max())):.0f} m/s, "
              f"qv_max {qv.max()*1000:.1f} g/kg")
        if fh == 0:
            init_fields = (u, v, th, pi, qv)

    # taban profilleri: t0 alanlarinin duz-z tablosuna alan ortalamasi
    npf = int(zt // 100) + 2
    ztab = np.arange(npf) * 100.0
    zt3 = np.broadcast_to(ztab[:, None, None], (npf, ny, nx))
    th_tab = vinterp(init_fields[2].astype(np.float64), z3, zt3).mean(axis=(1, 2))
    qv_tab = vinterp(init_fields[4].astype(np.float64), z3, zt3).mean(axis=(1, 2))
    u_tab = vinterp(init_fields[0].astype(np.float64), z3, zt3).mean(axis=(1, 2))

    with open(outdir / "wfe_init.bin", "wb") as f:
        ztab.astype(np.float32).tofile(f)
        th_tab.astype(np.float32).tofile(f)
        qv_tab.astype(np.float32).tofile(f)
        u_tab.astype(np.float32).tofile(f)
        h.astype(np.float32).tofile(f)
        fcor.astype(np.float32).tofile(f)
        tsk.astype(np.float32).tofile(f)
        land.astype(np.float32).tofile(f)
        plat.astype(np.float32).tofile(f)
        plon.astype(np.float32).tofile(f)
        soilw.astype(np.float32).tofile(f)
        for a in init_fields:
            a.astype(np.float32).tofile(f)

    meta = [
        f"version = 3",
        f"nx = {nx}", f"ny = {ny}", f"nz = {nz}",
        f"np_prof = {npf}",
        f"start = {args.date}{args.cycle}",
        f"bdy_interval = {args.bdy_interval * 3600}",
        f"n_bdy = {len(hours)}",
        f"hours = {args.hours}",
    ]
    (outdir / "wfe_input.ini").write_text("\n".join(meta) + "\n")
    print(f"tamam: {outdir}/wfe_init.bin + {len(hours)} sinir dosyasi")


if __name__ == "__main__":
    main()
