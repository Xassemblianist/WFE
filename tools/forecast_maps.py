"""WFE tahmin harita urunleri.

Her cikti zamani icin 4 panelli PNG: yuzey sicakligi + 10m-esdeger ruzgar,
orta seviye jet, bulutluluk (kolon qc+qr), donem yagisi. Cartopy varsa
Lambert projeksiyonlu kiyi cizgili; yoksa arazi konturlu duz cizim.

Kullanim: python tools/forecast_maps.py cases/turkey.ini [--steps all|last]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


def read_ini(path):
    kv = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].split(";")[0]
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--steps", default="all")
    args = ap.parse_args()

    cfg = read_ini(args.case)
    outdir = Path(cfg["out_dir"])
    meta = json.loads((outdir / "meta.json").read_text())
    nx, ny, nz, dt = meta["nx"], meta["ny"], meta["nz"], meta["dt"]
    start = read_ini(Path(cfg["input_dir"]) / "wfe_input.ini").get("start", "?")

    # projeksiyon ve koordinatlar
    lat0, lon0 = float(cfg["proj_lat0"]), float(cfg["proj_lon0"])
    lat1 = float(cfg.get("proj_lat1", lat0 - 5))
    lat2 = float(cfg.get("proj_lat2", lat0 + 5))
    prep = Path(cfg["input_dir"])
    npf = int(read_ini(prep / "wfe_input.ini")["np_prof"])
    raw = np.fromfile(prep / "wfe_init.bin", dtype=np.float32)
    o = 4 * npf
    n2 = nx * ny
    h = raw[o:o + n2].reshape(ny, nx); o += 2 * n2       # h (fcor atla)
    o += 2 * n2                                            # tsk, land atla
    plat = raw[o:o + n2].reshape(ny, nx); o += n2
    plon = raw[o:o + n2].reshape(ny, nx); o += n2

    thb = None  # gerekirse ilk adimdan turet

    def ld(v, s, lev=None):
        lev = lev or nz
        p = outdir / f"{v}_{s:06d}.bin"
        if not p.exists():
            return None
        return np.fromfile(p, dtype=np.float32).reshape(lev, ny, nx)

    def ld2(v, s):
        p = outdir / f"{v}_{s:06d}.bin"
        return np.fromfile(p, dtype=np.float32).reshape(ny, nx) if p.exists() else None

    steps = sorted(int(p.stem.split("_")[1]) for p in outdir.glob("thp_*.bin"))
    if args.steps == "last":
        steps = [steps[-1]]

    if HAS_CARTOPY:
        proj = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0,
                                     standard_parallels=(lat1, lat2))
        pc = ccrs.PlateCarree()

    def make_ax(fig, pos):
        if HAS_CARTOPY:
            ax = fig.add_subplot(2, 2, pos, projection=proj)
            ax.coastlines(resolution="50m", linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="gray")
            return ax, {"transform": pc}
        ax = fig.add_subplot(2, 2, pos)
        ax.contour(plon, plat, h, levels=[500, 1500], colors="gray", linewidths=0.5)
        ax.set_aspect(1.3)
        return ax, {}

    prev_rain = 0.0
    for si, step in enumerate(steps):
        fh = step * dt / 3600.0
        thp = ld("thp", step)
        u = ld("u", step)
        v = ld("v", step)
        qc = ld("qc", step)
        qr = ld("qr", step)
        rain = ld2("rain", step)
        tsk = ld2("tsk", step)

        fig = plt.figure(figsize=(15, 9))
        sk = max(nx // 24, 1)
        qv_kw = dict(color="k", width=0.0022)

        # 1) yuzey sicakligi + alt seviye ruzgar
        ax, tr = make_ax(fig, 1)
        if tsk is not None:
            fld, label = tsk - 273.15, "yuzey sicakligi [C]"
        else:
            fld, label = thp[0], "theta' k=0 [K]"
        cf = ax.contourf(plon, plat, fld, levels=16, cmap="turbo", **tr)
        ax.quiver(plon[::sk, ::sk], plat[::sk, ::sk], u[0, ::sk, ::sk],
                  v[0, ::sk, ::sk], **qv_kw, **tr)
        plt.colorbar(cf, ax=ax, shrink=0.85, label=label)
        ax.set_title(f"yuzey sicakligi + alt ruzgar  t+{fh:.0f}h")

        # 2) orta seviye jet
        ax, tr = make_ax(fig, 2)
        k5 = int(nz * 0.35)
        spd = np.sqrt(u[k5] ** 2 + v[k5] ** 2)
        cf = ax.contourf(plon, plat, spd, levels=15, cmap="YlOrRd", **tr)
        ax.quiver(plon[::sk, ::sk], plat[::sk, ::sk], u[k5, ::sk, ::sk],
                  v[k5, ::sk, ::sk], **qv_kw, **tr)
        plt.colorbar(cf, ax=ax, shrink=0.85, label="ruzgar [m/s]")
        ax.set_title(f"orta troposfer ruzgari  t+{fh:.0f}h")

        # 3) bulutluluk
        ax, tr = make_ax(fig, 3)
        cloud = (qc + qr).sum(axis=0) * 1000 if qc is not None else thp[5] * 0
        cf = ax.contourf(plon, plat, cloud,
                         levels=np.linspace(0, max(cloud.max(), 0.5), 12),
                         cmap="Blues", **tr)
        plt.colorbar(cf, ax=ax, shrink=0.85, label="kolon bulut+yagmur suyu [g/kg]")
        ax.set_title(f"bulutluluk  t+{fh:.0f}h")

        # 4) donem yagisi
        ax, tr = make_ax(fig, 4)
        if rain is not None:
            dr = rain - (prev_rain if si > 0 else 0)
            prev_rain = rain
            cf = ax.contourf(plon, plat, dr, levels=np.linspace(0, max(dr.max(), 1), 12),
                             cmap="YlGnBu", **tr)
            plt.colorbar(cf, ax=ax, shrink=0.85, label="donem yagisi [mm]")
            ax.set_title(f"yagis (onceki ciktidan beri, maks {dr.max():.1f} mm)")

        fig.suptitle(f"WFE tahmini | baslangic {start} UTC | t+{fh:.0f} saat",
                     fontsize=14)
        out = outdir / f"map_{step:06d}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out}")


if __name__ == "__main__":
    main()
