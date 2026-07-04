"""WFE binary ciktilarini tek bir CF-esintili NetCDF dosyasina toplar.

Kullanim: python tools/to_netcdf.py cases/turkey.ini [-o out.nc]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def read_ini(path):
    kv = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].split(";")[0]
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


UNITS = {"u": "m s-1", "v": "m s-1", "w": "m s-1", "thp": "K", "pip": "1",
         "qv": "kg kg-1", "qc": "kg kg-1", "qr": "kg kg-1", "rain": "mm",
         "tsk": "K"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    cfg = read_ini(args.case)
    outdir = Path(cfg["out_dir"])
    meta = json.loads((outdir / "meta.json").read_text())
    nx, ny, nz, dt = meta["nx"], meta["ny"], meta["nz"], meta["dt"]

    steps = sorted(int(p.stem.split("_")[1]) for p in outdir.glob("thp_*.bin"))
    times = np.array([s * dt for s in steps], dtype=np.float64)

    coords = {
        "time": ("time", times, {"units": "seconds since model start"}),
        "x": ("x", (np.arange(nx) + 0.5) * meta["dx"], {"units": "m"}),
        "y": ("y", (np.arange(ny) + 0.5) * meta["dy"], {"units": "m"}),
        "z": ("z", np.arange(nz, dtype=np.float64), {"long_name": "model level"}),
    }
    zc_p = outdir / "zc.bin"
    data = {}
    if zc_p.exists():
        zc = np.fromfile(zc_p, dtype=np.float32).reshape(nz, ny, nx)
        data["zc"] = (("z", "y", "x"), zc, {"units": "m", "long_name": "cell height"})

    for var, lev in meta["vars"].items():
        arrs = []
        for s in steps:
            p = outdir / f"{var}_{s:06d}.bin"
            if not p.exists():
                arrs = None
                break
            a = np.fromfile(p, dtype=np.float32)
            arrs.append(a.reshape(lev, ny, nx) if lev > 1 else a.reshape(ny, nx))
        if arrs is None:
            continue
        stack = np.stack(arrs)
        dims = ("time", "z", "y", "x") if lev == nz else (
            ("time", "zw", "y", "x") if lev == nz + 1 else ("time", "y", "x"))
        data[var] = (dims, stack, {"units": UNITS.get(var, "")})

    ds = xr.Dataset(data, coords=coords,
                    attrs={"title": "WFE regional forecast", "source": "WFE",
                           "dx_m": meta["dx"], "dt_s": dt})
    out = args.out or str(outdir / "wfe_out.nc")
    enc = {v: {"zlib": True, "complevel": 3} for v in ds.data_vars}
    ds.to_netcdf(out, encoding=enc)
    print(f"yazildi: {out} ({Path(out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
