"""WFE tahmin dogrulamasi: GFS hedef alanlarina karsi seviye-bazli RMSE/beceri.

Persistans (t=0 alani) referans alinir; beceri = 1 - RMSE_wfe/RMSE_pers.
Kullanim: python tools/verify.py cases/turkey.ini --fhour 24
"""

import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--fhour", type=int, default=24)
    ap.add_argument("--rim", type=int, default=12, help="disarida birakilacak kenar")
    args = ap.parse_args()

    cfg = read_ini(args.case)
    outdir = Path(cfg["out_dir"])
    prep = Path(cfg["input_dir"])
    meta = json.loads((outdir / "meta.json").read_text())
    nx, ny, nz, dt = meta["nx"], meta["ny"], meta["nz"], meta["dt"]
    n3 = nx * ny * nz
    step = int(round(args.fhour * 3600 / dt))

    def ldbdy(fh):
        a = np.fromfile(prep / f"wfe_bdy_{fh:03d}.bin", dtype=np.float32)
        return [a[i * n3:(i + 1) * n3].reshape(nz, ny, nx) for i in range(5)]

    def ld(v, s, lev=None):
        lev = lev or nz
        return np.fromfile(outdir / f"{v}_{s:06d}.bin",
                           dtype=np.float32).reshape(lev, ny, nx)

    ana = ldbdy(args.fhour)
    per = ldbdy(0)
    thb = per[2] - ld("thp", 0)
    uF = ld("u", step)
    fc = {
        "u": 0.5 * (uF[:, :, :-1] + uF[:, :, 1:]),
        "v": 0.5 * (ld("v", step)[:, :-1, :] + ld("v", step)[:, 1:, :]),
        "theta": ld("thp", step) + thb,
        "qv": ld("qv", step),
    }
    an = {"u": ana[0][:, :, :-1], "v": ana[1][:, :-1, :], "theta": ana[2],
          "qv": ana[4]}
    pe = {"u": per[0][:, :, :-1], "v": per[1][:, :-1, :], "theta": per[2],
          "qv": per[4]}

    m = args.rim
    print(f"dogrulama: t+{args.fhour}h, GFS hedefine karsi (kenar {m} hucre haric)")
    print(f"{'alan':8s} {'WFE':>8s} {'pers':>8s} {'beceri':>8s}   birim")
    units = {"u": "m/s", "v": "m/s", "theta": "K", "qv": "g/kg"}
    for name in ["theta", "u", "v", "qv"]:
        f = fc[name][:, m:ny - m, m:fc[name].shape[2] - m]
        a = an[name][:, m:ny - m, m:an[name].shape[2] - m]
        p = pe[name][:, m:ny - m, m:pe[name].shape[2] - m]
        sc = 1000 if name == "qv" else 1
        rf = np.sqrt(np.mean((f - a) ** 2)) * sc
        rp = np.sqrt(np.mean((p - a) ** 2)) * sc
        print(f"{name:8s} {rf:8.3f} {rp:8.3f} {100*(1-rf/rp):+7.0f}%   {units[name]}")

    print("\nu seviye-bazli beceri:")
    for k in range(2, nz, 4):
        f = fc["u"][k, m:ny - m, m:-m - 1]
        a = an["u"][k, m:ny - m, m:-m - 1]
        p = pe["u"][k, m:ny - m, m:-m - 1]
        rf = np.sqrt(np.mean((f - a) ** 2))
        rp = np.sqrt(np.mean((p - a) ** 2))
        print(f"  k={k:2d}: WFE={rf:5.2f} pers={rp:5.2f}  beceri={100*(1-rf/rp):+4.0f}%")


if __name__ == "__main__":
    main()
