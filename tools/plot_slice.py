"""WFE cikti gorsellestirme: x-z dikey kesit (alan ortasi y).

Kullanim: python tools/plot_slice.py out/warm_bubble --var thp --step 4000
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(outdir: Path, var: str, step: int):
    meta = json.loads((outdir / "meta.json").read_text())
    nzlev = meta["vars"][var]
    a = np.fromfile(outdir / f"{var}_{step:06d}.bin", dtype=np.float32)
    a = a.reshape(nzlev, meta["ny"], meta["nx"])
    return a, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--var", default="thp")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    a, meta = load(args.outdir, args.var, args.step)
    jmid = meta["ny"] // 2
    sl = a[:, jmid, :]

    x = (np.arange(meta["nx"]) + 0.5) * meta["dx"] / 1000.0
    z = (np.arange(sl.shape[0]) + (0.0 if sl.shape[0] > meta["nz"] else 0.5)) * meta["dz"] / 1000.0

    t = args.step * meta["dt"]
    fig, ax = plt.subplots(figsize=(9, 5))
    vmax = max(abs(sl.min()), abs(sl.max()), 1e-12)
    im = ax.pcolormesh(x, z, sl, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(im, ax=ax, label=args.var)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("z [km]")
    ax.set_title(f"{args.var}  t = {t:.0f} s  (y-orta kesit)  min={sl.min():.3f} max={sl.max():.3f}")
    out = args.out or args.outdir / f"{args.var}_{args.step:06d}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"kaydedildi: {out}")


if __name__ == "__main__":
    main()
