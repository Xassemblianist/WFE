"""WFE topluluk (ensemble) tahmini: N uye, baslangic pertubasyonlu.

Her uye farkli seed ile kucuk theta' pertubasyonundan baslar; buyume
belirsizligi orneklenir. Topluluk ortalamasi (en olasi) + yayilim (spread,
guven) hesaplanir ve haritalanir.

Kullanim (once prep yapilmis olmali):
  python tools/run_ensemble.py cases/turkey.ini --members 8 --amp 0.5 --fhour 24
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--amp", type=float, default=0.5, help="IC theta' pert. RMS [K]")
    ap.add_argument("--fhour", type=int, default=24)
    ap.add_argument("--nudge-tau", type=float, default=86400,
                    help="uye ic nudging [s] (uzun=zayif, yayilima izin verir)")
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()

    cfg = read_ini(args.case)
    base_out = Path(cfg["out_dir"])
    exe = str(Path("build") / "wfe.exe")

    members = []
    for mth in range(args.members):
        mdir = f"{base_out}_ens/m{mth:02d}"
        members.append(mdir)
        if args.skip_run:
            continue
        seed = 1000 + mth * 7
        # uye 0 kontrol (pertubasyonsuz), digerleri pertubasyonlu.
        # ic nudging zayiflatilir (nudge_tau uzun) ki uyeler ayrisip belirsizligi
        # ornekleyebilsin (guclu nudging yayilimi bastirir).
        amp = 0.0 if mth == 0 else args.amp
        print(f"uye {mth} (seed {seed}, amp {amp}) ...", flush=True)
        r = subprocess.run([exe, args.case, f"out_dir={mdir}",
                            f"ic_perturb={amp}", f"ens_seed={seed}",
                            f"nudge_tau={args.nudge_tau}"],
                           stdout=subprocess.DEVNULL)
        if r.returncode != 0:
            sys.exit(f"uye {mth} kosusu basarisiz (kod {r.returncode})")

    # topluluk istatistikleri
    meta = json.loads((Path(members[0]) / "meta.json").read_text())
    nx, ny, nz, dt = meta["nx"], meta["ny"], meta["nz"], meta["dt"]
    step = int(round(args.fhour * 3600 / dt))

    def load2(mdir, var):
        p = Path(mdir) / f"{var}_{step:06d}.bin"
        return np.fromfile(p, dtype=np.float32).reshape(ny, nx) if p.exists() else None

    def load3(mdir, var, k):
        a = np.fromfile(Path(mdir) / f"{var}_{step:06d}.bin", dtype=np.float32)
        return a.reshape(nz, ny, nx)[k]

    prep = Path(cfg["input_dir"])
    npf = int(read_ini(prep / "wfe_input.ini")["np_prof"])
    raw = np.fromfile(prep / "wfe_init.bin", dtype=np.float32)
    n2 = nx * ny
    o = 4 * npf + 2 * n2 + 2 * n2
    plat = raw[o:o + n2].reshape(ny, nx); o += n2
    plon = raw[o:o + n2].reshape(ny, nx)

    # 850hPa ~ k=3 seviyesi ruzgar hizi ve t2m topluluk
    t2m = np.stack([load2(m, "t2m") for m in members])
    k5 = int(nz * 0.35)
    u = np.stack([load3(m, "u", k5) for m in members])   # ham (ny,nx), yari-hucre kaymasi onemsiz
    v = np.stack([load3(m, "v", k5) for m in members])
    spd = np.sqrt(u ** 2 + v ** 2)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    x, y = plon, plat
    ax = axs[0, 0]
    cf = ax.contourf(x, y, t2m.mean(0) - 273.15, levels=16, cmap="turbo")
    fig.colorbar(cf, ax=ax, label="2m T ort [C]"); ax.set_title("Topluluk ortalamasi: 2m sicaklik")
    ax = axs[0, 1]
    cf = ax.contourf(x, y, t2m.std(0), levels=12, cmap="plasma")
    fig.colorbar(cf, ax=ax, label="2m T yayilim [K]")
    ax.set_title(f"Yayilim (belirsizlik): 2m sicaklik  (maks {t2m.std(0).max():.1f}K)")
    ax = axs[1, 0]
    cf = ax.contourf(x, y, spd.mean(0), levels=15, cmap="YlOrRd")
    fig.colorbar(cf, ax=ax, label="ruzgar ort [m/s]"); ax.set_title("Topluluk ort.: orta seviye ruzgar")
    ax = axs[1, 1]
    cf = ax.contourf(x, y, spd.std(0), levels=12, cmap="plasma")
    fig.colorbar(cf, ax=ax, label="ruzgar yayilim [m/s]")
    ax.set_title(f"Yayilim: ruzgar  (maks {spd.std(0).max():.1f} m/s)")
    for a in axs.flat:
        a.set_xlabel("lon"); a.set_aspect(1.3)
    fig.suptitle(f"WFE {args.members}-uyeli topluluk | t+{args.fhour}h", fontsize=13)
    out = base_out.parent / f"{base_out.name}_ens" / "ensemble.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\ntopluluk yayilimi: 2m T ort {t2m.std(0).mean():.2f}K "
          f"(maks {t2m.std(0).max():.2f}K), ruzgar ort {spd.std(0).mean():.2f} m/s")
    print(f"kaydedildi: {out}")


if __name__ == "__main__":
    main()
