"""WFE cok-donguli dogrulama: birkaç GFS dongusu icin prep+kosu+skill toplar.

Tek vakadan sistematik hatayi ayirmak icin (or. v-ruzgar hatasi hep u'nun
~2 kati mi, yoksa vakaya mi bagli). Her dongu icin GFS f024'e karsi alan-ort
RMSE + seviye-bazli u/v hatasi; sonda vakalar-arasi ozet + v/u orani.

Kullanim:
  python tools/verify_multi.py cases/turkey.ini \
      --cycles 20260704:12 20260705:00 20260705:12 --hours 24
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PY = sys.executable
TOOLS = Path(__file__).parent


def read_ini(path):
    kv = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].split(";")[0]
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def skill(case_ini, fhour, rim=12):
    """Bir kosunun GFS f024'e karsi skill'i (verify.py mantigi, ic)."""
    cfg = read_ini(case_ini)
    outdir = Path(cfg["out_dir"])
    prep = Path(cfg["input_dir"])
    meta = json.loads((outdir / "meta.json").read_text())
    nx, ny, nz, dt = meta["nx"], meta["ny"], meta["nz"], meta["dt"]
    n3 = nx * ny * nz
    step = int(round(fhour * 3600 / dt))

    def ldbdy(fh):
        a = np.fromfile(prep / f"wfe_bdy_{fh:03d}.bin", dtype=np.float32)
        return [a[i * n3:(i + 1) * n3].reshape(nz, ny, nx) for i in range(5)]

    def ld(v, s, lev=None):
        return np.fromfile(outdir / f"{v}_{s:06d}.bin",
                           dtype=np.float32).reshape(lev or nz, ny, nx)

    ana, per = ldbdy(fhour), ldbdy(0)
    thb = per[2] - ld("thp", 0)
    uF = ld("u", step); vF = ld("v", step)
    uc = 0.5 * (uF[:, :, :-1] + uF[:, :, 1:])
    vc = 0.5 * (vF[:, :-1, :] + vF[:, 1:, :])
    thF = ld("thp", step) + thb
    qvF = ld("qv", step)
    m = rim
    fc = {"theta": thF, "qv": qvF,
          "u": uc, "v": vc}
    an = {"theta": ana[2], "qv": ana[4],
          "u": ana[0][:, :, :-1], "v": ana[1][:, :-1, :]}
    pe = {"theta": per[2], "qv": per[4],
          "u": per[0][:, :, :-1], "v": per[1][:, :-1, :]}
    out = {}
    for name in ["theta", "u", "v", "qv"]:
        f = fc[name][:, m:ny - m, m:fc[name].shape[2] - m]
        a = an[name][:, m:ny - m, m:an[name].shape[2] - m]
        p = pe[name][:, m:ny - m, m:pe[name].shape[2] - m]
        rf = float(np.sqrt(np.mean((f - a) ** 2)))
        rp = float(np.sqrt(np.mean((p - a) ** 2)))
        out[name] = (rf, rp)
    # seviye-ort u,v mutlak hatasi (v/u orani icin)
    uu = fc["u"][:, m:ny - m, m:-m]; au = an["u"][:, m:ny - m, m:-m]
    vv = fc["v"][:, m:ny - m, m:-m]; av = an["v"][:, m:ny - m, m:-m]
    out["_uerr"] = float(np.sqrt(np.mean((uu - au) ** 2)))
    out["_verr"] = float(np.sqrt(np.mean((vv - av) ** 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--cycles", nargs="+", required=True, help="YYYYMMDD:HH ...")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    rows = []
    for cyc in args.cycles:
        date, hh = cyc.split(":")
        print(f"\n=== dongu {date} {hh}Z ===", flush=True)
        r = subprocess.run([PY, str(TOOLS / "prep_gfs.py"), args.case,
                            "--date", date, "--cycle", hh, "--hours", str(args.hours)])
        if r.returncode != 0:
            print("  prep basarisiz, atlaniyor"); continue
        r = subprocess.run([str(Path("build") / "wfe.exe"), args.case],
                           stdout=subprocess.DEVNULL)
        if r.returncode != 0:
            print("  kosu basarisiz, atlaniyor"); continue
        s = skill(args.case, args.hours)
        vu = s["_verr"] / max(s["_uerr"], 1e-6)
        print(f"  theta {s['theta'][0]:.2f}/{s['theta'][1]:.2f}  "
              f"u {s['u'][0]:.2f}/{s['u'][1]:.2f}  v {s['v'][0]:.2f}/{s['v'][1]:.2f}  "
              f"qv {s['qv'][0]:.3f}/{s['qv'][1]:.3f}  v/u={vu:.2f}")
        rows.append((cyc, s, vu))

    if not rows:
        print("hic dongu tamamlanmadi"); return
    print(f"\n{'='*60}\nVAKALAR-ARASI OZET ({len(rows)} dongu)")
    print(f"{'alan':8s} {'WFE-ort':>9s} {'pers-ort':>9s} {'beceri':>8s}")
    for name in ["theta", "u", "v", "qv"]:
        wf = np.mean([r[1][name][0] for r in rows])
        pr = np.mean([r[1][name][1] for r in rows])
        print(f"{name:8s} {wf:9.3f} {pr:9.3f} {100*(1-wf/pr):+7.0f}%")
    vus = [r[2] for r in rows]
    print(f"\nv/u hata orani: ort {np.mean(vus):.2f}, aralik [{min(vus):.2f}, {max(vus):.2f}]")
    print("  ~sabit >1.5 => SISTEMATIK v zaafiyeti; degisken => vaka-ozel meteoroloji")


if __name__ == "__main__":
    main()
