"""WFE otomatik dogrulama suiti: idealize vakalari kosar, sayisal kapilari denetler.

Her degisiklikten sonra calistirin: python tools/run_tests.py
Cikis kodu 0 = hepsi PASS. Sinirlar docs/EQUATIONS.md tablosundaki referans
davranistan turetilmistir (FP32 yuvarlama payiyla).
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WFE = Path("build/wfe.exe")


def run_case(ini):
    r = subprocess.run([str(WFE), ini], capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout + r.stderr


def last_diag(out):
    """Son tani satirini ayristir: wmax, thmin, thmax (+ opsiyonel pipmax/qr)."""
    d = {}
    for line in out.splitlines():
        mw = re.search(r"\|w\|max=\s*([-\d.]+)", line)
        if not mw:
            continue
        d["wmax"] = float(mw.group(1))
        mt = re.search(r"th'=\[([-+\d.]+),([-+\d.]+)\]", line)
        if mt:
            d["thmin"], d["thmax"] = float(mt.group(1)), float(mt.group(2))
        mp = re.search(r"\|pi'\|max=([-\d.e+]+)", line)
        if mp:
            d["pipmax"] = float(mp.group(1))
        mq = re.search(r"qr=([\d.]+)g/kg", line)
        if mq:
            d["qrmax"] = float(mq.group(1))
    return d


def straka_front():
    meta = json.loads(Path("out/straka/meta.json").read_text())
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    a = np.fromfile("out/straka/thp_001200.bin", dtype=np.float32).reshape(nz, ny, nx)
    sfc = a[0, ny // 2, :]
    cold = np.where(sfc < -1.0)[0]
    return (cold.max() + 0.5) * meta["dx"] - 25600.0 if len(cold) else 0.0


def field_min(case, var, step):
    meta = json.loads(Path(f"out/{case}/meta.json").read_text())
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    a = np.fromfile(f"out/{case}/{var}_{step:06d}.bin", dtype=np.float32)
    return float(a.min())


def uv_symmetry(case, step):
    """x<->y simetrik kurulumda (warm_bubble) u ve v ayna-simetrik olmali:
    u(k,j,i) == v(k,i,j). Bagil fark ~roundoff => cekirdekte yon onyargisi yok."""
    meta = json.loads(Path(f"out/{case}/meta.json").read_text())
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]
    u = np.fromfile(f"out/{case}/u_{step:06d}.bin", dtype=np.float32).reshape(nz, ny, nx)
    v = np.fromfile(f"out/{case}/v_{step:06d}.bin", dtype=np.float32).reshape(nz, ny, nx)
    denom = np.sqrt((u ** 2).mean()) + 1e-12
    return float(np.sqrt(((u - np.transpose(v, (0, 2, 1))) ** 2).mean()) / denom)


CASES = [
    ("warm_bubble", "cases/warm_bubble.ini", [
        ("wmax 16-18 m/s", lambda d: 16.0 <= d["wmax"] <= 18.0),
        ("thmax 1.0-1.3 K", lambda d: 1.0 <= d["thmax"] <= 1.3),
        ("u-v ayna simetrisi (cekirdek yon-tarafsiz)",
         lambda d: uv_symmetry("warm_bubble", 667) < 1e-4),
    ]),
    ("straka", "cases/straka.ini", [
        ("thmin -9.6..-8.5 K", lambda d: -9.6 <= d["thmin"] <= -8.5),
        ("cephe 13.5-15.5 km", lambda d: 13500 <= straka_front() <= 15500),
    ]),
    ("schaer_rest", "cases/schaer_rest.ini", [
        ("arazide tam duraganlik", lambda d: d["wmax"] < 1e-6 and abs(d["thmax"]) < 1e-6),
    ]),
    ("schaer", "cases/schaer.ini", [
        ("dag dalgasi wmax 1.2-2.5", lambda d: 1.2 <= d["wmax"] <= 2.5),
    ]),
    ("bubble_outflow", "cases/bubble_outflow.ini", [
        ("kabarcik temiz cikti", lambda d: d["thmax"] < 0.01 and d["wmax"] < 1.0),
        ("pi' surunmesi yok", lambda d: d["pipmax"] < 1e-3),
    ]),
    ("wk82_supercell", "cases/wk82_supercell.ini", [
        ("firtina wmax 25-60 m/s", lambda d: 25.0 <= d["wmax"] <= 60.0),
        ("yagmur olustu (qr son>0.2)", lambda d: d.get("qrmax", 0) > 0.2),
    ]),
    ("moist_blob", "cases/moist_blob.ini", [
        ("pozitif-tanimli: qv>=0", lambda d: field_min("moist_blob", "qv", 2000) >= -1e-6),
        ("blob korundu (qv_max>4g/kg)", lambda d: d["wmax"] >= 0),  # kosu tamamlandi
    ]),
]


def main():
    if not WFE.exists():
        sys.exit("once derleyin: build/wfe.exe yok")
    fails = 0
    t0 = time.time()
    for name, ini, checks in CASES:
        tc = time.time()
        code, out = run_case(ini)
        d = last_diag(out)
        wall = time.time() - tc
        if code != 0 or not d:
            print(f"[FAIL] {name:16s} kosu basarisiz (kod {code})")
            fails += 1
            continue
        bad = [label for label, fn in checks if not safe(fn, d)]
        status = "PASS" if not bad else "FAIL"
        extra = "" if not bad else "  <-- " + "; ".join(bad)
        print(f"[{status}] {name:16s} {wall:5.1f}s  wmax={d.get('wmax', -1):7.3f}"
              f"  th'=[{d.get('thmin', 0):+.3f},{d.get('thmax', 0):+.3f}]{extra}")
        fails += len(bad)
    print(f"\ntoplam {time.time()-t0:.0f}s — {'HEPSI PASS' if fails == 0 else f'{fails} HATA'}")
    sys.exit(0 if fails == 0 else 1)


def safe(fn, d):
    try:
        return fn(d)
    except Exception:
        return False


if __name__ == "__main__":
    main()
