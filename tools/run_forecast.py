"""WFE operasyonel tahmin pipeline'i: tek komutla indir -> koss -> haritala.

En guncel kullanilabilir GFS dongusunu NOMADS'ta bulur, on isler, modeli
kosar, harita urunlerini ve dogrulama raporunu uretir.

Kullanim:
  python tools/run_forecast.py cases/turkey.ini [--hours 24] [--date YYYYMMDD --cycle HH]
"""

import argparse
import datetime as dtm
import subprocess
import sys
import urllib.request
from pathlib import Path

PY = sys.executable
TOOLS = Path(__file__).parent


def cycle_available(date, cyc, fh):
    url = (f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
           f"?dir=%2Fgfs.{date}%2F{cyc}%2Fatmos&file=gfs.t{cyc}z.pgrb2.0p25.f{fh:03d}"
           f"&var_TMP=on&lev_500_mb=on&subregion=&leftlon=30&rightlon=31&toplat=40&bottomlat=39")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read(4) == b"GRIB"
    except Exception:
        return False


def latest_cycle(hours):
    """Istenen tahmin uzunlugunun tamami hazir olan en guncel dongu."""
    now = dtm.datetime.now(dtm.timezone.utc)
    for back in range(0, 5):
        t = now - dtm.timedelta(hours=6 * back)
        cyc = f"{(t.hour // 6) * 6:02d}"
        date = t.strftime("%Y%m%d")
        if cycle_available(date, cyc, hours):
            return date, cyc
    raise RuntimeError("uygun GFS dongusu bulunamadi")


def run(cmd):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        sys.exit(f"HATA: {cmd[0]} kodu {r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--date", default=None)
    ap.add_argument("--cycle", default=None)
    ap.add_argument("--skip-model", action="store_true", help="yalniz urunler")
    args = ap.parse_args()

    if args.date and args.cycle:
        date, cyc = args.date, args.cycle
    else:
        print("en guncel GFS dongusu araniyor...", flush=True)
        date, cyc = latest_cycle(args.hours)
    print(f"GFS dongusu: {date} {cyc}Z, +{args.hours}h")

    if not args.skip_model:
        run([PY, TOOLS / "prep_gfs.py", args.case, "--date", date, "--cycle", cyc,
             "--hours", args.hours])
        # t_end'i istenen tahmin uzunluguna gore ayarla (ini'deki degeri gecersiz kil)
        run([Path("build") / "wfe.exe", args.case, f"t_end={args.hours * 3600}"])
    run([PY, TOOLS / "forecast_maps.py", args.case])
    run([PY, TOOLS / "to_netcdf.py", args.case])
    run([PY, TOOLS / "verify.py", args.case, "--fhour", args.hours])
    print("\npipeline tamam.")


if __name__ == "__main__":
    main()
