"""WFE yerel operasyonel dongu: gozetimsiz surekli tahmin (1-2 hafta).

En guncel uygun GFS dongusunu bulur, her bolge icin tam pipeline'i
(prep + model + harita + netcdf + dogrulama) `run_forecast.py` ile kosar,
urunleri saklar, her seyi loglar, ~25 dk uyur ve tekrarlar.

Tasarim ilkeleri:
  * ROBUST — her bolge kosusu try/except icinde; bir bolge/dongu cokerse
    LOGLA ve DEVAM ET. Dongu ASLA olmez. Ag/GFS aksamasinda dongu atlanir.
  * YENIDEN BASLATILABILIR — islenmis (tarih,dongu) durum dosyasina yazilir;
    yeniden baslatinca tamamlananlar tekrar kosulmaz (kaldigi yerden devam).
  * DISK YONETIMI — GFS onbellegi ve harita arsivi --keep-days'ten eski
    olunca budanir (out/<bolge> zaten her kosuda uzerine yazilir = sabit boyut).
  * LATEST — out/<bolge> her zaman en guncel kosu (API bunu okur). Ayrica
    out/operational_status.json her bolgenin son basarili dongusunu tutar.

Windows'ta cron yok — bu dongu acik birakilarak (veya Gorev Zamanlayici ile)
calisir. Log: out/operational.log (ana) + out/logs/<bolge>_<dongu>.log (kosu).

Kullanim:
  python tools/run_operational.py --days 14 --regions turkey6km antalya
  python tools/run_operational.py --once --regions antalya      # tek dongu (test)
  python tools/run_operational.py --days 7 --hours 24 --sleep-min 25 --keep-days 3
"""

import argparse
import datetime as dtm
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
PY = sys.executable

# run_forecast.py'nin dongu-bulma mantigini yeniden kullan (tools/ path'te)
sys.path.insert(0, str(TOOLS))
from run_forecast import latest_cycle, read_ini  # noqa: E402


# --------------------------------------------------------------------------
# loglama
# --------------------------------------------------------------------------
def now_utc():
    return dtm.datetime.now(dtm.timezone.utc)


def stamp():
    return now_utc().strftime("%Y-%m-%d %H:%M:%SZ")


_LOG_PATH = ROOT / "out" / "operational.log"


def log(msg):
    line = f"[{stamp()}] {msg}"
    print(line, flush=True)
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass  # loglama asla dongu oldurmez


# --------------------------------------------------------------------------
# durum (yeniden baslatilabilirlik)
# --------------------------------------------------------------------------
def load_state(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"processed": {}, "regions": {}}


def save_state(path, state):
    try:
        Path(path).write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as e:
        log(f"UYARI durum yazilamadi: {e}")


def is_done(state, region, cycle_key):
    return cycle_key in state.get("processed", {}).get(region, [])


def mark_done(state, region, cycle_key):
    state.setdefault("processed", {}).setdefault(region, [])
    if cycle_key not in state["processed"][region]:
        state["processed"][region].append(cycle_key)
        # yalniz son ~40 dongu anahtarini tut (durum dosyasi sismesin)
        state["processed"][region] = state["processed"][region][-40:]


# --------------------------------------------------------------------------
# disk yonetimi
# --------------------------------------------------------------------------
def prune_old_files(directory, keep_days):
    """Bir dizindeki --keep-days'ten eski dosyalari sil (GFS onbellegi)."""
    d = Path(directory)
    if not d.is_dir():
        return
    cutoff = time.time() - keep_days * 86400
    n = 0
    for p in d.iterdir():
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
                n += 1
        except OSError:
            pass
    if n:
        log(f"disk: {d.name}/ icinden {n} eski dosya budandi (>{keep_days}g)")


def prune_old_dirs(parent, keep_days):
    """Zaman-damgali arsiv alt-dizinlerini buda (>keep_days)."""
    d = Path(parent)
    if not d.is_dir():
        return
    cutoff = time.time() - keep_days * 86400
    n = 0
    for sub in d.iterdir():
        try:
            if sub.is_dir() and sub.stat().st_mtime < cutoff:
                shutil.rmtree(sub, ignore_errors=True)
                n += 1
        except OSError:
            pass
    if n:
        log(f"disk: {d}/ icinden {n} eski arsiv dizini budandi (>{keep_days}g)")


def archive_maps(region, outdir, cycle_key):
    """Urunleri zaman-damgali arsive kopyala: harita PNG + meta + YUZEY
    alanlari (t2m/u10/rain, 2B kucuk dosyalar) — web arayuzunde "gecmis
    kosular" gezilebilsin. 3B alanlar (qc/pip/u/v) boyut nedeniyle arsivlenmez
    (arsiv kosularinda bulut/basinc katmani ve partikuller kapali kalir)."""
    dest = ROOT / "out" / "archive" / region / cycle_key
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for pat in ("map_*.png", "meta.json", "run_info.txt",
                    "t2m_*.bin", "u10_*.bin", "rain_*.bin"):
            for p in Path(outdir).glob(pat):
                shutil.copy2(p, dest / p.name)
    except OSError as e:
        log(f"UYARI {region} arsivleme basarisiz: {e}")


# --------------------------------------------------------------------------
# dogrulama ozeti ayristirma
# --------------------------------------------------------------------------
def parse_verify_summary(text):
    """run_forecast ciktisindan kompakt dogrulama+saglik ozeti."""
    skills = []
    for name in ("theta", "u", "v", "qv"):
        m = re.search(rf"^{name}\s+[-\d.]+\s+[-\d.]+\s+([+-]?\d+)%", text, re.M)
        if m:
            skills.append(f"{name}{int(m.group(1)):+d}%")
    flags = []
    low = text.lower()
    if re.search(r"patla|blow|nan|inf\b|acil", low):
        flags.append("PATLAMA/NaN?")
    if re.search(r"cfl.*(asil|exceed|>)", low):
        flags.append("CFL?")
    summary = "beceri " + " ".join(skills) if skills else "dogrulama-yok"
    if flags:
        summary += "  [" + ";".join(flags) + "]"
    return summary


# --------------------------------------------------------------------------
# bir bolge kosusu
# --------------------------------------------------------------------------
def _stream_subprocess(cmd, runlog, timeout_s=3 * 3600):
    """Alt-sureci kos; ciktisini CANLI olarak log dosyasina akit (gozetimsiz
    sistemde ilerleme gorunur olsun). (out_metni, donus_kodu) dondur. Zaman
    asiminda sureci oldur (rc=124). Tam cikti her zaman log dosyasinda; bellekte
    yalniz son satirlar (ozet ayristirma icin) tutulur."""
    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
    except Exception as e:  # noqa — baslatma hatasi dongu oldurmesin
        try:
            Path(runlog).write_text(f"[baslatilamadi] {e}\n", encoding="utf-8")
        except OSError:
            pass
        return f"[baslatilamadi] {e}", 1
    timed_out = {"v": False}
    timer = threading.Timer(timeout_s,
                            lambda: (timed_out.__setitem__("v", True), proc.kill()))
    timer.start()
    lines = []
    try:
        with open(runlog, "w", encoding="utf-8") as lf:
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                lines.append(line)
                if len(lines) > 8000:      # bellek siniri; log dosyasi tam kalir
                    del lines[:3000]
            proc.wait()
    finally:
        timer.cancel()
    rc = 124 if timed_out["v"] else proc.returncode
    out = "".join(lines)
    if timed_out["v"]:
        out += "\n[TIMEOUT asildi — surec olduruldu]"
    return out, rc


def run_region(region, date, cyc, hours):
    """Tam pipeline'i bir bolge icin kos. (rc, sure_s, ozet) dondur."""
    case = ROOT / "cases" / f"{region}.ini"
    if not case.exists():
        return 127, 0.0, f"case yok: {case}"
    logdir = ROOT / "out" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    runlog = logdir / f"{region}_{date}{cyc}.log"
    cmd = [PY, str(TOOLS / "run_forecast.py"), str(case),
           "--hours", str(hours), "--date", date, "--cycle", cyc]
    t0 = time.time()
    out, rc = _stream_subprocess(cmd, runlog)
    dt_s = time.time() - t0
    summary = parse_verify_summary(out)
    if rc == 0:
        cfg = read_ini(case)
        archive_maps(region, ROOT / cfg["out_dir"], f"{date}{cyc}")
    return rc, dt_s, summary


def update_status(region, date, cyc, rc, summary, secs):
    """out/operational_status.json — bolge basina son dongu ozeti."""
    p = ROOT / "out" / "operational_status.json"
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {}
    init = None
    try:
        init = dtm.datetime.strptime(f"{date}{cyc}", "%Y%m%d%H").replace(
            tzinfo=dtm.timezone.utc).isoformat()
    except ValueError:
        pass
    st[region] = {
        "cycle": f"{date}{cyc}", "init": init,
        "state": "ok" if rc == 0 else "failed", "exit_code": rc,
        "summary": summary, "runtime_s": round(secs, 0),
        "updated": now_utc().isoformat(),
    }
    try:
        p.write_text(json.dumps(st, indent=2), encoding="utf-8")
    except OSError:
        pass


# --------------------------------------------------------------------------
# ana dongu
# --------------------------------------------------------------------------
def interruptible_sleep(minutes, deadline):
    """Kucuk parcalarla uyu; --days son tarihini asma."""
    end = time.time() + minutes * 60
    while time.time() < end:
        if deadline and now_utc() >= deadline:
            return
        time.sleep(min(30.0, end - time.time()))


def one_iteration(regions, hours, state, state_path, keep_days, do_archive):
    """Tek dongu adimi: en guncel dongu -> islenmemis bolgeleri kos."""
    try:
        date, cyc = latest_cycle(hours)
    except Exception as e:  # noqa — GFS/ag hatasi: dongu atla
        log(f"GFS dongusu bulunamadi ({e}); atlaniyor")
        return
    cycle_key = f"{date}{cyc}"
    pending = [r for r in regions if not is_done(state, r, cycle_key)]
    if not pending:
        log(f"dongu {cycle_key}Z zaten islenmis ({', '.join(regions)}); bekleniyor")
        return
    log(f"=== dongu {cycle_key}Z +{hours}h | islenecek: {', '.join(pending)} ===")

    for region in pending:
        try:
            log(f"[{region}] baslatiliyor ...")
            rc, secs, summary = run_region(region, date, cyc, hours)
            mins = secs / 60.0
            if rc == 0:
                log(f"[{region}] BASARILI ({mins:.1f} dk) | {summary}")
                mark_done(state, region, cycle_key)
            else:
                log(f"[{region}] BASARISIZ kod={rc} ({mins:.1f} dk) | {summary} "
                    f"| ayrinti: out/logs/{region}_{cycle_key}.log")
            update_status(region, date, cyc, rc, summary, secs)
            save_state(state_path, state)
        except Exception as e:  # noqa — bir bolge cokse bile dongu yasar
            log(f"[{region}] ISTISNA (dongu suruyor): {e}")

    # disk yonetimi (her dongu sonu)
    try:
        prune_old_files(ROOT / "out" / "gfs_cache", keep_days)
        if do_archive:
            for region in regions:
                prune_old_dirs(ROOT / "out" / "archive" / region, keep_days)
    except Exception as e:  # noqa
        log(f"UYARI disk budama hatasi: {e}")


def main():
    ap = argparse.ArgumentParser(description="WFE gozetimsiz operasyonel dongu")
    ap.add_argument("--regions", nargs="+", default=["turkey6km", "antalya"],
                    help="bolge (case) adlari; case = cases/<ad>.ini")
    ap.add_argument("--days", type=float, default=14.0, help="kac gun kossun")
    ap.add_argument("--hours", type=int, default=24, help="tahmin uzunlugu [saat]")
    ap.add_argument("--sleep-min", type=float, default=25.0,
                    help="donguler arasi uyku [dk]")
    ap.add_argument("--keep-days", type=int, default=3,
                    help="GFS onbellek + arsiv saklama [gun]")
    ap.add_argument("--state", default=str(ROOT / "out" / "operational_state.json"))
    ap.add_argument("--no-archive", action="store_true",
                    help="harita arsivlemeyi kapat")
    ap.add_argument("--once", action="store_true",
                    help="tek dongu adimi kos ve cik (test)")
    args = ap.parse_args()

    do_archive = not args.no_archive
    state = load_state(args.state)
    start = now_utc()
    deadline = None if args.once else start + dtm.timedelta(days=args.days)

    log("################################################################")
    log(f"WFE operasyonel dongu basladi | bolgeler={args.regions} "
        f"| +{args.hours}h | {'TEK DONGU' if args.once else f'{args.days:g} gun'}"
        f" | uyku {args.sleep_min:g} dk | saklama {args.keep_days} gun")
    log(f"Python: {PY}")

    n_iter = 0
    while True:
        n_iter += 1
        try:
            one_iteration(args.regions, args.hours, state, args.state,
                          args.keep_days, do_archive)
        except KeyboardInterrupt:
            log("kullanici durdurdu (KeyboardInterrupt); cikiliyor")
            break
        except Exception as e:  # noqa — ust duzey kalkan: dongu asla olmez
            log(f"BEKLENMEDIK ust-duzey hata (dongu suruyor): {e}")

        if args.once:
            log("tek dongu tamamlandi; cikiliyor")
            break
        if deadline and now_utc() >= deadline:
            log(f"{args.days:g} gun doldu; operasyonel dongu duruyor")
            break
        log(f"uyku {args.sleep_min:g} dk (dongu #{n_iter} bitti) ...")
        try:
            interruptible_sleep(args.sleep_min, deadline)
        except KeyboardInterrupt:
            log("uyku sirasinda durduruldu; cikiliyor")
            break

    log("operasyonel dongu sonlandi.")


if __name__ == "__main__":
    main()
