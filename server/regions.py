"""WFE tahmin bolgeleri (domain) kaydi. Her bolge bir case ini'ye baglanir."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REGIONS = {
    "turkey": {
        "case": "cases/turkey.ini",
        "title": "Türkiye (12 km)",
        "desc": "Tüm Türkiye ve çevresi, GFS-güdümlü bölgesel model.",
        "default_hours": 24,
    },
    "antalya": {
        "case": "cases/antalya.ini",
        "title": "Antalya (2.5 km)",
        "desc": "Antalya körfezi + Toros, yüksek çözünürlük, gerçek arazi.",
        "default_hours": 24,
    },
    "antalya1km": {
        "case": "cases/antalya1km.ini",
        "title": "Antalya (1 km)",
        "desc": "Konveksiyon-çözücü iç alan.",
        "default_hours": 24,
    },
}


def read_ini(path):
    kv = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].split(";")[0]
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def region_cfg(name):
    r = REGIONS.get(name)
    if not r:
        return None
    return read_ini(ROOT / r["case"])
