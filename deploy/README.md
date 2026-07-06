# WFE — Google Cloud dağıtımı (spot GPU, maliyet-verimli)

Mimari: **hesaplama kısa, servis ucuz.**

```
Cloud Scheduler (4×/gün, GFS döngülerinden ~4h sonra)
   → spot GPU VM açılır → WFE koşusu (~dakikalar) → ürünler GCS'e → VM kendini siler
   → Cloud Run / küçük VM (GPU'suz) API + web'i GCS'ten servis eder (hep açık, ucuz)
```

Maliyet tahmini (T4 spot ~$0.11/saat): koşu ~10-20 dk × 4/gün ≈ **ayda ~$5-10**.
$500 kredi ile ~yıllarca. Serving (Cloud Run) neredeyse ücretsiz.

## Ön koşullar (senin yapman gerekenler)

```bash
gcloud auth login
gcloud config set project <PROJE_ID>
gcloud services enable compute.googleapis.com run.googleapis.com \
    cloudscheduler.googleapis.com
gsutil mb -l europe-west4 gs://<BUCKET>          # urun deposu (Antalya'ya yakin bolge)
gsutil iam ch allUsers:objectViewer gs://<BUCKET> # web icin herkese-okuma (istege bagli)
```

`deploy/config.env` dosyasına kendi değerlerini yaz (PROJE_ID, BUCKET, ZONE, GPU).

## 1) Tahmin koşusu (spot GPU)

`launch_forecast.sh` bir spot GPU VM'i başlatır; `vm_startup.sh` içinde repoyu çeker,
derler, koşar, ürünleri GCS'e yükler ve VM'i siler.

```bash
bash deploy/launch_forecast.sh turkey 24     # bolge, saat
```

## 2) Zamanlama (Cloud Scheduler)

`schedule.sh` GFS döngülerine (00/06/12/18Z + ~4h) göre 4 tetikleyici kurar.
Her tetikleyici bir spot koşusu başlatan küçük bir Cloud Function/PubSub'a bağlanır.

## 3) Servis (API + web)

`serve/` — GCS'ten ürünleri okuyan hafif Cloud Run servisi (server/app.py, GPU'suz).
Web arayüzü (web/) aynı servisten. Deploy: `bash deploy/deploy_serve.sh`.

> NOT: Bu scriptler senin GCP hesabınla test edilmelidir (bende GCP erişimi yok).
> Placeholder'ları (`<...>`, config.env) doldur; ilk koşuda logları izle.
