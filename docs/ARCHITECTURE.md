# Kod mimarisi

## Temel ilkeler

- **Tek bellek şeması:** Tüm alanlar (staggered dahil) aynı tampon boyutunu ve
  `GDims::idx(i,j,k)` indekslemesini paylaşır: i-en-hızlı (coalesced erişim),
  her yönde `ng=3` ghost, z'de w için +1 seviye. Staggered değişkenler sadece
  geçerli aralıklarıyla ayrışır (bkz. src/core/grid.hpp üstündeki yorum).
  Bu, kernel'lerde offset aritmetiğini tekilleştirir; bellek israfı ihmal edilir.
- **Hassasiyet politikası:** `real = float` (WFE_DOUBLE ile double). Turing'de FP64
  1/32 hızında; operasyonel NWP'de FP32 standarttır (ECMWF IFS). Kernel'lerde çıplak
  double sabiti yazılmaz.
- **Taban durumu ayrımı:** Hidrostatik dengedeki ρ̄, θ̄, π̄ profilleri 1B dizi olarak
  kernel'lere gider (`DevProf`); prognostik alanlar sapmadır. Bu FP32'de dinamik
  aralığı korur (tam basınç yerine ~10⁻⁴'lük π' taşınır).
- **Ghost doldurma sırası:** dikey → x periyodik (tüm j,k) → y periyodik (x ghost'ları
  dahil tüm i). Köşe tutarlılığı bu sıraya bağlıdır; değiştirme.

## Veri akışı (bir zaman adımı)

```
Integrator::step(dt)                          [src/dynamics/integrator.cpp]
  3 RK3 aşaması için:
    apply_bcs(cur)                            ghost'lar güncellenir
    compute_divergence(cur) -> div            ρ̄V diverjansı (merkezlerde)
    compute_tendencies(cur, div) -> tend      5 kernel: u,v,w,thp,pip
    update_state: s_stage = s_n + dt_rk*tend  tend ghost'ları hep 0 => ghost'lar s_n'den kopyalanır
  s_n <-> s_stage (pointer swap)
```

Tendency kernel'leri şimdilik "naif" (her thread kendi 6 akısını üst üste hesaplar,
shared memory yok). RTX 2060'ta 500k hücre 4000 adım / 9.3 s — optimizasyon Faz 6'nın
işi; önce doğruluk ve kapsam.

## Yeni prognostik değişken ekleme kontrol listesi

1. `State`'e Field3D alanı (src/dynamics/state.hpp) — alloc/copy_from/swap'a da ekle
2. `DevState`'e pointer (kernels.cu)
3. Tendency kernel'i + `compute_tendencies`'e launch
4. `update_state`'e çağrı
5. BC: periyodik listeye ekle + uygun dikey BC kerneli
6. `Writer::write`'a satır + meta.json vars listesi
7. Görselleştirme: tools/plot_slice.py otomatik çalışır (meta.json'dan okur)

## Faz 1'de değişecekler (bilinçli borç)

- Split-explicit akustik alt-adımlama: `compute_tendencies` yavaş/hızlı terimlere
  ayrılacak, dikey implicit tridiagonal çözücü (thread=kolon) eklenecek.
- Arazi-takip eden koordinat: `GDims`e metrik terimler (dz/dx haritaları), taban
  durumu 3B'leşecek (sadece dikey profil olmaktan çıkar).
- Açık yanal sınırlar: apply_bcs'e Klemp-Lilly radyasyon BC varyantı.
