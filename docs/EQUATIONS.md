# Yönetici denklemler ve ayrıklaştırma

## Denklem seti (Faz 0: kuru, düz zemin)

Tam sıkıştırılabilir, non-hidrostatik Euler denklemleri, Klemp–Wilhelmson (1978)
pertürbasyon formülasyonu. Taban durumu yatay-homojen ve hidrostatik dengede:
`θ = θ̄(z) + θ'`, `π = π̄(z) + π'` (π: Exner fonksiyonu, `π = (p/p00)^(Rd/cp)`).

```
∂u/∂t = -ADV(u) - cp θ̄ ∂π'/∂x
∂v/∂t = -ADV(v) - cp θ̄ ∂π'/∂y
∂w/∂t = -ADV(w) - cp θ̄ʷ ∂π'/∂z + g θ'/θ̄ʷ
∂θ'/∂t = -ADV(θ') - w dθ̄/dz
∂π'/∂t = -(Rd π̄ / (cv ρ̄ θ̄)) ∇·(ρ̄ θ̄ V)
```

**Bilinçli yaklaşımlar (Faz 0):**
- π' adveksiyonu ihmal edilir (KW78 standardı; akustik enerji açısından önemsiz).
- Kaldırma kuvvetinde nem/yoğunluk sıcaklığı yok (kuru model; Faz 2'de θ_ρ gelecek).
- Coriolis, karışım (turbülans kapanımı) ve tüm fizik parametrizasyonları henüz yok.

## Ayrıklaştırma

- **Grid:** Arakawa C — skalarlar hücre merkezinde, u/v/w yüzlerde. Düzgün aralıklı
  kartezyen (Faz 1'de arazi-takip eden dikey koordinat + harita projeksiyonu).
- **Adveksiyon:** 5. mertebe upwind-eğilimli arayüz değerleri (Wicker & Skamarock 2002),
  akı formu + advektif-tutarlılık düzeltmesi:
  `tend = -(1/ρ̄)[∇·(ρ̄ V q) - q ∇·(ρ̄ V)]`
  Böylece sabit alan tam korunur; upwind terimi RK3 ile birlikte içsel sönümleme sağlar,
  ayrıca açık filtre gerekmez.
- **Zaman:** WS2002 RK3 (aşamalar dt/3, dt/2, dt). Şu an **tamamen explicit** — zaman
  adımı ses dalgası CFL'iyle sınırlı: `dt ≲ 0.5 dx/c ≈ 0.5·dx/350`. Faz 1'de
  split-explicit akustik alt-adımlama (yatay explicit, dikey implicit) gelecek;
  o zaman dt yaklaşık 6-8 kat büyüyecek.
- **Sınırlar:** x/y periyodik; alt/üst rijit, serbest-kayma (w=0, diğerleri sıfır-gradyan
  ghost, w ghost'ları tek-simetrik). Faz 1'de açık/radyasyon yanal sınırlar.

## Doğrulama: sıcak kabarcık (cases/warm_bubble.ini)

20×20×10 km alan, Δ=200 m, θ0=300 K izentropik taban, 2 K cos² kabarcık (r=2 km, z=2 km).
2026-07-03 çalıştırması: t=800 s'de mantar termali, tepe ~8 km, w_max≈19.2 m/s,
θ'_max 2K→1.2K sönümleme, |π'|~4·10⁻⁴, 1000 s boyunca stabil, NaN yok.
Referans davranış: Wicker & Skamarock (1998), Bryan & Fritsch (2002) 3B kuru termaller.
