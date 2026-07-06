# Yönetici denklemler ve ayrıklaştırma

## Denklem seti (Faz 2: nemli, arazi-takip eden koordinat)

Tam sıkıştırılabilir, non-hidrostatik Euler denklemleri, Klemp–Wilhelmson (1978)
pertürbasyon formülasyonu. Taban durumu fiziksel z'de yatay-homojen ve hidrostatik
dengede: `θ = θ̄(z) + θ'`, `π = π̄(z) + π'` (π: Exner, `π = (p/p00)^(Rd/cp)`).

Gal-Chen (BTF) koordinatı: `ζ ∈ [0, zt]`, `z = h(x,y) + ζ (zt − h)/zt`,
Jacobian `J = ∂z/∂ζ = (zt−h)/zt` (kolonda sabit), `∂z/∂x|ζ = h_x (1 − ζ/zt)`.
Kontravariant dikey hız: `Ω = (w − u ∂z/∂x − v ∂z/∂y)/J`; prognostik w fizikseldir.

```
∂u/∂t = -ADV(u) - cp θ̄ [∂π'/∂x|ζ - (z_x/J) ∂π'/∂ζ] + f(v-v_b) - α(u-u_b)
∂v/∂t = -ADV(v) - cp θ̄ [∂π'/∂y|ζ - (z_y/J) ∂π'/∂ζ] - f(u-u_b) - α v
∂w/∂t = -ADV(w) - (cp θ̄ʷ/J) ∂π'/∂ζ + g θ'/θ̄ʷ - α w
∂θ'/∂t = -ADV(θ') - w dθ̄/dz - α θ'
∂π'/∂t = -(Rd π̄/(cv ρ̄ θ̄ J)) [∂x(ρ̄θ̄J u) + ∂y(ρ̄θ̄J v) + ∂ζ(ρ̄ʷθ̄ʷ(w - u z_x - v z_y))]
ADV(q) = (1/ρ̃)[∇·(MF q) - q ∇·MF],  ρ̃ = ρ̄J,  MF = (ρ̃u, ρ̃v, ρ̄ʷ(w - uz_x - vz_y))
```

α(ζ): üst Rayleigh sönümleme profili (sin², `rayleigh_zd` üstünde); f: f-plane Coriolis
(pertürbasyon formu). İsteğe bağlı sabit-K difüzyon idealize testler için.

**Nem (Faz 2):** qv, qc, qr tam alan prognostikleri (adveksiyon + Kessler
mikrofiziği). Kaldırma: `B = g[θ'/θ̄ + 0.61(qv−q̄v) − qc − qr]` (KW78 nemli form).
Hidrostatik taban ve PGF/süreklilik katsayıları sanal potansiyel sıcaklık
`θ̄v = θ̄(1+0.61 q̄v)` ile kurulur (q̄v−π̄ iteratif dengeleme). Mikrofizik zaman
adımı sonunda operatör bölmesiyle: sedimentasyon (kolon upwind, CFL alt-adımlı,
Vr=36.34(ρqr)^0.1364·√(ρ0/ρ)) → otokonversiyon/akresyon → yağmur buharlaşması →
doygunluk ayarlaması (Tetens, tek Newton adımı, gizli ısı → θ').

**Bilinçli yaklaşımlar:**
- π' adveksiyonu ihmal (KW78 standardı); π' denkleminde nem kaynak terimi yok.
- cp, Rd kuru hava değerleri (nem düzeltmesi θ̄v üzerinden).
- Nem adveksiyonu opsiyonel pozitif-tanımlı (`pd_moist`, Skamarock 2006): yüksek
  mertebe akılar hücre başına çıkan kütle mevcut kütleyi aşmayacak şekilde upwind
  oranıyla renormalize edilir → qv,qc,qr ≥ 0 garanti, kütle korunur (telescoping),
  sahte negatif su ve Gibbs aşımları kalkar. Kapalıyken negatifler mikrofizikte kırpılır.
- Pertürbasyon formu sayesinde durağan atmosfer arazi üstünde TAM korunur
  (schaer_rest testi: |w| = 0.0, makine kesinliğinde).

## Ayrıklaştırma

- **Grid:** Arakawa C; ζ'de gerilebilir seviyeler (`stretch=geometric`).
- **Adveksiyon:** 5. mertebe upwind arayüz değerleri (WS2002), kütle akılarıyla
  akı formu + advektif-tutarlılık düzeltmesi (sabit alan tam korunur; upwind
  sönümlemesi RK3 ile eşleşir, açık filtre gerekmez).
- **Zaman:** WS2002 RK3 + Klemp-Wilhelmson **split-explicit**: yavaş terimler
  (adveksiyon, Coriolis, difüzyon, Rayleigh) RK3 aşamalarında; hızlı terimler
  (PGF, kaldırma, süreklilik, stratifikasyon) akustik alt-adımlarda
  (`acoustic_ns`, aşama başına {1, ns/2, ns}). Yatayda forward-backward +
  diverjans sönümleme (π' ileri ağırlıklama, `acoustic_smdiv`); dikeyde
  off-centered implicit (`acoustic_beta`) w-π' tridiagonal çözücü (Thomas,
  kolon/thread) → dt yalnız adveksiyon/yerçekimi dalgası CFL'iyle sınırlı.
- **Sınırlar:** yanal periyodik veya açık (`bc_x/bc_y = open`): girişte taban
  durumuna sabitleme, çıkışta KW radyasyonu (faz hızı u±c*, her akustik
  alt-adımda); skalarlarda sıfır-gradyan ghost. Altta Ω=0 (w yüzeyde
  diagnostik: w = u z_x + v z_y), üstte rijit w=0 + Rayleigh katmanı.

## Doğrulama süiti (2026-07-04 çalıştırmaları)

| Test | Sonuç | Referans |
|---|---|---|
| Sıcak kabarcık (warm_bubble.ini) | mantar termali, w_max≈19.2 @ t≈830s; dt=1.5s (split-explicit) dt=0.25s (explicit) ile birebir | WK98 tipi |
| Straka yoğunluk akıntısı (straka.ini) | t=900s: cephe 14.45 km, θ'min=−9.0…−9.2K, 3 KH rotoru | Straka 1993: ~15 km, −8.9…−9.8K |
| Galilean değişmezlik (bubble_outflow.ini) | 20 m/s ortam akışında w evrimi durağan durumla aynı; sınırdan yansımasız çıkış | — |
| Durağanlık arazide (schaer_rest.ini) | 1 saat: |w|=0.000, θ'=0 (tam) | analitik |
| Schär dağ dalgası (schaer.ini) | küçük ölçek evanescent, büyük ölçek yukarı yayılan eğik faz; w_max≈1.5-1.9 m/s | Schär 2002 |
| WK82 süperhücre (wk82_supercell.ini) | fırtına bölünmesi (ayna-simetrik çift), w_max 40-48 m/s, çift yağış şeridi, stratosferik taşan tepeler | Weisman-Klemp 1982 |
| Gerçek tahmin (turkey.ini, GFS 2026-07-04 00Z) | 24h stabil; fizik v1 ile GFS f024'e karşı: θ +%15, u@8km +%23, u@0.9km −%32 (fiziksiz −%48), 3B u −%11 (fiziksiz −%19) | GFS analizi |
| Pozitif-tanımlı adveksiyon (moist_blob.ini) | qv≥0 makine kesinliğinde (0 negatif hücre); PD kapalı: 5265 negatif hücre, min −0.97 g/kg | analitik (pozitiflik) |
| Nonlocal PBL (turkey.ini, pbl=nonlocal) | PBLH gündüz döngüsü: kara gece 460m → öğleden sonra 1405m (iç bölge 3600m), deniz ~800m sabit; qv becerisi local'e karşı −%50→−%38 | Troen-Mahrt 1986 / Hong-Pan 1996 |
| Çekirdek u-v ayna simetrisi (warm_bubble) | x↔y simetrik kurulumda RMS(u−vᵀ)/RMS(u) = 2×10⁻⁶ (makine kesinliği) → dinamik çekirdek yön-tarafsız, yatay momentumda kod asimetrisi yok | analitik (simetri) |
| Gerçek tahmin + iç nudging (turkey.ini, nudge_tau=6h) | GFS f024: u +%17, qv +%1 persistansı yener, θ RMSE 2.36K; METAR 2m T yanlılık −2.5°C (yüks.düz.), 10m rüzgâr yanlılık ~0 | GFS + METAR |

**Gerçek tahmin notları (Faz 3):** PGF tam θv ile (linearizasyon kaldırıldı);
Coriolis tam rüzgâra, f(lat) 2D; harita faktörleri henüz yok (m=1, bölgesel
alanlarda ~%1-2); fizik yok (radyasyon/PBL/yüzey → Faz 4) — alt seviye rüzgâr
hatasının ana nedeni.
