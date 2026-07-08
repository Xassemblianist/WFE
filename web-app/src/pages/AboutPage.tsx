import { useNavigate } from 'react-router-dom'
import { IcArrowRight, IcGithub } from '../components/icons'

interface VRow {
  test: string
  result: string
  ref: string
  win?: boolean
}

const VALIDATION: VRow[] = [
  {
    test: 'Gerçek tahmin + iç nudging (Türkiye, 24 sa)',
    result: 'GFS f024: rüzgâr becerisi kalıcılığı yener (u +%17); θ RMSE 2.36 K; METAR 10 m rüzgâr yanlılığı ~0',
    ref: 'GFS + METAR',
    win: true,
  },
  {
    test: 'Çok-döngülü doğrulama (3 döngü)',
    result: 'u her döngüde kalıcılığı yener (+%14 / +%18 / +%17 — tutarlı)',
    ref: 'GFS f024',
    win: true,
  },
  {
    test: 'Uzatılmış menzil (48 saat)',
    result: 'Stabil; GFS f048: u +%53, θ +%14, v +%9, nem +%33 — model belirgin değer katar',
    ref: 'GFS f048',
    win: true,
  },
  {
    test: 'Nonlocal PBL (sınır tabakası)',
    result: 'PBLH gündüz döngüsü fiziksel: kara gece 460 m → öğleden sonra 1405 m; nem becerisi −%50 → −%38',
    ref: 'Troen-Mahrt 1986',
  },
  {
    test: 'Straka yoğunluk akıntısı',
    result: 't=900 s: cephe 14.45 km, θ′min −9.0…−9.2 K, 3 Kelvin-Helmholtz rotoru',
    ref: 'Straka 1993',
  },
  {
    test: 'WK82 süperhücre',
    result: 'Fırtına bölünmesi (ayna-simetrik çift), w_max 40–48 m/s, çift yağış şeridi',
    ref: 'Weisman-Klemp 1982',
  },
  {
    test: 'Çekirdek u-v simetri testi',
    result: 'RMS(u−vᵀ)/RMS(u) = 2×10⁻⁶ (makine kesinliği) → dinamik çekirdek yön-tarafsız',
    ref: 'analitik (simetri)',
  },
  {
    test: 'Pozitif-tanımlı nem adveksiyonu',
    result: 'qv ≥ 0 makine kesinliğinde (0 negatif hücre) → sahte yağış yok',
    ref: 'analitik',
  },
]

export default function AboutPage() {
  const nav = useNavigate()
  return (
    <div className="page">
      <div className="container">
        <section className="hero" style={{ padding: '44px 40px' }}>
          <div className="kicker">
            <span className="pulse" />
            Açık kaynak · MIT
          </div>
          <h1 style={{ fontSize: 'clamp(26px,3.6vw,40px)' }}>
            WFE — <span className="grad">Weather Forecast Engine</span>
          </h1>
          <p>
            Sıfırdan yazılmış, GPU üzerinde çalışan bölgesel sayısal hava tahmin modeli. Hedef: WRF
            muadili, tam denetlenebilir, tek makinede operasyonel bir tahmin motoru.
          </p>
          <div className="hero-stats">
            <div className="hstat">
              <b>C++ / CUDA</b>
              <span>çekirdek, sıfırdan</span>
            </div>
            <div className="hstat">
              <b>Split-explicit</b>
              <span>sıkıştırılabilir Euler</span>
            </div>
            <div className="hstat">
              <b>GFS</b>
              <span>başlangıç + sınır koşulu</span>
            </div>
            <div className="hstat">
              <b>1× RTX</b>
              <span>tek GPU'da operasyonel</span>
            </div>
          </div>
        </section>

        <div className="prose">
          <h2>Model nedir?</h2>
          <p>
            WFE, sıkıştırılabilir Euler denklemlerini bölünmüş-açık (split-explicit) zaman
            integrasyonuyla çözen, araziyi izleyen (terrain-following) koordinatta çalışan bir
            atmosfer modelidir. Çekirdek C++/CUDA ile yazılmıştır ve tek bir RTX sınıfı GPU'da koşar.
            Gerçek tahminlerde başlangıç ve sınır koşulları GFS küresel modelinden alınır; iç bölge
            analiz-nudging ile gözlemlere yakın tutulur.
          </p>
          <p>
            İki operasyonel alan üretilir: tüm Türkiye (~6 km, gerçek yüksek çözünürlüklü arazi) ve
            Antalya körfezi + Toros (2.5 km). Fizik paketi yüzey katmanı, nonlocal PBL, levha toprak
            modeli, basit radyasyon ve tek-moment mikrofizik içerir.
          </p>

          <h2>Neden güvenilir?</h2>
          <p>
            Model hem idealize testlerle (bilinen analitik/literatür çözümleri) hem de gerçek
            tahminlerde GFS analizine ve METAR istasyon gözlemlerine karşı doğrulanmıştır.{' '}
            <span style={{ color: 'var(--good)', fontWeight: 650 }}>Yeşil</span> satırlar modelin basit
            kalıcılık (persistans) tahminini istatistiksel olarak yendiği durumlardır.
          </p>

          <div className="vtable-wrap">
            <table className="vtable">
              <thead>
                <tr>
                  <th>Test</th>
                  <th>Sonuç</th>
                  <th>Referans</th>
                </tr>
              </thead>
              <tbody>
                {VALIDATION.map((r, i) => (
                  <tr key={i}>
                    <td style={{ fontWeight: 620, color: 'var(--tx)' }}>
                      {r.win && (
                        <span className="badge-dot" style={{ background: 'var(--good)', marginRight: 7 }} />
                      )}
                      {r.test}
                    </td>
                    <td>{r.result}</td>
                    <td className="muted" style={{ whiteSpace: 'nowrap' }}>
                      {r.ref}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2>Haritadaki katmanlar</h2>
          <p>
            Katmanlar model ham çıktısından (float32) tarayıcıda renklendirilir ve tahmin saatleri
            arasında zamanda interpolasyonlanır. Rüzgâr katmanındaki akış animasyonu, modelin gerçek
            alt-seviye rüzgâr bileşenleriyle taşınan partiküllerdir. Yağış katmanı 3 saatlik birikimi,
            sıcaklık katmanı 2 m sıcaklığı, basınç katmanı deniz seviyesine indirgenmiş basıncı
            gösterir. Tahmin alanı, modelin Lambert konformal ızgarasının dört köşesine göre
            georeferanslanır.
          </p>

          <p style={{ marginTop: 24, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            <button className="btn primary" onClick={() => nav('/harita')}>
              Haritayı keşfet <IcArrowRight size={16} />
            </button>
            <a
              className="btn"
              href="https://github.com/Xassemblianist/WFE"
              target="_blank"
              rel="noreferrer"
              style={{ textDecoration: 'none' }}
            >
              <IcGithub size={16} /> Kaynak kod
            </a>
          </p>
        </div>
      </div>
    </div>
  )
}
