// WFE web arayüzü — API'den bölgeler, haritalar ve nokta tahminini çeker.
const API = location.origin;   // API ile aynı sunucudan servis edilirse
let manifest = null;

async function j(url) { const r = await fetch(url); if (!r.ok) throw new Error(r.status); return r.json(); }

async function init() {
  const regions = await j(`${API}/regions`);
  const rsel = document.getElementById('region');
  regions.forEach(r => { const o = document.createElement('option'); o.value = r.id; o.textContent = r.title; rsel.appendChild(o); });
  rsel.onchange = loadRegion;
  document.getElementById('step').onchange = showMap;
  await loadRegion();
}

async function loadRegion() {
  const region = document.getElementById('region').value;
  manifest = await j(`${API}/products/${region}`);
  document.getElementById('init').textContent = manifest.init ? `başlangıç: ${manifest.init.slice(0,16)}Z` : 'koşu yok';
  const ssel = document.getElementById('step'); ssel.innerHTML = '';
  (manifest.maps || []).forEach(name => {
    const step = parseInt(name.match(/_(\d+)\.png/)[1]);
    const fh = manifest.steps.find(s => s.step === step);
    const o = document.createElement('option'); o.value = name;
    o.textContent = fh ? `t+${fh.fhour}s` : name; ssel.appendChild(o);
  });
  showMap();
}

function showMap() {
  const region = document.getElementById('region').value;
  const name = document.getElementById('step').value;
  if (name) document.getElementById('map').src = `${API}/products/${region}/map/${name}`;
}

async function loadPoint() {
  const region = document.getElementById('region').value;
  const lat = document.getElementById('lat').value, lon = document.getElementById('lon').value;
  let d;
  try { d = await j(`${API}/point/${region}?lat=${lat}&lon=${lon}`); }
  catch { document.getElementById('ptinfo').textContent = 'nokta alınamadı'; return; }
  document.getElementById('ptinfo').textContent =
    `grid ${d.grid.grid_lat},${d.grid.grid_lon} · yükseklik ${d.grid.elev_m} m`;
  const rows = d.series.filter(s => s.t2m_C !== null);
  const tb = document.getElementById('pttable');
  tb.innerHTML = '<tr><th>saat</th><th>°C</th><th>rüzgâr m/s</th><th>yağış mm</th></tr>' +
    rows.map(s => `<tr><td>${s.valid.slice(11,16)}</td><td>${s.t2m_C}</td><td>${s.wind10_ms}</td><td>${s.precip_mm}</td></tr>`).join('');
  drawChart(rows.map(s => s.t2m_C));
}

function drawChart(vals) {
  const c = document.getElementById('chart'), ctx = c.getContext('2d');
  const W = c.width = c.clientWidth, H = c.height = 120;
  ctx.clearRect(0, 0, W, H);
  if (!vals.length) return;
  const mn = Math.min(...vals) - 1, mx = Math.max(...vals) + 1;
  ctx.strokeStyle = '#3fa7ff'; ctx.lineWidth = 2; ctx.beginPath();
  vals.forEach((v, i) => { const x = i / (vals.length - 1) * (W - 10) + 5;
    const y = H - 20 - (v - mn) / (mx - mn) * (H - 30);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  ctx.fillStyle = '#8aa0b4'; ctx.font = '11px sans-serif';
  ctx.fillText(`${mx.toFixed(0)}°`, 2, 12); ctx.fillText(`${mn.toFixed(0)}°`, 2, H - 5);
}

init();
