// WebGL alan boyacısı — motorun kalbi.
//
// 16-bit R/G paketli veri-PNG'ler doğrudan doku olarak yüklenir; çözme,
// iki kare arası zaman harmanı, lapse-rate arazi düzeltmesi, renk LUT'u ve
// dithering tamamı fragment shader'da yapılır. Kod çözme (v = lo + (256R+G)k)
// kanal başına LİNEER olduğundan GPU'nun bilineer doku filtresi değerin
// bilineer interpolasyonuna denktir — CPU'suz pürüzsüz büyütme.
//
// Sıcaklıkta detaylandırma: T(px) = T_model(bilineer) + Γ·(z_model(bilineer) − z_hires(px))
// (meteoblue tarzı fiziksel downscaling; Γ = 6.5 K/km, z_hires gerçek DEM).
import type { LUT } from './lut'

const VS = `
attribute vec2 aPos;
varying vec2 vUV;
void main() {
  vUV = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`

const FS = `
precision highp float;
varying vec2 vUV;
uniform sampler2D uT0, uT1, uLUT, uZH, uZM;
uniform vec2 uGrid;
uniform float uFrac, uDLo, uDHi, uLLo, uLHi;
uniform float uEnhance, uLapse, uELo, uEHi;

float dec01(sampler2D t, vec2 uv) {
  vec4 c = texture2D(t, uv);
  return (c.r * 65280.0 + c.g * 255.0) / 65535.0;
}
float dec(sampler2D t, vec2 uv, float lo, float hi) {
  return lo + dec01(t, uv) * (hi - lo);
}

// Catmull-Rom agirligi (a=-0.5) — bilineer elmas artefaktlarini giderir
float cr(float x) {
  x = abs(x);
  if (x <= 1.0) return 1.5*x*x*x - 2.5*x*x + 1.0;
  if (x <  2.0) return -0.5*(x*x*x - 5.0*x*x + 8.0*x - 4.0);
  return 0.0;
}

// 16-tap bicubic ornekleme (alan dokusu icin; yakinlastirmada purussuz)
float bicubic01(sampler2D t, vec2 uv) {
  vec2 p  = uv * uGrid - 0.5;
  vec2 ip = floor(p);
  vec2 f  = p - ip;
  float v = 0.0;
  float ws = 0.0;
  for (int j = -1; j <= 2; j++) {
    float wy = cr(f.y - float(j));
    for (int i = -1; i <= 2; i++) {
      float w = wy * cr(f.x - float(i));
      vec2 suv = (ip + vec2(float(i), float(j)) + 0.5) / uGrid;
      v  += w * dec01(t, suv);
      ws += w;
    }
  }
  return v / ws;
}

void main() {
  float v01 = bicubic01(uT0, vUV);
  if (uFrac > 0.001) v01 = mix(v01, bicubic01(uT1, vUV), uFrac);
  float v = uDLo + v01 * (uDHi - uDLo);
  if (uEnhance > 0.5) {
    float zm = dec(uZM, vUV, uELo, uEHi);
    float zh = dec(uZH, vUV, uELo, uEHi);
    v += uLapse * (zm - zh);
  }
  float x = clamp((v - uLLo) / (uLHi - uLLo), 0.0, 1.0);
  // hafif gurultu dither — bantlasmayi kirar
  float d = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  x += (d - 0.5) * (2.0 / 1024.0);
  vec4 col = texture2D(uLUT, vec2(clamp(x, 0.0, 1.0), 0.5));
  gl_FragColor = vec4(col.rgb * col.a, col.a);  // premultiplied
}`

type TexKey = string

export class GLPainter {
  readonly canvas: HTMLCanvasElement
  readonly ok: boolean
  private gl: WebGLRenderingContext | null = null
  private prog: WebGLProgram | null = null
  private uni: Record<string, WebGLUniformLocation | null> = {}
  private texCache = new Map<TexKey, WebGLTexture>()
  private lutTex: WebGLTexture | null = null
  private lutKey = ''

  constructor() {
    this.canvas = document.createElement('canvas')
    const gl = this.canvas.getContext('webgl', {
      alpha: true,
      premultipliedAlpha: true,
      preserveDrawingBuffer: true, // maplibre canvas-source buffer'ı okur
      antialias: false,
      depth: false,
      stencil: false,
    })
    if (!gl) {
      this.ok = false
      return
    }
    this.gl = gl
    const vs = gl.createShader(gl.VERTEX_SHADER)!
    gl.shaderSource(vs, VS)
    gl.compileShader(vs)
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!
    gl.shaderSource(fs, FS)
    gl.compileShader(fs)
    const prog = gl.createProgram()!
    gl.attachShader(prog, vs)
    gl.attachShader(prog, fs)
    gl.linkProgram(prog)
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error('GLPainter link:', gl.getProgramInfoLog(prog))
      this.ok = false
      return
    }
    this.prog = prog
    gl.useProgram(prog)
    const buf = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buf)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW)
    const loc = gl.getAttribLocation(prog, 'aPos')
    gl.enableVertexAttribArray(loc)
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0)
    for (const n of ['uT0', 'uT1', 'uLUT', 'uZH', 'uZM', 'uGrid', 'uFrac', 'uDLo', 'uDHi', 'uLLo', 'uLHi', 'uEnhance', 'uLapse', 'uELo', 'uEHi']) {
      this.uni[n] = gl.getUniformLocation(prog, n)
    }
    this.ok = true
  }

  /** Görüntüyü doku olarak yükle/önbellekten getir. flipY: satır-0 kuzey → GL alt-satır. */
  texture(key: TexKey, img: HTMLImageElement): WebGLTexture {
    const gl = this.gl!
    let t = this.texCache.get(key)
    if (t) return t
    t = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, t)
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1)
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 0)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    this.texCache.set(key, t)
    return t
  }

  /** Bölge/alan değişiminde kare dokularını bırak (arazi/LUT kalır). */
  dropFrames(prefix: string) {
    const gl = this.gl!
    for (const [k, t] of this.texCache) {
      if (k.startsWith(prefix)) {
        gl.deleteTexture(t)
        this.texCache.delete(k)
      }
    }
  }

  setLUT(lut: LUT, key: string) {
    if (key === this.lutKey) return
    const gl = this.gl!
    if (!this.lutTex) this.lutTex = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, this.lutTex)
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1024, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array(lut.rgba.buffer.slice(0)))
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    this.lutKey = key
  }

  paint(opts: {
    tex0: WebGLTexture
    tex1: WebGLTexture | null
    frac: number
    outW: number
    outH: number
    gridW: number
    gridH: number
    dataRange: [number, number]
    lutRange: [number, number]
    enhance: { zh: WebGLTexture; zm: WebGLTexture; elevRange: [number, number]; lapse: number } | null
  }) {
    const gl = this.gl!
    if (this.canvas.width !== opts.outW || this.canvas.height !== opts.outH) {
      this.canvas.width = opts.outW
      this.canvas.height = opts.outH
    }
    gl.viewport(0, 0, opts.outW, opts.outH)
    gl.useProgram(this.prog)
    const bind = (unit: number, tex: WebGLTexture | null, uname: string) => {
      gl.activeTexture(gl.TEXTURE0 + unit)
      gl.bindTexture(gl.TEXTURE_2D, tex)
      gl.uniform1i(this.uni[uname], unit)
    }
    bind(0, opts.tex0, 'uT0')
    bind(1, opts.tex1 ?? opts.tex0, 'uT1')
    bind(2, this.lutTex, 'uLUT')
    const e = opts.enhance
    bind(3, e ? e.zh : opts.tex0, 'uZH')
    bind(4, e ? e.zm : opts.tex0, 'uZM')
    gl.uniform2f(this.uni.uGrid, opts.gridW, opts.gridH)
    gl.uniform1f(this.uni.uFrac, opts.tex1 ? opts.frac : 0)
    gl.uniform1f(this.uni.uDLo, opts.dataRange[0])
    gl.uniform1f(this.uni.uDHi, opts.dataRange[1])
    gl.uniform1f(this.uni.uLLo, opts.lutRange[0])
    gl.uniform1f(this.uni.uLHi, opts.lutRange[1])
    gl.uniform1f(this.uni.uEnhance, e ? 1 : 0)
    gl.uniform1f(this.uni.uLapse, e ? e.lapse : 0)
    gl.uniform1f(this.uni.uELo, e ? e.elevRange[0] : 0)
    gl.uniform1f(this.uni.uEHi, e ? e.elevRange[1] : 1)
    gl.drawArrays(gl.TRIANGLES, 0, 3)
  }

  destroy() {
    const gl = this.gl
    if (!gl) return
    for (const t of this.texCache.values()) gl.deleteTexture(t)
    this.texCache.clear()
    if (this.lutTex) gl.deleteTexture(this.lutTex)
  }
}
