export const N_MIN = 2;
export const N_MAX = 1000;
export const FULL_HASSE_N = 7;
export const SVG_W = 720;
export const SLIDER_STEPS = 1000;
export const N_JUMPS = [
  2, 4, 6, 7, 8, 10, 12, 16, 20, 50, 80, 100, 200, 500, 800, 1000,
];

export type ColorBy = "trait" | "fitness";

export type NodePos = { x: number; y: number; k: number };
export type Edge = { x1: number; y1: number; x2: number; y2: number };
export type LayerBar = { k: number; x: number; y: number; w: number; h: number; count: number };

export type HasseLayout =
  | {
      mode: "nodes";
      pos: Map<number, NodePos>;
      edges: Edge[];
      r: number;
      colGap: number;
      left: number;
      top: number;
      plotH: number;
      height: number;
      counts: number[];
      bars?: undefined;
    }
  | {
      mode: "layers";
      pos: Map<number, NodePos>;
      edges: Edge[];
      r: number;
      colGap: number;
      left: number;
      top: number;
      plotH: number;
      height: number;
      counts: number[];
      bars: LayerBar[];
    };

export function formatCount(x: number): string {
  if (!Number.isFinite(x)) return "∞";
  if (x >= 1e9) {
    const [mant, exp] = x.toExponential(2).split("e");
    return `${mant}×10^${Number(exp)}`;
  }
  if (x >= 10_000) return Math.round(x).toLocaleString("en-US");
  return String(Math.round(x));
}

export function formatPow2(n: number): string {
  if (n <= 20) return (2 ** n).toLocaleString("en-US");
  return `2^${n}`;
}

export function formatEdges(n: number): string {
  if (n <= 20) return formatCount(n * 2 ** (n - 1));
  return `${n}×2^${n - 1}`;
}

export function nToSlider(n: number): number {
  const t = Math.log(n / N_MIN) / Math.log(N_MAX / N_MIN);
  return Math.round(SLIDER_STEPS * Math.min(1, Math.max(0, t)));
}

export function sliderToN(t: number): number {
  return clampN(N_MIN * (N_MAX / N_MIN) ** (Number(t) / SLIDER_STEPS));
}

export function kLabelStep(n: number): number {
  if (n <= 12) return 1;
  if (n <= 20) return 2;
  if (n <= 50) return 5;
  if (n <= 100) return 10;
  if (n <= 200) return 25;
  if (n <= 500) return 50;
  return 100;
}

export function clampN(value: number): number {
  const n = Math.round(Number(value));
  if (!Number.isFinite(n)) return 4;
  return Math.min(N_MAX, Math.max(N_MIN, n));
}

export function popcount(g: number): number {
  let w = 0;
  let x = g >>> 0;
  while (x) {
    w += x & 1;
    x >>>= 1;
  }
  return w;
}

export function bitstring(g: number, n: number): string {
  return g.toString(2).padStart(n, "0");
}

export function binomial(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  k = Math.min(k, n - k);
  let v = 1;
  for (let i = 1; i <= k; i++) v *= (n - k + i) / i;
  return v;
}

export function binomialLayerCounts(n: number): number[] {
  const c = new Array<number>(n + 1);
  c[0] = 1;
  for (let k = 1; k <= n; k++) c[k] = (c[k - 1] * (n - k + 1)) / k;
  return c;
}

export function traitFill(z: number): string {
  const v = Math.round(255 * (1 - z));
  return `rgb(${v}, ${v}, ${v})`;
}

export function clamp01(value: number): number {
  const x = Number(value);
  if (!Number.isFinite(x)) return 0.5;
  return Math.min(1, Math.max(0, x));
}

export const VS_MIN = 0.001;
export const VS_MAX = 1;
export const VS_DEFAULT = 0.0625; // s = 1/(2 Vs) = 8
export const VS_SLIDER_STEPS = 1000;

export function clampVs(value: number): number {
  const x = Number(value);
  if (!Number.isFinite(x)) return VS_DEFAULT;
  return Math.min(VS_MAX, Math.max(VS_MIN, x));
}

export function vsToSlider(vs: number): number {
  const t = Math.log(vs / VS_MIN) / Math.log(VS_MAX / VS_MIN);
  return Math.round(VS_SLIDER_STEPS * Math.min(1, Math.max(0, t)));
}

export function sliderToVs(t: number): number {
  return clampVs(VS_MIN * (VS_MAX / VS_MIN) ** (Number(t) / VS_SLIDER_STEPS));
}

export function sFromVs(vs: number): number {
  return 1 / (2 * vs);
}

export const BOLTZMANN_N_JUMPS = [
  2, 3, 4, 5, 6, 7, 8, 10, 16, 20, 50, 100, 200, 500, 1000,
];
export const NPOP_MIN = 5;
export const NPOP_MAX = 200;
export const NPOP_DEFAULT = 50;
export const ALPHA_MIN = 0.2;
export const ALPHA_MAX = 10;
export const ALPHA_DEFAULT = 2;
export const ALPHA_SLIDER_STEPS = 400;
export const ALPHA0_MIN = 1;
export const ALPHA0_MAX = 65536;
export const ALPHA0_DEFAULT = 32;
export const ALPHA0_SLIDER_STEPS = 400;
export const ALPHA0_JUMPS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

export function clampNpop(value: number): number {
  const x = Number(value);
  if (!Number.isFinite(x)) return NPOP_DEFAULT;
  return Math.min(NPOP_MAX, Math.max(NPOP_MIN, Math.round(x)));
}

export function clampAlpha(value: number): number {
  const x = Number(value);
  if (!Number.isFinite(x)) return ALPHA_DEFAULT;
  return Math.min(ALPHA_MAX, Math.max(ALPHA_MIN, x));
}

export function alphaToSlider(a: number): number {
  const t = Math.log(a / ALPHA_MIN) / Math.log(ALPHA_MAX / ALPHA_MIN);
  return Math.round(ALPHA_SLIDER_STEPS * Math.min(1, Math.max(0, t)));
}

export function sliderToAlpha(t: number): number {
  return clampAlpha(
    ALPHA_MIN * (ALPHA_MAX / ALPHA_MIN) ** (Number(t) / ALPHA_SLIDER_STEPS),
  );
}

export function clampAlpha0(value: number): number {
  const x = Number(value);
  if (!Number.isFinite(x)) return ALPHA0_DEFAULT;
  return Math.min(ALPHA0_MAX, Math.max(ALPHA0_MIN, x));
}

export function alpha0ToSlider(a: number): number {
  const t = Math.log(a / ALPHA0_MIN) / Math.log(ALPHA0_MAX / ALPHA0_MIN);
  return Math.round(ALPHA0_SLIDER_STEPS * Math.min(1, Math.max(0, t)));
}

export function sliderToAlpha0(t: number): number {
  return clampAlpha0(
    ALPHA0_MIN * (ALPHA0_MAX / ALPHA0_MIN) ** (Number(t) / ALPHA0_SLIDER_STEPS),
  );
}

export function logAlphaFromAlpha0(alpha0: number, n: number): number {
  return Math.log(alpha0) - n * Math.LN2;
}

export function formatSci(x: number, digits = 2): string {
  if (!Number.isFinite(x)) return "—";
  if (x === 0) return "0";
  const ax = Math.abs(x);
  if (ax >= 1 && ax < 10000) return x.toFixed(ax >= 100 ? 0 : 2);
  if (ax >= 0.01 && ax < 1) return x.toFixed(3);
  return x.toExponential(digits);
}

export function impliedAlpha0(alpha: number, n: number): number {
  const l = Math.log(alpha) + n * Math.LN2;
  return l > 700 ? Infinity : Math.exp(l);
}

export function impliedAlpha(alpha0: number, n: number): number {
  return Math.exp(logAlphaFromAlpha0(alpha0, n));
}

export const SIMPSON_MIN = 0.05;
export const SIMPSON_MAX = 0.9;
export const SIMPSON_DEFAULT = 0.3;
export const SIMPSON_SLIDER_STEPS = 400;
export const SIMPSON_JUMPS = [0.26, 0.3, 0.4, 0.5, 0.6, 0.75];
export const DEMO_N_JUMPS = [4, 8, 16, 32, 64, 128, 256];

export type DemoScaling = "c" | "alpha0";
export type DemoRegime = {
  id: string;
  label: string;
  hint: string;
  scaling: DemoScaling;
  n: number;
  npop: number;
  vs: number;
  zStar: number;
  c?: number;
  alpha0?: number;
  sweep: { n: number; eta: number }[];
};

/** Offline Hamming-layer Boltzmann; even n. For the talk, pick one and step n. */
export const DEMO_REGIMES: DemoRegime[] = [
  {
    id: "default-c",
    label: "default c=0.3",
    hint: "E[S]=c held. η ≈ 7 → 19 → 36 from n=4 to 16 to 256, then slows.",
    scaling: "c",
    n: 4,
    npop: 50,
    vs: 0.0625,
    zStar: 0.5,
    c: 0.3,
    sweep: [
      { n: 4, eta: 7.25 },
      { n: 8, eta: 14.6 },
      { n: 16, eta: 18.8 },
      { n: 32, eta: 22.8 },
      { n: 64, eta: 27.5 },
      { n: 128, eta: 32.9 },
      { n: 256, eta: 36.1 },
      { n: 1000, eta: 38.2 },
    ],
  },
  {
    id: "poster",
    label: "steep: c=0.75, N=100",
    hint: "Best live demo. η ≈ 27 → 103 → 375 from n=4 to 16 to 256.",
    scaling: "c",
    n: 4,
    npop: 100,
    vs: 0.0625,
    zStar: 0.5,
    c: 0.75,
    sweep: [
      { n: 4, eta: 26.7 },
      { n: 8, eta: 57.5 },
      { n: 16, eta: 103 },
      { n: 32, eta: 164 },
      { n: 64, eta: 206 },
      { n: 128, eta: 274 },
      { n: 256, eta: 375 },
      { n: 1000, eta: 597 },
    ],
  },
  {
    id: "strong",
    label: "stronger selection",
    hint: "c=0.3, Vs=0.02. Rise continues through n=1000.",
    scaling: "c",
    n: 4,
    npop: 50,
    vs: 0.02,
    zStar: 0.5,
    c: 0.3,
    sweep: [
      { n: 4, eta: 4.04 },
      { n: 8, eta: 8.68 },
      { n: 16, eta: 12.2 },
      { n: 32, eta: 17.9 },
      { n: 64, eta: 21.9 },
      { n: 128, eta: 24.7 },
      { n: 256, eta: 31.2 },
      { n: 1000, eta: 38.4 },
    ],
  },
  {
    id: "alpha0-4",
    label: "α₀=4 fixed",
    hint: "Other same-neutral: hold Kα. Even n. η ≈ 9.6 → 22.9 from n=4 to 256.",
    scaling: "alpha0",
    n: 4,
    npop: 50,
    vs: 0.0625,
    zStar: 0.5,
    alpha0: 4,
    sweep: [
      { n: 4, eta: 9.63 },
      { n: 8, eta: 14.4 },
      { n: 16, eta: 11.4 },
      { n: 32, eta: 15.7 },
      { n: 64, eta: 18.5 },
      { n: 128, eta: 18.9 },
      { n: 256, eta: 22.9 },
    ],
  },
];

export function demoMatches(
  r: DemoRegime,
  npop: number,
  vs: number,
  zStar: number,
  simpsonC: number,
  alpha0: number,
): boolean {
  if (r.npop !== npop) return false;
  if (Math.abs(r.vs - vs) > 1e-6) return false;
  if (Math.abs(r.zStar - zStar) > 1e-6) return false;
  if (r.scaling === "c") return Math.abs((r.c ?? 0) - simpsonC) < 0.005;
  return Math.abs((r.alpha0 ?? 0) - alpha0) < 1e-9;
}

export function clampSimpson(value: number): number {
  const x = Number(value);
  if (!Number.isFinite(x)) return SIMPSON_DEFAULT;
  return Math.min(SIMPSON_MAX, Math.max(SIMPSON_MIN, x));
}

export function simpsonToSlider(c: number): number {
  return Math.round(
    SIMPSON_SLIDER_STEPS * ((c - SIMPSON_MIN) / (SIMPSON_MAX - SIMPSON_MIN)),
  );
}

export function sliderToSimpson(t: number): number {
  return clampSimpson(
    SIMPSON_MIN + (Number(t) / SIMPSON_SLIDER_STEPS) * (SIMPSON_MAX - SIMPSON_MIN),
  );
}

export function simpsonFloor(n: number): number {
  if (n >= 50) return 0;
  return 2 ** -n;
}

export function expectedSimpson(alpha: number, n: number, logAlpha?: number): number {
  const la =
    logAlpha !== undefined && Number.isFinite(logAlpha)
      ? logAlpha
      : Math.log(alpha);
  if (!Number.isFinite(la)) return NaN;
  const alpha0 = Math.exp(la + n * Math.LN2);
  const a = Math.exp(la);
  if (!Number.isFinite(alpha0)) return 0;
  if (!(a > 0) || !Number.isFinite(a)) return 1 / (alpha0 + 1);
  return (a + 1) / (alpha0 + 1);
}

export function alphaFromSimpson(
  c: number,
  n: number,
): { ok: true; logAlpha: number; alpha: number; alpha0: number } | { ok: false } {
  const floor = simpsonFloor(n);
  if (!(c > floor && c < 1)) return { ok: false };
  if (n >= 50) {
    const logAlpha = Math.log(1 - c) - Math.log(c) - n * Math.LN2;
    return { ok: true, logAlpha, alpha: Math.exp(logAlpha), alpha0: (1 - c) / c };
  }
  const K = 2 ** n;
  const den = c * K - 1;
  if (den <= 0) return { ok: false };
  const a = (1 - c) / den;
  if (!(a > 0) || !Number.isFinite(a)) return { ok: false };
  return { ok: true, logAlpha: Math.log(a), alpha: a, alpha0: K * a };
}

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng: () => number): number {
  const u = Math.max(rng(), 1e-12);
  const v = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function randGamma(a: number, rng: () => number): number {
  if (a < 1) {
    const u = Math.max(rng(), 1e-12);
    return randGamma(a + 1, rng) * u ** (1 / a);
  }
  const d = a - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x = 0;
    let v = 0;
    do {
      x = randn(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function logBinoms(n: number): number[] {
  const L = new Array<number>(n + 1);
  L[0] = 0;
  for (let k = 1; k <= n; k++) {
    L[k] = L[k - 1] + Math.log(n - k + 1) - Math.log(k);
  }
  return L;
}

function omegaLayers(n: number, vs: number, zStar: number): number[] {
  const w = new Array<number>(n + 1);
  for (let k = 0; k <= n; k++) {
    const d = k / n - zStar;
    w[k] = Math.exp(-(d * d) / (2 * vs));
  }
  return w;
}

function sampleLayerDirichlet(
  n: number,
  logAlpha: number,
  logB: number[],
  rng: () => number,
): number[] {
  const m = n + 1;
  const logg = new Array<number>(m);
  for (let k = 0; k < m; k++) {
    const logA = logAlpha + logB[k];
    if (logA > 700) {
      logg[k] = logA;
    } else if (logA > 20) {
      const ak = Math.exp(logA);
      logg[k] = Math.log(Math.max(ak + Math.sqrt(ak) * randn(rng), 1e-300));
    } else if (logA < -40) {
      logg[k] = -1e9;
    } else {
      logg[k] = Math.log(Math.max(randGamma(Math.exp(logA), rng), 1e-300));
    }
  }
  let maxL = -Infinity;
  for (let k = 0; k < m; k++) if (logg[k] > maxL) maxL = logg[k];
  let s = 0;
  const x = new Array<number>(m);
  for (let k = 0; k < m; k++) {
    x[k] = Math.exp(logg[k] - maxL);
    s += x[k];
  }
  const inv = 1 / s;
  for (let k = 0; k < m; k++) x[k] *= inv;
  return x;
}

export type BoltzResult = {
  D: number;
  C: number;
  eta: number;
  ess: number;
  samples: number;
  meanLogW: number;
  meanW: number;
  F: number;
};

/** Hamming-layer Dirichlet Boltzmann. Exact for W, C, D, F, η when ω=ω(k). */
export function boltzmannMaint(
  n: number,
  vs: number,
  zStar: number,
  Npop: number,
  logAlpha: number,
  seedTag: number,
): BoltzResult | null {
  if (n < 1 || n > N_MAX) return null;
  const layers = n + 1;
  const omega = omegaLayers(n, vs, zStar);
  const logB = logBinoms(n);
  const samples = n <= 20 ? 4000 : 2500;
  const rng = mulberry32(
    (n * 10007 +
      Math.round(vs * 1e6) +
      Math.round(zStar * 1e4) +
      Npop * 13 +
      Math.round(logAlpha * 1e6) +
      seedTag * 99991) >>>
      0,
  );
  const logTilt = new Array<number>(samples);
  const meanFit = new Array<number>(samples);
  const C = new Array<number>(samples);
  let maxLt = -Infinity;
  for (let s = 0; s < samples; s++) {
    const x = sampleLayerDirichlet(n, logAlpha, logB, rng);
    let W = 0;
    for (let k = 0; k < layers; k++) W += x[k] * omega[k];
    W = Math.max(W, 1e-300);
    meanFit[s] = W;
    const lt = 2 * Npop * Math.log(W);
    logTilt[s] = lt;
    if (lt > maxLt) maxLt = lt;
    let cost = 0;
    for (let k = 0; k < layers; k++) {
      const r = Math.max(omega[k] / W, 1e-300);
      cost += x[k] * r * Math.log2(r);
    }
    C[s] = cost;
  }
  let sumExp = 0;
  for (let s = 0; s < samples; s++) {
    sumExp += Math.exp(logTilt[s] - maxLt);
  }
  let essInv = 0;
  let eLogTilt = 0;
  let eC = 0;
  let eW = 0;
  for (let s = 0; s < samples; s++) {
    const ps = Math.exp(logTilt[s] - maxLt) / sumExp;
    essInv += ps * ps;
    eLogTilt += ps * logTilt[s];
    eC += ps * C[s];
    eW += ps * meanFit[s];
  }
  const logZ = maxLt + Math.log(sumExp / samples);
  const twoN = 2 * Npop;
  const Dbits = (eLogTilt - logZ) / Math.LN2;
  const meanLogW = eLogTilt / twoN;
  const F = logZ / twoN;
  const eta = eC > 1e-18 ? Dbits / (twoN * eC) : NaN;
  return {
    D: Dbits,
    C: eC,
    eta,
    ess: 1 / essInv,
    samples,
    meanLogW,
    meanW: eW,
    F,
  };
}

export function fitnessW(z: number, vs: number, zStar: number): number {
  const d = z - zStar;
  return Math.exp(-(d * d) / (2 * vs));
}

export function fitnessFill(w: number): string {
  const t = Math.max(0, Math.min(1, w));
  const r = Math.round(196 * (1 - t) + 42 * t);
  const g = Math.round(48 * (1 - t) + 150 * t);
  const b = Math.round(48 * (1 - t) + 74 * t);
  return `rgb(${r}, ${g}, ${b})`;
}

export function luminance(fill: string): number {
  const m = fill.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (!m) return 128;
  return 0.299 * Number(m[1]) + 0.587 * Number(m[2]) + 0.114 * Number(m[3]);
}

export function layoutHasse(n: number): HasseLayout {
  const left = 28;
  const right = 28;
  const top = 18;
  const bottom = n <= 12 ? 56 : 40;
  const plotW = SVG_W - left - right;
  const colGap = plotW / n;

  if (n <= FULL_HASSE_N) {
    const layers: number[][] = Array.from({ length: n + 1 }, () => []);
    const nGen = 1 << n;
    for (let g = 0; g < nGen; g++) layers[popcount(g)].push(g);

    const maxLayer = Math.max(...layers.map((L) => L.length));
    const r =
      n <= 2 ? 16 : n <= 3 ? 13 : n <= 4 ? 9 : n <= 5 ? 6.2 : n <= 6 ? 4.6 : 3.5;
    const spacing =
      n <= 2 ? 52 : n <= 3 ? 42 : n <= 4 ? 28 : n <= 5 ? 18 : n <= 6 ? 13 : 10.2;
    const plotH = Math.max(spacing * maxLayer, 96);
    const height = top + plotH + bottom;

    const pos = new Map<number, NodePos>();
    for (let k = 0; k <= n; k++) {
      const L = layers[k];
      const x = left + k * colGap;
      const m = L.length;
      for (let i = 0; i < m; i++) {
        const y = top + plotH / 2 + (i - (m - 1) / 2) * spacing;
        pos.set(L[i], { x, y, k });
      }
    }

    const edges: Edge[] = [];
    for (let g = 0; g < nGen; g++) {
      const a = pos.get(g);
      if (!a) continue;
      for (let b = 0; b < n; b++) {
        if ((g & (1 << b)) === 0) {
          const h = g | (1 << b);
          const c = pos.get(h);
          if (!c) continue;
          edges.push({ x1: a.x, y1: a.y, x2: c.x, y2: c.y });
        }
      }
    }

    return {
      mode: "nodes",
      pos,
      edges,
      r,
      colGap,
      left,
      top,
      plotH,
      height,
      counts: layers.map((L) => L.length),
    };
  }

  const counts = binomialLayerCounts(n);
  const maxC = counts.reduce((m, x) => (Number.isFinite(x) && x > m ? x : m), 1);
  const maxBar = 210;
  const minBar = n >= 80 ? 4 : 12;
  const plotH = maxBar + 20;
  const height = top + plotH + bottom;
  const gapless = n > 40;
  const barW = gapless ? Math.max(colGap, 0.75) : Math.min(24, colGap * 0.7);
  const bars = counts.map((count, k) => {
    const ratio = Number.isFinite(count) ? Math.min(1, count / maxC) : 1;
    const h = minBar + (maxBar - minBar) * Math.sqrt(ratio);
    const x = left + k * colGap;
    const y = top + plotH / 2 - h / 2;
    return { k, x, y, w: barW, h, count };
  });

  return {
    mode: "layers",
    pos: new Map(),
    edges: [],
    r: 0,
    colGap,
    left,
    top,
    plotH,
    height,
    counts,
    bars,
  };
}
