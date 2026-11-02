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

export const VS_MIN = 0.02;
export const VS_MAX = 1;
export const VS_DEFAULT = 0.0625; // s = 1/(2 Vs) = 8

export function clampVs(value: number): number {
  const x = Number(value);
  if (!Number.isFinite(x)) return VS_DEFAULT;
  return Math.min(VS_MAX, Math.max(VS_MIN, x));
}

export function sFromVs(vs: number): number {
  return 1 / (2 * vs);
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
