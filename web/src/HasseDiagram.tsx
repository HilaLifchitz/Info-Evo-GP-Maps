import { useMemo } from "react";
import {
  SVG_W,
  bitstring,
  fitnessFill,
  fitnessW,
  kLabelStep,
  layoutHasse,
  luminance,
  traitFill,
  type ColorBy,
} from "./model";

const stroke = {
  primary: "var(--stroke)",
  tertiary: "var(--stroke-muted)",
};
const text = {
  secondary: "var(--muted)",
  tertiary: "var(--faint)",
};

export function HasseDiagram({
  n,
  colorBy = "trait",
  vs = 0.0625,
  zStar = 0.5,
}: {
  n: number;
  colorBy?: ColorBy;
  vs?: number;
  zStar?: number;
}) {
  const layout = useMemo(() => layoutHasse(n), [n]);
  const showLabels = n <= 4;
  const edgeOpacity = n <= 3 ? 0.5 : n <= 4 ? 0.38 : n <= 5 ? 0.22 : n <= 6 ? 0.14 : 0.1;
  const fontSize = n <= 2 ? 9 : n <= 3 ? 7.5 : 6;
  const labelEvery = kLabelStep(n);
  const showK = (k: number) => k === 0 || k === n || k % labelEvery === 0;
  const kTicks: number[] = [];
  for (let k = 0; k <= n; k++) if (showK(k)) kTicks.push(k);

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${SVG_W} ${layout.height}`}
      role="img"
      aria-label={
        colorBy === "fitness"
          ? `Boolean lattice for ${n} loci, colored by stabilizing fitness`
          : `Boolean lattice for ${n} loci, Hamming weight increasing left to right`
      }
    >
      {kTicks.map((k) => {
        const x = layout.left + k * layout.colGap;
        return (
          <g key={`lab-${k}`}>
            {layout.mode === "nodes" ? (
              <line
                x1={x}
                y1={layout.top}
                x2={x}
                y2={layout.top + layout.plotH}
                stroke={stroke.tertiary}
                strokeWidth={0.5}
                opacity={0.35}
              />
            ) : null}
            {showK(k) ? (
              <text
                x={x}
                y={layout.height - (n <= 12 ? 32 : 18)}
                textAnchor="middle"
                fontSize={n <= 12 ? 10 : 8}
                fill={text.secondary}
                fontFamily="inherit"
              >
                {n <= 12 ? `k = ${k}` : k}
              </text>
            ) : null}
            {n <= 12 && showK(k) ? (
              <text
                x={x}
                y={layout.height - 18}
                textAnchor="middle"
                fontSize={10}
                fill={text.secondary}
                fontFamily="inherit"
              >
                z = {k === 0 ? "0" : k === n ? "1" : `${k}/${n}`}
              </text>
            ) : null}
          </g>
        );
      })}

      {layout.mode === "layers" && layout.bars && n <= 40
        ? layout.bars.slice(0, -1).map((bar, i) => {
            const nxt = layout.bars[i + 1];
            return (
              <line
                key={`e-${i}`}
                x1={bar.x}
                y1={bar.y + bar.h / 2}
                x2={nxt.x}
                y2={nxt.y + nxt.h / 2}
                stroke={stroke.primary}
                strokeWidth={1}
                opacity={0.28}
              />
            );
          })
        : layout.mode === "nodes"
          ? layout.edges.map((e, i) => (
              <line
                key={i}
                x1={e.x1}
                y1={e.y1}
                x2={e.x2}
                y2={e.y2}
                stroke={stroke.primary}
                strokeWidth={n <= 4 ? 1.1 : 0.7}
                opacity={edgeOpacity}
              />
            ))
          : null}

      {colorBy === "fitness" ? (
        <line
          x1={layout.left + zStar * n * layout.colGap}
          y1={layout.top}
          x2={layout.left + zStar * n * layout.colGap}
          y2={layout.top + layout.plotH}
          stroke={stroke.primary}
          strokeWidth={1.1}
          strokeDasharray="4 3"
          opacity={0.75}
        />
      ) : null}

      {layout.mode === "layers" && layout.bars
        ? layout.bars.map((bar) => {
            const z = bar.k / n;
            const w = fitnessW(z, vs, zStar);
            const fill = colorBy === "fitness" ? fitnessFill(w) : traitFill(z);
            return (
              <rect
                key={`bar-${bar.k}`}
                x={bar.x - bar.w / 2}
                y={bar.y}
                width={bar.w}
                height={bar.h}
                rx={n > 40 ? 0 : 3}
                fill={fill}
                stroke={n > 40 ? fill : stroke.primary}
                strokeWidth={n > 40 ? 0 : 0.8}
              />
            );
          })
        : Array.from(layout.pos.entries()).map(([g, p]) => {
            const z = p.k / n;
            const w = fitnessW(z, vs, zStar);
            const fill = colorBy === "fitness" ? fitnessFill(w) : traitFill(z);
            const labelFill =
              colorBy === "fitness"
                ? luminance(fill) > 140
                  ? "black"
                  : "white"
                : z > 0.55
                  ? "white"
                  : "black";
            return (
              <g key={g}>
                <circle
                  cx={p.x}
                  cy={p.y}
                  r={layout.r}
                  fill={fill}
                  stroke={stroke.primary}
                  strokeWidth={n <= 4 ? 1.2 : 0.8}
                />
                {showLabels ? (
                  <text
                    x={p.x}
                    y={p.y + fontSize * 0.35}
                    textAnchor="middle"
                    fontSize={fontSize}
                    fontFamily="inherit"
                    fill={labelFill}
                  >
                    {bitstring(g, n)}
                  </text>
                ) : null}
              </g>
            );
          })}

      <text
        x={SVG_W / 2}
        y={layout.height - 4}
        textAnchor="middle"
        fontSize={10}
        fill={text.tertiary}
        fontFamily="inherit"
      >
        Hamming weight k →
      </text>
    </svg>
  );
}

export function ColorBar() {
  const steps = 48;
  const w = 160;
  const h = 10;
  return (
    <svg width={w + 36} height={28} aria-label="Trait color scale from 0 to 1">
      {Array.from({ length: steps }, (_, i) => (
        <rect
          key={i}
          x={(i * w) / steps}
          y={8}
          width={w / steps + 0.4}
          height={h}
          fill={traitFill(i / (steps - 1))}
        />
      ))}
      <text x={0} y={7} fontSize={9} fill={text.secondary} fontFamily="inherit">
        z = 0
      </text>
      <text x={w} y={7} fontSize={9} fill={text.secondary} fontFamily="inherit" textAnchor="end">
        z = 1
      </text>
    </svg>
  );
}

export function FitnessColorBar() {
  const steps = 48;
  const w = 160;
  const h = 10;
  return (
    <svg width={w + 48} height={28} aria-label="Fitness color scale from low to high">
      {Array.from({ length: steps }, (_, i) => (
        <rect
          key={i}
          x={(i * w) / steps}
          y={8}
          width={w / steps + 0.4}
          height={h}
          fill={fitnessFill(i / (steps - 1))}
        />
      ))}
      <text x={0} y={7} fontSize={9} fill={text.secondary} fontFamily="inherit">
        low W
      </text>
      <text x={w} y={7} fontSize={9} fill={text.secondary} fontFamily="inherit" textAnchor="end">
        high W
      </text>
    </svg>
  );
}

export function FitnessProfile({ vs, zStar }: { vs: number; zStar: number }) {
  const Wsvg = 720;
  const H = 78;
  const left = 40;
  const right = 16;
  const top = 10;
  const bottom = 22;
  const innerW = Wsvg - left - right;
  const innerH = H - top - bottom;
  const nPts = 96;
  const pts: string[] = [];
  for (let i = 0; i <= nPts; i++) {
    const z = i / nPts;
    const w = fitnessW(z, vs, zStar);
    const x = left + z * innerW;
    const y = top + (1 - w) * innerH;
    pts.push(`${x},${y}`);
  }
  const xStar = left + zStar * innerW;
  return (
    <svg
      width="100%"
      viewBox={`0 0 ${Wsvg} ${H}`}
      role="img"
      aria-label="Stabilizing fitness W as a function of trait z"
    >
      <text x={4} y={top + 4} fontSize={9} fill={text.tertiary} fontFamily="inherit">
        W
      </text>
      <line x1={left} y1={top} x2={left} y2={top + innerH} stroke={stroke.tertiary} strokeWidth={0.8} />
      <line
        x1={left}
        y1={top + innerH}
        x2={left + innerW}
        y2={top + innerH}
        stroke={stroke.tertiary}
        strokeWidth={0.8}
      />
      <polyline points={pts.join(" ")} fill="none" stroke="var(--fg)" strokeWidth={1.6} />
      <line
        x1={xStar}
        y1={top}
        x2={xStar}
        y2={top + innerH}
        stroke={stroke.primary}
        strokeWidth={1}
        strokeDasharray="3 3"
        opacity={0.7}
      />
      <text x={xStar} y={top + innerH + 14} textAnchor="middle" fontSize={10} fill={text.secondary} fontFamily="inherit">
        z* = {zStar.toFixed(2)}
      </text>
      <text x={left} y={H - 2} fontSize={9} fill={text.tertiary} fontFamily="inherit">
        z = 0
      </text>
      <text x={left + innerW} y={H - 2} textAnchor="end" fontSize={9} fill={text.tertiary} fontFamily="inherit">
        z = 1
      </text>
    </svg>
  );
}

function peakOf(y: number[]): number {
  let m = 1e-12;
  for (let i = 0; i < y.length; i++) if (y[i] > m) m = y[i];
  return m;
}

function scaleToPeak(y: number[], peak: number): number[] {
  const out = new Array<number>(y.length);
  for (let i = 0; i < y.length; i++) out[i] = y[i] / peak;
  return out;
}

function fmtPeak(x: number): string {
  return x >= 0.01 ? x.toFixed(3) : x.toExponential(2);
}

function binTrait(
  n: number,
  phiZ: number[],
  psiZ: number[],
  maxBins: number,
): { z: number[]; phi: number[]; psi: number[] } {
  const m = n + 1;
  if (m <= maxBins) {
    const z = new Array<number>(m);
    const phi = new Array<number>(m);
    const psi = new Array<number>(m);
    for (let k = 0; k < m; k++) {
      z[k] = k / n;
      phi[k] = phiZ[k] * n;
      psi[k] = psiZ[k] * n;
    }
    return { z, phi, psi };
  }
  const z: number[] = [];
  const phi: number[] = [];
  const psi: number[] = [];
  for (let i = 0; i < maxBins; i++) {
    const k0 = Math.floor((i * m) / maxBins);
    const k1 = Math.floor(((i + 1) * m) / maxBins);
    const dk = k1 - k0;
    if (dk <= 0) continue;
    let p = 0;
    let q = 0;
    for (let k = k0; k < k1; k++) {
      p += phiZ[k];
      q += psiZ[k];
    }
    z.push((k0 + k1 - 1) / (2 * n));
    phi.push((p * n) / dk);
    psi.push((q * n) / dk);
  }
  return { z, phi, psi };
}

function polylinePts(
  z: number[],
  y: number[],
  left: number,
  top: number,
  innerW: number,
  innerH: number,
  ymax: number,
): string {
  const pts: string[] = [];
  for (let i = 0; i < z.length; i++) {
    const x = left + z[i] * innerW;
    const yy = top + (1 - y[i] / ymax) * innerH;
    pts.push(`${x},${yy}`);
  }
  return pts.join(" ");
}

export function TraitDistPlot({
  n,
  zStar,
  phiZ,
  psiZ,
}: {
  n: number;
  zStar: number;
  phiZ: number[];
  psiZ: number[];
}) {
  const bins = binTrait(n, phiZ, psiZ, 64);
  const phiPeak = peakOf(phiZ);
  const psiPeak = peakOf(psiZ);
  const phiShape = scaleToPeak(bins.phi, peakOf(bins.phi));
  const psiShape = scaleToPeak(bins.psi, peakOf(bins.psi));
  const Wsvg = 720;
  const H = 178;
  const left = 40;
  const right = 16;
  const top = 26;
  const bottom = 28;
  const innerW = Wsvg - left - right;
  const innerH = H - top - bottom;
  const xStar = left + zStar * innerW;
  const phiPts = polylinePts(bins.z, phiShape, left, top, innerW, innerH, 1);
  const psiPts = polylinePts(bins.z, psiShape, left, top, innerW, innerH, 1);
  const phiFill = `${left},${top + innerH} ${phiPts} ${left + innerW},${top + innerH}`;
  const psiFill = `${left},${top + innerH} ${psiPts} ${left + innerW},${top + innerH}`;
  return (
    <svg
      width="100%"
      viewBox={`0 0 ${Wsvg} ${H}`}
      role="img"
      aria-label="Trait-distribution shapes, each curve scaled to its own peak"
    >
      <text x={left} y={12} fontSize={11} fill={text.secondary} fontFamily="inherit">
        Shape of P(Z); each curve scaled to its own peak
      </text>
      <line
        x1={left + innerW - 300}
        y1={8}
        x2={left + innerW - 276}
        y2={8}
        stroke={text.secondary}
        strokeWidth={2}
        strokeDasharray="5 4"
      />
      <text x={left + innerW - 272} y={12} fontSize={10} fill={text.secondary} fontFamily="inherit">
        φ^Z peak height {fmtPeak(phiPeak)}
      </text>
      <line
        x1={left + innerW - 138}
        y1={8}
        x2={left + innerW - 114}
        y2={8}
        stroke="var(--accent)"
        strokeWidth={2}
      />
      <text x={left + innerW - 110} y={12} fontSize={10} fill={text.secondary} fontFamily="inherit">
        ψ^Z peak height {fmtPeak(psiPeak)}
      </text>
      <polygon points={psiFill} fill="var(--accent)" opacity={0.2} />
      <polygon points={phiFill} fill={text.tertiary} opacity={0.12} />
      <polyline points={psiPts} fill="none" stroke="var(--accent)" strokeWidth={2} />
      <polyline
        points={phiPts}
        fill="none"
        stroke={text.secondary}
        strokeWidth={2}
        strokeDasharray="5 4"
      />
      {psiPeak / phiPeak < 1.08 && phiPeak / psiPeak < 1.08 ? (
        <text
          x={left + innerW / 2}
          y={top + 16}
          textAnchor="middle"
          fontSize={10}
          fill={text.secondary}
          fontFamily="inherit"
        >
          ψ^Z ≈ φ^Z — selection is not moving the trait
        </text>
      ) : null}
      <line x1={left} y1={top} x2={left} y2={top + innerH} stroke={stroke.tertiary} strokeWidth={0.8} />
      <line
        x1={left}
        y1={top + innerH}
        x2={left + innerW}
        y2={top + innerH}
        stroke={stroke.tertiary}
        strokeWidth={0.8}
      />
      <line
        x1={xStar}
        y1={top}
        x2={xStar}
        y2={top + innerH}
        stroke={stroke.primary}
        strokeWidth={1}
        strokeDasharray="3 3"
        opacity={0.7}
      />
      <text x={4} y={top + 8} fontSize={9} fill={text.tertiary} fontFamily="inherit">
        own peak
      </text>
      <text
        x={xStar}
        y={top + innerH + 14}
        textAnchor="middle"
        fontSize={10}
        fill={text.secondary}
        fontFamily="inherit"
      >
        z* = {zStar.toFixed(2)}
      </text>
      <text x={left} y={H - 2} fontSize={9} fill={text.tertiary} fontFamily="inherit">
        z = 0
      </text>
      <text x={left + innerW} y={H - 2} textAnchor="end" fontSize={9} fill={text.tertiary} fontFamily="inherit">
        z = 1
      </text>
    </svg>
  );
}
