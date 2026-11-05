import type { CSSProperties, ReactNode } from "react";
import { TEX, type TexName } from "./tex";

export function Stack({
  children,
  gap = 16,
}: {
  children: ReactNode;
  gap?: number;
}) {
  return <div style={{ display: "flex", flexDirection: "column", gap }}>{children}</div>;
}

export function Row({
  children,
  gap = 8,
  wrap = false,
  align = "center",
}: {
  children: ReactNode;
  gap?: number;
  wrap?: boolean;
  align?: CSSProperties["alignItems"];
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        gap,
        flexWrap: wrap ? "wrap" : "nowrap",
        alignItems: align,
      }}
    >
      {children}
    </div>
  );
}

export function H1({ children }: { children: ReactNode }) {
  return <h1 className="h1">{children}</h1>;
}
export function H2({ children }: { children: ReactNode }) {
  return <h2 className="h2">{children}</h2>;
}
export function H3({ children }: { children: ReactNode }) {
  return <h3 className="h3">{children}</h3>;
}

export function Text({
  children,
  tone = "primary",
  size = "body",
  weight,
}: {
  children: ReactNode;
  tone?: "primary" | "secondary" | "tertiary";
  size?: "body" | "small";
  weight?: "medium";
}) {
  return (
    <p className={`text text-${tone} text-${size}${weight ? " text-medium" : ""}`}>
      {children}
    </p>
  );
}

export function Card({ children }: { children: ReactNode }) {
  return <section className="card">{children}</section>;
}
export function CardHeader({
  children,
  trailing,
}: {
  children: ReactNode;
  trailing?: ReactNode;
}) {
  return (
    <header className="card-header">
      <div>{children}</div>
      {trailing ? <div className="card-trailing">{trailing}</div> : null}
    </header>
  );
}
export function CardBody({ children }: { children: ReactNode }) {
  return <div className="card-body">{children}</div>;
}

export function Pill({
  children,
  active,
  onClick,
}: {
  children: ReactNode;
  active?: boolean;
  onClick?: () => void;
}) {
  return (
    <button type="button" className={`pill${active ? " pill-active" : ""}`} onClick={onClick}>
      {children}
    </button>
  );
}

export function Stat({
  value,
  label,
  tone,
}: {
  value: string;
  label: string;
  tone?: "danger";
}) {
  return (
    <div className={`stat${tone === "danger" ? " stat-eta" : ""}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

export function Callout({ title, children }: { title: string; children: ReactNode }) {
  return (
    <aside className="callout">
      <strong>{title}</strong>
      <div>{children}</div>
    </aside>
  );
}

export function Table({ headers, rows }: { headers: string[]; rows: string[][] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {headers.map((h) => (
              <th key={h}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Divider() {
  return <hr className="divider" />;
}

export function TeX({ name, label }: { name: TexName; label: string }) {
  return (
    <div
      className="tex"
      aria-label={label}
      title={label}
      dangerouslySetInnerHTML={{ __html: TEX[name] }}
    />
  );
}
