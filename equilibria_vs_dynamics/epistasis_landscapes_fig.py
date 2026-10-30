"""Recreate the six canonical two-locus fitness-landscape sketches (A-F).

Each panel is a two-locus / two-allele system whose four genotypes sit at the
corners of a square:

        10 ---- 11
        |        |
        00 ---- 01

A single scale parameter ``a`` sets every height.  Fitness values (in units of
``a``) per panel:

    A Asymmetric additive : 00=0  10=a/2  01=3a/2  11=2a
    B Additive            : 00=0  10=a    01=a     11=2a
    C Synergy             : 00=0  10=a/2  01=a/2   11=2a
    D Sign epistasis      : 00=0  10=a    01=-a    11=2a
    E Reciprocal sign     : 00=0  10=-a   01=-a    11=2a
    F Neutral intermediate: 00=0  10=0    01=0     11=2a

Each corner is drawn as a pseudo-3D (oblique-projection) bar whose signed
height is its fitness and whose colour is a diverging blue->purple->pink->red
map of that fitness.  Arrows along the four edges point *uphill* (toward the
fitter genotype); flat edges get a plain connector.  The number of monotone
uphill routes 00 -> 11 is the "accessible" count shown under each panel.

Run:  python3 epistasis_landscapes_fig.py
"""
from __future__ import annotations

from fractions import Fraction

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb


# --------------------------------------------------------------------------- #
# Panel definitions.  Values are coefficients c such that fitness = c * a.
# --------------------------------------------------------------------------- #
PANELS = [
    ("A", "Asymmetric additive",
     {"00": 0.0, "10": 0.5, "01": 1.5, "11": 2.0},
     "one route steeper, no epistasis"),
    ("B", "Additive",
     {"00": 0.0, "10": 1.0, "01": 1.0, "11": 2.0},
     "no epistasis"),
    ("C", "Synergy",
     {"00": 0.0, "10": 0.5, "01": 0.5, "11": 2.0},
     "+ epistasis"),
    ("D", "Sign epistasis",
     {"00": 0.0, "10": 1.0, "01": -1.0, "11": 2.0},
     ""),
    ("E", "Neutral intermediates",
     {"00": 0.0, "10": 0.0, "01": 0.0, "11": 2.0},
     "neutral plateau"),
    ("F", "Reciprocal sign epistasis",
     {"00": 0.0, "10": -1.0, "01": -1.0, "11": 2.0},
     "fitness valley"),
]

# Screen (axes-fraction) position of the front-bottom-left foot of each bar.
POS = {
    "00": (0.16, 0.26),
    "01": (0.56, 0.26),
    "10": (0.28, 0.58),
    "11": (0.68, 0.58),
}
EDGES = [("00", "10"), ("00", "01"), ("10", "11"), ("01", "11")]

BW = 0.13          # bar width
ODX, ODY = 0.075, 0.065   # oblique depth offset
VSCALE = 0.135     # screen height per unit of a


# --------------------------------------------------------------------------- #
# Colour handling.  Fitness ranges over [-1, 2] (units of a); 0 is the neutral
# baseline, so the diverging palettes are centred on 0.  Each palette is a
# (colormap, norm) pair.  Pick one via the PALETTE global / build_figure().
# --------------------------------------------------------------------------- #
def _seg(stops):
    return LinearSegmentedColormap.from_list("epi", stops)


# Every positive fitness (0 .. 2a) is the SAME hue, just a darker/more saturated
# shade as fitness grows, so a/2 reads as a lighter version of a and of 2a.
# Deleterious fitness (negative) gets its own, clearly different hue.
#   pos      : sequential single-hue colormap for coef >= 0
#   pos_norm : maps coef in [~0, 2] onto that ramp (lo < 0 keeps 0 faintly tinted)
#   neg      : colormap for the magnitude |coef| of deleterious genotypes
#   neg_norm : maps |coef| onto the negative ramp
PALETTES = {
    # Bright red gradient (light -> vivid red) with blue for deleterious.
    "reds": dict(
        pos=_seg([(0.00, "#ffe6e1"), (0.35, "#ff9d90"),
                  (0.65, "#f24a3d"), (1.00, "#c3130b")]),
        pos_norm=Normalize(vmin=-0.25, vmax=2.0),
        neg=_seg([(0.0, "#cfe0f2"), (1.0, "#2f5f9e")]),
        neg_norm=Normalize(vmin=0.0, vmax=1.4),
    ),
    # Deep single-hue crimson gradient with a muted steel-blue for deleterious.
    "crimson": dict(
        pos=_seg([(0.00, "#fceceb"), (0.35, "#e79c98"),
                  (0.70, "#cf5148"), (1.00, "#8f1d1a")]),
        pos_norm=Normalize(vmin=-0.25, vmax=2.0),
        neg=_seg([(0.0, "#dbe4f0"), (1.0, "#28527a")]),
        neg_norm=Normalize(vmin=0.0, vmax=1.4),
    ),
    # Burgundy/wine gradient with teal for deleterious (print-friendly).
    "wine": dict(
        pos=_seg([(0.00, "#f6e8ea"), (0.40, "#d99aa2"),
                  (0.75, "#a83f4e"), (1.00, "#6e1327")]),
        pos_norm=Normalize(vmin=-0.25, vmax=2.0),
        neg=_seg([(0.0, "#dcecec"), (1.0, "#1b6d78")]),
        neg_norm=Normalize(vmin=0.0, vmax=1.4),
    ),
}

PALETTE = "reds"   # default; override in build_figure()


def value_color(coef: float, palette: str = None):
    p = PALETTES[palette or PALETTE]
    if coef < 0:
        return p["neg"](p["neg_norm"](abs(coef)))
    return p["pos"](p["pos_norm"](coef))


def shade(color, factor: float):
    r, g, b = to_rgb(color)
    return (min(1, r * factor), min(1, g * factor), min(1, b * factor))


# --------------------------------------------------------------------------- #
# Label formatting: turn a coefficient into a nice "a" expression.
# --------------------------------------------------------------------------- #
def coef_label(coef: float) -> str:
    if coef == 0:
        return "0"
    fr = Fraction(coef).limit_denominator(8)
    sign = "-" if fr < 0 else ""
    fr = abs(fr)
    num, den = fr.numerator, fr.denominator
    if den == 1:
        body = "a" if num == 1 else f"{num}a"
    else:
        body = f"a/{den}" if num == 1 else f"{num}a/{den}"
    return sign + body


# --------------------------------------------------------------------------- #
# Drawing primitives
# --------------------------------------------------------------------------- #
def draw_bar(ax, bx: float, by: float, height: float, base_color, z: float):
    """Oblique-projected box rising (or hanging) from foot (bx, by)."""
    top_c = shade(base_color, 1.22)
    front_c = base_color
    side_c = shade(base_color, 0.72)

    top_edge = by + height          # y of the far end (may be below by)
    cap_y = by + max(height, 0.0)   # y where the visible top cap sits

    right = [(bx + BW, by), (bx + BW + ODX, by + ODY),
             (bx + BW + ODX, top_edge + ODY), (bx + BW, top_edge)]
    front = [(bx, by), (bx + BW, by), (bx + BW, top_edge), (bx, top_edge)]
    cap = [(bx, cap_y), (bx + BW, cap_y),
           (bx + BW + ODX, cap_y + ODY), (bx + ODX, cap_y + ODY)]

    ax.add_patch(Polygon(right, closed=True, facecolor=side_c,
                         edgecolor="black", linewidth=0.8, zorder=z))
    ax.add_patch(Polygon(front, closed=True, facecolor=front_c,
                         edgecolor="black", linewidth=0.8, zorder=z + 0.1))
    ax.add_patch(Polygon(cap, closed=True, facecolor=top_c,
                         edgecolor="black", linewidth=0.8, zorder=z + 0.2))


def foot_center(g: str):
    bx, by = POS[g]
    return bx + BW / 2.0, by


def draw_edge(ax, p_lo, p_hi, c_lo: float, c_hi: float,
              shrinkA: float = 16, shrinkB: float = 30, zorder: float = 1):
    """Arrow between two anchor points, pointing to the fitter genotype.

    Equal fitness -> plain (headless) connector.
    """
    if c_lo == c_hi:
        ax.annotate("", xy=p_hi, xytext=p_lo,
                    arrowprops=dict(arrowstyle="-", color="0.4",
                                    lw=1.6, shrinkA=18, shrinkB=18),
                    zorder=zorder)
        return

    src, dst = (p_lo, p_hi) if c_hi > c_lo else (p_hi, p_lo)
    ax.annotate("", xy=dst, xytext=src,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0,
                                shrinkA=shrinkA, shrinkB=shrinkB,
                                mutation_scale=24),
                zorder=zorder)


# Per-panel arrow tweaks: (panel letter, edge) -> shrinkB in points.
# Panel A's tall 01 bar carries the "3a/2" label right in the path of the
# 10 -> 11 top arrow, so we shorten that arrow to keep its tip clear.
EDGE_SHRINK = {("A", ("10", "11")): 55}


def accessible_routes(coefs: dict) -> int:
    """Count strictly-uphill two-step routes 00 -> {01,10} -> 11."""
    n = 0
    for mid in ("10", "01"):
        if coefs["00"] < coefs[mid] < coefs["11"]:
            n += 1
    return n


def draw_zero_plane(ax):
    """Translucent oblique square through the four feet = the fitness-0 plane.

    Bars rise above it; deleterious genotypes (negative fitness) visibly dip
    below it, which is what disambiguates e.g. the back-row -a bar in panel E.
    """
    quad = [
        (POS["00"][0] - 0.02, POS["00"][1]),
        (POS["01"][0] + BW + 0.02, POS["01"][1]),
        (POS["11"][0] + BW + 0.02, POS["11"][1]),
        (POS["10"][0] - 0.02, POS["10"][1]),
    ]
    ax.add_patch(Polygon(quad, closed=True, facecolor="0.6", alpha=0.14,
                         edgecolor="0.55", linewidth=0.8,
                         linestyle=(0, (4, 3)), zorder=0))


def draw_fitness_axis(ax):
    """Vertical scale on the left making explicit that bar height = fitness.

    Zero is aligned with the front baseline of the oblique plane; ticks mark
    -a, 0, a, 2a on the same VSCALE used for every bar.
    """
    x = 0.07
    y0 = POS["00"][1]

    def yv(c):
        return y0 + c * VSCALE

    lo, hi = yv(-1) - 0.02, yv(2) + 0.045
    ax.annotate("", xy=(x, hi), xytext=(x, lo),
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.4,
                                mutation_scale=14), zorder=1)
    for c, lab in [(-1, "-a"), (0, "0"), (1, "a"), (2, "2a")]:
        y = yv(c)
        ax.plot([x - 0.014, x], [y, y], color="0.25", lw=1.2, zorder=1)
        ax.text(x - 0.02, y, lab, ha="right", va="center",
                fontsize=9, color="0.25", zorder=6)
    ax.text(x - 0.058, yv(0.6), "log fitness", rotation=90, ha="center",
            va="center", fontsize=10, color="0.25", fontweight="bold")


def draw_panel(ax, letter, title, coefs, note):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_zero_plane(ax)
    draw_fitness_axis(ax)

    # Arrows first so the bars occlude their lower parts (arrows sit in gaps).
    for g_lo, g_hi in EDGES:
        p_lo, p_hi = foot_center(g_lo), foot_center(g_hi)
        shrinkA, shrinkB, zorder = 16, 30, 1
        # Shift the 01->11 arrow to the right of the 01 bar (same direction /
        # angle for every panel, so top and bottom rows line up exactly) and
        # lift it on top so a tall 01 bar can't swallow it.
        if (g_lo, g_hi) == ("01", "11"):
            dx = 0.125
            p_lo = (p_lo[0] + dx, p_lo[1])
            p_hi = (p_hi[0] + dx, p_hi[1])
            shrinkA, shrinkB, zorder = 10, 14, 5
        shrinkB = EDGE_SHRINK.get((letter, (g_lo, g_hi)), shrinkB)
        draw_edge(ax, p_lo, p_hi, coefs[g_lo], coefs[g_hi],
                  shrinkA=shrinkA, shrinkB=shrinkB, zorder=zorder)

    # Bars: back row (10, 11) first, then front row (00, 01) on top.
    for g in ("10", "11", "00", "01"):
        bx, by = POS[g]
        coef = coefs[g]
        h = coef * VSCALE
        z = 2 + (0 if g in ("10", "11") else 2)
        draw_bar(ax, bx, by, h, value_color(coef), z)

        # Value label near the cap (above for positive, below for negative).
        cx = bx + BW / 2 + ODX / 2
        if h >= 0:
            ly, va = by + h + ODY + 0.015, "bottom"
        else:
            ly, va = by + h - 0.015, "top"
        ax.text(cx, ly, coef_label(coef), ha="center", va=va,
                fontsize=12, fontweight="bold",
                color=shade(value_color(coef), 0.75), zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.7))

    # Genotype labels around the base.
    lab_kw = dict(fontsize=11, color="0.2", zorder=6)
    ax.text(POS["00"][0] - 0.01, POS["00"][1] - 0.06, "00",
            ha="right", va="center", **lab_kw)
    ax.text(POS["01"][0] + BW + 0.16, POS["01"][1] - 0.03, "01",
            ha="left", va="center", **lab_kw)
    ax.text(POS["10"][0] - 0.05, POS["10"][1] + 0.02, "10",
            ha="right", va="center", **lab_kw)
    ax.text(POS["11"][0] + BW + 0.14, POS["11"][1] + 0.02, "11",
            ha="left", va="center", **lab_kw)

    # Panel letter (top-left) and title.
    ax.text(0.0, 1.05, letter, transform=ax.transAxes,
            fontsize=17, fontweight="bold", va="top", ha="left")
    ax.text(0.5, 1.05, title, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="center")

    # Accessibility caption.
    n = accessible_routes(coefs)
    caption = f"{n} accessible" if n else "0"
    if note:
        caption += f"\n({note})" if not n else f"\n({note})"
    ax.text(0.5, -0.02, caption, transform=ax.transAxes,
            fontsize=15, va="top", ha="center", color="0.15")


def build_figure(a: float = 1.0, palette: str = "science"):
    global PALETTE
    PALETTE = palette
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("white")
    for ax, (letter, title, coefs, note) in zip(axes.ravel(), PANELS):
        draw_panel(ax, letter, title, coefs, note)
    fig.suptitle("Two-locus fitness landscapes  (bar height = log fitness)",
                 fontsize=15, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.08,
                        wspace=0.10, hspace=0.35)
    return fig


if __name__ == "__main__":
    import sys

    # Render one palette if named on the CLI, else render them all.
    names = sys.argv[1:] or list(PALETTES)
    for name in names:
        fig = build_figure(a=1.0, palette=name)
        out = f"epistasis_landscapes_{name}.png"
        fig.savefig(out, dpi=200, facecolor="white")
        plt.close(fig)
        print(f"wrote {out}")
