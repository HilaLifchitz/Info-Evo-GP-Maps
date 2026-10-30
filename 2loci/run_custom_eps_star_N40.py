#!/usr/bin/env python3
"""Run one-off eps_star gallery cases (N=40) with custom u,v,r."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from log_Z_eps_star_gallery import (  # noqa: E402
    GalleryCase,
    _fmt_float,
    _run_case,
    predict_eps_star,
)
from log_Z_param_sweep import Scenario  # noqa: E402

OUT = SCRIPT_DIR / "plots" / "f2b" / "eps_star_gallery_N40"
N = 40
S1, S2 = 1.0e-3, -5.0e-4

VARIANTS: tuple[tuple[str, tuple[float, float], tuple[float, float], float], ...] = (
    ("mix_pp_sm_uv_x_r1e-1", (0.005, 0.001), (0.001, 0.005), 0.1),
    ("mix_pp_sm_vu_x", (0.001, 0.005), (0.005, 0.001), 0.0),
    ("mix_pp_sm_u5e3", (0.005, 0.005), (0.005, 0.005), 0.0),
)


def _make_case(tag: str, u: tuple[float, float], v: tuple[float, float], r: float) -> GalleryCase:
    label = f"N{N}_{tag}_s1{_fmt_float(S1)}_s2{_fmt_float(S2)}"
    title = f"mix pp sm {tag}  ($N={N}$, $s_1={S1:g}$, $s_2={S2:g}$)"
    sc = Scenario(label, title, N, S1, S2, u, v, r)
    eps_star = predict_eps_star(N, S1, S2, u, v, r)
    return GalleryCase(label, title, sc, float(eps_star))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (tag, u, v, r) in enumerate(VARIANTS, 1):
        case = _make_case(tag, u, v, r)
        print(f"\n[{i}/{len(VARIANTS)}] {case.label}  u={u} v={v} r={r}  eps*={case.eps_star:+.2e}", flush=True)
        _run_case(case, OUT, step=5e-4, pergen_max_t=500, reuse=False, minimal_plots=False)
    print(f"\nDone — {len(VARIANTS)} cases in {OUT}/", flush=True)


if __name__ == "__main__":
    main()
