"""
Cache for neutral (Sigma=0) stationary distribution phi on the simplex grid.

One .npz file holds phi vectors for several N values at a fixed (u, v, r).
Filename is derived from (u, v, r):

  cache/neutral_u0_1em3_u1_1em3_v0_2em3_v1_2em3_r0.npz

Build:
  python build_neutral_cache.py --config neutral_cache_config.yaml

Runtime (simplex.py) loads phi_N{N} when u, v, r and N match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CACHE_VERSION = 1


def _rate_tag(x: float) -> str:
    """Float → filesystem-safe scientific notation (1e-4 → 1em4)."""
    x = float(x)
    if x == 0.0:
        return "0"
    mant_str, exp_str = f"{x:.15e}".split("e")
    mant = float(mant_str)
    exp = int(exp_str)
    if abs(mant - round(mant)) < 1e-9:
        mant_part = str(int(round(mant)))
    else:
        mant_part = f"{mant:g}".replace(".", "p")
    if exp == 0:
        return mant_part
    exp_part = f"em{abs(exp)}" if exp < 0 else f"ep{exp}"
    return f"{mant_part}{exp_part}"


def cache_basename(u: np.ndarray, v: np.ndarray, r: float) -> str:
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    if u.size != 2 or v.size != 2:
        raise ValueError("u and v must have length 2")
    return (
        f"neutral_u0_{_rate_tag(u[0])}_u1_{_rate_tag(u[1])}_"
        f"v0_{_rate_tag(v[0])}_v1_{_rate_tag(v[1])}_"
        f"r{_rate_tag(r)}.npz"
    )


def cache_path_from_params(u: np.ndarray, v: np.ndarray, r: float, cache_dir: Path) -> Path:
    return cache_dir / cache_basename(u, v, r)


def _phi_key(N: int) -> str:
    return f"phi_N{int(N)}"


def _legacy_pi_key(N: int) -> str:
    """Old cache key (still read for backward compatibility)."""
    return f"pi_N{int(N)}"


def params_signature(u: np.ndarray, v: np.ndarray, r: float) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "u": [float(x) for x in np.asarray(u, dtype=float).ravel()],
        "v": [float(x) for x in np.asarray(v, dtype=float).ravel()],
        "r": float(r),
        "basename": cache_basename(u, v, r),
    }


def params_match(meta: dict[str, Any], u: np.ndarray, v: np.ndarray, r: float, atol: float = 1e-15) -> bool:
    if meta.get("version") != CACHE_VERSION:
        return False
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if len(meta.get("u", [])) != 2 or len(meta.get("v", [])) != 2:
        return False
    if not np.allclose(meta["u"], u, atol=atol, rtol=0):
        return False
    if not np.allclose(meta["v"], v, atol=atol, rtol=0):
        return False
    if not np.isclose(float(meta.get("r", float("nan"))), float(r), atol=atol, rtol=0):
        return False
    return True


def list_cached_N(path: Path) -> list[int]:
    if not path.exists():
        return []
    found: set[int] = set()
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            if key.startswith("phi_N"):
                found.add(int(key[5:]))
            elif key.startswith("pi_N"):
                found.add(int(key[4:]))
    return sorted(found)


def load_neutral_phi(
    path: Path,
    N: int,
    u: np.ndarray,
    v: np.ndarray,
    r: float,
) -> np.ndarray | None:
    """Return cached phi for N, or None if missing / params mismatch."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        if "meta_json" not in z.files:
            return None
        meta = json.loads(str(z["meta_json"]))
        if not params_match(meta, u, v, r):
            return None
        key = _phi_key(N)
        if key not in z.files:
            key = _legacy_pi_key(N)
            if key not in z.files:
                return None
        phi = np.asarray(z[key], dtype=float)
    s = float(phi.sum())
    if s <= 0.0:
        return None
    return phi / s


def save_neutral_phi(
    path: Path,
    N: int,
    phi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    r: float,
) -> None:
    """Append or overwrite phi for one N; preserve other N in the same file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    key = _phi_key(N)
    meta = params_signature(u, v, r)

    existing: dict[str, np.ndarray | str] = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            old_meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else None
            if old_meta is not None and not params_match(old_meta, u, v, r):
                raise ValueError(
                    f"Cache {path} belongs to different u/v/r ({old_meta.get('basename')}). "
                    f"Expected {meta['basename']}."
                )
            for name in z.files:
                if name != "meta_json":
                    existing[name] = z[name]

    phi = np.asarray(phi, dtype=float)
    phi = phi / float(phi.sum())
    existing[key] = phi
    existing["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **existing)


def resolve_cache_path(
    cfg: dict[str, Any],
    config_dir: Path,
    u: np.ndarray,
    v: np.ndarray,
    r: float,
) -> Path | None:
    if cfg.get("neutral_cache") is False:
        return None
    raw = cfg.get("neutral_cache_file")
    if raw is not None and raw is not False and raw != "":
        p = Path(str(raw))
        return p if p.is_absolute() else config_dir / p
    cache_dir = cfg.get("neutral_cache_dir", cfg.get("cache_dir", "cache"))
    if cache_dir is None or cache_dir is False:
        return None
    d = Path(str(cache_dir))
    if not d.is_absolute():
        d = config_dir / d
    return cache_path_from_params(u, v, r, d)
