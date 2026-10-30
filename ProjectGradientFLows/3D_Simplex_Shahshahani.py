import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SimplexGrid:
    x1: np.ndarray
    x2: np.ndarray
    x3: np.ndarray
    x4: np.ndarray
    mask: np.ndarray
    dx: float


def make_simplex_grid(n: int) -> SimplexGrid:
    dx = 1.0 / n
    coords = (np.arange(n) + 0.5) * dx
    x1, x2, x3 = np.meshgrid(coords, coords, coords, indexing="ij")
    x4 = 1.0 - x1 - x2 - x3
    mask = x4 >= 0.0
    return SimplexGrid(x1=x1, x2=x2, x3=x3, x4=x4, mask=mask, dx=dx)


def normalize_density(rho: np.ndarray, mask: np.ndarray, dx: float) -> np.ndarray:
    rho = np.where(mask, rho, 0.0)
    mass = np.sum(rho) * dx**3
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("Invalid mass for normalization.")
    return rho / mass


def entropy_density(rho: np.ndarray, mask: np.ndarray, dx: float) -> float:
    eps = 1e-14
    safe = np.where(mask, np.maximum(rho, eps), 0.0)
    return float(np.sum(safe * (np.log(safe) - 1.0)) * dx**3)


def diffusion_tensor(grid: SimplexGrid, ne: float, eps: float = 1e-12):
    x1 = np.clip(grid.x1, eps, 1.0)
    x2 = np.clip(grid.x2, eps, 1.0)
    x3 = np.clip(grid.x3, eps, 1.0)
    # M = 0.5 * B, and B has the 1/(2 Ne) factor in your PDF.
    # So diffusion tensor uses scale 1/(4 Ne).
    scale = 1.0 / (4.0 * ne)
    b11 = scale * x1 * (1.0 - x1)
    b22 = scale * x2 * (1.0 - x2)
    b33 = scale * x3 * (1.0 - x3)
    b12 = scale * (-x1 * x2)
    b13 = scale * (-x1 * x3)
    b23 = scale * (-x2 * x3)
    return b11, b22, b33, b12, b13, b23


def _neighbor_forward(arr: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    forward = np.empty_like(arr)
    mask_f = np.empty_like(mask)
    if axis == 0:
        forward[:-1, :, :] = arr[1:, :, :]
        forward[-1, :, :] = arr[-1, :, :]
        mask_f[:-1, :, :] = mask[1:, :, :]
        mask_f[-1, :, :] = False
    elif axis == 1:
        forward[:, :-1, :] = arr[:, 1:, :]
        forward[:, -1, :] = arr[:, -1, :]
        mask_f[:, :-1, :] = mask[:, 1:, :]
        mask_f[:, -1, :] = False
    else:
        forward[:, :, :-1] = arr[:, :, 1:]
        forward[:, :, -1] = arr[:, :, -1]
        mask_f[:, :, :-1] = mask[:, :, 1:]
        mask_f[:, :, -1] = False
    return np.where(mask_f, forward, arr)


def _neighbor_backward(arr: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    backward = np.empty_like(arr)
    mask_b = np.empty_like(mask)
    if axis == 0:
        backward[1:, :, :] = arr[:-1, :, :]
        backward[0, :, :] = arr[0, :, :]
        mask_b[1:, :, :] = mask[:-1, :, :]
        mask_b[0, :, :] = False
    elif axis == 1:
        backward[:, 1:, :] = arr[:, :-1, :]
        backward[:, 0, :] = arr[:, 0, :]
        mask_b[:, 1:, :] = mask[:, :-1, :]
        mask_b[:, 0, :] = False
    else:
        backward[:, :, 1:] = arr[:, :, :-1]
        backward[:, :, 0] = arr[:, :, 0]
        mask_b[:, :, 1:] = mask[:, :, :-1]
        mask_b[:, :, 0] = False
    return np.where(mask_b, backward, arr)


def _central_diff(arr: np.ndarray, mask: np.ndarray, axis: int, dx: float) -> np.ndarray:
    forward = _neighbor_forward(arr, mask, axis)
    backward = _neighbor_backward(arr, mask, axis)
    return (forward - backward) / (2.0 * dx)


def _divergence(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray, mask: np.ndarray, dx: float) -> np.ndarray:
    dfx = _central_diff(fx, mask, axis=0, dx=dx)
    dfy = _central_diff(fy, mask, axis=1, dx=dx)
    dfz = _central_diff(fz, mask, axis=2, dx=dx)
    div = dfx + dfy + dfz
    return np.where(mask, div, 0.0)


def apply_shahshahani_laplacian(
    rho: np.ndarray,
    grid: SimplexGrid,
    b11: np.ndarray,
    b22: np.ndarray,
    b33: np.ndarray,
    b12: np.ndarray,
    b13: np.ndarray,
    b23: np.ndarray,
) -> np.ndarray:
    gradx = _central_diff(rho, grid.mask, axis=0, dx=grid.dx)
    grady = _central_diff(rho, grid.mask, axis=1, dx=grid.dx)
    gradz = _central_diff(rho, grid.mask, axis=2, dx=grid.dx)

    fx = b11 * gradx + b12 * grady + b13 * gradz
    fy = b12 * gradx + b22 * grady + b23 * gradz
    fz = b13 * gradx + b23 * grady + b33 * gradz

    return _divergence(fx, fy, fz, grid.mask, grid.dx)


def cg_solve(matvec, b, x0=None, tol=1e-10, max_iter=400):
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - matvec(x)
    p = r.copy()
    rsold = float(np.dot(r, r))
    if rsold <= tol**2:
        return x, 0, np.sqrt(rsold)
    for it in range(1, max_iter + 1):
        ap = matvec(p)
        denom = float(np.dot(p, ap)) + 1e-30
        alpha = rsold / denom
        x += alpha * p
        r -= alpha * ap
        rsnew = float(np.dot(r, r))
        if rsnew <= tol**2:
            return x, it, np.sqrt(rsnew)
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x, max_iter, np.sqrt(rsold)


def implicit_euler_step(
    rho: np.ndarray,
    grid: SimplexGrid,
    tau: float,
    b11: np.ndarray,
    b22: np.ndarray,
    b33: np.ndarray,
    b12: np.ndarray,
    b13: np.ndarray,
    b23: np.ndarray,
    tol: float,
    max_iter: int,
):
    mask = grid.mask
    idx = np.where(mask)
    b_vec = rho[idx]

    def unpack(vec):
        full = np.zeros_like(rho)
        full[idx] = vec
        return full

    def matvec(vec):
        field = unpack(vec)
        lap = apply_shahshahani_laplacian(field, grid, b11, b22, b33, b12, b13, b23)
        return (field - tau * lap)[idx]

    x, iters, res = cg_solve(matvec, b_vec, x0=b_vec, tol=tol, max_iter=max_iter)
    rho_new = unpack(x)
    rho_new = np.where(mask, np.maximum(rho_new, 0.0), 0.0)
    rho_new = normalize_density(rho_new, mask, grid.dx)
    return rho_new, iters, res


def explicit_euler_step(
    rho: np.ndarray,
    grid: SimplexGrid,
    tau: float,
    b11: np.ndarray,
    b22: np.ndarray,
    b33: np.ndarray,
    b12: np.ndarray,
    b13: np.ndarray,
    b23: np.ndarray,
):
    lap = apply_shahshahani_laplacian(rho, grid, b11, b22, b33, b12, b13, b23)
    rho_new = rho + tau * lap
    rho_new = np.where(grid.mask, np.maximum(rho_new, 0.0), 0.0)
    rho_new = normalize_density(rho_new, grid.mask, grid.dx)
    return rho_new


def initialize_density(grid: SimplexGrid, center=(0.2, 0.2, 0.2), sigma=0.08):
    dx = grid.dx
    cx, cy, cz = center
    dist2 = (grid.x1 - cx) ** 2 + (grid.x2 - cy) ** 2 + (grid.x3 - cz) ** 2
    rho = np.exp(-dist2 / (2.0 * sigma**2))
    rho = np.where(grid.mask, rho, 0.0)
    return normalize_density(rho, grid.mask, dx)


def plot_xy_marginal(rho: np.ndarray, grid: SimplexGrid, title: str, save_path: str):
    dx = grid.dx
    mass_xy = np.sum(rho, axis=2) * dx
    mask_xy = grid.x1[:, :, 0] + grid.x2[:, :, 0] <= 1.0
    display = np.where(mask_xy, mass_xy, np.nan)

    plt.figure(figsize=(6, 5))
    plt.imshow(display.T, origin="lower", cmap="viridis")
    plt.colorbar(label="marginal mass")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_simulation(args):
    grid = make_simplex_grid(args.n)
    b11, b22, b33, b12, b13, b23 = diffusion_tensor(grid, args.ne)
    rho = initialize_density(grid, center=(args.c1, args.c2, args.c3), sigma=args.sigma)

    snapshots = []
    times = []
    for step in range(args.steps + 1):
        if step % args.snapshot_every == 0 or step == args.steps:
            snapshots.append(rho.copy())
            times.append(step * args.tau)

        if step == args.steps:
            break

        if args.scheme == "explicit":
            rho = explicit_euler_step(rho, grid, args.tau, b11, b22, b33, b12, b13, b23)
            iters = 0
            res = 0.0
        else:
            rho, iters, res = implicit_euler_step(
                rho,
                grid,
                args.tau,
                b11,
                b22,
                b33,
                b12,
                b13,
                b23,
                args.cg_tol,
                args.cg_max_iter,
            )

        if args.verbose and (step % args.print_every == 0 or step == args.steps - 1):
            ent = entropy_density(rho, grid.mask, grid.dx)
            print(
                f"step {step + 1:4d} | mass={np.sum(rho)*grid.dx**3:.6f} "
                f"| min={rho[grid.mask].min():.3e} | entropy={ent:.6f} "
                f"| cg_iters={iters} | cg_res={res:.2e}"
            )

    os.makedirs(args.out_dir, exist_ok=True)
    for snap, t in zip(snapshots, times):
        fname = f"simplex3d_{args.scheme}_t{t:.4f}_n{args.n}.png"
        plot_xy_marginal(
            snap,
            grid,
            title=f"Shahshahani {args.scheme} (t={t:.4f})",
            save_path=os.path.join(args.out_dir, fname),
        )


def main():
    parser = argparse.ArgumentParser(
        description="JKO-equivalent diffusion on the 3D simplex with Shahshahani metric."
    )
    parser.add_argument("--n", type=int, default=32, help="Grid resolution per axis.")
    parser.add_argument("--tau", type=float, default=1e-4, help="Time step size.")
    parser.add_argument("--steps", type=int, default=30, help="Number of time steps.")
    parser.add_argument("--scheme", choices=["implicit", "explicit"], default="implicit")
    parser.add_argument("--ne", type=float, default=50.0, help="Effective population size.")
    parser.add_argument("--c1", type=float, default=0.2, help="Initial center x1.")
    parser.add_argument("--c2", type=float, default=0.2, help="Initial center x2.")
    parser.add_argument("--c3", type=float, default=0.2, help="Initial center x3.")
    parser.add_argument("--sigma", type=float, default=0.08, help="Initial bump width.")
    parser.add_argument("--cg-tol", type=float, default=1e-9, help="CG tolerance.")
    parser.add_argument("--cg-max-iter", type=int, default=400, help="CG max iterations.")
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_simulation(args)


if __name__ == "__main__":
    main()
