#!/usr/bin/env python3
"""
WFE Phase 2 — Density Current Visualizer
Plots θ' and w snapshots from CSV output (numpy only, no pandas).
Usage: python3 scripts/plot_density_current.py [results_dir] [snap_index|all]
"""
import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_snapshot(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    x, z = data[:, 0], data[:, 1]
    u, w, theta = data[:, 3], data[:, 4], data[:, 5]
    xs = np.unique(x); zs = np.unique(z)
    nx, nz = len(xs), len(zs)
    X = x.reshape(nz, nx)
    Z = z.reshape(nz, nx)
    W = w.reshape(nz, nx)
    TH = theta.reshape(nz, nx)
    return X, Z, W, TH


def plot_snapshot(path, out_png):
    X, Z, W, TH = load_snapshot(path)
    dtheta = TH - 300.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # θ' (potential temperature perturbation) — Robert (1993) diagnostic
    vmax = max(abs(dtheta).max(), 0.01)
    levels_th = np.linspace(-vmax, vmax * 0.1, 33)
    cf = axes[0].contourf(X / 1e3, Z / 1e3, dtheta,
                          levels=levels_th, cmap="RdBu_r", extend="both")
    axes[0].set_ylabel("z [km]")
    axes[0].set_title(f"θ' [K]  —  {os.path.basename(path)}")
    plt.colorbar(cf, ax=axes[0], fraction=0.02, pad=0.02)

    # w (vertical velocity)
    vw = max(abs(W).max(), 0.01)
    levels_w = np.linspace(-vw, vw, 33)
    cf2 = axes[1].contourf(X / 1e3, Z / 1e3, W,
                           levels=levels_w, cmap="seismic", extend="both")
    axes[1].set_xlabel("x [km]")
    axes[1].set_ylabel("z [km]")
    axes[1].set_title("w [m/s]")
    plt.colorbar(cf2, ax=axes[1], fraction=0.02, pad=0.02)

    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_png}")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results_2d"
    snap_arg    = sys.argv[2] if len(sys.argv) > 2 else "all"

    csvs = sorted(glob.glob(os.path.join(results_dir, "density_current_*.csv")))
    if not csvs:
        print(f"No CSV files found in {results_dir}"); sys.exit(1)

    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

    targets = csvs if snap_arg == "all" else [csvs[int(snap_arg)]]
    for csv in targets:
        base = os.path.splitext(os.path.basename(csv))[0]
        plot_snapshot(csv, os.path.join(results_dir, "plots", base + ".png"))

    print(f"Done. {len(targets)} plot(s) written to {results_dir}/plots/")


if __name__ == "__main__":
    main()
