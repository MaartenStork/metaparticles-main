#!/usr/bin/env python3
"""
Mean bending energy vs time from a LAMMPS dump using MDAnalysis + membrane-curvature.

Energy model (Helfrich, mean term only):
    E_H(t) = (κ/2) * ∫ (2H - C0)^2 dA

Curvatures are computed on a regular (Nx × Ny) grid per frame from a selection
of atoms (here: membrane beads, typically a single type).

Differences vs original Bending_Energy_Helfrich.py:
- Only the mean curvature term is used (no Gaussian curvature term, no κG).
- Trajectory file is a *positional* argument instead of -i/--input.
- Membrane bead type default is 1 (your new model).
- Outputs only: Time, EH, I_2H2, OccupiedArea.

Assumptions:
  - Membrane beads have LAMMPS type == --mem-type (default 1)
  - Use 'wrap=True' so atoms are folded into the box before gridding
  - Length units: Å. κ is in the energy units you want to report (e.g. ε, kBT).

Usage: 
python BE.py position.lammpstrj --kappa 20 --step 3000 --dt 0.01 --nx 24 --ny 24
--smooth-sigma 1.0 --out-prefix newmem_bend
"""

import argparse
import numpy as np

# Matplotlib headless backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional smoothing (install scipy if not available)
try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

import MDAnalysis as mda
from membrane_curvature.base import MembraneCurvature


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute membrane mean bending energy vs time (Helfrich, mean term only)."
    )

    # Positional: trajectory
    p.add_argument(
        "input",
        help="LAMMPS dump file (e.g. wrapped_final.lammpstrj)",
    )

    # Membrane selection
    p.add_argument(
        "--mem-type", type=int, default=1,
        help="LAMMPS type id for membrane beads (default 1)",
    )

    # Grid
    p.add_argument(
        "--nx", type=int, default=24,
        help="Grid bins in x (default 24)",
    )
    p.add_argument(
        "--ny", type=int, default=24,
        help="Grid bins in y (default 24)",
    )

    # Helfrich parameters
    p.add_argument(
        "--kappa", type=float, default=10.0,
        help="Bending rigidity κ (energy units of your choice; default 10.0)",
    )
    p.add_argument(
        "--C0", type=float, default=0.0,
        help="Spontaneous curvature C0 (Å^-1), default 0",
    )

    # Time mapping
    p.add_argument(
        "--step", type=int, default=3000,
        help="Dump stride in MD timesteps between frames (default 3000)",
    )
    p.add_argument(
        "--dt", type=float, default=0.01,
        help="Timestep size per MD step (e.g., τ units; default 0.01)",
    )

    # Output
    p.add_argument(
        "--out-prefix", default="bending_mean",
        help="Prefix for output files (default 'bending_mean')",
    )

    # Curvature controls
    p.add_argument(
        "--mean-is-2H", action="store_true",
        help="Set if MembraneCurvature.mean already equals (k1+k2)=2H."
    )
    p.add_argument(
        "--smooth-sigma", type=float, default=0.0,
        help="Gaussian smoothing sigma (in grid bins) for H. 0 = off.",
    )
    p.add_argument(
        "--min-count", type=int, default=5,
        help="Minimum bead count per cell to accept curvature (default 5).",
    )
    p.add_argument(
        "--print-interval", type=int, default=50,
        help="How often to print diagnostics (in frames). Default 50.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.smooth_sigma and not _HAS_SCIPY:
        raise RuntimeError(
            "Smoothing requested but scipy is not available. "
            "Install scipy or set --smooth-sigma 0.0"
        )

    # 1) Load trajectory
    print(f"Loading dump: {args.input}")
    u = mda.Universe(args.input, format="LAMMPSDUMP")
    nframes = u.trajectory.n_frames
    print(f"  frames: {nframes}, atoms: {len(u.atoms)}")

    # 2) Curvature on membrane beads
    sel_mem = f"type {args.mem_type}"
    print(f"Running MembraneCurvature on selection: '{sel_mem}' with grid {args.nx}x{args.ny}")
    mc = (
        MembraneCurvature(
            universe=u,
            select=sel_mem,
            n_x_bins=args.nx,
            n_y_bins=args.ny,
            wrap=True,
        )
        .run()
    )

    H_grid = mc.results.mean.astype(float)        # shape: (frames, nx, ny)
    counts = getattr(mc.results, "counts", None)  # may not exist in older versions

    if H_grid.shape[0] != nframes:
        print("Warning: curvature frames != trajectory frames (using intersection).")
        nframes = min(nframes, H_grid.shape[0])

    # 3) Energy vs time
    times = np.arange(nframes, dtype=float) * (args.step * args.dt)  # physical time
    EH = np.zeros(nframes, dtype=float)

    # Diagnostics
    I_2H2_series = np.zeros(nframes, dtype=float)
    occ_area_series = np.zeros(nframes, dtype=float)

    for fi in range(nframes):
        ts = u.trajectory[fi]  # advance trajectory
        Lx, Ly = float(ts.dimensions[0]), float(ts.dimensions[1])  # Å
        dA = (Lx / args.nx) * (Ly / args.ny)  # Å^2 per cell

        H = H_grid[fi].copy()

        # Handle NaNs/infs from blank cells
        H[~np.isfinite(H)] = 0.0

        # Optional smoothing
        if args.smooth_sigma > 0.0:
            H = gaussian_filter(H, sigma=args.smooth_sigma, mode="wrap")

        # Occupancy mask
        if counts is not None:
            C = counts[fi].astype(int)
            mask = (C >= args.min_count)
        else:
            mask = np.isfinite(H)

        # Build 2H consistently
        if args.mean_is_2H:
            twoH = H  # library already gives (k1 + k2)
        else:
            twoH = 2.0 * H  # library gives mean H = (k1 + k2)/2

        # Apply mask
        twoH_masked = twoH.copy()
        twoH_masked[~mask] = 0.0
        occ_area = float(np.count_nonzero(mask)) * dA
        occ_area_series[fi] = occ_area

        # Diagnostic invariant: ∫(2H)^2 dA
        I_2H2 = np.sum(twoH_masked ** 2) * dA
        I_2H2_series[fi] = I_2H2

        # Mean term of Helfrich
        I_H_shifted = np.sum((twoH_masked - args.C0) ** 2) * dA  # ∫(2H - C0)^2 dA
        EH_frame = 0.5 * args.kappa * I_H_shifted
        EH[fi] = EH_frame

        if (fi % args.print_interval) == 0 or fi == nframes - 1:
            print(
                f"frame {fi+1:5d}/{nframes}  Lx={Lx:8.3f} Ly={Ly:8.3f}  "
                f"occ_area={occ_area:9.3f}  I_2H2={I_2H2:8.3f}  EH={EH_frame:9.3f}"
            )

    # 4) Save CSV
    csv_name = f"{args.out_prefix}_vs_time.csv"
    with open(csv_name, "w", encoding="utf-8") as fh:
        fh.write("Time,EH,I_2H2,OccupiedArea\n")
        for t, eh, ih2, oa in zip(times, EH, I_2H2_series, occ_area_series):
            fh.write(f"{t:.6f},{eh:.8e},{ih2:.8e},{oa:.8e}\n")
    print(f"✓ Saved: {csv_name}")

    # 5) Plot EH vs time
    plt.figure(figsize=(6.6, 3.6), dpi=140)
    plt.plot(times, EH, lw=1.8, label="E_H (mean term)")
    plt.xlabel("Time")
    plt.ylabel("Energy (units of κ)")
    plt.title("Mean bending energy vs time")
    plt.tight_layout()
    plt.legend(frameon=False)
    fig_name = f"{args.out_prefix}.png"
    plt.savefig(fig_name, dpi=200)
    print(f"✓ Saved: {fig_name}")

    # 6) Console summary
    print("\nSummary:")
    print(f"  ⟨E_H⟩ = {np.mean(EH):.4e}   min={np.min(EH):.4e}   max={np.max(EH):.4e}")
    print(f"  κ = {args.kappa}, C0 = {args.C0}, grid = {args.nx}×{args.ny}")
    print(f"  mean I_2H2 = {np.mean(I_2H2_series):.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
