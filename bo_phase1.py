#!/usr/bin/env python3
"""
Phase 1: Bayesian Optimization of 3 MP-membrane interaction parameters.

60 MP beads are split into 3 azimuthal wedge groups of 20.
Each group shares one epsilon value for membrane-MP LJ interaction.
BO (BoTorch GP + EI) finds the optimal 3 epsilon values to maximize wrapping.

Usage:
    python bo_phase1.py --dry-run          # verify grouping + generated LAMMPS input
    python bo_phase1.py --n-initial 10 --n-iter 100
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LAMMPS_BINARY = os.path.join(PROJECT_ROOT, "lammps-22Jul2025", "build", "lmp")
TEMPLATE_LMP = os.path.join(PROJECT_ROOT, "lammps_script", "run_sphere.lmp")
MP_DATA_FILE = os.path.join(PROJECT_ROOT, "MP_datafile_massimiliano", "MP60_eq.data")
STRUCTURE_FILE = os.path.join(PROJECT_ROOT, "lammps_script", "merged_structure.data")
RUN_DIR = os.path.join(PROJECT_ROOT, "bo_runs")

EPS_BOUNDS = [1.0, 9.0]  # search range for each epsilon
N_GROUPS = 3
N_BEADS = 60
MPI_NP = 4
SIM_TIMEOUT = 3600  # seconds

# ---------------------------------------------------------------------------
# 1. Bead Grouping
# ---------------------------------------------------------------------------
def compute_bead_groups(mp_file=MP_DATA_FILE):
    """
    Split 60 MP beads into 3 groups of 20 by azimuthal angle (120-deg wedges).

    Returns:
        groups: dict {0: [lammps_types], 1: [...], 2: [...]}
        bead_info: list of dicts with bead details
    """
    sys.path.insert(0, PROJECT_ROOT)
    from utilities import read_MP_data_v2

    coords, bonds, Lbox = read_MP_data_v2(mp_file)
    cm = np.mean(coords, axis=0)
    centered = coords - cm

    thetas = np.degrees(np.arctan2(centered[:, 1], centered[:, 0]))
    sorted_indices = np.argsort(thetas)

    groups = {}
    bead_info = []

    for g in range(N_GROUPS):
        start = g * 20
        end = (g + 1) * 20
        bead_indices = sorted_indices[start:end]
        lammps_types = (bead_indices + 2).tolist()  # bead 0 -> type 2
        groups[g] = sorted(lammps_types)
        for bi in bead_indices:
            bead_info.append({
                "bead_index": int(bi),
                "lammps_type": int(bi + 2),
                "theta_deg": float(thetas[bi]),
                "group": g,
                "xyz": centered[bi].tolist(),
            })

    return groups, bead_info


def save_bead_groups(groups, bead_info, outdir=RUN_DIR):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "bead_groups.json")
    with open(path, "w") as f:
        json.dump({"groups": {str(k): v for k, v in groups.items()},
                    "bead_info": bead_info}, f, indent=2)
    print(f"Saved bead groups to {path}")


# ---------------------------------------------------------------------------
# 2. LAMMPS Input Generator
# ---------------------------------------------------------------------------
def write_lammps_input(epsilons, groups, output_path):
    """
    Read template, inject per-group pair_coeff overrides, write to output_path.
    """
    with open(TEMPLATE_LMP) as f:
        lines = f.readlines()

    # Build type -> eps mapping
    type_to_eps = {}
    for g, eps in enumerate(epsilons):
        for lammps_type in groups[g]:
            type_to_eps[lammps_type] = eps

    out_lines = []
    for line in lines:
        stripped = line.strip()

        # Replace relative read_data path with absolute
        if stripped.startswith("read_data"):
            out_lines.append(f"read_data {STRUCTURE_FILE}\n")
            continue

        out_lines.append(line)

        # After the default membrane-MP pair_coeff line, insert overrides
        if stripped.startswith("pair_coeff 1 2*61 cosine/squared"):
            for ltype in sorted(type_to_eps.keys()):
                eps = type_to_eps[ltype]
                out_lines.append(
                    f"pair_coeff 1 {ltype} cosine/squared {eps:.6f} 1.75 1.85 wca\n"
                )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(out_lines)


# ---------------------------------------------------------------------------
# 3. Simulation Runner
# ---------------------------------------------------------------------------
def run_simulation(lmp_input, run_dir):
    """
    Run LAMMPS via mpirun with live progress output.
    Parses LAMMPS thermo output to show timestep progress.
    Returns (success, wall_time_s).
    """
    cmd = [
        "mpirun", "-np", str(MPI_NP),
        LAMMPS_BINARY,
        "-i", os.path.basename(lmp_input),
        "-v", "ktilt", "12.0",
        "-v", "ksplay", "1.0",
        "-v", "rcut", "2.5",
        "-v", "wc", "2.0",
        "-v", "zeta", "5.0",
    ]

    # Total steps: 1000 (warmup) + 250000 (production) = 251000
    total_steps = 500100
    stdout_log = open(os.path.join(run_dir, "stdout.log"), "w")
    stderr_log = open(os.path.join(run_dir, "stderr.log"), "w")

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, cwd=run_dir,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

        current_step = 0
        for line in proc.stdout:
            stdout_log.write(line)

            # Parse LAMMPS thermo lines: "Step Temp CPURemain PE Ebond Epair Vol"
            # Data lines start with a number
            parts = line.split()
            if len(parts) >= 4:
                try:
                    step = int(parts[0])
                    temp = float(parts[1])
                    pe = float(parts[3])
                    pct = min(100.0, step / total_steps * 100)
                    elapsed = time.time() - t0
                    eta = (elapsed / max(pct, 0.1)) * (100 - pct) if pct > 0 else 0
                    print(f"\r    step {step:>7d}/{total_steps} ({pct:5.1f}%) | "
                          f"T={temp:.3f} PE={pe:.1f} | "
                          f"elapsed {elapsed:.0f}s eta {eta:.0f}s", end="", flush=True)
                    current_step = step
                except (ValueError, IndexError):
                    pass

        # Read stderr
        stderr_output = proc.stderr.read()
        stderr_log.write(stderr_output)

        proc.wait(timeout=SIM_TIMEOUT)
        wall_time = time.time() - t0

        if current_step > 0:
            print()  # newline after progress

        stdout_log.close()
        stderr_log.close()

        return proc.returncode == 0, wall_time
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout_log.close()
        stderr_log.close()
        return False, time.time() - t0


# ---------------------------------------------------------------------------
# 4. Objective Extraction
# ---------------------------------------------------------------------------
def parse_last_frame(dump_path):
    """
    Read the last frame from a LAMMPS dump file.
    Returns dict: atom_id -> (type, x, y, z)
    """
    # Find the byte offset of the last ITEM: TIMESTEP
    last_offset = -1
    with open(dump_path, "rb") as f:
        # Read in chunks from the end for efficiency
        f.seek(0, 2)
        file_size = f.tell()
        chunk_size = min(file_size, 5_000_000)  # read last 5MB
        f.seek(max(0, file_size - chunk_size))
        data = f.read().decode("utf-8", errors="replace")

    # Find last occurrence of ITEM: TIMESTEP
    idx = data.rfind("ITEM: TIMESTEP")
    if idx < 0:
        raise ValueError(f"No ITEM: TIMESTEP found in {dump_path}")

    frame_text = data[idx:]
    lines = frame_text.strip().split("\n")

    # Parse header
    # Line 0: ITEM: TIMESTEP
    # Line 1: timestep value
    # Line 2: ITEM: NUMBER OF ATOMS
    # Line 3: N atoms
    # Line 4: ITEM: BOX BOUNDS
    # Lines 5-7: box bounds
    # Line 8: ITEM: ATOMS id type xu yu zu ix iy iz mux muy muz mol fx fy fz
    # Lines 9+: atom data
    n_atoms = int(lines[3])
    atoms = {}
    for i in range(9, 9 + n_atoms):
        parts = lines[i].split()
        aid = int(parts[0])
        atype = int(parts[1])
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        atoms[aid] = (atype, x, y, z)

    return atoms


def solid_angle_coverage(mp_positions, mem_positions, contact_cutoff=1.85, n_bins=12):
    """
    Compute the fraction of the unit sphere around the MP COM that is covered
    by contacting membrane beads.

    Each contact point is spread over a small angular cone (smoothing_angle)
    to account for the fact that discrete membrane beads represent a continuous
    sheet. Uses a coarse (theta, phi) grid — n_bins=12 gives 15-deg resolution,
    appropriate for ~300-800 discrete contacts.

    Returns: (coverage [0-1], n_contacts)
    """
    from scipy.spatial import cKDTree
    from scipy.ndimage import maximum_filter

    mp_com = mp_positions.mean(axis=0)

    # Find membrane beads within cutoff of any MP bead
    mp_tree = cKDTree(mp_positions)
    mem_tree = cKDTree(mem_positions)
    contact_lists = mp_tree.query_ball_tree(mem_tree, r=contact_cutoff)
    contact_ids = set()
    for cl in contact_lists:
        contact_ids.update(cl)
    n_contacts = len(contact_ids)

    if n_contacts == 0:
        return 0.0, 0

    # Direction vectors from MP COM to each contacting membrane bead
    contact_pos = mem_positions[list(contact_ids)]
    directions = contact_pos - mp_com
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    directions = directions / norms

    # Bin onto a grid in (theta, phi) space
    theta = np.arccos(np.clip(directions[:, 2], -1, 1))  # [0, pi]
    phi = np.arctan2(directions[:, 1], directions[:, 0])  # [-pi, pi]

    # Coarse grid: n_bins in theta, 2*n_bins in phi
    n_phi_bins = 2 * n_bins
    theta_edges = np.linspace(0, np.pi, n_bins + 1)
    phi_edges = np.linspace(-np.pi, np.pi, n_phi_bins + 1)

    hist, _, _ = np.histogram2d(theta, phi, bins=[theta_edges, phi_edges])

    # Dilate by 1 bin in each direction to bridge small gaps between
    # discrete contact points (membrane beads are ~1 sigma apart, so
    # neighboring beads on the sheet fill adjacent angular bins)
    dilated = maximum_filter(hist, size=3, mode='wrap')
    occupied = (dilated > 0).astype(float)

    # Weight each bin by its solid angle: sin(theta) * dtheta * dphi
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    dtheta = theta_edges[1] - theta_edges[0]
    dphi = phi_edges[1] - phi_edges[0]
    solid_angle_weights = np.sin(theta_centers) * dtheta * dphi  # shape (n_bins,)

    # Total solid angle of occupied bins / 4*pi
    covered = np.sum(occupied * solid_angle_weights[:, np.newaxis])
    coverage = covered / (4 * np.pi)

    return float(np.clip(coverage, 0.0, 1.0)), int(n_contacts)


def endocytosis_score(mp_positions, mem_positions):
    """
    Score endocytosis using two physical signals:

    Signal 1 – Penetration depth:  max(0, 1 - r/R)
        r = MP COM distance from vesicle center, R = median membrane radius.

    Signal 2 – Solid angle coverage:
        Fraction of the unit sphere around the MP COM that is covered by
        contacting membrane beads (within 1.85 cutoff).
        0 = no wrapping, 0.5 = hemisphere, 1.0 = fully engulfed.

    Combined:  score = depth * (0.5 + 0.5 * coverage)

    Returns: (score, details_dict)
    """
    mp_com = mp_positions.mean(axis=0)
    mem_com = mem_positions.mean(axis=0)

    # Penetration depth
    mem_radii = np.linalg.norm(mem_positions - mem_com, axis=1)
    R = np.median(mem_radii)
    r = np.linalg.norm(mp_com - mem_com)
    radial_ratio = r / R if R > 0 else 999.0
    depth = max(0.0, 1.0 - radial_ratio)

    # Solid angle coverage
    coverage, n_contacts = solid_angle_coverage(mp_positions, mem_positions)

    score = depth * (0.5 + 0.5 * coverage)

    details = {
        "radial_ratio": float(radial_ratio),
        "depth": float(depth),
        "coverage": float(coverage),
        "n_contacts": int(n_contacts),
        "membrane_R": float(R),
    }
    return float(score), details


def extract_objective(run_dir):
    """
    Compute objective from the last frame of the trajectory.
    Returns (objective_value, metadata_dict).
    """
    dump_path = os.path.join(run_dir, "position.lammpstrj")
    if not os.path.exists(dump_path):
        return -1e6, {"status": "no_dump"}

    try:
        atoms = parse_last_frame(dump_path)
    except Exception as e:
        return -1e6, {"status": f"parse_error: {e}"}

    mp_positions = np.array(
        [[x, y, z] for aid, (atype, x, y, z) in atoms.items() if 2 <= atype <= 61]
    )
    mem_positions = np.array(
        [[x, y, z] for aid, (atype, x, y, z) in atoms.items() if atype == 1]
    )

    if len(mp_positions) == 0:
        return -1e6, {"status": "no_mp_atoms"}

    mp_com = mp_positions.mean(axis=0)
    score, details = endocytosis_score(mp_positions, mem_positions)

    metadata = {
        "z_com": float(mp_com[2]),
        "x_com": float(mp_com[0]),
        "y_com": float(mp_com[1]),
        **details,
    }
    return score, metadata


# ---------------------------------------------------------------------------
# 5. Data Storage
# ---------------------------------------------------------------------------
def save_results(results, filepath):
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, filepath)


# ---------------------------------------------------------------------------
# 6. BO Loop
# ---------------------------------------------------------------------------
def run_bo(n_initial=10, n_iter=100, resume=True):
    import torch
    from botorch.acquisition import ExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from botorch.utils.sampling import draw_sobol_samples
    from gpytorch.mlls import ExactMarginalLogLikelihood

    bounds = torch.tensor(
        [[EPS_BOUNDS[0]] * N_GROUPS, [EPS_BOUNDS[1]] * N_GROUPS],
        dtype=torch.double,
    )

    # Bead groups
    groups, bead_info = compute_bead_groups()
    save_bead_groups(groups, bead_info)

    results_path = os.path.join(RUN_DIR, "bo_results.json")
    all_results = []
    start_eval = 0

    # Resume from existing results if available
    if resume and os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        start_eval = len(all_results)
        print(f"Resuming from evaluation {start_eval}")

    total_evals = n_initial + n_iter

    # Build training data from existing results
    train_X_list = []
    train_Y_list = []
    for r in all_results:
        train_X_list.append(r["epsilons"])
        train_Y_list.append(r["objective"])

    def print_summary_table(results):
        """Print a formatted table of all evaluations so far."""
        successful = [r for r in results if r.get("status") == "success"]
        if not successful:
            return
        best_obj = max(r["objective"] for r in successful)
        print(f"\n    {'eval':>4s}  {'eps_0':>6s} {'eps_1':>6s} {'eps_2':>6s}  "
              f"{'score':>5s}  {'r/R':>5s}  {'time':>5s}  {'':>2s}")
        print(f"    {'─' * 55}")
        for r in results:
            if r.get("status") == "success":
                marker = " ★" if r["objective"] == best_obj else ""
                print(f"    {r['eval_id']:>4d}  "
                      f"{r['epsilons'][0]:>6.3f} {r['epsilons'][1]:>6.3f} {r['epsilons'][2]:>6.3f}  "
                      f"{r['objective']:>5.3f}  "
                      f"{r.get('radial_ratio', 0):>5.2f}  "
                      f"{r.get('wall_time_s', 0):>5.0f}s{marker}")
            else:
                print(f"    {r['eval_id']:>4d}  "
                      f"{r['epsilons'][0]:>6.3f} {r['epsilons'][1]:>6.3f} {r['epsilons'][2]:>6.3f}  "
                      f"{'FAIL':>5s}  {'─':>5s}  "
                      f"{r.get('wall_time_s', 0):>5.0f}s")

    def evaluate(epsilons, eval_id):
        run_dir = os.path.join(RUN_DIR, f"eval_{eval_id:04d}")
        lmp_path = os.path.join(run_dir, "run.lmp")

        write_lammps_input(epsilons, groups, lmp_path)

        phase = "SOBOL" if eval_id < n_initial else "BO"
        print(f"\n{'=' * 65}")
        print(f"  [{phase}] Eval {eval_id}/{total_evals - 1}")
        print(f"  eps = [{epsilons[0]:.4f}, {epsilons[1]:.4f}, {epsilons[2]:.4f}]")
        print(f"  dir = {run_dir}")
        print(f"{'─' * 65}")

        success, wall_time = run_simulation(lmp_path, run_dir)

        if not success:
            print(f"  ✗ FAILED after {wall_time:.1f}s")
            return -1e6, {"status": "failed", "wall_time_s": wall_time}

        obj, metadata = extract_objective(run_dir)
        metadata["status"] = "success"
        metadata["wall_time_s"] = wall_time

        print(f"{'─' * 65}")
        print(f"  ✓ score = {obj:.3f}  "
              f"(r/R={metadata.get('radial_ratio', 0):.2f})  "
              f"time={wall_time:.0f}s")

        return obj, metadata

    # Initial sampling (Sobol)
    sobol_X = draw_sobol_samples(bounds=bounds, n=n_initial, q=1).squeeze(1)

    for i in range(start_eval, min(n_initial, total_evals)):
        if not any(r["eval_id"] == i for r in all_results):
            epsilons = sobol_X[i].tolist()
            obj, metadata = evaluate(epsilons, i)
            train_X_list.append(epsilons)
            train_Y_list.append(obj)
            all_results.append({"eval_id": i, "epsilons": epsilons, "objective": obj, **metadata})
            save_results(all_results, results_path)
            print_summary_table(all_results)

    # BO iterations
    for i in range(max(start_eval, n_initial), total_evals):
        if len(train_X_list) < 2:
            print("Not enough data to fit GP, need at least 2 evaluations")
            break

        train_X = torch.tensor(train_X_list, dtype=torch.double)
        train_Y = torch.tensor(train_Y_list, dtype=torch.double).unsqueeze(-1)

        # Filter out failed evaluations for GP fitting
        valid = train_Y.squeeze(-1) > -1e5
        if valid.sum() < 2:
            print("Not enough successful evaluations to fit GP")
            break

        gp = SingleTaskGP(train_X[valid], train_Y[valid])
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        best_f = train_Y[valid].max()
        acq = ExpectedImprovement(model=gp, best_f=best_f)

        candidate, acq_value = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )

        epsilons = candidate.squeeze(0).tolist()
        obj, metadata = evaluate(epsilons, i)

        train_X_list.append(epsilons)
        train_Y_list.append(obj)
        all_results.append({"eval_id": i, "epsilons": epsilons, "objective": obj, **metadata})
        save_results(all_results, results_path)
        print_summary_table(all_results)

    print("\nOptimization complete.")
    successful = [r for r in all_results if r.get("status") == "success"]
    if successful:
        best = max(successful, key=lambda r: r["objective"])
        print(f"Best result: eval {best['eval_id']}")
        print(f"  epsilons = {best['epsilons']}")
        print(f"  score    = {best['objective']:.3f}")
        print(f"  r/R      = {best.get('radial_ratio', '?')}")


# ---------------------------------------------------------------------------
# 7. Dry Run
# ---------------------------------------------------------------------------
def dry_run():
    """Verify bead grouping and LAMMPS input generation without running sims."""
    print("=" * 60)
    print("DRY RUN: Verifying bead grouping and LAMMPS input generation")
    print("=" * 60)

    groups, bead_info = compute_bead_groups()
    save_bead_groups(groups, bead_info)

    print(f"\nBead grouping ({N_BEADS} beads -> {N_GROUPS} groups of {N_BEADS // N_GROUPS}):")
    for g in range(N_GROUPS):
        types = groups[g]
        beads_in_group = [b for b in bead_info if b["group"] == g]
        theta_min = min(b["theta_deg"] for b in beads_in_group)
        theta_max = max(b["theta_deg"] for b in beads_in_group)
        print(f"  Group {g}: {len(types)} beads, theta=[{theta_min:.1f}, {theta_max:.1f}] deg")
        print(f"    LAMMPS types: {types}")

    # Generate a test LAMMPS input
    test_eps = [0.5, 1.0, 2.0]
    test_dir = os.path.join(RUN_DIR, "dry_run")
    test_lmp = os.path.join(test_dir, "run.lmp")
    write_lammps_input(test_eps, groups, test_lmp)
    print(f"\nGenerated test LAMMPS input: {test_lmp}")
    print(f"  Test epsilons: {test_eps}")

    # Show the pair_coeff section
    with open(test_lmp) as f:
        in_pair = False
        for line in f:
            if "pair_style" in line or "pair_coeff" in line or "pair_modify" in line:
                print(f"  {line.rstrip()}")

    # Check prerequisites
    print(f"\nPrerequisites:")
    print(f"  LAMMPS binary: {LAMMPS_BINARY} -> {'EXISTS' if os.path.isfile(LAMMPS_BINARY) else 'MISSING'}")
    print(f"  Structure file: {STRUCTURE_FILE} -> {'EXISTS' if os.path.isfile(STRUCTURE_FILE) else 'MISSING'}")
    print(f"  Template: {TEMPLATE_LMP} -> {'EXISTS' if os.path.isfile(TEMPLATE_LMP) else 'MISSING'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 1: BO optimization of 3 MP-membrane epsilons")
    parser.add_argument("--dry-run", action="store_true", help="Verify setup without running simulations")
    parser.add_argument("--n-initial", type=int, default=10, help="Number of initial Sobol samples (default: 10)")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of BO iterations (default: 100)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore existing results")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        run_bo(n_initial=args.n_initial, n_iter=args.n_iter, resume=not args.no_resume)


if __name__ == "__main__":
    main()
