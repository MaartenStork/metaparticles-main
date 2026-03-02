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
STRUCTURE_FILE = os.path.join(PROJECT_ROOT, "structures", "sphere_16.3_dist_0.8.lammps_60.data")
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
        if stripped.startswith("pair_coeff 1 2*61 lj/cut"):
            for ltype in sorted(type_to_eps.keys()):
                eps = type_to_eps[ltype]
                out_lines.append(
                    f"pair_coeff 1 {ltype} lj/cut {eps:.6f} 1.75 2.5\n"
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
    total_steps = 251000
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


def endocytosis_score(mp_positions, mem_positions):
    """
    Clustering-based endocytosis detection.

    The KEY idea: run DBSCAN on ALL particles (membrane + MP) together.
    - If the MP has truly passed through the membrane and is floating inside
      with a gap, DBSCAN will assign MP beads to their OWN cluster, separate
      from the membrane cluster. This is the SUCCESS signal.
    - If the MP is stuck in the membrane wall, wrapped on the surface, or
      touching the membrane, MP beads will be in the SAME cluster as membrane
      beads. This is NOT success.

    Four signals combined into [0, 1]:

    1. MP SEPARATION × INSIDE (weight 0.50):
       DBSCAN on all particles — are MP beads NOT in a membrane cluster?
       Gated by radial position — is MP COM inside the membrane sphere?
       separation × inside_score:
       - separated + inside  = 1.0 (true endocytosis!)
       - separated + outside = 0.0 (bud formation, not endocytosis)
       - not separated       = 0.0 (stuck in membrane wall)

    2. SOLID ANGLE COVERAGE (weight 0.30):
       What fraction of directions from MP COM have membrane beads?

    3. MEMBRANE INTEGRITY (weight 0.20):
       Is the membrane still one connected piece? Rupture = bad.

    Outcomes:
        Bounced off (far):     gsep=0  cov≈0.0  int=1.0  → 0.20
        Surface contact:       gsep=0  cov≈0.5  int=1.0  → 0.35
        Partial wrap:          gsep=0  cov≈0.7  int=1.0  → 0.41
        Bud (wrapped outside): gsep=0  cov≈0.9  int=1.0  → 0.47  (caught!)
        Stuck in membrane:     gsep=0  cov≈0.9  int=1.0  → 0.47
        Fully inside + gap:    gsep=1  cov≈0.9  int=1.0  → 0.97 ★
        Membrane ruptured:     gsep=?  cov=var  int≈0.3  → 0.3-0.5

    Returns: (score, details_dict)
    """
    from scipy.spatial import cKDTree
    from sklearn.cluster import DBSCAN

    mp_com = mp_positions.mean(axis=0)
    mp_radius = np.max(np.linalg.norm(mp_positions - mp_com, axis=1))
    n_mp = len(mp_positions)

    mem_com = mem_positions.mean(axis=0)
    mem_dists = np.linalg.norm(mem_positions - mem_com, axis=1)
    mem_radius = np.median(mem_dists)

    # ── 1. DBSCAN on ALL particles ──────────────────────────────────────
    # Combine membrane + MP positions, track which are MP
    all_positions = np.vstack([mem_positions, mp_positions])
    is_mp = np.zeros(len(all_positions), dtype=bool)
    is_mp[len(mem_positions):] = True

    # eps=1.0: membrane beads (spacing ~0.85) stay connected,
    # but any MP bead further than 1.0 from the membrane won't merge in.
    # Small eps = strict separation detection.
    db = DBSCAN(eps=1.0, min_samples=3).fit(all_positions)
    labels = db.labels_

    mp_labels = labels[is_mp]
    mem_labels = labels[~is_mp]

    mem_cluster_ids = set(mem_labels) - {-1}

    # Count MP beads that ended up in a membrane cluster (= stuck/touching).
    # Everything else (own cluster OR noise) = separated from membrane.
    mp_in_mem_cluster = sum(
        int(np.sum(mp_labels == c)) for c in mem_cluster_ids
    )
    mp_beads_separated = n_mp - mp_in_mem_cluster
    mp_separation = mp_beads_separated / n_mp if n_mp > 0 else 0.0

    # ── 2. Solid angle coverage ─────────────────────────────────────────
    n_theta, n_phi = 12, 18
    thetas = np.linspace(0, np.pi, n_theta + 1)
    phis = np.linspace(-np.pi, np.pi, n_phi + 1)

    coverage_cutoff = mem_radius + 5.0

    vecs = mem_positions - mp_com
    dists_to_mp = np.linalg.norm(vecs, axis=1)
    nearby = dists_to_mp < coverage_cutoff
    vecs_nearby = vecs[nearby]
    dists_nearby = dists_to_mp[nearby]

    if len(vecs_nearby) == 0:
        coverage = 0.0
    else:
        unit_v = vecs_nearby / dists_nearby[:, None]
        beam_theta = np.arccos(np.clip(unit_v[:, 2], -1, 1))
        beam_phi = np.arctan2(unit_v[:, 1], unit_v[:, 0])

        theta_bins = np.clip(np.digitize(beam_theta, thetas) - 1, 0, n_theta - 1)
        phi_bins = np.clip(np.digitize(beam_phi, phis) - 1, 0, n_phi - 1)

        bin_centers_theta = 0.5 * (thetas[:-1] + thetas[1:])
        bin_weights = np.sin(bin_centers_theta)
        bin_weights /= bin_weights.sum()

        occupied = np.zeros((n_theta, n_phi), dtype=bool)
        occupied[theta_bins, phi_bins] = True

        weighted_occ = 0.0
        for ti in range(n_theta):
            weighted_occ += bin_weights[ti] * (occupied[ti].sum() / n_phi)
        coverage = float(weighted_occ)

    # ── 3. Membrane integrity ───────────────────────────────────────────
    n_mem_clusters = len(mem_cluster_ids)
    n_noise = int((mem_labels == -1).sum())
    noise_frac = n_noise / len(mem_positions) if len(mem_positions) > 0 else 0

    if n_mem_clusters == 1:
        integrity = 1.0
    elif n_mem_clusters == 2:
        sizes = [np.sum(mem_labels == c) for c in mem_cluster_ids]
        ratio = min(sizes) / max(sizes)
        integrity = 0.6 + 0.3 * (1 - ratio)
    else:
        integrity = max(0.0, 0.4 - 0.1 * (n_mem_clusters - 2))
    integrity *= (1.0 - noise_frac)

    # ── 4. Radial position: is MP inside the membrane sphere? ──────────
    # Without this, bud formation (wrapped on outside) looks identical
    # to true endocytosis (floating inside).
    mp_dist_from_center = float(np.linalg.norm(mp_com - mem_com))
    radial_ratio = mp_dist_from_center / mem_radius if mem_radius > 0 else 999
    # ratio < 1 = inside, ratio > 1 = outside
    # Smooth sigmoid: 1.0 when deep inside, 0.0 when at/outside the shell
    # Steep transition around ratio = 0.85 (must be well inside, not near wall)
    inside_score = 1.0 / (1.0 + np.exp(30.0 * (radial_ratio - 0.85)))
    inside_score = float(inside_score)

    # ── Diagnostics ─────────────────────────────────────────────────────
    mp_tree = cKDTree(mp_positions)
    mem_tree = cKDTree(mem_positions)
    min_dists, _ = mem_tree.query(mp_positions, k=1)
    gap = float(np.min(min_dists))

    contact_pairs = mp_tree.query_ball_tree(mem_tree, r=3.0)
    contacts = sum(len(c) for c in contact_pairs)

    # ── Combine ─────────────────────────────────────────────────────────
    # separation × inside_score acts as a gate:
    #   - separated + inside  → high (true endocytosis)
    #   - separated + outside → low  (bud / bounced)
    #   - not separated       → low  (stuck in membrane)
    gated_sep = mp_separation * inside_score
    gated_cov = coverage * inside_score

    w_sep, w_cov, w_int = 0.50, 0.30, 0.20
    score = w_sep * gated_sep + w_cov * gated_cov + w_int * integrity

    details = {
        "mp_separation": round(float(mp_separation), 4),
        "inside_score": round(inside_score, 4),
        "gated_sep": round(float(gated_sep), 4),
        "gated_cov": round(float(gated_cov), 4),
        "coverage": round(coverage, 4),
        "integrity": round(integrity, 4),
        "gap": round(gap, 3),
        "radial_ratio": round(radial_ratio, 4),
        "n_mem_clusters": n_mem_clusters,
        "mp_beads_separated": int(mp_beads_separated),
        "n_mem_noise": n_noise,
        "contacts": contacts,
        "mp_radius": round(float(mp_radius), 3),
        "mem_radius": round(float(mem_radius), 3),
    }

    return float(np.clip(score, 0.0, 1.0)), details


def extract_objective(run_dir):
    """
    Compute objective from the last frame of the trajectory.

    Uses clustering + solid-angle coverage to produce a continuous [0, 1] score
    that naturally distinguishes all interaction modes (bounced, wrapped,
    stuck, half-through, fully endocytosed, ruptured).

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
              f"{'score':>5s}  {'gsep':>5s} {'cov':>5s} {'int':>4s} {'r/R':>5s}  "
              f"{'time':>5s}  {'':>2s}")
        print(f"    {'─' * 75}")
        for r in results:
            if r.get("status") == "success":
                marker = " ★" if r["objective"] == best_obj else ""
                print(f"    {r['eval_id']:>4d}  "
                      f"{r['epsilons'][0]:>6.3f} {r['epsilons'][1]:>6.3f} {r['epsilons'][2]:>6.3f}  "
                      f"{r['objective']:>5.3f}  "
                      f"{r.get('gated_sep', 0):>5.3f} "
                      f"{r.get('coverage', 0):>5.3f} "
                      f"{r.get('integrity', 0):>4.2f} "
                      f"{r.get('radial_ratio', 0):>5.2f}  "
                      f"{r.get('wall_time_s', 0):>5.0f}s{marker}")
            else:
                print(f"    {r['eval_id']:>4d}  "
                      f"{r['epsilons'][0]:>6.3f} {r['epsilons'][1]:>6.3f} {r['epsilons'][2]:>6.3f}  "
                      f"{'FAIL':>5s}  {'─':>5s} {'─':>5s} {'─':>4s} {'─':>5s}  "
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
        print(f"  ✓ score     = {obj:.3f}  (1.0 = fully endocytosed)")
        print(f"    gated_sep = {metadata.get('gated_sep', 0):.3f}  "
              f"(sep={metadata.get('mp_separation', 0):.2f} × "
              f"inside={metadata.get('inside_score', 0):.2f}  "
              f"r/R={metadata.get('radial_ratio', 0):.2f})")
        print(f"    coverage  = {metadata.get('coverage', 0):.3f}")
        print(f"    integrity = {metadata.get('integrity', 0):.3f}  "
              f"(membrane clusters: {metadata.get('n_mem_clusters', '?')})")
        print(f"    gap       = {metadata.get('gap', 0):.3f}")
        print(f"    wall time = {wall_time:.1f}s")

        return obj, metadata

    # Initial sampling (Sobol)
    for i in range(start_eval, min(n_initial, total_evals)):
        if i == start_eval and not train_X_list:
            # Generate all initial points at once
            sobol_X = draw_sobol_samples(bounds=bounds, n=n_initial, q=1).squeeze(1)

        idx_in_sobol = i
        if idx_in_sobol < n_initial and not any(r["eval_id"] == i for r in all_results):
            epsilons = sobol_X[idx_in_sobol].tolist() if i < n_initial else None
            if epsilons is None:
                break
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
        print(f"  epsilons   = {best['epsilons']}")
        print(f"  score      = {best['objective']:.3f}")
        print(f"  gated_sep  = {best.get('gated_sep', '?')} "
              f"(sep={best.get('mp_separation', '?')} × "
              f"inside={best.get('inside_score', '?')})")
        print(f"  r/R        = {best.get('radial_ratio', '?')}")
        print(f"  coverage   = {best.get('coverage', '?')}")
        print(f"  integrity  = {best.get('integrity', '?')} "
              f"(clusters: {best.get('n_mem_clusters', '?')})")


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
