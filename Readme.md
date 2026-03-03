# Metaparticles

Bayesian optimization of anisotropic nanoparticle-membrane interactions for endocytosis. A coarse-grained metaparticle (60 beads, 3 azimuthal groups) interacts with a lipid vesicle via tunable cosine/squared + WCA potentials. BO (BoTorch GP + EI) searches over the 3 group epsilon values to maximize membrane wrapping and penetration depth.

Built on LAMMPS with the `membrane_sillanov2` pair style for the lipid bilayer.

## Quick start

```bash
# 1. Compile LAMMPS with the custom potential (see potential_files/)
# 2. Generate initial structures
python write_structures.py

# 3. Calibrate the membrane
cd lammps_script && python merge_calibrated.py

# 4. Run Bayesian optimization
python bo_phase1.py --n-initial 10 --n-iter 100
```

## Structure

- `bo_phase1.py` -- BO loop (Sobol init + GP/EI acquisition)
- `lammps_script/` -- LAMMPS input templates and membrane calibration
- `analyze_bo.ipynb` -- visualization of BO results, potential curves, wrapping analysis
- `structures/` -- generated initial configurations
- `MP_datafile_massimiliano/` -- metaparticle bead coordinates
- `potential_files/` -- custom LAMMPS pair style source
