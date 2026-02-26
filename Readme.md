# Pipeline

## Installation
You have to have compiled your custom installation of lammps with the new potential, see readme.md in potential_files/

## Initial configurations
The idea is to combine membrane initial configuration contained in `membrane_data/` with MP data in `MP_datafile_massimiliano`. The script `write_structures.py` can generate initial configurations for single or multiple MP on plane (or spherical vesicle).

The generate combined datafile will be in `structures/`.

## Run Simulation
Move to `lammps_script` and run `mpirun -np 4 lmp_bin -i plane_MP.lmp -v ktilt 12.0 -v ksplay 1.0 -v rcut 2.5 -v wc 2.0 -v zeta 5.0`



# ISSUES
The membrane is stable but it seems to be some issues with the initialization of Metaparticles. In particular lots of warning for FENE bonds. Maybe I missed some parameters, check that!

# How to collab on Github
1. Create Your Branch
Never work directly on the main branch. Create a new branch.

```Bash
git checkout -b feature/your-feature-name
```

2. Commit Your Changes
As you work, save your progress with clear, concise commit messages.

```Bash
git add .
git commit -m "Brief description of what you changed"
git push
```
