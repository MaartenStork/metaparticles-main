# New potential for particle based membrane simulations
This repository contains the new potential for particle based membrane simulations. The potential has been implemented in LAMMPS following the general guidelines for a pair interaction (LAMMPS pair style).


# LAMMPS building instructions
1. Download and unpack with
```
wget --no-check-certificate https://download.lammps.org/tars/lammps-stable.tar.gz
tar -xvf lammps-stable.tar.gz
cd lammps-29Aug2024/
```

2. Move `pair_membrane_sillano_v2.cpp` and `pair_membrane_sillano_v2.h` files from `lammps_file/cpp_files`
 in `src` lammps directory. These files are the new potential and enable the pair_style 'membrane_sillanov2' in LAMMPS.

3. Move `compute_pressure.cpp` and `compute_pressure.h` in `src` overwriting the already existing files. This add an additional parameter control in the pressure computing by LAMMPS. Used when we apply a barostat to the simulation.

4. Install or load required compilation tools
```
apt-get install cmake openmpi # if you are using linux/ubuntu/WSL
```
```
module load 2024r1 cmake openmpi # if you are using DelftBlue
```
5. Compile
```
mkdir build
cd build
ccmake  -D BUILD_MPI=yes -D PKG_BROWNIAN=yes -D PKG_MOLFILE=yes -D PKG_EXTRA-PAIR=yes -D PKG_MOLECULE=yes -D PKG_DIPOLE=yes ../cmake
make -j8
make install
```

# Self assembly run
You can run a simulation with `mpirun -np 4 lmp -i self_assembly/self_assembly.lmp -v ktilt 7.0 -v ksplay 7.0 -v N 400 -v rcut 3.0 -v wc 3.0`

Most of simulation produce a `position.lammpstrj` trajectoy file that can be visualized with OVITO.
