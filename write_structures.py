import numpy as np
from utilities import *

# TODO list
# - add initial rotation of MPs


mem_path = 'membrane_data/'
MP_path  = 'MP_datafile_massimiliano/'
out_path = 'structures/'



# Single structure in Plane 
mem_file ='planar_d_0.90_N_2125.lammps'
N_MP = 60

shift = np.array([0,0,10])
tot_type,tot_mol_id,tot_coords,tot_mu,MP_bonds,Lbox = combine_structures(mem_path+mem_file,read_membrane_data,N_MP,shift)


outfile = out_path + mem_file + "_" + str(N_MP) + ".data"
write_structures_data(outfile,tot_type,tot_mol_id,tot_coords,tot_mu,MP_bonds,Lbox)


# Single structure in Sphere
mem_file = 'sphere_16.3_dist_0.8.lammps'
N_MP = 60

shift = np.array([0,0,22])
tot_type,tot_mol_id,tot_coords,tot_mu,MP_bonds,Lbox = combine_structures(mem_path + mem_file ,read_membrane_data,N_MP,shift)

outfile = out_path + mem_file + "_" + str(N_MP) + ".data"
write_structures_data(outfile,tot_type,tot_mol_id,tot_coords,tot_mu,MP_bonds,Lbox)


# # Multiple structures with sphere
# structure_list = [MP_path+"MP20_eq.data",MP_path+"MP20_eq.data"]
# shift_list = [[0.0,-35.0,10.0],
#               [0.0,-20.0,10.0]
#               ]
# tot_type,tot_mol_id,tot_coords,tot_mu,MP_bonds,Lbox = combine_multiple_structures(mem_path + mem_file,read_membrane_data,structure_list,shift_list)

# outfile = out_path + mem_file + '_2struc'
# write_structures_data(outfile,tot_type,tot_mol_id,tot_coords,tot_mu,MP_bonds,Lbox)


# Multiple structures with plane
# probably it is better to use N=10000 (100x100) lattice
mem_file ='planar_d_0.90_N_2125.lammps'
structure_list = [MP_path+"MP48_eq.data",MP_path+"MP48_eq.data",MP_path+"MP48_eq.data"]
# equilateral triangle
shift_list = [[-10,0.0,8.0],
              [10.0,0.0,8.0],
              [0,17.32,8.0],
              ]
tot_type,tot_mol_id,tot_coords,tot_mu,MP_bonds,Lbox = combine_multiple_structures(mem_path + mem_file ,read_membrane_data,structure_list,shift_list)

outfile = out_path + mem_file + '_3_MP48'
write_structures_data(outfile,tot_type,tot_mol_id,tot_coords,tot_mu,MP_bonds,Lbox)

