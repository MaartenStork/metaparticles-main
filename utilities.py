import numpy as np
# NOTES
# Atom type for MP should be Bond (minimal)
# Atom type for Membrane should be Sphere Dipole
# Atom type together should be Bond Sphere Dipole


# We have Dump and Data reader for each combination of only membrane, only MP, Mem + MP

def read_MP_membrane_data(filename,WRAP=True):
    # this one can read a Bond Sphere Dipole data file
    fh = open(filename,'r')
    fh.readline()
    N_ATOMS,_ = fh.readline().strip()
    N_BONDS,_ = fh.readline().strip().split()
    N_ATOMS = int(N_ATOMS)
    N_BONDS = int(N_BONDS)
    fh.readline()
    fh.readline()
    fh.readline()
    xlo, xhi = fh.readline().strip().split()
    ylo, yhi = fh.readline().strip().split()
    zlo, zhi = fh.readline().strip().split()

    xlo = float(xlo)
    ylo = float(ylo)
    zlo = float(zlo)
    xhi = float(xhi)
    yhi = float(yhi)
    zhi = float(zhi)

    Lx = xhi - xlo
    Ly = yhi - ylo
    Lz = zhi - zlo
    
    coords = np.empty((N_ATOMS, 3), dtype=np.float64) # type: ignore
    bonds  = np.empty((N_BONDS, 3), dtype=np.float64) # type: ignore
     
    mu = np.empty((N_ATOMS, 3), dtype=np.float64) # type: ignore
    atom_type = np.empty(N_ATOMS) 
    mol_id =  np.empty(N_ATOMS)
    fh.readline()

    for i in range(N_ATOMS):
        s = fh.readline().strip().split()
        if len(s) != 12:
            print(s)
            raise ValueError('Unexpected number of columns in the dump file \n the correct columns are: \n id type molid x y z ix iy iz mux muy muz')

        if len(s)==12:
            idx = int(s[0])
            atom_type[idx-1] = int(s[1])
            mol_id[idx-1] = int(s[2])
            
            x = float(s[3])
            y = float(s[4])
            z = float(s[5])

            mu[idx-1,0] = float(s[9])
            mu[idx-1,1] = float(s[10])
            mu[idx-1,2] = float(s[11])
        
        if WRAP:
            # already wrapped coordinates    
            x = (x - xlo) % Lx + xlo
            y = (y - ylo) % Ly + ylo
            z = (z - zlo) % Lz + zlo
        
        coords[idx-1,0] = x
        coords[idx-1,1] = y
        coords[idx-1,2] = z

    fh.readline()
    fh.readline()
    fh.readline()
    
    for i in range(N_BONDS):
        s = fh.readline().strip().split()
        idx = int(s[0])
        bonds[idx-1,0]  = int(s[2])
        bonds[idx-1,1]  = int(s[3])
    
    fh.close()
    return atom_type,mol_id,coords,mu

def read_membrane_dump(filename,WRAP=True):
    # this one can read a Bond Sphere Dipole dump file
    fh = open(filename,'r')
    fh.readline()
    fh.readline()
    t = int(fh.readline().strip())
    fh.readline()
    N_ATOMS = int(fh.readline().strip())
    fh.readline()
    xlo, xhi = fh.readline().strip().split()
    ylo, yhi = fh.readline().strip().split()
    zlo, zhi = fh.readline().strip().split()

    xlo = float(xlo)
    ylo = float(ylo)
    zlo = float(zlo)
    xhi = float(xhi)
    yhi = float(yhi)
    zhi = float(zhi)

    Lx = xhi - xlo
    Ly = yhi - ylo
    Lz = zhi - zlo
    
    Lbox = [(xlo,xhi),(ylo,yhi),(zlo,zhi)]

    
    coords = np.empty((N_ATOMS, 3), dtype=np.float64) # type: ignore  
    mu = np.empty((N_ATOMS, 3), dtype=np.float64) # type: ignore
    
    fh.readline()

    for i in range(N_ATOMS):
        s = fh.readline().strip().split()
        if len(s) != 11:
            print(s)
            raise ValueError('Unexpected number of columns in the dump file \n the correct columns are: \n id type molid x y z ix iy iz mux muy muz')

        idx = int(s[0])
        
        x = float(s[2])
        y = float(s[3])
        z = float(s[4])

        mu[idx-1,0] = float(s[8])
        mu[idx-1,1] = float(s[9])
        mu[idx-1,2] = float(s[10])
        
        if WRAP:
            # already wrapped coordinates    
            x = (x - xlo) % Lx + xlo
            y = (y - ylo) % Ly + ylo
            z = (z - zlo) % Lz + zlo
        
        coords[idx-1,0] = x
        coords[idx-1,1] = y
        coords[idx-1,2] = z
    
    fh.close()
    
    return coords, mu, Lbox

def read_membrane_data(filename,WRAP=True):
    
    # this one can read a Bond Sphere Dipole data file
    # this is usually the files I generated with lammps_input_configuration scripts
    # not sure if it works with the ovito generated one or the one saved from lammps
    fh = open(filename,'r')
    fh.readline()
    fh.readline()

    N_ATOMS,_ = fh.readline().strip().split(sep=' ')
    N_ATOMS = int(N_ATOMS)
    print(fh.readline())
    print(fh.readline())
    print(fh.readline())


    xlo, xhi,_,_ = fh.readline().strip().split()
    ylo, yhi,_,_ = fh.readline().strip().split()
    zlo, zhi,_,_ = fh.readline().strip().split()

    xlo = float(xlo)
    ylo = float(ylo)
    zlo = float(zlo)
    xhi = float(xhi)
    yhi = float(yhi)
    zhi = float(zhi)

    Lx = xhi - xlo
    Ly = yhi - ylo
    Lz = zhi - zlo
       
    Lbox = [(xlo,xhi),(ylo,yhi),(zlo,zhi)]

    
    coords = np.empty((N_ATOMS, 3), dtype=np.float64) # type: ignore  
    mu = np.empty((N_ATOMS, 3), dtype=np.float64) # type: ignore
    
    fh.readline()
    fh.readline()
    fh.readline()
    fh.readline()


    for i in range(N_ATOMS):
        s = fh.readline().strip().split()
        if len(s) != 12:
            raise ValueError('Unexpected number of columns in the dump file \n the correct columns are: \n id type molid x y z ix iy iz mux muy muz')

        idx = int(s[0])
        
        x = float(s[2])
        y = float(s[3])
        z = float(s[4])

        mu[idx-1,0] = float(s[8])
        mu[idx-1,1] = float(s[9])
        mu[idx-1,2] = float(s[10])
        
        if WRAP:
            # already wrapped coordinates    
            x = (x - xlo) % Lx + xlo
            y = (y - ylo) % Ly + ylo
            z = (z - zlo) % Lz + zlo
        
        coords[idx-1,0] = x
        coords[idx-1,1] = y
        coords[idx-1,2] = z
    
    fh.close()
    
    return coords, mu, Lbox

def read_MP_membrane_dump(filename,WRAP=True):
    # this one can read a Bond Sphere Dipole dump file
    fh = open(filename,'r')
    fh.readline()
    t = int(fh.readline().strip())
    fh.readline()
    N_ATOMS = int(fh.readline().strip())
    fh.readline()
    xlo, xhi = fh.readline().strip().split()
    ylo, yhi = fh.readline().strip().split()
    zlo, zhi = fh.readline().strip().split()

    xlo = float(xlo)
    ylo = float(ylo)
    zlo = float(zlo)
    xhi = float(xhi)
    yhi = float(yhi)
    zhi = float(zhi)

    Lx = xhi - xlo
    Ly = yhi - ylo
    Lz = zhi - zlo
    Lbox = [(xlo,xhi),(ylo,yhi),(zlo,zhi)]

    
    coords = np.empty((N_ATOMS, 3), dtype=np.float64) # type: ignore  
    mu = np.empty((N_ATOMS, 3), dtype=np.float64) # type: ignore
    
    fh.readline()

    for i in range(N_ATOMS):
        s = fh.readline().strip().split()
        if len(s) != 12:
            print(s)
            raise ValueError('Unexpected number of columns in the dump file \n the correct columns are: \n id type molid x y z ix iy iz mux muy muz')

        if len(s)==12:
            idx = int(s[0])
            
            x = float(s[3])
            y = float(s[4])
            z = float(s[5])

            mu[idx-1,0] = float(s[9])
            mu[idx-1,1] = float(s[10])
            mu[idx-1,2] = float(s[11])
        
        if WRAP:
            # already wrapped coordinates    
            x = (x - xlo) % Lx + xlo
            y = (y - ylo) % Ly + ylo
            z = (z - zlo) % Lz + zlo
        
        coords[idx-1,0] = x
        coords[idx-1,1] = y
        coords[idx-1,2] = z
    
    fh.close()
    
    return coords,mu

def read_MP_data(infile):
    # This function is to read Massimiliano data file and save the bare minimum to initialize the structures
    fh = open(infile,'r')
    fh.readline()
    fh.readline()
    N_ATOMS,_ = fh.readline().strip().split()
    N_BONDS,_ = fh.readline().strip().split()
    N_ATOMS = int(N_ATOMS)
    N_BONDS = int(N_BONDS)
    fh.readline()
    fh.readline()
    fh.readline()

    xlo, xhi, _, _ = fh.readline().strip().split()
    ylo, yhi, _, _  = fh.readline().strip().split()
    zlo, zhi, _, _  = fh.readline().strip().split()

    xlo = float(xlo)
    ylo = float(ylo)
    zlo = float(zlo)
    xhi = float(xhi)
    yhi = float(yhi)
    zhi = float(zhi)

    Lx = xhi - xlo
    Ly = yhi - ylo
    Lz = zhi - zlo
    Lbox = [(xlo,xhi),(ylo,yhi),(zlo,zhi)]
    
    coords = np.empty((N_ATOMS, 3), dtype=np.float64) # type: ignore
    bonds  = np.empty((N_BONDS,2),dtype=np.int32)

    mol_id =  np.empty(N_ATOMS)
    fh.readline()
    fh.readline() # MASSES
    fh.readline()
    fh.readline()
    fh.readline()
    fh.readline()
    fh.readline()

    for i in range(N_ATOMS):
        s = fh.readline().strip().split()
        
        idx = int(s[0])
        
        coords[idx-1,0]  = float(s[2])
        coords[idx-1,1]  = float(s[3])
        coords[idx-1,2]  = float(s[4])

        
    fh.readline()
    fh.readline()
    fh.readline()
    
    for i in range(N_BONDS):
        s = fh.readline().strip().split()
        idx = int(s[0])
        bonds[idx-1,0]  = int(s[2])
        bonds[idx-1,1]  = int(s[3])
        
    return coords, bonds, Lbox

def read_MP_data_v2(infile):
    # This function is to read Massimiliano data file ellipsoids version
    fh = open(infile,'r')
    print(infile)
    fh.readline()
    fh.readline()
    N_ATOMS,_ = fh.readline().strip().split()
    fh.readline()
    N_BONDS,_ = fh.readline().strip().split()
    N_ATOMS = int(N_ATOMS)
    N_BONDS = int(N_BONDS)
    fh.readline()
    fh.readline()
    fh.readline()


    xlo, xhi, _, _ = fh.readline().strip().split()
    ylo, yhi, _, _  = fh.readline().strip().split()
    zlo, zhi, _, _  = fh.readline().strip().split()

    xlo = float(xlo)
    ylo = float(ylo)
    zlo = float(zlo)
    xhi = float(xhi)
    yhi = float(yhi)
    zhi = float(zhi)

    Lx = xhi - xlo
    Ly = yhi - ylo
    Lz = zhi - zlo
    Lbox = [(xlo,xhi),(ylo,yhi),(zlo,zhi)]
    
    coords = np.empty((N_ATOMS, 3), dtype=np.float64) # type: ignore
    bonds  = np.empty((N_BONDS,2),dtype=np.int16)

    for i in range(15):
        fh.readline()


    for i in range(N_ATOMS):
        s = fh.readline().strip().split()
        idx = int(s[0])
        coords[idx-1,0]  = float(s[2])
        coords[idx-1,1]  = float(s[3])
        coords[idx-1,2]  = float(s[4])

        
    fh.readline()
    fh.readline()
    fh.readline()
    
    for i in range(N_ATOMS+3): #velocities section
        fh.readline()


    for i in range(N_BONDS):
        s = fh.readline().strip().split()
        idx = int(s[0])
        bonds[idx-1,0]  = int(s[2])
        bonds[idx-1,1]  = int(s[3])
        
    return coords, bonds, Lbox

def write_MP_data_only(outfile,coords,bonds,Lbox):

    # This one write MP bond type only
    # atom-ID atom-type x y z molecule-ID

    lammps_data = f"""
    {coords.shape[0]} atoms
    {bonds.shape[0]} bonds
    1 atom types
    1 bond types

    {Lbox[0][0]:.6f} {Lbox[0][1]:.6f} xlo xhi
    {Lbox[1][0]:.6f} {Lbox[1][1]:.6f} ylo yhi
    {Lbox[2][0]:.6f} {Lbox[2][1]:.6f} zlo zhi

    Atoms \n
    """
    atom_type = 1
    mol_id = 1
    for i in range(coords.shape[0]):
        lammps_data += f"{i+1:d} {atom_type:d} {coords[i,0]:.6f} {coords[i,1]:.6f} {coords[i,2]:.6f} {mol_id:d} \n"

    lammps_data += f"""\n Bonds \n
    """
    
    bond_type = 1
    for i in range(bonds.shape[0]):
        lammps_data += f"{i+1:d} {bond_type:d} {bonds[i,0]:d} {bonds[i,1]:d}\n"
    
    # Write the LAMMPS data to a file
    with open(outfile, "w") as file:
        file.write(lammps_data)

    print(f"LAMMPS data file {outfile} has been generated.")
       
def adjust_box_to_fit_coords(MP_coords, box, padding=5):
    """
    Ensure an N×3 coordinate array fits inside the given box.
    If not, expand the affected bounds by `padding`.

    Parameters
    ----------
    MP_coords : np.ndarray
        Array of shape (N, 3).
    box : list of tuples
        [(xlo, xhi), (ylo, yhi), (zlo, zhi)].
    padding : float
        Value to expand the box when exceeded.

    Returns
    -------
    new_box : list of tuples
        Adjusted box bounds.
    """

    # Current bounds
    xlo, xhi = box[0]
    ylo, yhi = box[1]
    zlo, zhi = box[2]

    # Coordinate mins/maxes
    mins = MP_coords.min(axis=0)
    maxs = MP_coords.max(axis=0)

    # Adjust
    if mins[0] < xlo:
        xlo = mins[0] - padding
    if maxs[0] > xhi:
        xhi = maxs[0] + padding

    if mins[1] < ylo:
        ylo = mins[1] - padding
    if maxs[1] > yhi:
        yhi = maxs[1] + padding

    if mins[2] < zlo:
        zlo = mins[2] - padding
    if maxs[2] > zhi:
        zhi = maxs[2] + padding

    return [(xlo, xhi), (ylo, yhi), (zlo, zhi)]

def combine_structures(mem_file,read_function,N_MP,shift):
    # this function can combine MPs with membrane structure

    # open equilibrated plane or equilibrated sphere
    # read file using standard sphere dipole reader
    mem_coords, mem_mu, mem_Lbox = read_function(mem_file,WRAP=True)
    
    # read MP structure
    if N_MP==20 or N_MP==36:
        MP_coords, MP_bonds, MP_Lbox = read_MP_data(f"MP_datafile_massimiliano/MP{N_MP}_eq.data")    
    else:
        MP_coords, MP_bonds, MP_Lbox = read_MP_data_v2(f"MP_datafile_massimiliano/MP{N_MP}_eq.data")    
    
    # center MP coords in the box
    cm = np.mean(MP_coords,axis=0)
    MP_coords = MP_coords - cm
    
    # shift MP coords
    MP_coords = MP_coords + shift
    
    # Offset bond ids
    MP_bonds = np.add(MP_bonds,len(mem_coords))
    
    # assegn types - each MP bead gets its own type (2, 3, 4, ..., N_MP+1)
    mem_type = np.ones(mem_coords.shape[0], dtype=int)
    MP_type = np.arange(2, 2 + MP_coords.shape[0], dtype=int)
    
    # dummy orientation for MPs
    MP_mu = np.ones_like(MP_coords)
    
    # merge Lboxes (and enlarge if needed)
    Lbox = adjust_box_to_fit_coords(MP_coords, mem_Lbox, padding=5)
    
    # combine arrays
    tot_coords = np.vstack((mem_coords,MP_coords))
    tot_mu = np.vstack((mem_mu,MP_mu))
    tot_type = np.hstack((mem_type,MP_type))
    tot_mol_id = np.hstack((mem_type,MP_type))
    
    # print(tot_coords.shape,tot_mu.shape,tot_type.shape,tot_mol_id.shape)
    return tot_type,tot_mol_id,tot_coords,tot_mu,MP_bonds,Lbox

def combine_multiple_structures(mem_file,read_function,structure_list,shift_list):
    # this function can combine MPs with membrane structure

    # open equilibrated plane or equilibrated sphere
    # read file using standard sphere dipole reader
    mem_coords, mem_mu, mem_Lbox = read_function(mem_file,WRAP=True)
    mem_type = np.ones(mem_coords.shape[0], dtype=int)

    tot_coords = mem_coords 
    tot_mu = mem_mu
    tot_type = mem_type
    tot_mol_id = mem_type
    tot_bonds = np.empty((0, 2),dtype=np.int32)
    mol_idx = 1
    
    if len(structure_list) != len(shift_list):
        raise ValueError('Number of structures and number of initial positions does not match')   
    
     
    for file,shift in zip(structure_list,shift_list):
        
        if file=="MP_datafile_massimiliano/MP20_eq.data" or file=="MP_datafile_massimiliano/MP36_eq.data":
            MP_coords, MP_bonds, MP_Lbox = read_MP_data(file)    
        else:
            MP_coords, MP_bonds, MP_Lbox = read_MP_data_v2(file)    
        
        # center MP coords in the box
        cm = np.mean(MP_coords,axis=0)
        MP_coords = MP_coords - cm
        
        # shift MP coords
        MP_coords = MP_coords + np.array(shift, dtype=float)
    
        # Offset bond ids
        MP_bonds = np.add(MP_bonds,len(tot_coords))
        
        
        # offset mol_id index
        mol_idx = mol_idx + 1
        MP_id = mol_idx*np.ones(MP_coords.shape[0], dtype=int)
        
        # assegn types - each MP bead gets its own type, continuing from last used type
        next_type = int(tot_type.max()) + 1
        MP_type = np.arange(next_type, next_type + MP_coords.shape[0], dtype=int)
        
        # dummy orientation for MPs
        MP_mu = np.ones_like(MP_coords)
        
        # combine arrays after each MP addition
        tot_coords = np.vstack((tot_coords,MP_coords))
        tot_bonds  = np.vstack((tot_bonds,MP_bonds))
        tot_mu = np.vstack((tot_mu,MP_mu))
        tot_type = np.hstack((tot_type,MP_type))
        tot_mol_id = np.hstack((tot_mol_id,MP_id))
             

    
    # merge Lboxes (and enlarge if needed)
    Lbox = adjust_box_to_fit_coords(tot_coords, mem_Lbox, padding=5)
        
    # print(tot_coords.shape,tot_mu.shape,tot_bonds.shape,tot_type.shape,tot_mol_id.shape)
    return tot_type,tot_mol_id,tot_coords,tot_mu,tot_bonds,Lbox

def write_structures_data(outfile,tot_type,tot_mol_id,tot_coords,tot_mu,MP_bonds,Lbox):

    # This one writes Membrane + MP lammps data input

    # atom-ID atom-type x y z diameter density q mux muy muz molecule-ID
    density = 1.0
    diameter = 1.0
    q = 1.0 # charge


    n_atom_types = int(tot_type.max())
    lammps_data = f"""

    {len(tot_coords)} atoms
    {len(MP_bonds)} bonds
    {n_atom_types} atom types
    1 bond types
    
    {Lbox[0][0]:.6f} {Lbox[0][1]:.6f} xlo xhi
    {Lbox[1][0]:.6f} {Lbox[1][1]:.6f} ylo yhi
    {Lbox[2][0]:.6f} {Lbox[2][1]:.6f} zlo zhi


    Atoms \n
    """
    # atom-ID atom-type x y z diameter density q mux muy muz molecule-ID

    # Append atom positions to the LAMMPS data
    idx = 1
    for i in range(tot_coords.shape[0]):
        lammps_data += f"{idx:d} {tot_type[i]:d} {tot_coords[i,0]:.6f} {tot_coords[i,1]:.6f} {tot_coords[i,2]:.6f} {diameter} {density} {q} {tot_mu[i,0]:.6f} {tot_mu[i,1]:.6f} {tot_mu[i,2]:.6f} {int(tot_mol_id[i]):d} \n"
        idx = idx + 1

    lammps_data += f"""\n Bonds \n
    """
    
    bond_type = 1
    for i in range(MP_bonds.shape[0]):
        lammps_data += f"{i+1:d} {bond_type:d} {MP_bonds[i,0]:d} {MP_bonds[i,1]:d}\n"
        
    with open(outfile, "w") as file:
        file.write(lammps_data)

    print(f"LAMMPS data file {outfile} has been generated.")

def main():
    return 0
    
    
