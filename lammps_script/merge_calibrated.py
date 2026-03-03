"""
Merge calibrated membrane positions/dipoles into the original data file.
Reads calibrated_membrane.data (from write_data) and the original structure,
replaces membrane atom coords+dipoles, writes merged_structure.data.

Original format:  id type x y z diam dens ??? mux muy muz mol
Calibrated format: id type x y z diam dens mol mux muy muz mol ix iy iz
We extract x y z mux muy muz from calibrated and patch into original format.
"""

ORIGINAL = "../structures/sphere_16.3_dist_0.8.lammps_60.data"
CALIBRATED = "calibrated_membrane.data"
OUTPUT = "merged_structure.data"

def parse_calibrated_atoms(filepath):
    """Parse calibrated data file, extract id -> (x, y, z, mux, muy, muz)."""
    atoms = {}
    in_atoms = False
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("Atoms"):
                in_atoms = True
                continue
            if in_atoms and stripped == "":
                if atoms:
                    break
                continue
            if in_atoms and stripped:
                cols = stripped.split()
                atom_id = int(cols[0])
                # write_data hybrid format: id type x y z diam dens mol mux muy muz ...
                x, y, z = cols[2], cols[3], cols[4]
                mux, muy, muz = cols[8], cols[9], cols[10]
                atoms[atom_id] = (x, y, z, mux, muy, muz)
    return atoms

cal = parse_calibrated_atoms(CALIBRATED)
print(f"Read {len(cal)} calibrated membrane atoms")

# Read original, replace membrane atom positions and dipoles
in_atoms = False
past_blank = False
with open(ORIGINAL) as f, open(OUTPUT, "w") as out:
    for line in f:
        stripped = line.strip()
        if stripped == "Atoms":
            in_atoms = True
            past_blank = False
            out.write(line)
            continue
        if in_atoms and stripped == "":
            if past_blank:
                in_atoms = False
            else:
                past_blank = True
            out.write(line)
            continue
        if in_atoms and stripped:
            past_blank = True
            cols = stripped.split()
            atom_id = int(cols[0])
            if atom_id in cal:
                x, y, z, mux, muy, muz = cal[atom_id]
                # Original format: id type x y z diam dens ??? mux muy muz mol
                cols[2] = x
                cols[3] = y
                cols[4] = z
                cols[8] = mux
                cols[9] = muy
                cols[10] = muz
                out.write(" ".join(cols) + " \n")
            else:
                out.write(line)
            continue
        out.write(line)

print(f"Written merged structure to {OUTPUT}")
