import numpy as np
import matplotlib.pyplot as plt
from icosphere import icosphere
from sklearn.neighbors import BallTree
from scipy.spatial.distance import pdist

def write_positions(path,atom_positions, face_normals, pred_radius):
    """
    Write the atom positions to a initial configuration LAMMPS data file.
    """

    # atom-ID atom-type x y z diameter density q mux muy muz molecule-ID

    atomtype = 1 # atom type
    # density = 1.0 / (4.0/3.0*np.pi)
    density = 1.0 # density of the particles
    diameter = 1.0 # diameter of the particles
    mol_id = 1 # molecule ID
    q = 1.0 # charge
    lammps_data = f"""\
    Radius equal to {pred_radius:.6f}

    {len(atom_positions)} atoms

    1 atom types

    {- atom_positions.max(0)[0]*2.0:.6f} {atom_positions.max(0)[0]*2.0:.6f} xlo xhi
    {- atom_positions.max(0)[1]*2.0:.6f} {atom_positions.max(0)[1]*2.0:.6f} ylo yhi
    {- atom_positions.max(0)[2]*2.0:.6f} {atom_positions.max(0)[2]*2.0:.6f} zlo zhi


    Atoms \n
    """
    # atom-ID atom-type x y z diameter density q mux muy muz molecule-ID

    # Append atom positions and normals for orientations to the LAMMPS data
    for i in range(atom_positions.shape[0]):
        atom_id = i + 1  # LAMMPS atom IDs start from 1
        lammps_data += f"{atom_id} {atomtype} {atom_positions[i,0]:.6f} {atom_positions[i,1]:.6f} {atom_positions[i,2]:.6f} {diameter} {density} {q} {face_normals[i,0]} {face_normals[i,1]} {face_normals[i,2]} {mol_id}\n"

    # Write the LAMMPS data to a file
    with open(f"{path}/sphere_{pred_radius:.1f}_dist_{target_dist}.lammps", "w") as file:
        file.write(lammps_data)

    print(f"LAMMPS data file 'sphere_{pred_radius:.1f}_dist_{target_dist}.lammps' has been generated.")

NOISE = False
path = '.' # current directory

# python spherical_vesicle.py


print("Choose the iteration value according to the sphere size or number of particles to generate")
print("The following values are an example:")

for nu in np.arange(5, 30, 5):
    nr_vertex = 12 + 10 * (nu**2 - 1)
    nr_face = 20 * nu**2
    print(f"Iteration: {nu}, N particles: {nr_face}, pred radius: {nu / 1.1:.1f}")


nu = int(input("Iteration number: "))

vertices, faces = icosphere(nu)
triangles = vertices[faces]

if NOISE:
    triangles[:,0] = triangles[:,0] + 0.3 * np.random.randn()
    triangles[:,1] = triangles[:,1] + 0.3 * np.random.randn()
    triangles[:,2] = triangles[:,2] + 0.3 * np.random.randn()


# calculate face normals
face_normals = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                        vertices[faces[:, 2]]-vertices[faces[:, 0]])
face_normals /= np.sqrt(np.sum(face_normals**2, axis=1, keepdims=True))

# calculate the center of mass of the triangles
cm = (triangles[:, 0, :] + triangles[:, 1, :] + triangles[:, 2, :]) / 3
vertices = cm  # Use the center of mass of the triangle as position for the particle

# calculate the mean distance between the particles
tree = BallTree(vertices)
dist_neighbours, index = tree.query(vertices, k=6)
# np.mean(dist_neighbours[0], axis=1)
mean_dist = np.mean(dist_neighbours)
print(f"Mean dist: {mean_dist}")

cm_sphere = np.mean(vertices,axis=0)
print(f"Center of mass of the sphere",cm_sphere)
# scale the particles to the target distance, this is something that can be adjusted according to the model
# 0.8 works for Yuan's model
target_dist = 0.8



scale_factor = target_dist / mean_dist
print(f"Scale factor: {scale_factor}")

new_vertices = vertices * scale_factor

cm_sphere = np.mean(new_vertices,axis=0)
print(f"Center of mass of the sphere",cm_sphere)
tree = BallTree(new_vertices)
dist_neighbours, index = tree.query(new_vertices, k=6)
mean_dist = np.mean(dist_neighbours)
print(f"Mean dist: {mean_dist}")


plt.plot(vertices[:, 0], vertices[:, 1], 'bo')
plt.plot(new_vertices[:, 0], new_vertices[:, 1], 'ro')
plt.axis('equal')
plt.show()


pred_radius = np.max(pdist(vertices, metric='euclidean')) / 2.0
print(f"Old sphere radius: {pred_radius}")

vertices = new_vertices


pred_radius = np.max(pdist(vertices, metric='euclidean')) / 2.0
print(f"New scaled radius: {pred_radius}")

write_positions(path,vertices, face_normals, pred_radius)

# render_curvature_pv(vertices)


###########################################################
# The following part is to generate spheres with a for loop sequence, if you need to generate a bunch of them ########
###########################################################

# radius_list = []
# nu_list = np.arange(40, 60, 5)
# factor_list = []

# for nu in nu_list:
#     print(nu)
#     vertices, faces = icosphere(nu)
#     triangles = vertices[faces]

#     face_normals = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
#                             vertices[faces[:, 2]]-vertices[faces[:, 0]])
#     face_normals /= np.sqrt(np.sum(face_normals**2, axis=1, keepdims=True))

#     cm = triangles[:, 0, :] + triangles[:, 1, :] + triangles[:, 2, :] / 3
#     vertices = cm  # uso il cm del triangle come posizione per la particle


#     tree = BallTree(vertices)
#     dist_neighbours, index = tree.query(vertices, k=6)
#     # np.mean(dist_neighbours[0], axis=1)
#     mean_dist = np.mean(dist_neighbours)
#     target_dist = 0.8

#     scale_factor = target_dist / mean_dist
#     print(f"Scale factor: {scale_factor}")

#     new_vertices = vertices * scale_factor
#     factor_list.append(scale_factor)

#     vertices = new_vertices



#     pred_radius = np.max(pdist(vertices, metric='euclidean')) / 2.0
#     print(f"Radius: {pred_radius}")
#     radius_list.append(pred_radius)


#     write_positions(path,vertices, face_normals, pred_radius)


# plt.plot(nu_list,radius_list,'bo')
# plt.plot(nu_list,nu_list,'k--')
# plt.show()


# plt.plot(nu_list,factor_list,'bo')
# plt.show()


# plt.plot(radius_list,factor_list,'bo')
# plt.show()