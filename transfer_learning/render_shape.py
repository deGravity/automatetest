from train_latent_space import BRepDS, BRepFaceAutoencoder, BRepFaceEncoder
import torch
import numpy as np
import meshplot as mp


def load_model(ckpt_path = 'D:/fusion360segmentation/BRepFaceAutoencoder_64_1024_4.ckpt'):
    model = BRepFaceAutoencoder(64,1024,4)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    return model

def load_test_set(
    index='D:/fusion360segmentation/simple_train_test.json', 
    datadir='D:/fusion360segmentation/simple_preprocessed'
):
    return BRepDS(index, datadir, 'test', preshuffle=False)



def predict(data, model, N=100):
    n_faces = data.face_surfaces.shape[0]
    line = torch.linspace(-0.1,1.1,N)
    grid = torch.cartesian_prod(line, line)
    grids = grid.repeat(n_faces,1)
    indices = torch.arange(n_faces).repeat_interleave(N*N, dim=0)
    with torch.no_grad():
        preds = model(data, grids, indices)
    return preds

def plot_part(p):
    e_i = np.concatenate([
        p.mesh_topology.edge_to_topology[:,1],
        p.mesh_topology.edge_to_topology[:,2],
        p.mesh_topology.edge_to_topology[:,0]
    ])
    e_i = e_i[e_i >= 0]
    f_i = p.mesh_topology.face_to_topology
    e = np.concatenate([
        p.mesh.F[:,[0,1]][(p.mesh_topology.edge_to_topology[:,1] >= 0)],
        p.mesh.F[:,[1,2]][(p.mesh_topology.edge_to_topology[:,2] >= 0)],
        p.mesh.F[:,[2,0]][(p.mesh_topology.edge_to_topology[:,0] >= 0)]
    ],axis=0)
    plot = mp.plot(p.mesh.V, p.mesh.F, return_plot=True)
    plot.add_edges(p.mesh.V, e)

def plot_part_data(m, grid_pred, interior = True):
    mask = (grid_pred[:,-1] <= 0)#(grid_pred[:,:,-1] <= 0).flatten()

    positions = grid_pred[:,:3]#.reshape((-1,3))
    dists = grid_pred[:,-1]#.flatten()

    masked_positions = positions[mask] if interior else positions
    masked_dists = dists[mask] if interior else dists

    plot = mp.plot(masked_positions.numpy(), c=masked_dists.numpy(), shading={'point_size':0.1}, return_plot=True)
    plot_edges(m, plot)

def plot_edges(m, plot):
    all_lines = []
    all_points = []
    offset = 0
    for curve in m.curve_samples:
        lines = [(i+offset,i+1+offset) for i in range(len(curve) - 1)]
        lines = np.array(lines)
        all_lines.append(lines)
        all_points.append(curve[:,:3])
        offset += curve.shape[0]
    points = np.concatenate(all_points, 0)
    lines = np.concatenate(all_lines,0)
    plot.add_edges(points, lines)
def plot_part_data(m, grid_pred, interior = True):
    mask = (grid_pred[:,-1] <= 0)#(grid_pred[:,:,-1] <= 0).flatten()

    positions = grid_pred[:,:3]#.reshape((-1,3))
    dists = grid_pred[:,-1]#.flatten()

    masked_positions = positions[mask] if interior else positions
    masked_dists = dists[mask] if interior else dists

    plot = mp.plot(masked_positions.numpy(), c=masked_dists.numpy(), shading={'point_size':0.1}, return_plot=True)
    plot_edges(m, plot)

def preds_to_mesh(preds, N):
    num_faces = int(preds.shape[0] / (N**2))
    v = lambda f,i,j: f*N**2+i*N+j
    tris = np.array([
        [
            [ v(f,i,j),    v(f,i,j+1),    v(f,i+1,j+1) ],
            [ v(f,i,j+1),   v(f,i+1,j+1), v(f,i+1,j)   ],
            [ v(f,i+1,j+1), v(f,i+1,j),   v(f,i,j)     ],
            [ v(f,i+1,j),   v(f,i,j),     v(f,i,j+1)   ]
        ]
        for f in range(num_faces) for i in range(N-1) for j in range(N-1) 
    ]).reshape((-1,3))

    indices = torch.arange(num_faces).repeat_interleave(N*N, dim=0).numpy()

    verts = preds.numpy()[:,:3]
    dists = preds.numpy()[:,-1]

    tri_faces = indices[tris].max(axis=1)
    tri_mask = np.all(dists[tris] <= 0, axis=1)

    return verts, tris[tri_mask], tri_faces[tri_mask]