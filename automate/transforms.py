import functools
import torch
from scipy.spatial.transform import Rotation as R

def helper_add_point_cloud(n, v, f, use_normals=False):
    #print(f.shape)
    v0 = v[f[0,:]]
    v1 = v[f[1,:]]
    v2 = v[f[2,:]]

    normals = (v1 - v0).cross(v2 - v0)
    area = normals.norm(dim=1)
    normals /= area.unsqueeze(-1)

    sum_area = area.sum()#dim=1)
    p_area = area / sum_area
    tris = p_area.multinomial(n, replacement=True)

    v0 = v0[tris]
    v1 = v1[tris]
    v2 = v2[tris]

    u = torch.rand(n, 1)
    v = torch.rand(n, 1)
    bounds_check = u + v > 1
    u[bounds_check] = 1 - u[bounds_check]
    v[bounds_check] = 1 - v[bounds_check]
    w = 1 - (u + v)

    pc = u * v0 + v * v1 + w * v2

    if use_normals:
        return pc.float(), normals[tris].float(), tris
    else:
        return pc.float(), tris


def motion_points(points, normals, axis, origin, motion_type, displacement):
    """
    given a pair of meshes, return another pair of meshes displaced relative to each other based on motion type
    """

    points = points.copy()
    normals = normals.copy()
    #p = mp.plot(*meshes[0])
    if motion_type == 'ROTATE':
        r = R.from_rotvec(axis * displacement)
        mat = torch.from_numpy(r.as_matrix()).float()
        points[0] = (mat @ (points[0] - origin).T).T + origin
        normals[0] = (mat @ normals[0].T).T
    elif motion_type == 'SLIDE':
        points[0] = points[0] + axis * displacement
    else:
        raise ValueError
    return points, normals



def compose(*fs):
    return functools.reduce(lambda f, g: lambda *a, **kw: f(g(*a, **kw)), fs)


def fix_edge_sets(data):
    data.__edge_sets__['flat_topos_to_graph_idx'] = ['graph_idx']
    data.__edge_sets__['mcf_to_graph_idx'] = ['graph_idx']
    return data



def remap_type_labels(data):
    remap_indices = torch.tensor([-1, -1, -1, 0, 1, 2, -1, 3])
    data.mate_labels = remap_indices[data.mate_labels]
    return data


def sample_points(npoints, assembly_points, normalize, combined):
    def _sample_points(data):
        facet_to_part_id = data.flat_topos_to_graph_idx[0][data.face_to_flat_topos[1][data.F_to_faces[0]]]
        if assembly_points:
            assembly_pc, assembly_normals, assembly_tris = helper_add_point_cloud(npoints*2 if combined else npoints, data.V, data.F, use_normals=True)
            point_to_part_idx = facet_to_part_id[assembly_tris]

        allpoints = []
        all_assembly_points = []
        for i in range(data.part_edges.shape[1]):
            pair = data.part_edges[:,i]
            facets_both = [data.F[:,facet_to_part_id == p] for p in pair]
            bothpoints = []
            bothnormals = []
            for p,facets in enumerate(facets_both):
                pcs, normals, tris = helper_add_point_cloud(npoints, data.V, facets, use_normals=True)
                bothpoints.append(pcs)
                bothnormals.append(normals)
            

            if normalize:
                minPt = torch.minimum(bothpoints[0].min(0)[0], bothpoints[1].min(0)[0])
                maxPt = torch.maximum(bothpoints[0].max(0)[0], bothpoints[1].max(0)[0])
                dims = maxPt - minPt
                maxDim = dims.max()
                median = (maxPt + minPt) / 2
                bothpoints[0] -= median
                bothpoints[0] /= maxDim
                bothpoints[1] -= median
                bothpoints[1] /= maxDim

            

            if combined:
                pointnormals = [torch.cat([pt, nt, torch.full((npoints, 1), p)], dim=1) for p,(pt, nt) in enumerate(zip(bothpoints, bothnormals))]
                pointnormals = torch.cat(pointnormals, dim=0)
            else:
                pointnormals = [torch.cat([pt, nt], dim=1) for pt, nt in zip(bothpoints, bothnormals)]
                pointnormals = torch.stack(pointnormals)

            if assembly_points:
            
                assembly_feats = torch.cat([assembly_pc, assembly_normals, torch.full((npoints*2, 1), -1, dtype=torch.float)], dim=1)
                assembly_feats[point_to_part_idx == pair[0],6] = 0
                assembly_feats[point_to_part_idx == pair[1],6] = 1
                all_assembly_points.append(assembly_feats)
                

            allpoints.append(pointnormals)
        
        data.pcs = torch.stack(allpoints)
        if assembly_points:
            data.global_pcs = torch.stack(all_assembly_points)
        return data
    return _sample_points


def sample_motions(npoints, displacement, angle, debug=False):
    def _sample_motions(data):
        facet_to_part_id = data.flat_topos_to_graph_idx[0][data.face_to_flat_topos[1][data.F_to_faces[0]]]
        allpoints = []
        if debug:
            data.pc, _ = helper_add_point_cloud(npoints, data.V, data.F, use_normals=False)
            data.debug_mesh_pairs = []
        for i in range(data.part_edges.shape[1]):
            pair = data.part_edges[:,i]
            mcf = data.mcfs[data.axis_labels[i]]
            axis = mcf[:3]
            origin = mcf[3:]
            facets_both = [data.F[:,facet_to_part_id == p] for p in pair]
            bothpoints = []
            bothnormals = []
            for p,facets in enumerate(facets_both):
                pcs, normals, tris = helper_add_point_cloud(npoints, data.V, facets, use_normals=True)
                bothpoints.append(pcs)
                bothnormals.append(normals)
            
            if debug:
                data.debug_mesh_pairs.append(torch.cat(facets_both, dim=1).T)

            slide_points_0, _ = motion_points(bothpoints, bothnormals, axis, origin, 'SLIDE', displacement=displacement)
            slide_points_1, _ = motion_points(bothpoints, bothnormals, axis, origin, 'SLIDE', displacement=-displacement)
            rotate_points_0, rotate_normals_0 = motion_points(bothpoints, bothnormals, axis, origin, 'ROTATE', displacement=angle)
            rotate_points_1, rotate_normals_1 = motion_points(bothpoints, bothnormals, axis, origin, 'ROTATE', displacement=-angle)

            allpoints_time = [bothpoints, slide_points_0, slide_points_1, rotate_points_0, rotate_points_1]
            allnormals_time = [bothnormals, bothnormals, bothnormals, rotate_normals_0, rotate_normals_1]

            pointnormals = [torch.cat([torch.cat([points, norms, torch.full((npoints, 1), p)], dim=1) for p,(points, norms) in enumerate(zip(bp, bn))], dim=0) for bp, bn in zip(allpoints_time, allnormals_time)]

            pointnormals = torch.stack(pointnormals)

            #point_to_face = data.F_to_faces[tris] #Todo: concatenate face features?
            
            allpoints.append(pointnormals)
        
        data.motion_points = torch.stack(allpoints)
        return data
    return _sample_motions