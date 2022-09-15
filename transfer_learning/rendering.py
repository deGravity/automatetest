import numpy as np

import os
import platform
if platform.system() != 'Windows':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

import pyrender
import trimesh
import scipy
from matplotlib import pyplot as plt

import json
from zipfile import ZipFile
from pspy import Part, PartOptions
from tqdm import tqdm


def look_at(point, pos, up):
    z = pos - point
    x = np.cross(up, z)
    y = np.cross(z, x)
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)
    pose = np.eye(4)
    pose[:3,0] = x
    pose[:3,1] = y
    pose[:3,2] = z
    pose[:3,3] = pos
    return pose


def find_best_angle_from_part(part, cubify=True, normalize=True):
    return find_best_angle(part.mesh.V, part.mesh.F, part.mesh_topology.face_to_topology, cubify, normalize)

def find_best_angle(V, F, FtoT, cubify=True, normalize=True):
    if normalize: # assuming centered part for now
        max_coord = np.abs(V).max()
        scale = 1 / max_coord
        V = V * scale
    bb = np.stack([V.min(axis = 0), V.max(axis=0)])
    face_labels = np.tile(FtoT, (3,1)).T.astype(int)
    mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=face_labels)
    pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800, point_size=1.0)
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
    
    c = bb.sum(axis=0) # bounding box center
    
    if cubify:
        s = np.abs(bb - c).max()
        bb = np.array([
            [c[0]-s,c[1]-s,c[2]-s],
            [c[0]+s,c[1]+s,c[2]+s]
        ])
        
        
    side_positions = [
        [c[0],c[1],bb[0,2]*1.1], # front
        [c[0],c[1],bb[1,2]*1.1], # back
        [bb[0,0]*1.1,c[1],c[2]], # left
        [bb[1,0]*1.1,c[1],c[2]], # right
    ]    
    
    top_corner_positions = [
        [bb[0,0],bb[1,1],bb[0,2]], # front top left
        [bb[1,0],bb[1,1],bb[0,2]], # front top right
        [bb[0,0],bb[1,1],bb[1,2]], # back top left
        [bb[1,0],bb[1,1],bb[1,2]], # back top right
    ]
    
    bottom_corner_positions = [
        [bb[0,0],bb[0,1],bb[0,2]], # front bottom left
        [bb[1,0],bb[0,1],bb[0,2]], # front bottom right   
        [bb[0,0],bb[0,1],bb[1,2]], # back bottom left
        [bb[1,0],bb[0,1],bb[1,2]], # back bottom right
    ]
    
    side_top_position = [
        [c[0],bb[1,1],bb[0,2]*1.1], # front top
        [c[0],bb[1,1],bb[1,2]*1.1], # back top
        [bb[0,0]*1.1,bb[1,1],c[2]], # left top
        [bb[1,0]*1.1,bb[1,1],c[2]], # right top
    ]
    
    side_bottom_position = [
        [c[0],bb[0,1],bb[0,2]*1.1], # front bottom
        [c[0],bb[0,1],bb[1,2]*1.1], # back bottom
        [bb[0,0]*1.1,bb[0,1],c[2]], # left bottom
        [bb[1,0]*1.1,bb[0,1],c[2]], # right bottom
    ]
    
    candidate_positions = top_corner_positions

    if cubify:
        s = np.abs(bb - c).max()
        bb = np.array([
            [c[0]-s,c[1]-s,c[2]-s],
            [c[0]+s,c[1]+s,c[2]+s]
        ])
    
        top_corner_positions = [
            [bb[0,0],bb[1,1],bb[0,2]], # front top left
            [bb[1,0],bb[1,1],bb[0,2]], # front top right
            [bb[0,0],bb[1,1],bb[1,2]], # back top left
            [bb[1,0],bb[1,1],bb[1,2]], # back top right
        ]
        
        candidate_positions += top_corner_positions
    
        
    n_visible_faces = []
    image_areas = []
    candidate_poses = []
    zooms = []
    depths = []
    #candidate_renderings = []
    for pos in candidate_positions:
        camera_pose = look_at(c, pos, [0,1,0])
        candidate_poses.append(camera_pose)
        zoom = 1.2 * np.abs(camera_pose[:3,:3].dot(bb.T)).max()
        zooms.append(zoom)
        cam = pyrender.OrthographicCamera(xmag=zoom, ymag=zoom)
        scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
        scene.add(pmesh, pose=np.eye(4))
        scene.add(cam, pose=camera_pose)
        color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES)
        n_visible_faces.append(len(np.unique(color[:,:,0].flatten()[color[:,:,0].flatten() <= face_labels[:,0].max()])))
        image_areas.append((depth > 0).sum())
        depths.append(depth)
        #candidate_renderings.append(color)
    ordering = sorted(enumerate(zip(n_visible_faces, image_areas)), key=lambda x: x[1], reverse=True)
    best_idx = ordering[0][0]
    best_zoom = zooms[best_idx]
    # Rescale to maximize image in frame
    depth = depths[best_idx]
    mask = depth > 0
    mg = np.stack(np.meshgrid(np.linspace(-1,1,800),np.linspace(-1,1,800)),axis=-1)
    zoom_factor = np.abs(mg.reshape((-1,2))[mask.flatten()]).max() * 1.05 # add 5% for some margin
    return candidate_poses[best_idx], zoom_factor * best_zoom

def get_camera_params(index_path, zip_path, split='test'):
    poses = []
    zooms = []
    opts = PartOptions()
    opts.default_mcfs = False
    opts.num_uv_samples = 0
    opts.sample_normals = 0
    opts.sample_tangents = False
    with open(index_path, 'r') as f:
        index = json.load(f)
    parts_list = [index['template'].format(*x) for x in index[split]]
    with ZipFile(zip_path, 'r') as zf:
        for part_path in tqdm(parts_list):
            part = Part(zf.open(part_path).read().decode('utf-8'), opts)
            pose, zoom = find_best_angle_from_part(part, cubify=True)
            poses.append(pose)
            zooms.append(zoom)
    return poses, zooms

def render_part(
        part, camera_pose, zoom, 
        max_labels = None, face_labels=None, cmap='tab20', 
        renderer = None,viewport_width=800, viewport_height=800, point_size=1.0,
        normalize=True
):
    V = part.mesh.V
    if normalize: # assuming centered part for now
        max_coord = np.abs(V).max()
        scale = 1 / max_coord
        V = V * scale
    if not max_labels:
        max_labels = (part.mesh_topology.face_to_topology.max() + 1)
    palette = plt.cm.get_cmap(cmap, lut=max_labels)
    
    tri_labels = part.mesh_topology.face_to_topology
    
    face_colors = (palette(tri_labels)*255).astype(int)
    
    mesh = trimesh.Trimesh(vertices=V, faces=part.mesh.F, face_colors=face_colors)
    pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    
    cam = pyrender.OrthographicCamera(xmag=zoom, ymag=zoom)
    
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
    scene.add(pmesh, pose=np.eye(4))
    scene.add(cam, pose=camera_pose)
    
    if not renderer:
        r = pyrender.OffscreenRenderer(viewport_width=viewport_width, viewport_height=viewport_height, point_size=point_size)
    
    # Rendering an image with independent face colors. Do this first regardless of overridding
    # color labels in order to compute hard edges for edge rendering
    color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES)
    
    # Detect Edges to add to image
    laplace_kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    edge_mask = (
        (scipy.ndimage.convolve(color[:,:,0], laplace_kernel) 
         + scipy.ndimage.convolve(color[:,:,1], laplace_kernel) 
         + scipy.ndimage.convolve(color[:,:,2], laplace_kernel)) == 0
    ).astype(int)
    
    # If we have custom face coloring labels, re-render the colors
    if face_labels is not None:
        tri_labels = face_labels[tri_labels]
        face_colors = (palette(tri_labels)*255).astype(int)
        mesh = trimesh.Trimesh(vertices=V, faces=part.mesh.F, face_colors=face_colors)
        pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
        scene.add(pmesh, pose=np.eye(4))
        scene.add(cam, pose=camera_pose)
        color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES)

    
    color_with_edges = np.stack([edge_mask]*3,axis=-1)*color
    
    if not renderer:
        r.delete()
    
    return color_with_edges

def render_mesh(
        V, F, labels, camera_pose, zoom, 
        max_labels = None, face_labels=None, cmap='tab20', 
        renderer = None,viewport_width=800, viewport_height=800, point_size=1.0
):
    
    if not max_labels:
        max_labels = (labels.max() + 1)
    palette = plt.cm.get_cmap(cmap, lut=max_labels)
    
    tri_labels = labels
    
    face_colors = (palette(tri_labels)*255).astype(int)
    
    mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=face_colors)
    pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    
    cam = pyrender.OrthographicCamera(xmag=zoom, ymag=zoom)
    
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
    scene.add(pmesh, pose=np.eye(4))
    scene.add(cam, pose=camera_pose)
    
    if not renderer:
        r = pyrender.OffscreenRenderer(viewport_width=viewport_width, viewport_height=viewport_height, point_size=point_size)
    
    # Rendering an image with independent face colors. Do this first regardless of overridding
    # color labels in order to compute hard edges for edge rendering
    color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES)
    
    # Detect Edges to add to image
    laplace_kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    edge_mask = (
        (scipy.ndimage.convolve(color[:,:,0], laplace_kernel) 
         + scipy.ndimage.convolve(color[:,:,1], laplace_kernel) 
         + scipy.ndimage.convolve(color[:,:,2], laplace_kernel)) == 0
    ).astype(int)
    
    # If we have custom face coloring labels, re-render the colors
    if face_labels:
        tri_labels = face_labels[tri_labels]
        face_colors = (palette(tri_labels)*255).astype(int)
        mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=face_colors)
        pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
        scene.add(pmesh, pose=np.eye(4))
        scene.add(cam, pose=camera_pose)
        color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES)

    
    color_with_edges = np.stack([edge_mask]*3,axis=-1)*color
    
    if not renderer:
        r.delete()
    
    return color_with_edges


def render_part2(
        part, camera_pose, zoom, 
        max_labels = None, face_labels=None, cmap='tab20', 
        renderer = None,viewport_width=800, viewport_height=800, point_size=1.0,
        normalize=True
):
    V = part.mesh.V
    if normalize: # assuming centered part for now
        max_coord = np.abs(V).max()
        scale = 1 / max_coord
        V = V * scale
    
    inferred_max_labels = (part.mesh_topology.face_to_topology.max() + 1)
    if max_labels and face_labels is None:
        inferred_max_labels = max_labels
    if not max_labels:
        max_labels = inferred_max_labels

    if isinstance(cmap, str):
        palette = plt.cm.get_cmap(cmap, lut=inferred_max_labels)
    else:
        palette = plt.cm.get_cmap('tab20', lut=inferred_max_labels)
    
    tri_labels = part.mesh_topology.face_to_topology
    
    face_colors = (palette(tri_labels)*255).astype(int)
    
    mesh = trimesh.Trimesh(vertices=V, faces=part.mesh.F, face_colors=face_colors)
    pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    
    cam = pyrender.OrthographicCamera(xmag=zoom, ymag=zoom)
    
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
    scene.add(pmesh, pose=np.eye(4))
    scene.add(cam, pose=camera_pose)
    
    if not renderer:
        r = pyrender.OffscreenRenderer(viewport_width=viewport_width, viewport_height=viewport_height, point_size=point_size)
    
    # Rendering an image with independent face colors. Do this first regardless of overridding
    # color labels in order to compute hard edges for edge rendering
    color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES)
    
    # Detect Edges to add to image
    laplace_kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    edge_mask = (
        (scipy.ndimage.convolve(color[:,:,0], laplace_kernel) 
         + scipy.ndimage.convolve(color[:,:,1], laplace_kernel) 
         + scipy.ndimage.convolve(color[:,:,2], laplace_kernel)) == 0
    ).astype(int)
    
    # If we have custom face coloring labels, re-render the colors
    if face_labels is not None or not isinstance(cmap, str):
        if isinstance(cmap, str):
            palette = plt.cm.get_cmap(cmap, lut=max_labels)
        else:
            palette = cmap
        tri_labels = face_labels[tri_labels]
        face_colors = (np.array(palette(tri_labels))*255).astype(int)
        mesh = trimesh.Trimesh(vertices=V, faces=part.mesh.F, face_colors=face_colors)
        pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
        scene.add(pmesh, pose=np.eye(4))
        scene.add(cam, pose=camera_pose)
        color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES)

    
    color_with_edges = np.stack([edge_mask]*3,axis=-1)*color
    
    if not renderer:
        r.delete()
    
    return color_with_edges
def render_mesh2(
        V, F, labels, camera_pose, zoom, 
        max_labels = None, face_ids = None, face_labels=None, cmap='tab20', 
        renderer = None,viewport_width=800, viewport_height=800, point_size=1.0
):
    
    inferred_max_labels = (labels.max() + 1)
    if max_labels and face_labels is None:
        inferred_max_labels = max_labels
    if not max_labels:
        max_labels = inferred_max_labels
    
    if isinstance(cmap, str):
        palette = plt.cm.get_cmap(cmap, lut=inferred_max_labels)
    else:
        palette = plt.cm.get_cmap('tab20', lut=inferred_max_labels)
    
    tri_labels = labels
    
    face_colors = (palette(tri_labels)*255).astype(int)
    
    mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=face_colors)
    pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    
    cam = pyrender.OrthographicCamera(xmag=zoom, ymag=zoom)
    
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
    scene.add(pmesh, pose=np.eye(4))
    scene.add(cam, pose=camera_pose)
    
    if not renderer:
        r = pyrender.OffscreenRenderer(viewport_width=viewport_width, viewport_height=viewport_height, point_size=point_size)
    
    # Rendering an image with independent face colors. Do this first regardless of overridding
    # color labels in order to compute hard edges for edge rendering
    color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES)
    
    # Detect Edges to add to image
    laplace_kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    edge_mask = (
        (scipy.ndimage.convolve(color[:,:,0], laplace_kernel) 
         + scipy.ndimage.convolve(color[:,:,1], laplace_kernel) 
         + scipy.ndimage.convolve(color[:,:,2], laplace_kernel)) == 0
    ).astype(int)
    
    # If we have custom face coloring labels, re-render the colors
    if face_labels is not None or not isinstance(cmap, str):
        if isinstance(cmap, str):
            palette = plt.cm.get_cmap(cmap, lut=max_labels)
        else:
            palette = cmap
        tri_labels = face_labels[tri_labels]
        face_colors = (np.array(palette(tri_labels))*255).astype(int)
        mesh = trimesh.Trimesh(vertices=V, faces=F, face_colors=face_colors)
        pmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
        scene.add(pmesh, pose=np.eye(4))
        scene.add(cam, pose=camera_pose)
        color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT | pyrender.constants.RenderFlags.SKIP_CULL_FACES)

    
    color_with_edges = np.stack([edge_mask]*3,axis=-1)*color
    
    if not renderer:
        r.delete()
    
    return color_with_edges

