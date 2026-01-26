import mujoco
import numpy as np


def sample_surface_points(verts, faces, n_points):
    """Uniformly sample points on a triangular mesh by surface area."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    tri_probs = tri_areas / np.sum(tri_areas)
    tri_idx = np.random.choice(len(faces), size=n_points, p=tri_probs)

    r1, r2 = np.sqrt(np.random.rand(n_points)), np.random.rand(n_points)
    a, b, c = 1 - r1, r1 * (1 - r2), r1 * r2
    pts = a[:, None] * v0[tri_idx] + b[:, None] * v1[tri_idx] + c[:, None] * v2[tri_idx]
    return pts


def get_scene_point_cloud_surface(model, data, samples_per_geom=2000):
    """
    Generate a uniform-density point cloud using MuJoCo's world-frame geom transforms.
    """
    points_world = []

    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue

        # world transform (already includes local offset)
        geom_pos_world = data.geom_xpos[geom_id]
        geom_rot_world = data.geom_xmat[geom_id].reshape(3, 3)
        mesh_id = model.geom_dataid[geom_id]

        # vertices & faces (raw mesh)
        v_start, v_count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        verts_local = model.mesh_vert[v_start : v_start + v_count].copy()
        f_start, f_count = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        faces = model.mesh_face[f_start : f_start + f_count].reshape(-1, 3)

        # area-based sampling
        pts_local = sample_surface_points(verts_local, faces, samples_per_geom)

        # world transform (no double offsets)
        pts_world = pts_local @ geom_rot_world.T + geom_pos_world
        points_world.append(pts_world)

    return np.concatenate(points_world, axis=0) if points_world else np.zeros((0, 3))


# Example
if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("tmp.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    scene_points = get_scene_point_cloud_surface(model, data, samples_per_geom=3000)
    print("Scene point cloud:", scene_points.shape)

    # Optional visualization
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_points))
        o3d.visualization.draw_geometries([pcd])
    except ImportError:
        print("Install open3d for visualization: pip install open3d")
