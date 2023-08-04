import os

import matplotlib.cm
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
from diskcache import FanoutCache
from scipy import stats
from sklearn.neighbors import NearestNeighbors

cache = FanoutCache(
    directory=os.path.join("/tmp", "fanoutcache_seg"),
    shards=64,
    timeout=1,
    size_limit=3e11,
)


class SerializablePcd:
    def __init__(self, pcd: o3d.geometry.PointCloud):
        self.points = np.asarray(pcd.points)
        self.normals = np.asarray(pcd.normals)
        self.colors = np.asarray(pcd.colors)

    def to_open3d(self) -> o3d.geometry.PointCloud:

        pcl = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(np.asarray(self.points))
        )
        pcl.normals = o3d.utility.Vector3dVector(np.asarray(self.normals))
        pcl.colors = o3d.utility.Vector3dVector(np.asarray(self.colors))
        return pcl


class SerializableMesh:
    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        self.vertices = np.asarray(mesh.vertices)
        self.triangles = np.asarray(mesh.triangles)
        self.vertex_normals = np.asarray(mesh.vertex_normals)
        self.vertex_colors = np.asarray(mesh.vertex_colors)

    def to_open3d(self) -> o3d.geometry.TriangleMesh:

        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(np.asarray(self.vertices)),
            triangles=o3d.utility.Vector3iVector(np.asarray(self.triangles)),
        )
        mesh.vertex_normals = o3d.utility.Vector3dVector(
            np.asarray(self.vertex_normals)
        )
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(self.vertex_colors))
        return mesh


def serialize_o3d(o3d_object):
    if isinstance(o3d_object, o3d.geometry.PointCloud):
        return SerializablePcd(o3d_object)
    elif isinstance(o3d_object, o3d.geometry.TriangleMesh):
        return SerializableMesh(o3d_object)
    else:
        raise ValueError("Only PointCloud and TriangleMesh supported.")


def check_o3d_type(o3d_object):
    if isinstance(o3d_object, o3d.geometry.PointCloud):
        o3d_object_type = "PointCloud"
    elif isinstance(o3d_object, o3d.geometry.TriangleMesh):
        o3d_object_type = "TriangleMesh"
    else:
        raise ValueError("Only PointCloud and TriangleMesh supported.")
    return o3d_object_type


def np2o3d(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz)[:, :3])
    if xyz.shape[1] == 6:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(xyz)[:, 3:6])
    return pcd

def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def compute_occlusion_likelihood(verts):
    # compute distance to stem
    verts = np.asarray(verts)
    likelihood = verts[:, 1] - verts[0, 1]
    # normalize
    likelihood /= np.max(likelihood)
    # print("shapes", verts.shape, likelihood.shape)
    return likelihood

def visualize_point(point, radius=5):
    point_vis = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=5)
    point_vis.translate(point)
    point_vis.paint_uniform_color(np.array((0, 0, 1)))
    return point_vis


def generate_random_masks(output_shape: torch.Size, subsample_n=10000):
    """Generate a random binary mask with for sampling.

    Args:
        output_shape (torch.Size): shape of the random mask
        subsample_n (int, optional): Number of samples. Defaults to 10000.

    Returns:
        torch.tensor: random binary mask
    """
    # subsampling mask
    mask = torch.zeros(output_shape, dtype=torch.bool)
    mask[:subsample_n] = 1
    rand_perm = torch.randperm(output_shape)
    subsample_mask = mask[rand_perm]
    return subsample_mask


def generate_shuffled_masks(output_shape: torch.Size, subsample_n=10000):
    """Generate a random binary mask with for sampling.

    Args:
        output_shape (torch.Size): shape of the random mask
        subsample_n (int, optional): Number of samples. Defaults to 10000.

    Returns:
        torch.tensor: random binary mask
    """
    # subsampling mask
    # mask = torch.zeros(output_shape, dtype=torch.bool)
    mask = torch.arange(output_shape, dtype=torch.long)
    # mask[:subsample_n] = 1
    rand_perm = torch.randperm(output_shape)
    subsample_mask = mask[rand_perm][:subsample_n]
    return subsample_mask


def compute_plant_center_pcds(plant_offsets, plant_ids):
    plant_center_pcds = []
    for plant in plant_ids.unique()[1:]:
        plant_mask = plant_ids == plant
        plant_center = plant_offsets[plant_mask].mean(dim=0)
        plant_center_pcds.append(visualize_point(plant_center, radius=0.01))
    return plant_center_pcds


def visualize_labeled_cloud(points, labels, visualize_plant_centers=False):
    leaf_seg_pcl = compute_leaf_colors(points, labels)
    if visualize_plant_centers:
        compute_plant_center_pcds()
    visualize_o3d([leaf_seg_pcl])


def visualize_prediction_errors(points, preds, labels):
    leaf_seg_pcl = compute_leaf_colors(points, preds)

    visualize_o3d([leaf_seg_pcl])


def visualize_ious(points, preds, labels, leaf_list, gt_ious, pred_ious):
    pred_pcl = compute_leaf_colors(points.detach().cpu(), preds.detach().cpu())
    label_pcl = compute_leaf_colors(points.detach().cpu(), labels.detach().cpu())
    gui.Application.instance.initialize()

    window = gui.Application.instance.create_window("IoUs", 1024, 1024)

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)

    window.add_child(scene)

    scene.scene.add_geometry("pcl", pred_pcl, rendering.MaterialRecord())
    scene.scene.add_geometry(
        "label_pcl",
        label_pcl.translate(np.array((1.0, 0.0, 0.0))),
        rendering.MaterialRecord(),
    )

    bounds = pred_pcl.get_axis_aligned_bounding_box()
    scene.setup_camera(60, bounds, bounds.get_center())
    for idx, leaf in enumerate(leaf_list):
        if not (gt_ious[idx] != -1 and (pred_ious[idx] - gt_ious[idx]) > 0.2):
            continue
        center = points[preds == leaf].mean(dim=-2)
        label = "gt {:.2f}".format(gt_ious[idx]) + "pred {:.2f}".format(pred_ious[idx])
        l = scene.add_3d_label(center.detach().cpu().numpy(), label)

    gui.Application.instance.run()


def log_labeled_cloud(points, labels, trainer_obj, log_name, plant_centers=None):
    leaf_seg_pcl = compute_leaf_colors(points, labels)
    tensorboard_pcl = leaf_seg_pcl
    point_positions = torch.tensor(np.asarray(tensorboard_pcl.points)).unsqueeze_(0)
    # scale to 0-255 range for tensorboard
    point_colors = torch.from_numpy(
        np.asarray(tensorboard_pcl.colors) * 255
    ).unsqueeze_(0)
    trainer_obj.logger.experiment.add_mesh(
        log_name,
        vertices=point_positions,
        colors=point_colors,
        global_step=trainer_obj.global_step,
    )

def compute_leaf_colors(points, leaf_ids):
    leaf_ids = np.unique(leaf_ids, return_inverse=True)[1]
    leaf_list = np.unique(leaf_ids)
    cmap_leaves = matplotlib.cm.get_cmap("gist_rainbow")
    colors = (leaf_list / leaf_list.max()) * 0.9 + 0.1
    leaf_colors = cmap_leaves(colors)[:, :3]
    leaf_colors = leaf_colors * 0.6
    np.random.seed(0)
    np.random.shuffle(leaf_colors)

    point_colors = np.zeros_like(points)

    for idx in leaf_list:
        if idx == 0:
            color = (0.5, 0.5, 0.5)
        else:
            color = leaf_colors[idx]
        leaf_mask = leaf_ids == idx
        point_colors[leaf_mask] = np.asarray(color)
    # Create an o3d pointcloud
    leaf_seg_pcl = np2o3d(points)
    leaf_seg_pcl.colors = o3d.utility.Vector3dVector(point_colors)

    return leaf_seg_pcl


def batch_instances(
    instance_ids,
    point_embeds,
    points,
    instance_list,
    use_emb_feats=False,
    input_var=True,
):
    trgt_shape = [
        instance_ids.shape[0],
        instance_list.shape[-1],
        instance_ids.shape[-1],
    ]
    masks = instance_ids.unsqueeze(-2).expand(trgt_shape) == instance_list.unsqueeze(
        -1
    ).expand(trgt_shape)
    trgt_shape_embed = [
        instance_ids.shape[0],
        instance_list.shape[-1],
        instance_ids.shape[-1],
        point_embeds.shape[-1],
    ]
    active_instances = instance_list >= 0
    batched_points = points[torch.where(active_instances)[0]]
    if use_emb_feats:
        masks = masks.unsqueeze(dim=-1)[active_instances]
        point_embeds = point_embeds.unsqueeze(dim=1).expand(trgt_shape_embed)[
            active_instances
        ]
        pred_centers = point_embeds.clone()
        pred_centers[~masks.squeeze(dim=-1)] = 0
        pred_centers = pred_centers.sum(dim=-2)
        pred_centers /= masks[..., 0].bool().sum(dim=-1).unsqueeze(-1)

        normalized_center_preds = point_embeds - pred_centers.unsqueeze(dim=-2)
        masks_with_feats = normalized_center_preds * masks
        return batched_points, masks_with_feats, masks
    elif input_var:
        masks = masks.unsqueeze(-1).expand(trgt_shape_embed)[active_instances]
        point_embeds = point_embeds.unsqueeze(dim=1).expand(trgt_shape_embed)[
            active_instances
        ]
        vars = torch.zeros((len(masks), 3), device=masks.device)
        for batch in range(len(masks)):
            vars[batch] = torch.var(point_embeds[batch][masks[batch, :, 0]], dim=0)
        masks = masks * vars.unsqueeze(-2).expand_as(masks)
        return batched_points, masks, None

    else:
        masks = masks.unsqueeze(-1).expand(trgt_shape_embed)[active_instances]
        # masks = masks.unsqueeze(-1).expand(trgt_shape_embed)
        return batched_points, masks, None  # .unsqueeze(-1)


def find_min_kernel_size(points, n_neighbors=10, percent_inliers=80):
    points = points.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(points)
    dists, ids = nbrs.kneighbors(points)
    return torch.tensor(stats.scoreatpercentile(dists[:, -1], percent_inliers))


class SerializablePcdT:
    def __init__(
        self,
        pcd: o3d.t.geometry.PointCloud,
        attributes=[
            "colors",
            "normals",
            "leaf_ids",
            "plant_ids",
            "confidence",
            "label",
            "keypt_ids",
        ],
    ):
        self.attributes = attributes
        self.points = pcd.point["positions"].numpy()
        for attr in self.attributes:
            if pcd.point.__contains__(attr):
                setattr(self, attr, pcd.point[attr].numpy())

    def to_open3d(self) -> o3d.t.geometry.PointCloud:
        pcd = o3d.t.geometry.PointCloud()
        pcd.point["positions"] = o3d.core.Tensor(self.points)
        for attr in self.attributes:
            if hasattr(self, attr):
                pcd.point[attr] = o3d.core.Tensor(getattr(self, attr))
        return pcd
    
def visualize_o3d(shape_list, name="Open3d", show_frame=False, non_blocking=False):
    # o3d.visualization.draw_geometries(shape_list, mesh_show_back_face=True, mesh_show_wireframe=True, window_name=name)
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    for geometry in shape_list:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    if show_frame:
        opt.show_coordinate_frame = True
    opt.mesh_show_back_face = True
    opt.mesh_show_wireframe = True
    # opt.window_name=name
    # opt.background_color = np.asarray([0.5, 0.5, 0.5])
    if not non_blocking:
        viewer.run()
        viewer.destroy_window()
    else:
        return viewer