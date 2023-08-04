import os

import numpy as np
import open3d as o3d
import scipy
import torch
import torch.nn.functional as F
from diskcache import FanoutCache
# import data_processor
from pytorch_lightning import LightningDataModule
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import find_min_kernel_size, generate_random_masks, pytimer, SerializablePcdT

cache = FanoutCache(
    directory=os.path.join("/tmp", "fanoutcache_seg"),
    shards=64,
    timeout=1,
    size_limit=3e12,
)

class EmbedDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if self.cfg["debug"]["overfit"]:
            self.train_path = os.path.join(self.cfg["paths"]["overfit_path"], "test")
            self.val_path = os.path.join(self.cfg["paths"]["overfit_path"], "test")
            self.test_path = os.path.join(self.cfg["paths"]["overfit_path"], "test")
        else:
            self.train_path = os.path.join(self.cfg["paths"]["data_path"], "train")
            self.val_path = os.path.join(self.cfg["paths"]["data_path"], "val")
            self.test_path = os.path.join(self.cfg["paths"]["data_path"], "test")

        self.train_data = SBUBDataset(
            cfg=self.cfg,
            data_path=self.train_path,
            file_list=os.listdir(self.train_path),
            n_samples=len(os.listdir(self.train_path))*10,
            augment=self.cfg["data"]["augment"],
            compute_min_kernel=True,
            compute_offset_labels=True
            if self.cfg["offset_model"]["loss"] == "offset"
            else False,
        )

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        self.val_data = SBUBDataset(
            cfg=self.cfg,
            data_path=self.val_path,
            file_list=os.listdir(self.val_path),
            n_samples=len(os.listdir(self.val_path)),
            augment=False,
            compute_offset_labels=True
            if self.cfg["offset_model"]["loss"] == "offset"
            else False,
        )
        self.test_data = SBUBDataset(
            cfg=self.cfg,
            data_path=self.test_path,
            file_list=os.listdir(self.test_path),
            n_samples=len(os.listdir(self.test_path)),
            augment=False,
            compute_offset_labels=True
            if self.cfg["offset_model"]["loss"] == "offset"
            else False,
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_data,
            batch_size=self.cfg["data"]["batch_size_embed"],
            num_workers=self.cfg["train"]["num_workers"],
            shuffle=True,
            drop_last=True,
        )

        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_data,
            batch_size=self.cfg["data"]["batch_size_embed"],
            num_workers=self.cfg["train"]["num_workers"],
        )

        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_data,
            batch_size=self.cfg["data"]["batch_size_embed"],
            num_workers=self.cfg["train"]["num_workers"],
        )

        return loader


class SBUBDataset(Dataset):
    def __init__(
        self,
        cfg,
        data_path,
        file_list,
        n_samples,
        augment,
        compute_min_kernel=False,
        compute_offset_labels=False,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # Store array for filenames and pass it as length

        super().__init__()
        self.processing_plant = 0

        self.n_samples = n_samples
        self.cfg = cfg
        self.sample_size = cfg["data"]["sample_size"]
        self.downsample_size = cfg["data"]["resolution"]
        self.min_leaf_area = cfg["data"]["min_leaf_area"]
        self.only_labeled = cfg["data"]["only_labeled"]
        self.use_poses_as_feats = cfg["offset_model"]["use_poses_as_feats"]

        (
            point_clouds,
            plant_lists,
            leaf_lists,
            keypt_vector_lists,
            ground_heights,
            pcd_centroids,
        ) = self.load_data(
            None,
            file_list,
            data_path,
            self.downsample_size,
            self.min_leaf_area,
            self.only_labeled,
            enable_plants=cfg["offset_model"]["enable_plants"],
        )
        print("Parsed {} pointclouds.".format(len(point_clouds)))
        positions = []
        leaf_ids = []
        colors = []
        plant_ids = []
        for o3d_pcd in point_clouds:
            o3d_pcd = o3d_pcd.to_open3d()
            positions.append(torch.from_numpy(o3d_pcd.point["positions"].numpy()))
            leaf_ids.append(torch.from_numpy(o3d_pcd.point["leaf_ids"].numpy()))
            colors.append(torch.from_numpy(o3d_pcd.point["colors"].numpy()))
            if cfg["offset_model"]["enable_plants"]:
                plant_ids.append(torch.from_numpy(o3d_pcd.point["plant_ids"].numpy()))

        self.dataset_frame = {
            "positions": positions,
            "leaf_ids": leaf_ids,
            "plant_ids": plant_ids,
            "colors": colors,
            "leaf_lists": leaf_lists,
            "plant_lists": plant_lists,
            "ground_heights": ground_heights,
            "centroids": pcd_centroids,
        }
        self.plant_list = plant_lists
        self.n_plants = len(point_clouds)
        self.randomize_plants = True
        self.augment = augment
        self.enable_offset_labels = compute_offset_labels
        if (
            "leaf_center_offsets" in cfg["data"].keys()
            and cfg["data"]["leaf_center_offsets"]
        ):
            instance_means = self.compute_instance_centers()
            self.dataset_frame["instance_means"] = instance_means
        elif self.enable_offset_labels:
            instance_means = self.compute_instance_means()
            self.dataset_frame["instance_means"] = instance_means
            
        self.min_kernel_size = self.compute_min_kernel()
        print("Computed min kernel size:", self.min_kernel_size)

    def compute_min_kernel(self):
        min_kernel_size = []
        for idx in range(len(self.dataset_frame["positions"])):
            point_pos = self.dataset_frame["positions"][idx]
            downsmpl_mask = generate_random_masks(len(point_pos), self.sample_size)
            point_pos = point_pos[downsmpl_mask]
            min_kernel_size.append(
                find_min_kernel_size(point_pos, n_neighbors=10, percent_inliers=60)
            )
        return torch.tensor(min_kernel_size).mean()

    def compute_plant_centers(self, points, plant_labels, leaf_labels):
        unique_ids = plant_labels.unique()
        unique_ids = unique_ids[unique_ids >= 0]
        center_mean = torch.full_like(points, torch.nan)
        for plant_id in unique_ids:
            center_mask = torch.logical_and(
                plant_labels == plant_id, leaf_labels == -1
            ).squeeze()
            center_mean[plant_labels == plant_id] = points[center_mask].mean(dim=-2)

        return center_mean

    @staticmethod
    def verify_plant_centers(file_name, plant_labels, leaf_labels):
        unique_ids = plant_labels.unique()
        unique_ids = unique_ids[unique_ids >= 0]
        for plant_id in unique_ids:
            center_mask = torch.logical_and(
                plant_labels == plant_id, leaf_labels == -1
            ).squeeze()
            if center_mask.sum() == 0:
                raise ValueError(
                    "Plant {} in {} has no center".format(plant_id, file_name)
                )

    def compute_instance_means(self):
        instance_means = []
        for idx in range(len(self.dataset_frame["positions"])):
            points = self.dataset_frame["positions"][idx]
            labels = self.dataset_frame["leaf_ids"][idx]
            unique_ids = labels.unique()
            unique_ids = unique_ids[unique_ids >= 0]
            instance_mean = torch.full_like(points, torch.nan)
            for leaf_id in unique_ids:
                leaf_mask = (labels == leaf_id).squeeze()
                instance_mean[leaf_mask] = points[leaf_mask].mean(dim=-2)

            instance_means.append(instance_mean)

        return instance_means

    def compute_instance_centers(self):
        instance_centers = []
        for idx in range(len(self.dataset_frame["positions"])):
            points = self.dataset_frame["positions"][idx]
            labels = self.dataset_frame["leaf_ids"][idx]
            unique_ids = labels.unique()
            unique_ids = unique_ids[unique_ids != (-1)]
            instance_mean = torch.full_like(points, torch.nan)
            for leaf_id in unique_ids:
                leaf_mask = (labels == leaf_id).squeeze()
                leaf_mean = points[leaf_mask].mean(dim=-2)
                dists = (points[leaf_mask] - leaf_mean).norm(dim=-1)
                instance_mean[leaf_mask] = points[leaf_mask][torch.argmin(dists)]
            instance_centers.append(instance_mean)
            
            pcd = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(points.detach().cpu().numpy())
            )
            pcd.paint_uniform_color(np.array((0, 0, 1.0)))
            shifted = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(instance_mean.detach().cpu().numpy())
            )
            shifted.paint_uniform_color(np.array((1.0, 0, 0)))
            o3d.visualization.draw_geometries([pcd, shifted])

        return instance_centers

    @staticmethod
    @cache.memoize(typed=True)
    def load_data(
        data_proc, file_list, data_path, downsample_size, min_leaf_area, only_labeled, enable_plants
    ):
        point_clouds = []
        plant_lists = []
        leaf_lists = []
        keypt_vector_lists = []
        ground_heights = []
        pcd_centroids = []

        for file in tqdm(file_list):
            print("Loading file", file)
            if not file.split(".")[-1] == "ply":
                continue
            plant_lists.append(file)
            
            # read pointcloud
            plant_pcd = o3d.t.io.read_point_cloud(os.path.join(data_path, file))

            # downsample if needed
            if downsample_size > 0:
                plant_pcd = plant_pcd.voxel_down_sample(downsample_size)

            if enable_plants:
                # check that at least one plant center is labeled
                if (
                    torch.logical_and(
                        torch.from_numpy(plant_pcd.point["plant_ids"].numpy()) >= 0,
                        torch.from_numpy(plant_pcd.point["leaf_ids"].numpy()) == -1,
                    ).sum()
                    == 0
                ):
                    continue

            if only_labeled:
                if plant_pcd.point.__contains__("leaf_ids"):
                    plant_mask = np.where(plant_pcd.point["leaf_ids"].numpy() >= 0)[0]
                    plant_pcd = pcd_select_by_id(plant_pcd, plant_mask)
                else:
                    raise ValueError("No leaf ids present")
            else:
                # filter by green to remove ground
                plant_pcd = filter_pcd_by_green(plant_pcd)[0]
                
            current_leaf_label_list = compute_filtered_leaf_list(
                plant_pcd, min_leaf_area
            )
            pcd_centroid = plant_pcd.get_center().numpy()
            plant_pcd.translate(-pcd_centroid)
            
            pcd_centroids.append(torch.tensor(pcd_centroid))
            
            point_clouds.append(SerializablePcdT(plant_pcd))

            leaf_lists.append(current_leaf_label_list)

        return point_clouds, plant_lists, leaf_lists, keypt_vector_lists, ground_heights, pcd_centroids

    def __len__(self):
        len = self.n_samples
        return int(len)  # Batch size according to number of PointCloud

    def elastic_distortion(self, pointcloud, granularity, magnitude):
        """Apply elastic distortion on sparse coordinate space.
        Thanks to chrischoy for providing the code at https://github.com/chrischoy/SpatioTemporalSegmentation.

        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords = pointcloud[:, :3]
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)
        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=0, fill_value=0
        )
        pointcloud[:, :3] = coords + interp(coords) * magnitude
        return pointcloud

    def __getitem__(self, idx):
        self.processing_plant = idx % self.n_plants

        points = self.dataset_frame["positions"][self.processing_plant]
        plant_name = self.dataset_frame["plant_lists"][self.processing_plant]

        # fix seed if augmentation disabled to get always the same subsample
        if not self.augment:
            np.random.seed(0)
            torch.manual_seed(0)

        mask = generate_random_masks(len(points), self.sample_size)
        
        points = points[mask]

        colors = self.dataset_frame["colors"][self.processing_plant][mask]
        if self.cfg["offset_model"]["enable_leaves"]:
            leaf_ids = self.dataset_frame["leaf_ids"][self.processing_plant][mask].squeeze()
        if self.cfg["offset_model"]["enable_plants"]:
            plant_ids = self.dataset_frame["plant_ids"][self.processing_plant][mask].squeeze()
        
        if self.augment:
            if np.random.rand() < 0.95:
                points = points.numpy()
                for granularity, magnitude in ((0.02, 0.04), (0.08, 0.16)):
                    points = self.elastic_distortion(points.astype("float32"), granularity, magnitude)
                points = torch.from_numpy(points)
            if np.random.rand() < 0.5:
                # flip x
                points = points * torch.tensor((-1, 1, 1))
            if np.random.rand() < 0.5:
                # flip y
                points = points * torch.tensor((1, -1, 1))
            rand_z = np.random.randint(-180, 180)
            rand_y = np.random.randint(-25, 25)
            rand_x = np.random.randint(-25, 25)
            random_r = torch.from_numpy(
                Rotation.from_euler(
                    "zyx", [rand_z, rand_y, rand_x], degrees=True
                ).as_matrix()
            )
            
            points = torch.mm(random_r, points.double().T).T.float()
        
        leaf_list = self.dataset_frame["leaf_lists"][self.processing_plant]
        padded_leaf_list = np.full(100, float("nan"))
        padded_leaf_list[: len(leaf_list)] = leaf_list

        if self.use_poses_as_feats:
            in_feats = torch.cat((colors.float(), points), dim=-1)
        else:
            in_feats = colors.float()

        # compute leaf offset labels
        leaf_instance_mean = torch.full_like(points, torch.nan)
        for leaf_id in leaf_list:
            leaf_mask = (leaf_ids == leaf_id).squeeze()
            leaf_instance_mean[leaf_mask] = points[leaf_mask].mean(dim=-2)
        leaf_offset_labels = leaf_instance_mean - points
        if self.cfg["offset_model"]["enable_plants"]:
            # and plant offset labels
            plant_centers = self.compute_plant_centers(points, plant_ids, leaf_ids)
            plant_offset_labels = plant_centers - points
        sample = {
            "points": points,
            "colors": colors,
            "features": in_feats,
            "num_points": points.shape[0],
            "leaf_labels": leaf_ids,
            "leaf_list": padded_leaf_list,
            "plant_name": plant_name,
            "leaf_offset_labels": leaf_offset_labels,
            "original_centroid": self.dataset_frame["centroids"][
                 self.processing_plant
             ],
        }
        if self.cfg["offset_model"]["enable_plants"]:
            sample["plant_labels"] = plant_ids
            sample["plant_offset_labels"] = plant_offset_labels
            sample["plant_centers"] = plant_centers
        return sample

def compute_filtered_leaf_list(pcd, min_leaf_points):
    sparse_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    leaf_label_list, counts = np.unique(
        sparse_pcd.point["leaf_ids"].numpy(), return_counts=True
    )
    # filter nans
    leaf_label_list = leaf_label_list[~np.isnan(leaf_label_list)]
    # remove plant center and crap
    mask = leaf_label_list >= 0
    leaf_label_list = leaf_label_list[mask]
    counts = counts[mask]
    if min_leaf_points != None:
        leaf_label_list = leaf_label_list[counts > min_leaf_points]

    return leaf_label_list

def filter_pcd_by_green(in_pcd, thres=0.1, invert=False):
    pcd = in_pcd.clone()
    colors = pcd.point["colors"].numpy()
    colors = colors / 255
    R = 0
    G = 1
    B = 2
    ExG = colors[:, G] * 2 - colors[:, R] - colors[:, B]
    # ExR = colors[:,R]*1.4 - colors[:,G]
    # index_diff = ExG - ExR
    if invert:
        mask = np.argwhere(ExG < thres)
    else:
        mask = np.argwhere(ExG > thres)
        
    pcd = pcd_select_by_id(pcd, mask)

    return pcd, mask

def pcd_select_by_id(pcd, ids):
    ids = np.squeeze(ids)
    attributes = ["colors", "normals", "leaf_ids", "plant_ids", "confidence", "label"]
    sub_pcd = o3d.t.geometry.PointCloud()
    sub_pcd.point["positions"] = o3d.core.Tensor(pcd.point["positions"].numpy()[ids])
    for attr in attributes:
        if pcd.point.__contains__(attr):
            sub_pcd.point[attr] = o3d.core.Tensor(pcd.point[attr].numpy()[ids])

    return sub_pcd