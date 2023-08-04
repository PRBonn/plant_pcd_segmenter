import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import find_min_kernel_size, generate_random_masks, cache


class ConfidDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if self.cfg["debug"]["overfit"]:
            train_path = self.cfg["paths"]["overfit_path"]
            val_path = self.cfg["paths"]["overfit_path"]
            test_path = self.cfg["paths"]["overfit_path"]
        else:
            train_path = os.path.join(self.cfg["paths"]["data_path"], "train")
            val_path = os.path.join(self.cfg["paths"]["data_path"], "val")
            test_path = os.path.join(self.cfg["paths"]["data_path"], "test")
        self.train_data = ConfidDataset(
            data_path=train_path,
            min_confid_thres=self.cfg["confid_model"]["min_confid_thres"]
            if "min_confid_thres" in self.cfg["confid_model"].keys()
            else 0.4,
            enable_data_balancing=True,
            compute_min_kernel=True,
        )

    def setup(self, stage=None):
        # Create datasets
        if self.cfg["debug"]["overfit"]:
            train_path = self.cfg["paths"]["overfit_path"]
            val_path = self.cfg["paths"]["overfit_path"]
            test_path = self.cfg["paths"]["overfit_path"]
        else:
            train_path = os.path.join(self.cfg["paths"]["data_path"], "train")
            val_path = os.path.join(self.cfg["paths"]["data_path"], "val")
            test_path = os.path.join(self.cfg["paths"]["data_path"], "test")

        self.min_kernel_size = self.train_data.min_kernel_size
        self.val_data = ConfidDataset(
            data_path=val_path,
            min_confid_thres=self.cfg["confid_model"]["min_confid_thres"]
            if "min_confid_thres" in self.cfg["confid_model"].keys()
            else 0.4,
            enable_data_balancing=False,
        )
        self.test_data = ConfidDataset(
            data_path=test_path,
            min_confid_thres=self.cfg["confid_model"]["min_confid_thres"]
            if "min_confid_thres" in self.cfg["confid_model"].keys()
            else 0.4,
            enable_data_balancing=False,
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_data,
            batch_size=self.cfg["data"]["batch_size_confid"],
            num_workers=self.cfg["train"]["num_workers"],
            shuffle=True,
            drop_last=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_data,
            batch_size=self.cfg["data"]["batch_size_confid"],
            num_workers=self.cfg["train"]["num_workers"],
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_data,
            batch_size=self.cfg["data"]["batch_size_confid"],
            num_workers=self.cfg["train"]["num_workers"],
        )
        return loader


class ConfidDataset(Dataset):
    def __init__(
        self,
        data_path,
        min_confid_thres,
        enable_data_balancing=False,
        compute_min_kernel=False,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # Store array for filenames and pass it as length

        super().__init__()
        n_bins = 5
        self.min_confid_thres = min_confid_thres

        balanced_data = self.prepare_data(data_path, enable_data_balancing, n_bins)
        print(
            "Data IoU histog: ",
            torch.histogram(
                balanced_data["gt_ious"][balanced_data["gt_ious"] >= 0].cpu(), 10
            )[0].int(),
        )
        self.dataset_frame = {
            "confid_data": balanced_data,
        }
        if compute_min_kernel:
            self.min_kernel_size = self.compute_min_kernel(
                self.dataset_frame["confid_data"]["points"]
            )

    @cache.memoize(typed=True)
    def prepare_data(self, data_path, enable_data_balancing, n_bins):

        confid_data = self.load_data(data_path)
        ser_data = self.serialize_data(confid_data)
        print(
            "Unbalanced data IoU histog: ",
            torch.histogram(ser_data["gt_ious"][ser_data["gt_ious"] >= 0].cpu(), 10)[
                0
            ].int(),
        )
        if enable_data_balancing:
            balanced_data = self.balance_data(ser_data, n_bins)
        else:
            balanced_data = ser_data
        return balanced_data

    def load_data(self, data_path):
        confid_data = []

        for file in tqdm(os.listdir(data_path)):
            if not file.split(".")[-1] == "pt":
                continue
            confid_data.append(
                torch.load(os.path.join(data_path, file), map_location="cpu")
            )
        return confid_data

    def serialize_data(self, data):
        serialized_data = {}
        for field in data[0].keys():
            print(field)
            field_list = [x[field] for x in data]

            if field == "file_name":
                serialized_data[field] = field_list
            else:
                serialized_data[field] = torch.vstack(field_list)
                serialized_data[field] = serialized_data[field].detach()
        return serialized_data

    def balance_data(self, data, n_bins):
        bin_masks = torch.zeros(
            [
                n_bins,
            ]
            + list(data["gt_ious"].shape),
            dtype=torch.bool,
        )
        bin_masks[0] = torch.logical_and(
            data["gt_ious"] < self.min_confid_thres, data["gt_ious"] != -1
        )
        bin_size = (1 - self.min_confid_thres) / n_bins
        for bin in range(1, n_bins):
            bin_masks[bin] = torch.logical_and(
                data["gt_ious"] > self.min_confid_thres + bin * bin_size,
                data["gt_ious"] < self.min_confid_thres + (bin + 1) * bin_size,
            )
        min_bin_elem = bin_masks[1:, ...].sum(-1).sum(-1).min()

        subsample_mask = torch.zeros_like(data["gt_ious"], dtype=torch.bool)
        for bin in range(n_bins):
            subsample_mask[bin_masks[bin]] = generate_random_masks(
                bin_masks[bin].sum(), min_bin_elem
            )
        out_data = {}
        out_data["points"] = data["points"][subsample_mask.sum(-1) > 0]
        out_data["leaf_ids"] = data["leaf_ids"][subsample_mask.sum(-1) > 0]
        if "hdbscan_probs" in data.keys():
            out_data["hdbscan_probs"] = data["hdbscan_probs"][
                subsample_mask.sum(-1) > 0
            ]
        if "cluster_var" in data.keys():
            out_data["cluster_var"] = data["cluster_var"][subsample_mask.sum(-1) > 0]
        out_data["leaf_labels"] = data["leaf_labels"][subsample_mask.sum(-1) > 0]
        out_data["embeddings"] = data["embeddings"][subsample_mask.sum(-1) > 0]
        out_data["file_name"] = [
            (data["file_name"] * 10)[x]
            for x in torch.where(subsample_mask.sum(-1) > 0)[0]
        ]
        gt_ious = data["gt_ious"]
        gt_ious[~subsample_mask] = -1
        out_data["gt_ious"] = gt_ious[subsample_mask.sum(-1) > 0]

        instance_list = data["instance_list"]
        instance_list[~subsample_mask] = -1
        out_data["instance_list"] = instance_list[subsample_mask.sum(-1) > 0]
        return out_data

    @staticmethod
    def compute_min_kernel(points):
        mask = generate_random_masks(len(points), 100)
        points = points[mask]
        min_kernel_size = []
        for point in points:
            min_kernel_size.append(find_min_kernel_size(point, percent_inliers=50))
        return torch.tensor(min_kernel_size).mean()

    def __len__(self):
        data_len = len(self.dataset_frame["confid_data"]["points"])
        return data_len

    def __getitem__(self, idx):
        subsample_mask = generate_random_masks(
            self.dataset_frame["confid_data"]["points"][0].shape[-2], 50000
        )
        sample = {}
        sample["points"] = self.dataset_frame["confid_data"]["points"][idx][
            subsample_mask
        ]
        sample["file_name"] = self.dataset_frame["confid_data"]["file_name"][idx]
        sample["leaf_ids"] = self.dataset_frame["confid_data"]["leaf_ids"][idx][
            subsample_mask
        ]

        if "hdbscan_probs" in self.dataset_frame["confid_data"].keys():
            sample["hdbscan_probs"] = self.dataset_frame["confid_data"][
                "hdbscan_probs"
            ][idx][subsample_mask]
        if "cluster_var" in self.dataset_frame["confid_data"].keys():
            sample["cluster_var"] = self.dataset_frame["confid_data"]["cluster_var"][
                idx
            ][subsample_mask]
        sample["leaf_labels"] = self.dataset_frame["confid_data"]["leaf_labels"][idx][
            subsample_mask
        ]
        sample["gt_ious"] = self.dataset_frame["confid_data"]["gt_ious"][idx]
        sample["instance_list"] = self.dataset_frame["confid_data"]["instance_list"][
            idx
        ]
        sample["embeddings"] = self.dataset_frame["confid_data"]["embeddings"][idx][
            subsample_mask
        ]
        return sample
