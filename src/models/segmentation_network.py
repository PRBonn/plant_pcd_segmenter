import os

import open3d as o3d
import torch
from datasets.sbub_dataset import SBUBDataset
from pytorch_lightning import LightningModule
from utils import (compute_leaf_colors, compute_plant_center_pcds,
                   generate_random_masks, visualize_ious, visualize_point, Timer, visualize_o3d)

from .loss import OffsetLoss
from .postprocessing import Evaluator, cluster_embeddings, compute_gt_ious
from .segmentation_model import OffsetModel



class SegNetwork(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        # name your hyperparameter hparams, then it will be saved automagically.
        self.save_hyperparameters(hparams)
        
        self.offset_model = OffsetModel(cfg=self.hparams["offset_model"])
        
        self.loss_type = self.hparams["offset_model"]["loss"]
        self.leaf_offset_critereon = OffsetLoss(cfg=self.hparams["offset_model"])
        self.plant_offset_critereon = OffsetLoss(cfg=self.hparams["offset_model"])

        self.leaf_evaluator = Evaluator(
            self, "leaf", "leaf_ids", "leaf_id_probs", "leaf_labels"
        )
        self.plant_evaluator = Evaluator(
            self,
            "plant",
            "plant_ids",
            "plant_id_probs",
            "plant_labels",
            plant_offset_size=self.hparams["offset_model"]["plant_offset_size"],
        )

        self.last_render = 0
        self.last_val_render = 0
        self.max_render_freq = self.hparams["debug"]["max_render_freq"]
        self.best_train_loss = 1e6  # just a high number
        self.best_val_loss = 1e6
        self.best_val_vscore = 0
        self.best_train_vscore = 0
        self.best_val_kp_error = 1e6
        self.optimizer_state = None

    def getLoss(self, x: torch.Tensor, y: torch.Tensor):

        return (x - y) ** 2

    def generate_confid_sample(self, batch, log_vals):
        leaf_ids = log_vals["leaf_ids"]
        if leaf_ids != None:
            _, instance_list, instance_max_ious = compute_gt_ious(
                leaf_ids,
                batch["leaf_labels"][:, log_vals["sampled_indices"]],
                log_vals["min_n_points"],
            )
            if (instance_max_ious[0][instance_max_ious[0] != -1] < 0.1).any():
                visualize_ious(
                    batch["points"][0][log_vals["sampled_indices"]],
                    log_vals["leaf_ids"][0],
                    batch["leaf_labels"][0][log_vals["sampled_indices"]],
                    instance_list[0],
                    instance_max_ious[0],
                    instance_max_ious[0],
                )
            if not (instance_list == -1).all():
                log_vals["leaf_ids"] = leaf_ids
                log_vals["instance_max_ious"] = instance_max_ious
                log_vals["leaf_labels"] = batch["leaf_labels"]
                log_vals["instance_list"] = instance_list

                self.save_confid_data(batch, log_vals)

    def save_confid_data(self, batch, log_vals):
        data_sample = {}
        data_sample["points"] = batch["points"][:, log_vals["sampled_indices"]]
        data_sample["leaf_labels"] = batch["leaf_labels"][
            :, log_vals["sampled_indices"]
        ]
        data_sample["leaf_ids"] = log_vals["leaf_ids"]
        data_sample["gt_ious"] = log_vals["instance_max_ious"]
        data_sample["instance_list"] = log_vals["instance_list"]
        data_sample["embeddings"] = log_vals["embeddings"][
            :, log_vals["sampled_indices"]
        ]
        data_sample["file_name"] = batch["plant_name"]
        confid_data_dir = (
            self.hparams["paths"]["log_dir"] + "/confid_data/" + log_vals["epoch_type"]
        )
        os.makedirs(confid_data_dir, exist_ok=True)

        data_sample["hdbscan_probs"] = log_vals["hdbscan_probs"]

        assert len(log_vals["leaf_ids"]) == 1
        cluster_var = torch.full_like(log_vals["hdbscan_probs"][0], -1)
        center_preds = log_vals["points"] + log_vals["embeddings"]
        leaf_list = log_vals["leaf_ids"].unique()[1:]
        for leaf in leaf_list:
            leaf_mask = (log_vals["leaf_ids"] == leaf).squeeze()
            cluster_var[leaf_mask] = center_preds[0][leaf_mask].var(dim=-2).mean()
        data_sample["cluster_var"] = cluster_var.unsqueeze(0)
        torch.save(
            data_sample,
            os.path.join(
                confid_data_dir,
                str(self.global_step) + str(log_vals["batch_idx"]) + ".pt",
            ),
        )

    def placeholder_pred(self):
        confid_loss = torch.tensor(0.0).cuda()
        confid_loss.requires_grad = True
        log_vals = {}
        log_vals["confidences"] = None
        log_vals["leaf_ids"] = None
        log_vals["instance_max_ious"] = None
        return confid_loss, log_vals

    def cluster(self, log_vals, cfg):
        min_n_points = int(
            cfg["data"]["eval_n_samples"]
            * cfg["data"]["min_leaf_point_ratio_inference"]
        )
        log_vals["min_n_points"] = min_n_points
        min_samples = int(min_n_points * cfg["data"]["min_samples_hdbscan_ratio"])
        if self.hparams["offset_model"]["enable_leaves"]:
            leaf_center_preds = log_vals["points"] + log_vals["leaf_offsets"]
            log_vals["leaf_ids"], log_vals["leaf_hdbscan_probs"] = cluster_embeddings(
                leaf_center_preds,
                log_vals["sampled_indices"],
                min_n_points,
                min_samples,
                cdist_cluster_metric=True,
            )

        if self.hparams["offset_model"]["enable_plants"]:
            plant_center_preds = (
                log_vals["points"][
                    ..., : self.hparams["offset_model"]["plant_offset_size"]
                ]
                + log_vals["plant_offsets"]
            )
            log_vals["plant_ids"], log_vals["plant_hdbscan_probs"] = cluster_embeddings(
                plant_center_preds,
                log_vals["sampled_indices"],
                min_n_points,
                min_samples,
                cdist_cluster_metric=True,
            )
        return log_vals

    def offset_step(self, batch, batch_idx, log_vals):
        leaf_offsets, plant_offsets = self.offset_model(
            batch["points"], batch["features"]
        )
        
        log_vals["leaf_offsets"] = leaf_offsets
        log_vals["plant_offsets"] = plant_offsets
        if self.hparams["offset_model"]["enable_leaves"]:
            leaf_loss = self.leaf_offset_critereon(
                leaf_offsets, batch["leaf_offset_labels"]
            )
            self.log(log_vals["epoch_type"] + "/leaf_loss", leaf_loss.detach())
        else:
            leaf_loss = 0
        if self.hparams["offset_model"]["enable_plants"]:
            plant_loss = self.plant_offset_critereon(
                plant_offsets,
                batch["plant_offset_labels"][
                    ..., : self.hparams["offset_model"]["plant_offset_size"]
                ],
            )
            self.log(log_vals["epoch_type"] + "/plant_loss", plant_loss.detach())
        else:
            plant_loss = 0
        loss = leaf_loss + plant_loss
        self.log(log_vals["epoch_type"] + "/loss", loss.detach(), prog_bar=True)

        return loss, log_vals

    def training_step(self, batch: dict, batch_idx, optimizer_idx=0):
        log_vals = {}
        log_vals["batch_idx"] = batch_idx
        log_vals["epoch_type"] = "train"
        log_vals["points"] = batch["points"]
        log_vals["sampled_indices"] = generate_random_masks(
            batch["points"].shape[-2], self.hparams["data"]["eval_n_samples"]
        )
        
        combined_loss, log_vals = self.offset_step(batch, batch_idx, log_vals)
        if self.hparams["data"]["generate_confid_dataset"]:
            # Cluster embeddings
            log_vals = self.cluster(log_vals, self.hparams)
            self.generate_confid_sample(batch, log_vals)
        return_elems = {"loss": combined_loss}

        if (
            "visualize_train_data" in self.hparams["debug"].keys()
            and self.hparams["debug"]["visualize_train_data"]
        ):
            leaf_pcd = compute_leaf_colors(
                batch["points"][0].detach().cpu(),
                batch["leaf_labels"][0].detach().cpu(),
            )
            plant_center_pcds = compute_plant_center_pcds(
                batch["points"][0].detach().cpu()
                + batch["plant_offset_labels"][0].detach().cpu(),
                batch["plant_labels"][0].detach().cpu(),
            )
            visualize_o3d(
                [
                    leaf_pcd,
                ]
                + plant_center_pcds
            )
        return return_elems

    def validation_step(self, batch: dict, batch_idx):
        log_vals = {}
        log_vals["batch_idx"] = batch_idx
        log_vals["epoch_type"] = "val"
        log_vals["points"] = batch["points"]
        log_vals["sampled_indices"] = generate_random_masks(
            batch["points"].shape[-2], self.hparams["data"]["eval_n_samples"]
        )
        combined_loss, log_vals = self.offset_step(batch, batch_idx, log_vals)
        if self.current_epoch >= self.hparams["train"]["warmup_epochs"]:
            log_vals = self.cluster(log_vals, self.hparams)

            if self.hparams["data"]["generate_confid_dataset"]:
                self.generate_confid_sample(batch, log_vals)
            metrics = {}

            if self.hparams["offset_model"]["enable_leaves"]:
                leaf_metrics = self.leaf_evaluator.compute_PQ(
                    log_vals, data_batch=batch, epoch_type="val"
                )
                self.leaf_evaluator.log_predictions(
                    epoch_type="val",
                    batch_data=batch,
                    log_vals=log_vals,
                    log_labels=True,
                )
                metrics.update(leaf_metrics)

            if self.hparams["offset_model"]["enable_plants"]:
                (
                    plant_center_metrics,
                    pred_plant_centers,
                ) = self.plant_evaluator.evaluate_plant_center(
                    log_vals, data_batch=batch, epoch_type="val"
                )
                metrics.update(plant_center_metrics)

                plant_metrics = self.plant_evaluator.compute_PQ(
                    log_vals, data_batch=batch, epoch_type="val"
                )
                self.plant_evaluator.log_predictions(
                    epoch_type="val",
                    batch_data=batch,
                    log_vals=log_vals,
                    log_labels=True,
                    pred_plant_centers=pred_plant_centers,
                )
                metrics.update(plant_metrics)

            return {
                "val_loss": combined_loss,
                self.hparams["train"]["val_metric"]: torch.tensor(
                    metrics[self.hparams["train"]["val_metric"]]
                ),
            }
        else:
            return {
                "val_loss": combined_loss,
                self.hparams["train"]["val_metric"]: torch.tensor(0.0),
            }

    def write_eval_ply(self, batch, log_vals, cfg):
        out_pcd = o3d.t.geometry.PointCloud(
            o3d.core.Tensor(
                (batch["points"][0] + batch["original_centroid"]).cpu().numpy()
            )
        )
        out_pcd.point["pred_leaf_ids"] = o3d.core.Tensor(
            log_vals["leaf_ids"][0].int().unsqueeze(-1).cpu().numpy()
        )
        out_pcd.point["gt_leaf_ids"] = o3d.core.Tensor(
            batch["leaf_labels"][0].int().unsqueeze(-1).cpu().numpy()
        )
        out_pcd.point["hdbscan_probs"] = o3d.core.Tensor(
            log_vals["hdbscan_probs"][0].float().unsqueeze(-1).cpu().numpy()
        )

        cluster_var = torch.full_like(log_vals["hdbscan_probs"][0].float(), -1)
        center_preds = log_vals["points"] + log_vals["embeddings"]
        leaf_list = log_vals["leaf_ids"].unique()[1:]
        for leaf in leaf_list:
            leaf_mask = (log_vals["leaf_ids"] == leaf).squeeze()
            cluster_var[leaf_mask] = center_preds[0][leaf_mask].var(dim=-2).mean()
        out_pcd.point["cluster_var"] = o3d.core.Tensor(
            cluster_var.float().unsqueeze(-1).cpu().numpy()
        )

        o3d.t.io.write_point_cloud(
            os.path.join(
                cfg["paths"]["preds_dir"], batch["plant_name"][0].split(".")[0] + ".ply"
            ),
            out_pcd,
        )

    def write_plant_eval_ply(self, batch, log_vals, cfg):
        points = batch["points"][0] + batch["original_centroid"][0]
        leaf_ids = log_vals["leaf_ids"][0]
        plant_ids = log_vals["plant_ids"][0]
        leaf_labels = batch["leaf_labels"][0]

        # compute ground heights
        pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(points.cpu().numpy()))
        pcd.point["colors"] = o3d.core.Tensor(batch["colors"][0].cpu().numpy())
        pcd.point["plant_ids"] = o3d.core.Tensor(
            plant_ids.int().unsqueeze(-1).cpu().numpy()
        )
        ground_heights = torch.tensor(SBUBDataset.compute_ground_heights(pcd))

        plant_list = log_vals["plant_ids"].unique()[1:]
        plant_center_pcds = []
        for idx in range(len(plant_list)):
            plant_center = (
                log_vals["pred_plant_centers"][idx] + batch["original_centroid"][0][:2]
            )
            plant_center = torch.hstack((plant_center, ground_heights[idx]))

            # add plant center as point
            points = torch.vstack((points, plant_center))
            # and label it accordingly
            leaf_ids = torch.hstack((leaf_ids, torch.tensor((-1.0))))
            plant_ids = torch.hstack((plant_ids, plant_list[idx]))
            leaf_labels = torch.hstack((leaf_labels, torch.tensor((-1.0))))
            plant_center_pcds.append(
                visualize_point(plant_center.cpu().numpy().T, radius=0.01)
            )

        out_pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(points.cpu().numpy()))
        out_pcd.point["pred_leaf_ids"] = o3d.core.Tensor(
            leaf_ids.int().unsqueeze(-1).cpu().numpy()
        )
        out_pcd.point["pred_plant_ids"] = o3d.core.Tensor(
            plant_ids.int().unsqueeze(-1).cpu().numpy()
        )
        out_pcd.point["gt_leaf_ids"] = o3d.core.Tensor(
            leaf_labels.int().unsqueeze(-1).cpu().numpy()
        )

        o3d.t.io.write_point_cloud(
            os.path.join(
                cfg["paths"]["preds_dir"], batch["plant_name"][0].split(".")[0] + ".ply"
            ),
            out_pcd,
        )

        leaf_pcd = compute_leaf_colors(points.detach().cpu(), plant_ids.detach().cpu())
        visualize_o3d(
            [
                leaf_pcd,
            ]
            + plant_center_pcds
        )

    def test_step(self, batch: dict, batch_idx):
        log_vals = {}
        log_vals["batch_idx"] = batch_idx
        log_vals["epoch_type"] = "test"
        log_vals["points"] = batch["points"]
        log_vals["sampled_indices"] = torch.ones(batch["points"].shape[-2]).bool()
        
        combined_loss, log_vals = self.offset_step(batch, batch_idx, log_vals)

        log_vals = self.cluster(log_vals, self.hparams)
        metrics = {}

        if self.hparams["offset_model"]["enable_leaves"]:
            leaf_metrics = self.leaf_evaluator.compute_PQ(
                log_vals, data_batch=batch, epoch_type="val"
            )
            metrics.update(leaf_metrics)

        if self.hparams["offset_model"]["enable_plants"]:
            (
                plant_center_metrics,
                pred_plant_centers,
            ) = self.plant_evaluator.evaluate_plant_center(
                log_vals, data_batch=batch, epoch_type="val"
            )
            log_vals["pred_plant_centers"] = pred_plant_centers
            metrics.update(plant_center_metrics)

            plant_metrics = self.plant_evaluator.compute_PQ(
                log_vals, data_batch=batch, epoch_type="val"
            )
            metrics.update(plant_metrics)

        self.write_plant_eval_ply(batch, log_vals, self.hparams)
        return {
            "PQ": torch.tensor(metrics["PQ_plant"]),
        }

    def configure_optimizers(self):
        optimizers = []
        if (
            self.hparams["offset_model"]["enabled"]
            and not self.hparams["data"]["generate_confid_dataset"]
        ):
            optimizers.append(
                torch.optim.AdamW(
                    self.offset_model.parameters(),
                    lr=float(self.hparams["train"]["lr_emb"]),
                )
            )
        else:
            optimizers = None
        return optimizers
