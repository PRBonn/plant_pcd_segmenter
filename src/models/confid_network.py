import os

import open3d as o3d
import torch
from pytorch_lightning import LightningModule

from .confid_model import ConfidModel
from .loss import ConfidIoULoss
from utils import batch_instances


class ConfidNet(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        # name you hyperparameter hparams, then it will be saved automagically.
        self.save_hyperparameters(hparams)
        self.confid_model = ConfidModel(cfg=self.hparams["confid_model"])
        self.confid_criterion = ConfidIoULoss(cfg=self.hparams["confid_model"])
        self.last_render = 0
        self.last_val_render = 0
        self.max_render_freq = self.hparams["debug"]["max_render_freq"]
        self.best_train_loss = 1e6  # just a high number
        self.best_val_loss = 1e6
        self.best_val_vscore = 0
        self.best_train_vscore = 0
        self.best_val_kp_error = 1e6
        self.max_kp_offset_len = self.hparams["offset_model"]["max_kp_offset_len"]
        self.optimizer_state = None

    def getLoss(self, x: torch.Tensor, y: torch.Tensor):

        return (x - y) ** 2

    def confid_forward(
        self, x: torch.Tensor, f: torch.Tensor, point_embeds, instance_list
    ):
        pred_ious = torch.full_like(instance_list, -1.0)

        batched_points, feats, masks = batch_instances(
            f,
            x + point_embeds,
            x,
            instance_list,
            use_emb_feats=self.hparams["confid_model"]["use_emb_feats"],
        )

        if self.hparams["confid_model"]["use_poses_as_feats"]:
            raise ValueError("Not implemented")

        tmp_preds = torch.zeros(len(batched_points)).to(x.device)
        mini_batch_size = 20
        for sub_batch in range(
            torch.ceil(torch.tensor(len(batched_points) / mini_batch_size)).int()
        ):
            confid_output = self.confid_model(
                batched_points[
                    sub_batch * mini_batch_size : (sub_batch + 1) * mini_batch_size
                ],
                feats.float()[
                    sub_batch * mini_batch_size : (sub_batch + 1) * mini_batch_size
                ],
            )
            if confid_output == None:
                return None
            tmp_preds[
                sub_batch * mini_batch_size : (sub_batch + 1) * mini_batch_size
            ] = confid_output.squeeze()
        pred_ious[instance_list >= 0] = tmp_preds
        return pred_ious

    def reshape_batches(self, batch):
        batch_shape = [
            batch.shape[0] * batch.shape[1],
        ] + list(batch.shape[2:])
        return batch.reshape(batch_shape)

    def forward_step(self, batch, epoch_type):
        points = batch["points"]
        leaf_ids = batch["leaf_ids"]
        instance_list = batch["instance_list"]
        gt_ious = batch["gt_ious"]
        point_embeds = batch["embeddings"]
        confidences = self.confid_forward(points, leaf_ids, point_embeds, instance_list)
        if confidences == None:
            return None
        confid_loss = self.confid_criterion(confidences, gt_ious)
        valid_inst_mask = gt_ious >= 0
        confid_errors = confidences[valid_inst_mask] - gt_ious[valid_inst_mask]
        confid_error = confid_errors.abs().mean()

        overestim_error = confid_errors[confid_errors > 0].sum() / valid_inst_mask.sum()

        self.log(epoch_type + "/confid_loss", confid_loss.detach(), prog_bar=True)
        self.log(epoch_type + "/confid_error", confid_error.detach(), prog_bar=True)
        self.log(
            epoch_type + "/overestim_error", overestim_error.detach(), prog_bar=True
        )
        return confid_loss, confidences

    def training_step(self, batch: dict, batch_idx, optimizer_idx=0):
        return_elems = self.forward_step(batch, "train")
        if return_elems == None:
            print("Skipped batch")
            return None
        combined_loss, _ = return_elems
        return_elems = {"loss": combined_loss}

        return return_elems

    def validation_step(self, batch: dict, batch_idx):
        return_elems = self.forward_step(batch, "val")
        if return_elems == None:
            print("Skipped batch")
            return None
        combined_loss, _ = return_elems

        return {
            "val_loss": combined_loss,
        }

    def write_eval_ply(self, batch, confidences, gt_ious, cfg):
        out_pcd = o3d.t.geometry.PointCloud(
            o3d.core.Tensor(batch["points"][0].cpu().numpy())
        )
        assert len(batch["leaf_ids"]) == 1
        out_pcd.point["pred_leaf_ids"] = o3d.core.Tensor(
            batch["leaf_ids"][0].int().unsqueeze(-1).cpu().numpy()
        )
        out_pcd.point["gt_leaf_ids"] = o3d.core.Tensor(
            batch["leaf_labels"][0].int().unsqueeze(-1).cpu().numpy()
        )
        # compute pointwise confidences
        pt_confid = torch.zeros_like(batch["leaf_ids"][0]).float()
        pt_gt_iou = torch.zeros_like(batch["leaf_ids"][0]).float()
        for id, pred_label in enumerate(batch["instance_list"][0]):
            mask = batch["leaf_ids"] == pred_label
            pt_confid[mask.squeeze()] = confidences[0][id]
            pt_gt_iou[mask.squeeze()] = gt_ious[0][id]
        out_pcd.point["pred_leaf_confid"] = o3d.core.Tensor(
            pt_confid.unsqueeze(-1).cpu().numpy()
        )

        out_pcd.point["gt_iou"] = o3d.core.Tensor(pt_gt_iou.unsqueeze(-1).cpu().numpy())
        out_pcd.point["hdbscan_probs"] = o3d.core.Tensor(
            batch["hdbscan_probs"][0].float().unsqueeze(-1).cpu().numpy()
        )

        cluster_var = torch.full_like(batch["hdbscan_probs"][0].float(), -1)
        center_preds = batch["points"] + batch["embeddings"]
        leaf_list = batch["leaf_ids"].unique()[1:]
        for leaf in leaf_list:
            leaf_mask = (batch["leaf_ids"] == leaf).squeeze()
            cluster_var[leaf_mask] = center_preds[0][leaf_mask].var(dim=-2).mean()
        out_pcd.point["cluster_var"] = o3d.core.Tensor(
            cluster_var.float().unsqueeze(-1).cpu().numpy()
        )

        o3d.t.io.write_point_cloud(
            os.path.join(cfg["paths"]["preds_dir"], batch["file_name"][0][0]), out_pcd
        )

    def test_step(self, batch: dict, batch_idx):
        return_elems = self.forward_step(batch, "test")
        if return_elems == None:
            print("Skipped batch")
            return None
        combined_loss, confidences = return_elems
        self.write_eval_ply(batch, confidences, batch["gt_ious"], self.hparams)
        return {"test_loss": combined_loss}

    def configure_optimizers(self):
        optimizers = []
        optimizers.append(
            torch.optim.Adam(
                self.confid_model.parameters(),
                lr=float(self.hparams["train"]["lr_confid"]),
            )
        )
        return optimizers
