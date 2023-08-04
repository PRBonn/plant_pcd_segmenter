import numpy as np
import open3d as o3d
import torch
from hdbscan import HDBSCAN as hdbscan_cpu
from sklearn.cluster import MeanShift
from utils import PanopticEval, log_labeled_cloud


def compute_gt_ious(predicted_instance_ids, gt_instance_ids, min_points=60):
    """Compute ground truth ious.
    predicted_instance_ids contains cluster ids [-1, n_clusters], where -1 are points not assigned to any cluster.

    Args:
        predicted_instance_ids (_type_): _description_
        gt_instance_ids (_type_): _description_

    Returns:
        _type_: _description_
    """
    predicted_instance_ids = predicted_instance_ids.long()
    gt_instance_ids = gt_instance_ids.long()
    device = predicted_instance_ids.device
    processed_gt_ious = torch.full(
        predicted_instance_ids.shape, -1.0, device=device, dtype=torch.float
    )
    instance_list = torch.full((len(predicted_instance_ids), 1000), -1.0, device=device)
    instance_max_ious = torch.full(
        (len(predicted_instance_ids), 1000), -1.0, device=device, dtype=torch.float
    )
    max_len = 0
    for B in range(len(predicted_instance_ids)):
        predicted_instance_ids_batch = predicted_instance_ids[B]
        gt_instance_ids_batch = gt_instance_ids[B]

        # generate the areas for each unique instance prediction
        unique_pred, counts_pred = torch.unique(
            predicted_instance_ids_batch[predicted_instance_ids_batch >= 0],
            return_counts=True,
        )
        valid_mask = counts_pred > min_points
        try:
            instance_list[B, : len(unique_pred[valid_mask])] = unique_pred[valid_mask]
        except:
            raise ValueError(
                "Max instances probably to low. Try adjusting instance_list and instance_max_ious size."
            )
        max_len = max(len(unique_pred[valid_mask]), max_len)
        id2idx_pred = {id.item(): idx for idx, id in enumerate(unique_pred)}

        # generate the areas for each unique instance gt_np
        unique_gt, counts_gt = torch.unique(gt_instance_ids_batch, return_counts=True)
        id2idx_gt = {id.item(): idx for idx, id in enumerate(unique_gt)}

        # generate intersection using offset
        offset = 1000

        valid_combos = predicted_instance_ids_batch >= 0
        offset_combo = (
            predicted_instance_ids_batch[valid_combos]
            + offset * gt_instance_ids_batch[valid_combos]
        )
        unique_combo, counts_combo = torch.unique(offset_combo, return_counts=True)

        # generate an intersection map
        # count the intersections with over 0.5 IoU as TP
        gt_labels = torch.div(unique_combo, offset, rounding_mode="floor")
        pred_labels = unique_combo % offset
        gt_areas = torch.empty(len(gt_labels), device=device)
        for i, id in enumerate(gt_labels):
            gt_areas[i] = counts_gt[id2idx_gt[id.item()]]

        pred_areas = torch.empty(len(pred_labels), device=device)
        for i, id in enumerate(pred_labels):
            pred_areas[i] = counts_pred[id2idx_pred[id.item()]]

        intersections = counts_combo
        unions = gt_areas + pred_areas - intersections
        ious = intersections.float() / unions.float()
        for pred_label in pred_labels:
            relevant_ious = ious[pred_labels == pred_label]
            relevant_intersections = intersections[pred_labels == pred_label]
            most_overlapping_id = relevant_intersections.argmax()
            # find the gt id of the cluster with largest overlap with current prediction
            max_gt_id = gt_labels[pred_labels == pred_label][most_overlapping_id]
            # set max_iou to -1 if associated gt is unlabeled (-1)
            if max_gt_id == -1:
                max_iou = -1
            else:
                max_iou = relevant_ious[most_overlapping_id]

            processed_gt_ious[B][predicted_instance_ids_batch == pred_label] = max_iou
            instance_max_ious[B][instance_list[B] == pred_label] = max_iou

    return processed_gt_ious, instance_list, instance_max_ious


def map_instances(
    predicted_instance_ids,
    gt_instance_ids,
    min_points=60,
    allow_invalid_association=False,
):
    """Compute ground truth ious.
    predicted_instance_ids contains cluster ids [-1, n_clusters], where -1 are points not assigned to any cluster.

    Args:
        predicted_instance_ids (_type_): _description_
        gt_instance_ids (_type_): _description_

    Returns:
        _type_: _description_
    """
    predicted_instance_ids = predicted_instance_ids.long()
    gt_instance_ids = gt_instance_ids.long()
    device = predicted_instance_ids.device
    instance_list = torch.full((len(predicted_instance_ids), 1000), -1.0, device=device)
    max_len = 0
    for B in range(len(predicted_instance_ids)):
        predicted_instance_ids_batch = predicted_instance_ids[B]
        gt_instance_ids_batch = gt_instance_ids[B]

        # generate the areas for each unique instance prediction
        unique_pred, counts_pred = torch.unique(
            predicted_instance_ids_batch[predicted_instance_ids_batch >= 0],
            return_counts=True,
        )
        valid_mask = counts_pred > min_points
        try:
            instance_list[B, : len(unique_pred[valid_mask])] = unique_pred[valid_mask]
        except:
            raise ValueError(
                "Max instances probably to low. Try adjusting instance_list and instance_max_ious size."
            )
        max_len = max(len(unique_pred[valid_mask]), max_len)
        id2idx_pred = {id.item(): idx for idx, id in enumerate(unique_pred)}

        # generate the areas for each unique instance gt_np
        unique_gt, counts_gt = torch.unique(gt_instance_ids_batch, return_counts=True)
        id2idx_gt = {id.item(): idx for idx, id in enumerate(unique_gt)}

        # generate intersection using offset
        offset = 1000

        valid_combos = predicted_instance_ids_batch >= 0
        offset_combo = (
            predicted_instance_ids_batch[valid_combos]
            + offset * gt_instance_ids_batch[valid_combos]
        )
        unique_combo, counts_combo = torch.unique(offset_combo, return_counts=True)

        # generate an intersection map
        # count the intersections with over 0.5 IoU as TP
        gt_labels = torch.div(unique_combo, offset, rounding_mode="floor")
        pred_labels = unique_combo % offset
        gt_areas = torch.empty(len(gt_labels), device=device)
        for i, id in enumerate(gt_labels):
            gt_areas[i] = counts_gt[id2idx_gt[id.item()]]

        pred_areas = torch.empty(len(pred_labels), device=device)
        for i, id in enumerate(pred_labels):
            pred_areas[i] = counts_pred[id2idx_pred[id.item()]]

        intersections = counts_combo
        unions = gt_areas + pred_areas - intersections
        ious = intersections.float() / unions.float()
        for pred_label in pred_labels:
            relevant_intersections = intersections[pred_labels == pred_label]
            most_overlapping_id = relevant_intersections.argmax()
            # find the gt id of the cluster with largest overlap with current prediction
            max_gt_id = gt_labels[pred_labels == pred_label][most_overlapping_id]
            if not max_gt_id == -1 or allow_invalid_association:
                predicted_instance_ids[predicted_instance_ids == pred_label] = max_gt_id
    return predicted_instance_ids


def compute_pq(pred_leaf_ids, leaf_labels, cfg):
    n_samples = len(pred_leaf_ids)
    panQual_calculator = PanopticEval(
        n_classes=3,
        min_points=len(pred_leaf_ids[0])
        * cfg["data"]["min_leaf_point_ratio_inference"],
        ignore=[
            2,
        ],
    )
    for sample in range(n_samples):
        valid_points = leaf_labels[sample] >= 0
        # use semantic mask to compute panoptic quality only on valid points by ignoring sem class 2
        valid_mask = torch.ones_like(pred_leaf_ids[sample]).long().numpy()
        valid_mask[~valid_points] = 2
        panQual_calculator.addBatch(
            pred_leaf_ids[sample].long().numpy() > 0,
            pred_leaf_ids[sample].long().numpy(),
            valid_mask,
            leaf_labels[sample].long().numpy(),
        )
    (
        class_all_PQ,
        class_all_SQ,
        class_all_RQ,
        class_all_PR,
        class_all_RC,
    ) = panQual_calculator.getPQ(return_pr_rc=True)
    return class_all_PQ, class_all_SQ, class_all_RQ, class_all_PR, class_all_RC




def eval_only(batch, cfg):
    """Evaluates a batch

    Args:
        batch (dict): Contains:
                        points: Nx3 tensor with point positions
                        leaf_labels: Nx1 tensor containing leaf labels
                        sampled_indices: Nx1 binary mask for postprocessed points (during validation steps postprocessing is performed on downsample cloud to speed up)
                        pred_leaf_ids: Nx1 tensor with predicted leaf ids
                        pred_leaf_id_probs: Nx1 tensor with predicted leaf id probabilities
        cfg (dict): Parameters from config file
    """
    pred_leaf_ids = batch["pred_leaf_ids"]
    pred_leaf_id_probs = batch["pred_leaf_id_probs"]
    leaf_labels = batch["leaf_labels"]
    metrics = {}
    PQ, SQ, RQ, PR, RC = compute_pq(pred_leaf_ids, leaf_labels, cfg)
    metrics["PQ"] = PQ
    metrics["SQ"] = SQ
    metrics["RQ"] = RQ
    metrics["PR"] = PR
    metrics["RC"] = RC

    return metrics


class Evaluator:
    def __init__(
        self,
        trainer_obj,
        type,
        instance_field_name,
        instance_probs_field_name,
        label_field_name,
        plant_offset_size=3,
    ) -> None:
        self.trainer_obj = trainer_obj
        self.type = type
        self.instance_field_name = instance_field_name
        self.instance_probs_field_name = instance_probs_field_name
        self.label_field_name = label_field_name
        self.plant_offset_size = plant_offset_size

    def compute_PQ(self, log_vals, data_batch, epoch_type):
        instance_ids = log_vals[self.instance_field_name]
        sampled_indices = log_vals["sampled_indices"]
        panQual_calculator = PanopticEval(
            n_classes=3,
            min_points=len(instance_ids[0])
            * self.trainer_obj.hparams["data"]["min_leaf_point_ratio_inference"],
            ignore=[
                2,
            ],
        )
        for sample in range(len(instance_ids)):
            valid_points = (
                data_batch[self.label_field_name][sample].cpu()[sampled_indices] >= 0
            )

            # use semantic mask to compute panoptic quality only on valid points by ignoring sem class 2
            valid_mask = torch.ones_like(instance_ids[sample]).long().cpu().numpy()
            valid_mask[~valid_points] = 2
            panQual_calculator.addBatch(
                instance_ids[sample].long().cpu().numpy() > 0,
                instance_ids[sample].long().cpu().numpy(),
                valid_mask,  # target_mask,
                data_batch[self.label_field_name][sample]
                .cpu()[sampled_indices]
                .long()
                .cpu()
                .numpy(),
            )

        (
            class_all_PQ,
            class_all_SQ,
            class_all_RQ,
        ) = panQual_calculator.getPQ()

        # log metrics
        self.trainer_obj.log(
            epoch_type + "/PQ_" + self.type, class_all_PQ, prog_bar=True
        )
        metrics = {}
        metrics["PQ_" + self.type] = class_all_PQ
        self.trainer_obj.log(epoch_type + "/RQ_" + self.type, class_all_RQ)
        self.trainer_obj.log(epoch_type + "/SQ_" + self.type, class_all_SQ)
        return metrics

    def evaluate_plant_center(self, log_vals, data_batch, epoch_type):
        pred_ids = log_vals["plant_ids"].clone()
        pred_ids[pred_ids >= 0] += 1
        remapped_plant_preds = map_instances(
            pred_ids,
            data_batch["plant_labels"][:, log_vals["sampled_indices"]],
            allow_invalid_association=True,
        )

        pred_plant_centers = torch.empty((0, self.plant_offset_size)).to(
            data_batch["points"].device
        )
        gt_plant_centers = torch.empty((0, self.plant_offset_size)).to(
            data_batch["points"].device
        )
        for plant_id in torch.unique(remapped_plant_preds)[1:]:
            plant_mask = remapped_plant_preds == plant_id
            pred_plant_centers = torch.vstack(
                (
                    pred_plant_centers,
                    (
                        data_batch["points"][:, log_vals["sampled_indices"]][
                            plant_mask
                        ][:, : self.plant_offset_size]
                        + log_vals["plant_offsets"][:, log_vals["sampled_indices"]][
                            plant_mask
                        ]
                    ).mean(dim=0),
                )
            )
            gt_plant_mask = data_batch["plant_labels"] == plant_id
            gt_plant_centers = torch.vstack(
                (
                    gt_plant_centers,
                    data_batch["plant_centers"][gt_plant_mask][0][
                        : self.plant_offset_size
                    ],
                )
            )
        if len(pred_plant_centers) == 0:
            plant_center_error = np.nan
        else:
            plant_center_error = np.nanmean(
                (pred_plant_centers - gt_plant_centers).norm(dim=-1).cpu().numpy(),
                axis=0,
            )
        if not np.isnan(plant_center_error):
            self.trainer_obj.log(
                epoch_type + "/plant_center_pred_error",
                plant_center_error,
                prog_bar=True,
            )
        metrics = {}
        metrics["plant_center_pred_error"] = plant_center_error

        # redo plant center computation for all plants predictions
        pred_plant_centers = torch.empty((0, self.plant_offset_size)).to(
            data_batch["points"].device
        )
        for plant_id in torch.unique(log_vals["plant_ids"])[1:]:
            plant_mask = log_vals["plant_ids"] == plant_id
            pred_plant_centers = torch.vstack(
                (
                    pred_plant_centers,
                    (
                        data_batch["points"][:, log_vals["sampled_indices"]][
                            plant_mask
                        ][:, : self.plant_offset_size]
                        + log_vals["plant_offsets"][:, log_vals["sampled_indices"]][
                            plant_mask
                        ]
                    ).mean(dim=0),
                )
            )
        return metrics, pred_plant_centers

    def log_predictions(
        self,
        epoch_type,
        batch_data,
        log_vals,
        log_labels=False,
        pred_plant_centers=None,
    ):
        pcl_name = epoch_type + ":preds_" + self.type
        log_labeled_cloud(
            batch_data["points"][0][log_vals["sampled_indices"]].detach().cpu(),
            log_vals[self.instance_field_name][0].cpu(),
            self.trainer_obj,
            pcl_name,
        )
        if log_labels:
            log_labeled_cloud(
                batch_data["points"][0][log_vals["sampled_indices"]].detach().cpu(),
                batch_data["leaf_labels"][0].cpu()[log_vals["sampled_indices"]].cpu(),
                self.trainer_obj,
                epoch_type + ":" + "labels_" + self.type,
            )


def cluster_embeddings(
    embs,
    sampled_indices,
    min_n_points,
    min_samples,
    cluster_device="cpu",
    cdist_cluster_metric=False,
    clustering_algo="hdbscan",
):
    """Cluster the points into individual leaves based on embeddings.

    Args:
        embs ([type]): Embedding vectors
        sampled_indices ([type]): Subset of points to use for faster computation
        batch ([type]): Batch dataframe

    Returns:
        [type]: Cluster id of each point
    """
    device = embs.device
    embs = embs.detach()
    leaf_ids = torch.empty((len(embs), sampled_indices.sum()), device=embs.device)
    leaf_id_probs = torch.empty((len(embs), sampled_indices.sum()))
    for sample in range(len(embs)):
        if clustering_algo == "hdbscan":
            if cluster_device == "cpu":
                if cdist_cluster_metric:
                    embs_batch = embs[sample][sampled_indices].cpu()
                    clustering = hdbscan_cpu(
                        min_cluster_size=min_n_points,
                        min_samples=min_samples,
                        metric="minkowski",
                        p=2.0,
                    ).fit(
                        # input_array
                        embs_batch.numpy()
                    )
                else:
                    clustering = hdbscan_cpu(
                        min_cluster_size=min_n_points, min_samples=min_samples, p=None
                    ).fit(
                        # input_array
                        embs[sample][sampled_indices]
                        .cpu()
                        .numpy()
                    )
            else:
                raise ValueError("Not implemented")
            leaf_ids[sample] = torch.as_tensor(clustering.labels_, device=embs.device)
            leaf_id_probs[sample] = torch.as_tensor(
                clustering.probabilities_, device=embs.device
            )
        elif clustering_algo == "meanshift":
            clustering = MeanShift(bandwidth=0.02, n_jobs=-1).fit(
                # input_array
                embs[sample][sampled_indices]
                .cpu()
                .numpy()
            )
            leaf_ids[sample] = torch.as_tensor(clustering.labels_, device=embs.device)
            leaf_id_probs[sample] = torch.ones_like(
                leaf_ids[sample], device=embs.device
            )
    leaf_ids = leaf_ids.to(device)
    return leaf_ids, leaf_id_probs
